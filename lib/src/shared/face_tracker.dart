import 'dart:async';
import 'dart:math' as math;

import 'face_model_config.dart' show kDefaultMaxMissedFrames;
import 'face_types.dart';

/// Throws [ArgumentError] if the tracking configuration is unusable.
///
/// Called before any model loads so bad configuration fails fast, matching
/// how `validateFaceGates` guards the detection gates.
void validateTrackingConfig({required int maxMissedFrames}) {
  if (maxMissedFrames < 0) {
    throw ArgumentError.value(
      maxMissedFrames,
      'maxMissedFrames',
      'must be zero or greater',
    );
  }
}

/// Stateful geometric association used by `FaceDetector` when temporal
/// tracking is enabled.
///
/// This deliberately uses only normalized detector boxes. It does not inspect
/// face embeddings and therefore must not be treated as identity recognition.
/// The class lives under `src/` and is shared by the native and web detector
/// implementations; it is not part of the package's public API.
class TemporalFaceTracker {
  /// Creates a tracker with conservative defaults for camera/video frames.
  TemporalFaceTracker({
    this.maxMissedFrames = kDefaultMaxMissedFrames,
    this.maxNormalizedCenterDistance = 1.5,
    this.minScaleSimilarity = 0.25,
  }) : assert(maxMissedFrames >= 0),
       // Admission is decided by center distance alone (see [_candidate]), and
       // that only holds while the limit cannot exclude an overlapping pair.
       assert(maxNormalizedCenterDistance >= 1.0),
       assert(minScaleSimilarity >= 0.0),
       assert(minScaleSimilarity <= 1.0);

  /// Number of fully processed frames for which an absent face is retained.
  final int maxMissedFrames;

  /// Maximum center displacement, measured in average box diagonals.
  ///
  /// Must be at least 1.0. Two boxes that overlap at all are never more than
  /// 1.0 average diagonals apart, so any value at or above 1.0 admits every
  /// overlapping pair; see [_candidate].
  final double maxNormalizedCenterDistance;

  /// Smallest `min(area) / max(area)` accepted for a match.
  final double minScaleSimilarity;

  final Map<int, _FaceTrack> _tracks = <int, _FaceTrack>{};
  int _nextId = 1;

  /// Associates [faces] with recent tracks and returns faces carrying IDs.
  ///
  /// Input ordering is preserved. IDs are monotonically allocated until
  /// [reset] is called. A constant-velocity box prediction helps retain IDs
  /// during movement and short detector dropouts.
  List<Face> update(List<Face> faces) {
    final List<_TrackBox> boxes = <_TrackBox>[
      for (final Face face in faces) _TrackBox.fromFace(face),
    ];
    final List<_MatchCandidate> candidates = <_MatchCandidate>[];

    for (final _FaceTrack track in _tracks.values) {
      for (
        int detectionIndex = 0;
        detectionIndex < boxes.length;
        detectionIndex++
      ) {
        final _MatchCandidate? candidate = _candidate(
          track,
          detectionIndex,
          boxes[detectionIndex],
        );
        if (candidate != null) candidates.add(candidate);
      }
    }

    // A global score ordering avoids making association depend on detector
    // output order. Track ID and detection index provide deterministic ties.
    candidates.sort((_MatchCandidate a, _MatchCandidate b) {
      final int scoreOrder = b.score.compareTo(a.score);
      if (scoreOrder != 0) return scoreOrder;
      final int trackOrder = a.track.id.compareTo(b.track.id);
      if (trackOrder != 0) return trackOrder;
      return a.detectionIndex.compareTo(b.detectionIndex);
    });

    final Set<int> matchedTrackIds = <int>{};
    final Set<int> matchedDetectionIndices = <int>{};
    final List<int?> assignments = List<int?>.filled(faces.length, null);

    for (final _MatchCandidate candidate in candidates) {
      if (matchedTrackIds.contains(candidate.track.id) ||
          matchedDetectionIndices.contains(candidate.detectionIndex)) {
        continue;
      }
      candidate.track.match(boxes[candidate.detectionIndex]);
      matchedTrackIds.add(candidate.track.id);
      matchedDetectionIndices.add(candidate.detectionIndex);
      assignments[candidate.detectionIndex] = candidate.track.id;
    }

    for (final _FaceTrack track in _tracks.values) {
      if (!matchedTrackIds.contains(track.id)) track.missedFrames++;
    }
    _tracks.removeWhere(
      (int _, _FaceTrack track) => track.missedFrames > maxMissedFrames,
    );

    for (int i = 0; i < faces.length; i++) {
      if (assignments[i] != null) continue;
      final int id = _nextId++;
      _tracks[id] = _FaceTrack(id, boxes[i]);
      assignments[i] = id;
    }

    return <Face>[
      for (int i = 0; i < faces.length; i++)
        faces[i].withTrackingId(assignments[i]!),
    ];
  }

  _MatchCandidate? _candidate(
    _FaceTrack track,
    int detectionIndex,
    _TrackBox detection,
  ) {
    final _TrackBox predicted = track.predictedBox;
    if (!predicted.isValid || !detection.isValid) return null;

    final double maxArea = math.max(predicted.area, detection.area);
    final double scaleSimilarity = maxArea == 0.0
        ? 0.0
        : math.min(predicted.area, detection.area) / maxArea;
    if (scaleSimilarity < minScaleSimilarity) return null;

    final double iou = predicted.intersectionOverUnion(detection);
    final double dx = predicted.centerX - detection.centerX;
    final double dy = predicted.centerY - detection.centerY;
    final double centerDistance = math.sqrt(dx * dx + dy * dy);
    final double referenceDiagonal = math.max(
      0.05,
      (predicted.diagonal + detection.diagonal) * 0.5,
    );
    final double normalizedDistance = centerDistance / referenceDiagonal;
    final double distanceLimit =
        maxNormalizedCenterDistance + track.missedFrames * 0.25;

    // Admission is center distance only. This used to also admit a pair whose
    // IoU cleared a 0.02 floor, but that branch could never decide anything:
    // two boxes that overlap at all are at most 1.0 average diagonals apart,
    // while the limit starts at 1.5 and only widens with missed frames, so the
    // distance test had already admitted every overlapping pair. IoU still does
    // the heavy lifting in the score below, where it is the dominant term.
    if (normalizedDistance > distanceLimit) return null;

    final double proximity = (1.0 - normalizedDistance / distanceLimit).clamp(
      0.0,
      1.0,
    );
    final double score = iou * 0.65 + proximity * 0.25 + scaleSimilarity * 0.10;
    return _MatchCandidate(track, detectionIndex, score);
  }

  /// Drops all temporal state and restarts ID allocation from one.
  void reset() {
    _tracks.clear();
    _nextId = 1;
  }
}

/// Serializes tracking-enabled detector calls and owns their association
/// lifecycle.
///
/// Keeping this controller shared ensures native and web apply frames in the
/// same invocation order even if their asynchronous inference work would
/// otherwise complete out of order.
class TemporalTrackingController {
  TemporalFaceTracker _tracker = TemporalFaceTracker();
  Future<void> _tail = Future<void>.value();
  bool _enabled = false;
  int _maxMissedFrames = kDefaultMaxMissedFrames;
  int _generation = 0;

  bool get isEnabled => _enabled;

  /// Frames a face may go undetected before its ID is retired.
  int get maxMissedFrames => _maxMissedFrames;

  /// Applies a new tracking configuration and clears prior stream state.
  ///
  /// Rebuilds the tracker so [maxMissedFrames] takes effect, which also drops
  /// any associations carried over from a previous configuration.
  void configure({
    required bool enabled,
    int maxMissedFrames = kDefaultMaxMissedFrames,
  }) {
    validateTrackingConfig(maxMissedFrames: maxMissedFrames);
    _enabled = enabled;
    _maxMissedFrames = maxMissedFrames;
    _tracker = TemporalFaceTracker(maxMissedFrames: maxMissedFrames);
    reset();
  }

  /// Clears associations and invalidates results from already-running calls.
  void reset() {
    _generation++;
    _tracker.reset();
  }

  /// Runs [operation] in invocation order and applies [attachTracking].
  ///
  /// When tracking is disabled, the operation runs immediately and the value
  /// passes through untouched. If [reset] occurs while an older operation is
  /// running, that stale result is returned without entering the new track
  /// state.
  Future<T> run<T>(
    Future<T> Function() operation,
    T Function(T value) attachTracking,
  ) {
    if (!_enabled) return operation();

    final Future<void> predecessor = _tail;
    final Completer<void> completion = Completer<void>();
    _tail = completion.future;
    final int generation = _generation;

    Future<T> execute() async {
      try {
        await predecessor;
        final T value = await operation();
        if (generation != _generation) return value;
        return attachTracking(value);
      } finally {
        completion.complete();
      }
    }

    return execute();
  }

  List<Face> attachFaces(List<Face> faces) => _tracker.update(faces);
}

class _MatchCandidate {
  const _MatchCandidate(this.track, this.detectionIndex, this.score);

  final _FaceTrack track;
  final int detectionIndex;
  final double score;
}

class _FaceTrack {
  _FaceTrack(this.id, this.box);

  final int id;
  _TrackBox box;
  double velocityX = 0.0;
  double velocityY = 0.0;
  int missedFrames = 0;
  int hits = 1;

  _TrackBox get predictedBox => box.shifted(
    velocityX * (missedFrames + 1),
    velocityY * (missedFrames + 1),
  );

  void match(_TrackBox observed) {
    final int elapsedFrames = missedFrames + 1;
    final double observedVelocityX =
        (observed.centerX - box.centerX) / elapsedFrames;
    final double observedVelocityY =
        (observed.centerY - box.centerY) / elapsedFrames;
    if (hits == 1) {
      velocityX = observedVelocityX;
      velocityY = observedVelocityY;
    } else {
      velocityX = velocityX * 0.6 + observedVelocityX * 0.4;
      velocityY = velocityY * 0.6 + observedVelocityY * 0.4;
    }
    box = observed;
    missedFrames = 0;
    hits++;
  }
}

class _TrackBox {
  const _TrackBox(this.xmin, this.ymin, this.xmax, this.ymax);

  factory _TrackBox.fromFace(Face face) {
    final RectF box = face.detectionData.boundingBox;
    return _TrackBox(box.xmin, box.ymin, box.xmax, box.ymax);
  }

  final double xmin;
  final double ymin;
  final double xmax;
  final double ymax;

  double get width => xmax - xmin;
  double get height => ymax - ymin;
  double get area => width * height;
  double get centerX => (xmin + xmax) * 0.5;
  double get centerY => (ymin + ymax) * 0.5;
  double get diagonal => math.sqrt(width * width + height * height);

  bool get isValid =>
      xmin.isFinite &&
      ymin.isFinite &&
      xmax.isFinite &&
      ymax.isFinite &&
      width > 0.0 &&
      height > 0.0;

  _TrackBox shifted(double dx, double dy) =>
      _TrackBox(xmin + dx, ymin + dy, xmax + dx, ymax + dy);

  double intersectionOverUnion(_TrackBox other) {
    final double intersectionWidth = math.max(
      0.0,
      math.min(xmax, other.xmax) - math.max(xmin, other.xmin),
    );
    final double intersectionHeight = math.max(
      0.0,
      math.min(ymax, other.ymax) - math.max(ymin, other.ymin),
    );
    final double intersection = intersectionWidth * intersectionHeight;
    final double union = area + other.area - intersection;
    return union <= 0.0 ? 0.0 : intersection / union;
  }
}
