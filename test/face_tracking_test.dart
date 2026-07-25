import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:face_detection_tflite/src/shared/face_tracker.dart';
import 'package:flutter_test/flutter_test.dart';

Face _face(double xmin, double ymin, double xmax, double ymax) {
  const Size imageSize = Size(1000, 1000);
  return Face(
    detection: Detection(
      boundingBox: RectF(xmin, ymin, xmax, ymax),
      score: 0.9,
      keypointsXY: List<double>.filled(12, 0.5),
      imageSize: imageSize,
    ),
    mesh: null,
    irises: const <Point>[],
    originalSize: imageSize,
  );
}

/// Square face of half-extent `0.05 * scale` centered on (0.45, 0.45).
Face _scaledFace(double scale) {
  const double centre = 0.45;
  final double half = 0.05 * scale;
  return _face(centre - half, centre - half, centre + half, centre + half);
}

SegmentationMask _mask() => SegmentationMask.internal(
  internalData: Float32List(4),
  width: 2,
  height: 2,
  originalWidth: 2,
  originalHeight: 2,
  padding: const <double>[0, 0, 0, 0],
);

void main() {
  test('FaceDetector tracking is opt-in and reset is safe before init', () {
    final FaceDetector detector = FaceDetector();

    expect(detector.isTrackingEnabled, isFalse);
    expect(detector.maxMissedFrames, kDefaultMaxMissedFrames);
    expect(detector.resetTracking, returnsNormally);
  });

  test('validateTrackingConfig accepts zero and rejects negatives', () {
    expect(() => validateTrackingConfig(maxMissedFrames: 0), returnsNormally);
    expect(() => validateTrackingConfig(maxMissedFrames: 3), returnsNormally);
    expect(
      () => validateTrackingConfig(maxMissedFrames: -1),
      throwsA(
        isA<ArgumentError>().having(
          (ArgumentError e) => e.name,
          'name',
          'maxMissedFrames',
        ),
      ),
    );
  });

  group('Face trackingId', () {
    test('defaults to null and can be attached without changing detection', () {
      final Face original = _face(0.1, 0.1, 0.3, 0.3);
      final Face tracked = original.withTrackingId(42);

      expect(original.trackingId, isNull);
      expect(tracked.trackingId, 42);
      expect(identical(tracked.detectionData, original.detectionData), isTrue);
      expect(tracked.boundingBox.topLeft.x, original.boundingBox.topLeft.x);
    });

    test('round-trips through map serialization and accepts legacy maps', () {
      final Face original = _face(0.1, 0.1, 0.3, 0.3).withTrackingId(7);
      final Map<String, dynamic> map = original.toMap();

      expect(Face.fromMap(map).trackingId, 7);

      map.remove('trackingId');
      expect(Face.fromMap(map).trackingId, isNull);
    });
  });

  group('TemporalFaceTracker', () {
    test('keeps IDs stable when detector output order changes', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();

      final List<Face> first = tracker.update(<Face>[
        _face(0.10, 0.20, 0.30, 0.50),
        _face(0.65, 0.20, 0.85, 0.50),
      ]);
      final List<Face> second = tracker.update(<Face>[
        _face(0.63, 0.20, 0.83, 0.50),
        _face(0.12, 0.20, 0.32, 0.50),
      ]);

      expect(first.map((Face face) => face.trackingId), <int?>[1, 2]);
      expect(second.map((Face face) => face.trackingId), <int?>[2, 1]);
    });

    test('uses motion prediction to retain IDs while faces cross', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();

      expect(
        tracker
            .update(<Face>[
              _face(0.10, 0.2, 0.25, 0.5),
              _face(0.75, 0.2, 0.90, 0.5),
            ])
            .map((Face face) => face.trackingId),
        <int?>[1, 2],
      );
      expect(
        tracker
            .update(<Face>[
              _face(0.60, 0.2, 0.75, 0.5),
              _face(0.25, 0.2, 0.40, 0.5),
            ])
            .map((Face face) => face.trackingId),
        <int?>[2, 1],
      );
      expect(
        tracker
            .update(<Face>[
              _face(0.40, 0.2, 0.55, 0.5),
              _face(0.45, 0.2, 0.60, 0.5),
            ])
            .map((Face face) => face.trackingId),
        <int?>[1, 2],
      );
      expect(
        tracker
            .update(<Face>[
              _face(0.30, 0.2, 0.45, 0.5),
              _face(0.55, 0.2, 0.70, 0.5),
            ])
            .map((Face face) => face.trackingId),
        <int?>[2, 1],
      );
    });

    test('survives three missed frames and expires on the fourth', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();
      expect(
        tracker.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        1,
      );

      tracker.update(const <Face>[]);
      tracker.update(const <Face>[]);
      tracker.update(const <Face>[]);
      expect(
        tracker.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        1,
      );

      tracker.update(const <Face>[]);
      tracker.update(const <Face>[]);
      tracker.update(const <Face>[]);
      tracker.update(const <Face>[]);
      expect(
        tracker.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        2,
      );
    });

    test('allocates a new ID for an unrelated distant face', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();

      expect(
        tracker.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        1,
      );
      expect(
        tracker.update(<Face>[_face(0.7, 0.2, 0.9, 0.5)]).single.trackingId,
        2,
      );
    });

    test('reset clears associations and restarts IDs', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();

      expect(
        tracker.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        1,
      );
      tracker.reset();
      expect(
        tracker.update(<Face>[_face(0.7, 0.2, 0.9, 0.5)]).single.trackingId,
        1,
      );
    });

    test('holds one ID for a stationary face across a long sequence', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();
      final Set<int?> ids = <int?>{};

      for (int frame = 0; frame < 50; frame++) {
        ids.add(
          tracker.update(<Face>[_face(0.4, 0.4, 0.5, 0.6)]).single.trackingId,
        );
      }

      expect(ids, <int>{1});
    });

    test('holds one ID for a face crossing the frame at steady speed', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();
      final Set<int?> ids = <int?>{};

      for (int frame = 0; frame < 20; frame++) {
        final double xmin = 0.05 + frame * 0.04;
        ids.add(
          tracker
              .update(<Face>[_face(xmin, 0.4, xmin + 0.10, 0.6)])
              .single
              .trackingId,
        );
      }

      expect(ids, <int>{1});
    });

    test('keeps neighbouring faces distinct as they converge', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();

      expect(
        tracker
            .update(<Face>[
              _face(0.20, 0.4, 0.30, 0.6),
              _face(0.60, 0.4, 0.70, 0.6),
            ])
            .map((Face face) => face.trackingId),
        <int?>[1, 2],
      );
      expect(
        tracker
            .update(<Face>[
              _face(0.42, 0.4, 0.52, 0.6),
              _face(0.46, 0.4, 0.56, 0.6),
            ])
            .map((Face face) => face.trackingId),
        <int?>[1, 2],
      );
    });

    test('tolerates growth up to the scale gate, then reassigns', () {
      // minScaleSimilarity is 0.25, i.e. a doubling of linear size between two
      // processed frames is the limit. Approaching the camera at a normal rate
      // stays well inside it; a jump past it is treated as a different face.
      for (final double scale in <double>[1.1, 1.5, 1.95]) {
        final TemporalFaceTracker tracker = TemporalFaceTracker();
        tracker.update(<Face>[_scaledFace(1.0)]);
        expect(
          tracker.update(<Face>[_scaledFace(scale)]).single.trackingId,
          1,
          reason: 'growth of ${scale}x should keep the track',
        );
      }

      final TemporalFaceTracker tracker = TemporalFaceTracker();
      tracker.update(<Face>[_scaledFace(1.0)]);
      expect(tracker.update(<Face>[_scaledFace(2.5)]).single.trackingId, 2);
    });

    // Box overlap carries 65% of the match score, so these two cases exist to
    // make that weight load-bearing. Both are built so proximity and scale
    // favour the WRONG candidate and only overlap picks the right one: the
    // decoy shares the track's exact centre and exact area (perfect proximity,
    // perfect scale similarity) but is transposed so it barely overlaps, while
    // the true continuation has moved half its own width away, giving up
    // proximity while still overlapping substantially.
    //
    // The displacement is tuned so the two scores cross at an IoU weight of
    // roughly 0.42: at the real 0.65 the overlapping face wins by a clear
    // margin, but drop the weight to 0.4 or below and the decoy takes the ID.
    // A smaller displacement would only catch the weight being zeroed.
    Face wideFace(double xmin) => _face(xmin, 0.45, xmin + 0.40, 0.55);
    // Same centre and area as wideFace(0.30), transposed so overlap is small.
    final Face rotatedDecoy = _face(0.45, 0.30, 0.55, 0.70);

    test('ranks candidates by overlap, not centre distance alone', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();
      expect(tracker.update(<Face>[wideFace(0.30)]).single.trackingId, 1);

      // The displaced-but-overlapping face must keep the track; the perfectly
      // centred, perfectly scaled, barely-overlapping decoy must not.
      final List<Face> second = tracker.update(<Face>[
        wideFace(0.50),
        rotatedDecoy,
      ]);

      expect(
        second.map((Face f) => f.trackingId),
        <int?>[1, 2],
        reason: 'overlap must outrank the better-centred decoy',
      );
    });

    test('overlap ranking is independent of detector output order', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker();
      expect(tracker.update(<Face>[wideFace(0.30)]).single.trackingId, 1);

      // Same frame, decoy listed first. The assignment must not move.
      final List<Face> second = tracker.update(<Face>[
        rotatedDecoy,
        wideFace(0.50),
      ]);

      expect(
        second.map((Face f) => f.trackingId),
        <int?>[2, 1],
        reason: 'the overlapping face keeps the ID regardless of position',
      );
    });

    test('breaks overlap-free ties by proximity, not input order', () {
      // Mirror of the overlap cases for the proximity term. Both candidates
      // clear the track's box entirely, so IoU is 0 for each, and both are the
      // same size, so scale similarity ties at 1.0. Centre distance is the only
      // signal left. The nearer candidate is listed SECOND on purpose: with
      // proximity weighted the scores separate and it still wins, but drop that
      // weight and every score collapses to the scale term, leaving a tie that
      // the deterministic tie-break hands to whichever came first.
      final TemporalFaceTracker tracker = TemporalFaceTracker();
      expect(
        tracker.update(<Face>[_face(0.40, 0.45, 0.50, 0.55)]).single.trackingId,
        1,
      );

      final List<Face> second = tracker.update(<Face>[
        _face(0.60, 0.45, 0.70, 0.55), // farther: 1.41 diagonals away
        _face(0.52, 0.45, 0.62, 0.55), // nearer:  0.85 diagonals away
      ]);

      expect(
        second.map((Face f) => f.trackingId),
        <int?>[2, 1],
        reason: 'the nearer face keeps the ID despite being listed second',
      );
    });

    test('honours a custom maxMissedFrames for track expiry', () {
      // Default is 3; a tracker given 6 must still hold the ID after five
      // empty frames, where a default tracker would already have retired it.
      final TemporalFaceTracker patient = TemporalFaceTracker(
        maxMissedFrames: 6,
      );
      expect(
        patient.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        1,
      );
      for (int i = 0; i < 5; i++) {
        patient.update(const <Face>[]);
      }
      expect(
        patient.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        1,
      );

      final TemporalFaceTracker impatient = TemporalFaceTracker();
      expect(
        impatient.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        1,
      );
      for (int i = 0; i < 5; i++) {
        impatient.update(const <Face>[]);
      }
      expect(
        impatient.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        2,
      );
    });

    test('maxMissedFrames of zero retires a track on the first miss', () {
      final TemporalFaceTracker tracker = TemporalFaceTracker(
        maxMissedFrames: 0,
      );
      expect(
        tracker.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        1,
      );
      tracker.update(const <Face>[]);
      expect(
        tracker.update(<Face>[_face(0.1, 0.2, 0.3, 0.5)]).single.trackingId,
        2,
      );
    });

    test(
      'gives degenerate and non-finite boxes fresh IDs without matching',
      () {
        final TemporalFaceTracker zeroArea = TemporalFaceTracker();
        expect(
          zeroArea.update(<Face>[_face(0.3, 0.3, 0.3, 0.3)]).single.trackingId,
          1,
        );
        expect(
          zeroArea.update(<Face>[_face(0.3, 0.3, 0.3, 0.3)]).single.trackingId,
          2,
        );

        final TemporalFaceTracker nonFinite = TemporalFaceTracker();
        expect(
          nonFinite
              .update(<Face>[_face(double.nan, 0.2, 0.3, 0.5)])
              .single
              .trackingId,
          1,
        );
        expect(
          nonFinite
              .update(<Face>[_face(double.nan, 0.2, 0.3, 0.5)])
              .single
              .trackingId,
          2,
        );
      },
    );
  });

  group('TemporalTrackingController', () {
    test('passes values through untouched while disabled', () async {
      final TemporalTrackingController controller =
          TemporalTrackingController();
      final List<Face> input = <Face>[_face(0.1, 0.2, 0.3, 0.5)];

      final List<Face> output = await controller.run(
        () async => input,
        controller.attachFaces,
      );

      expect(identical(output, input), isTrue);
      expect(output.single.trackingId, isNull);
    });

    test('runs enabled operations in invocation order', () async {
      final TemporalTrackingController controller = TemporalTrackingController()
        ..configure(enabled: true);
      final Completer<List<Face>> firstResult = Completer<List<Face>>();
      bool secondStarted = false;

      final Future<List<Face>> first = controller.run(
        () => firstResult.future,
        controller.attachFaces,
      );
      final Future<List<Face>> second = controller.run(() async {
        secondStarted = true;
        return <Face>[_face(0.12, 0.2, 0.32, 0.5)];
      }, controller.attachFaces);

      await Future<void>.delayed(Duration.zero);
      expect(secondStarted, isFalse);

      firstResult.complete(<Face>[_face(0.1, 0.2, 0.3, 0.5)]);
      expect((await first).single.trackingId, 1);
      expect((await second).single.trackingId, 1);
      expect(secondStarted, isTrue);
    });

    test(
      'reset prevents a stale in-flight result from entering new state',
      () async {
        final TemporalTrackingController controller =
            TemporalTrackingController()..configure(enabled: true);
        final Completer<List<Face>> staleResult = Completer<List<Face>>();
        final Future<List<Face>> stale = controller.run(
          () => staleResult.future,
          controller.attachFaces,
        );

        await Future<void>.delayed(Duration.zero);
        controller.reset();
        staleResult.complete(<Face>[_face(0.1, 0.2, 0.3, 0.5)]);

        expect((await stale).single.trackingId, isNull);
        final List<Face> fresh = await controller.run(
          () async => <Face>[_face(0.7, 0.2, 0.9, 0.5)],
          controller.attachFaces,
        );
        expect(fresh.single.trackingId, 1);
      },
    );

    test('a failed operation propagates and leaves the queue usable', () async {
      final TemporalTrackingController controller = TemporalTrackingController()
        ..configure(enabled: true);

      await expectLater(
        controller.run<List<Face>>(
          () async => throw StateError('decode failed'),
          controller.attachFaces,
        ),
        throwsStateError,
      );

      // A mid-stream failure must not wedge the sequencing chain: without the
      // completer being settled on the error path this would never resolve.
      final Future<List<Face>> next = controller.run(
        () async => <Face>[_face(0.1, 0.2, 0.3, 0.5)],
        controller.attachFaces,
      );
      expect(
        await next.timeout(const Duration(seconds: 5)),
        isA<List<Face>>().having(
          (List<Face> faces) => faces.single.trackingId,
          'trackingId',
          1,
        ),
      );
    });

    test('applies maxMissedFrames to the tracker it builds', () async {
      final TemporalTrackingController controller = TemporalTrackingController()
        ..configure(enabled: true, maxMissedFrames: 5);

      expect(controller.maxMissedFrames, 5);

      Future<List<Face>> frame(List<Face> faces) =>
          controller.run(() async => faces, controller.attachFaces);

      expect(
        (await frame(<Face>[_face(0.1, 0.2, 0.3, 0.5)])).single.trackingId,
        1,
      );
      for (int i = 0; i < 4; i++) {
        await frame(const <Face>[]);
      }
      // Four misses is inside the configured budget of five, so the ID holds.
      expect(
        (await frame(<Face>[_face(0.1, 0.2, 0.3, 0.5)])).single.trackingId,
        1,
      );
    });

    test('defaults to kDefaultMaxMissedFrames and rejects a negative', () {
      final TemporalTrackingController controller = TemporalTrackingController()
        ..configure(enabled: true);

      expect(controller.maxMissedFrames, kDefaultMaxMissedFrames);
      expect(kDefaultMaxMissedFrames, 3);
      expect(
        () => controller.configure(enabled: true, maxMissedFrames: -1),
        throwsArgumentError,
      );
    });

    test('reconfiguring drops associations from the previous stream', () async {
      final TemporalTrackingController controller = TemporalTrackingController()
        ..configure(enabled: true);
      final List<Face> first = await controller.run(
        () async => <Face>[_face(0.1, 0.2, 0.3, 0.5)],
        controller.attachFaces,
      );
      expect(first.single.trackingId, 1);

      controller.configure(enabled: true, maxMissedFrames: 8);
      final List<Face> afterReconfigure = await controller.run(
        () async => <Face>[_face(0.1, 0.2, 0.3, 0.5)],
        controller.attachFaces,
      );
      expect(afterReconfigure.single.trackingId, 1);
      expect(controller.maxMissedFrames, 8);
    });

    test('tracks combined detection and segmentation results', () async {
      final TemporalTrackingController controller = TemporalTrackingController()
        ..configure(enabled: true);
      final SegmentationMask mask = _mask();

      Future<DetectionWithSegmentationResult> run(double xmin) {
        return controller.run<DetectionWithSegmentationResult>(
          () async => DetectionWithSegmentationResult(
            faces: <Face>[_face(xmin, 0.2, xmin + 0.2, 0.5)],
            segmentationMask: mask,
            detectionTimeMs: 11,
            segmentationTimeMs: 22,
          ),
          (DetectionWithSegmentationResult result) =>
              DetectionWithSegmentationResult(
                faces: controller.attachFaces(result.faces),
                segmentationMask: result.segmentationMask,
                detectionTimeMs: result.detectionTimeMs,
                segmentationTimeMs: result.segmentationTimeMs,
              ),
        );
      }

      final DetectionWithSegmentationResult first = await run(0.10);
      final DetectionWithSegmentationResult second = await run(0.12);

      expect(first.faces.single.trackingId, 1);
      expect(second.faces.single.trackingId, 1);
      // The mask and both timings must survive the rebuild that attaches IDs.
      expect(identical(first.segmentationMask, mask), isTrue);
      expect(second.detectionTimeMs, 11);
      expect(second.segmentationTimeMs, 22);
    });
  });
}
