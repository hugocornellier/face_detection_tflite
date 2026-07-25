// ignore_for_file: avoid_print

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';

/// End-to-end tests for the detection gates (`minScore` / `minFaceSize`) on the
/// real native pipeline, using a genuine multi-face image.
///
/// These are self-checking: the expected filtered sets are DERIVED from an
/// ungated baseline detection on the same image, not hardcoded. That keeps the
/// test stable across model/weight changes (which may shift exact boxes and
/// counts) while still proving the gate wiring end to end.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const String groupShot = 'assets/samples/group-shot-bounding-box-ex1.jpeg';
  const String singleFace = 'assets/samples/landmark-ex1.jpg';
  const FaceDetectionMode mode =
      FaceDetectionMode.fast; // gates need only bbox+score

  // The presence gate reads the mesh model's face flag, so unlike the
  // score/size gates it needs a mode that actually computes a mesh.
  const FaceDetectionMode meshMode = FaceDetectionMode.standard;

  Future<Uint8List> loadBytes(String path) async {
    final ByteData data = await rootBundle.load(path);
    return data.buffer.asUint8List();
  }

  group('Detection gates on a real multi-face image', () {
    test('baseline detects multiple faces and widthFraction stays in [0,1]',
        () async {
      final detector = FaceDetector();
      await detector.initialize();
      final faces = await detector.detectFacesFromBytes(
        await loadBytes(groupShot),
        mode: mode,
      );
      await detector.dispose();

      expect(faces.length, greaterThan(1),
          reason: 'group shot should contain several faces');
      for (final f in faces) {
        expect(f.widthFraction, inInclusiveRange(0.0, 1.0));
      }
    });

    test('minFaceSize keeps exactly the faces at or above the threshold',
        () async {
      final bytes = await loadBytes(groupShot);

      // Ungated baseline.
      final baseDet = FaceDetector();
      await baseDet.initialize();
      final baseline = await baseDet.detectFacesFromBytes(bytes, mode: mode);
      await baseDet.dispose();

      final widths = baseline.map((f) => f.widthFraction).toList()..sort();
      expect(widths.first, lessThan(widths.last),
          reason: 'faces should vary in size for a meaningful size gate');

      // A threshold strictly between the smallest and largest face width.
      final double threshold = (widths.first + widths.last) / 2.0;
      final int expectedKept =
          baseline.where((f) => f.widthFraction >= threshold).length;

      // Gated detector with the same image.
      final gatedDet = FaceDetector();
      await gatedDet.initialize(minFaceSize: threshold);
      final gated = await gatedDet.detectFacesFromBytes(bytes, mode: mode);
      await gatedDet.dispose();

      expect(gated.length, expectedKept);
      expect(gated.length, lessThan(baseline.length),
          reason: 'the size gate should drop the smallest face(s)');
      for (final f in gated) {
        expect(f.widthFraction, greaterThanOrEqualTo(threshold));
      }
    });

    test('minScore keeps exactly the faces at or above the threshold',
        () async {
      final bytes = await loadBytes(groupShot);

      final baseDet = FaceDetector();
      await baseDet.initialize();
      final baseline = await baseDet.detectFacesFromBytes(bytes, mode: mode);
      await baseDet.dispose();

      final scores = baseline.map((f) => f.score).toList()..sort();
      // Only meaningful above the internal 0.5 floor; pick a threshold between
      // the observed min and max score (skip if scores do not vary).
      if (scores.first >= scores.last) {
        return; // all identical: nothing to separate, gate is a no-op
      }
      final double threshold = (scores.first + scores.last) / 2.0;
      final int expectedKept =
          baseline.where((f) => f.score >= threshold).length;

      final gatedDet = FaceDetector();
      await gatedDet.initialize(minScore: threshold);
      final gated = await gatedDet.detectFacesFromBytes(bytes, mode: mode);
      await gatedDet.dispose();

      expect(gated.length, expectedKept);
      for (final f in gated) {
        expect(f.score, greaterThanOrEqualTo(threshold));
      }
    });

    test('zero gates return the full baseline; extreme gates return none',
        () async {
      final bytes = await loadBytes(groupShot);

      final baseDet = FaceDetector();
      await baseDet.initialize();
      final baseline = await baseDet.detectFacesFromBytes(bytes, mode: mode);
      await baseDet.dispose();

      final zeroDet = FaceDetector();
      await zeroDet.initialize(minScore: 0.0, minFaceSize: 0.0);
      final zero = await zeroDet.detectFacesFromBytes(bytes, mode: mode);
      await zeroDet.dispose();
      expect(zero.length, baseline.length,
          reason: 'zero gates must not change results');

      // A face cannot be wider than the image, so minFaceSize just above 1.0 is
      // impossible to satisfy; 1.0 itself is only met by a full-width face.
      final strictDet = FaceDetector();
      await strictDet.initialize(minScore: 1.0);
      final strict = await strictDet.detectFacesFromBytes(bytes, mode: mode);
      await strictDet.dispose();
      expect(strict, isEmpty,
          reason: 'a perfect-confidence threshold should drop all faces');
    });
  });

  /// End-to-end tests for the MediaPipe `min_face_presence_confidence` gate.
  ///
  /// Same self-checking approach as the group above: expected sets are derived
  /// from an ungated baseline on the same image rather than hardcoded, so the
  /// tests survive model/weight changes that shift exact mesh scores.
  ///
  /// This gate is second-stage: it filters on the mesh model's face-flag output
  /// (`Face.meshScore`), which only exists in `standard`/`full` mode. That makes
  /// it unreachable from unit tests with synthetic `Face` objects, which is why
  /// it needs coverage here.
  ///
  /// NOT covered: the gate's headline purpose, rejecting a first-stage false
  /// positive (a hand or palm that clears BlazeFace but scores near zero on the
  /// mesh model) at the default 0.5. No bundled sample produces such a
  /// detection: the hand and palm images available here yield zero BlazeFace
  /// detections at all, so there is nothing for the mesh stage to reject.
  /// Closing this needs a fixture image that genuinely fools the detector.
  group('Face-presence gate (minFacePresenceConfidence)', () {
    test('defaults to 0.5, matching MediaPipe', () async {
      final detector = FaceDetector();
      await detector.initialize();
      final double actual = detector.minFacePresenceConfidence;
      await detector.dispose();

      expect(actual, 0.5,
          reason: 'the documented MediaPipe-parity default is 0.5, and '
              'changing it silently changes which faces callers get back');
    });

    test('keeps exactly the faces whose meshScore clears the threshold',
        () async {
      final bytes = await loadBytes(groupShot);

      // Ungated baseline: gate off, mesh computed.
      final baseDet = FaceDetector();
      await baseDet.initialize(minFacePresenceConfidence: 0.0);
      final baseline =
          await baseDet.detectFacesFromBytes(bytes, mode: meshMode);
      await baseDet.dispose();

      expect(baseline, isNotEmpty,
          reason: 'group shot should detect faces with the gate off');
      for (final f in baseline) {
        expect(f.meshScore, isNotNull,
            reason: 'standard mode must produce a mesh score to gate on');
        expect(f.meshScore, inInclusiveRange(0.0, 1.0));
      }

      // Sweeping the threshold upward must shrink the kept set monotonically,
      // and each run must equal the corresponding baseline subset exactly.
      //
      // Note on strength: real faces in this image all score above 0.99, so the
      // low thresholds are expected no-ops and only the top of the sweep
      // actually filters. `droppedSomewhere` below guards against this test
      // silently decaying into an all-no-op pass if mesh scores ever shift.
      int previousKept = baseline.length;
      bool droppedSomewhere = false;
      for (final double threshold in <double>[0.25, 0.5, 0.75, 1.0]) {
        final int expectedKept = baseline
            .where((f) => (f.meshScore ?? double.infinity) >= threshold)
            .length;

        final gatedDet = FaceDetector();
        await gatedDet.initialize(minFacePresenceConfidence: threshold);
        final gated =
            await gatedDet.detectFacesFromBytes(bytes, mode: meshMode);
        await gatedDet.dispose();

        expect(gated.length, expectedKept,
            reason: 'threshold $threshold should keep exactly the baseline '
                'faces at or above it');
        for (final f in gated) {
          expect(f.meshScore, greaterThanOrEqualTo(threshold));
        }
        expect(gated.length, lessThanOrEqualTo(previousKept),
            reason: 'raising the threshold must never admit more faces');
        if (gated.length < baseline.length) droppedSomewhere = true;
        previousKept = gated.length;
      }

      expect(droppedSomewhere, isTrue,
          reason: 'no threshold in the sweep filtered anything, so this test '
              'proved only that the gate is harmless, not that it works');
    });

    test('has no effect in fast mode, where there is no mesh score', () async {
      final bytes = await loadBytes(groupShot);

      final baseDet = FaceDetector();
      await baseDet.initialize(minFacePresenceConfidence: 0.0);
      final baseline = await baseDet.detectFacesFromBytes(bytes, mode: mode);
      await baseDet.dispose();

      expect(baseline, isNotEmpty);
      for (final f in baseline) {
        expect(f.meshScore, isNull, reason: 'fast mode computes no mesh');
      }

      // A null presence score means "cannot evaluate", never "reject", so even
      // an impossible threshold must leave fast-mode results untouched.
      final strictDet = FaceDetector();
      await strictDet.initialize(minFacePresenceConfidence: 1.0);
      final strict = await strictDet.detectFacesFromBytes(bytes, mode: mode);
      await strictDet.dispose();

      expect(strict.length, baseline.length,
          reason: 'a null meshScore must always pass the gate');
    });

    test('a genuine face clears the default gate', () async {
      final detector = FaceDetector();
      await detector.initialize(); // default 0.5
      final faces = await detector.detectFacesFromBytes(
        await loadBytes(singleFace),
        mode: meshMode,
      );
      await detector.dispose();

      expect(faces, isNotEmpty,
          reason: 'the default gate must not reject a real face');
      for (final f in faces) {
        expect(f.meshScore, greaterThanOrEqualTo(0.5));
      }
    });

    test('rejects out-of-range and NaN thresholds', () async {
      for (final double bad in <double>[-0.1, 1.1, double.nan]) {
        final detector = FaceDetector();
        await expectLater(
          detector.initialize(minFacePresenceConfidence: bad),
          throwsArgumentError,
          reason: 'threshold $bad is outside [0.0, 1.0] and must fail fast',
        );
      }
    });
  });
}
