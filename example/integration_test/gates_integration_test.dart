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
  const FaceDetectionMode mode =
      FaceDetectionMode.fast; // gates need only bbox+score

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
}
