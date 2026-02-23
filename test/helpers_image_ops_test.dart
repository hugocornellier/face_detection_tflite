import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_config.dart';

void main() {
  globalTestSetup();

  group('faceDetectionToRoi', () {
    test('expands and centers bounding box to square ROI', () {
      final bbox = RectF(0.2, 0.3, 0.6, 0.7);

      final roi = faceDetectionToRoi(bbox, expandFraction: 0.6);

      expect(roi.xmin, closeTo(0.08, 1e-6));
      expect(roi.ymin, closeTo(0.18, 1e-6));
      expect(roi.xmax, closeTo(0.72, 1e-6));
      expect(roi.ymax, closeTo(0.82, 1e-6));
      expect(roi.w, closeTo(0.64, 1e-6));
      expect(roi.h, closeTo(0.64, 1e-6));
    });
  });
}
