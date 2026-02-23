import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

import 'test_config.dart';

void main() {
  globalTestSetup();

  group('FaceDetectionModel enum', () {
    test('should have all expected model variants', () {
      expect(FaceDetectionModel.values.length, greaterThanOrEqualTo(3));
      expect(
        FaceDetectionModel.values.map((m) => m.name),
        containsAll(['backCamera', 'frontCamera', 'shortRange']),
      );
    });

    test('each model should have a name', () {
      for (final model in FaceDetectionModel.values) {
        expect(model.name, isNotEmpty);
      }
    });
  });

  group('FaceDetectionMode enum', () {
    test('should have fast, standard, and full modes', () {
      expect(FaceDetectionMode.values, contains(FaceDetectionMode.fast));
      expect(FaceDetectionMode.values, contains(FaceDetectionMode.standard));
      expect(FaceDetectionMode.values, contains(FaceDetectionMode.full));
    });
  });

  group('PerformanceConfig', () {
    test('should have default auto mode', () {
      const config = PerformanceConfig();
      expect(config.mode, PerformanceMode.auto);
      expect(config.numThreads, isNull);
    });

    test('should create disabled config', () {
      expect(PerformanceConfig.disabled.mode, PerformanceMode.disabled);
    });

    test('should create xnnpack config', () {
      final config = PerformanceConfig.xnnpack();
      expect(config.mode, PerformanceMode.xnnpack);
    });

    test('should create xnnpack config with custom threads', () {
      final config = PerformanceConfig.xnnpack(numThreads: 2);
      expect(config.mode, PerformanceMode.xnnpack);
      expect(config.numThreads, 2);
    });
  });

  group('PerformanceMode enum', () {
    test('should have all expected modes', () {
      expect(PerformanceMode.values, contains(PerformanceMode.disabled));
      expect(PerformanceMode.values, contains(PerformanceMode.auto));
      expect(PerformanceMode.values, contains(PerformanceMode.xnnpack));
      expect(PerformanceMode.values, contains(PerformanceMode.gpu));
    });
  });

  group('DecodedBox', () {
    test('should store bounding box and keypoints', () {
      final box = DecodedBox(RectF(0.1, 0.2, 0.8, 0.9), [0.3, 0.4, 0.5, 0.6]);
      expect(box.boundingBox.xmin, 0.1);
      expect(box.boundingBox.ymin, 0.2);
      expect(box.keypointsXY.length, 4);
    });
  });

  group('FaceLandmarkType', () {
    test('should have correct indices', () {
      expect(FaceLandmarkType.leftEye.index, 0);
      expect(FaceLandmarkType.rightEye.index, 1);
    });

    test('should have all 6 landmarks', () {
      expect(FaceLandmarkType.values.length, 6);
    });
  });
}
