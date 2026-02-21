import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

import 'test_config.dart';

void main() {
  globalTestSetup();

  group('kMinSegmentationInputSize', () {
    test('should be 16', () {
      expect(kMinSegmentationInputSize, 16);
    });
  });

  group('SegmentationClass', () {
    test('class indices should be sequential 0-5', () {
      expect(SegmentationClass.background, 0);
      expect(SegmentationClass.hair, 1);
      expect(SegmentationClass.bodySkin, 2);
      expect(SegmentationClass.faceSkin, 3);
      expect(SegmentationClass.clothes, 4);
      expect(SegmentationClass.other, 5);
    });

    test('allPerson should contain all non-background classes', () {
      expect(SegmentationClass.allPerson, [1, 2, 3, 4, 5]);
      expect(SegmentationClass.allPerson, isNot(contains(0)));
    });
  });

  group('SegmentationConfig', () {
    test('default config should use general model', () {
      const config = SegmentationConfig();
      expect(config.model, SegmentationModel.general);
    });

    test('safe config should have validation enabled', () {
      expect(SegmentationConfig.safe.validateModel, isTrue);
    });

    test('should support custom configuration', () {
      final config = SegmentationConfig(
        model: SegmentationModel.multiclass,
        maxOutputSize: 512,
        validateModel: false,
      );
      expect(config.model, SegmentationModel.multiclass);
      expect(config.maxOutputSize, 512);
      expect(config.validateModel, isFalse);
    });
  });

  group('SegmentationModel', () {
    test('should have all expected models', () {
      expect(SegmentationModel.values.length, 3);
      expect(
        SegmentationModel.values.map((m) => m.name),
        containsAll(['general', 'landscape', 'multiclass']),
      );
    });
  });
}
