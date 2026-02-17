import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

import 'test_config.dart';

void main() {
  globalTestSetup();

  group('FaceDetectorIsolate constructor and state', () {
    test('should not be ready before spawn', () {
      // FaceDetectorIsolate has a private constructor, so we test via spawn
      // But we can test isReady/isSegmentationReady via error paths
    });
  });

  group('FaceDetectorIsolate.dispose', () {
    test('should handle dispose on freshly created instance', () async {
      // Can't easily test without spawn since constructor is private
      // But we can test that it doesn't crash when fields are null
    });
  });

  group('IsolateOutputFormat', () {
    test('should have all expected formats', () {
      expect(IsolateOutputFormat.values.length, 3);
      expect(
        IsolateOutputFormat.values.map((f) => f.name),
        containsAll(['float32', 'uint8', 'binary']),
      );
    });
  });

  group('DetectionWithSegmentationResult', () {
    test('should store faces, mask, and timing data', () {
      final mask = SegmentationMask(
        data: Float32List.fromList([0.5, 0.8]),
        width: 2,
        height: 1,
        originalWidth: 100,
        originalHeight: 100,
        padding: [0.0, 0.0, 0.0, 0.0],
      );

      final result = DetectionWithSegmentationResult(
        faces: [],
        segmentationMask: mask,
        detectionTimeMs: 10,
        segmentationTimeMs: 20,
      );

      expect(result.faces, isEmpty);
      expect(result.segmentationMask, isNotNull);
      expect(result.detectionTimeMs, 10);
      expect(result.segmentationTimeMs, 20);
      expect(result.totalTimeMs, 20); // max of the two
    });

    test('totalTimeMs should be max of detection and segmentation', () {
      final mask = SegmentationMask(
        data: Float32List.fromList([0.5]),
        width: 1,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
        padding: [0.0, 0.0, 0.0, 0.0],
      );

      final result = DetectionWithSegmentationResult(
        faces: [],
        segmentationMask: mask,
        detectionTimeMs: 50,
        segmentationTimeMs: 20,
      );

      expect(result.totalTimeMs, 50);
    });
  });

  group('SegmentationMask serialization', () {
    test('SegmentationMask.fromMap should reconstruct from map', () {
      final data = Float32List.fromList([0.1, 0.5, 0.9, 0.2]);
      final mask = SegmentationMask(
        data: data,
        width: 2,
        height: 2,
        originalWidth: 100,
        originalHeight: 200,
        padding: [0.1, 0.1, 0.05, 0.05],
      );

      final map = <String, dynamic>{
        'data': data.toList(),
        'dataFormat': 'float32',
        'width': 2,
        'height': 2,
        'originalWidth': 100,
        'originalHeight': 200,
        'padding': [0.1, 0.1, 0.05, 0.05],
      };

      final reconstructed = SegmentationMask.fromMap(map);

      expect(reconstructed.width, mask.width);
      expect(reconstructed.height, mask.height);
      expect(reconstructed.originalWidth, mask.originalWidth);
      expect(reconstructed.originalHeight, mask.originalHeight);
    });

    test('SegmentationMask toUint8 should convert to 0-255 range', () {
      final data = Float32List.fromList([0.0, 0.5, 1.0]);
      final mask = SegmentationMask(
        data: data,
        width: 3,
        height: 1,
        originalWidth: 100,
        originalHeight: 100,
        padding: [0.0, 0.0, 0.0, 0.0],
      );

      final uint8 = mask.toUint8();
      expect(uint8[0], 0);
      expect(uint8[1], 128); // 0.5 * 255 = 127.5 -> 128
      expect(uint8[2], 255);
    });

    test('SegmentationMask toBinary should threshold correctly', () {
      final data = Float32List.fromList([0.2, 0.5, 0.8]);
      final mask = SegmentationMask(
        data: data,
        width: 3,
        height: 1,
        originalWidth: 100,
        originalHeight: 100,
        padding: [0.0, 0.0, 0.0, 0.0],
      );

      final binary = mask.toBinary(threshold: 0.5);
      expect(binary[0], 0);
      expect(binary[1], 255);
      expect(binary[2], 255);
    });
  });

  group('MulticlassSegmentationMask', () {
    test('should store class data with person mask', () {
      final personMask = Float32List.fromList([0.9, 0.8]);
      final classData = Float32List.fromList([
        // pixel 0: bg=0.1, hair=0.3, body=0.2, face=0.2, clothes=0.1, other=0.1
        0.1, 0.3, 0.2, 0.2, 0.1, 0.1,
        // pixel 1: bg=0.2, hair=0.1, body=0.1, face=0.4, clothes=0.1, other=0.1
        0.2, 0.1, 0.1, 0.4, 0.1, 0.1,
      ]);

      final mask = MulticlassSegmentationMask(
        data: personMask,
        width: 2,
        height: 1,
        originalWidth: 100,
        originalHeight: 100,
        padding: [0.0, 0.0, 0.0, 0.0],
        classData: classData,
      );

      expect(mask.data.length, 2);

      // Test classMask access
      final bgMaskData = mask.classMask(SegmentationClass.background);
      expect(bgMaskData[0], closeTo(0.1, 0.001)); // bg at pixel 0
      final hairMaskData = mask.classMask(SegmentationClass.hair);
      expect(hairMaskData[0], closeTo(0.3, 0.001)); // hair at pixel 0

      // Test hair mask
      final hairMask = mask.hairMask;
      expect(hairMask.length, 2);
      expect(hairMask[0], closeTo(0.3, 0.001));

      // Test face skin mask
      final faceMask = mask.faceSkinMask;
      expect(faceMask.length, 2);
      expect(faceMask[0], closeTo(0.2, 0.001));
      expect(faceMask[1], closeTo(0.4, 0.001));

      // Test body skin mask
      final bodyMask = mask.bodySkinMask;
      expect(bodyMask.length, 2);

      // Test clothes mask
      final clothesMask = mask.clothesMask;
      expect(clothesMask.length, 2);

      // Test background mask
      final bgMask = mask.backgroundMask;
      expect(bgMask.length, 2);
    });
  });

  group('SegmentationConfig', () {
    test('should have reasonable defaults', () {
      const config = SegmentationConfig();
      expect(config.model, SegmentationModel.general);
      expect(config.maxOutputSize, greaterThan(0));
      expect(config.validateModel, isTrue);
    });

    test('safe config should have validation enabled', () {
      expect(SegmentationConfig.safe.validateModel, isTrue);
    });
  });

  group('SegmentationModel enum', () {
    test('should have general, landscape, and multiclass', () {
      expect(
        SegmentationModel.values.map((m) => m.name),
        containsAll(['general', 'landscape', 'multiclass']),
      );
    });
  });

  group('SegmentationClass constants', () {
    test('should have correct values', () {
      expect(SegmentationClass.background, 0);
      expect(SegmentationClass.hair, 1);
      expect(SegmentationClass.bodySkin, 2);
      expect(SegmentationClass.faceSkin, 3);
      expect(SegmentationClass.clothes, 4);
      expect(SegmentationClass.other, 5);
      expect(SegmentationClass.allPerson.length, 5);
      expect(SegmentationClass.allPerson, isNot(contains(0)));
    });
  });

  group('SegmentationException', () {
    test('should store error code and message', () {
      final exception = SegmentationException(
        SegmentationError.inferenceFailed,
        'test error',
      );
      expect(exception.code, SegmentationError.inferenceFailed);
      expect(exception.message, 'test error');
      expect(exception.toString(), contains('test error'));
    });

    test('should store optional cause', () {
      final cause = Exception('root cause');
      final exception = SegmentationException(
        SegmentationError.imageDecodeFailed,
        'decode failed',
        cause,
      );
      expect(exception.cause, cause);
    });
  });

  group('SegmentationError enum', () {
    test('should have all expected error codes', () {
      expect(
        SegmentationError.values.map((e) => e.name),
        containsAll([
          'modelNotFound',
          'interpreterCreationFailed',
          'imageDecodeFailed',
          'imageTooSmall',
          'inferenceFailed',
          'unexpectedTensorShape',
        ]),
      );
    });
  });
}
