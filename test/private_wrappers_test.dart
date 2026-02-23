import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

void main() {
  globalTestSetup();

  group('testNormalizeEmbedding', () {
    test('should produce unit-length vector', () {
      final input = Float32List.fromList([3.0, 4.0]);
      final result = testNormalizeEmbedding(input);
      // L2 norm of [3,4] = 5, so normalized = [0.6, 0.8]
      expect(result[0], closeTo(0.6, 0.0001));
      expect(result[1], closeTo(0.8, 0.0001));

      // Verify unit length
      double norm = 0.0;
      for (int i = 0; i < result.length; i++) {
        norm += result[i] * result[i];
      }
      expect(math.sqrt(norm), closeTo(1.0, 0.0001));
    });

    test('should handle already-unit-norm vector', () {
      final input = Float32List.fromList([1.0, 0.0, 0.0]);
      final result = testNormalizeEmbedding(input);
      expect(result[0], closeTo(1.0, 0.0001));
      expect(result[1], closeTo(0.0, 0.0001));
      expect(result[2], closeTo(0.0, 0.0001));
    });

    test('should return original for zero vector', () {
      final input = Float32List.fromList([0.0, 0.0, 0.0]);
      final result = testNormalizeEmbedding(input);
      // Zero vector can't be normalized, returned as-is
      expect(result[0], 0.0);
      expect(result[1], 0.0);
      expect(result[2], 0.0);
    });

    test('should handle negative values', () {
      final input = Float32List.fromList([-3.0, 4.0]);
      final result = testNormalizeEmbedding(input);
      expect(result[0], closeTo(-0.6, 0.0001));
      expect(result[1], closeTo(0.8, 0.0001));
    });

    test('should handle single element', () {
      final input = Float32List.fromList([5.0]);
      final result = testNormalizeEmbedding(input);
      expect(result[0], closeTo(1.0, 0.0001));
    });

    test('should handle typical 192-dim embedding', () {
      final input = Float32List(192);
      for (int i = 0; i < 192; i++) {
        input[i] = (i - 96).toDouble();
      }
      final result = testNormalizeEmbedding(input);

      double norm = 0.0;
      for (int i = 0; i < result.length; i++) {
        norm += result[i] * result[i];
      }
      expect(math.sqrt(norm), closeTo(1.0, 0.0001));
    });
  });

  group('testTransformIrisToAbsolute', () {
    test('zero rotation, no flip', () {
      final roi = AlignedRoi(100, 100, 50, 0.0);
      final lmNorm = [
        [0.5, 0.5, 0.0], // center
      ];
      final result = testTransformIrisToAbsolute(lmNorm, roi, false);
      // center of normalized (0.5, 0.5) with zero rotation maps to (cx, cy)
      expect(result[0][0], closeTo(100.0, 0.0001));
      expect(result[0][1], closeTo(100.0, 0.0001));
      expect(result[0][2], 0.0);
    });

    test('zero rotation, corner point', () {
      final roi = AlignedRoi(100, 100, 50, 0.0);
      final lmNorm = [
        [0.0, 0.0, 1.0], // top-left
      ];
      final result = testTransformIrisToAbsolute(lmNorm, roi, false);
      // (0-0.5)*50 = -25, (0-0.5)*50 = -25
      // cx + (-25)*cos(0) - (-25)*sin(0) = 100 - 25 = 75
      // cy + (-25)*sin(0) + (-25)*cos(0) = 100 - 25 = 75
      expect(result[0][0], closeTo(75.0, 0.0001));
      expect(result[0][1], closeTo(75.0, 0.0001));
      expect(result[0][2], 1.0);
    });

    test('isRight flips x coordinate', () {
      final roi = AlignedRoi(100, 100, 50, 0.0);
      final lmNorm = [
        [0.2, 0.5, 0.0], // left of center
      ];
      final resultLeft = testTransformIrisToAbsolute(lmNorm, roi, false);
      final resultRight = testTransformIrisToAbsolute(lmNorm, roi, true);

      // For isRight, px = 1.0 - 0.2 = 0.8, so it flips horizontally
      // Left:  (0.2 - 0.5)*50 = -15  => cx + (-15) = 85
      // Right: (0.8 - 0.5)*50 = 15   => cx + 15 = 115
      expect(resultLeft[0][0], closeTo(85.0, 0.0001));
      expect(resultRight[0][0], closeTo(115.0, 0.0001));
      // Y should be same for both
      expect(resultLeft[0][1], closeTo(resultRight[0][1], 0.0001));
    });

    test('90-degree rotation', () {
      final roi = AlignedRoi(100, 100, 50, math.pi / 2);
      final lmNorm = [
        [1.0, 0.5, 0.0], // right of center
      ];
      final result = testTransformIrisToAbsolute(lmNorm, roi, false);
      // lx2 = (1.0 - 0.5)*50 = 25, ly2 = (0.5 - 0.5)*50 = 0
      // cos(pi/2) ≈ 0, sin(pi/2) ≈ 1
      // x = 100 + 25*0 - 0*1 = 100
      // y = 100 + 25*1 + 0*0 = 125
      expect(result[0][0], closeTo(100.0, 0.001));
      expect(result[0][1], closeTo(125.0, 0.001));
    });

    test('multiple landmarks', () {
      final roi = AlignedRoi(50, 50, 20, 0.0);
      final lmNorm = [
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
      ];
      final result = testTransformIrisToAbsolute(lmNorm, roi, false);
      expect(result.length, 3);
      expect(result[0][0], closeTo(50.0, 0.0001));
      expect(result[0][1], closeTo(50.0, 0.0001));
      expect(result[2][2], 2.0); // z preserved
    });

    test('empty landmarks list', () {
      final roi = AlignedRoi(100, 100, 50, 0.0);
      final result = testTransformIrisToAbsolute([], roi, false);
      expect(result, isEmpty);
    });
  });

  group('testFastExp', () {
    test('should return 0.0 for x < -20', () {
      expect(testFastExp(-21.0), 0.0);
      expect(testFastExp(-100.0), 0.0);
      expect(testFastExp(-20.1), 0.0);
    });

    test('should return clamped value for x > 20', () {
      expect(testFastExp(21.0), closeTo(485165195.4, 0.1));
      expect(testFastExp(100.0), closeTo(485165195.4, 0.1));
    });

    test('should match math.exp for normal range', () {
      for (double x = -20.0; x <= 20.0; x += 0.5) {
        expect(testFastExp(x), closeTo(math.exp(x), math.exp(x) * 0.01));
      }
    });

    test('should return 1.0 for x = 0', () {
      expect(testFastExp(0.0), closeTo(1.0, 0.0001));
    });

    test('boundary at exactly -20', () {
      // x = -20 is NOT < -20, so it should use math.exp
      expect(testFastExp(-20.0), closeTo(math.exp(-20.0), 1e-15));
    });

    test('boundary at exactly 20', () {
      // x = 20 is NOT > 20, so it should use math.exp
      expect(
          testFastExp(20.0), closeTo(math.exp(20.0), math.exp(20.0) * 0.001));
    });
  });

  group('testComputeClassProbabilities', () {
    test('equal logits should produce uniform probabilities', () {
      // 1 pixel, 6 channels all zero
      final raw = Float32List.fromList([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
      final result = testComputeClassProbabilities(raw, 1, 1);
      expect(result.length, 6);
      for (int i = 0; i < 6; i++) {
        expect(result[i], closeTo(1.0 / 6.0, 0.0001));
      }
    });

    test('dominant logit should produce high probability', () {
      // One very large logit
      final raw = Float32List.fromList([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
      final result = testComputeClassProbabilities(raw, 1, 1);
      expect(result[0], greaterThan(0.99));
      for (int i = 1; i < 6; i++) {
        expect(result[i], lessThan(0.01));
      }
    });

    test('probabilities should sum to 1 for each pixel', () {
      // 2x2 with various values
      final raw = Float32List.fromList([
        1.0, 2.0, 3.0, 0.5, -1.0, 0.0, // pixel 0
        -5.0, 10.0, 0.0, 0.0, 0.0, 0.0, // pixel 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // pixel 2
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, // pixel 3
      ]);
      final result = testComputeClassProbabilities(raw, 2, 2);
      expect(result.length, 24);

      for (int p = 0; p < 4; p++) {
        double sum = 0.0;
        for (int c = 0; c < 6; c++) {
          sum += result[p * 6 + c];
          expect(result[p * 6 + c], greaterThanOrEqualTo(0.0));
          expect(result[p * 6 + c], lessThanOrEqualTo(1.0));
        }
        expect(sum, closeTo(1.0, 0.0001));
      }
    });

    test('should handle large negative logits', () {
      final raw = Float32List.fromList(
          [-100.0, -100.0, -100.0, -100.0, -100.0, -100.0]);
      final result = testComputeClassProbabilities(raw, 1, 1);
      // All equal, so uniform
      double sum = 0.0;
      for (int i = 0; i < 6; i++) {
        sum += result[i];
      }
      expect(sum, closeTo(1.0, 0.0001));
    });

    test('order of probabilities matches order of logits', () {
      // Logits in ascending order
      final raw = Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      final result = testComputeClassProbabilities(raw, 1, 1);
      // Probabilities should be in ascending order too
      for (int i = 0; i < 5; i++) {
        expect(result[i + 1], greaterThan(result[i]));
      }
    });
  });

  group('testSerializeMask / testDeserializeMask', () {
    SegmentationMask createMask({
      List<double>? data,
      int width = 2,
      int height = 2,
    }) {
      return SegmentationMask(
        data: Float32List.fromList(data ?? [0.1, 0.5, 0.8, 1.0]),
        width: width,
        height: height,
        originalWidth: width * 2,
        originalHeight: height * 2,
      );
    }

    test('float32 format round-trip', () {
      final mask = createMask();
      final serialized =
          testSerializeMask(mask, IsolateOutputFormat.float32, 0.5);
      expect(serialized['dataFormat'], 'float32');
      expect(serialized['width'], 2);
      expect(serialized['height'], 2);
      expect(serialized['originalWidth'], 4);
      expect(serialized['originalHeight'], 4);

      final restored = testDeserializeMask(serialized);
      expect(restored.width, 2);
      expect(restored.height, 2);
      expect(restored.data[0], closeTo(0.1, 0.0001));
      expect(restored.data[3], closeTo(1.0, 0.0001));
    });

    test('uint8 format round-trip', () {
      final mask = createMask();
      final serialized =
          testSerializeMask(mask, IsolateOutputFormat.uint8, 0.5);
      expect(serialized['dataFormat'], 'uint8');
      final data = serialized['data'] as List;
      // 0.1*255 ≈ 26, 0.5*255 ≈ 128, 0.8*255 ≈ 204, 1.0*255 = 255
      expect(data[0], closeTo(26, 1));
      expect(data[1], closeTo(128, 1));
      expect(data[3], 255);
    });

    test('binary format round-trip', () {
      final mask = createMask();
      final serialized =
          testSerializeMask(mask, IsolateOutputFormat.binary, 0.5);
      expect(serialized['dataFormat'], 'binary');
      expect(serialized['binaryThreshold'], 0.5);
      final data = serialized['data'] as List;
      expect(data[0], 0); // 0.1 < 0.5
      expect(data[1], 255); // 0.5 >= 0.5
      expect(data[2], 255); // 0.8 >= 0.5
      expect(data[3], 255); // 1.0 >= 0.5
    });

    test('binary format with custom threshold', () {
      final mask = createMask();
      final serialized =
          testSerializeMask(mask, IsolateOutputFormat.binary, 0.9);
      final data = serialized['data'] as List;
      expect(data[0], 0); // 0.1 < 0.9
      expect(data[1], 0); // 0.5 < 0.9
      expect(data[2], 0); // 0.8 < 0.9
      expect(data[3], 255); // 1.0 >= 0.9
    });

    test('MulticlassSegmentationMask preserves classData', () {
      final classData = Float32List.fromList([
        // 1 pixel, 6 classes
        0.1, 0.2, 0.3, 0.15, 0.15, 0.1,
      ]);
      final mask = MulticlassSegmentationMask(
        data: Float32List.fromList([0.5]),
        width: 1,
        height: 1,
        originalWidth: 2,
        originalHeight: 2,
        classData: classData,
      );
      final serialized =
          testSerializeMask(mask, IsolateOutputFormat.float32, 0.5);
      expect(serialized['classData'], isNotNull);
      expect((serialized['classData'] as List).length, 6);

      final restored = testDeserializeMask(serialized);
      expect(restored, isA<MulticlassSegmentationMask>());
      final multi = restored as MulticlassSegmentationMask;
      expect(multi.width, 1);
      expect(multi.height, 1);
    });

    test('standard mask has no classData', () {
      final mask = createMask();
      final serialized =
          testSerializeMask(mask, IsolateOutputFormat.float32, 0.5);
      expect(serialized.containsKey('classData'), isFalse);

      final restored = testDeserializeMask(serialized);
      expect(restored, isNot(isA<MulticlassSegmentationMask>()));
    });

    test('preserves padding', () {
      final mask = SegmentationMask(
        data: Float32List.fromList([0.5, 0.5, 0.5, 0.5]),
        width: 2,
        height: 2,
        originalWidth: 4,
        originalHeight: 4,
        padding: [0.1, 0.2, 0.3, 0.4],
      );
      final serialized =
          testSerializeMask(mask, IsolateOutputFormat.float32, 0.5);
      expect(serialized['padding'], [0.1, 0.2, 0.3, 0.4]);
    });
  });
}
