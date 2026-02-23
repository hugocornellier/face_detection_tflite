import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

void main() {
  globalTestSetup();

  group('testComputeFaceAlignment', () {
    Detection makeDet({
      double leftEyeX = 0.3,
      double leftEyeY = 0.4,
      double rightEyeX = 0.7,
      double rightEyeY = 0.4,
      double mouthX = 0.5,
      double mouthY = 0.7,
    }) {
      return Detection(
        boundingBox: RectF(0.2, 0.2, 0.8, 0.8),
        score: 0.9,
        keypointsXY: [
          leftEyeX, leftEyeY,
          rightEyeX, rightEyeY,
          0.5, 0.5, // noseTip
          mouthX, mouthY,
          0.1, 0.4, // leftEyeTragion
          0.9, 0.4, // rightEyeTragion
        ],
      );
    }

    test('horizontal eyes should produce theta near 0', () {
      final result = testComputeFaceAlignment(
        makeDet(leftEyeX: 0.3, leftEyeY: 0.5, rightEyeX: 0.7, rightEyeY: 0.5),
        100.0,
        100.0,
      );
      expect(result.theta, closeTo(0.0, 0.01));
    });

    test('tilted eyes should produce non-zero theta', () {
      final result = testComputeFaceAlignment(
        makeDet(leftEyeX: 0.3, leftEyeY: 0.5, rightEyeX: 0.7, rightEyeY: 0.3),
        100.0,
        100.0,
      );
      expect(result.theta, lessThan(0.0)); // tilted upward to the right
    });

    test('size should be proportional to eye distance', () {
      final narrow = testComputeFaceAlignment(
        makeDet(
            leftEyeX: 0.4,
            leftEyeY: 0.5,
            rightEyeX: 0.6,
            rightEyeY: 0.5,
            mouthX: 0.5,
            mouthY: 0.6),
        100.0,
        100.0,
      );
      final wide = testComputeFaceAlignment(
        makeDet(
            leftEyeX: 0.2,
            leftEyeY: 0.5,
            rightEyeX: 0.8,
            rightEyeY: 0.5,
            mouthX: 0.5,
            mouthY: 0.6),
        100.0,
        100.0,
      );
      expect(wide.size, greaterThan(narrow.size));
    });

    test('center should be influenced by eye and mouth position', () {
      final result = testComputeFaceAlignment(
        makeDet(
            leftEyeX: 0.3,
            leftEyeY: 0.3,
            rightEyeX: 0.7,
            rightEyeY: 0.3,
            mouthX: 0.5,
            mouthY: 0.7),
        100.0,
        100.0,
      );
      // Center should be between eye center and mouth
      expect(result.cx, closeTo(50.0, 5.0));
      expect(result.cy, greaterThan(30.0)); // shifted down from eye center
    });

    test('should scale with image dimensions', () {
      final det = makeDet();
      final small = testComputeFaceAlignment(det, 100.0, 100.0);
      final large = testComputeFaceAlignment(det, 200.0, 200.0);
      expect(large.size, closeTo(small.size * 2, 1.0));
      expect(large.cx, closeTo(small.cx * 2, 1.0));
    });
  });

  group('testTransformMeshToAbsolute', () {
    test('identity transform (theta=0, size=1, center=0.5,0.5)', () {
      final lmNorm = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
      ];
      final result = testTransformMeshToAbsolute(lmNorm, 0.5, 0.5, 1.0, 0.0);
      expect(result.length, 3);
      // At theta=0: tx = cx - 0.5*size = 0.0, ty = cy - 0.5*size = 0.0
      expect(result[0].x, closeTo(0.0, 0.0001));
      expect(result[0].y, closeTo(0.0, 0.0001));
      expect(result[1].x, closeTo(1.0, 0.0001));
      expect(result[1].y, closeTo(1.0, 0.0001));
      expect(result[2].x, closeTo(0.5, 0.0001));
      expect(result[2].y, closeTo(0.5, 0.0001));
    });

    test('90 degree rotation', () {
      // Verify rotation effect: a point at normalized (1, 0) should rotate
      // sct = size*cos(theta) ≈ 0, sst = size*sin(theta) ≈ 100
      // tx = 50 - 0.5*0 + 0.5*100 = 100
      // ty = 50 - 0.5*100 - 0.5*0 = 0
      // x = tx + sct*1.0 - sst*0.0 = 100 + 0 - 0 = 100
      // y = ty + sst*1.0 + sct*0.0 = 0 + 100 + 0 = 100
      final lmNorm = [
        [1.0, 0.0, 0.0],
      ];
      final result =
          testTransformMeshToAbsolute(lmNorm, 50.0, 50.0, 100.0, math.pi / 2);
      expect(result[0].x, closeTo(100.0, 0.5));
      expect(result[0].y, closeTo(100.0, 0.5));
    });

    test('translation with size=100', () {
      final lmNorm = [
        [0.5, 0.5, 0.0], // center in normalized space
      ];
      final result =
          testTransformMeshToAbsolute(lmNorm, 200.0, 300.0, 100.0, 0.0);
      // tx = 200 - 50 = 150, ty = 300 - 50 = 250
      // x = 150 + 100*0.5 = 200, y = 250 + 100*0.5 = 300
      expect(result[0].x, closeTo(200.0, 0.0001));
      expect(result[0].y, closeTo(300.0, 0.0001));
    });

    test('Z coordinate is scaled by size', () {
      final lmNorm = [
        [0.5, 0.5, 0.1],
      ];
      final result = testTransformMeshToAbsolute(lmNorm, 0.0, 0.0, 200.0, 0.0);
      expect(result[0].z, closeTo(0.1 * 200.0, 0.0001));
    });

    test('should handle empty input', () {
      final result = testTransformMeshToAbsolute([], 50.0, 50.0, 100.0, 0.0);
      expect(result, isEmpty);
    });
  });

  group('_InferenceLock (via testCreateInferenceLockRunner)', () {
    test('should execute single task', () async {
      final run = testCreateInferenceLockRunner();
      final result = await run<int>(() async => 42);
      expect(result, 42);
    });

    test('should serialize concurrent tasks', () async {
      final run = testCreateInferenceLockRunner();
      final order = <int>[];

      final f1 = run<void>(() async {
        await Future.delayed(const Duration(milliseconds: 50));
        order.add(1);
      });
      final f2 = run<void>(() async {
        order.add(2);
      });

      await Future.wait([f1, f2]);
      expect(order, [1, 2]); // f1 should complete before f2 starts
    });

    test('should not block subsequent tasks after exception', () async {
      final run = testCreateInferenceLockRunner();

      try {
        await run<void>(() async {
          throw Exception('test error');
        });
      } catch (_) {}

      // Should still work after exception
      final result = await run<int>(() async => 99);
      expect(result, 99);
    });

    test('should maintain order across many tasks', () async {
      final run = testCreateInferenceLockRunner();
      final order = <int>[];

      final futures = List.generate(5, (i) {
        return run<void>(() async {
          await Future.delayed(Duration(milliseconds: (5 - i) * 10));
          order.add(i);
        });
      });

      await Future.wait(futures);
      expect(order, [0, 1, 2, 3, 4]);
    });
  });

  group('testFindIrisCenterFromPoints', () {
    test('should return (0,0,0) for empty list', () {
      final result = testFindIrisCenterFromPoints([]);
      expect(result.x, 0);
      expect(result.y, 0);
      expect(result.z, 0);
    });

    test('should return the single point for length-1 list', () {
      final result = testFindIrisCenterFromPoints([const Point(42, 99, 5)]);
      expect(result.x, 42);
      expect(result.y, 99);
      expect(result.z, 5);
    });

    test('should find center point closest to centroid', () {
      // 5 iris contour points forming a diamond with center at (10, 10)
      final points = [
        const Point(10, 10), // center - closest to centroid
        const Point(10, 8), // top
        const Point(12, 10), // right
        const Point(10, 12), // bottom
        const Point(8, 10), // left
      ];
      final result = testFindIrisCenterFromPoints(points);
      expect(result.x, 10);
      expect(result.y, 10);
    });

    test('should pick point closest to centroid when no exact center', () {
      // Asymmetric points: centroid ≈ (5.33, 5.33)
      final points = [
        const Point(2, 2),
        const Point(4, 4),
        const Point(10, 10),
      ];
      final result = testFindIrisCenterFromPoints(points);
      // centroid = (16/3, 16/3) ≈ (5.33, 5.33), closest is (4, 4)
      expect(result.x, 4);
      expect(result.y, 4);
    });

    test('should handle typical 5-point iris contour', () {
      // Typical iris: center + 4 cardinal points
      final points = [
        const Point(100.0, 100.0, 0.5), // actual center
        const Point(100.0, 95.0, 0.5), // top
        const Point(105.0, 100.0, 0.5), // right
        const Point(100.0, 105.0, 0.5), // bottom
        const Point(95.0, 100.0, 0.5), // left
      ];
      final result = testFindIrisCenterFromPoints(points);
      expect(result.x, 100.0);
      expect(result.y, 100.0);
    });

    test('should handle two points', () {
      final points = [
        const Point(0, 0),
        const Point(10, 10),
      ];
      final result = testFindIrisCenterFromPoints(points);
      // Centroid is (5, 5), both equidistant, should return first (idx 0)
      expect(result.x, 0);
      expect(result.y, 0);
    });
  });
}
