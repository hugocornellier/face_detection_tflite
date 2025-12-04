import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart' show Size;
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

/// Tests for pure Dart helper functions in helpers.dart
/// These tests don't require TFLite models or image processing
void main() {
  globalTestSetup();

  group('Tensor Allocation - createNHWCTensor4D', () {
    test('should create 4D tensor with correct dimensions', () {
      final tensor = createNHWCTensor4D(128, 128);

      expect(tensor.length, 1); // Batch dimension
      expect(tensor[0].length, 128); // Height
      expect(tensor[0][0].length, 128); // Width
      expect(tensor[0][0][0].length, 3); // Channels (RGB)
    });

    test('should initialize all values to zero', () {
      final tensor = createNHWCTensor4D(2, 2);

      for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 2; w++) {
          for (int c = 0; c < 3; c++) {
            expect(tensor[0][h][w][c], 0.0);
          }
        }
      }
    });

    test('should create different sized tensors', () {
      final small = createNHWCTensor4D(32, 32);
      final medium = createNHWCTensor4D(128, 128);
      final large = createNHWCTensor4D(256, 256);

      expect(small[0].length, 32);
      expect(medium[0].length, 128);
      expect(large[0].length, 256);
    });

    test('should create non-square tensors', () {
      final tensor = createNHWCTensor4D(100, 200);

      expect(tensor[0].length, 100); // Height
      expect(tensor[0][0].length, 200); // Width
    });
  });

  group('Tensor Filling - fillNHWC4D', () {
    test('should fill tensor from flat array correctly', () {
      final flat = Float32List.fromList([
        1.0, 2.0, 3.0, // Pixel (0, 0)
        4.0, 5.0, 6.0, // Pixel (0, 1)
        7.0, 8.0, 9.0, // Pixel (1, 0)
        10.0, 11.0, 12.0, // Pixel (1, 1)
      ]);

      final tensor = createNHWCTensor4D(2, 2);
      fillNHWC4D(flat, tensor, 2, 2);

      // Check pixel (0, 0)
      expect(tensor[0][0][0][0], 1.0);
      expect(tensor[0][0][0][1], 2.0);
      expect(tensor[0][0][0][2], 3.0);

      // Check pixel (0, 1)
      expect(tensor[0][0][1][0], 4.0);
      expect(tensor[0][0][1][1], 5.0);
      expect(tensor[0][0][1][2], 6.0);

      // Check pixel (1, 0)
      expect(tensor[0][1][0][0], 7.0);
      expect(tensor[0][1][0][1], 8.0);
      expect(tensor[0][1][0][2], 9.0);

      // Check pixel (1, 1)
      expect(tensor[0][1][1][0], 10.0);
      expect(tensor[0][1][1][1], 11.0);
      expect(tensor[0][1][1][2], 12.0);
    });

    test('should handle row-major order correctly', () {
      // Create 3x2 image (3 rows, 2 columns)
      final flat = Float32List.fromList([
        1.0, 1.0, 1.0, 2.0, 2.0, 2.0, // Row 0
        3.0, 3.0, 3.0, 4.0, 4.0, 4.0, // Row 1
        5.0, 5.0, 5.0, 6.0, 6.0, 6.0, // Row 2
      ]);

      final tensor = createNHWCTensor4D(3, 2);
      fillNHWC4D(flat, tensor, 3, 2);

      // Verify row 0
      expect(tensor[0][0][0][0], 1.0);
      expect(tensor[0][0][1][0], 2.0);

      // Verify row 1
      expect(tensor[0][1][0][0], 3.0);
      expect(tensor[0][1][1][0], 4.0);

      // Verify row 2
      expect(tensor[0][2][0][0], 5.0);
      expect(tensor[0][2][1][0], 6.0);
    });

    test('should handle single pixel', () {
      final flat = Float32List.fromList([0.5, 0.6, 0.7]);
      final tensor = createNHWCTensor4D(1, 1);
      fillNHWC4D(flat, tensor, 1, 1);

      expect(tensor[0][0][0][0], 0.5);
      expect(tensor[0][0][0][1], 0.6);
      expect(tensor[0][0][0][2], 0.7);
    });
  });

  group('Tensor Allocation - allocTensorShape', () {
    test('should allocate 1D tensor', () {
      final tensor = allocTensorShape([5]) as List<double>;

      expect(tensor.length, 5);
      expect(tensor, everyElement(equals(0.0)));
    });

    test('should allocate 2D tensor', () {
      final tensor = allocTensorShape([3, 4]) as List<List<double>>;

      expect(tensor.length, 3);
      expect(tensor[0].length, 4);
      expect(tensor[2][3], 0.0);
    });

    test('should allocate 3D tensor', () {
      final tensor = allocTensorShape([2, 3, 4]) as List<List<List<double>>>;

      expect(tensor.length, 2);
      expect(tensor[0].length, 3);
      expect(tensor[0][0].length, 4);
      expect(tensor[1][2][3], 0.0);
    });

    test('should allocate 4D tensor', () {
      final tensor =
          allocTensorShape([1, 2, 3, 4]) as List<List<List<List<double>>>>;

      expect(tensor.length, 1);
      expect(tensor[0].length, 2);
      expect(tensor[0][0].length, 3);
      expect(tensor[0][0][0].length, 4);
      expect(tensor[0][1][2][3], 0.0);
    });

    test('should handle empty shape', () {
      final tensor = allocTensorShape([]) as List<double>;

      expect(tensor, isEmpty);
    });

    test('should initialize all values to zero', () {
      final tensor = allocTensorShape([2, 2]) as List<List<double>>;

      for (var row in tensor) {
        for (var val in row) {
          expect(val, 0.0);
        }
      }
    });
  });

  group('Tensor Flattening - flattenDynamicTensor', () {
    test('should flatten 1D list', () {
      final input = [1.0, 2.0, 3.0, 4.0];
      final result = flattenDynamicTensor(input);

      expect(result.length, 4);
      expect(result[0], 1.0);
      expect(result[3], 4.0);
    });

    test('should flatten 2D list', () {
      final input = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
      ];
      final result = flattenDynamicTensor(input);

      expect(result.length, 6);
      expect(result[0], 1.0);
      expect(result[5], 6.0);
    });

    test('should flatten 3D list', () {
      final input = [
        [
          [1.0, 2.0],
          [3.0, 4.0]
        ],
        [
          [5.0, 6.0],
          [7.0, 8.0]
        ]
      ];
      final result = flattenDynamicTensor(input);

      expect(result.length, 8);
      expect(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    });

    test('should flatten 4D list (NHWC format)', () {
      final input = [
        [
          [
            [1.0, 2.0, 3.0]
          ]
        ]
      ];
      final result = flattenDynamicTensor(input);

      expect(result.length, 3);
      expect(result, [1.0, 2.0, 3.0]);
    });

    test('should throw on null input', () {
      expect(() => flattenDynamicTensor(null), throwsA(isA<TypeError>()));
    });

    test('should handle empty nested lists', () {
      final input = [[]];
      final result = flattenDynamicTensor(input);

      expect(result, isEmpty);
    });

    test('should preserve order in flattening', () {
      final input = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ];
      final result = flattenDynamicTensor(input);

      expect(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    });
  });

  group('RectF Utilities', () {
    test('should create normalized rectangle', () {
      final rect = RectF(0.1, 0.2, 0.9, 0.8);

      expect(rect.xmin, 0.1);
      expect(rect.ymin, 0.2);
      expect(rect.xmax, 0.9);
      expect(rect.ymax, 0.8);
    });

    test('should calculate width and height', () {
      final rect = RectF(0.2, 0.3, 0.7, 0.8);

      expect(rect.w, closeTo(0.5, 0.0001));
      expect(rect.h, closeTo(0.5, 0.0001));
    });

    test('should expand rectangle uniformly', () {
      final rect = RectF(0.4, 0.4, 0.6, 0.6);
      final expanded = rect.expand(0.5);

      // Original size: 0.2 x 0.2
      // Expanded size: 0.3 x 0.3 (1.5x)
      expect(expanded.w, closeTo(0.3, 0.0001));
      expect(expanded.h, closeTo(0.3, 0.0001));

      // Center should stay at (0.5, 0.5)
      final cx = (expanded.xmin + expanded.xmax) / 2;
      final cy = (expanded.ymin + expanded.ymax) / 2;
      expect(cx, closeTo(0.5, 0.0001));
      expect(cy, closeTo(0.5, 0.0001));
    });

    test('should scale rectangle', () {
      final rect = RectF(0.1, 0.2, 0.3, 0.6);
      final scaled = rect.scale(2.0, 3.0);

      expect(scaled.xmin, closeTo(0.2, 0.0001));
      expect(scaled.ymin, closeTo(0.6, 0.0001));
      expect(scaled.xmax, closeTo(0.6, 0.0001));
      expect(scaled.ymax, closeTo(1.8, 0.0001));
    });

    test('should handle zero-size rectangle', () {
      final rect = RectF(0.5, 0.5, 0.5, 0.5);

      expect(rect.w, 0.0);
      expect(rect.h, 0.0);

      final expanded = rect.expand(2.0);
      expect(expanded.w, 0.0);
      expect(expanded.h, 0.0);
    });
  });

  group('OutputTensorInfo', () {
    test('should store tensor metadata', () {
      final shape = [1, 128, 128, 3];
      final buffer = Float32List(128 * 128 * 3);

      final info = OutputTensorInfo(shape, buffer);

      expect(info.shape, equals(shape));
      expect(info.buffer, equals(buffer));
      expect(info.buffer.length, 128 * 128 * 3);
    });

    test('should handle different shapes', () {
      final shape1D = [100];
      final buffer1D = Float32List(100);
      final info1D = OutputTensorInfo(shape1D, buffer1D);

      expect(info1D.shape.length, 1);
      expect(info1D.shape[0], 100);

      final shape4D = [1, 32, 32, 64];
      final buffer4D = Float32List(32 * 32 * 64);
      final info4D = OutputTensorInfo(shape4D, buffer4D);

      expect(info4D.shape.length, 4);
      expect(info4D.shape[3], 64);
    });
  });

  group('AlignedRoi', () {
    test('should store ROI metadata', () {
      final roi = AlignedRoi(100.0, 100.0, 50.0, 0.5);

      expect(roi.cx, 100.0);
      expect(roi.cy, 100.0);
      expect(roi.size, 50.0);
      expect(roi.theta, 0.5);
    });

    test('should handle negative rotation', () {
      final roi = AlignedRoi(50.0, 50.0, 30.0, -1.57); // ~-90 degrees

      expect(roi.theta, -1.57);
    });

    test('should handle zero size', () {
      final roi = AlignedRoi(100.0, 100.0, 0.0, 0.0);

      expect(roi.size, 0.0);
    });
  });

  group('DecodedBox', () {
    test('should store detection box and keypoints', () {
      final bbox = RectF(0.2, 0.3, 0.8, 0.7);
      final keypoints = [0.3, 0.4, 0.7, 0.4, 0.5, 0.6];

      final box = DecodedBox(bbox, keypoints);

      expect(box.boundingBox, equals(bbox));
      expect(box.keypointsXY, equals(keypoints));
      expect(box.keypointsXY.length, 6);
    });

    test('should handle empty keypoints', () {
      final bbox = RectF(0.2, 0.3, 0.8, 0.7);
      final box = DecodedBox(bbox, []);

      expect(box.keypointsXY, isEmpty);
    });
  });

  group('Point Utilities', () {
    test('should create and compare points', () {
      final p1 = Point(10.0, 20.0);
      final p2 = Point(10.0, 20.0);
      final p3 = Point(10.0, 20.0, 5.0);

      expect(p1 == p2, true);
      expect(p1 == p3, false);
      expect(p1.hashCode, equals(p2.hashCode));
    });

    test('should detect 2D vs 3D points', () {
      final point2D = Point(10.0, 20.0);
      final point3D = Point(10.0, 20.0, 5.0);

      expect(point2D.is3D, false);
      expect(point3D.is3D, true);
    });

    test('should format toString correctly', () {
      final point2D = Point(10.0, 20.0);
      final point3D = Point(10.0, 20.0, 5.0);

      expect(point2D.toString(), 'Point(10.0, 20.0)');
      expect(point3D.toString(), 'Point(10.0, 20.0, 5.0)');
    });
  });

  group('BoundingBox Geometry', () {
    test('should calculate dimensions correctly', () {
      final bbox = BoundingBox(
        topLeft: Point(10.0, 20.0),
        topRight: Point(90.0, 20.0),
        bottomRight: Point(90.0, 80.0),
        bottomLeft: Point(10.0, 80.0),
      );

      expect(bbox.width, 80.0);
      expect(bbox.height, 60.0);
    });

    test('should calculate center point', () {
      final bbox = BoundingBox(
        topLeft: Point(0.0, 0.0),
        topRight: Point(100.0, 0.0),
        bottomRight: Point(100.0, 100.0),
        bottomLeft: Point(0.0, 100.0),
      );

      final center = bbox.center;
      expect(center.x, 50.0);
      expect(center.y, 50.0);
    });

    test('should provide corners in order', () {
      final topLeft = Point(10.0, 20.0);
      final topRight = Point(90.0, 20.0);
      final bottomRight = Point(90.0, 80.0);
      final bottomLeft = Point(10.0, 80.0);

      final bbox = BoundingBox(
        topLeft: topLeft,
        topRight: topRight,
        bottomRight: bottomRight,
        bottomLeft: bottomLeft,
      );

      final corners = bbox.corners;
      expect(corners[0], equals(topLeft));
      expect(corners[1], equals(topRight));
      expect(corners[2], equals(bottomRight));
      expect(corners[3], equals(bottomLeft));
    });

    test('should handle rotated bounding box', () {
      // Diamond shape
      final bbox = BoundingBox(
        topLeft: Point(50.0, 0.0),
        topRight: Point(100.0, 50.0),
        bottomRight: Point(50.0, 100.0),
        bottomLeft: Point(0.0, 50.0),
      );

      // Width/height calculated from differences
      expect(bbox.width, 50.0); // topRight.x - topLeft.x
      expect(bbox.height, 50.0); // bottomLeft.y - topLeft.y
      expect(bbox.center.x, 50.0);
      expect(bbox.center.y, 50.0);
    });
  });

  group('Coordinate Validation', () {
    test('should validate pixel coordinates within bounds', () {
      final point = Point(320.0, 240.0);
      final imageSize = Size(640, 480);

      expect(
        TestUtils.isValidPixelCoordinate(point, imageSize),
        true,
      );
    });

    test('should invalidate pixel coordinates outside bounds', () {
      final point = Point(700.0, 240.0);
      final imageSize = Size(640, 480);

      expect(
        TestUtils.isValidPixelCoordinate(point, imageSize),
        false,
      );
    });

    test('should handle boundary coordinates', () {
      final point = Point(640.0, 480.0);
      final imageSize = Size(640, 480);

      expect(
        TestUtils.isValidPixelCoordinate(point, imageSize),
        true,
      );
    });

    test('should handle negative coordinates', () {
      final point = Point(-10.0, 240.0);
      final imageSize = Size(640, 480);

      expect(
        TestUtils.isValidPixelCoordinate(point, imageSize),
        false,
      );
    });
  });

  group('Floating Point Comparison', () {
    test('should detect approximately equal doubles', () {
      expect(
        TestUtils.approximatelyEqual(0.1 + 0.2, 0.3),
        true,
      );
    });

    test('should detect different doubles', () {
      expect(
        TestUtils.approximatelyEqual(0.1, 0.2),
        false,
      );
    });

    test('should use epsilon tolerance', () {
      expect(
        TestUtils.approximatelyEqual(1.0, 1.00005, epsilon: 0.001),
        true,
      );

      expect(
        TestUtils.approximatelyEqual(1.0, 1.005, epsilon: 0.001),
        false,
      );
    });

    test('should detect approximately equal points', () {
      final p1 = Point(0.1 + 0.2, 0.3);
      final p2 = Point(0.3, 0.3);

      expect(
        TestUtils.pointsApproximatelyEqual(p1, p2),
        true,
      );
    });

    test('should detect different points', () {
      final p1 = Point(0.1, 0.2);
      final p2 = Point(0.3, 0.4);

      expect(
        TestUtils.pointsApproximatelyEqual(p1, p2),
        false,
      );
    });
  });

  group('Test Data Generation', () {
    test('should generate valid keypoints', () {
      final keypoints = TestUtils.generateValidKeypoints();

      expect(keypoints.length, 12); // 6 landmarks * 2 (x,y)
      expect(keypoints[0], inRange(0.0, 1.0)); // All normalized
      expect(keypoints[1], inRange(0.0, 1.0));
    });

    test('should generate custom keypoints', () {
      final keypoints = TestUtils.generateValidKeypoints(
        leftEye: Point(0.25, 0.35),
        rightEye: Point(0.75, 0.35),
      );

      expect(keypoints[0], 0.25); // leftEye.x
      expect(keypoints[1], 0.35); // leftEye.y
      expect(keypoints[2], 0.75); // rightEye.x
      expect(keypoints[3], 0.35); // rightEye.y
    });

    test('should create dummy image bytes', () {
      final bytes = TestUtils.createDummyImageBytes();

      expect(bytes, isNotEmpty);
      expect(bytes.length, greaterThan(40)); // Minimal PNG is ~50 bytes
      expect(bytes[0], 0x89); // PNG signature
      expect(bytes[1], 0x50);
      expect(bytes[2], 0x4E);
      expect(bytes[3], 0x47);
    });
  });
}

/// Helper matcher to check if value is in range
Matcher inRange(num min, num max) {
  return predicate<num>(
    (value) => value >= min && value <= max,
    'value in range [$min, $max]',
  );
}
