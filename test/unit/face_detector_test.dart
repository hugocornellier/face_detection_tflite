import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart' show Size;
import 'package:face_detection_tflite/face_detection_tflite.dart';

import '../test_config.dart';

/// Comprehensive tests for the Detection class and related data structures.
///
/// This test suite covers:
/// - Detection.landmarks conversion and validation
/// - RectF operations (scale, expand, getters)
/// - Edge cases and error handling
/// - Coordinate transformations
void main() {
  setUpAll(() {
    globalTestSetup();
  });

  group('Detection', () {
    group('landmarks', () {
      test('converts normalized keypoints into pixel coordinates', () {
        // Arrange: Create detection with known normalized keypoints
        const imgSize = Size(200, 100);
        final keypointsXY = <double>[
          0.25, 0.50, // leftEye
          0.75, 0.40, // rightEye
          0.5, 0.3, // noseTip
          0.5, 0.8, // mouth
          0.2, 0.5, // leftEyeTragion
          0.8, 0.5, // rightEyeTragion
        ];

        final det = Detection(
          bbox: const RectF(0.2, 0.3, 0.4, 0.6),
          score: 0.9,
          keypointsXY: keypointsXY,
          imageSize: imgSize,
        );

        // Act: Get landmarks in pixel coordinates
        final lm = det.landmarks;

        // Assert: Verify correct conversion (normalized * size = pixels)
        expect(lm[FaceIndex.leftEye], const math.Point<double>(50.0, 50.0),
            reason: '0.25 * 200 = 50, 0.50 * 100 = 50');

        expect(lm[FaceIndex.rightEye], const math.Point<double>(150.0, 40.0),
            reason: '0.75 * 200 = 150, 0.40 * 100 = 40');

        expect(lm[FaceIndex.noseTip], const math.Point<double>(100.0, 30.0),
            reason: '0.5 * 200 = 100, 0.3 * 100 = 30');

        expect(lm[FaceIndex.mouth], const math.Point<double>(100.0, 80.0),
            reason: '0.5 * 200 = 100, 0.8 * 100 = 80');

        expect(lm[FaceIndex.leftEyeTragion], const math.Point<double>(40.0, 50.0),
            reason: '0.2 * 200 = 40, 0.5 * 100 = 50');

        expect(lm[FaceIndex.rightEyeTragion], const math.Point<double>(160.0, 50.0),
            reason: '0.8 * 200 = 160, 0.5 * 100 = 50');
      });

      test('returns all FaceIndex enum values in landmarks map', () {
        // Arrange: Create detection with valid data
        const imgSize = Size(100, 100);
        final keypointsXY = TestUtils.generateValidKeypoints();

        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: keypointsXY,
          imageSize: imgSize,
        );

        // Act: Get landmarks
        final lm = det.landmarks;

        // Assert: All FaceIndex values should be present
        expect(lm.keys.length, FaceIndex.values.length,
            reason: 'Should contain all FaceIndex enum values');

        for (final index in FaceIndex.values) {
          expect(lm.containsKey(index), isTrue,
              reason: 'Should contain key for $index');
          expect(lm[index], isNotNull,
              reason: 'Landmark for $index should not be null');
        }
      });

      test('handles edge coordinates (0.0 and 1.0) correctly', () {
        // Arrange: Create detection with boundary normalized coordinates
        const imgSize = Size(100, 200);
        final keypointsXY = <double>[
          0.0, 0.0, // leftEye at top-left
          1.0, 0.0, // rightEye at top-right
          0.5, 0.5, // noseTip at center
          0.5, 1.0, // mouth at bottom center
          0.0, 0.5, // leftEyeTragion at left edge
          1.0, 1.0, // rightEyeTragion at bottom-right
        ];

        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: keypointsXY,
          imageSize: imgSize,
        );

        // Act
        final lm = det.landmarks;

        // Assert: Boundary coordinates should map correctly
        expect(lm[FaceIndex.leftEye], const math.Point<double>(0.0, 0.0));
        expect(lm[FaceIndex.rightEye], const math.Point<double>(100.0, 0.0));
        expect(lm[FaceIndex.noseTip], const math.Point<double>(50.0, 100.0));
        expect(lm[FaceIndex.mouth], const math.Point<double>(50.0, 200.0));
        expect(lm[FaceIndex.leftEyeTragion], const math.Point<double>(0.0, 100.0));
        expect(lm[FaceIndex.rightEyeTragion], const math.Point<double>(100.0, 200.0));
      });

      test('handles fractional pixel coordinates correctly', () {
        // Arrange: Test that fractional normalized values produce fractional pixels
        const imgSize = Size(300, 300);
        final keypointsXY = <double>[
          0.333, 0.333, // leftEye
          0.666, 0.333, // rightEye
          0.5, 0.5, // noseTip
          0.5, 0.777, // mouth
          0.111, 0.5, // leftEyeTragion
          0.888, 0.5, // rightEyeTragion
        ];

        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: keypointsXY,
          imageSize: imgSize,
        );

        // Act
        final lm = det.landmarks;

        // Assert: Verify fractional calculations
        expect(lm[FaceIndex.leftEye]!.x, closeTo(99.9, 0.01),
            reason: '0.333 * 300 = 99.9');
        expect(lm[FaceIndex.rightEye]!.x, closeTo(199.8, 0.01),
            reason: '0.666 * 300 = 199.8');
        expect(lm[FaceIndex.mouth]!.y, closeTo(233.1, 0.01),
            reason: '0.777 * 300 = 233.1');
      });

      test('handles non-square images correctly', () {
        // Arrange: Test with very different width and height
        const imgSize = Size(1920, 1080); // 16:9 aspect ratio
        final keypointsXY = <double>[
          0.5, 0.5, // leftEye at center
          0.5, 0.5, // rightEye at center
          0.5, 0.5, // noseTip at center
          0.5, 0.5, // mouth at center
          0.5, 0.5, // leftEyeTragion at center
          0.5, 0.5, // rightEyeTragion at center
        ];

        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: keypointsXY,
          imageSize: imgSize,
        );

        // Act
        final lm = det.landmarks;

        // Assert: Center should be at (960, 540)
        expect(lm[FaceIndex.leftEye], const math.Point<double>(960.0, 540.0));
        expect(lm[FaceIndex.noseTip], const math.Point<double>(960.0, 540.0));
      });

      test('throws StateError if imageSize is null', () {
        // Arrange: Create detection without imageSize
        final detNoSize = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: const [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.0],
          imageSize: null,
        );

        // Act & Assert: Should throw when accessing landmarks
        expect(
              () => detNoSize.landmarks,
          throwsA(isA<StateError>()),
          reason: 'Should throw StateError when imageSize is null',
        );
      });

      test('throws StateError with descriptive message when imageSize is null', () {
        // Arrange
        final detNoSize = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: const [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.0],
          imageSize: null,
        );

        // Act & Assert: Check error message
        expect(
              () => detNoSize.landmarks,
          throwsA(
            isA<StateError>().having(
                  (e) => e.message,
              'message',
              contains('imageSize is null'),
            ),
          ),
          reason: 'Error message should mention imageSize is null',
        );
      });

      test('handles very large image dimensions', () {
        // Arrange: Test with 4K resolution
        const imgSize = Size(3840, 2160);
        final keypointsXY = <double>[
          0.25, 0.25, // leftEye
          0.75, 0.25, // rightEye
          0.5, 0.5, // noseTip
          0.5, 0.75, // mouth
          0.1, 0.5, // leftEyeTragion
          0.9, 0.5, // rightEyeTragion
        ];

        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: keypointsXY,
          imageSize: imgSize,
        );

        // Act
        final lm = det.landmarks;

        // Assert: Large coordinates should be calculated correctly
        expect(lm[FaceIndex.leftEye], const math.Point<double>(960.0, 540.0));
        expect(lm[FaceIndex.rightEye], const math.Point<double>(2880.0, 540.0));
        expect(lm[FaceIndex.noseTip], const math.Point<double>(1920.0, 1080.0));
        expect(lm[FaceIndex.mouth], const math.Point<double>(1920.0, 1620.0));
      });

      test('handles very small image dimensions', () {
        // Arrange: Test with tiny image
        const imgSize = Size(10, 10);
        final keypointsXY = <double>[
          0.5, 0.5, // leftEye
          0.5, 0.5, // rightEye
          0.5, 0.5, // noseTip
          0.5, 0.5, // mouth
          0.5, 0.5, // leftEyeTragion
          0.5, 0.5, // rightEyeTragion
        ];

        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: keypointsXY,
          imageSize: imgSize,
        );

        // Act
        final lm = det.landmarks;

        // Assert: Small coordinates should work
        expect(lm[FaceIndex.leftEye], const math.Point<double>(5.0, 5.0));
        expect(lm[FaceIndex.noseTip], const math.Point<double>(5.0, 5.0));
      });
    });

    group('keypointsXY accessor', () {
      test('allows direct access to keypoint values by index', () {
        // Arrange
        final keypointsXY = <double>[
          0.1, 0.2, // leftEye
          0.3, 0.4, // rightEye
          0.5, 0.6, // noseTip
          0.7, 0.8, // mouth
          0.9, 1.0, // leftEyeTragion
          0.0, 0.0, // rightEyeTragion
        ];

        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: keypointsXY,
        );

        // Act & Assert: Test operator[] access
        expect(det[0], 0.1, reason: 'First X coordinate');
        expect(det[1], 0.2, reason: 'First Y coordinate');
        expect(det[4], 0.5, reason: 'Third X coordinate (noseTip)');
        expect(det[5], 0.6, reason: 'Third Y coordinate (noseTip)');
        expect(det[11], 0.0, reason: 'Last Y coordinate');
      });

      test('provides access to all keypoint coordinates', () {
        // Arrange
        final keypointsXY = TestUtils.generateValidKeypoints();
        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: keypointsXY,
        );

        // Act & Assert: Verify we can access all 12 values (6 points * 2 coords)
        for (var i = 0; i < 12; i++) {
          expect(() => det[i], returnsNormally,
              reason: 'Should access index $i without error');
          expect(det[i], isA<double>(),
              reason: 'Value at index $i should be a double');
        }
      });
    });

    group('bbox', () {
      test('stores bounding box correctly', () {
        // Arrange & Act
        const bbox = RectF(0.1, 0.2, 0.9, 0.8);
        final det = Detection(
          bbox: bbox,
          score: 0.95,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        // Assert
        expect(det.bbox, same(bbox), reason: 'Should return same bbox instance');
        expect(det.bbox.xmin, 0.1);
        expect(det.bbox.ymin, 0.2);
        expect(det.bbox.xmax, 0.9);
        expect(det.bbox.ymax, 0.8);
      });

      test('bbox is independent of imageSize', () {
        // Arrange: bbox should remain in normalized coords regardless of imageSize
        const bbox = RectF(0.1, 0.2, 0.3, 0.4);

        final detWithSize = Detection(
          bbox: bbox,
          score: 0.9,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: const Size(1000, 1000),
        );

        final detWithoutSize = Detection(
          bbox: bbox,
          score: 0.9,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        // Assert: bbox should be the same regardless of imageSize
        expect(detWithSize.bbox.xmin, detWithoutSize.bbox.xmin);
        expect(detWithSize.bbox.xmax, detWithoutSize.bbox.xmax);
      });
    });

    group('score', () {
      test('stores confidence score correctly', () {
        // Arrange & Act
        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 0.87654321,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        // Assert
        expect(det.score, 0.87654321, reason: 'Score should be exact value provided');
      });

      test('accepts valid score range [0.0, 1.0]', () {
        // Arrange & Act: Test boundary values
        final lowScore = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 0.0,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        final highScore = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        final midScore = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 0.5,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        // Assert: All scores should be stored correctly
        expect(lowScore.score, 0.0);
        expect(highScore.score, 1.0);
        expect(midScore.score, 0.5);
      });

      test('accepts scores outside [0.0, 1.0] range', () {
        // Note: The Detection class doesn't validate score range,
        // so it accepts any double value. This documents current behavior.

        // Arrange & Act: Test out-of-range values
        final negativeScore = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: -0.5,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        final overOne = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.5,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        // Assert: Values are accepted (no validation)
        expect(negativeScore.score, -0.5);
        expect(overOne.score, 1.5);
      });
    });

    group('imageSize', () {
      test('stores imageSize when provided', () {
        // Arrange & Act
        const imgSize = Size(640, 480);
        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: imgSize,
        );

        // Assert
        expect(det.imageSize, imgSize);
        expect(det.imageSize!.width, 640);
        expect(det.imageSize!.height, 480);
      });

      test('imageSize is null when not provided', () {
        // Arrange & Act
        final det = Detection(
          bbox: const RectF(0.0, 0.0, 1.0, 1.0),
          score: 1.0,
          keypointsXY: TestUtils.generateValidKeypoints(),
        );

        // Assert
        expect(det.imageSize, isNull, reason: 'imageSize should default to null');
      });
    });
  });

  group('RectF', () {
    group('constructor and getters', () {
      test('creates rectangle with correct bounds', () {
        // Arrange & Act
        const rect = RectF(10.0, 20.0, 100.0, 80.0);

        // Assert
        expect(rect.xmin, 10.0);
        expect(rect.ymin, 20.0);
        expect(rect.xmax, 100.0);
        expect(rect.ymax, 80.0);
      });

      test('calculates width correctly', () {
        // Arrange
        const rect = RectF(10.0, 20.0, 50.0, 80.0);

        // Act & Assert
        expect(rect.w, 40.0, reason: 'Width should be xmax - xmin = 50 - 10 = 40');
      });

      test('calculates height correctly', () {
        // Arrange
        const rect = RectF(10.0, 20.0, 50.0, 80.0);

        // Act & Assert
        expect(rect.h, 60.0, reason: 'Height should be ymax - ymin = 80 - 20 = 60');
      });

      test('handles normalized coordinates [0.0, 1.0]', () {
        // Arrange & Act
        const rect = RectF(0.1, 0.2, 0.7, 0.9);

        // Assert
        expect(rect.w, closeTo(0.6, 0.0001));
        expect(rect.h, closeTo(0.7, 0.0001));
      });

      test('handles negative coordinates', () {
        // Arrange & Act: Rectangle in negative space
        const rect = RectF(-50.0, -30.0, -10.0, -5.0);

        // Assert
        expect(rect.w, 40.0);
        expect(rect.h, 25.0);
      });

      test('handles zero-width rectangle', () {
        // Arrange & Act
        const rect = RectF(10.0, 20.0, 10.0, 50.0);

        // Assert
        expect(rect.w, 0.0);
        expect(rect.h, 30.0);
      });

      test('handles zero-height rectangle', () {
        // Arrange & Act
        const rect = RectF(10.0, 20.0, 50.0, 20.0);

        // Assert
        expect(rect.w, 40.0);
        expect(rect.h, 0.0);
      });

      test('handles point (zero width and height)', () {
        // Arrange & Act
        const rect = RectF(10.0, 20.0, 10.0, 20.0);

        // Assert
        expect(rect.w, 0.0);
        expect(rect.h, 0.0);
      });
    });

    group('scale', () {
      test('scales rectangle proportionally', () {
        // Arrange
        const rect = RectF(10.0, 20.0, 30.0, 40.0);

        // Act
        final scaled = rect.scale(2.0, 3.0);

        // Assert: All coordinates should be multiplied by scale factors
        expect(scaled.xmin, 20.0, reason: 'xmin should be doubled: 10 * 2 = 20');
        expect(scaled.ymin, 60.0, reason: 'ymin should be tripled: 20 * 3 = 60');
        expect(scaled.xmax, 60.0, reason: 'xmax should be doubled: 30 * 2 = 60');
        expect(scaled.ymax, 120.0, reason: 'ymax should be tripled: 40 * 3 = 120');
        expect(scaled.w, 40.0, reason: 'Width should be doubled: 20 * 2 = 40');
        expect(scaled.h, 60.0, reason: 'Height should be tripled: 20 * 3 = 60');
      });

      test('scales with equal x and y factors', () {
        // Arrange
        const rect = RectF(5.0, 10.0, 15.0, 20.0);

        // Act
        final scaled = rect.scale(3.0, 3.0);

        // Assert
        expect(scaled.xmin, 15.0);
        expect(scaled.ymin, 30.0);
        expect(scaled.xmax, 45.0);
        expect(scaled.ymax, 60.0);
      });

      test('handles scale factor of 1.0 (no change)', () {
        // Arrange
        const rect = RectF(10.0, 20.0, 30.0, 40.0);

        // Act
        final scaled = rect.scale(1.0, 1.0);

        // Assert: Rectangle should be unchanged
        expect(scaled.xmin, rect.xmin);
        expect(scaled.ymin, rect.ymin);
        expect(scaled.xmax, rect.xmax);
        expect(scaled.ymax, rect.ymax);
      });

      test('handles scale factor of 0.0 (collapse to origin)', () {
        // Arrange
        const rect = RectF(10.0, 20.0, 30.0, 40.0);

        // Act
        final scaled = rect.scale(0.0, 0.0);

        // Assert: All coordinates should become 0
        expect(scaled.xmin, 0.0);
        expect(scaled.ymin, 0.0);
        expect(scaled.xmax, 0.0);
        expect(scaled.ymax, 0.0);
        expect(scaled.w, 0.0);
        expect(scaled.h, 0.0);
      });

      test('handles negative scale factors (flip)', () {
        // Arrange
        const rect = RectF(10.0, 20.0, 30.0, 40.0);

        // Act
        final scaled = rect.scale(-1.0, -1.0);

        // Assert: Coordinates should be negated
        expect(scaled.xmin, -10.0);
        expect(scaled.ymin, -20.0);
        expect(scaled.xmax, -30.0);
        expect(scaled.ymax, -40.0);
      });

      test('handles fractional scale factors', () {
        // Arrange
        const rect = RectF(10.0, 20.0, 30.0, 40.0);

        // Act
        final scaled = rect.scale(0.5, 0.25);

        // Assert
        expect(scaled.xmin, 5.0);
        expect(scaled.ymin, 5.0);
        expect(scaled.xmax, 15.0);
        expect(scaled.ymax, 10.0);
        expect(scaled.w, 10.0, reason: 'Width should be halved: 20 * 0.5 = 10');
        expect(scaled.h, 5.0, reason: 'Height should be quartered: 20 * 0.25 = 5');
      });

      test('scaling normalized rect to pixel coordinates', () {
        // Arrange: Normalized rectangle
        const normalizedRect = RectF(0.1, 0.2, 0.9, 0.8);
        const imageWidth = 640.0;
        const imageHeight = 480.0;

        // Act: Scale to image dimensions
        final pixelRect = normalizedRect.scale(imageWidth, imageHeight);

        // Assert
        expect(pixelRect.xmin, 64.0, reason: '0.1 * 640 = 64');
        expect(pixelRect.ymin, 96.0, reason: '0.2 * 480 = 96');
        expect(pixelRect.xmax, 576.0, reason: '0.9 * 640 = 576');
        expect(pixelRect.ymax, 384.0, reason: '0.8 * 480 = 384');
      });

      test('returns new RectF instance (immutability)', () {
        // Arrange
        const original = RectF(10.0, 20.0, 30.0, 40.0);

        // Act
        final scaled = original.scale(2.0, 2.0);

        // Assert: Original should be unchanged
        expect(original.xmin, 10.0);
        expect(original.ymin, 20.0);
        expect(identical(original, scaled), isFalse,
            reason: 'Should return new instance, not modify original');
      });
    });

    group('expand', () {
      test('expands rectangle by fraction while maintaining center', () {
        // Arrange: Rectangle centered at (50, 100) with size 100x200
        const rect = RectF(0.0, 0.0, 100.0, 200.0);

        // Act: Expand by 50% (becomes 150x300)
        final expanded = rect.expand(0.5);

        // Assert: Size should increase
        expect(expanded.w, 150.0, reason: '100 * (1 + 0.5) = 150');
        expect(expanded.h, 300.0, reason: '200 * (1 + 0.5) = 300');

        // Assert: Center should remain the same
        const centerX = 50.0;
        const centerY = 100.0;
        expect((expanded.xmin + expanded.xmax) / 2, centerX,
            reason: 'Center X should remain at 50');
        expect((expanded.ymin + expanded.ymax) / 2, centerY,
            reason: 'Center Y should remain at 100');
      });

      test('expand by 0.0 returns same size rectangle', () {
        // Arrange
        const rect = RectF(10.0, 20.0, 50.0, 80.0);

        // Act
        final expanded = rect.expand(0.0);

        // Assert: Size should be unchanged
        expect(expanded.w, rect.w);
        expect(expanded.h, rect.h);

        // Center should still be the same
        final centerX = (rect.xmin + rect.xmax) / 2;
        final centerY = (rect.ymin + rect.ymax) / 2;
        expect((expanded.xmin + expanded.xmax) / 2, centerX);
        expect((expanded.ymin + expanded.ymax) / 2, centerY);
      });

      test('expand by 1.0 doubles the rectangle size', () {
        // Arrange
        const rect = RectF(25.0, 50.0, 75.0, 150.0); // 50x100 centered at (50, 100)

        // Act
        final expanded = rect.expand(1.0);

        // Assert: Size should double
        expect(expanded.w, 100.0, reason: '50 * (1 + 1.0) = 100');
        expect(expanded.h, 200.0, reason: '100 * (1 + 1.0) = 200');

        // Center should remain at (50, 100)
        expect((expanded.xmin + expanded.xmax) / 2, 50.0);
        expect((expanded.ymin + expanded.ymax) / 2, 100.0);
      });

      test('expand with negative fraction shrinks rectangle', () {
        // Arrange
        const rect = RectF(0.0, 0.0, 100.0, 100.0); // 100x100 centered at (50, 50)

        // Act: Shrink by 50%
        final shrunk = rect.expand(-0.5);

        // Assert: Size should decrease
        expect(shrunk.w, 50.0, reason: '100 * (1 - 0.5) = 50');
        expect(shrunk.h, 50.0, reason: '100 * (1 - 0.5) = 50');

        // Center should remain at (50, 50)
        expect((shrunk.xmin + shrunk.xmax) / 2, 50.0);
        expect((shrunk.ymin + shrunk.ymax) / 2, 50.0);
      });

      test('expand by -1.0 collapses to center point', () {
        // Arrange
        const rect = RectF(0.0, 0.0, 100.0, 100.0);

        // Act
        final collapsed = rect.expand(-1.0);

        // Assert: Should collapse to a point at center
        expect(collapsed.w, 0.0, reason: '100 * (1 - 1.0) = 0');
        expect(collapsed.h, 0.0, reason: '100 * (1 - 1.0) = 0');
        expect(collapsed.xmin, 50.0);
        expect(collapsed.xmax, 50.0);
        expect(collapsed.ymin, 50.0);
        expect(collapsed.ymax, 50.0);
      });

      test('handles small fractional expansions', () {
        // Arrange
        const rect = RectF(40.0, 40.0, 60.0, 60.0); // 20x20 centered at (50, 50)

        // Act: Expand by 10%
        final expanded = rect.expand(0.1);

        // Assert
        expect(expanded.w, 22.0, reason: '20 * 1.1 = 22');
        expect(expanded.h, 22.0);

        // Center preserved
        expect((expanded.xmin + expanded.xmax) / 2, 50.0);
        expect((expanded.ymin + expanded.ymax) / 2, 50.0);
      });

      test('maintains center for off-origin rectangles', () {
        // Arrange: Rectangle not centered at origin
        const rect = RectF(100.0, 200.0, 140.0, 260.0); // 40x60 centered at (120, 230)

        // Act
        final expanded = rect.expand(0.5);

        // Assert: Size increases
        expect(expanded.w, 60.0, reason: '40 * 1.5 = 60');
        expect(expanded.h, 90.0, reason: '60 * 1.5 = 90');

        // Center preserved
        expect((expanded.xmin + expanded.xmax) / 2, 120.0);
        expect((expanded.ymin + expanded.ymax) / 2, 230.0);
      });

      test('handles normalized coordinates', () {
        // Arrange: Normalized rectangle
        const rect = RectF(0.3, 0.4, 0.7, 0.8); // 0.4x0.4 centered at (0.5, 0.6)

        // Act
        final expanded = rect.expand(0.5);

        // Assert
        expect(expanded.w, closeTo(0.6, 0.0001), reason: '0.4 * 1.5 = 0.6');
        expect(expanded.h, closeTo(0.6, 0.0001), reason: '0.4 * 1.5 = 0.6');

        // Center preserved
        expect((expanded.xmin + expanded.xmax) / 2, closeTo(0.5, 0.0001));
        expect((expanded.ymin + expanded.ymax) / 2, closeTo(0.6, 0.0001));
      });

      test('returns new RectF instance (immutability)', () {
        // Arrange
        const original = RectF(10.0, 20.0, 30.0, 40.0);

        // Act
        final expanded = original.expand(0.5);

        // Assert: Original should be unchanged
        expect(original.w, 20.0);
        expect(original.h, 20.0);
        expect(identical(original, expanded), isFalse,
            reason: 'Should return new instance');
      });
    });
  });

  group('DetectionPoints extension', () {
    test('landmarksPoints returns same result as landmarks', () {
      // Arrange
      const imgSize = Size(100, 100);
      final det = Detection(
        bbox: const RectF(0.0, 0.0, 1.0, 1.0),
        score: 1.0,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: imgSize,
      );

      // Act
      final landmarks = det.landmarks;
      final landmarksPoints = det.landmarksPoints;

      // Assert: Extension method should return identical result
      expect(landmarksPoints.length, landmarks.length);
      for (final key in landmarks.keys) {
        expect(landmarksPoints[key], landmarks[key],
            reason: 'landmarksPoints should match landmarks for $key');
      }
    });
  });
}