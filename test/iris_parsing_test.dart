import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

/// Tests for Face iris parsing logic, particularly Face._parseIris()
/// and Face.eyes getter which handles multiple data formats
void main() {
  globalTestSetup();

  // Helper to create a Face with specific iris points
  Face createFaceWithIrisPoints(List<Point> irisPoints) {
    final detection = Detection(
      boundingBox: RectF(0.2, 0.3, 0.8, 0.7),
      score: 0.95,
      keypointsXY: TestUtils.generateValidKeypoints(),
      imageSize: TestConstants.mediumImage,
    );

    final mesh = FaceMesh(
      List.generate(468, (i) => Point(i.toDouble(), i.toDouble())),
    );

    return Face(
      detection: detection,
      mesh: mesh,
      irises: irisPoints,
      originalSize: TestConstants.mediumImage,
    );
  }

  group('Face.eyes - Empty and Null Cases', () {
    test('should return null when iris points are empty', () {
      final face = createFaceWithIrisPoints([]);
      expect(face.eyes, isNull);
    });

    test('should return null when iris points less than 5', () {
      final irisPoints = [
        Point(100.0, 100.0),
        Point(110.0, 100.0),
        Point(100.0, 110.0),
      ];
      final face = createFaceWithIrisPoints(irisPoints);
      expect(face.eyes, isNull);
    });
  });

  group('Face.eyes - Legacy Format (5 Points)', () {
    test('should parse 5 iris points as single left eye', () {
      // Create 5 points where point at index 2 is the center
      // (center has minimum sum of squared distances to other points)
      final irisPoints = [
        Point(95.0, 100.0), // Left
        Point(105.0, 100.0), // Right
        Point(100.0, 100.0), // Center (equidistant from all)
        Point(100.0, 95.0), // Top
        Point(100.0, 105.0), // Bottom
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNull);

      final leftEye = eyes.leftEye!;
      expect(leftEye.irisCenter, equals(Point(100.0, 100.0)));
      expect(leftEye.irisContour.length, 4);
      expect(leftEye.mesh, isEmpty);

      // Contour should be the 4 non-center points
      expect(leftEye.irisContour, contains(Point(95.0, 100.0)));
      expect(leftEye.irisContour, contains(Point(105.0, 100.0)));
      expect(leftEye.irisContour, contains(Point(100.0, 95.0)));
      expect(leftEye.irisContour, contains(Point(100.0, 105.0)));
    });

    test('should identify center as point with minimum distance sum', () {
      // Create points where first point is clearly the center
      final irisPoints = [
        Point(100.0, 100.0), // Center - equidistant
        Point(90.0, 90.0), // Far corner
        Point(110.0, 90.0), // Far corner
        Point(110.0, 110.0), // Far corner
        Point(90.0, 110.0), // Far corner
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      expect(eyes!.leftEye!.irisCenter, equals(Point(100.0, 100.0)));
    });

    test('should handle corner points as iris points', () {
      // Test with actual corner configuration
      final irisPoints = [
        Point(50.0, 50.0),
        Point(60.0, 50.0),
        Point(55.0, 55.0), // This should be center
        Point(50.0, 60.0),
        Point(60.0, 60.0),
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      expect(eyes!.leftEye!.irisCenter, equals(Point(55.0, 55.0)));
      expect(eyes.leftEye!.irisContour.length, 4);
    });
  });

  group('Face.eyes - Full Format (76 Points)', () {
    test('should parse 76 points into eye mesh and iris', () {
      // First 71 points are eye mesh, last 5 are iris points
      final meshPoints = List.generate(71, (i) => Point(i.toDouble(), 50.0));
      final irisPoints = [
        Point(80.0, 50.0),
        Point(90.0, 50.0),
        Point(85.0, 55.0), // Center
        Point(80.0, 60.0),
        Point(90.0, 60.0),
      ];
      final allPoints = [...meshPoints, ...irisPoints];

      final face = createFaceWithIrisPoints(allPoints);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNull);

      final leftEye = eyes.leftEye!;
      expect(leftEye.mesh.length, 71);
      expect(leftEye.irisCenter, equals(Point(85.0, 55.0)));
      expect(leftEye.irisContour.length, 4);

      // Verify mesh points are correct
      expect(leftEye.mesh[0], equals(Point(0.0, 50.0)));
      expect(leftEye.mesh[70], equals(Point(70.0, 50.0)));

      // Verify contour extraction (first 15 mesh points)
      final contour = leftEye.contour;
      expect(contour.length, 15);
      expect(contour[0], equals(Point(0.0, 50.0)));
      expect(contour[14], equals(Point(14.0, 50.0)));
    });

    test('should handle 76-point format with different center', () {
      final meshPoints = List.generate(71, (i) => Point(100.0 + i, 100.0));
      final irisPoints = [
        Point(200.0, 100.0), // Far left
        Point(220.0, 100.0), // Far right
        Point(230.0, 100.0), // Even further right
        Point(210.0, 95.0), // Top middle
        Point(210.0, 105.0), // Bottom middle (center)
      ];
      final allPoints = [...meshPoints, ...irisPoints];

      final face = createFaceWithIrisPoints(allPoints);
      final eyes = face.eyes;

      expect(eyes!.leftEye, isNotNull);
      expect(eyes.leftEye!.mesh.length, 71);
      expect(eyes.leftEye!.irisContour.length, 4);
    });
  });

  group('Face.eyes - Two Eyes Format (10 and 152 Points)', () {
    test('should parse 10 points as two eyes (5 each)', () {
      // Left eye (5 points)
      final leftIrisPoints = [
        Point(45.0, 50.0),
        Point(55.0, 50.0),
        Point(50.0, 50.0), // Center
        Point(50.0, 45.0),
        Point(50.0, 55.0),
      ];

      // Right eye (5 points)
      final rightIrisPoints = [
        Point(145.0, 50.0),
        Point(155.0, 50.0),
        Point(150.0, 50.0), // Center
        Point(150.0, 45.0),
        Point(150.0, 55.0),
      ];

      final allPoints = [...leftIrisPoints, ...rightIrisPoints];
      final face = createFaceWithIrisPoints(allPoints);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      expect(eyes.leftEye!.irisCenter, equals(Point(50.0, 50.0)));
      expect(eyes.rightEye!.irisCenter, equals(Point(150.0, 50.0)));

      expect(eyes.leftEye!.irisContour.length, 4);
      expect(eyes.rightEye!.irisContour.length, 4);

      expect(eyes.leftEye!.mesh, isEmpty);
      expect(eyes.rightEye!.mesh, isEmpty);
    });

    test('should parse 152 points as two eyes (76 each)', () {
      // Left eye (76 points: 71 mesh + 5 iris)
      final leftMesh = List.generate(71, (i) => Point(i.toDouble(), 50.0));
      final leftIris = [
        Point(80.0, 50.0),
        Point(90.0, 50.0),
        Point(85.0, 50.0), // Center
        Point(85.0, 45.0),
        Point(85.0, 55.0),
      ];

      // Right eye (76 points: 71 mesh + 5 iris)
      final rightMesh = List.generate(71, (i) => Point(100.0 + i, 50.0));
      final rightIris = [
        Point(180.0, 50.0),
        Point(190.0, 50.0),
        Point(185.0, 50.0), // Center
        Point(185.0, 45.0),
        Point(185.0, 55.0),
      ];

      final allPoints = [...leftMesh, ...leftIris, ...rightMesh, ...rightIris];
      final face = createFaceWithIrisPoints(allPoints);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      expect(eyes.leftEye!.mesh.length, 71);
      expect(eyes.rightEye!.mesh.length, 71);

      expect(eyes.leftEye!.irisCenter, equals(Point(85.0, 50.0)));
      expect(eyes.rightEye!.irisCenter, equals(Point(185.0, 50.0)));

      expect(eyes.leftEye!.irisContour.length, 4);
      expect(eyes.rightEye!.irisContour.length, 4);
    });
  });

  group('Face.eyes - Variable Length Formats', () {
    test('should parse 20 points as two eyes (10 each)', () {
      // Left eye: 5 mesh + 5 iris
      final leftPoints = List.generate(10, (i) => Point(i.toDouble(), 50.0));
      // Right eye: 5 mesh + 5 iris
      final rightPoints = List.generate(10, (i) => Point(20.0 + i, 50.0));

      final allPoints = [...leftPoints, ...rightPoints];
      final face = createFaceWithIrisPoints(allPoints);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      // Each eye should have 5 mesh points and 5 iris points
      expect(eyes.leftEye!.mesh.length, 5);
      expect(eyes.rightEye!.mesh.length, 5);
    });

    test('should parse 30 points as two eyes (15 each)', () {
      // Even number > 10 should split evenly
      final leftPoints = List.generate(15, (i) => Point(i.toDouble(), 50.0));
      final rightPoints = List.generate(15, (i) => Point(30.0 + i, 50.0));

      final allPoints = [...leftPoints, ...rightPoints];
      final face = createFaceWithIrisPoints(allPoints);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      // Each eye: 10 mesh + 5 iris
      expect(eyes.leftEye!.mesh.length, 10);
      expect(eyes.rightEye!.mesh.length, 10);
    });

    test('should handle 7 points as single left eye', () {
      // Odd number >= 5 but <= 10 should be treated as single eye
      final points = List.generate(7, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNull);

      // 2 mesh + 5 iris
      expect(eyes.leftEye!.mesh.length, 2);
      expect(eyes.leftEye!.irisContour.length, 4);
    });

    test('should handle 9 points as single left eye', () {
      // 9 points: 4 mesh + 5 iris
      final points = List.generate(9, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNull);
    });
  });

  group('Face.eyes - Center Point Detection Algorithm', () {
    test('should identify center with equal distances to corners', () {
      // Perfect square with center at origin
      final irisPoints = [
        Point(100.0, 100.0), // Center
        Point(90.0, 90.0), // Top-left
        Point(110.0, 90.0), // Top-right
        Point(110.0, 110.0), // Bottom-right
        Point(90.0, 110.0), // Bottom-left
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      expect(eyes!.leftEye!.irisCenter, equals(Point(100.0, 100.0)));
    });

    test('should identify center with minimum sum of squared distances', () {
      // Create configuration where one point is clearly central
      final irisPoints = [
        Point(50.0, 50.0), // Outlier
        Point(101.0, 99.0), // Near center
        Point(100.0, 100.0), // True center
        Point(99.0, 101.0), // Near center
        Point(102.0, 98.0), // Near center
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      // The point at (100, 100) should have the minimum sum of squared distances
      expect(eyes!.leftEye!.irisCenter, equals(Point(100.0, 100.0)));
    });

    test('should handle collinear points', () {
      // All points on a line
      final irisPoints = [
        Point(100.0, 100.0),
        Point(110.0, 100.0),
        Point(105.0, 100.0), // Center
        Point(102.0, 100.0),
        Point(108.0, 100.0),
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      // Middle point should be center
      expect(eyes!.leftEye!.irisCenter, equals(Point(105.0, 100.0)));
    });
  });

  group('Face.eyes - Contour Extraction', () {
    test('should exclude center from contour', () {
      final irisPoints = [
        Point(100.0, 100.0),
        Point(110.0, 100.0),
        Point(105.0, 105.0), // Center
        Point(100.0, 110.0),
        Point(110.0, 110.0),
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;
      final contour = eyes!.leftEye!.irisContour;

      expect(contour.length, 4);
      expect(contour, isNot(contains(Point(105.0, 105.0))));
    });

    test('should include all non-center points in contour', () {
      final irisPoints = [
        Point(100.0, 100.0), // Center
        Point(95.0, 95.0),
        Point(105.0, 95.0),
        Point(105.0, 105.0),
        Point(95.0, 105.0),
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;
      final contour = eyes!.leftEye!.irisContour;

      expect(contour, contains(Point(95.0, 95.0)));
      expect(contour, contains(Point(105.0, 95.0)));
      expect(contour, contains(Point(105.0, 105.0)));
      expect(contour, contains(Point(95.0, 105.0)));
    });
  });

  group('Face.eyes - Edge Cases', () {
    test('should return null when all parsing returns null', () {
      // 4 points - less than minimum
      final points = List.generate(4, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);

      expect(face.eyes, isNull);
    });

    test('should handle exactly 5 points', () {
      final points = List.generate(5, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNull);
    });

    test('should handle exactly 10 points', () {
      final points = List.generate(10, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);
    });

    test('should handle exactly 76 points', () {
      final points = List.generate(76, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNull);
    });

    test('should handle exactly 152 points', () {
      final points = List.generate(152, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);
    });

    test('should handle large even number of points', () {
      // 200 points should split as 100 each
      final points = List.generate(200, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      // Each eye: 95 mesh + 5 iris
      expect(eyes.leftEye!.mesh.length, 95);
      expect(eyes.rightEye!.mesh.length, 95);
    });
  });
}
