import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

/// Tests for Face iris parsing logic, particularly Face._parseIris()
/// and Face.eyes getter which handles multiple data formats
void main() {
  globalTestSetup();

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
      final irisPoints = [
        Point(95.0, 100.0),
        Point(105.0, 100.0),
        Point(100.0, 100.0),
        Point(100.0, 95.0),
        Point(100.0, 105.0),
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

      expect(leftEye.irisContour, contains(Point(95.0, 100.0)));
      expect(leftEye.irisContour, contains(Point(105.0, 100.0)));
      expect(leftEye.irisContour, contains(Point(100.0, 95.0)));
      expect(leftEye.irisContour, contains(Point(100.0, 105.0)));
    });

    test('should identify center as point with minimum distance sum', () {
      final irisPoints = [
        Point(100.0, 100.0),
        Point(90.0, 90.0),
        Point(110.0, 90.0),
        Point(110.0, 110.0),
        Point(90.0, 110.0),
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      expect(eyes!.leftEye!.irisCenter, equals(Point(100.0, 100.0)));
    });

    test('should handle corner points as iris points', () {
      final irisPoints = [
        Point(50.0, 50.0),
        Point(60.0, 50.0),
        Point(55.0, 55.0),
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
      final meshPoints = List.generate(71, (i) => Point(i.toDouble(), 50.0));
      final irisPoints = [
        Point(80.0, 50.0),
        Point(90.0, 50.0),
        Point(85.0, 55.0),
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

      expect(leftEye.mesh[0], equals(Point(0.0, 50.0)));
      expect(leftEye.mesh[70], equals(Point(70.0, 50.0)));

      final contour = leftEye.contour;
      expect(contour.length, 15);
      expect(contour[0], equals(Point(0.0, 50.0)));
      expect(contour[14], equals(Point(14.0, 50.0)));
    });

    test('should handle 76-point format with different center', () {
      final meshPoints = List.generate(71, (i) => Point(100.0 + i, 100.0));
      final irisPoints = [
        Point(200.0, 100.0),
        Point(220.0, 100.0),
        Point(230.0, 100.0),
        Point(210.0, 95.0),
        Point(210.0, 105.0),
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
      final leftIrisPoints = [
        Point(45.0, 50.0),
        Point(55.0, 50.0),
        Point(50.0, 50.0),
        Point(50.0, 45.0),
        Point(50.0, 55.0),
      ];

      final rightIrisPoints = [
        Point(145.0, 50.0),
        Point(155.0, 50.0),
        Point(150.0, 50.0),
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
      final leftMesh = List.generate(71, (i) => Point(i.toDouble(), 50.0));
      final leftIris = [
        Point(80.0, 50.0),
        Point(90.0, 50.0),
        Point(85.0, 50.0),
        Point(85.0, 45.0),
        Point(85.0, 55.0),
      ];

      final rightMesh = List.generate(71, (i) => Point(100.0 + i, 50.0));
      final rightIris = [
        Point(180.0, 50.0),
        Point(190.0, 50.0),
        Point(185.0, 50.0),
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
      final leftPoints = List.generate(10, (i) => Point(i.toDouble(), 50.0));
      final rightPoints = List.generate(10, (i) => Point(20.0 + i, 50.0));

      final allPoints = [...leftPoints, ...rightPoints];
      final face = createFaceWithIrisPoints(allPoints);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      expect(eyes.leftEye!.mesh.length, 5);
      expect(eyes.rightEye!.mesh.length, 5);
    });

    test('should parse 30 points as two eyes (15 each)', () {
      final leftPoints = List.generate(15, (i) => Point(i.toDouble(), 50.0));
      final rightPoints = List.generate(15, (i) => Point(30.0 + i, 50.0));

      final allPoints = [...leftPoints, ...rightPoints];
      final face = createFaceWithIrisPoints(allPoints);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      expect(eyes.leftEye!.mesh.length, 10);
      expect(eyes.rightEye!.mesh.length, 10);
    });

    test('should handle 7 points as single left eye', () {
      final points = List.generate(7, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNull);

      expect(eyes.leftEye!.mesh.length, 2);
      expect(eyes.leftEye!.irisContour.length, 4);
    });

    test('should handle 9 points as single left eye', () {
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
      final irisPoints = [
        Point(100.0, 100.0),
        Point(90.0, 90.0),
        Point(110.0, 90.0),
        Point(110.0, 110.0),
        Point(90.0, 110.0),
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      expect(eyes!.leftEye!.irisCenter, equals(Point(100.0, 100.0)));
    });

    test('should identify center with minimum sum of squared distances', () {
      final irisPoints = [
        Point(50.0, 50.0),
        Point(101.0, 99.0),
        Point(100.0, 100.0),
        Point(99.0, 101.0),
        Point(102.0, 98.0),
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      expect(eyes!.leftEye!.irisCenter, equals(Point(100.0, 100.0)));
    });

    test('should handle collinear points', () {
      final irisPoints = [
        Point(100.0, 100.0),
        Point(110.0, 100.0),
        Point(105.0, 100.0),
        Point(102.0, 100.0),
        Point(108.0, 100.0),
      ];

      final face = createFaceWithIrisPoints(irisPoints);
      final eyes = face.eyes;

      expect(eyes!.leftEye!.irisCenter, equals(Point(105.0, 100.0)));
    });
  });

  group('Face.eyes - Contour Extraction', () {
    test('should exclude center from contour', () {
      final irisPoints = [
        Point(100.0, 100.0),
        Point(110.0, 100.0),
        Point(105.0, 105.0),
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
        Point(100.0, 100.0),
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
      final points = List.generate(200, (i) => Point(i.toDouble(), 50.0));
      final face = createFaceWithIrisPoints(points);
      final eyes = face.eyes;

      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      expect(eyes.leftEye!.mesh.length, 95);
      expect(eyes.rightEye!.mesh.length, 95);
    });
  });

  group('Face.landmarks - Iris Center Replacement', () {
    test(
        'should replace eye landmarks with iris centers when iris data present',
        () {
      // 10 iris points = 5 left + 5 right, so eyes returns both
      final irisPoints = [
        // Left eye iris: center-ish at (100, 100)
        const Point(100, 100),
        const Point(95, 100),
        const Point(105, 100),
        const Point(100, 95),
        const Point(100, 105),
        // Right eye iris: center-ish at (200, 100)
        const Point(200, 100),
        const Point(195, 100),
        const Point(205, 100),
        const Point(200, 95),
        const Point(200, 105),
      ];
      final face = createFaceWithIrisPoints(irisPoints);

      final eyes = face.eyes;
      expect(eyes, isNotNull);
      expect(eyes!.leftEye, isNotNull);
      expect(eyes.rightEye, isNotNull);

      final landmarks = face.landmarks;

      // The leftEye landmark should be replaced with the iris center
      expect(landmarks.leftEye!.x, eyes.leftEye!.irisCenter.x);
      expect(landmarks.leftEye!.y, eyes.leftEye!.irisCenter.y);

      // The rightEye landmark should be replaced with the iris center
      expect(landmarks.rightEye!.x, eyes.rightEye!.irisCenter.x);
      expect(landmarks.rightEye!.y, eyes.rightEye!.irisCenter.y);
    });

    test('should not replace eye landmarks when no iris data', () {
      final face = createFaceWithIrisPoints([]);
      final landmarks = face.landmarks;

      // Original detection keypoints should be used
      expect(landmarks.leftEye, isNotNull);
      expect(landmarks.rightEye, isNotNull);
    });
  });
}
