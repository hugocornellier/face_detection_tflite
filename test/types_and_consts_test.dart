import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart' show Size;
import 'package:image/image.dart' as img;
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

void main() {
  globalTestSetup();

  group('Point', () {
    test('should create 2D point without z coordinate', () {
      final point = Point(10.0, 20.0);

      expect(point.x, 10.0);
      expect(point.y, 20.0);
      expect(point.z, isNull);
      expect(point.is3D, false);
    });

    test('should create 3D point with z coordinate', () {
      final point = Point(10.0, 20.0, 5.0);

      expect(point.x, 10.0);
      expect(point.y, 20.0);
      expect(point.z, 5.0);
      expect(point.is3D, true);
    });

    test('should format toString for 2D point', () {
      final point = Point(10.5, 20.3);
      expect(point.toString(), 'Point(10.5, 20.3)');
    });

    test('should format toString for 3D point', () {
      final point = Point(10.5, 20.3, -2.1);
      expect(point.toString(), 'Point(10.5, 20.3, -2.1)');
    });

    test('should support equality for identical 2D points', () {
      final p1 = Point(10.0, 20.0);
      final p2 = Point(10.0, 20.0);

      expect(p1, equals(p2));
      expect(p1.hashCode, equals(p2.hashCode));
    });

    test('should support equality for identical 3D points', () {
      final p1 = Point(10.0, 20.0, 5.0);
      final p2 = Point(10.0, 20.0, 5.0);

      expect(p1, equals(p2));
      expect(p1.hashCode, equals(p2.hashCode));
    });

    test('should not equal different 2D points', () {
      final p1 = Point(10.0, 20.0);
      final p2 = Point(10.0, 21.0);

      expect(p1, isNot(equals(p2)));
    });

    test('should not equal 2D vs 3D point with same x,y', () {
      final p1 = Point(10.0, 20.0);
      final p2 = Point(10.0, 20.0, 5.0);

      expect(p1, isNot(equals(p2)));
    });

    test('should handle identical reference', () {
      final p1 = Point(10.0, 20.0);

      expect(p1, equals(p1));
    });

    test('should support negative coordinates', () {
      final point = Point(-10.0, -20.0, -5.0);

      expect(point.x, -10.0);
      expect(point.y, -20.0);
      expect(point.z, -5.0);
    });

    test('toMap/fromMap round-trip for 2D point', () {
      final point = Point(10.5, 20.3);
      final map = point.toMap();
      final restored = Point.fromMap(map);

      expect(restored, equals(point));
      expect(map['z'], isNull);
    });

    test('toMap/fromMap round-trip for 3D point', () {
      final point = Point(10.5, 20.3, -5.7);
      final map = point.toMap();
      final restored = Point.fromMap(map);

      expect(restored, equals(point));
      expect(map['z'], -5.7);
    });
  });

  group('FaceMesh', () {
    test('should create mesh with 468 points', () {
      final points = List.generate(
        468,
        (i) => Point(i.toDouble(), i.toDouble()),
      );
      final mesh = FaceMesh(points);

      expect(mesh.length, 468);
      expect(mesh.points.length, 468);
    });

    test('should support indexing', () {
      final points = List.generate(
        468,
        (i) => Point(i.toDouble(), i.toDouble()),
      );
      final mesh = FaceMesh(points);

      expect(mesh[0], equals(Point(0.0, 0.0)));
      expect(mesh[1], equals(Point(1.0, 1.0)));
      expect(mesh[467], equals(Point(467.0, 467.0)));
    });

    test('should format toString', () {
      final points = List.generate(
        468,
        (i) => Point(i.toDouble(), i.toDouble()),
      );
      final mesh = FaceMesh(points);

      expect(mesh.toString(), 'FaceMesh(468 points)');
    });

    test('should assert on wrong number of points', () {
      final points = List.generate(
        100,
        (i) => Point(i.toDouble(), i.toDouble()),
      );

      expect(() => FaceMesh(points), throwsA(isA<AssertionError>()));
    });

    test('should return same points list reference', () {
      final points = List.generate(
        468,
        (i) => Point(i.toDouble(), i.toDouble()),
      );
      final mesh = FaceMesh(points);

      expect(mesh.points, same(points));
    });

    test('toMap/fromMap round-trip preserves all points', () {
      final points = List.generate(
        468,
        (i) => Point(i.toDouble(), i * 2.0, i * 0.1),
      );
      final mesh = FaceMesh(points);
      final map = mesh.toMap();
      final restored = FaceMesh.fromMap(map);

      expect(restored.length, 468);
      for (int i = 0; i < 468; i++) {
        expect(restored[i].x, mesh[i].x);
        expect(restored[i].y, mesh[i].y);
        expect(restored[i].z, mesh[i].z);
      }
    });
  });

  group('Eye', () {
    test('should create eye with iris center and contour', () {
      final center = Point(100.0, 100.0);
      final contour = [
        Point(90.0, 90.0),
        Point(110.0, 90.0),
        Point(110.0, 110.0),
        Point(90.0, 110.0),
      ];
      final mesh = <Point>[];

      final eye = Eye(irisCenter: center, irisContour: contour, mesh: mesh);

      expect(eye.irisCenter, equals(center));
      expect(eye.irisContour.length, 4);
      expect(eye.mesh.length, 0);
    });

    test('should create eye with mesh data', () {
      final center = Point(100.0, 100.0);
      final contour = List.generate(
        4,
        (i) => Point(i.toDouble(), i.toDouble()),
      );
      final mesh = List.generate(71, (i) => Point(i.toDouble(), i.toDouble()));

      final eye = Eye(irisCenter: center, irisContour: contour, mesh: mesh);

      expect(eye.mesh.length, 71);
    });

    test('should return first 15 mesh points as contour', () {
      final center = Point(100.0, 100.0);
      final irisContour = List.generate(
        4,
        (i) => Point(i.toDouble(), i.toDouble()),
      );
      final mesh = List.generate(71, (i) => Point(100.0 + i, 100.0 + i));

      final eye = Eye(irisCenter: center, irisContour: irisContour, mesh: mesh);
      final contour = eye.contour;

      expect(contour.length, 15);
      expect(contour[0], equals(Point(100.0, 100.0)));
      expect(contour[14], equals(Point(114.0, 114.0)));
    });

    test(
      'should return full mesh as contour when mesh has less than 15 points',
      () {
        final center = Point(100.0, 100.0);
        final irisContour = List.generate(
          4,
          (i) => Point(i.toDouble(), i.toDouble()),
        );
        final mesh = List.generate(
          10,
          (i) => Point(i.toDouble(), i.toDouble()),
        );

        final eye = Eye(
          irisCenter: center,
          irisContour: irisContour,
          mesh: mesh,
        );

        expect(eye.contour.length, 10);
        expect(eye.contour, equals(mesh));
      },
    );

    test('should return empty contour when mesh is empty', () {
      final center = Point(100.0, 100.0);
      final irisContour = List.generate(
        4,
        (i) => Point(i.toDouble(), i.toDouble()),
      );
      final mesh = <Point>[];

      final eye = Eye(irisCenter: center, irisContour: irisContour, mesh: mesh);

      expect(eye.contour, isEmpty);
    });

    test('toMap/fromMap round-trip preserves all eye data', () {
      final center = Point(100.0, 100.0, 5.0);
      final irisContour = [
        Point(90.0, 90.0, 4.0),
        Point(110.0, 90.0, 4.0),
        Point(110.0, 110.0, 4.0),
        Point(90.0, 110.0, 4.0),
      ];
      final mesh = List.generate(
        71,
        (i) => Point(i.toDouble(), i * 2.0, i * 0.1),
      );

      final eye = Eye(irisCenter: center, irisContour: irisContour, mesh: mesh);
      final map = eye.toMap();
      final restored = Eye.fromMap(map);

      expect(restored.irisCenter.x, eye.irisCenter.x);
      expect(restored.irisCenter.y, eye.irisCenter.y);
      expect(restored.irisCenter.z, eye.irisCenter.z);
      expect(restored.irisContour.length, 4);
      expect(restored.mesh.length, 71);
    });
  });

  group('EyePair', () {
    test('should create with both eyes', () {
      final leftEye = Eye(
        irisCenter: Point(50.0, 50.0),
        irisContour: [],
        mesh: [],
      );
      final rightEye = Eye(
        irisCenter: Point(150.0, 50.0),
        irisContour: [],
        mesh: [],
      );

      final pair = EyePair(leftEye: leftEye, rightEye: rightEye);

      expect(pair.leftEye, isNotNull);
      expect(pair.rightEye, isNotNull);
    });

    test('should create with only left eye', () {
      final leftEye = Eye(
        irisCenter: Point(50.0, 50.0),
        irisContour: [],
        mesh: [],
      );

      final pair = EyePair(leftEye: leftEye, rightEye: null);

      expect(pair.leftEye, isNotNull);
      expect(pair.rightEye, isNull);
    });

    test('should create with only right eye', () {
      final rightEye = Eye(
        irisCenter: Point(150.0, 50.0),
        irisContour: [],
        mesh: [],
      );

      final pair = EyePair(leftEye: null, rightEye: rightEye);

      expect(pair.leftEye, isNull);
      expect(pair.rightEye, isNotNull);
    });

    test('should create with both eyes null', () {
      final pair = EyePair(leftEye: null, rightEye: null);

      expect(pair.leftEye, isNull);
      expect(pair.rightEye, isNull);
    });

    test('toMap/fromMap round-trip with both eyes', () {
      final leftEye = Eye(
        irisCenter: Point(50.0, 50.0),
        irisContour: [
          Point(45.0, 50.0),
          Point(55.0, 50.0),
          Point(50.0, 45.0),
          Point(50.0, 55.0),
        ],
        mesh: [],
      );
      final rightEye = Eye(
        irisCenter: Point(150.0, 50.0),
        irisContour: [
          Point(145.0, 50.0),
          Point(155.0, 50.0),
          Point(150.0, 45.0),
          Point(150.0, 55.0),
        ],
        mesh: [],
      );

      final pair = EyePair(leftEye: leftEye, rightEye: rightEye);
      final map = pair.toMap();
      final restored = EyePair.fromMap(map);

      expect(restored.leftEye, isNotNull);
      expect(restored.rightEye, isNotNull);
      expect(restored.leftEye!.irisCenter.x, 50.0);
      expect(restored.rightEye!.irisCenter.x, 150.0);
    });

    test('toMap/fromMap round-trip with null eyes', () {
      final pair = EyePair(leftEye: null, rightEye: null);
      final map = pair.toMap();
      final restored = EyePair.fromMap(map);

      expect(restored.leftEye, isNull);
      expect(restored.rightEye, isNull);
    });
  });

  group('FaceLandmarks', () {
    late Map<FaceLandmarkType, Point> landmarkMap;

    setUp(() {
      landmarkMap = {
        FaceLandmarkType.leftEye: Point(30.0, 40.0),
        FaceLandmarkType.rightEye: Point(70.0, 40.0),
        FaceLandmarkType.noseTip: Point(50.0, 60.0),
        FaceLandmarkType.mouth: Point(50.0, 75.0),
        FaceLandmarkType.leftEyeTragion: Point(10.0, 40.0),
        FaceLandmarkType.rightEyeTragion: Point(90.0, 40.0),
      };
    });

    test('should access landmarks by named getters', () {
      final landmarks = FaceLandmarks(landmarkMap);

      expect(landmarks.leftEye, equals(Point(30.0, 40.0)));
      expect(landmarks.rightEye, equals(Point(70.0, 40.0)));
      expect(landmarks.noseTip, equals(Point(50.0, 60.0)));
      expect(landmarks.mouth, equals(Point(50.0, 75.0)));
      expect(landmarks.leftEyeTragion, equals(Point(10.0, 40.0)));
      expect(landmarks.rightEyeTragion, equals(Point(90.0, 40.0)));
    });

    test('should support indexing by FaceLandmarkType', () {
      final landmarks = FaceLandmarks(landmarkMap);

      expect(landmarks[FaceLandmarkType.leftEye], equals(Point(30.0, 40.0)));
      expect(landmarks[FaceLandmarkType.noseTip], equals(Point(50.0, 60.0)));
    });

    test('should return null for missing landmarks', () {
      final partialMap = {FaceLandmarkType.leftEye: Point(30.0, 40.0)};
      final landmarks = FaceLandmarks(partialMap);

      expect(landmarks.rightEye, isNull);
      expect(landmarks.noseTip, isNull);
    });

    test('should provide values iterable', () {
      final landmarks = FaceLandmarks(landmarkMap);
      final values = landmarks.values.toList();

      expect(values.length, 6);
      expect(values, contains(Point(30.0, 40.0)));
      expect(values, contains(Point(70.0, 40.0)));
    });

    test('should provide keys iterable', () {
      final landmarks = FaceLandmarks(landmarkMap);
      final keys = landmarks.keys.toList();

      expect(keys.length, 6);
      expect(keys, contains(FaceLandmarkType.leftEye));
      expect(keys, contains(FaceLandmarkType.noseTip));
    });

    test('should convert to map', () {
      final landmarks = FaceLandmarks(landmarkMap);
      final map = landmarks.toMap();

      expect(map.length, 6);
      expect(map[FaceLandmarkType.leftEye], equals(Point(30.0, 40.0)));
    });

    test('toSerializableMap/fromSerializableMap round-trip', () {
      final landmarks = FaceLandmarks(landmarkMap);
      final map = landmarks.toSerializableMap();
      final restored = FaceLandmarks.fromSerializableMap(map);

      expect(restored.leftEye, equals(Point(30.0, 40.0)));
      expect(restored.rightEye, equals(Point(70.0, 40.0)));
      expect(restored.noseTip, equals(Point(50.0, 60.0)));
      expect(restored.mouth, equals(Point(50.0, 75.0)));
    });
  });

  group('BoundingBox', () {
    test('should calculate width correctly', () {
      final bbox = BoundingBox(
        topLeft: Point(10.0, 20.0),
        topRight: Point(90.0, 20.0),
        bottomRight: Point(90.0, 80.0),
        bottomLeft: Point(10.0, 80.0),
      );

      expect(bbox.width, 80.0);
    });

    test('should calculate height correctly', () {
      final bbox = BoundingBox(
        topLeft: Point(10.0, 20.0),
        topRight: Point(90.0, 20.0),
        bottomRight: Point(90.0, 80.0),
        bottomLeft: Point(10.0, 80.0),
      );

      expect(bbox.height, 60.0);
    });

    test('should calculate center point', () {
      final bbox = BoundingBox(
        topLeft: Point(10.0, 20.0),
        topRight: Point(90.0, 20.0),
        bottomRight: Point(90.0, 80.0),
        bottomLeft: Point(10.0, 80.0),
      );

      expect(bbox.center.x, 50.0);
      expect(bbox.center.y, 50.0);
    });

    test('should return corners in correct order', () {
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
      expect(corners.length, 4);
      expect(corners[0], equals(topLeft));
      expect(corners[1], equals(topRight));
      expect(corners[2], equals(bottomRight));
      expect(corners[3], equals(bottomLeft));
    });

    test('should handle non-axis-aligned box', () {
      final bbox = BoundingBox(
        topLeft: Point(20.0, 10.0),
        topRight: Point(80.0, 20.0),
        bottomRight: Point(70.0, 90.0),
        bottomLeft: Point(10.0, 80.0),
      );

      expect(bbox.width, 60.0);
      expect(bbox.height, 70.0);
    });

    test('should handle zero-size box', () {
      final bbox = BoundingBox(
        topLeft: Point(50.0, 50.0),
        topRight: Point(50.0, 50.0),
        bottomRight: Point(50.0, 50.0),
        bottomLeft: Point(50.0, 50.0),
      );

      expect(bbox.width, 0.0);
      expect(bbox.height, 0.0);
      expect(bbox.center, equals(Point(50.0, 50.0)));
    });

    test('toMap/fromMap round-trip preserves corners', () {
      final bbox = BoundingBox(
        topLeft: Point(10.0, 20.0),
        topRight: Point(90.0, 20.0),
        bottomRight: Point(90.0, 80.0),
        bottomLeft: Point(10.0, 80.0),
      );
      final map = bbox.toMap();
      final restored = BoundingBox.fromMap(map);

      expect(restored.topLeft, equals(bbox.topLeft));
      expect(restored.topRight, equals(bbox.topRight));
      expect(restored.bottomRight, equals(bbox.bottomRight));
      expect(restored.bottomLeft, equals(bbox.bottomLeft));
    });
  });

  group('RectF', () {
    test('should calculate width', () {
      final rect = RectF(0.2, 0.3, 0.8, 0.7);
      expect(rect.w, closeTo(0.6, 0.0001));
    });

    test('should calculate height', () {
      final rect = RectF(0.2, 0.3, 0.8, 0.7);
      expect(rect.h, closeTo(0.4, 0.0001));
    });

    test('should scale correctly', () {
      final rect = RectF(0.2, 0.3, 0.8, 0.7);
      final scaled = rect.scale(2.0, 3.0);

      expect(scaled.xmin, closeTo(0.4, 0.0001));
      expect(scaled.ymin, closeTo(0.9, 0.0001));
      expect(scaled.xmax, closeTo(1.6, 0.0001));
      expect(scaled.ymax, closeTo(2.1, 0.0001));
    });

    test('should expand preserving center', () {
      final rect = RectF(0.4, 0.4, 0.6, 0.6);
      final expanded = rect.expand(0.5);

      final centerX = (expanded.xmin + expanded.xmax) / 2;
      final centerY = (expanded.ymin + expanded.ymax) / 2;

      expect(centerX, closeTo(0.5, 0.0001));
      expect(centerY, closeTo(0.5, 0.0001));

      expect(expanded.w, closeTo(0.3, 0.0001));
      expect(expanded.h, closeTo(0.3, 0.0001));
    });

    test('should expand with negative fraction (shrink)', () {
      final rect = RectF(0.3, 0.3, 0.7, 0.7);
      final shrunk = rect.expand(-0.25);

      final centerX = (shrunk.xmin + shrunk.xmax) / 2;
      final centerY = (shrunk.ymin + shrunk.ymax) / 2;

      expect(centerX, closeTo(0.5, 0.0001));
      expect(centerY, closeTo(0.5, 0.0001));

      expect(shrunk.w, closeTo(0.3, 0.0001));
      expect(shrunk.h, closeTo(0.3, 0.0001));
    });

    test('should handle zero-size rect', () {
      final rect = RectF(0.5, 0.5, 0.5, 0.5);

      expect(rect.w, 0.0);
      expect(rect.h, 0.0);
    });

    test('toMap/fromMap round-trip preserves coordinates', () {
      final rect = RectF(0.2, 0.3, 0.8, 0.7);
      final map = rect.toMap();
      final restored = RectF.fromMap(map);

      expect(restored.xmin, rect.xmin);
      expect(restored.ymin, rect.ymin);
      expect(restored.xmax, rect.xmax);
      expect(restored.ymax, rect.ymax);
    });
  });

  group('Detection', () {
    test('should create detection with normalized coordinates', () {
      final bbox = RectF(0.2, 0.3, 0.8, 0.7);
      final keypoints = TestUtils.generateValidKeypoints();

      final detection = Detection(
        boundingBox: bbox,
        score: 0.95,
        keypointsXY: keypoints,
        imageSize: TestConstants.mediumImage,
      );

      expect(detection.boundingBox, equals(bbox));
      expect(detection.score, 0.95);
      expect(detection.keypointsXY.length, 12);
    });

    test('should support operator indexing', () {
      final bbox = RectF(0.2, 0.3, 0.8, 0.7);
      final keypoints = TestUtils.generateValidKeypoints();

      final detection = Detection(
        boundingBox: bbox,
        score: 0.95,
        keypointsXY: keypoints,
        imageSize: TestConstants.mediumImage,
      );

      expect(detection[0], keypoints[0]);
      expect(detection[1], keypoints[1]);
    });

    test('should denormalize landmarks to pixel coordinates', () {
      final bbox = RectF(0.2, 0.3, 0.8, 0.7);
      final keypoints = [
        0.5,
        0.5,
        0.0,
        0.0,
        1.0,
        1.0,
        0.25,
        0.75,
        0.1,
        0.2,
        0.9,
        0.8,
      ];

      final detection = Detection(
        boundingBox: bbox,
        score: 0.95,
        keypointsXY: keypoints,
        imageSize: Size(640, 480),
      );

      final landmarks = detection.landmarks;

      expect(landmarks[FaceLandmarkType.leftEye]!.x, 320.0);
      expect(landmarks[FaceLandmarkType.leftEye]!.y, 240.0);
      expect(landmarks[FaceLandmarkType.rightEye]!.x, 0.0);
      expect(landmarks[FaceLandmarkType.rightEye]!.y, 0.0);
      expect(landmarks[FaceLandmarkType.noseTip]!.x, 640.0);
      expect(landmarks[FaceLandmarkType.noseTip]!.y, 480.0);
    });

    test('should throw when landmarks called without imageSize', () {
      final bbox = RectF(0.2, 0.3, 0.8, 0.7);
      final keypoints = TestUtils.generateValidKeypoints();

      final detection = Detection(
        boundingBox: bbox,
        score: 0.95,
        keypointsXY: keypoints,
        imageSize: null,
      );

      expect(() => detection.landmarks, throwsA(isA<StateError>()));
    });

    test('toMap/fromMap round-trip preserves all data', () {
      final bbox = RectF(0.2, 0.3, 0.8, 0.7);
      final keypoints = TestUtils.generateValidKeypoints();

      final detection = Detection(
        boundingBox: bbox,
        score: 0.95,
        keypointsXY: keypoints,
        imageSize: TestConstants.mediumImage,
      );

      final map = detection.toMap();
      final restored = Detection.fromMap(map);

      expect(restored.boundingBox.xmin, detection.boundingBox.xmin);
      expect(restored.boundingBox.ymax, detection.boundingBox.ymax);
      expect(restored.score, detection.score);
      expect(restored.keypointsXY.length, detection.keypointsXY.length);
      expect(restored.imageSize!.width, detection.imageSize!.width);
      expect(restored.imageSize!.height, detection.imageSize!.height);
    });

    test('toMap/fromMap handles null imageSize', () {
      final bbox = RectF(0.2, 0.3, 0.8, 0.7);
      final keypoints = TestUtils.generateValidKeypoints();

      final detection = Detection(
        boundingBox: bbox,
        score: 0.95,
        keypointsXY: keypoints,
        imageSize: null,
      );

      final map = detection.toMap();
      final restored = Detection.fromMap(map);

      expect(restored.imageSize, isNull);
    });
  });

  group('ImageTensor', () {
    test('should store tensor data and metadata', () {
      final tensor = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final padding = [0.1, 0.2, 0.3, 0.4];

      final imageTensor = ImageTensor(tensor, padding, 128, 128);

      expect(imageTensor.tensorNHWC, equals(tensor));
      expect(imageTensor.padding, equals(padding));
      expect(imageTensor.width, 128);
      expect(imageTensor.height, 128);
    });
  });

  group('AlignedFace', () {
    test('should store aligned face data', () {
      final faceCrop = img.Image(width: 2, height: 2);

      final aligned = AlignedFace(
        cx: 100.0,
        cy: 100.0,
        size: 50.0,
        theta: 0.5,
        faceCrop: faceCrop,
      );

      expect(aligned.cx, 100.0);
      expect(aligned.cy, 100.0);
      expect(aligned.size, 50.0);
      expect(aligned.theta, 0.5);
      expect(aligned.faceCrop, equals(faceCrop));
    });
  });

  group('DecodedRgb', () {
    test('should store decoded image data', () {
      final rgb = Uint8List.fromList([255, 0, 0, 0, 255, 0]);

      final decoded = DecodedRgb(2, 1, rgb);

      expect(decoded.width, 2);
      expect(decoded.height, 1);
      expect(decoded.rgb, equals(rgb));
    });
  });

  group('Face', () {
    test('should compute pixel-space bounding box from normalized values', () {
      final detection = Detection(
        boundingBox: RectF(0.1, 0.2, 0.4, 0.6),
        score: 0.9,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: const Size(200, 100),
      );

      final face = Face(
        detection: detection,
        mesh: null,
        irises: const [],
        originalSize: const Size(200, 100),
      );

      final bbox = face.boundingBox;
      expect(bbox.topLeft, equals(Point(20.0, 20.0)));
      expect(bbox.topRight, equals(Point(80.0, 20.0)));
      expect(bbox.bottomRight, equals(Point(80.0, 60.0)));
      expect(bbox.bottomLeft, equals(Point(20.0, 60.0)));
    });

    test('should override eye landmarks with iris centers when available', () {
      final detection = Detection(
        boundingBox: RectF(0.0, 0.0, 1.0, 1.0),
        score: 0.9,
        keypointsXY: TestUtils.generateValidKeypoints(
          leftEye: Point(0.25, 0.25),
          rightEye: Point(0.75, 0.25),
        ),
        imageSize: const Size(100, 100),
      );

      final leftEyePoints = [
        const Point(10.0, 10.0),
        const Point(20.0, 10.0),
        const Point(15.0, 15.0),
        const Point(15.0, 8.0),
        const Point(15.0, 22.0),
      ];
      final rightEyePoints = [
        const Point(80.0, 10.0),
        const Point(90.0, 10.0),
        const Point(85.0, 15.0),
        const Point(85.0, 8.0),
        const Point(85.0, 22.0),
      ];

      final face = Face(
        detection: detection,
        mesh: null,
        irises: [...leftEyePoints, ...rightEyePoints],
        originalSize: const Size(100, 100),
      );

      final landmarks = face.landmarks;
      expect(landmarks.leftEye, equals(const Point(15.0, 15.0)));
      expect(landmarks.rightEye, equals(const Point(85.0, 15.0)));

      expect(landmarks.noseTip, equals(const Point(50.0, 60.0)));
    });

    test('toMap/fromMap round-trip preserves face with mesh and iris', () {
      final detection = Detection(
        boundingBox: RectF(0.1, 0.2, 0.4, 0.6),
        score: 0.9,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: const Size(200, 100),
      );

      final meshPoints = List.generate(
        468,
        (i) => Point(i.toDouble(), i * 2.0, i * 0.1),
      );
      final mesh = FaceMesh(meshPoints);

      final irisPoints = List.generate(
        10,
        (i) => Point(i * 10.0, i * 5.0, i * 0.5),
      );

      final face = Face(
        detection: detection,
        mesh: mesh,
        irises: irisPoints,
        originalSize: const Size(200, 100),
      );

      final map = face.toMap();
      final restored = Face.fromMap(map);

      expect(
        restored.boundingBox.topLeft.x,
        closeTo(face.boundingBox.topLeft.x, 0.01),
      );
      expect(
        restored.boundingBox.topLeft.y,
        closeTo(face.boundingBox.topLeft.y, 0.01),
      );

      expect(restored.mesh, isNotNull);
      expect(restored.mesh!.length, 468);
      expect(restored.mesh![0].x, face.mesh![0].x);
      expect(restored.mesh![0].z, face.mesh![0].z);

      expect(restored.irisPoints.length, 10);
      expect(restored.irisPoints[0].x, face.irisPoints[0].x);

      expect(restored.originalSize.width, 200);
      expect(restored.originalSize.height, 100);
    });

    test('toMap/fromMap round-trip handles null mesh', () {
      final detection = Detection(
        boundingBox: RectF(0.1, 0.2, 0.4, 0.6),
        score: 0.9,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: const Size(200, 100),
      );

      final face = Face(
        detection: detection,
        mesh: null,
        irises: const [],
        originalSize: const Size(200, 100),
      );

      final map = face.toMap();
      final restored = Face.fromMap(map);

      expect(restored.mesh, isNull);
      expect(restored.irisPoints, isEmpty);
    });
  });

  group('PerformanceConfig constructors', () {
    test('xnnpack constructor sets correct mode', () {
      const config = PerformanceConfig.xnnpack(numThreads: 4);

      expect(config.mode, PerformanceMode.xnnpack);
      expect(config.numThreads, 4);
    });

    test('xnnpack constructor with default threads', () {
      const config = PerformanceConfig.xnnpack();

      expect(config.mode, PerformanceMode.xnnpack);
      expect(config.numThreads, isNull);
    });

    test('gpu constructor sets correct mode', () {
      const config = PerformanceConfig.gpu(numThreads: 2);

      expect(config.mode, PerformanceMode.gpu);
      expect(config.numThreads, 2);
    });

    test('gpu constructor with default threads', () {
      const config = PerformanceConfig.gpu();

      expect(config.mode, PerformanceMode.gpu);
      expect(config.numThreads, isNull);
    });

    test('auto constructor sets correct mode', () {
      const config = PerformanceConfig.auto(numThreads: 8);

      expect(config.mode, PerformanceMode.auto);
      expect(config.numThreads, 8);
    });

    test('auto constructor with default threads', () {
      const config = PerformanceConfig.auto();

      expect(config.mode, PerformanceMode.auto);
      expect(config.numThreads, isNull);
    });

    test('disabled constant has correct mode', () {
      expect(PerformanceConfig.disabled.mode, PerformanceMode.disabled);
      expect(PerformanceConfig.disabled.numThreads, isNull);
    });

    test('default constructor sets correct mode', () {
      const config = PerformanceConfig(mode: PerformanceMode.xnnpack);

      expect(config.mode, PerformanceMode.xnnpack);
      expect(config.numThreads, isNull);
    });
  });

  group('Getter caching', () {
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

    List<Point> generate152Points() {
      return List.generate(152, (i) => Point(i.toDouble(), i.toDouble()));
    }

    test('Face.eyes returns identical object on repeated access', () {
      final face = createFaceWithIrisPoints(generate152Points());

      final eyes1 = face.eyes;
      final eyes2 = face.eyes;

      expect(eyes1, isNotNull);
      expect(
        identical(eyes1, eyes2),
        isTrue,
        reason: 'Face.eyes should return cached object on repeated access',
      );
    });

    test('Face.eyes caches null result for empty iris points', () {
      final face = createFaceWithIrisPoints([]);

      final eyes1 = face.eyes;
      final eyes2 = face.eyes;

      expect(eyes1, isNull);
      expect(identical(eyes1, eyes2), isTrue);
    });

    test(
      'Eye.contour returns identical object when using optimized constructor',
      () {
        final mesh = List.generate(
          71,
          (i) => Point(i.toDouble(), i.toDouble()),
        );
        final eye = Eye.optimized(
          irisCenter: Point(100, 100),
          irisContour: [Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)],
          mesh: mesh,
        );

        final contour1 = eye.contour;
        final contour2 = eye.contour;

        expect(contour1.length, kMaxEyeLandmark);
        expect(
          identical(contour1, contour2),
          isTrue,
          reason:
              'Eye.contour should return cached object when using optimized constructor',
        );
      },
    );

    test('Eye.contour computes on access for const constructor', () {
      const eye = Eye(
        irisCenter: Point(100, 100, 0),
        irisContour: [Point(1, 1, 0)],
      );

      expect(eye.contour, isEmpty);
    });

    test('Eye.fromMap uses optimized constructor with caching', () {
      final mesh = List.generate(71, (i) => Point(i.toDouble(), i.toDouble()));
      final originalEye = Eye.optimized(
        irisCenter: Point(100, 100),
        irisContour: [Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)],
        mesh: mesh,
      );

      final map = originalEye.toMap();
      final restoredEye = Eye.fromMap(map);

      final contour1 = restoredEye.contour;
      final contour2 = restoredEye.contour;

      expect(
        identical(contour1, contour2),
        isTrue,
        reason: 'Eye.fromMap should use optimized constructor with caching',
      );
    });

    test('Face serialization round-trip recomputes cache correctly', () {
      final face = createFaceWithIrisPoints(generate152Points());
      final originalEyes = face.eyes;

      final map = face.toMap();
      final restored = Face.fromMap(map);
      final restoredEyes = restored.eyes;

      expect(
        identical(originalEyes, restoredEyes),
        isFalse,
        reason: 'Serialization creates new objects',
      );
      expect(
        restoredEyes?.leftEye?.irisCenter,
        equals(originalEyes?.leftEye?.irisCenter),
      );
      expect(
        restoredEyes?.rightEye?.irisCenter,
        equals(originalEyes?.rightEye?.irisCenter),
      );

      final restoredEyes2 = restored.eyes;
      expect(identical(restoredEyes, restoredEyes2), isTrue);
    });
  });
}
