// ignore_for_file: avoid_print

import 'dart:math' show sqrt;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Test helper to create a minimal 1x1 PNG image
class TestUtils {
  static Uint8List createDummyImageBytes() {
    return Uint8List.fromList([
      0x89,
      0x50,
      0x4E,
      0x47,
      0x0D,
      0x0A,
      0x1A,
      0x0A,
      0x00,
      0x00,
      0x00,
      0x0D,
      0x49,
      0x48,
      0x44,
      0x52,
      0x00,
      0x00,
      0x00,
      0x01,
      0x00,
      0x00,
      0x00,
      0x01,
      0x08,
      0x06,
      0x00,
      0x00,
      0x00,
      0x1F,
      0x15,
      0xC4,
      0x89,
      0x00,
      0x00,
      0x00,
      0x0A,
      0x49,
      0x44,
      0x41,
      0x54,
      0x78,
      0x9C,
      0x63,
      0x00,
      0x01,
      0x00,
      0x00,
      0x05,
      0x00,
      0x01,
      0x0D,
      0x0A,
      0x2D,
      0xB4,
      0x00,
      0x00,
      0x00,
      0x00,
      0x49,
      0x45,
      0x4E,
      0x44,
      0xAE,
      0x42,
      0x60,
      0x82
    ]);
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('FaceDetector - Initialization and Disposal', () {
    test('should initialize successfully with default options', () async {
      final detector = FaceDetector();
      expect(detector.isReady, false);

      await detector.initialize();
      expect(detector.isReady, true);

      detector.dispose();
    });

    test('should allow re-initialization', () async {
      final detector = FaceDetector();
      await detector.initialize();
      expect(detector.isReady, true);

      await detector.initialize();
      expect(detector.isReady, true);

      detector.dispose();
    });

    test('should handle multiple dispose calls', () async {
      final detector = FaceDetector();
      await detector.initialize();
      detector.dispose();

      try {
        detector.dispose();
      } catch (e) {
        expect(e, isA<StateError>());
      }
    });
  });

  group('FaceDetector - Error Handling', () {
    test('should throw StateError when detectFaces() called before initialize',
        () async {
      final detector = FaceDetector();
      final bytes = TestUtils.createDummyImageBytes();

      expect(
        () => detector.detectFaces(bytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    test('should handle invalid image bytes gracefully', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

      try {
        final results = await detector.detectFaces(invalidBytes);
        expect(results, isEmpty);
      } catch (e) {
        expect(e, isNotNull);
      }

      detector.dispose();
    });
  });

  group('FaceDetector - detectFaces() with real images', () {
    test('should detect faces in landmark-ex1.jpg with full mode', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        expect(face.boundingBox, isNotNull);
        expect(face.boundingBox.topLeft.x, greaterThanOrEqualTo(0));
        expect(face.boundingBox.topLeft.y, greaterThanOrEqualTo(0));
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.boundingBox.height, greaterThan(0));

        expect(face.landmarks.leftEye, isNotNull);
        expect(face.landmarks.rightEye, isNotNull);
        expect(face.landmarks.noseTip, isNotNull);
        expect(face.landmarks.mouth, isNotNull);
        expect(face.landmarks.leftEyeTragion, isNotNull);
        expect(face.landmarks.rightEyeTragion, isNotNull);

        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);

        expect(face.irisPoints, isNotEmpty);

        expect(face.originalSize.width, greaterThan(0));
        expect(face.originalSize.height, greaterThan(0));
      }

      detector.dispose();
    });

    test('should detect faces in iris-detection-ex1.jpg', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      expect(results, isNotEmpty);
      detector.dispose();
    });

    test('should detect faces in iris-detection-ex2.jpg', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/iris-detection-ex2.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      expect(results, isNotEmpty);
      detector.dispose();
    });

    test('should detect multiple faces in group-shot-bounding-box-ex1.jpeg',
        () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      expect(results.length, 4,
          reason: 'group-shot-bounding-box-ex1.jpeg should have 4 faces');

      for (final face in results) {
        expect(face.boundingBox, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.landmarks.leftEye, isNotNull);
        expect(face.landmarks.rightEye, isNotNull);
        expect(face.landmarks.noseTip, isNotNull);
        expect(face.landmarks.mouth, isNotNull);
        expect(face.landmarks.leftEyeTragion, isNotNull);
        expect(face.landmarks.rightEyeTragion, isNotNull);
        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);
        expect(
          face.irisPoints,
          isNotEmpty,
          reason: 'Iris points expected for all faces in group image',
        );
      }

      detector.dispose();
    });

    test('should detect face in mesh-ex1.jpeg', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/mesh-ex1.jpeg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.standard,
      );

      expect(results, isNotEmpty);
      expect(results.first.mesh, isNotNull);
      expect(results.first.mesh!.points.length, 468);

      detector.dispose();
    });
  });

  group('FaceDetector - Different Detection Modes', () {
    test('should work with fast mode (bounding box + keypoints only)',
        () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.fast,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        expect(face.boundingBox, isNotNull);
        expect(face.landmarks, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));

        expect(face.mesh, isNull);
        expect(face.irisPoints, isEmpty);
      }

      detector.dispose();
    });

    test('should work with standard mode (bounding box + keypoints + mesh)',
        () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.standard,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        expect(face.boundingBox, isNotNull);
        expect(face.landmarks, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);

        expect(face.irisPoints, isEmpty);
      }

      detector.dispose();
    });

    test('should work with full mode (all features including iris)', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        expect(face.boundingBox, isNotNull);
        expect(face.landmarks, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);
        expect(face.irisPoints, isNotEmpty);
      }

      detector.dispose();
    });
  });

  group('FaceDetector - Landmark Access', () {
    late FaceDetector detector;
    late List<Face> faces;

    setUpAll(() async {
      detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      faces = await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
    });

    tearDownAll(() {
      detector.dispose();
    });

    test('should access facial keypoints', () {
      expect(faces, isNotEmpty);
      final face = faces.first;

      final leftEye = face.landmarks.leftEye;
      expect(leftEye, isNotNull);
      expect(leftEye!.x, greaterThanOrEqualTo(0));
      expect(leftEye.y, greaterThanOrEqualTo(0));

      final rightEye = face.landmarks.rightEye;
      expect(rightEye, isNotNull);
      expect(rightEye!.x, greaterThanOrEqualTo(0));
      expect(rightEye.y, greaterThanOrEqualTo(0));

      final noseTip = face.landmarks.noseTip;
      expect(noseTip, isNotNull);
      expect(noseTip!.x, greaterThanOrEqualTo(0));
      expect(noseTip.y, greaterThanOrEqualTo(0));

      final mouth = face.landmarks.mouth;
      expect(mouth, isNotNull);
      expect(mouth!.x, greaterThanOrEqualTo(0));
      expect(mouth.y, greaterThanOrEqualTo(0));

      final leftEyeTragion = face.landmarks.leftEyeTragion;
      expect(leftEyeTragion, isNotNull);

      final rightEyeTragion = face.landmarks.rightEyeTragion;
      expect(rightEyeTragion, isNotNull);
    });

    test('should have valid mesh coordinates', () {
      final face = faces.first;
      final mesh = face.mesh;

      expect(mesh, isNotNull);
      expect(mesh!.points.length, 468);

      for (final point in mesh.points) {
        expect(point.x, greaterThanOrEqualTo(0));
        expect(point.x, lessThanOrEqualTo(face.originalSize.width.toDouble()));
        expect(point.y, greaterThanOrEqualTo(0));
        expect(point.y, lessThanOrEqualTo(face.originalSize.height.toDouble()));

        if (point.is3D) {
          expect(point.z, isNotNull);
        }
      }
    });

    test('should access eyes and iris data', () {
      final face = faces.first;
      final eyes = face.eyes;

      if (face.irisPoints.isNotEmpty && eyes != null) {
        final leftEye = eyes.leftEye;
        if (leftEye != null) {
          expect(leftEye.irisCenter, isNotNull);
          expect(leftEye.irisContour.length, 4);
          expect(leftEye.mesh.length, 71);
        }

        final rightEye = eyes.rightEye;
        if (rightEye != null) {
          expect(rightEye.irisCenter, isNotNull);
          expect(rightEye.irisContour.length, 4);
          expect(rightEye.mesh.length, 71);
        }
      }
    });

    test('should access bounding box properties', () {
      final face = faces.first;
      final bbox = face.boundingBox;

      expect(bbox.topLeft.x, greaterThanOrEqualTo(0));
      expect(bbox.topLeft.y, greaterThanOrEqualTo(0));
      expect(bbox.width, greaterThan(0));
      expect(bbox.height, greaterThan(0));

      expect(bbox.topLeft.x,
          lessThanOrEqualTo(face.originalSize.width.toDouble()));
      expect(bbox.topLeft.y,
          lessThanOrEqualTo(face.originalSize.height.toDouble()));
      expect(bbox.bottomRight.x,
          lessThanOrEqualTo(face.originalSize.width.toDouble()));
      expect(bbox.bottomRight.y,
          lessThanOrEqualTo(face.originalSize.height.toDouble()));
    });
  });

  group('FaceDetector - Multiple Images', () {
    test('should process multiple images sequentially', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final images = [
        'assets/samples/landmark-ex1.jpg',
        'assets/samples/iris-detection-ex1.jpg',
        'assets/samples/iris-detection-ex2.jpg',
      ];

      for (final imagePath in images) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();
        final List<Face> results = await detector.detectFaces(bytes);

        expect(results, isNotEmpty, reason: 'Failed to detect in $imagePath');
      }

      detector.dispose();
    });

    test('should handle different image sizes', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final images = [
        'assets/samples/landmark-ex1.jpg',
        'assets/samples/mesh-ex1.jpeg',
        'assets/samples/group-shot-bounding-box-ex1.jpeg',
      ];

      for (final imagePath in images) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();
        final List<Face> results = await detector.detectFaces(bytes);

        if (results.isNotEmpty) {
          for (final face in results) {
            expect(face.originalSize.width, greaterThan(0));
            expect(face.originalSize.height, greaterThan(0));
          }
        }
      }

      detector.dispose();
    });
  });

  group('FaceDetector - Edge Cases', () {
    test('should handle 1x1 image', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes = TestUtils.createDummyImageBytes();
      final List<Face> results = await detector.detectFaces(bytes);

      expect(results, isNotNull);

      detector.dispose();
    });

    test('Face.toString() should not crash', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(bytes);

      expect(results, isNotEmpty);

      final faceString = results.first.toString();
      expect(faceString, isNotEmpty);
      expect(faceString, contains('Face'));

      detector.dispose();
    });

    test('should handle normal detection', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Face> results = await detector.detectFaces(bytes);

      expect(results, isNotEmpty);

      for (final face in results) {
        expect(face.boundingBox, isNotNull);
      }

      detector.dispose();
    });

    test('should track iris statistics', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      for (int i = 0; i < 3; i++) {
        await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
      }

      final totalIrisAttempts = detector.irisOkCount + detector.irisFailCount;
      expect(totalIrisAttempts, greaterThan(0));

      detector.dispose();
    });
  });

  group('FaceDetector - OpenCV API (detectFacesFromMat)', () {
    test('should detect faces using cv.Mat input', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      expect(mat.isEmpty, false);

      final List<Face> results = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.full,
      );

      mat.dispose();

      expect(results, isNotEmpty);

      for (final face in results) {
        expect(face.boundingBox, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);
      }

      detector.dispose();
    });

    test('should detect faces using detectFaces (OpenCV accelerated)',
        () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        expect(face.boundingBox, isNotNull);
        expect(face.mesh, isNotNull);
        expect(face.irisPoints, isNotEmpty);
      }

      detector.dispose();
    });

    test('detectFacesFromMat should produce same results as detectFaces',
        () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Face> oldResults = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      final List<Face> newResults = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.full,
      );
      mat.dispose();

      expect(newResults.length, oldResults.length);

      for (int i = 0; i < oldResults.length; i++) {
        final oldBbox = oldResults[i].boundingBox;
        final newBbox = newResults[i].boundingBox;

        expect((oldBbox.topLeft.x - newBbox.topLeft.x).abs(), lessThan(5));
        expect((oldBbox.topLeft.y - newBbox.topLeft.y).abs(), lessThan(5));
      }

      detector.dispose();
    });

    test(
        'detectFacesFromMat should produce consistent eye keypoints and iris centers',
        () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final List<Face> bytesResults = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      final List<Face> matResults = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.full,
      );
      mat.dispose();

      expect(matResults.length, bytesResults.length);
      expect(bytesResults, isNotEmpty,
          reason: 'Should detect at least one face');

      for (int i = 0; i < bytesResults.length; i++) {
        final bytesLandmarks = bytesResults[i].landmarks;
        final matLandmarks = matResults[i].landmarks;

        final bytesLeftEye = bytesLandmarks.leftEye;
        final matLeftEye = matLandmarks.leftEye;
        expect(bytesLeftEye, isNotNull,
            reason: 'Bytes API should have left eye');
        expect(matLeftEye, isNotNull, reason: 'Mat API should have left eye');

        const double tolerance = 5.0;
        expect(
          (bytesLeftEye!.x - matLeftEye!.x).abs(),
          lessThan(tolerance),
          reason: 'Left eye X should match within $tolerance pixels',
        );
        expect(
          (bytesLeftEye.y - matLeftEye.y).abs(),
          lessThan(tolerance),
          reason: 'Left eye Y should match within $tolerance pixels',
        );

        final bytesRightEye = bytesLandmarks.rightEye;
        final matRightEye = matLandmarks.rightEye;
        expect(bytesRightEye, isNotNull,
            reason: 'Bytes API should have right eye');
        expect(matRightEye, isNotNull, reason: 'Mat API should have right eye');

        expect(
          (bytesRightEye!.x - matRightEye!.x).abs(),
          lessThan(tolerance),
          reason: 'Right eye X should match within $tolerance pixels',
        );
        expect(
          (bytesRightEye.y - matRightEye.y).abs(),
          lessThan(tolerance),
          reason: 'Right eye Y should match within $tolerance pixels',
        );

        expect(
          bytesResults[i].irisPoints,
          isNotEmpty,
          reason: 'Bytes API should have iris points in full mode',
        );
        expect(
          matResults[i].irisPoints,
          isNotEmpty,
          reason: 'Mat API should have iris points in full mode',
        );
      }

      detector.dispose();
    });

    test('should throw StateError when detectFacesFromMat called before init',
        () async {
      final detector = FaceDetector();
      final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);

      expect(
        () => detector.detectFacesFromMat(mat),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );

      mat.dispose();
    });

    test('should properly dispose cv.Mat after detection', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      for (int i = 0; i < 5; i++) {
        final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
        await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);
        mat.dispose();
      }

      expect(true, isTrue);

      detector.dispose();
    });

    test('should work with different detection modes using cv.Mat', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final fastResults = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.fast,
      );
      expect(fastResults, isNotEmpty);
      expect(fastResults.first.mesh, isNull);

      final standardResults = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.standard,
      );
      expect(standardResults, isNotEmpty);
      expect(standardResults.first.mesh, isNotNull);
      expect(standardResults.first.irisPoints, isEmpty);

      final fullResults = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.full,
      );
      expect(fullResults, isNotEmpty);
      expect(fullResults.first.mesh, isNotNull);
      expect(fullResults.first.irisPoints, isNotEmpty);

      mat.dispose();
      detector.dispose();
    });
  });

  group('FaceDetectorIsolate - Background Isolate Detection', () {
    test('spawn creates initialized instance', () async {
      final detector = await FaceDetectorIsolate.spawn();

      expect(detector.isReady, true);

      await detector.dispose();
    });

    test('isReady returns false after dispose', () async {
      final detector = await FaceDetectorIsolate.spawn();
      expect(detector.isReady, true);

      await detector.dispose();

      expect(detector.isReady, false);
    });

    test('detectFaces returns empty list for image with no faces', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final bytes = TestUtils.createDummyImageBytes();
      final faces = await detector.detectFaces(bytes);

      expect(faces, isEmpty);

      await detector.dispose();
    });

    test('detectFaces throws after dispose', () async {
      final detector = await FaceDetectorIsolate.spawn();
      await detector.dispose();

      final bytes = TestUtils.createDummyImageBytes();

      expect(
        () => detector.detectFaces(bytes),
        throwsA(isA<StateError>()),
      );
    });

    test('dispose can be called multiple times safely', () async {
      final detector = await FaceDetectorIsolate.spawn();

      await detector.dispose();
      await detector.dispose();

      expect(detector.isReady, false);
    });

    test('spawn with custom configuration', () async {
      final detector = await FaceDetectorIsolate.spawn(
        model: FaceDetectionModel.frontCamera,
        performanceConfig: PerformanceConfig.xnnpack(numThreads: 2),
        meshPoolSize: 2,
      );

      expect(detector.isReady, true);

      await detector.dispose();
    });

    test('detectFaces with real image returns valid Face objects', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty);

      for (final face in faces) {
        expect(face.boundingBox, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.boundingBox.height, greaterThan(0));

        expect(face.landmarks.leftEye, isNotNull);
        expect(face.landmarks.rightEye, isNotNull);
        expect(face.landmarks.noseTip, isNotNull);

        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);

        expect(face.irisPoints, isNotEmpty);
      }

      await detector.dispose();
    });

    test('detectFaces respects detection mode', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final fastFaces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(fastFaces, isNotEmpty);
      expect(fastFaces.first.mesh, isNull);
      expect(fastFaces.first.irisPoints, isEmpty);

      final standardFaces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.standard);
      expect(standardFaces, isNotEmpty);
      expect(standardFaces.first.mesh, isNotNull);
      expect(standardFaces.first.irisPoints, isEmpty);

      final fullFaces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
      expect(fullFaces, isNotEmpty);
      expect(fullFaces.first.mesh, isNotNull);
      expect(fullFaces.first.irisPoints, isNotEmpty);

      await detector.dispose();
    });

    test('multiple concurrent detectFaces calls work correctly', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final futures = [
        detector.detectFaces(bytes, mode: FaceDetectionMode.fast),
        detector.detectFaces(bytes, mode: FaceDetectionMode.fast),
        detector.detectFaces(bytes, mode: FaceDetectionMode.fast),
      ];

      final results = await Future.wait(futures);

      expect(results.length, 3);
      for (final result in results) {
        expect(result, isA<List<Face>>());
      }
      final nonEmptyCount = results.where((r) => r.isNotEmpty).length;
      expect(nonEmptyCount, greaterThan(0),
          reason: 'At least one concurrent call should detect faces');

      await detector.dispose();
    });

    test('FaceDetectorIsolate produces same results as FaceDetector', () async {
      final isolateDetector = await FaceDetectorIsolate.spawn();
      final regularDetector = FaceDetector();
      await regularDetector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final isolateFaces = await isolateDetector.detectFaces(bytes,
          mode: FaceDetectionMode.full);
      final regularFaces = await regularDetector.detectFaces(bytes,
          mode: FaceDetectionMode.full);

      expect(isolateFaces.length, regularFaces.length);

      for (int i = 0; i < isolateFaces.length; i++) {
        final isolateBbox = isolateFaces[i].boundingBox;
        final regularBbox = regularFaces[i].boundingBox;

        expect(
            (isolateBbox.topLeft.x - regularBbox.topLeft.x).abs(), lessThan(1));
        expect(
            (isolateBbox.topLeft.y - regularBbox.topLeft.y).abs(), lessThan(1));
        expect((isolateBbox.width - regularBbox.width).abs(), lessThan(1));
        expect((isolateBbox.height - regularBbox.height).abs(), lessThan(1));
      }

      await isolateDetector.dispose();
      regularDetector.dispose();
    });

    test('detectFacesFromMat works with cv.Mat input', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty);
      expect(faces.first.boundingBox, isNotNull);
      expect(faces.first.mesh, isNotNull);
      expect(faces.first.irisPoints, isNotEmpty);

      mat.dispose();
      await detector.dispose();
    });

    test(
        'detectFacesFromMat produces same results as detectFacesFromMat on FaceDetector',
        () async {
      final isolateDetector = await FaceDetectorIsolate.spawn();
      final regularDetector = FaceDetector();
      await regularDetector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final isolateFaces = await isolateDetector.detectFacesFromMat(mat,
          mode: FaceDetectionMode.full);
      final regularFaces = await regularDetector.detectFacesFromMat(mat,
          mode: FaceDetectionMode.full);

      expect(isolateFaces.length, regularFaces.length);

      for (int i = 0; i < isolateFaces.length; i++) {
        final isolateBbox = isolateFaces[i].boundingBox;
        final regularBbox = regularFaces[i].boundingBox;

        expect(
            (isolateBbox.topLeft.x - regularBbox.topLeft.x).abs(), lessThan(1));
        expect(
            (isolateBbox.topLeft.y - regularBbox.topLeft.y).abs(), lessThan(1));
      }

      mat.dispose();
      await isolateDetector.dispose();
      regularDetector.dispose();
    });

    test('detectFacesFromMatBytes works with raw BGR bytes', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      final bgrBytes = mat.data;
      final width = mat.cols;
      final height = mat.rows;

      final faces = await detector.detectFacesFromMatBytes(
        bgrBytes,
        width: width,
        height: height,
        mode: FaceDetectionMode.full,
      );

      expect(faces, isNotEmpty);
      expect(faces.first.boundingBox, isNotNull);
      expect(faces.first.mesh, isNotNull);

      mat.dispose();
      await detector.dispose();
    });

    test('detectFacesFromMat respects detection mode', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final fastFaces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);
      expect(fastFaces, isNotEmpty);
      expect(fastFaces.first.mesh, isNull);
      expect(fastFaces.first.irisPoints, isEmpty);

      final standardFaces = await detector.detectFacesFromMat(mat,
          mode: FaceDetectionMode.standard);
      expect(standardFaces, isNotEmpty);
      expect(standardFaces.first.mesh, isNotNull);
      expect(standardFaces.first.irisPoints, isEmpty);

      final fullFaces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);
      expect(fullFaces, isNotEmpty);
      expect(fullFaces.first.mesh, isNotNull);
      expect(fullFaces.first.irisPoints, isNotEmpty);

      mat.dispose();
      await detector.dispose();
    });

    test('detectFacesFromMat throws after dispose', () async {
      final detector = await FaceDetectorIsolate.spawn();
      await detector.dispose();

      final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);

      expect(
        () => detector.detectFacesFromMat(mat),
        throwsA(isA<StateError>()),
      );

      mat.dispose();
    });
  });

  group('FaceDetector - Face Embedding', () {
    test('should generate face embedding from detected face', () async {
      final detector = FaceDetector();
      await detector.initialize();

      expect(detector.isEmbeddingReady, true);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty);

      final embedding = await detector.getFaceEmbedding(faces.first, bytes);

      expect(embedding, isNotNull);
      expect(embedding.length, greaterThan(0));
      print('Embedding dimension: ${embedding.length}');

      double norm = 0.0;
      for (final v in embedding) {
        norm += v * v;
      }
      norm = sqrt(norm);
      expect(norm, closeTo(1.0, 0.01),
          reason: 'Embedding should be L2-normalized');

      detector.dispose();
    });

    test('should generate batch embeddings for multiple faces', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces.length, greaterThan(1));
      print('Detected ${faces.length} faces in group photo');

      final embeddings = await detector.getFaceEmbeddings(faces, bytes);

      expect(embeddings.length, faces.length);

      int validCount = 0;
      for (final emb in embeddings) {
        if (emb != null) {
          validCount++;
          expect(emb.length, greaterThan(0));
        }
      }
      print('Generated $validCount valid embeddings out of ${faces.length}');
      expect(validCount, greaterThan(0));

      detector.dispose();
    });

    test('same face should have high similarity across images', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data1 =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final ByteData data2 =
          await rootBundle.load('assets/samples/iris-detection-ex2.jpg');

      final bytes1 = data1.buffer.asUint8List();
      final bytes2 = data2.buffer.asUint8List();

      final faces1 =
          await detector.detectFaces(bytes1, mode: FaceDetectionMode.fast);
      final faces2 =
          await detector.detectFaces(bytes2, mode: FaceDetectionMode.fast);

      expect(faces1, isNotEmpty);
      expect(faces2, isNotEmpty);

      final emb1 = await detector.getFaceEmbedding(faces1.first, bytes1);
      final emb2 = await detector.getFaceEmbedding(faces2.first, bytes2);

      final similarity = FaceDetector.compareFaces(emb1, emb2);
      print(
          'Same person similarity (iris-detection-ex1 vs ex2): ${similarity.toStringAsFixed(3)}');

      expect(similarity, greaterThan(0.0));

      detector.dispose();
    });

    test('different faces should have lower similarity', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces.length, greaterThanOrEqualTo(2));

      final embeddings = await detector.getFaceEmbeddings(faces, bytes);

      final validEmbeddings =
          embeddings.where((e) => e != null).take(2).toList();
      expect(validEmbeddings.length, 2);

      final similarity =
          FaceDetector.compareFaces(validEmbeddings[0]!, validEmbeddings[1]!);
      print('Different people similarity: ${similarity.toStringAsFixed(3)}');

      expect(similarity, isNotNull);

      detector.dispose();
    });

    test('compareFaces and faceDistance should be consistent', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty);

      final emb = await detector.getFaceEmbedding(faces.first, bytes);

      final selfSimilarity = FaceDetector.compareFaces(emb, emb);
      final selfDistance = FaceDetector.faceDistance(emb, emb);

      expect(selfSimilarity, closeTo(1.0, 0.001));
      expect(selfDistance, closeTo(0.0, 0.001));

      detector.dispose();
    });

    test('should throw StateError when embedding called before initialize',
        () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty);
      final face = faces.first;

      detector.dispose();

      final uninitDetector = FaceDetector();

      expect(
        () => uninitDetector.getFaceEmbedding(face, bytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    test('getFaceEmbeddingFromMat should work with cv.Mat', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final bytes = data.buffer.asUint8List();

      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty);

      final embedding =
          await detector.getFaceEmbeddingFromMat(faces.first, mat);

      expect(embedding, isNotNull);
      expect(embedding.length, greaterThan(0));

      mat.dispose();
      detector.dispose();
    });
  });

  group('FaceDetectorIsolate - Face Embedding', () {
    test('should generate embedding in background isolate', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty);

      final embedding = await detector.getFaceEmbedding(faces.first, bytes);

      expect(embedding, isNotNull);
      expect(embedding.length, greaterThan(0));

      double norm = 0.0;
      for (final v in embedding) {
        norm += v * v;
      }
      norm = sqrt(norm);
      expect(norm, closeTo(1.0, 0.01));

      await detector.dispose();
    });

    test('should generate batch embeddings in background isolate', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces.length, greaterThan(1));

      final embeddings = await detector.getFaceEmbeddings(faces, bytes);

      expect(embeddings.length, faces.length);

      int validCount = 0;
      for (final emb in embeddings) {
        if (emb != null) {
          validCount++;
        }
      }
      expect(validCount, greaterThan(0));
      print(
          'Isolate generated $validCount embeddings out of ${faces.length} faces');

      await detector.dispose();
    });

    test('FaceDetectorIsolate embeddings should match FaceDetector embeddings',
        () async {
      final isolateDetector = await FaceDetectorIsolate.spawn();
      final regularDetector = FaceDetector();
      await regularDetector.initialize();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final bytes = data.buffer.asUint8List();

      final isolateFaces = await isolateDetector.detectFaces(bytes,
          mode: FaceDetectionMode.fast);
      final regularFaces = await regularDetector.detectFaces(bytes,
          mode: FaceDetectionMode.fast);

      expect(isolateFaces.length, regularFaces.length);
      expect(isolateFaces, isNotEmpty);

      final isolateEmb =
          await isolateDetector.getFaceEmbedding(isolateFaces.first, bytes);
      final regularEmb =
          await regularDetector.getFaceEmbedding(regularFaces.first, bytes);

      expect(isolateEmb.length, regularEmb.length);

      final similarity = FaceDetector.compareFaces(isolateEmb, regularEmb);
      print(
          'Isolate vs Regular embedding similarity: ${similarity.toStringAsFixed(4)}');

      expect(similarity, greaterThan(0.99));

      await isolateDetector.dispose();
      regularDetector.dispose();
    });

    test('getFaceEmbedding throws after dispose', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty);
      final face = faces.first;

      await detector.dispose();

      expect(
        () => detector.getFaceEmbedding(face, bytes),
        throwsA(isA<StateError>()),
      );
    });
  });

  group('Face Recognition - Find Matching Face', () {
    test('should identify same person across images', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData refData =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final refBytes = refData.buffer.asUint8List();

      final ByteData groupData = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final groupBytes = groupData.buffer.asUint8List();

      final refFaces =
          await detector.detectFaces(refBytes, mode: FaceDetectionMode.fast);
      expect(refFaces, isNotEmpty);
      final refEmbedding =
          await detector.getFaceEmbedding(refFaces.first, refBytes);
      print('Reference embedding generated (${refEmbedding.length} dims)');

      final groupFaces =
          await detector.detectFaces(groupBytes, mode: FaceDetectionMode.fast);
      expect(groupFaces.length, greaterThan(1));
      print('Found ${groupFaces.length} faces in group image');

      Face? bestMatch;
      double bestSimilarity = -1.0;
      int bestIndex = -1;

      for (int i = 0; i < groupFaces.length; i++) {
        try {
          final embedding =
              await detector.getFaceEmbedding(groupFaces[i], groupBytes);
          final similarity = FaceDetector.compareFaces(refEmbedding, embedding);
          print(
              'Face $i: similarity = ${similarity.toStringAsFixed(3)}, bbox = ${groupFaces[i].boundingBox.center}');

          if (similarity > bestSimilarity) {
            bestSimilarity = similarity;
            bestMatch = groupFaces[i];
            bestIndex = i;
          }
        } catch (e) {
          print('Face $i: failed to get embedding - $e');
        }
      }

      expect(bestMatch, isNotNull);
      print(
          '\nBest match: Face $bestIndex with similarity ${bestSimilarity.toStringAsFixed(3)}');

      detector.dispose();
    });
  });

  group('Benchmark - FaceDetector vs FaceDetectorIsolate', () {
    test('compare detection latency across modes', () async {
      final regularDetector = FaceDetector();
      await regularDetector.initialize();
      final isolateDetector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      const int warmupRuns = 3;
      const int benchmarkRuns = 10;

      final results = <String, Map<String, double>>{};

      for (final mode in FaceDetectionMode.values) {
        for (int i = 0; i < warmupRuns; i++) {
          await regularDetector.detectFaces(bytes, mode: mode);
          await isolateDetector.detectFaces(bytes, mode: mode);
        }

        final regularTimes = <int>[];
        for (int i = 0; i < benchmarkRuns; i++) {
          final sw = Stopwatch()..start();
          await regularDetector.detectFaces(bytes, mode: mode);
          sw.stop();
          regularTimes.add(sw.elapsedMicroseconds);
        }

        final isolateTimes = <int>[];
        for (int i = 0; i < benchmarkRuns; i++) {
          final sw = Stopwatch()..start();
          await isolateDetector.detectFaces(bytes, mode: mode);
          sw.stop();
          isolateTimes.add(sw.elapsedMicroseconds);
        }

        final regularAvg =
            regularTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
        final isolateAvg =
            isolateTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
        final overhead = isolateAvg - regularAvg;
        final overheadPct = (overhead / regularAvg) * 100;

        results[mode.name] = {
          'regular_ms': regularAvg,
          'isolate_ms': isolateAvg,
          'overhead_ms': overhead,
          'overhead_pct': overheadPct,
        };
      }

      print('\n${'=' * 70}');
      print('BENCHMARK: FaceDetector vs FaceDetectorIsolate');
      print('=' * 70);
      print('Runs: $benchmarkRuns (after $warmupRuns warmup)');
      print('-' * 70);
      print('Mode'.padRight(12) +
          'Regular'.padLeft(12) +
          'Isolate'.padLeft(12) +
          'Overhead'.padLeft(12) +
          'Overhead %'.padLeft(12));
      print('-' * 70);

      for (final mode in FaceDetectionMode.values) {
        final r = results[mode.name]!;
        print(mode.name.padRight(12) +
            '${r['regular_ms']!.toStringAsFixed(2)} ms'.padLeft(12) +
            '${r['isolate_ms']!.toStringAsFixed(2)} ms'.padLeft(12) +
            '${r['overhead_ms']!.toStringAsFixed(2)} ms'.padLeft(12) +
            '${r['overhead_pct']!.toStringAsFixed(1)}%'.padLeft(12));
      }
      print('=' * 70);

      for (final mode in FaceDetectionMode.values) {
        final regularFaces =
            await regularDetector.detectFaces(bytes, mode: mode);
        final isolateFaces =
            await isolateDetector.detectFaces(bytes, mode: mode);
        expect(regularFaces.length, isolateFaces.length,
            reason: 'Face count should match for ${mode.name}');
      }

      regularDetector.dispose();
      await isolateDetector.dispose();
    });

    test('compare Mat-based detection latency', () async {
      final regularDetector = FaceDetector();
      await regularDetector.initialize();
      final isolateDetector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      const int warmupRuns = 3;
      const int benchmarkRuns = 10;
      const mode = FaceDetectionMode.full;

      for (int i = 0; i < warmupRuns; i++) {
        await regularDetector.detectFacesFromMat(mat, mode: mode);
        await isolateDetector.detectFacesFromMat(mat, mode: mode);
      }

      final regularTimes = <int>[];
      for (int i = 0; i < benchmarkRuns; i++) {
        final sw = Stopwatch()..start();
        await regularDetector.detectFacesFromMat(mat, mode: mode);
        sw.stop();
        regularTimes.add(sw.elapsedMicroseconds);
      }

      final isolateTimes = <int>[];
      for (int i = 0; i < benchmarkRuns; i++) {
        final sw = Stopwatch()..start();
        await isolateDetector.detectFacesFromMat(mat, mode: mode);
        sw.stop();
        isolateTimes.add(sw.elapsedMicroseconds);
      }

      final regularAvg =
          regularTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
      final isolateAvg =
          isolateTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
      final overhead = isolateAvg - regularAvg;

      print('\n${'=' * 70}');
      print('BENCHMARK: Mat-based Detection (full mode)');
      print('=' * 70);
      print(
          'FaceDetector.detectFacesFromMat:        ${regularAvg.toStringAsFixed(2)} ms');
      print(
          'FaceDetectorIsolate.detectFacesFromMat: ${isolateAvg.toStringAsFixed(2)} ms');
      print(
          'Overhead:                               ${overhead.toStringAsFixed(2)} ms');
      print('=' * 70);

      mat.dispose();
      regularDetector.dispose();
      await isolateDetector.dispose();
    });

    test('measure serialization overhead specifically', () async {
      final isolateDetector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces = await isolateDetector.detectFaces(bytes,
          mode: FaceDetectionMode.full);
      expect(faces, isNotEmpty);

      final face = faces.first;

      const int runs = 1000;
      final toMapSw = Stopwatch()..start();
      for (int i = 0; i < runs; i++) {
        face.toMap();
      }
      toMapSw.stop();

      final map = face.toMap();
      final fromMapSw = Stopwatch()..start();
      for (int i = 0; i < runs; i++) {
        Face.fromMap(map);
      }
      fromMapSw.stop();

      final toMapAvg = toMapSw.elapsedMicroseconds / runs;
      final fromMapAvg = fromMapSw.elapsedMicroseconds / runs;
      final totalPerFace = toMapAvg + fromMapAvg;

      print('\n${'=' * 70}');
      print('BENCHMARK: Face Serialization Overhead');
      print('=' * 70);
      print('Face.toMap():   ${toMapAvg.toStringAsFixed(1)} µs');
      print('Face.fromMap(): ${fromMapAvg.toStringAsFixed(1)} µs');
      print(
          'Total per face: ${totalPerFace.toStringAsFixed(1)} µs (${(totalPerFace / 1000).toStringAsFixed(3)} ms)');
      print('');
      print('Mesh points: ${face.mesh?.length ?? 0}');
      print('Iris points: ${face.irisPoints.length}');
      print('=' * 70);

      await isolateDetector.dispose();
    });
  });
}
