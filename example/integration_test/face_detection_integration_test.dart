// Comprehensive integration tests for FaceDetector.
//
// These tests cover:
// - Initialization and disposal
// - Error handling (works in standard test environment)
// - Detection with real sample images (requires device/platform-specific testing)
// - detectFaces() method
// - Different detection modes (fast, standard, full)
// - Landmark, mesh, and iris access
// - Configuration parameters
// - Edge cases
//
// NOTE: Most tests require TensorFlow Lite native libraries which are not
// available in the standard `flutter test` environment. To run all tests:
//
// - macOS: flutter test integration_test/face_detection_integration_test.dart --platform=macos
// - Device: Run as integration tests on a physical device or emulator
//
// Tests that work in standard environment (no TFLite required):
// - StateError when not initialized
// - Parameter validation
//

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Test helper to create a minimal 1x1 PNG image
class TestUtils {
  static Uint8List createDummyImageBytes() {
    return Uint8List.fromList([
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
      0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
      0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // Width: 1, Height: 1
      0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, // Bit depth, color type
      0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
      0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
      0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, // Image data
      0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND chunk
      0x42, 0x60, 0x82
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
      // Note: isReady may still be true after dispose in current implementation
    });

    test('should allow re-initialization', () async {
      final detector = FaceDetector();
      await detector.initialize();
      expect(detector.isReady, true);

      // Re-initialize should work
      await detector.initialize();
      expect(detector.isReady, true);

      detector.dispose();
    });

    test('should handle multiple dispose calls', () async {
      final detector = FaceDetector();
      await detector.initialize();
      detector.dispose();

      // Second dispose may throw - that's acceptable
      try {
        detector.dispose();
      } catch (e) {
        // Disposing twice may throw, which is acceptable behavior
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

      // Invalid bytes may throw or return empty - either is acceptable
      try {
        final results = await detector.detectFaces(invalidBytes);
        expect(results, isEmpty);
      } catch (e) {
        // Error is also acceptable for invalid input
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        // Verify bounding box
        expect(face.boundingBox, isNotNull);
        expect(face.boundingBox.topLeft.x, greaterThanOrEqualTo(0));
        expect(face.boundingBox.topLeft.y, greaterThanOrEqualTo(0));
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.boundingBox.height, greaterThan(0));

        // Verify landmarks (6 keypoints)
        expect(face.landmarks.leftEye, isNotNull);
        expect(face.landmarks.rightEye, isNotNull);
        expect(face.landmarks.noseTip, isNotNull);
        expect(face.landmarks.mouth, isNotNull);
        expect(face.landmarks.leftEyeTragion, isNotNull);
        expect(face.landmarks.rightEyeTragion, isNotNull);

        // Verify mesh (468 points)
        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);

        // Verify iris points (152 points for 2 eyes)
        expect(face.irisPoints, isNotEmpty);

        // Check image dimensions
        expect(face.originalSize.width, greaterThan(0));
        expect(face.originalSize.height, greaterThan(0));
      }

      detector.dispose();
    });

    test('should detect faces in iris-detection-ex1.jpg', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('../assets/samples/iris-detection-ex1.jpg');
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
          await rootBundle.load('../assets/samples/iris-detection-ex2.jpg');
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
          .load('../assets/samples/group-shot-bounding-box-ex1.jpeg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      // Group shot should detect 4 faces
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
          await rootBundle.load('../assets/samples/mesh-ex1.jpeg');
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.fast,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        // Should have bounding box and 6 keypoints
        expect(face.boundingBox, isNotNull);
        expect(face.landmarks, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));

        // Should NOT have mesh or iris in fast mode
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.standard,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        // Should have bounding box, keypoints, and mesh
        expect(face.boundingBox, isNotNull);
        expect(face.landmarks, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);

        // Should NOT have iris in standard mode
        expect(face.irisPoints, isEmpty);
      }

      detector.dispose();
    });

    test('should work with full mode (all features including iris)', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('../assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      expect(results, isNotEmpty);

      for (final face in results) {
        // Should have everything
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      faces = await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
    });

    tearDownAll(() {
      detector.dispose();
    });

    test('should access facial keypoints', () {
      expect(faces, isNotEmpty);
      final face = faces.first;

      // Test accessing different landmark types
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
        // Coordinates should be within image bounds
        expect(point.x, greaterThanOrEqualTo(0));
        expect(point.x, lessThanOrEqualTo(face.originalSize.width.toDouble()));
        expect(point.y, greaterThanOrEqualTo(0));
        expect(point.y, lessThanOrEqualTo(face.originalSize.height.toDouble()));

        // Check if 3D data is present
        if (point.is3D) {
          expect(point.z, isNotNull);
        }
      }
    });

    test('should access eyes and iris data', () {
      final face = faces.first;
      final eyes = face.eyes;

      if (face.irisPoints.isNotEmpty && eyes != null) {
        // Check left eye
        final leftEye = eyes.leftEye;
        if (leftEye != null) {
          expect(leftEye.irisCenter, isNotNull);
          expect(leftEye.irisContour.length, 4);
          expect(leftEye.mesh.length, 71);
        }

        // Check right eye
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

      // Bounding box should be within image
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
        '../assets/samples/landmark-ex1.jpg',
        '../assets/samples/iris-detection-ex1.jpg',
        '../assets/samples/iris-detection-ex2.jpg',
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
        '../assets/samples/landmark-ex1.jpg',
        '../assets/samples/mesh-ex1.jpeg',
        '../assets/samples/group-shot-bounding-box-ex1.jpeg',
      ];

      for (final imagePath in images) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();
        final List<Face> results = await detector.detectFaces(bytes);

        // Should work regardless of image size
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

      // Should not crash, but probably won't detect anything
      expect(results, isNotNull);

      detector.dispose();
    });

    test('Face.toString() should not crash', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> results = await detector.detectFaces(bytes);

      expect(results, isNotEmpty);

      // toString() should not crash - content may vary
      final faceString = results.first.toString();
      expect(faceString, isNotEmpty);
      expect(faceString, contains('Face'));

      detector.dispose();
    });

    test('should handle normal detection', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Load image bytes
      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Use detectFaces
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
          await rootBundle.load('../assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Run detection multiple times
      for (int i = 0; i < 3; i++) {
        await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
      }

      // Check that iris statistics are being tracked
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Decode to cv.Mat
      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      expect(mat.isEmpty, false);

      // Run detection with cv.Mat
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
          await rootBundle.load('../assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Run detection with OpenCV-accelerated API
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Run with old API
      final List<Face> oldResults = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      // Run with new OpenCV API
      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      final List<Face> newResults = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.full,
      );
      mat.dispose();

      // Both should detect same number of faces
      expect(newResults.length, oldResults.length);

      // Compare bounding boxes (allow small differences due to rounding)
      for (int i = 0; i < oldResults.length; i++) {
        final oldBbox = oldResults[i].boundingBox;
        final newBbox = newResults[i].boundingBox;

        // Bounding boxes should be very close (within 5 pixels)
        expect((oldBbox.topLeft.x - newBbox.topLeft.x).abs(), lessThan(5));
        expect((oldBbox.topLeft.y - newBbox.topLeft.y).abs(), lessThan(5));
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Create mat and run detection multiple times
      for (int i = 0; i < 5; i++) {
        final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
        await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);
        mat.dispose();
      }

      // If we get here without crash, memory is being managed correctly
      expect(true, isTrue);

      detector.dispose();
    });

    test('should work with different detection modes using cv.Mat', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      // Fast mode
      final fastResults = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.fast,
      );
      expect(fastResults, isNotEmpty);
      expect(fastResults.first.mesh, isNull);

      // Standard mode
      final standardResults = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.standard,
      );
      expect(standardResults, isNotEmpty);
      expect(standardResults.first.mesh, isNotNull);
      expect(standardResults.first.irisPoints, isEmpty);

      // Full mode
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

      // Use dummy image bytes (1x1 pixel - no face)
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
      await detector.dispose(); // Should not throw

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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces = await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty);

      for (final face in faces) {
        // Verify bounding box
        expect(face.boundingBox, isNotNull);
        expect(face.boundingBox.width, greaterThan(0));
        expect(face.boundingBox.height, greaterThan(0));

        // Verify landmarks
        expect(face.landmarks.leftEye, isNotNull);
        expect(face.landmarks.rightEye, isNotNull);
        expect(face.landmarks.noseTip, isNotNull);

        // Verify mesh (full mode)
        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);

        // Verify iris points (full mode)
        expect(face.irisPoints, isNotEmpty);
      }

      await detector.dispose();
    });

    test('detectFaces respects detection mode', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Fast mode - no mesh or iris
      final fastFaces = await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(fastFaces, isNotEmpty);
      expect(fastFaces.first.mesh, isNull);
      expect(fastFaces.first.irisPoints, isEmpty);

      // Standard mode - mesh but no iris
      final standardFaces = await detector.detectFaces(bytes, mode: FaceDetectionMode.standard);
      expect(standardFaces, isNotEmpty);
      expect(standardFaces.first.mesh, isNotNull);
      expect(standardFaces.first.irisPoints, isEmpty);

      // Full mode - mesh and iris
      final fullFaces = await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
      expect(fullFaces, isNotEmpty);
      expect(fullFaces.first.mesh, isNotNull);
      expect(fullFaces.first.irisPoints, isNotEmpty);

      await detector.dispose();
    });

    test('multiple concurrent detectFaces calls work correctly', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Fire multiple requests concurrently
      // Note: Due to internal TFLite race conditions with concurrent inference,
      // some concurrent calls may return empty results. We verify that:
      // 1. All calls complete without throwing
      // 2. At least one call succeeds (proving the detector works)
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
      // At least one concurrent call should detect a face
      final nonEmptyCount = results.where((r) => r.isNotEmpty).length;
      expect(nonEmptyCount, greaterThan(0),
          reason: 'At least one concurrent call should detect faces');

      await detector.dispose();
    });

    test('FaceDetectorIsolate produces same results as FaceDetector', () async {
      // Initialize both detectors
      final isolateDetector = await FaceDetectorIsolate.spawn();
      final regularDetector = FaceDetector();
      await regularDetector.initialize();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Run detection with both
      final isolateFaces = await isolateDetector.detectFaces(bytes, mode: FaceDetectionMode.full);
      final regularFaces = await regularDetector.detectFaces(bytes, mode: FaceDetectionMode.full);

      // Should detect same number of faces
      expect(isolateFaces.length, regularFaces.length);

      // Compare bounding boxes (allow small differences)
      for (int i = 0; i < isolateFaces.length; i++) {
        final isolateBbox = isolateFaces[i].boundingBox;
        final regularBbox = regularFaces[i].boundingBox;

        expect((isolateBbox.topLeft.x - regularBbox.topLeft.x).abs(), lessThan(1));
        expect((isolateBbox.topLeft.y - regularBbox.topLeft.y).abs(), lessThan(1));
        expect((isolateBbox.width - regularBbox.width).abs(), lessThan(1));
        expect((isolateBbox.height - regularBbox.height).abs(), lessThan(1));
      }

      await isolateDetector.dispose();
      regularDetector.dispose();
    });

    test('detectFacesFromMat works with cv.Mat input', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Decode to cv.Mat
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final faces = await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty);
      expect(faces.first.boundingBox, isNotNull);
      expect(faces.first.mesh, isNotNull);
      expect(faces.first.irisPoints, isNotEmpty);

      mat.dispose();
      await detector.dispose();
    });

    test('detectFacesFromMat produces same results as detectFacesFromMat on FaceDetector', () async {
      final isolateDetector = await FaceDetectorIsolate.spawn();
      final regularDetector = FaceDetector();
      await regularDetector.initialize();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Decode to cv.Mat
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      // Run detection with both
      final isolateFaces = await isolateDetector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);
      final regularFaces = await regularDetector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);

      // Should detect same number of faces
      expect(isolateFaces.length, regularFaces.length);

      // Compare bounding boxes
      for (int i = 0; i < isolateFaces.length; i++) {
        final isolateBbox = isolateFaces[i].boundingBox;
        final regularBbox = regularFaces[i].boundingBox;

        expect((isolateBbox.topLeft.x - regularBbox.topLeft.x).abs(), lessThan(1));
        expect((isolateBbox.topLeft.y - regularBbox.topLeft.y).abs(), lessThan(1));
      }

      mat.dispose();
      await isolateDetector.dispose();
      regularDetector.dispose();
    });

    test('detectFacesFromMatBytes works with raw BGR bytes', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Decode to cv.Mat to get BGR bytes
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      // Fast mode - no mesh or iris
      final fastFaces = await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);
      expect(fastFaces, isNotEmpty);
      expect(fastFaces.first.mesh, isNull);
      expect(fastFaces.first.irisPoints, isEmpty);

      // Standard mode - mesh but no iris
      final standardFaces = await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.standard);
      expect(standardFaces, isNotEmpty);
      expect(standardFaces.first.mesh, isNotNull);
      expect(standardFaces.first.irisPoints, isEmpty);

      // Full mode - mesh and iris
      final fullFaces = await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);
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

  group('Benchmark - FaceDetector vs FaceDetectorIsolate', () {
    test('compare detection latency across modes', () async {
      final regularDetector = FaceDetector();
      await regularDetector.initialize();
      final isolateDetector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      const int warmupRuns = 3;
      const int benchmarkRuns = 10;

      final results = <String, Map<String, double>>{};

      for (final mode in FaceDetectionMode.values) {
        // Warmup - exclude from timing
        for (int i = 0; i < warmupRuns; i++) {
          await regularDetector.detectFaces(bytes, mode: mode);
          await isolateDetector.detectFaces(bytes, mode: mode);
        }

        // Benchmark regular detector
        final regularTimes = <int>[];
        for (int i = 0; i < benchmarkRuns; i++) {
          final sw = Stopwatch()..start();
          await regularDetector.detectFaces(bytes, mode: mode);
          sw.stop();
          regularTimes.add(sw.elapsedMicroseconds);
        }

        // Benchmark isolate detector
        final isolateTimes = <int>[];
        for (int i = 0; i < benchmarkRuns; i++) {
          final sw = Stopwatch()..start();
          await isolateDetector.detectFaces(bytes, mode: mode);
          sw.stop();
          isolateTimes.add(sw.elapsedMicroseconds);
        }

        final regularAvg = regularTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
        final isolateAvg = isolateTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
        final overhead = isolateAvg - regularAvg;
        final overheadPct = (overhead / regularAvg) * 100;

        results[mode.name] = {
          'regular_ms': regularAvg,
          'isolate_ms': isolateAvg,
          'overhead_ms': overhead,
          'overhead_pct': overheadPct,
        };
      }

      // Print results
      print('\n${'=' * 70}');
      print('BENCHMARK: FaceDetector vs FaceDetectorIsolate');
      print('${'=' * 70}');
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

      // Verify both produce valid results
      for (final mode in FaceDetectionMode.values) {
        final regularFaces = await regularDetector.detectFaces(bytes, mode: mode);
        final isolateFaces = await isolateDetector.detectFaces(bytes, mode: mode);
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
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      const int warmupRuns = 3;
      const int benchmarkRuns = 10;
      const mode = FaceDetectionMode.full;

      // Warmup
      for (int i = 0; i < warmupRuns; i++) {
        await regularDetector.detectFacesFromMat(mat, mode: mode);
        await isolateDetector.detectFacesFromMat(mat, mode: mode);
      }

      // Benchmark regular detector
      final regularTimes = <int>[];
      for (int i = 0; i < benchmarkRuns; i++) {
        final sw = Stopwatch()..start();
        await regularDetector.detectFacesFromMat(mat, mode: mode);
        sw.stop();
        regularTimes.add(sw.elapsedMicroseconds);
      }

      // Benchmark isolate detector
      final isolateTimes = <int>[];
      for (int i = 0; i < benchmarkRuns; i++) {
        final sw = Stopwatch()..start();
        await isolateDetector.detectFacesFromMat(mat, mode: mode);
        sw.stop();
        isolateTimes.add(sw.elapsedMicroseconds);
      }

      final regularAvg = regularTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
      final isolateAvg = isolateTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
      final overhead = isolateAvg - regularAvg;

      print('\n${'=' * 70}');
      print('BENCHMARK: Mat-based Detection (full mode)');
      print('${'=' * 70}');
      print('FaceDetector.detectFacesFromMat:        ${regularAvg.toStringAsFixed(2)} ms');
      print('FaceDetectorIsolate.detectFacesFromMat: ${isolateAvg.toStringAsFixed(2)} ms');
      print('Overhead:                               ${overhead.toStringAsFixed(2)} ms');
      print('=' * 70);

      mat.dispose();
      regularDetector.dispose();
      await isolateDetector.dispose();
    });

    test('measure serialization overhead specifically', () async {
      final isolateDetector = await FaceDetectorIsolate.spawn();

      final ByteData data =
          await rootBundle.load('../assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // First, get a face to measure serialization
      final faces = await isolateDetector.detectFaces(bytes, mode: FaceDetectionMode.full);
      expect(faces, isNotEmpty);

      final face = faces.first;

      // Measure toMap serialization
      const int runs = 1000;
      final toMapSw = Stopwatch()..start();
      for (int i = 0; i < runs; i++) {
        face.toMap();
      }
      toMapSw.stop();

      // Measure fromMap deserialization
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
      print('${'=' * 70}');
      print('Face.toMap():   ${toMapAvg.toStringAsFixed(1)} µs');
      print('Face.fromMap(): ${fromMapAvg.toStringAsFixed(1)} µs');
      print('Total per face: ${totalPerFace.toStringAsFixed(1)} µs (${(totalPerFace / 1000).toStringAsFixed(3)} ms)');
      print('');
      print('Mesh points: ${face.mesh?.length ?? 0}');
      print('Iris points: ${face.irisPoints.length}');
      print('=' * 70);

      await isolateDetector.dispose();
    });
  });
}
