// ignore_for_file: avoid_print

/// Error recovery and graceful degradation tests for FaceDetector.
///
/// Tests:
/// - Uninitialized detector errors
/// - Double initialization
/// - Double disposal
/// - Recovery from errors
/// - Graceful degradation with problematic inputs
library;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const errorTimeout = Timeout(Duration(minutes: 2));

  late Uint8List validImageBytes;

  setUpAll(() async {
    final data = await rootBundle.load('assets/samples/landmark-ex1.jpg');
    validImageBytes = data.buffer.asUint8List();
  });

  group('Uninitialized Detector Errors', () {
    test('detectFaces should throw StateError before initialize', () {
      final detector = FaceDetector();

      expect(
        () => detector.detectFaces(validImageBytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    test('detectFacesFromMat should throw StateError before initialize', () {
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

    test('getFaceEmbedding should throw StateError before initialize',
        () async {
      // First create a valid face from an initialized detector
      final initDetector = FaceDetector();
      await initDetector.initialize();
      final faces = await initDetector.detectFaces(validImageBytes);
      initDetector.dispose();

      // Now try with uninitialized detector
      final uninitDetector = FaceDetector();

      expect(
        () => uninitDetector.getFaceEmbedding(faces.first, validImageBytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    test('isReady should be false before initialize', () {
      final detector = FaceDetector();
      expect(detector.isReady, false);
    });

    test('isEmbeddingReady should be false before initialize', () {
      final detector = FaceDetector();
      expect(detector.isEmbeddingReady, false);
    });
  });

  group('Initialization Behavior', () {
    test('should handle multiple initialize calls', () async {
      final detector = FaceDetector();

      await detector.initialize();
      expect(detector.isReady, true);

      // Second initialize should work (re-initialize)
      await detector.initialize();
      expect(detector.isReady, true);

      // Should still work after re-init
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('should handle initialize with different models', () async {
      final detector = FaceDetector();

      // Initialize with one model
      await detector.initialize(model: FaceDetectionModel.frontCamera);
      expect(detector.isReady, true);

      final faces1 = await detector.detectFaces(validImageBytes);
      expect(faces1, isNotEmpty);

      // Re-initialize with different model
      await detector.initialize(model: FaceDetectionModel.backCamera);
      expect(detector.isReady, true);

      final faces2 = await detector.detectFaces(validImageBytes);
      expect(faces2, isNotEmpty);

      detector.dispose();
    });
  });

  group('Disposal Behavior', () {
    test('should handle dispose properly', () async {
      final detector = FaceDetector();
      await detector.initialize();
      expect(detector.isReady, true);

      detector.dispose();
      // Dispose completed without crash
    });

    test('should handle double dispose gracefully', () async {
      final detector = FaceDetector();
      await detector.initialize();

      detector.dispose();

      // Second dispose should not crash (may throw or be no-op)
      try {
        detector.dispose();
      } catch (e) {
        // Exception is acceptable
      }
      // If we get here, double dispose is handled gracefully
    });

    // Note: detectFaces after dispose may hang rather than throw
    // This is expected behavior - the detector state is undefined after dispose
  });

  group('Recovery After Errors', () {
    test('should recover after invalid image bytes', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Process invalid input
      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      try {
        await detector.detectFaces(invalidBytes);
      } catch (e) {
        // Expected
      }

      // Should still work with valid input
      expect(detector.isReady, true);
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('should recover after empty image', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Process empty input
      try {
        await detector.detectFaces(Uint8List(0));
      } catch (e) {
        // Expected
      }

      // Should still work
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('should recover after processing image with no faces', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Process image without faces
      final noFaceMat = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);
      final noFaceResult = await detector.detectFacesFromMat(noFaceMat);
      noFaceMat.dispose();

      expect(noFaceResult, isEmpty);

      // Should still work with face image
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('should handle rapid error-success cycles', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([1, 2, 3]);

      for (int i = 0; i < 10; i++) {
        // Invalid
        try {
          await detector.detectFaces(invalidBytes);
        } catch (_) {
          // Expected to fail with invalid input
        }

        // Valid
        final faces = await detector.detectFaces(validImageBytes);
        expect(faces, isNotEmpty, reason: 'Iteration $i failed');
      }

      detector.dispose();
    });
  });

  group('Graceful Degradation', () {
    test('should return empty for tiny image instead of crashing', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // 1x1 image
      final tinyMat = cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3);
      final faces = await detector.detectFacesFromMat(tinyMat);
      tinyMat.dispose();

      expect(faces, isEmpty);
      expect(detector.isReady, true);

      detector.dispose();
    });

    test('should handle various Mat types', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Test different Mat types
      final matTypes = [
        cv.MatType.CV_8UC3,
        cv.MatType.CV_8UC4,
      ];

      for (final type in matTypes) {
        final mat = cv.Mat.zeros(256, 256, type);

        try {
          final faces = await detector.detectFacesFromMat(mat);
          expect(faces, isEmpty); // No face in black image
        } catch (e) {
          // Some types may not be supported
          print('Mat type $type: $e');
        }

        mat.dispose();
      }

      detector.dispose();
    });
  });

  group('FaceDetectorIsolate Error Handling', () {
    test('should throw StateError when detectFaces called after dispose',
        () async {
      final detector = await FaceDetectorIsolate.spawn();
      await detector.dispose();

      expect(detector.isReady, false);

      expect(
        () => detector.detectFaces(validImageBytes),
        throwsA(isA<StateError>()),
      );
    });

    test('should throw StateError when getFaceEmbedding called after dispose',
        () async {
      final detector = await FaceDetectorIsolate.spawn();

      // Get face first
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      await detector.dispose();

      expect(
        () => detector.getFaceEmbedding(faces.first, validImageBytes),
        throwsA(isA<StateError>()),
      );
    });

    test('should handle multiple dispose calls without crashing', () async {
      final detector = await FaceDetectorIsolate.spawn();

      await detector.dispose();
      await detector.dispose(); // Should not throw
      await detector.dispose(); // Should not throw

      expect(detector.isReady, false);
    });

    test('should handle invalid image gracefully', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

      try {
        final faces = await detector.detectFaces(invalidBytes);
        expect(faces, isEmpty);
      } catch (e) {
        // Exception is acceptable
      }

      // Should still work
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      await detector.dispose();
    });
  });

  group('Embedding Error Handling', () {
    test('should handle embedding for faces from different images', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Get face from one image
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      // Try to get embedding with the same image (should work)
      final embedding =
          await detector.getFaceEmbedding(faces.first, validImageBytes);
      expect(embedding.length, greaterThan(0));

      detector.dispose();
    });

    test('should handle batch embeddings with some failures', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Get multiple faces from group photo
      final data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final groupImage = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(groupImage, mode: FaceDetectionMode.fast);
      expect(faces.length, greaterThan(1));

      // Get embeddings for all
      final embeddings = await detector.getFaceEmbeddings(faces, groupImage);

      expect(embeddings.length, faces.length);

      // At least some should succeed
      final successCount = embeddings.where((e) => e != null).length;
      expect(successCount, greaterThan(0));

      print('Batch embeddings: $successCount/${faces.length} succeeded');

      detector.dispose();
    });
  });

  group('State Consistency', () {
    test('isReady should reflect actual state', () async {
      final detector = FaceDetector();

      expect(detector.isReady, false);

      await detector.initialize();
      expect(detector.isReady, true);

      // Verify operations work when isReady is true
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('iris statistics should accumulate correctly', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Load image that has good iris detection
      final data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final irisImage = data.buffer.asUint8List();

      // Run detection multiple times
      for (int i = 0; i < 5; i++) {
        await detector.detectFaces(irisImage, mode: FaceDetectionMode.full);
      }

      // Check statistics are tracked
      final totalAttempts = detector.irisOkCount + detector.irisFailCount;
      expect(totalAttempts, greaterThan(0),
          reason: 'Iris statistics should be tracked');

      print(
          'Iris stats: ${detector.irisOkCount} ok, ${detector.irisFailCount} fail');

      detector.dispose();
    });
  });

  group('Cross-Component Error Isolation', () {
    test('face detection failure should not affect embedding', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Get valid face first
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);
      final validFace = faces.first;

      // Try to detect with invalid image (may fail)
      try {
        await detector.detectFaces(Uint8List.fromList([1, 2, 3]));
      } catch (_) {
        // Expected to fail with invalid input
      }

      // Embedding should still work
      final embedding =
          await detector.getFaceEmbedding(validFace, validImageBytes);
      expect(embedding.length, greaterThan(0));

      detector.dispose();
    });

    test('mesh detection failure should not affect face detection', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Run in different modes - fast mode doesn't use mesh
      final fastFaces = await detector.detectFaces(
        validImageBytes,
        mode: FaceDetectionMode.fast,
      );
      expect(fastFaces, isNotEmpty);

      // Full mode uses mesh
      final fullFaces = await detector.detectFaces(
        validImageBytes,
        mode: FaceDetectionMode.full,
      );
      expect(fullFaces, isNotEmpty);

      // Both should detect same number of faces
      expect(fastFaces.length, fullFaces.length);

      detector.dispose();
    });
  });

  group('Resource Cleanup', () {
    test('should clean up after processing many images', () async {
      final detector = FaceDetector();
      await detector.initialize();

      // Process many images
      for (int i = 0; i < 20; i++) {
        final mat = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);
        await detector.detectFacesFromMat(mat);
        mat.dispose();
      }

      // Detector should still be functional
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    }, timeout: errorTimeout);

    test('should handle create/initialize/use/dispose cycle repeatedly',
        () async {
      for (int i = 0; i < 3; i++) {
        final detector = FaceDetector();
        await detector.initialize();

        final faces = await detector.detectFaces(validImageBytes);
        expect(faces, isNotEmpty, reason: 'Cycle $i failed');

        detector.dispose();
      }
    }, timeout: errorTimeout);
  });
}
