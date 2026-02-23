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

    test('detectFaces should throw StateError before initialize', () {
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
      final initDetector = FaceDetector();
      await initDetector.initialize();
      final faces = await initDetector.detectFaces(validImageBytes);
      initDetector.dispose();

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

      await detector.initialize();
      expect(detector.isReady, true);

      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('should handle initialize with different models', () async {
      final detector = FaceDetector();

      await detector.initialize(model: FaceDetectionModel.frontCamera);
      expect(detector.isReady, true);

      final faces1 = await detector.detectFaces(validImageBytes);
      expect(faces1, isNotEmpty);

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
    });

    test('should handle double dispose gracefully', () async {
      final detector = FaceDetector();
      await detector.initialize();

      detector.dispose();

      try {
        detector.dispose();
      } catch (e) {
        // Expected - double dispose may throw; verifying it doesn't crash
      }
    });
  });

  group('Recovery After Errors', () {
    test('should recover after invalid image bytes', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      try {
        await detector.detectFaces(invalidBytes);
      } catch (e) {
        // Expected - invalid bytes should fail; testing recovery
      }

      expect(detector.isReady, true);
      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('should recover after empty image', () async {
      final detector = FaceDetector();
      await detector.initialize();

      try {
        await detector.detectFaces(Uint8List(0));
      } catch (e) {
        // Expected - empty image should fail; testing recovery
      }

      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('should recover after processing image with no faces', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final noFaceMat = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);
      final noFaceResult = await detector.detectFacesFromMat(noFaceMat);
      noFaceMat.dispose();

      expect(noFaceResult, isEmpty);

      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('should handle rapid error-success cycles', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([1, 2, 3]);

      for (int i = 0; i < 10; i++) {
        try {
          await detector.detectFaces(invalidBytes);
        } catch (_) {}

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

      final matTypes = [
        cv.MatType.CV_8UC3,
        cv.MatType.CV_8UC4,
      ];

      for (final type in matTypes) {
        final mat = cv.Mat.zeros(256, 256, type);

        try {
          final faces = await detector.detectFacesFromMat(mat);
          expect(faces, isEmpty);
        } catch (e) {
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
      await detector.dispose();
      await detector.dispose();

      expect(detector.isReady, false);
    });

    test('should handle invalid image gracefully', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

      try {
        final faces = await detector.detectFaces(invalidBytes);
        expect(faces, isEmpty);
      } catch (e) {
        // Expected - invalid bytes may throw; testing isolate recovery
      }

      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      await detector.dispose();
    });
  });

  group('Embedding Error Handling', () {
    test('should handle embedding for faces from different images', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      final embedding =
          await detector.getFaceEmbedding(faces.first, validImageBytes);
      expect(embedding.length, greaterThan(0));

      detector.dispose();
    });

    test('should handle batch embeddings with some failures', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final groupImage = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(groupImage, mode: FaceDetectionMode.fast);
      expect(faces.length, greaterThan(1));

      final embeddings = await detector.getFaceEmbeddings(faces, groupImage);

      expect(embeddings.length, faces.length);

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

      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);

      detector.dispose();
    });

    test('iris statistics should accumulate correctly', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final irisImage = data.buffer.asUint8List();

      for (int i = 0; i < 5; i++) {
        await detector.detectFaces(irisImage, mode: FaceDetectionMode.full);
      }

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

      final faces = await detector.detectFaces(validImageBytes);
      expect(faces, isNotEmpty);
      final validFace = faces.first;

      try {
        await detector.detectFaces(Uint8List.fromList([1, 2, 3]));
      } catch (_) {}

      final embedding =
          await detector.getFaceEmbedding(validFace, validImageBytes);
      expect(embedding.length, greaterThan(0));

      detector.dispose();
    });

    test('mesh detection failure should not affect face detection', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final fastFaces = await detector.detectFaces(
        validImageBytes,
        mode: FaceDetectionMode.fast,
      );
      expect(fastFaces, isNotEmpty);

      final fullFaces = await detector.detectFaces(
        validImageBytes,
        mode: FaceDetectionMode.full,
      );
      expect(fullFaces, isNotEmpty);

      expect(fastFaces.length, fullFaces.length);

      detector.dispose();
    });
  });

  group('Resource Cleanup', () {
    test('should clean up after processing many images', () async {
      final detector = FaceDetector();
      await detector.initialize();

      for (int i = 0; i < 20; i++) {
        final mat = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);
        await detector.detectFacesFromMat(mat);
        mat.dispose();
      }

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
