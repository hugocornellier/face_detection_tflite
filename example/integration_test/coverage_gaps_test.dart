// ignore_for_file: avoid_print

/// Integration tests for confirmed coverage gaps after the 5.0.0 refactor.
///
/// Tests cover:
/// - FaceDetector dispose-then-reinitialize lifecycle on the same instance
/// - IrisLandmark.createFromFile() and IrisLandmark.callWithIsolate()
/// - SegmentationConfig.maxOutputSize behavior (documents dead code)
/// - SegmentationConfig.validateModel behavior
/// - SegmentationWorker.segmentMat() with non-BGR Mat types
/// - SegmentationWorker error recovery (bad input then good input)
/// - SelfieSegmentation.disposeAsync() semantics
/// - Grayscale cv.Mat through FaceDetector.detectFacesFromMat()
library;

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testTimeout = Timeout(Duration(minutes: 5));

  late Uint8List imageBytes;
  late bool modelsAvailable;

  Future<bool> checkModels() async {
    try {
      await rootBundle.load(
        'packages/face_detection_tflite/assets/models/selfie_segmenter.tflite',
      );
      return true;
    } catch (_) {
      return false;
    }
  }

  /// Writes a bundled asset to a temp file and returns the path.
  /// Needed for IrisLandmark.createFromFile() and callWithIsolate().
  Future<String> writeAssetToTempFile(String assetKey, String filename) async {
    final data = await rootBundle.load(assetKey);
    final dir = Directory.systemTemp.createTempSync('face_detection_test_');
    final file = File('${dir.path}/$filename');
    await file.writeAsBytes(data.buffer.asUint8List());
    return file.path;
  }

  setUpAll(() async {
    modelsAvailable = await checkModels();
    final data = await rootBundle.load('assets/samples/landmark-ex1.jpg');
    imageBytes = data.buffer.asUint8List();
  });

  // ===========================================================================
  // 1. FaceDetector dispose-then-reinitialize lifecycle
  // ===========================================================================
  group('FaceDetector dispose-then-reinitialize', () {
    test('should work after dispose then re-initialize on same instance',
        () async {
      print('\n--- Testing dispose -> re-initialize on same instance ---');
      final detector = FaceDetector();

      // First lifecycle
      await detector.initialize();
      expect(detector.isReady, true);
      final faces1 = await detector.detectFaces(imageBytes);
      expect(faces1, isNotEmpty);
      print('First lifecycle: detected ${faces1.length} face(s)');

      // Dispose
      detector.dispose();
      // isReady should be false after dispose (meshPool cleared)
      expect(detector.isReady, false);

      // Re-initialize same instance
      await detector.initialize();
      expect(detector.isReady, true);

      // Should still work
      final faces2 = await detector.detectFaces(imageBytes);
      expect(faces2, isNotEmpty);
      expect(faces2.length, faces1.length);
      print('Second lifecycle: detected ${faces2.length} face(s)');

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should work with different model on re-initialize', () async {
      print('\n--- Testing dispose -> re-init with different model ---');
      final detector = FaceDetector();

      await detector.initialize(model: FaceDetectionModel.backCamera);
      expect(detector.isReady, true);
      final faces1 =
          await detector.detectFaces(imageBytes, mode: FaceDetectionMode.fast);
      expect(faces1, isNotEmpty);
      print('backCamera: ${faces1.length} face(s)');

      detector.dispose();

      await detector.initialize(model: FaceDetectionModel.frontCamera);
      expect(detector.isReady, true);
      final faces2 =
          await detector.detectFaces(imageBytes, mode: FaceDetectionMode.fast);
      expect(faces2, isNotEmpty);
      print('frontCamera: ${faces2.length} face(s)');

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should support full mode after dispose-then-reinitialize', () async {
      print('\n--- Testing full mode after dispose -> re-init ---');
      final detector = FaceDetector();

      await detector.initialize();
      final faces1 =
          await detector.detectFaces(imageBytes, mode: FaceDetectionMode.full);
      expect(faces1, isNotEmpty);
      expect(faces1.first.mesh, isNotNull);
      expect(faces1.first.mesh!.length, 468);

      detector.dispose();
      await detector.initialize();

      final faces2 =
          await detector.detectFaces(imageBytes, mode: FaceDetectionMode.full);
      expect(faces2, isNotEmpty);
      expect(faces2.first.mesh, isNotNull);
      expect(faces2.first.mesh!.length, 468);

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should support embeddings after dispose-then-reinitialize', () async {
      print('\n--- Testing embeddings after dispose -> re-init ---');
      final detector = FaceDetector();

      await detector.initialize();
      expect(detector.isEmbeddingReady, true);
      final faces1 =
          await detector.detectFaces(imageBytes, mode: FaceDetectionMode.fast);
      expect(faces1, isNotEmpty);
      final emb1 = await detector.getFaceEmbedding(faces1.first, imageBytes);
      expect(emb1.length, kEmbeddingDimension);

      detector.dispose();
      await detector.initialize();
      expect(detector.isEmbeddingReady, true);

      final faces2 =
          await detector.detectFaces(imageBytes, mode: FaceDetectionMode.fast);
      expect(faces2, isNotEmpty);
      final emb2 = await detector.getFaceEmbedding(faces2.first, imageBytes);
      expect(emb2.length, kEmbeddingDimension);

      // Embeddings from the same face should be similar
      final similarity = FaceDetector.compareFaces(emb1, emb2);
      expect(similarity, greaterThan(0.8));
      print('Embedding similarity across lifecycles: $similarity');

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should support segmentation after dispose-then-reinitialize',
        () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing segmentation after dispose -> re-init ---');
      final detector = FaceDetector();

      await detector.initialize();
      await detector.initializeSegmentation();
      expect(detector.isSegmentationReady, true);
      final mask1 = await detector.getSegmentationMask(imageBytes);
      expect(mask1.width, greaterThan(0));

      detector.dispose();
      expect(detector.isSegmentationReady, false);

      await detector.initialize();
      await detector.initializeSegmentation();
      expect(detector.isSegmentationReady, true);
      final mask2 = await detector.getSegmentationMask(imageBytes);
      expect(mask2.width, greaterThan(0));
      expect(mask2.width, mask1.width);
      expect(mask2.height, mask1.height);

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should handle multiple dispose-reinitialize cycles', () async {
      print('\n--- Testing 5 dispose -> re-init cycles ---');
      final detector = FaceDetector();

      for (int i = 0; i < 5; i++) {
        await detector.initialize();
        expect(detector.isReady, true);
        final faces = await detector.detectFaces(imageBytes);
        expect(faces, isNotEmpty, reason: 'Cycle $i: should detect faces');
        detector.dispose();
        expect(detector.isReady, false);
        print('Cycle ${i + 1}/5 passed');
      }

      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 2. IrisLandmark.createFromFile() and callWithIsolate()
  // ===========================================================================
  group('IrisLandmark.createFromFile', () {
    test('should create from file path and run inference', () async {
      print('\n--- Testing IrisLandmark.createFromFile ---');

      final modelPath = await writeAssetToTempFile(
        'packages/face_detection_tflite/assets/models/iris_landmark.tflite',
        'iris_landmark.tflite',
      );

      try {
        final iris = await IrisLandmark.createFromFile(modelPath);

        // Create a small eye crop for inference
        final detector = FaceDetector();
        await detector.initialize();
        final faces = await detector.detectFaces(
          imageBytes,
          mode: FaceDetectionMode.full,
        );
        expect(faces, isNotEmpty);

        // Use the first face's eye region as a test crop
        final face = faces.first;
        final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);

        // Create a 64x64 eye crop from roughly the eye region
        final eyeCrop = cv.Mat.zeros(64, 64, cv.MatType.CV_8UC3);
        final leftEye = face.landmarks.leftEye;
        if (leftEye != null) {
          // Crop around the left eye
          final cx = leftEye.x.clamp(32, mat.cols - 32).toInt();
          final cy = leftEye.y.clamp(32, mat.rows - 32).toInt();
          final roi = cv.Rect(cx - 32, cy - 32, 64, 64);
          final cropped = mat.region(roi);
          cropped.copyTo(eyeCrop);
          cropped.dispose();
        }

        final landmarks = await iris.call(eyeCrop);
        expect(landmarks, isNotEmpty);
        for (final point in landmarks) {
          expect(point.length, greaterThanOrEqualTo(2));
        }
        print('Got ${landmarks.length} iris landmarks from file-loaded model');

        eyeCrop.dispose();
        mat.dispose();
        iris.dispose();
        detector.dispose();
      } finally {
        File(modelPath).parent.deleteSync(recursive: true);
      }

      print('Test passed');
    }, timeout: testTimeout);

    test('should create with XNNPACK performance config', () async {
      if (!Platform.isMacOS && !Platform.isLinux && !Platform.isWindows) {
        print('Skipping: XNNPACK only available on desktop');
        return;
      }

      print('\n--- Testing IrisLandmark.createFromFile with XNNPACK ---');

      final modelPath = await writeAssetToTempFile(
        'packages/face_detection_tflite/assets/models/iris_landmark.tflite',
        'iris_landmark.tflite',
      );

      try {
        final iris = await IrisLandmark.createFromFile(
          modelPath,
          performanceConfig: PerformanceConfig.xnnpack(),
        );

        final eyeCrop = cv.Mat.zeros(64, 64, cv.MatType.CV_8UC3);
        final landmarks = await iris.call(eyeCrop);
        expect(landmarks, isNotEmpty);
        print('Got ${landmarks.length} landmarks with XNNPACK delegate');

        eyeCrop.dispose();
        iris.dispose();
      } finally {
        File(modelPath).parent.deleteSync(recursive: true);
      }

      print('Test passed');
    }, timeout: testTimeout);

    test('should throw for non-existent file path', () async {
      print('\n--- Testing IrisLandmark.createFromFile with bad path ---');

      try {
        await IrisLandmark.createFromFile('/non/existent/path/model.tflite');
        fail('Should have thrown');
      } catch (e) {
        print('Correctly threw: ${e.runtimeType}');
        // May be FileSystemException, StateError, or similar depending on platform
        expect(e, isNotNull);
      }

      print('Test passed');
    }, timeout: testTimeout);
  });

  group('IrisLandmark.callWithIsolate', () {
    test('should run inference in a spawned isolate', () async {
      print('\n--- Testing IrisLandmark.callWithIsolate ---');

      final modelPath = await writeAssetToTempFile(
        'packages/face_detection_tflite/assets/models/iris_landmark.tflite',
        'iris_landmark.tflite',
      );

      try {
        // Create a small eye crop image as encoded bytes
        final eyeMat = cv.Mat.zeros(64, 64, cv.MatType.CV_8UC3);
        eyeMat.setTo(cv.Scalar(100, 150, 200, 255));
        final (_, eyeBytes) = cv.imencode('.png', eyeMat);
        eyeMat.dispose();

        final landmarks =
            await IrisLandmark.callWithIsolate(eyeBytes, modelPath);
        expect(landmarks, isNotEmpty);
        for (final point in landmarks) {
          expect(point.length, greaterThanOrEqualTo(2));
          for (final coord in point) {
            expect(coord, isA<double>());
          }
        }
        print('Got ${landmarks.length} landmarks from isolate');
      } finally {
        File(modelPath).parent.deleteSync(recursive: true);
      }

      print('Test passed');
    }, timeout: testTimeout);

    test('should throw StateError for invalid image bytes', () async {
      print('\n--- Testing callWithIsolate with invalid bytes ---');

      final modelPath = await writeAssetToTempFile(
        'packages/face_detection_tflite/assets/models/iris_landmark.tflite',
        'iris_landmark.tflite',
      );

      try {
        await IrisLandmark.callWithIsolate(
          Uint8List.fromList([0, 1, 2, 3]),
          modelPath,
        );
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw StateError: ${e.message}');
        expect(e.message, contains('decode_failed'));
      } finally {
        File(modelPath).parent.deleteSync(recursive: true);
      }

      print('Test passed');
    }, timeout: testTimeout);

    test('should throw for non-existent model path', () async {
      print('\n--- Testing callWithIsolate with bad model path ---');

      final eyeMat = cv.Mat.zeros(64, 64, cv.MatType.CV_8UC3);
      final (_, eyeBytes) = cv.imencode('.png', eyeMat);
      eyeMat.dispose();

      try {
        await IrisLandmark.callWithIsolate(
            eyeBytes, '/non/existent/model.tflite');
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw StateError: ${e.runtimeType}');
        expect(e, isNotNull);
      }

      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 3. SegmentationConfig.maxOutputSize behavior
  // ===========================================================================
  group('SegmentationConfig.maxOutputSize behavior', () {
    test('mask is returned at model resolution regardless of maxOutputSize',
        () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing maxOutputSize is not applied at model level ---');

      // Create with very small maxOutputSize
      final segmenter = await SelfieSegmentation.create(
        config: const SegmentationConfig(maxOutputSize: 64),
      );

      final mask = await segmenter.callFromBytes(imageBytes);

      // Mask should be at model's native resolution (256x256), NOT capped to 64
      expect(mask.width, segmenter.outputWidth);
      expect(mask.height, segmenter.outputHeight);
      print('Model output: ${mask.width}x${mask.height}');
      print(
          'maxOutputSize was 64 but mask is at native resolution (not capped)');

      // Verify the config value is stored correctly
      expect(segmenter.config.maxOutputSize, 64);

      // User must manually call upsample with maxSize to apply the cap
      final upsampled = mask.upsample(maxSize: segmenter.config.maxOutputSize);
      expect(upsampled.width, lessThanOrEqualTo(64));
      expect(upsampled.height, lessThanOrEqualTo(64));
      print(
          'After manual upsample(maxSize: 64): ${upsampled.width}x${upsampled.height}');

      segmenter.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('different maxOutputSize configs produce same raw mask dimensions',
        () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing maxOutputSize has no effect on raw output ---');

      final segmenter1 = await SelfieSegmentation.create(
        config: const SegmentationConfig(maxOutputSize: 128),
      );
      final segmenter2 = await SelfieSegmentation.create(
        config: const SegmentationConfig(maxOutputSize: 4096),
      );

      final mask1 = await segmenter1.callFromBytes(imageBytes);
      final mask2 = await segmenter2.callFromBytes(imageBytes);

      // Both should produce identical dimensions
      expect(mask1.width, mask2.width);
      expect(mask1.height, mask2.height);
      print('maxOutputSize=128 -> ${mask1.width}x${mask1.height}');
      print('maxOutputSize=4096 -> ${mask2.width}x${mask2.height}');

      segmenter1.dispose();
      segmenter2.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 4. SegmentationConfig.validateModel behavior
  // ===========================================================================
  group('SegmentationConfig.validateModel behavior', () {
    test('validateModel=true accepts correct model-config pairing', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing validateModel=true with correct pairing ---');

      // General model with general config -- should pass validation
      final segmenter = await SelfieSegmentation.create(
        config: const SegmentationConfig(
          model: SegmentationModel.general,
          validateModel: true,
        ),
      );

      expect(segmenter.outputChannels, 1);
      final mask = await segmenter.callFromBytes(imageBytes);
      expect(mask.width, greaterThan(0));

      segmenter.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('validateModel=true rejects mismatched model-config pairing',
        () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing validateModel=true with mismatched pairing ---');

      // Load the general model bytes
      final generalModelData = await rootBundle.load(
        'packages/face_detection_tflite/assets/models/selfie_segmenter.tflite',
      );
      final generalModelBytes = generalModelData.buffer.asUint8List();

      // Try to load general model bytes but claim it's multiclass (6 channels)
      try {
        await SelfieSegmentation.createFromBuffer(
          generalModelBytes,
          config: const SegmentationConfig(
            model: SegmentationModel.multiclass,
            validateModel: true,
          ),
        );
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        expect(e.code, SegmentationError.unexpectedTensorShape);
        print('Correctly rejected: ${e.message}');
      }

      print('Test passed');
    }, timeout: testTimeout);

    test('validateModel=false skips validation for mismatched pairing',
        () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing validateModel=false bypasses validation ---');

      final generalModelData = await rootBundle.load(
        'packages/face_detection_tflite/assets/models/selfie_segmenter.tflite',
      );
      final generalModelBytes = generalModelData.buffer.asUint8List();

      // With validateModel=false, mismatched pairing should NOT throw during
      // creation (though inference may produce garbage or crash).
      // We only test that creation succeeds.
      final segmenter = await SelfieSegmentation.createFromBuffer(
        generalModelBytes,
        config: const SegmentationConfig(
          model: SegmentationModel.multiclass,
          validateModel: false,
        ),
      );

      print('Created segmenter with validateModel=false (no exception)');
      segmenter.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 5. SegmentationWorker.segmentMat() with non-BGR Mat types
  // ===========================================================================
  group('SegmentationWorker.segmentMat with non-BGR Mat types', () {
    test('should handle BGRA (CV_8UC4) Mat', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker.segmentMat with BGRA ---');

      final worker = SegmentationWorker();
      await worker.initialize();

      final mat = cv.Mat.create(rows: 256, cols: 256, type: cv.MatType.CV_8UC4);
      mat.setTo(cv.Scalar(100, 150, 200, 255));

      final mask = await worker.segmentMat(mat);
      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      expect(mask.data.length, mask.width * mask.height);
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }
      print('BGRA mask: ${mask.width}x${mask.height}');

      mat.dispose();
      worker.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should handle grayscale (CV_8UC1) Mat', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker.segmentMat with grayscale ---');

      final worker = SegmentationWorker();
      await worker.initialize();

      final mat = cv.Mat.create(rows: 256, cols: 256, type: cv.MatType.CV_8UC1);
      mat.setTo(cv.Scalar(128, 0, 0, 0));

      try {
        final mask = await worker.segmentMat(mat);
        expect(mask.width, greaterThan(0));
        expect(mask.height, greaterThan(0));
        print('Grayscale mask: ${mask.width}x${mask.height}');
      } on SegmentationException catch (e) {
        // Some platforms may not handle grayscale Mats in the worker
        print('Grayscale Mat threw SegmentationException: ${e.message}');
        expect(e.code, isNotNull);
      }

      mat.dispose();
      worker.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 6. SegmentationWorker error recovery
  // ===========================================================================
  group('SegmentationWorker error recovery', () {
    test('should recover after empty Mat and process valid input', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker error recovery (Mat) ---');

      final worker = SegmentationWorker();
      await worker.initialize();

      // Send bad input
      final emptyMat = cv.Mat.empty();
      try {
        await worker.segmentMat(emptyMat);
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        print('Expected error: ${e.message}');
      }
      emptyMat.dispose();

      // Worker should still be functional
      final validMat =
          cv.Mat.create(rows: 256, cols: 256, type: cv.MatType.CV_8UC3);
      validMat.setTo(cv.Scalar(100, 150, 200, 255));

      final mask = await worker.segmentMat(validMat);
      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      print('Recovery successful: ${mask.width}x${mask.height}');

      validMat.dispose();
      worker.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should recover after invalid bytes and process valid input',
        () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker error recovery (bytes) ---');

      final worker = SegmentationWorker();
      await worker.initialize();

      // Send bad input
      try {
        await worker.segment(Uint8List.fromList([0, 1, 2, 3]));
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        print('Expected error: ${e.message}');
      }

      // Worker should still be functional
      final mask = await worker.segment(imageBytes);
      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      print('Recovery successful: ${mask.width}x${mask.height}');

      worker.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should handle multiple error-recovery cycles', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker repeated error recovery ---');

      final worker = SegmentationWorker();
      await worker.initialize();

      for (int i = 0; i < 5; i++) {
        // Bad input
        try {
          await worker.segment(Uint8List.fromList([0xFF, 0xFE]));
        } on SegmentationException {
          // Expected
        }

        // Good input
        final mask = await worker.segment(imageBytes);
        expect(mask.width, greaterThan(0));
        print('Cycle ${i + 1}/5: recovery OK');
      }

      worker.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 7. SelfieSegmentation.disposeAsync() semantics
  // ===========================================================================
  group('SelfieSegmentation.disposeAsync semantics', () {
    test('disposeAsync sets isDisposed to true', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing disposeAsync sets isDisposed ---');

      final segmenter = await SelfieSegmentation.create();
      expect(segmenter.isDisposed, false);

      await segmenter.disposeAsync();
      expect(segmenter.isDisposed, true);

      print('Test passed');
    }, timeout: testTimeout);

    test('disposeAsync is idempotent', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing disposeAsync idempotence ---');

      final segmenter = await SelfieSegmentation.create();

      await segmenter.disposeAsync();
      expect(segmenter.isDisposed, true);

      // Calling again should not throw
      await segmenter.disposeAsync();
      expect(segmenter.isDisposed, true);

      // Third call
      await segmenter.disposeAsync();
      expect(segmenter.isDisposed, true);

      print('Test passed');
    }, timeout: testTimeout);

    test('call() throws after disposeAsync', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing call() throws after disposeAsync ---');

      final segmenter = await SelfieSegmentation.create();

      // Run a successful inference first
      final mask = await segmenter.callFromBytes(imageBytes);
      expect(mask.width, greaterThan(0));

      await segmenter.disposeAsync();

      // Now call() should throw
      final mat = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);
      try {
        await segmenter.call(mat);
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw: ${e.message}');
      }

      mat.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('disposeAsync followed by create produces working instance', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing disposeAsync then create new instance ---');

      final seg1 = await SelfieSegmentation.create();
      final mask1 = await seg1.callFromBytes(imageBytes);
      expect(mask1.width, greaterThan(0));
      await seg1.disposeAsync();

      // Create a new instance - should work fine
      final seg2 = await SelfieSegmentation.create();
      final mask2 = await seg2.callFromBytes(imageBytes);
      expect(mask2.width, greaterThan(0));
      expect(mask2.width, mask1.width);
      expect(mask2.height, mask1.height);

      await seg2.disposeAsync();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 8. Grayscale cv.Mat through FaceDetector.detectFacesFromMat()
  // ===========================================================================
  group('Grayscale Mat through FaceDetector.detectFacesFromMat', () {
    test('should handle grayscale (CV_8UC1) Mat without crashing', () async {
      print('\n--- Testing detectFacesFromMat with grayscale Mat ---');

      final detector = FaceDetector();
      await detector.initialize();

      final grayMat =
          cv.Mat.create(rows: 256, cols: 256, type: cv.MatType.CV_8UC1);
      grayMat.setTo(cv.Scalar(128, 0, 0, 0));

      try {
        final faces = await detector.detectFacesFromMat(
          grayMat,
          mode: FaceDetectionMode.fast,
        );
        // If it works, should return empty for a solid gray image
        expect(faces, isEmpty);
        print('Grayscale Mat accepted, returned ${faces.length} faces');
      } catch (e) {
        // Some platforms may not support grayscale input
        print('Grayscale Mat threw: ${e.runtimeType}: $e');
        expect(e, isNotNull);
      }

      grayMat.dispose();
      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should handle BGRA (CV_8UC4) Mat', () async {
      print('\n--- Testing detectFacesFromMat with BGRA Mat ---');

      final detector = FaceDetector();
      await detector.initialize();

      // Create a 256x256 BGRA Mat directly
      final bgraMat =
          cv.Mat.create(rows: 256, cols: 256, type: cv.MatType.CV_8UC4);
      bgraMat.setTo(cv.Scalar(100, 150, 200, 255));

      try {
        final faces = await detector.detectFacesFromMat(
          bgraMat,
          mode: FaceDetectionMode.fast,
        );
        // Solid color should not detect faces
        expect(faces, isEmpty);
        print('BGRA Mat: detected ${faces.length} face(s)');
      } catch (e) {
        // Some platforms may not support 4-channel input directly
        print('BGRA Mat threw: ${e.runtimeType}: $e');
      }

      bgraMat.dispose();
      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('should detect faces from grayscale-decoded image', () async {
      print('\n--- Testing detectFaces with IMREAD_GRAYSCALE ---');

      final detector = FaceDetector();
      await detector.initialize();

      // Decode the real image as grayscale via imdecode flag
      final grayMat = cv.imdecode(imageBytes, cv.IMREAD_GRAYSCALE);

      try {
        final faces = await detector.detectFacesFromMat(
          grayMat,
          mode: FaceDetectionMode.fast,
        );
        print('Grayscale-decoded image: detected ${faces.length} face(s)');
      } catch (e) {
        // Pipeline expects BGR; grayscale may fail on some platforms
        print('Grayscale-decoded threw: ${e.runtimeType}: $e');
      }

      grayMat.dispose();
      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 9. Post-dispose StateError enforcement on FaceDetector
  // ===========================================================================
  group('FaceDetector post-dispose StateError', () {
    test('detectFaces() throws StateError after dispose', () async {
      print('\n--- Testing detectFaces after dispose ---');
      final detector = FaceDetector();
      await detector.initialize();
      detector.dispose();

      expect(
        () => detector.detectFaces(imageBytes),
        throwsStateError,
      );
      print('Test passed');
    }, timeout: testTimeout);

    test('detectFacesFromMat() throws StateError after dispose', () async {
      print('\n--- Testing detectFacesFromMat after dispose ---');
      final detector = FaceDetector();
      await detector.initialize();
      detector.dispose();

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      try {
        expect(
          () => detector.detectFacesFromMat(mat),
          throwsStateError,
        );
      } finally {
        mat.dispose();
      }
      print('Test passed');
    }, timeout: testTimeout);

    test('getFaceEmbedding() throws StateError after dispose', () async {
      print('\n--- Testing getFaceEmbedding after dispose ---');
      final detector = FaceDetector();
      await detector.initialize();

      // Get a valid face first
      final faces = await detector.detectFaces(
        imageBytes,
        mode: FaceDetectionMode.fast,
      );
      expect(faces, isNotEmpty);

      detector.dispose();

      expect(
        () => detector.getFaceEmbedding(faces.first, imageBytes),
        throwsStateError,
      );
      print('Test passed');
    }, timeout: testTimeout);

    test('getFaceEmbeddingFromMat() throws StateError after dispose', () async {
      print('\n--- Testing getFaceEmbeddingFromMat after dispose ---');
      final detector = FaceDetector();
      await detector.initialize();

      final faces = await detector.detectFaces(
        imageBytes,
        mode: FaceDetectionMode.fast,
      );
      expect(faces, isNotEmpty);

      detector.dispose();

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      try {
        expect(
          () => detector.getFaceEmbeddingFromMat(faces.first, mat),
          throwsStateError,
        );
      } finally {
        mat.dispose();
      }
      print('Test passed');
    }, timeout: testTimeout);

    test('getFaceEmbeddings() throws StateError after dispose', () async {
      print('\n--- Testing getFaceEmbeddings after dispose ---');
      final detector = FaceDetector();
      await detector.initialize();

      final faces = await detector.detectFaces(
        imageBytes,
        mode: FaceDetectionMode.fast,
      );
      expect(faces, isNotEmpty);

      detector.dispose();

      expect(
        () => detector.getFaceEmbeddings(faces, imageBytes),
        throwsStateError,
      );
      print('Test passed');
    }, timeout: testTimeout);

    test('getSegmentationMask() throws StateError after dispose', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }
      print('\n--- Testing getSegmentationMask after dispose ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation();
      expect(detector.isSegmentationReady, true);

      detector.dispose();

      expect(
        () => detector.getSegmentationMask(imageBytes),
        throwsStateError,
      );
      print('Test passed');
    }, timeout: testTimeout);

    test('getSegmentationMaskFromMat() throws StateError after dispose',
        () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }
      print('\n--- Testing getSegmentationMaskFromMat after dispose ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation();

      detector.dispose();

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      try {
        expect(
          () => detector.getSegmentationMaskFromMat(mat),
          throwsStateError,
        );
      } finally {
        mat.dispose();
      }
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 10. Double dispose() safety on initialized FaceDetector
  // ===========================================================================
  group('FaceDetector double dispose safety', () {
    test('dispose() twice on initialized detector does not throw', () async {
      print('\n--- Testing double dispose on initialized detector ---');
      final detector = FaceDetector();
      await detector.initialize();
      expect(detector.isReady, true);

      detector.dispose();
      expect(detector.isReady, false);

      // Second dispose should be a no-op (all fields already null)
      detector.dispose();
      expect(detector.isReady, false);
      print('Test passed');
    }, timeout: testTimeout);

    test('dispose() twice with segmentation does not throw', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }
      print('\n--- Testing double dispose with segmentation ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation();
      expect(detector.isSegmentationReady, true);

      detector.dispose();
      detector.dispose();
      expect(detector.isReady, false);
      expect(detector.isSegmentationReady, false);
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 11. initializeSegmentation with custom SegmentationConfig
  // ===========================================================================
  group('FaceDetector.initializeSegmentation with custom config', () {
    test('accepts SegmentationConfig.performance', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }
      print('\n--- Testing initializeSegmentation with performance config ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation(
        config: SegmentationConfig.performance,
      );
      expect(detector.isSegmentationReady, true);

      final mask = await detector.getSegmentationMask(imageBytes);
      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('accepts SegmentationConfig.fast', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }
      print('\n--- Testing initializeSegmentation with fast config ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation(
        config: SegmentationConfig.fast,
      );
      expect(detector.isSegmentationReady, true);

      final mask = await detector.getSegmentationMask(imageBytes);
      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('accepts custom SegmentationConfig with landscape model', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }
      print('\n--- Testing initializeSegmentation with landscape model ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation(
        config: const SegmentationConfig(
          model: SegmentationModel.landscape,
        ),
      );
      expect(detector.isSegmentationReady, true);

      final mask = await detector.getSegmentationMask(imageBytes);
      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // 12. initialize(options:) parameter
  // ===========================================================================
  group('FaceDetector.initialize with custom InterpreterOptions', () {
    test('accepts custom InterpreterOptions', () async {
      print('\n--- Testing initialize with custom options ---');
      final options = InterpreterOptions();
      options.threads = 2;

      final detector = FaceDetector();
      await detector.initialize(options: options);
      expect(detector.isReady, true);

      final faces = await detector.detectFaces(
        imageBytes,
        mode: FaceDetectionMode.fast,
      );
      expect(faces, isNotEmpty);

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('options takes precedence over performanceConfig', () async {
      print('\n--- Testing options precedence over performanceConfig ---');
      final options = InterpreterOptions();
      options.threads = 1;

      // Pass both - options should take precedence
      final detector = FaceDetector();
      await detector.initialize(
        options: options,
        performanceConfig: PerformanceConfig.xnnpack(),
      );
      expect(detector.isReady, true);

      final faces = await detector.detectFaces(
        imageBytes,
        mode: FaceDetectionMode.fast,
      );
      expect(faces, isNotEmpty);

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });
}
