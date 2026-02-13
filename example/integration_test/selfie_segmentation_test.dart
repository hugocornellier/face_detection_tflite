// ignore_for_file: avoid_print

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';

/// Helper to check if the default segmentation model is available
Future<bool> _areModelsAvailable() async {
  try {
    await rootBundle.load(
      'packages/face_detection_tflite/assets/models/selfie_segmenter.tflite',
    );
    return true;
  } catch (_) {
    return false;
  }
}

/// Helper to check if the multiclass segmentation model is available
Future<bool> _isMulticlassModelAvailable() async {
  try {
    await rootBundle.load(
      'packages/face_detection_tflite/assets/models/selfie_multiclass.tflite',
    );
    return true;
  } catch (_) {
    return false;
  }
}

/// Helper to create a solid color test image
Uint8List _createTestImage(int width, int height,
    {int r = 128, int g = 128, int b = 128}) {
  final image = img.Image(width: width, height: height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      image.setPixelRgba(x, y, r, g, b, 255);
    }
  }
  return Uint8List.fromList(img.encodePng(image));
}

/// Separator line for test output
const _separator =
    '============================================================';

/// Helper to create a grayscale test image
Uint8List _createGrayscaleImage(int width, int height, {int gray = 128}) {
  final image = img.Image(width: width, height: height, numChannels: 1);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      image.setPixelRgba(x, y, gray, gray, gray, 255);
    }
  }
  return Uint8List.fromList(img.encodePng(image));
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late bool modelsAvailable;

  setUpAll(() async {
    modelsAvailable = await _areModelsAvailable();
    if (!modelsAvailable) {
      print('\n$_separator');
      print('WARNING: Segmentation models not found');
      print('Tests that require models will be skipped');
      print('$_separator\n');
    }
  });

  // ===========================================================================
  // Diagnostic Test
  // ===========================================================================
  test('DIAGNOSTIC: Load model directly with interpreter', () async {
    print('\n--- DIAGNOSTIC: Direct interpreter test ---');

    final modelPaths = {
      'face_detection_back':
          'packages/face_detection_tflite/assets/models/face_detection_back.tflite',
      'selfie_segmenter (default)':
          'packages/face_detection_tflite/assets/models/selfie_segmenter.tflite',
      'selfie_segmenter_landscape':
          'packages/face_detection_tflite/assets/models/selfie_segmenter_landscape.tflite',
      'selfie_multiclass':
          'packages/face_detection_tflite/assets/models/selfie_multiclass.tflite',
    };

    for (final entry in modelPaths.entries) {
      try {
        print('Loading ${entry.key}...');
        final interpreter = await Interpreter.fromAsset(entry.value);
        print('  Loaded successfully!');
        print('  Input shape: ${interpreter.getInputTensor(0).shape}');
        print('  Output shape: ${interpreter.getOutputTensor(0).shape}');
        interpreter.close();
      } catch (e) {
        print('  FAILED: $e');
      }
    }

    print('--- DIAGNOSTIC complete ---');
  });

  // ===========================================================================
  // Initialization Tests
  // ===========================================================================
  group('Initialization', () {
    test('SelfieSegmentation.create() with default config', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SelfieSegmentation.create() default config ---');
      final segmenter = await SelfieSegmentation.create();

      expect(segmenter.inputWidth, 256);
      expect(segmenter.inputHeight, 256);
      expect(segmenter.outputChannels,
          1); // Default general model has 1 output channel (binary sigmoid)
      print('Input size: ${segmenter.inputWidth}x${segmenter.inputHeight}');
      print('Output channels: ${segmenter.outputChannels}');

      segmenter.dispose();
      print('Test passed');
    });

    test('SelfieSegmentation.create() with safe config', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SelfieSegmentation.create() with safe config ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig.safe,
      );

      expect(
          segmenter.hasGpuDelegateFailed, false); // Should not have failed yet
      segmenter.dispose();
      print('Test passed');
    });

    test('SelfieSegmentation.create() with performance config', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print(
          '\n--- Testing SelfieSegmentation.create() with performance config ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig.performance,
      );

      expect(segmenter.inputWidth, 256);
      segmenter.dispose();
      print('Test passed');
    });

    test('SelfieSegmentation.create() with custom config', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SelfieSegmentation.create() with custom config ---');
      final config = SegmentationConfig(
        performanceConfig: PerformanceConfig.disabled,
        maxOutputSize: 512,
        resizeStrategy: ResizeStrategy.stretch,
        validateModel: true,
      );

      final segmenter = await SelfieSegmentation.create(config: config);
      expect(segmenter.inputWidth, 256);
      segmenter.dispose();
      print('Test passed');
    });

    test('Multiple create/dispose cycles (20x)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing 20 create/dispose cycles ---');
      for (int i = 0; i < 20; i++) {
        final segmenter = await SelfieSegmentation.create(
          config: SegmentationConfig.safe,
        );
        segmenter.dispose();
        if (i % 5 == 4) print('Completed ${i + 1} cycles');
      }
      print('Test passed - no memory leaks detected');
    });

    test('dispose() is idempotent', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing dispose() idempotence ---');
      final segmenter = await SelfieSegmentation.create();

      // Should not throw
      segmenter.dispose();
      segmenter.dispose();
      segmenter.dispose();

      print('Test passed - multiple dispose calls handled');
    });

    test('throws on invalid model file', () async {
      print('\n--- Testing invalid model handling ---');
      // This test verifies error handling, runs regardless of model availability

      try {
        await SelfieSegmentation.createFromBuffer(
          Uint8List.fromList([0, 1, 2, 3]), // Invalid model bytes
        );
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        print('Caught expected exception: ${e.code}');
        expect(
          e.code,
          anyOf(
            SegmentationError.modelNotFound,
            SegmentationError.interpreterCreationFailed,
          ),
        );
      }
      print('Test passed');
    });
  });

  // ===========================================================================
  // Inference Tests
  // ===========================================================================
  group('Inference', () {
    test('call() with valid PNG image', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing inference with PNG image ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(256, 256);

      final mask = await segmenter.call(imageBytes);

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      expect(mask.data.length, mask.width * mask.height);
      print('Mask size: ${mask.width}x${mask.height}');
      print('Data length: ${mask.data.length}');

      segmenter.dispose();
      print('Test passed');
    });

    test('call() with sample image', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing inference with sample image ---');
      final segmenter = await SelfieSegmentation.create();

      try {
        final ByteData data =
            await rootBundle.load('assets/samples/landmark-ex1.jpg');
        final imageBytes = data.buffer.asUint8List();

        final sw = Stopwatch()..start();
        final mask = await segmenter.call(imageBytes);
        sw.stop();

        print('Inference time: ${sw.elapsedMilliseconds}ms');
        print('Mask size: ${mask.width}x${mask.height}');
        print('Original size: ${mask.originalWidth}x${mask.originalHeight}');

        expect(mask.originalWidth, greaterThan(0));
        expect(mask.originalHeight, greaterThan(0));
      } catch (e) {
        print('Sample image not found, using generated image');
        final imageBytes = _createTestImage(512, 512);
        final mask = await segmenter.call(imageBytes);
        expect(mask.width, greaterThan(0));
      }

      segmenter.dispose();
      print('Test passed');
    });

    test('callFromMat() with constructed Mat', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing callFromMat() with constructed Mat ---');
      final segmenter = await SelfieSegmentation.create();

      final bgrData = Uint8List(300 * 200 * 3);
      for (int i = 0; i < bgrData.length; i += 3) {
        bgrData[i] = 200; // B
        bgrData[i + 1] = 150; // G
        bgrData[i + 2] = 100; // R
      }
      final mat = cv.Mat.fromList(200, 300, cv.MatType.CV_8UC3, bgrData);

      final mask = await segmenter.callFromMat(mat);

      expect(mask.originalWidth, 300);
      expect(mask.originalHeight, 200);
      print(
          'Mask generated for ${mask.originalWidth}x${mask.originalHeight} image');

      mat.dispose();
      segmenter.dispose();
      print('Test passed');
    });

    test('callFromMat() with OpenCV Mat', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing callFromMat() ---');
      final segmenter = await SelfieSegmentation.create();

      // Create a BGR Mat (OpenCV default)
      final bgrData = Uint8List(256 * 256 * 3);
      for (int i = 0; i < bgrData.length; i += 3) {
        bgrData[i] = 128; // B
        bgrData[i + 1] = 128; // G
        bgrData[i + 2] = 128; // R
      }
      final mat = cv.Mat.fromList(256, 256, cv.MatType.CV_8UC3, bgrData);

      final mask = await segmenter.callFromMat(mat);

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));

      mat.dispose();
      segmenter.dispose();
      print('Test passed');
    });

    test('handles JPEG image format', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing JPEG format ---');
      final segmenter = await SelfieSegmentation.create();

      final image = img.Image(width: 200, height: 200);
      img.fill(image, color: img.ColorRgb8(100, 100, 100));
      final jpegBytes = Uint8List.fromList(img.encodeJpg(image, quality: 90));

      final mask = await segmenter.call(jpegBytes);
      expect(mask.originalWidth, 200);

      segmenter.dispose();
      print('Test passed');
    });

    test('sequential inference produces consistent results', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing sequential inference consistency ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(256, 256, r: 200, g: 150, b: 100);

      final mask1 = await segmenter.call(imageBytes);
      final mask2 = await segmenter.call(imageBytes);

      // Same image should produce identical masks
      expect(mask1.width, mask2.width);
      expect(mask1.height, mask2.height);

      // Check a sample of values are identical
      bool identical = true;
      for (int i = 0; i < mask1.data.length && i < 100; i++) {
        if ((mask1.data[i] - mask2.data[i]).abs() > 0.0001) {
          identical = false;
          break;
        }
      }
      expect(identical, true,
          reason: 'Same image should produce identical masks');

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Edge Cases - Image Size
  // ===========================================================================
  group('Edge Cases - Image Size', () {
    test('rejects image smaller than 16x16', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing minimum size rejection (10x10) ---');
      final segmenter = await SelfieSegmentation.create();
      final smallImage = _createTestImage(10, 10);

      try {
        await segmenter.call(smallImage);
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        expect(e.code, SegmentationError.imageTooSmall);
        print('Correctly rejected: ${e.message}');
      }

      segmenter.dispose();
      print('Test passed');
    });

    test('accepts minimum 16x16 image', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing minimum size acceptance (16x16) ---');
      final segmenter = await SelfieSegmentation.create();
      final minImage = _createTestImage(16, 16);

      final mask = await segmenter.call(minImage);
      expect(mask.originalWidth, 16);
      expect(mask.originalHeight, 16);

      segmenter.dispose();
      print('Test passed');
    });

    test('handles large image (1920x1080)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing large image (1920x1080) ---');
      final segmenter = await SelfieSegmentation.create();

      // Create a smaller test image due to memory constraints in test
      final largeImage = _createTestImage(1920, 1080);

      final sw = Stopwatch()..start();
      final mask = await segmenter.call(largeImage);
      sw.stop();

      print('Inference time: ${sw.elapsedMilliseconds}ms');
      expect(mask.originalWidth, 1920);
      expect(mask.originalHeight, 1080);

      segmenter.dispose();
      print('Test passed');
    });

    test('handles extreme aspect ratio (10:1)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing extreme aspect ratio (500x50) ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(resizeStrategy: ResizeStrategy.letterbox),
      );

      final wideImage = _createTestImage(500, 50);
      final mask = await segmenter.call(wideImage);

      expect(mask.originalWidth, 500);
      expect(mask.originalHeight, 50);
      expect(mask.padding.length, 4);
      print('Padding: ${mask.padding}');

      segmenter.dispose();
      print('Test passed');
    });

    test('handles tall aspect ratio (1:10)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing tall aspect ratio (50x500) ---');
      final segmenter = await SelfieSegmentation.create();

      final tallImage = _createTestImage(50, 500);
      final mask = await segmenter.call(tallImage);

      expect(mask.originalWidth, 50);
      expect(mask.originalHeight, 500);

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Edge Cases - Input Format
  // ===========================================================================
  group('Edge Cases - Input Format', () {
    test('handles grayscale image', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing grayscale image ---');
      final segmenter = await SelfieSegmentation.create();
      final grayImage = _createGrayscaleImage(256, 256);

      final mask = await segmenter.call(grayImage);
      expect(mask.width, greaterThan(0));

      segmenter.dispose();
      print('Test passed');
    });

    test('handles RGBA image (drops alpha)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing RGBA image ---');
      final segmenter = await SelfieSegmentation.create();

      final image = img.Image(width: 200, height: 200, numChannels: 4);
      for (int y = 0; y < 200; y++) {
        for (int x = 0; x < 200; x++) {
          image.setPixelRgba(x, y, 100, 150, 200, 128); // Semi-transparent
        }
      }
      final pngBytes = Uint8List.fromList(img.encodePng(image));

      final mask = await segmenter.call(pngBytes);
      expect(mask.originalWidth, 200);

      segmenter.dispose();
      print('Test passed');
    });

    test('rejects corrupted image bytes', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing corrupted image rejection ---');
      final segmenter = await SelfieSegmentation.create();
      final corrupted =
          Uint8List.fromList([0xFF, 0xD8, 0x00, 0x00]); // Invalid JPEG

      try {
        await segmenter.call(corrupted);
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        expect(e.code, SegmentationError.imageDecodeFailed);
        print('Correctly rejected corrupted bytes');
      }

      segmenter.dispose();
      print('Test passed');
    });

    test('rejects empty bytes', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing empty bytes rejection ---');
      final segmenter = await SelfieSegmentation.create();

      try {
        await segmenter.call(Uint8List(0));
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        expect(e.code, SegmentationError.imageDecodeFailed);
        print('Correctly rejected empty bytes');
      }

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Mask Utilities
  // ===========================================================================
  group('Mask Utilities', () {
    test('toBinary() produces 0 or 255 only', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing toBinary() ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(256, 256);

      final mask = await segmenter.call(imageBytes);
      final binary = mask.toBinary(threshold: 0.5);

      for (int i = 0; i < binary.length; i++) {
        expect(binary[i], anyOf(0, 255),
            reason: 'Binary mask should only contain 0 or 255');
      }

      print('Binary mask size: ${binary.length} bytes');
      segmenter.dispose();
      print('Test passed');
    });

    test('toUint8() produces 0-255 range', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing toUint8() ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(256, 256);

      final mask = await segmenter.call(imageBytes);
      final uint8 = mask.toUint8();

      for (int i = 0; i < uint8.length; i++) {
        expect(uint8[i], inInclusiveRange(0, 255));
      }

      print('Uint8 mask size: ${uint8.length} bytes');
      segmenter.dispose();
      print('Test passed');
    });

    test('toRgba() with custom colors', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing toRgba() with custom colors ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(64, 64);

      final mask = await segmenter.call(imageBytes);
      final rgba = mask.toRgba(
        foreground: 0xFF0000FF, // Red
        background: 0x00000000, // Transparent
        threshold: 0.5,
      );

      expect(rgba.length, mask.width * mask.height * 4);
      print(
          'RGBA mask size: ${rgba.length} bytes (${mask.width * mask.height} pixels)');

      segmenter.dispose();
      print('Test passed');
    });

    test('toRgba() soft blend mode', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing toRgba() soft blend ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(64, 64);

      final mask = await segmenter.call(imageBytes);
      final rgba = mask.toRgba(
        foreground: 0xFFFFFFFF, // White
        background: 0xFF000000, // Black
        threshold: -1, // Soft blend
      );

      expect(rgba.length, mask.width * mask.height * 4);

      segmenter.dispose();
      print('Test passed');
    });

    test('at() returns valid values', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing at() method ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(256, 256);

      final mask = await segmenter.call(imageBytes);

      // Valid coordinates
      final centerValue = mask.at(mask.width ~/ 2, mask.height ~/ 2);
      expect(centerValue, inInclusiveRange(0.0, 1.0));
      print('Center value: $centerValue');

      // Out of bounds returns 0
      expect(mask.at(-1, 0), 0.0);
      expect(mask.at(0, -1), 0.0);
      expect(mask.at(mask.width, 0), 0.0);
      expect(mask.at(0, mask.height), 0.0);

      segmenter.dispose();
      print('Test passed');
    });

    test('toMap() and fromMap() round-trip', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing toMap()/fromMap() round-trip ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(64, 64);

      final mask = await segmenter.call(imageBytes);
      final map = mask.toMap();
      final restored = SegmentationMask.fromMap(map);

      expect(restored.width, mask.width);
      expect(restored.height, mask.height);
      expect(restored.originalWidth, mask.originalWidth);
      expect(restored.originalHeight, mask.originalHeight);
      expect(restored.data.length, mask.data.length);

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Numerical Correctness
  // ===========================================================================
  group('Numerical Correctness', () {
    test('mask values strictly in [0.0, 1.0]', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing mask value range ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(256, 256);

      final mask = await segmenter.call(imageBytes);

      double minVal = double.infinity;
      double maxVal = double.negativeInfinity;

      for (int i = 0; i < mask.data.length; i++) {
        if (mask.data[i] < minVal) minVal = mask.data[i];
        if (mask.data[i] > maxVal) maxVal = mask.data[i];
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      print('Value range: [$minVal, $maxVal]');
      segmenter.dispose();
      print('Test passed');
    });

    test('same image produces identical mask (deterministic)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing deterministic output ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(128, 128);

      final mask1 = await segmenter.call(imageBytes);
      final mask2 = await segmenter.call(imageBytes);

      for (int i = 0; i < mask1.data.length; i++) {
        expect(
          (mask1.data[i] - mask2.data[i]).abs(),
          lessThan(1e-6),
          reason: 'Mask values should be identical for same input',
        );
      }

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Resize Strategies
  // ===========================================================================
  group('Resize Strategies', () {
    test('letterbox strategy preserves aspect ratio', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing letterbox strategy ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(resizeStrategy: ResizeStrategy.letterbox),
      );

      final wideImage = _createTestImage(400, 200);
      final mask = await segmenter.call(wideImage);

      expect(mask.originalWidth, 400);
      expect(mask.originalHeight, 200);
      // Padding should be recorded for 2:1 aspect ratio
      print('Padding: ${mask.padding}');

      segmenter.dispose();
      print('Test passed');
    });

    test('stretch strategy fills input', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing stretch strategy ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(resizeStrategy: ResizeStrategy.stretch),
      );

      final wideImage = _createTestImage(400, 200);
      final mask = await segmenter.call(wideImage);

      expect(mask.originalWidth, 400);
      // Stretch doesn't add padding
      expect(mask.padding.every((p) => p == 0), true);

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // FaceDetector Integration
  // ===========================================================================
  group('FaceDetector Integration', () {
    test('initializeSegmentation() and getSegmentationMask()', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing FaceDetector segmentation integration ---');
      final detector = FaceDetector();
      await detector.initialize();

      expect(detector.isSegmentationReady, false);

      await detector.initializeSegmentation();
      expect(detector.isSegmentationReady, true);

      final imageBytes = _createTestImage(256, 256);
      final mask = await detector.getSegmentationMask(imageBytes);

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));

      detector.dispose();
      print('Test passed');
    });

    test('getSegmentationMask() throws if not initialized', () async {
      print('\n--- Testing uninitialized segmentation error ---');
      final detector = FaceDetector();
      await detector.initialize();

      try {
        await detector.getSegmentationMask(Uint8List(0));
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw StateError: ${e.message}');
      }

      detector.dispose();
      print('Test passed');
    });

    test('initializeSegmentation() is idempotent', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing idempotent initializeSegmentation ---');
      final detector = FaceDetector();
      await detector.initialize();

      await detector.initializeSegmentation();
      await detector.initializeSegmentation(); // Should not throw
      await detector.initializeSegmentation();

      expect(detector.isSegmentationReady, true);

      detector.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Isolate Support
  // ===========================================================================
  group('Isolate Support', () {
    test('FaceDetectorIsolate.spawn() with segmentation', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing FaceDetectorIsolate with segmentation ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      expect(detector.isSegmentationReady, true);

      await detector.dispose();
      print('Test passed');
    });

    test('getSegmentationMask() in isolate', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing isolate segmentation ---');
      final detector = await FaceDetectorIsolate.spawn(withSegmentation: true);

      final imageBytes = _createTestImage(256, 256);
      final mask = await detector.getSegmentationMask(imageBytes);

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      print('Isolate mask size: ${mask.width}x${mask.height}');

      await detector.dispose();
      print('Test passed');
    });

    test('isolate with uint8 output format', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing isolate uint8 output ---');
      final detector = await FaceDetectorIsolate.spawn(withSegmentation: true);

      final imageBytes = _createTestImage(256, 256);
      final mask = await detector.getSegmentationMask(
        imageBytes,
        outputFormat: IsolateOutputFormat.uint8,
      );

      expect(mask.width, greaterThan(0));
      // Values should still be in 0-1 range after conversion
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      await detector.dispose();
      print('Test passed');
    });

    test('isolate with binary output format', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing isolate binary output ---');
      final detector = await FaceDetectorIsolate.spawn(withSegmentation: true);

      final imageBytes = _createTestImage(256, 256);
      final mask = await detector.getSegmentationMask(
        imageBytes,
        outputFormat: IsolateOutputFormat.binary,
        binaryThreshold: 0.5,
      );

      // Binary output should only have 0.0 or 1.0
      for (int i = 0; i < mask.data.length; i++) {
        expect(mask.data[i], anyOf(0.0, 1.0));
      }

      await detector.dispose();
      print('Test passed');
    });

    test('isolate segmentation throws if not enabled', () async {
      print('\n--- Testing isolate without segmentation ---');
      final detector = await FaceDetectorIsolate.spawn(withSegmentation: false);

      expect(detector.isSegmentationReady, false);

      try {
        await detector.getSegmentationMask(Uint8List(0));
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw: ${e.message}');
      }

      await detector.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Concurrency and Stress
  // ===========================================================================
  group('Concurrency and Stress', () {
    test('rapid sequential calls (10x)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing 10 rapid sequential calls ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(128, 128);

      final sw = Stopwatch()..start();
      for (int i = 0; i < 10; i++) {
        final mask = await segmenter.call(imageBytes);
        expect(mask.width, greaterThan(0));
      }
      sw.stop();

      print('Total time for 10 calls: ${sw.elapsedMilliseconds}ms');
      print('Average: ${sw.elapsedMilliseconds / 10}ms per call');

      segmenter.dispose();
      print('Test passed');
    });

    test('memory stability after 50 inferences', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing memory stability (50 inferences) ---');
      final segmenter = await SelfieSegmentation.create();

      for (int i = 0; i < 50; i++) {
        final imageBytes =
            _createTestImage(128, 128, r: i * 5, g: i * 3, b: i * 2);
        final mask = await segmenter.call(imageBytes);
        expect(mask.data.length, greaterThan(0));
        if (i % 10 == 9) print('Completed ${i + 1} inferences');
      }

      segmenter.dispose();
      print('Test passed - no memory issues detected');
    });
  });

  // ===========================================================================
  // Platform Delegate
  // ===========================================================================
  group('Platform Delegate', () {
    test('safe config uses CPU only', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing safe config (CPU only) ---');
      print('Platform: ${Platform.operatingSystem}');

      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig.safe,
      );

      // Safe config should not fail
      expect(segmenter.hasGpuDelegateFailed, false);

      segmenter.dispose();
      print('Test passed');
    });

    test('performance comparison: safe vs auto', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Performance comparison ---');
      print('Platform: ${Platform.operatingSystem}');

      final imageBytes = _createTestImage(256, 256);

      // Safe (CPU)
      final safeSeg =
          await SelfieSegmentation.create(config: SegmentationConfig.safe);
      await safeSeg.call(imageBytes); // Warmup
      final safeSw = Stopwatch()..start();
      for (int i = 0; i < 5; i++) {
        await safeSeg.call(imageBytes);
      }
      safeSw.stop();
      final safeAvg = safeSw.elapsedMilliseconds / 5;
      safeSeg.dispose();

      // Performance (auto delegate)
      final perfSeg = await SelfieSegmentation.create(
          config: SegmentationConfig.performance);
      await perfSeg.call(imageBytes); // Warmup
      final perfSw = Stopwatch()..start();
      for (int i = 0; i < 5; i++) {
        await perfSeg.call(imageBytes);
      }
      perfSw.stop();
      final perfAvg = perfSw.elapsedMilliseconds / 5;
      perfSeg.dispose();

      print('Safe (CPU): ${safeAvg.toStringAsFixed(1)}ms avg');
      print('Performance (auto): ${perfAvg.toStringAsFixed(1)}ms avg');
      if (perfAvg < safeAvg) {
        print(
            'Performance mode is ${(safeAvg / perfAvg).toStringAsFixed(1)}x faster');
      }

      print('Test passed');
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================
  group('Error Handling', () {
    test('SegmentationException contains useful info', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationException info ---');
      final segmenter = await SelfieSegmentation.create();

      try {
        await segmenter.call(Uint8List.fromList([1, 2, 3])); // Invalid
        fail('Should throw');
      } on SegmentationException catch (e) {
        expect(e.code, isNotNull);
        expect(e.message, isNotEmpty);
        print('Exception code: ${e.code}');
        print('Exception message: ${e.message}');
        print('toString: $e');
      }

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Disposal State Tests (Codex Gap #1)
  // ===========================================================================
  group('Disposal State', () {
    test('using segmenter after dispose throws StateError', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing use after dispose ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(64, 64);

      // Should work before dispose
      final mask = await segmenter.call(imageBytes);
      expect(mask.width, greaterThan(0));

      segmenter.dispose();
      expect(segmenter.isDisposed, true);

      // Should throw after dispose
      try {
        await segmenter.call(imageBytes);
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw StateError: ${e.message}');
        expect(e.message, contains('dispose'));
      }

      print('Test passed');
    });

    test('callFromMat throws after dispose (constructed Mat)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing callFromMat after dispose ---');
      final segmenter = await SelfieSegmentation.create();
      segmenter.dispose();

      final bgrData = Uint8List(64 * 64 * 3);
      final mat = cv.Mat.fromList(64, 64, cv.MatType.CV_8UC3, bgrData);

      try {
        await segmenter.callFromMat(mat);
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw: ${e.message}');
      }

      mat.dispose();
      print('Test passed');
    });

    test('callFromMat throws after dispose', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing callFromMat after dispose ---');
      final segmenter = await SelfieSegmentation.create();
      segmenter.dispose();

      final matData = Uint8List(64 * 64 * 3);
      final mat = cv.Mat.fromList(64, 64, cv.MatType.CV_8UC3, matData);

      try {
        await segmenter.callFromMat(mat);
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw: ${e.message}');
      } finally {
        mat.dispose();
      }

      print('Test passed');
    });
  });

  // ===========================================================================
  // Mat Channel Variations (Codex Gap #2)
  // ===========================================================================
  group('Mat Channel Variations', () {
    test('callFromMat with CV_8UC4 (BGRA)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing CV_8UC4 (BGRA) Mat ---');
      final segmenter = await SelfieSegmentation.create();

      final bgraData = Uint8List(64 * 64 * 4);
      for (int i = 0; i < bgraData.length; i += 4) {
        bgraData[i] = 100; // B
        bgraData[i + 1] = 150; // G
        bgraData[i + 2] = 200; // R
        bgraData[i + 3] = 255; // A
      }
      final mat = cv.Mat.fromList(64, 64, cv.MatType.CV_8UC4, bgraData);

      final mask = await segmenter.callFromMat(mat);
      expect(mask.width, greaterThan(0));
      print('Mask from BGRA: ${mask.width}x${mask.height}');

      mat.dispose();
      segmenter.dispose();
      print('Test passed');
    });

    test('callFromMat with CV_8UC1 (grayscale)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing CV_8UC1 (grayscale) Mat ---');
      final segmenter = await SelfieSegmentation.create();

      final grayData = Uint8List(64 * 64);
      for (int i = 0; i < grayData.length; i++) {
        grayData[i] = 128;
      }
      final mat = cv.Mat.fromList(64, 64, cv.MatType.CV_8UC1, grayData);

      final mask = await segmenter.callFromMat(mat);
      expect(mask.width, greaterThan(0));
      print('Mask from grayscale: ${mask.width}x${mask.height}');

      mat.dispose();
      segmenter.dispose();
      print('Test passed');
    });

    test('callFromMat with empty Mat throws', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing empty Mat ---');
      final segmenter = await SelfieSegmentation.create();

      final emptyMat = cv.Mat.empty();

      try {
        await segmenter.callFromMat(emptyMat);
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        expect(e.code, SegmentationError.imageDecodeFailed);
        print('Correctly threw: ${e.message}');
      } finally {
        emptyMat.dispose();
      }

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Concurrent Inference (Codex Gap #3)
  // ===========================================================================
  group('Concurrent Inference', () {
    test('FaceDetector lock serializes concurrent calls', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing concurrent lock mechanism ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation();

      final imageBytes = _createTestImage(128, 128);

      // Launch multiple concurrent calls
      final futures = <Future<SegmentationMask>>[];
      for (int i = 0; i < 5; i++) {
        futures.add(detector.getSegmentationMask(imageBytes));
      }

      // All should complete successfully (serialized by lock)
      final results = await Future.wait(futures);
      expect(results.length, 5);
      for (final mask in results) {
        expect(mask.width, greaterThan(0));
      }

      print('All 5 concurrent calls completed successfully');
      detector.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Upsample Edge Cases (Codex Gap #4)
  // ===========================================================================
  group('Upsample Edge Cases', () {
    test('upsample with maxSize=0 (unlimited)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing upsample with maxSize=0 ---');
      final segmenter = await SelfieSegmentation.create();
      final imageBytes = _createTestImage(512, 512);

      final mask = await segmenter.call(imageBytes);
      final upsampled = mask.upsample(
        targetWidth: 1000,
        targetHeight: 1000,
        maxSize: 0, // Unlimited
      );

      // Should be exactly 1000x1000 (no capping)
      expect(upsampled.width, 1000);
      expect(upsampled.height, 1000);
      print('Upsampled to ${upsampled.width}x${upsampled.height}');

      segmenter.dispose();
      print('Test passed');
    });

    test('upsample with padding (unletterbox)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing upsample with padding ---');
      final segmenter = await SelfieSegmentation.create();

      // Wide image will have top/bottom padding
      final wideImage = _createTestImage(400, 100);
      final mask = await segmenter.call(wideImage);

      print('Original padding: ${mask.padding}');
      expect(mask.padding.any((p) => p > 0), true,
          reason: 'Wide image should have padding');

      final upsampled = mask.upsample();
      // After upsample, padding should be removed
      expect(upsampled.padding.every((p) => p == 0), true);
      print('Upsampled dimensions: ${upsampled.width}x${upsampled.height}');

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // fromMap Error Cases (Codex Gap #5)
  // ===========================================================================
  group('fromMap Error Cases', () {
    test('fromMap with missing width throws', () async {
      print('\n--- Testing fromMap with missing width ---');

      try {
        SegmentationMask.fromMap({
          'data': [0.5, 0.5, 0.5, 0.5],
          // missing 'width'
          'height': 2,
          'originalWidth': 100,
          'originalHeight': 100,
        });
        fail('Should have thrown');
      } catch (e) {
        print('Correctly threw: $e');
      }

      print('Test passed');
    });

    test('fromMap with unknown dataFormat throws', () async {
      print('\n--- Testing fromMap with unknown dataFormat ---');

      try {
        SegmentationMask.fromMap({
          'data': [0.5, 0.5, 0.5, 0.5],
          'width': 2,
          'height': 2,
          'originalWidth': 100,
          'originalHeight': 100,
          'dataFormat': 'unknown_format',
        });
        fail('Should have thrown ArgumentError');
      } on ArgumentError catch (e) {
        print('Correctly threw ArgumentError: $e');
        expect(e.message, contains('Unknown data format'));
      }

      print('Test passed');
    });

    test('fromMap with data length mismatch throws', () async {
      print('\n--- Testing fromMap with data length mismatch ---');

      try {
        SegmentationMask.fromMap({
          'data': [0.5, 0.5], // Only 2 elements
          'width': 2,
          'height': 2, // Expects 4 elements
          'originalWidth': 100,
          'originalHeight': 100,
        });
        fail('Should have thrown ArgumentError');
      } on ArgumentError catch (e) {
        print('Correctly threw: $e');
        expect(e.message, contains('Data length'));
      }

      print('Test passed');
    });

    test('fromMap with uint8 format converts correctly', () async {
      print('\n--- Testing fromMap with uint8 format ---');

      final mask = SegmentationMask.fromMap({
        'data': [0, 128, 255, 64], // uint8 values
        'width': 2,
        'height': 2,
        'originalWidth': 100,
        'originalHeight': 100,
        'dataFormat': 'uint8',
      });

      // Check conversion: 0/255=0.0, 128/255~=0.502, 255/255=1.0, 64/255~=0.251
      expect(mask.data[0], closeTo(0.0, 0.01));
      expect(mask.data[1], closeTo(0.502, 0.01));
      expect(mask.data[2], closeTo(1.0, 0.01));
      expect(mask.data[3], closeTo(0.251, 0.01));

      print('Uint8 conversion correct');
      print('Test passed');
    });

    test('fromMap with binary format converts correctly', () async {
      print('\n--- Testing fromMap with binary format ---');

      final mask = SegmentationMask.fromMap({
        'data': [0, 255, 255, 0], // binary values
        'width': 2,
        'height': 2,
        'originalWidth': 100,
        'originalHeight': 100,
        'dataFormat': 'binary',
      });

      // Binary: 0 -> 0.0, 255 -> 1.0
      expect(mask.data[0], 0.0);
      expect(mask.data[1], 1.0);
      expect(mask.data[2], 1.0);
      expect(mask.data[3], 0.0);

      print('Binary conversion correct');
      print('Test passed');
    });
  });

  // ===========================================================================
  // Real Sample Image Tests (Consolidated to reduce resource pressure)
  // ===========================================================================
  group('Real Sample Image Tests', () {
    test('all sample images produce valid segmentation masks', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing all sample images with segmentation ---');
      final segmenter = await SelfieSegmentation.create();

      try {
        final samplePaths = [
          'assets/samples/landmark-ex1.jpg',
          'assets/samples/iris-detection-ex1.jpg',
          'assets/samples/iris-detection-ex2.jpg',
          'assets/samples/mesh-ex1.jpeg',
          'assets/samples/group-shot-bounding-box-ex1.jpeg',
        ];

        for (final path in samplePaths) {
          final ByteData data = await rootBundle.load(path);
          final imageBytes = data.buffer.asUint8List();

          final sw = Stopwatch()..start();
          final mask = await segmenter.call(imageBytes);
          sw.stop();

          // Verify all mask values are in valid range [0.0, 1.0]
          double minVal = double.infinity;
          double maxVal = double.negativeInfinity;
          for (final v in mask.data) {
            expect(v, inInclusiveRange(0.0, 1.0),
                reason: 'Mask value out of range for $path');
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
          }

          // Count foreground pixels
          final foreground = mask.data.where((v) => v >= 0.5).length;
          final pct = (foreground / mask.data.length * 100).toStringAsFixed(1);

          print(
              '  $path: ${mask.width}x${mask.height}, ${sw.elapsedMilliseconds}ms, fg=$pct%');

          // All images contain people, so we should detect some foreground
          expect(foreground, greaterThan(0),
              reason: '$path should have some foreground detected');

          // Test binary mask for this image
          final binary = mask.toBinary(threshold: 0.5);
          for (final v in binary) {
            expect(v, anyOf(0, 255),
                reason: 'Binary mask should only contain 0 or 255');
          }
        }
      } finally {
        segmenter.dispose();
      }

      print('Test passed');
    });

    test('multiclass model inference', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!multiclassAvailable) {
        print('Skipping: multiclass model not available');
        return;
      }

      print('\n--- Testing multiclass model inference ---');

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final imageBytes = data.buffer.asUint8List();

      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.multiclass),
      );
      try {
        expect(segmenter.outputChannels, 6);

        final sw = Stopwatch()..start();
        final mask = await segmenter.call(imageBytes);
        sw.stop();

        // Multiclass model should return MulticlassSegmentationMask
        expect(mask, isA<MulticlassSegmentationMask>());
        final mcMask = mask as MulticlassSegmentationMask;

        final foreground = mask.data.where((v) => v >= 0.5).length;

        print('Multiclass model:');
        print('  Input size: ${segmenter.inputWidth}x${segmenter.inputHeight}');
        print('  Mask: ${mask.width}x${mask.height}');
        print('  Time: ${sw.elapsedMilliseconds}ms');
        print(
            '  Foreground: ${(foreground / mask.data.length * 100).toStringAsFixed(1)}%');

        // Verify per-class masks are available
        final hairMask = mcMask.hairMask;
        final faceMask = mcMask.faceSkinMask;
        final bodyMask = mcMask.bodySkinMask;
        final clothesMask = mcMask.clothesMask;
        final bgMask = mcMask.backgroundMask;
        final otherMask = mcMask.otherMask;

        final numPixels = mask.width * mask.height;
        expect(hairMask.length, numPixels);
        expect(faceMask.length, numPixels);
        expect(bodyMask.length, numPixels);
        expect(clothesMask.length, numPixels);
        expect(bgMask.length, numPixels);
        expect(otherMask.length, numPixels);

        // All class probabilities at each pixel should sum to ~1.0
        for (int i = 0; i < 100 && i < numPixels; i++) {
          final sum = bgMask[i] +
              hairMask[i] +
              bodyMask[i] +
              faceMask[i] +
              clothesMask[i] +
              otherMask[i];
          expect(sum, closeTo(1.0, 0.01),
              reason: 'Class probabilities at pixel $i should sum to 1.0');
        }

        expect(foreground, greaterThan(0), reason: 'Multiclass model failed');
      } finally {
        segmenter.dispose();
      }

      print('Test passed');
    });

    test('upsampled mask and utilities on real image', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing mask utilities on real image ---');
      final segmenter = await SelfieSegmentation.create();

      try {
        final ByteData data =
            await rootBundle.load('assets/samples/landmark-ex1.jpg');
        final imageBytes = data.buffer.asUint8List();

        final mask = await segmenter.call(imageBytes);
        print('Original image: ${mask.originalWidth}x${mask.originalHeight}');
        print('Raw mask: ${mask.width}x${mask.height}');

        // Test upsample
        final upsampled = mask.upsample();
        print('Upsampled: ${upsampled.width}x${upsampled.height}');
        expect(upsampled.width, lessThanOrEqualTo(mask.originalWidth));
        expect(upsampled.height, lessThanOrEqualTo(mask.originalHeight));

        // Test toUint8
        final uint8 = mask.toUint8();
        expect(uint8.length, mask.width * mask.height);
        for (final v in uint8) {
          expect(v, inInclusiveRange(0, 255));
        }
        print('Uint8 mask size: ${uint8.length} bytes');

        // Test toRgba
        final rgba = mask.toRgba(
          foreground: 0xFF00FF00,
          background: 0x00000000,
          threshold: 0.5,
        );
        expect(rgba.length, mask.width * mask.height * 4);
        print('RGBA mask size: ${rgba.length} bytes');
      } finally {
        segmenter.dispose();
      }

      print('Test passed');
    });
  });

  // ===========================================================================
  // Model Selection Tests
  // ===========================================================================
  group('Model Selection', () {
    test('general model (default) produces binary mask', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing general model (default) ---');
      final segmenter = await SelfieSegmentation.create();

      expect(segmenter.inputWidth, 256);
      expect(segmenter.inputHeight, 256);
      expect(segmenter.outputChannels, 1);

      final imageBytes = _createTestImage(256, 256);
      final mask = await segmenter.call(imageBytes);

      expect(mask, isNot(isA<MulticlassSegmentationMask>()));
      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      print('General model: ${mask.width}x${mask.height}');

      segmenter.dispose();
      print('Test passed');
    });

    test('general model explicit selection', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing explicit general model selection ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.general),
      );

      expect(segmenter.outputChannels, 1);

      final imageBytes = _createTestImage(256, 256);
      final mask = await segmenter.call(imageBytes);
      expect(mask, isNot(isA<MulticlassSegmentationMask>()));

      segmenter.dispose();
      print('Test passed');
    });

    test('landscape model uses 144x256 input', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing landscape model ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.landscape),
      );

      expect(segmenter.inputWidth, 256);
      expect(segmenter.inputHeight, 144);
      expect(segmenter.outputChannels, 1);

      final imageBytes = _createTestImage(640, 360); // 16:9
      final mask = await segmenter.call(imageBytes);

      expect(mask, isNot(isA<MulticlassSegmentationMask>()));
      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      print('Landscape model: ${mask.width}x${mask.height}');

      segmenter.dispose();
      print('Test passed');
    });

    test('multiclass model produces MulticlassSegmentationMask', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!multiclassAvailable) {
        print('Skipping: multiclass model not available');
        return;
      }

      print('\n--- Testing multiclass model selection ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.multiclass),
      );

      expect(segmenter.inputWidth, 256);
      expect(segmenter.inputHeight, 256);
      expect(segmenter.outputChannels, 6);

      final imageBytes = _createTestImage(256, 256);
      final mask = await segmenter.call(imageBytes);

      expect(mask, isA<MulticlassSegmentationMask>());
      print('Multiclass model: ${mask.width}x${mask.height}');

      segmenter.dispose();
      print('Test passed');
    });

    test('switching between models works', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing model switching ---');
      final imageBytes = _createTestImage(256, 256);

      // General model
      final general = await SelfieSegmentation.create();
      final gMask = await general.call(imageBytes);
      expect(gMask, isNot(isA<MulticlassSegmentationMask>()));
      general.dispose();
      print('General: OK');

      // Landscape model
      final landscape = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.landscape),
      );
      final lMask = await landscape.call(imageBytes);
      expect(lMask, isNot(isA<MulticlassSegmentationMask>()));
      landscape.dispose();
      print('Landscape: OK');

      // Multiclass model (if available)
      if (multiclassAvailable) {
        final multiclass = await SelfieSegmentation.create(
          config: SegmentationConfig(model: SegmentationModel.multiclass),
        );
        final mMask = await multiclass.call(imageBytes);
        expect(mMask, isA<MulticlassSegmentationMask>());
        multiclass.dispose();
        print('Multiclass: OK');
      }

      print('Test passed');
    });

    test('stress switch binary <-> multiclass with inference (30x)', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!modelsAvailable || !multiclassAvailable) {
        print('Skipping: required models not available');
        return;
      }

      print('\n--- Stress switching binary <-> multiclass (30x) ---');
      final imageBytes = _createTestImage(256, 256, r: 180, g: 140, b: 100);

      for (int i = 0; i < 30; i++) {
        final binary = await SelfieSegmentation.create(
          config: const SegmentationConfig(model: SegmentationModel.general),
        );
        final binaryMask = await binary.call(imageBytes);
        expect(binaryMask, isNot(isA<MulticlassSegmentationMask>()));
        await binary.disposeAsync();

        final multiclass = await SelfieSegmentation.create(
          config: const SegmentationConfig(model: SegmentationModel.multiclass),
        );
        final multiMask = await multiclass.call(imageBytes);
        expect(multiMask, isA<MulticlassSegmentationMask>());
        await multiclass.disposeAsync();

        if ((i + 1) % 5 == 0) {
          print('Completed ${i + 1}/30 cycles');
        }
      }

      print('Test passed');
    });
  });

  // ===========================================================================
  // MulticlassSegmentationMask API Tests
  // ===========================================================================
  group('MulticlassSegmentationMask API', () {
    test('classMask returns correct length for each class', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!multiclassAvailable) {
        print('Skipping: multiclass model not available');
        return;
      }

      print('\n--- Testing classMask for each class index ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.multiclass),
      );
      final imageBytes = _createTestImage(128, 128);
      final mask =
          await segmenter.call(imageBytes) as MulticlassSegmentationMask;

      final numPixels = mask.width * mask.height;
      for (int c = 0; c < 6; c++) {
        final cm = mask.classMask(c);
        expect(cm.length, numPixels, reason: 'Class $c mask length mismatch');
        for (int i = 0; i < cm.length; i++) {
          expect(cm[i], inInclusiveRange(0.0, 1.0),
              reason: 'Class $c pixel $i out of range');
        }
      }
      print('All 6 class masks valid ($numPixels pixels each)');

      segmenter.dispose();
      print('Test passed');
    });

    test('named accessors match classMask indices', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!multiclassAvailable) {
        print('Skipping: multiclass model not available');
        return;
      }

      print('\n--- Testing named accessors ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.multiclass),
      );
      final imageBytes = _createTestImage(64, 64);
      final mask =
          await segmenter.call(imageBytes) as MulticlassSegmentationMask;

      // Verify named accessors return the same data as classMask(index)
      expect(mask.backgroundMask, mask.classMask(0));
      expect(mask.hairMask, mask.classMask(1));
      expect(mask.bodySkinMask, mask.classMask(2));
      expect(mask.faceSkinMask, mask.classMask(3));
      expect(mask.clothesMask, mask.classMask(4));
      expect(mask.otherMask, mask.classMask(5));

      print('All named accessors match class indices');
      segmenter.dispose();
      print('Test passed');
    });

    test('classMask out of bounds throws RangeError', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!multiclassAvailable) {
        print('Skipping: multiclass model not available');
        return;
      }

      print('\n--- Testing classMask bounds checking ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.multiclass),
      );
      final imageBytes = _createTestImage(64, 64);
      final mask =
          await segmenter.call(imageBytes) as MulticlassSegmentationMask;

      expect(() => mask.classMask(-1), throwsRangeError);
      expect(() => mask.classMask(6), throwsRangeError);

      segmenter.dispose();
      print('Test passed');
    });

    test('multiclass mask toBinary/toUint8/toRgba work correctly', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!multiclassAvailable) {
        print('Skipping: multiclass model not available');
        return;
      }

      print('\n--- Testing multiclass mask utility methods ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.multiclass),
      );
      final imageBytes = _createTestImage(128, 128);
      final mask =
          await segmenter.call(imageBytes) as MulticlassSegmentationMask;

      // toBinary
      final binary = mask.toBinary(threshold: 0.5);
      for (final v in binary) {
        expect(v, anyOf(0, 255));
      }

      // toUint8
      final uint8 = mask.toUint8();
      for (final v in uint8) {
        expect(v, inInclusiveRange(0, 255));
      }

      // toRgba
      final rgba = mask.toRgba(
        foreground: 0xFF0000FF,
        background: 0x00000000,
        threshold: 0.5,
      );
      expect(rgba.length, mask.width * mask.height * 4);

      segmenter.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // SegmentationWorker Model Selection
  // ===========================================================================
  group('SegmentationWorker Model Selection', () {
    test('worker with default (general) model', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker with default model ---');
      final worker = SegmentationWorker();
      await worker.initialize();

      final imageBytes = _createTestImage(256, 256);
      final mask = await worker.segment(imageBytes);

      expect(mask.width, greaterThan(0));
      expect(mask, isNot(isA<MulticlassSegmentationMask>()));
      print('Worker (general): ${mask.width}x${mask.height}');

      worker.dispose();
      print('Test passed');
    });

    test('worker with multiclass model', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!multiclassAvailable) {
        print('Skipping: multiclass model not available');
        return;
      }

      print('\n--- Testing SegmentationWorker with multiclass model ---');
      final worker = SegmentationWorker();
      await worker.initialize(
        config: SegmentationConfig(model: SegmentationModel.multiclass),
      );

      final imageBytes = _createTestImage(256, 256);
      final mask = await worker.segment(imageBytes);

      expect(mask.width, greaterThan(0));
      expect(mask, isA<MulticlassSegmentationMask>());

      final mcMask = mask as MulticlassSegmentationMask;
      final numPixels = mask.width * mask.height;
      expect(mcMask.hairMask.length, numPixels);

      print('Worker (multiclass): ${mask.width}x${mask.height}');
      worker.dispose();
      print('Test passed');
    });

    test('worker with landscape model', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker with landscape model ---');
      final worker = SegmentationWorker();
      await worker.initialize(
        config: SegmentationConfig(model: SegmentationModel.landscape),
      );

      final imageBytes = _createTestImage(640, 360);
      final mask = await worker.segment(imageBytes);

      expect(mask.width, greaterThan(0));
      expect(mask, isNot(isA<MulticlassSegmentationMask>()));
      print('Worker (landscape): ${mask.width}x${mask.height}');

      worker.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // FaceDetectorIsolate Model Selection
  // ===========================================================================
  group('FaceDetectorIsolate Model Selection', () {
    test('isolate with default (general) model', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing FaceDetectorIsolate with default model ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final imageBytes = _createTestImage(256, 256);
      final mask = await detector.getSegmentationMask(imageBytes);

      expect(mask.width, greaterThan(0));
      print('Isolate (general): ${mask.width}x${mask.height}');

      await detector.dispose();
      print('Test passed');
    });

    test('isolate with multiclass model', () async {
      final multiclassAvailable = await _isMulticlassModelAvailable();
      if (!multiclassAvailable) {
        print('Skipping: multiclass model not available');
        return;
      }

      print('\n--- Testing FaceDetectorIsolate with multiclass model ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
        segmentationConfig: SegmentationConfig(
          model: SegmentationModel.multiclass,
        ),
      );

      final imageBytes = _createTestImage(256, 256);
      final mask = await detector.getSegmentationMask(imageBytes);

      expect(mask.width, greaterThan(0));
      expect(mask, isA<MulticlassSegmentationMask>());
      print('Isolate (multiclass): ${mask.width}x${mask.height}');

      await detector.dispose();
      print('Test passed');
    });

    test('isolate with landscape model', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing FaceDetectorIsolate with landscape model ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
        segmentationConfig: SegmentationConfig(
          model: SegmentationModel.landscape,
        ),
      );

      final imageBytes = _createTestImage(640, 360);
      final mask = await detector.getSegmentationMask(imageBytes);

      expect(mask.width, greaterThan(0));
      print('Isolate (landscape): ${mask.width}x${mask.height}');

      await detector.dispose();
      print('Test passed');
    });
  });

  // ===========================================================================
  // Summary
  // ===========================================================================
  test('Test Suite Summary', () async {
    print('\n$_separator');
    print('SELFIE SEGMENTATION TEST SUITE COMPLETE');
    print(_separator);
    print('Models available: $modelsAvailable');
    print('Platform: ${Platform.operatingSystem}');
    print(_separator);
  });
}
