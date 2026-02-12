// ignore_for_file: avoid_print

/// Integration tests for untested segmentation-related public methods.
///
/// Tests cover:
/// - FaceDetector.getSegmentationMaskFromMat()
/// - FaceDetectorIsolate.getSegmentationMaskFromMat()
/// - FaceDetectorIsolate.detectFacesWithSegmentation()
/// - FaceDetectorIsolate.detectFacesWithSegmentationFromMat()
/// - SegmentationWorker.segmentMat()
library;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
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

  setUpAll(() async {
    modelsAvailable = await checkModels();
    final data = await rootBundle.load('assets/samples/landmark-ex1.jpg');
    imageBytes = data.buffer.asUint8List();
  });

  // ===========================================================================
  // FaceDetector.getSegmentationMaskFromMat
  // ===========================================================================
  group('FaceDetector.getSegmentationMaskFromMat', () {
    test('returns valid mask from cv.Mat input', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing FaceDetector.getSegmentationMaskFromMat ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation();

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      final mask = await detector.getSegmentationMaskFromMat(mat);
      mat.dispose();

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      expect(mask.data.length, mask.width * mask.height);
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      print('Mask size: ${mask.width}x${mask.height}');
      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('output consistent with getSegmentationMask(bytes)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing consistency: Mat vs bytes ---');
      final detector = FaceDetector();
      await detector.initialize();
      await detector.initializeSegmentation();

      final maskFromBytes = await detector.getSegmentationMask(imageBytes);

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      final maskFromMat = await detector.getSegmentationMaskFromMat(mat);
      mat.dispose();

      expect(maskFromMat.width, maskFromBytes.width);
      expect(maskFromMat.height, maskFromBytes.height);

      print(
        'Bytes mask: ${maskFromBytes.width}x${maskFromBytes.height}, '
        'Mat mask: ${maskFromMat.width}x${maskFromMat.height}',
      );
      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('throws StateError before initializeSegmentation', () async {
      print('\n--- Testing StateError without segmentation ---');
      final detector = FaceDetector();
      await detector.initialize();

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      try {
        await detector.getSegmentationMaskFromMat(mat);
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw: ${e.message}');
      } finally {
        mat.dispose();
      }

      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // FaceDetectorIsolate.getSegmentationMaskFromMat
  // ===========================================================================
  group('FaceDetectorIsolate.getSegmentationMaskFromMat', () {
    test('returns valid mask with float32 format', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing Isolate.getSegmentationMaskFromMat (float32) ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      final mask = await detector.getSegmentationMaskFromMat(mat);
      mat.dispose();

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      expect(mask.data.length, mask.width * mask.height);
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      print('Mask size: ${mask.width}x${mask.height}');
      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('returns valid mask with uint8 format', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing Isolate.getSegmentationMaskFromMat (uint8) ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      final mask = await detector.getSegmentationMaskFromMat(
        mat,
        outputFormat: IsolateOutputFormat.uint8,
      );
      mat.dispose();

      expect(mask.width, greaterThan(0));
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('returns valid mask with binary format', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing Isolate.getSegmentationMaskFromMat (binary) ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      final mask = await detector.getSegmentationMaskFromMat(
        mat,
        outputFormat: IsolateOutputFormat.binary,
        binaryThreshold: 0.5,
      );
      mat.dispose();

      for (int i = 0; i < mask.data.length; i++) {
        expect(mask.data[i], anyOf(0.0, 1.0));
      }

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('throws StateError when spawned without segmentation', () async {
      print('\n--- Testing Isolate.getSegmentationMaskFromMat StateError ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: false,
      );

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      try {
        await detector.getSegmentationMaskFromMat(mat);
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw: ${e.message}');
      } finally {
        mat.dispose();
      }

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // FaceDetectorIsolate.detectFacesWithSegmentation
  // ===========================================================================
  group('FaceDetectorIsolate.detectFacesWithSegmentation', () {
    test('returns faces and segmentation mask from bytes', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing detectFacesWithSegmentation ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final result = await detector.detectFacesWithSegmentation(imageBytes);

      expect(result.faces, isNotEmpty);
      expect(result.segmentationMask, isNotNull);
      expect(result.segmentationMask!.width, greaterThan(0));
      expect(result.segmentationMask!.height, greaterThan(0));
      expect(result.detectionTimeMs, greaterThanOrEqualTo(0));
      expect(result.segmentationTimeMs, greaterThanOrEqualTo(0));
      expect(result.totalTimeMs, greaterThanOrEqualTo(0));

      // Verify mask values are valid
      final mask = result.segmentationMask!;
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      print(
        'Faces: ${result.faces.length}, '
        'Mask: ${mask.width}x${mask.height}, '
        'Detection: ${result.detectionTimeMs}ms, '
        'Segmentation: ${result.segmentationTimeMs}ms, '
        'Total: ${result.totalTimeMs}ms',
      );

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('works with uint8 output format', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing detectFacesWithSegmentation (uint8) ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final result = await detector.detectFacesWithSegmentation(
        imageBytes,
        outputFormat: IsolateOutputFormat.uint8,
      );

      expect(result.faces, isNotEmpty);
      expect(result.segmentationMask, isNotNull);
      for (int i = 0;
          i < 100 && i < result.segmentationMask!.data.length;
          i++) {
        expect(result.segmentationMask!.data[i], inInclusiveRange(0.0, 1.0));
      }

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('works with binary output format', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing detectFacesWithSegmentation (binary) ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final result = await detector.detectFacesWithSegmentation(
        imageBytes,
        outputFormat: IsolateOutputFormat.binary,
        binaryThreshold: 0.5,
      );

      expect(result.faces, isNotEmpty);
      expect(result.segmentationMask, isNotNull);
      for (final v in result.segmentationMask!.data) {
        expect(v, anyOf(0.0, 1.0));
      }

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('works with fast detection mode', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing detectFacesWithSegmentation (fast mode) ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final result = await detector.detectFacesWithSegmentation(
        imageBytes,
        mode: FaceDetectionMode.fast,
      );

      expect(result.faces, isNotEmpty);
      expect(result.segmentationMask, isNotNull);
      // Fast mode: no mesh
      expect(result.faces.first.mesh, isNull);

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('throws StateError when spawned without segmentation', () async {
      print('\n--- Testing detectFacesWithSegmentation StateError ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: false,
      );

      try {
        await detector.detectFacesWithSegmentation(imageBytes);
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw: ${e.message}');
      }

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // FaceDetectorIsolate.detectFacesWithSegmentationFromMat
  // ===========================================================================
  group('FaceDetectorIsolate.detectFacesWithSegmentationFromMat', () {
    test('returns faces and mask from cv.Mat', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing detectFacesWithSegmentationFromMat ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      final result = await detector.detectFacesWithSegmentationFromMat(mat);
      mat.dispose();

      expect(result.faces, isNotEmpty);
      expect(result.segmentationMask, isNotNull);
      expect(result.segmentationMask!.width, greaterThan(0));
      expect(result.detectionTimeMs, greaterThanOrEqualTo(0));
      expect(result.segmentationTimeMs, greaterThanOrEqualTo(0));

      print(
        'Faces: ${result.faces.length}, '
        'Mask: ${result.segmentationMask!.width}x${result.segmentationMask!.height}',
      );

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('face results consistent with detectFaces', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing consistency: combined vs separate ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);

      final facesOnly = await detector.detectFacesFromMat(
        mat,
        mode: FaceDetectionMode.full,
      );
      final combined = await detector.detectFacesWithSegmentationFromMat(
        mat,
        mode: FaceDetectionMode.full,
      );
      mat.dispose();

      expect(combined.faces.length, facesOnly.length);

      // Bounding boxes should be close (within 5 pixels)
      if (facesOnly.isNotEmpty) {
        final bb1 = facesOnly.first.boundingBox;
        final bb2 = combined.faces.first.boundingBox;
        expect((bb1.topLeft.x - bb2.topLeft.x).abs(), lessThan(5));
        expect((bb1.topLeft.y - bb2.topLeft.y).abs(), lessThan(5));
      }

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('throws StateError when spawned without segmentation', () async {
      print(
        '\n--- Testing detectFacesWithSegmentationFromMat StateError ---',
      );
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: false,
      );

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      try {
        await detector.detectFacesWithSegmentationFromMat(mat);
        fail('Should have thrown StateError');
      } on StateError catch (e) {
        print('Correctly threw: ${e.message}');
      } finally {
        mat.dispose();
      }

      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // SegmentationWorker.segmentMat
  // ===========================================================================
  group('SegmentationWorker.segmentMat', () {
    test('returns valid mask from cv.Mat input', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker.segmentMat ---');
      final worker = SegmentationWorker();
      await worker.initialize();

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      final mask = await worker.segmentMat(mat);
      mat.dispose();

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      expect(mask.data.length, mask.width * mask.height);
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      print('Mask size: ${mask.width}x${mask.height}');
      worker.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('output consistent with segment(bytes)', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker: Mat vs bytes ---');
      final worker = SegmentationWorker();
      await worker.initialize();

      final maskFromBytes = await worker.segment(imageBytes);

      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      final maskFromMat = await worker.segmentMat(mat);
      mat.dispose();

      expect(maskFromMat.width, maskFromBytes.width);
      expect(maskFromMat.height, maskFromBytes.height);

      print(
        'Bytes: ${maskFromBytes.width}x${maskFromBytes.height}, '
        'Mat: ${maskFromMat.width}x${maskFromMat.height}',
      );
      worker.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('throws SegmentationException on empty Mat', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing SegmentationWorker.segmentMat empty Mat ---');
      final worker = SegmentationWorker();
      await worker.initialize();

      final emptyMat = cv.Mat.empty();
      try {
        await worker.segmentMat(emptyMat);
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        print('Correctly threw: ${e.message}');
      } finally {
        emptyMat.dispose();
      }

      worker.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });
}
