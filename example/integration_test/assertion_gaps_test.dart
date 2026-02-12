// ignore_for_file: avoid_print

/// Integration tests for assertion gaps and under-covered areas.
///
/// Tests cover:
/// - Embedding dimension validation (kEmbeddingDimension == 192)
/// - FaceEmbedding static methods (cosineSimilarity, euclideanDistance)
/// - Public constants
/// - Mixed concurrent detection + segmentation
/// - Segmentation error recovery
/// - SelfieSegmentation landscape model + callFromMat
library;

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testTimeout = Timeout(Duration(minutes: 5));

  late Uint8List landmarkBytes;
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

  Uint8List createTestImage(int width, int height,
      {int r = 128, int g = 128, int b = 128}) {
    final image = img.Image(width: width, height: height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        image.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return Uint8List.fromList(img.encodePng(image));
  }

  setUpAll(() async {
    modelsAvailable = await checkModels();
    final data1 = await rootBundle.load('assets/samples/landmark-ex1.jpg');
    landmarkBytes = data1.buffer.asUint8List();
  });

  // ===========================================================================
  // Embedding dimension validation
  // ===========================================================================
  group('Embedding dimension validation', () {
    test('FaceDetector embedding has exactly kEmbeddingDimension elements',
        () async {
      print('\n--- Testing embedding dimension ---');
      final detector = FaceDetector();
      await detector.initialize();

      final faces = await detector.detectFaces(landmarkBytes);
      expect(faces, isNotEmpty);

      final embedding =
          await detector.getFaceEmbedding(faces.first, landmarkBytes);
      expect(embedding.length, kEmbeddingDimension);
      expect(embedding.length, 192);

      print(
          'Embedding length: ${embedding.length} (expected $kEmbeddingDimension)');
      detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);

    test('FaceDetectorIsolate embedding has exactly 192 elements', () async {
      print('\n--- Testing isolate embedding dimension ---');
      final detector = await FaceDetectorIsolate.spawn();

      final faces = await detector.detectFaces(landmarkBytes);
      expect(faces, isNotEmpty);

      final embedding =
          await detector.getFaceEmbedding(faces.first, landmarkBytes);
      expect(embedding.length, kEmbeddingDimension);
      expect(embedding.length, 192);

      print('Isolate embedding length: ${embedding.length}');
      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // FaceEmbedding static methods
  // ===========================================================================
  group('FaceEmbedding static methods', () {
    test('cosineSimilarity returns 1.0 for identical vectors', () {
      final vec = Float32List.fromList([0.5, 0.5, 0.5, 0.5]);
      final similarity = FaceEmbedding.cosineSimilarity(vec, vec);
      expect(similarity, closeTo(1.0, 0.001));
      print('cosineSimilarity(v, v) = $similarity');
    }, timeout: testTimeout);

    test('cosineSimilarity returns ~0 for orthogonal vectors', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0, 0.0]);
      final b = Float32List.fromList([0.0, 1.0, 0.0, 0.0]);
      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      expect(similarity, closeTo(0.0, 0.01));
      print('cosineSimilarity(orthogonal) = $similarity');
    }, timeout: testTimeout);

    test('cosineSimilarity returns -1.0 for opposite vectors', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0]);
      final b = Float32List.fromList([-1.0, 0.0, 0.0]);
      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      expect(similarity, closeTo(-1.0, 0.01));
      print('cosineSimilarity(opposite) = $similarity');
    }, timeout: testTimeout);

    test('cosineSimilarity throws ArgumentError for mismatched dimensions', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b = Float32List.fromList([1.0, 0.0, 0.0]);
      expect(
        () => FaceEmbedding.cosineSimilarity(a, b),
        throwsA(isA<ArgumentError>()),
      );
      print('cosineSimilarity: dimension mismatch throws correctly');
    }, timeout: testTimeout);

    test('euclideanDistance returns 0.0 for identical vectors', () {
      final vec = Float32List.fromList([0.5, 0.5, 0.5, 0.5]);
      final distance = FaceEmbedding.euclideanDistance(vec, vec);
      expect(distance, closeTo(0.0, 0.001));
      print('euclideanDistance(v, v) = $distance');
    }, timeout: testTimeout);

    test('euclideanDistance returns correct value for known vectors', () {
      final a = Float32List.fromList([0.0, 0.0, 0.0]);
      final b = Float32List.fromList([3.0, 4.0, 0.0]);
      final distance = FaceEmbedding.euclideanDistance(a, b);
      expect(distance, closeTo(5.0, 0.001));
      print('euclideanDistance([0,0,0], [3,4,0]) = $distance');
    }, timeout: testTimeout);

    test('euclideanDistance throws ArgumentError for mismatched dimensions',
        () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b = Float32List.fromList([1.0, 0.0, 0.0]);
      expect(
        () => FaceEmbedding.euclideanDistance(a, b),
        throwsA(isA<ArgumentError>()),
      );
      print('euclideanDistance: dimension mismatch throws correctly');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // Public constants
  // ===========================================================================
  group('Public constants', () {
    test('kMeshPoints is 468', () {
      expect(kMeshPoints, 468);
      print('kMeshPoints = $kMeshPoints');
    }, timeout: testTimeout);

    test('kEmbeddingDimension is 192', () {
      expect(kEmbeddingDimension, 192);
      print('kEmbeddingDimension = $kEmbeddingDimension');
    }, timeout: testTimeout);

    test('kEmbeddingInputSize is 112', () {
      expect(kEmbeddingInputSize, 112);
      print('kEmbeddingInputSize = $kEmbeddingInputSize');
    }, timeout: testTimeout);

    test('kMaxEyeLandmark is 15', () {
      expect(kMaxEyeLandmark, 15);
      print('kMaxEyeLandmark = $kMaxEyeLandmark');
    }, timeout: testTimeout);

    test('kMinSegmentationInputSize is 16', () {
      expect(kMinSegmentationInputSize, 16);
      print('kMinSegmentationInputSize = $kMinSegmentationInputSize');
    }, timeout: testTimeout);

    test('eyeLandmarkConnections has 15 entries', () {
      expect(eyeLandmarkConnections.length, 15);
      // Each connection should be a pair
      for (final conn in eyeLandmarkConnections) {
        expect(conn.length, 2);
      }
      print(
        'eyeLandmarkConnections: ${eyeLandmarkConnections.length} connections',
      );
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // Mixed concurrent detection + segmentation
  // ===========================================================================
  group('Mixed concurrent detection + segmentation', () {
    test('concurrent detection and segmentation on same isolate', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing mixed concurrent operations ---');
      final detector = await FaceDetectorIsolate.spawn(
        withSegmentation: true,
      );

      final futures = <Future>[];
      for (int i = 0; i < 5; i++) {
        futures.add(detector.detectFaces(
          landmarkBytes,
          mode: FaceDetectionMode.fast,
        ));
        futures.add(detector.getSegmentationMask(landmarkBytes));
      }

      final results = await Future.wait(futures);

      // 10 results total: 5 face lists + 5 masks alternating
      expect(results.length, 10);
      for (int i = 0; i < 10; i++) {
        if (i.isEven) {
          // Face detection result
          final faces = results[i] as List<Face>;
          expect(faces, isNotEmpty);
        } else {
          // Segmentation result
          final mask = results[i] as SegmentationMask;
          expect(mask.width, greaterThan(0));
        }
      }

      print('All 10 concurrent mixed operations succeeded');
      await detector.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // Segmentation error recovery
  // ===========================================================================
  group('Segmentation error recovery', () {
    test('segmentation works after SegmentationException', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing segmentation error recovery ---');
      final segmenter = await SelfieSegmentation.create();

      // Trigger SegmentationException with corrupted bytes
      try {
        await segmenter.call(Uint8List.fromList([0, 1, 2, 3, 4]));
        fail('Should have thrown SegmentationException');
      } on SegmentationException catch (e) {
        print('Got expected error: ${e.code}');
      }

      // Subsequent valid call should succeed
      final imageBytes = createTestImage(256, 256);
      final mask = await segmenter.call(imageBytes);

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      print('Recovery successful: ${mask.width}x${mask.height}');
      segmenter.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // SelfieSegmentation landscape model + callFromMat
  // ===========================================================================
  group('SelfieSegmentation landscape model + callFromMat', () {
    test('landscape model produces valid mask from Mat', () async {
      if (!modelsAvailable) {
        print('Skipping: models not available');
        return;
      }

      print('\n--- Testing landscape model + callFromMat ---');
      final segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: SegmentationModel.landscape),
      );

      expect(segmenter.inputWidth, 256);
      expect(segmenter.inputHeight, 144);

      // Create a landscape Mat
      final mat = cv.Mat.zeros(360, 640, cv.MatType.CV_8UC3);
      mat.setTo(cv.Scalar(128, 128, 128, 255));

      final mask = await segmenter.callFromMat(mat);
      mat.dispose();

      expect(mask.width, greaterThan(0));
      expect(mask.height, greaterThan(0));
      expect(mask.data.length, mask.width * mask.height);
      for (int i = 0; i < 100 && i < mask.data.length; i++) {
        expect(mask.data[i], inInclusiveRange(0.0, 1.0));
      }

      print('Landscape + Mat mask: ${mask.width}x${mask.height}');
      segmenter.dispose();
      print('Test passed');
    }, timeout: testTimeout);
  });
}
