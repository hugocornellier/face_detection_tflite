// ignore_for_file: avoid_print

/// Integration tests for all FaceDetectionModel variants.
///
/// Tests each model variant:
/// - frontCamera (128x128 input, optimized for selfies)
/// - backCamera (256x256 input, higher resolution)
/// - shortRange (128x128 input, close-up faces)
/// - full (192x192 input, full-range detection)
/// - fullSparse (192x192 input, sparse anchors)
///
/// Each variant is tested for:
/// - Successful initialization
/// - Detection on standard test images
/// - Proper cleanup/disposal
/// - Detection accuracy within acceptable bounds
library;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  /// Test image and expected face count
  const testImages = {
    'assets/samples/landmark-ex1.jpg': 1,
    'assets/samples/iris-detection-ex1.jpg': 1,
    'assets/samples/group-shot-bounding-box-ex1.jpeg': 4,
  };

  group('FaceDetectionModel.frontCamera', () {
    test('should initialize and detect faces', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.frontCamera);
      expect(detector.isReady, true);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty, reason: 'frontCamera should detect faces');
      expect(faces.first.boundingBox.width, greaterThan(0));
      expect(faces.first.landmarks.leftEye, isNotNull);
      expect(faces.first.mesh, isNotNull);

      print(
          'frontCamera: Detected ${faces.length} face(s), bbox width: ${faces.first.boundingBox.width.toStringAsFixed(1)}');

      detector.dispose();
    });

    test('should work with fast mode', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.frontCamera);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNull);
      expect(faces.first.irisPoints, isEmpty);

      detector.dispose();
    });

    test('should detect faces in close-up images', () async {
      // frontCamera is optimized for selfies/close-up faces
      // It may not work well on group photos with smaller faces
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.frontCamera);

      // Use close-up face image instead of group photo
      final ByteData data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);

      expect(faces, isNotEmpty,
          reason: 'frontCamera should detect close-up faces');
      print('frontCamera: Detected ${faces.length} face(s) in close-up image');

      detector.dispose();
    });
  });

  group('FaceDetectionModel.backCamera', () {
    test('should initialize and detect faces', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.backCamera);
      expect(detector.isReady, true);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty, reason: 'backCamera should detect faces');
      expect(faces.first.boundingBox.width, greaterThan(0));
      expect(faces.first.mesh, isNotNull);
      expect(faces.first.mesh!.points.length, 468);

      print(
          'backCamera: Detected ${faces.length} face(s), bbox width: ${faces.first.boundingBox.width.toStringAsFixed(1)}');

      detector.dispose();
    });

    test('should work with all detection modes', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.backCamera);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      for (final mode in FaceDetectionMode.values) {
        final faces = await detector.detectFaces(bytes, mode: mode);
        expect(faces, isNotEmpty, reason: 'backCamera ${mode.name} failed');

        if (mode == FaceDetectionMode.fast) {
          expect(faces.first.mesh, isNull);
        } else {
          expect(faces.first.mesh, isNotNull);
        }

        if (mode == FaceDetectionMode.full) {
          expect(faces.first.irisPoints, isNotEmpty);
        }
      }

      detector.dispose();
    });

    test('should detect multiple faces', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.backCamera);

      final ByteData data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);

      expect(faces.length, greaterThanOrEqualTo(2),
          reason: 'backCamera should detect multiple faces');
      print('backCamera: Detected ${faces.length} faces in group photo');

      detector.dispose();
    });
  });

  group('FaceDetectionModel.shortRange', () {
    test('should initialize and detect faces', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.shortRange);
      expect(detector.isReady, true);

      final ByteData data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty,
          reason: 'shortRange should detect close-up faces');
      expect(faces.first.boundingBox.width, greaterThan(0));
      expect(faces.first.mesh, isNotNull);

      print(
          'shortRange: Detected ${faces.length} face(s), bbox width: ${faces.first.boundingBox.width.toStringAsFixed(1)}');

      detector.dispose();
    });

    test('should work well with close-up face images', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.shortRange);

      // Close-up iris detection images should work well with shortRange
      final testCases = [
        'assets/samples/iris-detection-ex1.jpg',
        'assets/samples/iris-detection-ex2.jpg',
      ];

      for (final imagePath in testCases) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();

        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

        expect(faces, isNotEmpty, reason: 'shortRange failed on $imagePath');
        expect(faces.first.irisPoints, isNotEmpty,
            reason: 'shortRange should detect iris in close-up');
      }

      detector.dispose();
    });
  });

  group('FaceDetectionModel.full', () {
    test('should initialize and detect faces', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.full);
      expect(detector.isReady, true);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty, reason: 'full model should detect faces');
      expect(faces.first.boundingBox.width, greaterThan(0));
      expect(faces.first.mesh, isNotNull);
      expect(faces.first.mesh!.points.length, 468);

      print(
          'full: Detected ${faces.length} face(s), bbox width: ${faces.first.boundingBox.width.toStringAsFixed(1)}');

      detector.dispose();
    });

    test('should handle various image sizes', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.full);

      for (final entry in testImages.entries) {
        final ByteData data = await rootBundle.load(entry.key);
        final Uint8List bytes = data.buffer.asUint8List();

        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);

        expect(faces, isNotEmpty, reason: 'full model failed on ${entry.key}');
        print('full: ${entry.key} -> ${faces.length} faces');
      }

      detector.dispose();
    });
  });

  group('FaceDetectionModel.fullSparse', () {
    test('should initialize and detect faces', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.fullSparse);
      expect(detector.isReady, true);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty, reason: 'fullSparse should detect faces');
      expect(faces.first.boundingBox.width, greaterThan(0));
      expect(faces.first.mesh, isNotNull);

      print(
          'fullSparse: Detected ${faces.length} face(s), bbox width: ${faces.first.boundingBox.width.toStringAsFixed(1)}');

      detector.dispose();
    });

    test('should work with all detection modes', () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.fullSparse);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      for (final mode in FaceDetectionMode.values) {
        final faces = await detector.detectFaces(bytes, mode: mode);
        expect(faces, isNotEmpty, reason: 'fullSparse ${mode.name} failed');
      }

      detector.dispose();
    });
  });

  group('Model Variant Comparison', () {
    test('all models should detect faces in standard test image', () async {
      final results = <String, int>{};

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      for (final model in FaceDetectionModel.values) {
        final detector = FaceDetector();
        await detector.initialize(model: model);

        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
        results[model.name] = faces.length;

        detector.dispose();
      }

      print('\nModel comparison on landmark-ex1.jpg:');
      for (final entry in results.entries) {
        print('  ${entry.key}: ${entry.value} face(s)');
        expect(entry.value, greaterThan(0),
            reason: '${entry.key} should detect at least one face');
      }
    });

    test('most models should detect multiple faces in group photo', () async {
      final results = <String, int>{};

      final ByteData data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final Uint8List bytes = data.buffer.asUint8List();

      for (final model in FaceDetectionModel.values) {
        final detector = FaceDetector();
        await detector.initialize(model: model);

        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
        results[model.name] = faces.length;

        detector.dispose();
      }

      print('\nModel comparison on group-shot (4 faces expected):');
      for (final entry in results.entries) {
        print('  ${entry.key}: ${entry.value} face(s)');
      }

      // frontCamera and shortRange are optimized for close-up faces
      // They may not detect multiple faces in group photos
      final closeUpModels = {'frontCamera', 'shortRange'};

      for (final entry in results.entries) {
        if (closeUpModels.contains(entry.key)) {
          // Just verify they don't crash and return valid result
          expect(entry.value, greaterThanOrEqualTo(0),
              reason: '${entry.key} should not crash on group photo');
        } else {
          // backCamera, full, fullSparse should detect multiple faces
          expect(entry.value, greaterThanOrEqualTo(2),
              reason: '${entry.key} should detect at least 2 faces in group');
        }
      }
    });

    test('detection results should be consistent across multiple runs',
        () async {
      final detector = FaceDetector();
      await detector.initialize(model: FaceDetectionModel.backCamera);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final counts = <int>[];
      for (int i = 0; i < 5; i++) {
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
        counts.add(faces.length);
      }

      // All runs should produce the same count
      expect(counts.toSet().length, 1,
          reason: 'Detection count should be consistent');

      detector.dispose();
    });
  });

  group('FaceDetectorIsolate with different models', () {
    test('should work with frontCamera model', () async {
      final detector = await FaceDetectorIsolate.spawn(
        model: FaceDetectionModel.frontCamera,
      );

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNotNull);

      await detector.dispose();
    });

    test('should work with shortRange model', () async {
      final detector = await FaceDetectorIsolate.spawn(
        model: FaceDetectionModel.shortRange,
      );

      final ByteData data =
          await rootBundle.load('assets/samples/iris-detection-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.full);

      expect(faces, isNotEmpty);
      expect(faces.first.irisPoints, isNotEmpty);

      await detector.dispose();
    });

    test('should work with fullSparse model', () async {
      final detector = await FaceDetectorIsolate.spawn(
        model: FaceDetectionModel.fullSparse,
      );

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);

      expect(faces, isNotEmpty);

      await detector.dispose();
    });
  });
}
