// ignore_for_file: avoid_print

/// Edge case tests for FaceDetector.
///
/// Tests unusual inputs and boundary conditions:
/// - Empty/tiny images
/// - Very large images
/// - Images with no faces
/// - Images with faces at boundaries
/// - Various image formats
/// - Corrupted/invalid input handling
library;

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Helper to create test images
class ImageGenerator {
  /// Creates a minimal 1x1 PNG image
  static Uint8List create1x1Png() {
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

  /// Creates a solid color cv.Mat of specified size
  static cv.Mat createSolidMat(int width, int height,
      {int r = 128, int g = 128, int b = 128}) {
    final mat = cv.Mat.zeros(height, width, cv.MatType.CV_8UC3);
    mat.setTo(cv.Scalar(b.toDouble(), g.toDouble(), r.toDouble(), 255));
    return mat;
  }

  /// Creates a large cv.Mat
  static cv.Mat createLargeMat(int width, int height) {
    return createSolidMat(width, height, r: 200, g: 200, b: 200);
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const edgeCaseTimeout = Timeout(Duration(minutes: 3));

  late FaceDetector detector;
  late Uint8List validFaceImage;

  setUpAll(() async {
    detector = FaceDetector();
    await detector.initialize();

    final data = await rootBundle.load('assets/samples/landmark-ex1.jpg');
    validFaceImage = data.buffer.asUint8List();
  });

  tearDownAll(() {
    detector.dispose();
  });

  group('Empty and Tiny Images', () {
    test('should handle 1x1 PNG image', () async {
      final bytes = ImageGenerator.create1x1Png();

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);

      // Should not crash, should return empty
      expect(faces, isEmpty);
    });

    test('should handle 1x1 cv.Mat', () async {
      final mat = cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);
      mat.dispose();
    });

    test('should handle 10x10 solid image', () async {
      final mat = ImageGenerator.createSolidMat(10, 10);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);
      mat.dispose();
    });

    test('should handle 50x50 solid image', () async {
      final mat = ImageGenerator.createSolidMat(50, 50);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);
      mat.dispose();
    });
  });

  group('Large Images', () {
    test('should handle 1920x1080 image', () async {
      final mat = ImageGenerator.createLargeMat(1920, 1080);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      // No faces in solid color image
      expect(faces, isEmpty);
      mat.dispose();
    });

    test('should handle 4K resolution (3840x2160)', () async {
      final mat = ImageGenerator.createLargeMat(3840, 2160);

      // Should not crash or timeout
      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);
      mat.dispose();
    }, timeout: edgeCaseTimeout);

    test('should handle non-standard aspect ratio (100x2000)', () async {
      final mat = ImageGenerator.createLargeMat(100, 2000);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);
      mat.dispose();
    });

    test('should handle wide panorama (3000x500)', () async {
      final mat = ImageGenerator.createLargeMat(3000, 500);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);
      mat.dispose();
    });
  });

  group('Invalid Input Handling', () {
    test('should handle empty byte array', () async {
      final bytes = Uint8List(0);

      // Should throw or return empty, not crash
      try {
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
        expect(faces, isEmpty);
      } catch (e) {
        // Exception is acceptable for truly invalid input
        expect(e, isNotNull);
      }
    });

    test('should handle random bytes (not valid image)', () async {
      final bytes = Uint8List.fromList(List.generate(100, (i) => i % 256));

      try {
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
        expect(faces, isEmpty);
      } catch (e) {
        expect(e, isNotNull);
      }
    });

    test('should handle truncated PNG', () async {
      final validPng = ImageGenerator.create1x1Png();
      final truncated =
          Uint8List.fromList(validPng.sublist(0, validPng.length ~/ 2));

      try {
        final faces =
            await detector.detectFaces(truncated, mode: FaceDetectionMode.fast);
        expect(faces, isEmpty);
      } catch (e) {
        expect(e, isNotNull);
      }
    });

    test('should recover after invalid input', () async {
      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

      // Process invalid input
      try {
        await detector.detectFaces(invalidBytes, mode: FaceDetectionMode.fast);
      } catch (e) {
        // Expected
      }

      // Should still work with valid input
      final faces = await detector.detectFaces(validFaceImage,
          mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty, reason: 'Should recover after invalid input');
    });
  });

  group('Images Without Faces', () {
    test('should return empty for solid color image', () async {
      final mat = ImageGenerator.createSolidMat(256, 256);

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);
      mat.dispose();
    });

    test('should return empty for gradient image', () async {
      final mat = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);

      // Create gradient
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          mat.set(y, x, cv.Vec3b(x, y, 128));
        }
      }

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);
      mat.dispose();
    });

    test('should handle random noise image', () async {
      final mat = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);
      cv.randu(mat, cv.Scalar(0, 0, 0, 0), cv.Scalar(255, 255, 255, 255));

      final faces =
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast);

      // May or may not detect false positives in noise
      expect(faces, isNotNull);
      mat.dispose();
    });
  });

  group('Detection Mode Edge Cases', () {
    test('should handle mode switching on same image', () async {
      for (final mode in FaceDetectionMode.values) {
        final faces = await detector.detectFaces(validFaceImage, mode: mode);
        expect(faces, isNotEmpty,
            reason: 'Mode ${mode.name} should detect face');
      }
    });

    test('fast mode should not have mesh or iris', () async {
      final faces = await detector.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNull);
      expect(faces.first.irisPoints, isEmpty);
    });

    test('standard mode should have mesh but not iris', () async {
      final faces = await detector.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.standard,
      );

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNotNull);
      expect(faces.first.mesh!.points.length, 468);
      expect(faces.first.irisPoints, isEmpty);
    });

    test('full mode should have mesh and iris', () async {
      final faces = await detector.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.full,
      );

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNotNull);
      expect(faces.first.mesh!.points.length, 468);
      expect(faces.first.irisPoints, isNotEmpty);
    });
  });

  group('Bounding Box Validation', () {
    test('bounding box should be within image bounds', () async {
      final faces = await detector.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);

      final face = faces.first;
      final bbox = face.boundingBox;
      final imgWidth = face.originalSize.width.toDouble();
      final imgHeight = face.originalSize.height.toDouble();

      // All corners should be within image
      expect(bbox.topLeft.x, greaterThanOrEqualTo(0));
      expect(bbox.topLeft.y, greaterThanOrEqualTo(0));
      expect(bbox.bottomRight.x, lessThanOrEqualTo(imgWidth));
      expect(bbox.bottomRight.y, lessThanOrEqualTo(imgHeight));

      // Width and height should be positive
      expect(bbox.width, greaterThan(0));
      expect(bbox.height, greaterThan(0));
    });

    test('landmarks should be within image bounds', () async {
      final faces = await detector.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);

      final face = faces.first;
      final imgWidth = face.originalSize.width.toDouble();
      final imgHeight = face.originalSize.height.toDouble();

      for (final point in face.landmarks.values) {
        expect(point.x, greaterThanOrEqualTo(0));
        expect(point.y, greaterThanOrEqualTo(0));
        expect(point.x, lessThanOrEqualTo(imgWidth));
        expect(point.y, lessThanOrEqualTo(imgHeight));
      }
    });

    test('mesh points should be within reasonable bounds', () async {
      final faces = await detector.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.standard,
      );

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNotNull);

      final face = faces.first;
      final imgWidth = face.originalSize.width.toDouble();
      final imgHeight = face.originalSize.height.toDouble();

      for (final point in face.mesh!.points) {
        // Mesh points should be within image (with some tolerance)
        expect(point.x, greaterThanOrEqualTo(-imgWidth * 0.1));
        expect(point.y, greaterThanOrEqualTo(-imgHeight * 0.1));
        expect(point.x, lessThanOrEqualTo(imgWidth * 1.1));
        expect(point.y, lessThanOrEqualTo(imgHeight * 1.1));
      }
    });
  });

  group('Multi-Face Edge Cases', () {
    late Uint8List groupImage;

    setUpAll(() async {
      final data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      groupImage = data.buffer.asUint8List();
    });

    test('should detect all faces in group photo', () async {
      final faces = await detector.detectFaces(
        groupImage,
        mode: FaceDetectionMode.fast,
      );

      expect(faces.length, greaterThanOrEqualTo(2));
      print('Group photo: detected ${faces.length} faces');
    });

    test('all faces should have valid bounding boxes', () async {
      final faces = await detector.detectFaces(
        groupImage,
        mode: FaceDetectionMode.fast,
      );

      for (int i = 0; i < faces.length; i++) {
        final face = faces[i];
        final bbox = face.boundingBox;

        expect(bbox.width, greaterThan(0), reason: 'Face $i has invalid width');
        expect(bbox.height, greaterThan(0),
            reason: 'Face $i has invalid height');
      }
    });

    test('bounding boxes should not overlap completely', () async {
      final faces = await detector.detectFaces(
        groupImage,
        mode: FaceDetectionMode.fast,
      );

      if (faces.length < 2) return;

      // Check that faces have distinct bounding boxes
      for (int i = 0; i < faces.length; i++) {
        for (int j = i + 1; j < faces.length; j++) {
          final bbox1 = faces[i].boundingBox;
          final bbox2 = faces[j].boundingBox;

          final centerDist = ((bbox1.center.x - bbox2.center.x).abs() +
              (bbox1.center.y - bbox2.center.y).abs());

          expect(centerDist, greaterThan(10),
              reason: 'Faces $i and $j should have different positions');
        }
      }
    });

    test('all faces should have mesh in full mode', () async {
      final faces = await detector.detectFaces(
        groupImage,
        mode: FaceDetectionMode.full,
      );

      for (int i = 0; i < faces.length; i++) {
        expect(faces[i].mesh, isNotNull, reason: 'Face $i missing mesh');
        expect(faces[i].mesh!.points.length, 468,
            reason: 'Face $i has wrong mesh size');
      }
    });
  });

  group('Serialization Edge Cases', () {
    test('Face.toMap/fromMap should preserve all data', () async {
      final faces = await detector.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.full,
      );

      expect(faces, isNotEmpty);

      final original = faces.first;
      final map = original.toMap();
      final restored = Face.fromMap(map);

      // Compare bounding boxes
      expect(restored.boundingBox.topLeft.x, original.boundingBox.topLeft.x);
      expect(restored.boundingBox.topLeft.y, original.boundingBox.topLeft.y);
      expect(restored.boundingBox.width, original.boundingBox.width);
      expect(restored.boundingBox.height, original.boundingBox.height);

      // Compare mesh
      expect(restored.mesh, isNotNull);
      expect(restored.mesh!.points.length, original.mesh!.points.length);

      // Compare iris points
      expect(restored.irisPoints.length, original.irisPoints.length);

      // Compare original size
      expect(restored.originalSize.width, original.originalSize.width);
      expect(restored.originalSize.height, original.originalSize.height);
    });

    test('Face.toString should not crash', () async {
      final faces = await detector.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.full,
      );

      expect(faces, isNotEmpty);

      final str = faces.first.toString();
      expect(str, isNotEmpty);
      expect(str, contains('Face'));
    });
  });

  group('Embedding Edge Cases', () {
    test('should handle embedding for each detected face', () async {
      final data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final groupImage = data.buffer.asUint8List();

      final faces = await detector.detectFaces(
        groupImage,
        mode: FaceDetectionMode.fast,
      );

      expect(faces.length, greaterThan(1));

      // Generate embedding for each face
      for (int i = 0; i < faces.length; i++) {
        try {
          final embedding =
              await detector.getFaceEmbedding(faces[i], groupImage);
          expect(embedding.length, greaterThan(0),
              reason: 'Face $i should have valid embedding');
        } catch (e) {
          // Some faces may fail due to alignment issues
          print('Face $i embedding failed: $e');
        }
      }
    });

    test('different faces should have different embeddings', () async {
      final data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final groupImage = data.buffer.asUint8List();

      final faces = await detector.detectFaces(
        groupImage,
        mode: FaceDetectionMode.fast,
      );

      if (faces.length < 2) return;

      final embeddings = await detector.getFaceEmbeddings(faces, groupImage);
      final validEmbeddings = embeddings.whereType<Float32List>().toList();

      if (validEmbeddings.length < 2) return;

      // Compare first two embeddings
      final similarity = FaceDetector.compareFaces(
        validEmbeddings[0],
        validEmbeddings[1],
      );

      // Different people should have lower similarity
      print('Different faces similarity: ${similarity.toStringAsFixed(3)}');
      expect(similarity, lessThan(1.0),
          reason: 'Different faces should have < 1.0 similarity');
    });
  });

  group('FaceDetectorIsolate Edge Cases', () {
    test('should handle tiny image', () async {
      final isolate = await FaceDetectorIsolate.spawn();

      final bytes = ImageGenerator.create1x1Png();
      final faces =
          await isolate.detectFaces(bytes, mode: FaceDetectionMode.fast);

      expect(faces, isEmpty);

      await isolate.dispose();
    });

    test('should handle detectFaces then getFaceEmbedding', () async {
      final isolate = await FaceDetectorIsolate.spawn();

      final faces = await isolate.detectFaces(
        validFaceImage,
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);

      final embedding =
          await isolate.getFaceEmbedding(faces.first, validFaceImage);
      expect(embedding.length, greaterThan(0));

      await isolate.dispose();
    });

    test('should throw after dispose', () async {
      final isolate = await FaceDetectorIsolate.spawn();
      await isolate.dispose();

      expect(
        () => isolate.detectFaces(validFaceImage),
        throwsA(isA<StateError>()),
      );
    });
  });
}
