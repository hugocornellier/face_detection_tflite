import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'test_config.dart';

void main() {
  globalTestSetup();

  group('FaceDetector constructor and state', () {
    test('should create uninitialized detector', () {
      final detector = FaceDetector();
      expect(detector.isReady, isFalse);
      expect(detector.isEmbeddingReady, isFalse);
      expect(detector.isSegmentationReady, isFalse);
      expect(detector.irisOkCount, 0);
      expect(detector.irisFailCount, 0);
      expect(detector.irisUsedFallbackCount, 0);
      expect(detector.lastIrisTime, Duration.zero);
    });

    test('should safely dispose without initialization', () {
      final detector = FaceDetector();
      expect(() => detector.dispose(), returnsNormally);
    });

    test('should safely dispose twice', () {
      final detector = FaceDetector();
      detector.dispose();
      expect(() => detector.dispose(), returnsNormally);
    });
  });

  group('FaceDetector.compareFaces', () {
    test('should return 1.0 for identical vectors', () {
      final a = Float32List.fromList([0.5, 0.5, 0.5]);
      expect(FaceDetector.compareFaces(a, a), closeTo(1.0, 0.001));
    });

    test('should return -1.0 for opposite vectors', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b = Float32List.fromList([-1.0, 0.0]);
      expect(FaceDetector.compareFaces(a, b), closeTo(-1.0, 0.001));
    });
  });

  group('FaceDetector.faceDistance', () {
    test('should return 0.0 for identical vectors', () {
      final a = Float32List.fromList([0.5, 0.5, 0.5]);
      expect(FaceDetector.faceDistance(a, a), closeTo(0.0, 0.001));
    });

    test('should return correct distance', () {
      final a = Float32List.fromList([0.0, 0.0]);
      final b = Float32List.fromList([3.0, 4.0]);
      expect(FaceDetector.faceDistance(a, b), closeTo(5.0, 0.001));
    });
  });

  group('FaceDetector.splitMeshesIfConcatenated', () {
    test('should return empty list for empty input', () {
      final detector = FaceDetector();
      expect(detector.splitMeshesIfConcatenated([]), isEmpty);
      detector.dispose();
    });

    test('should return single mesh for 468 points', () {
      final detector = FaceDetector();
      final mesh = List.generate(468, (i) => Point(i.toDouble(), 0));
      final result = detector.splitMeshesIfConcatenated(mesh);
      expect(result.length, 1);
      expect(result[0].length, 468);
      detector.dispose();
    });

    test('should split concatenated meshes for 936 points', () {
      final detector = FaceDetector();
      final mesh = List.generate(936, (i) => Point(i.toDouble(), 0));
      final result = detector.splitMeshesIfConcatenated(mesh);
      expect(result.length, 2);
      expect(result[0].length, 468);
      expect(result[1].length, 468);
      detector.dispose();
    });

    test('should return single list for non-multiple of 468', () {
      final detector = FaceDetector();
      final mesh = List.generate(100, (i) => Point(i.toDouble(), 0));
      final result = detector.splitMeshesIfConcatenated(mesh);
      expect(result.length, 1);
      expect(result[0].length, 100);
      detector.dispose();
    });
  });

  group('FaceDetector.eyeRoisFromMesh', () {
    test('should return two ROIs from a valid mesh', () {
      final detector = FaceDetector();
      // Create a 468-point mesh with spread-out eye landmarks
      final mesh = List.generate(468, (i) => Point(i.toDouble(), i.toDouble()));
      // Mesh indices 33, 133 for left eye corners; 362, 263 for right eye
      final rois = detector.eyeRoisFromMesh(mesh);

      expect(rois.length, 2);
      // Check left eye ROI is centered between points 33 and 133
      expect(rois[0].cx, closeTo((33.0 + 133.0) / 2, 0.01));
      expect(rois[0].cy, closeTo((33.0 + 133.0) / 2, 0.01));
      // Check right eye ROI
      expect(rois[1].cx, closeTo((362.0 + 263.0) / 2, 0.01));
      expect(rois[1].cy, closeTo((362.0 + 263.0) / 2, 0.01));

      detector.dispose();
    });

    test('should calculate correct size based on eye distance', () {
      final detector = FaceDetector();
      final mesh = List.generate(468, (i) => Point(0, 0));
      // Set left eye corners
      mesh[33] = Point(10, 50);
      mesh[133] = Point(110, 50);
      // Set right eye corners
      mesh[362] = Point(200, 50);
      mesh[263] = Point(300, 50);

      final rois = detector.eyeRoisFromMesh(mesh);

      // Left eye distance = 100, size = 100 * 2.3 = 230
      expect(rois[0].size, closeTo(230, 0.1));
      expect(rois[0].cx, closeTo(60, 0.1));
      expect(rois[0].cy, closeTo(50, 0.1));

      // Right eye distance = 100, size = 100 * 2.3 = 230
      expect(rois[1].size, closeTo(230, 0.1));
      expect(rois[1].cx, closeTo(250, 0.1));
      expect(rois[1].cy, closeTo(50, 0.1));

      detector.dispose();
    });
  });

  group('FaceDetector error states', () {
    test('detectFaces throws StateError when not initialized', () async {
      final detector = FaceDetector();
      final bytes = TestUtils.createDummyImageBytes();

      expect(() => detector.detectFaces(bytes), throwsA(isA<StateError>()));
      detector.dispose();
    });
  });

  group('FaceDetector segmentation state', () {
    test('getSegmentationMask throws when not initialized', () {
      final detector = FaceDetector();
      expect(
        () => detector.getSegmentationMask(Uint8List(0)),
        throwsA(isA<StateError>()),
      );
      detector.dispose();
    });

    test('getSegmentationMaskFromMat throws when not initialized', () {
      final detector = FaceDetector();
      final mat = cv.Mat.zeros(10, 10, cv.MatType.CV_8UC3);
      expect(
        () => detector.getSegmentationMaskFromMat(mat),
        throwsA(isA<StateError>()),
      );
      mat.dispose();
      detector.dispose();
    });
  });

  group('FaceDetector embedding state', () {
    test('getFaceEmbedding throws when not initialized', () {
      final detector = FaceDetector();

      final detection = Detection(
        boundingBox: RectF(0.2, 0.3, 0.8, 0.7),
        score: 0.95,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: TestConstants.mediumImage,
      );
      final face = Face(
        detection: detection,
        mesh: null,
        irises: [],
        originalSize: TestConstants.mediumImage,
      );

      expect(
        () => detector.getFaceEmbedding(face, Uint8List(0)),
        throwsA(isA<StateError>()),
      );
      detector.dispose();
    });

    test('getFaceEmbeddings throws when not initialized', () {
      final detector = FaceDetector();
      expect(
        () => detector.getFaceEmbeddings([], Uint8List(0)),
        throwsA(isA<StateError>()),
      );
      detector.dispose();
    });
  });

  group('FaceDetector.detectFacesFromMat error states', () {
    test('detectFacesFromMat throws StateError when not initialized', () {
      final detector = FaceDetector();
      final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);
      expect(
        () => detector.detectFacesFromMat(mat),
        throwsA(isA<StateError>()),
      );
      mat.dispose();
      detector.dispose();
    });

    test('detectFacesFromMat with fast mode throws when not initialized', () {
      final detector = FaceDetector();
      final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);
      expect(
        () => detector.detectFacesFromMat(mat, mode: FaceDetectionMode.fast),
        throwsA(isA<StateError>()),
      );
      mat.dispose();
      detector.dispose();
    });

    test(
      'detectFacesFromMat with standard mode throws when not initialized',
      () {
        final detector = FaceDetector();
        final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);
        expect(
          () => detector.detectFacesFromMat(
            mat,
            mode: FaceDetectionMode.standard,
          ),
          throwsA(isA<StateError>()),
        );
        mat.dispose();
        detector.dispose();
      },
    );
  });

  group('FaceDetector.compareFaces edge cases', () {
    test('should return 0.0 for orthogonal vectors', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b = Float32List.fromList([0.0, 1.0]);
      expect(FaceDetector.compareFaces(a, b), closeTo(0.0, 0.001));
    });

    test('should handle multi-dimensional vectors', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0, 0.0]);
      final b = Float32List.fromList([1.0, 0.0, 0.0, 0.0]);
      expect(FaceDetector.compareFaces(a, b), closeTo(1.0, 0.001));
    });
  });

  group('FaceDetector.faceDistance edge cases', () {
    test('should compute correct L2 distance', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0]);
      final b = Float32List.fromList([0.0, 1.0, 0.0]);
      // sqrt(1 + 1) = sqrt(2)
      expect(FaceDetector.faceDistance(a, b), closeTo(1.414, 0.01));
    });
  });

  group('FaceDetector.splitMeshesIfConcatenated edge cases', () {
    test('should split three concatenated meshes', () {
      final detector = FaceDetector();
      final mesh = List.generate(468 * 3, (i) => Point(i.toDouble(), 0));
      final result = detector.splitMeshesIfConcatenated(mesh);
      expect(result.length, 3);
      for (final m in result) {
        expect(m.length, 468);
      }
      detector.dispose();
    });
  });
}
