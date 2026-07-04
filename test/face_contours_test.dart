import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/src/native/face_native_lib.dart';

import 'test_config.dart';

/// Structural tests for the ML Kit-style named face contours
/// (`Face.getContour` / `Face.contours` and the `faceContourMeshIndices` map).
///
/// These use a synthetic mesh where point `i` is `Point(i, i, i)`, so the
/// returned contour points reveal exactly which mesh indices were selected and
/// in what order. Real-image geometry (top/bottom, left/right, oval bounds) is
/// validated separately in the integration tests.
void main() {
  globalTestSetup();

  // Point i == (i, i, i): a contour point's x is its source mesh index.
  Face faceWithIdentityMesh() {
    final points = List.generate(
      kMeshPoints,
      (i) => Point(i.toDouble(), i.toDouble(), i.toDouble()),
    );
    return Face(
      detection: Detection(
        boundingBox: RectF(0.2, 0.3, 0.8, 0.7),
        score: 0.95,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: TestConstants.mediumImage,
      ),
      mesh: FaceMesh(points),
      irises: const [],
      originalSize: TestConstants.mediumImage,
    );
  }

  Face fastModeFace() => Face(
    detection: Detection(
      boundingBox: RectF(0.2, 0.3, 0.8, 0.7),
      score: 0.95,
      keypointsXY: TestUtils.generateValidKeypoints(),
      imageSize: TestConstants.mediumImage,
    ),
    mesh: null,
    irises: const [],
    originalSize: TestConstants.mediumImage,
  );

  group('faceContourMeshIndices table', () {
    test('covers every FaceContourType exactly once', () {
      expect(
        faceContourMeshIndices.keys.toSet(),
        FaceContourType.values.toSet(),
      );
    });

    test('all indices are valid mesh indices in [0, 468)', () {
      for (final entry in faceContourMeshIndices.entries) {
        expect(entry.value, isNotEmpty, reason: '${entry.key} is empty');
        for (final i in entry.value) {
          expect(i, greaterThanOrEqualTo(0), reason: '${entry.key} has $i');
          expect(i, lessThan(kMeshPoints), reason: '${entry.key} has $i');
        }
      }
    });

    test('expected point counts match ML Kit / MediaPipe groups', () {
      const expected = {
        FaceContourType.face: 36,
        FaceContourType.leftEyebrowTop: 5,
        FaceContourType.leftEyebrowBottom: 5,
        FaceContourType.rightEyebrowTop: 5,
        FaceContourType.rightEyebrowBottom: 5,
        FaceContourType.leftEye: 16,
        FaceContourType.rightEye: 16,
        FaceContourType.upperLipTop: 11,
        FaceContourType.upperLipBottom: 11,
        FaceContourType.lowerLipTop: 11,
        FaceContourType.lowerLipBottom: 11,
        FaceContourType.noseBridge: 6,
        FaceContourType.noseBottom: 5,
        FaceContourType.leftCheek: 1,
        FaceContourType.rightCheek: 1,
      };
      for (final type in FaceContourType.values) {
        expect(
          faceContourMeshIndices[type]!.length,
          expected[type],
          reason: 'point count for $type',
        );
      }
    });

    test('the face oval has no repeated indices', () {
      final oval = faceContourMeshIndices[FaceContourType.face]!;
      expect(oval.toSet().length, oval.length);
    });

    test('shared lip corner indices are consistent (61/291, 78/308)', () {
      // Outer arcs (upperLipTop, lowerLipBottom) share the outer mouth corners
      // 61 and 291; inner arcs (upperLipBottom, lowerLipTop) share 78 and 308.
      final upperTop = faceContourMeshIndices[FaceContourType.upperLipTop]!;
      final lowerBottom =
          faceContourMeshIndices[FaceContourType.lowerLipBottom]!;
      final upperBottom =
          faceContourMeshIndices[FaceContourType.upperLipBottom]!;
      final lowerTop = faceContourMeshIndices[FaceContourType.lowerLipTop]!;
      expect(upperTop.first, lowerBottom.first); // 61
      expect(upperTop.last, lowerBottom.last); // 291
      expect(upperBottom.first, lowerTop.first); // 78
      expect(upperBottom.last, lowerTop.last); // 308
    });
  });

  group('Face.getContour', () {
    test('returns the mapped mesh points in order', () {
      final face = faceWithIdentityMesh();
      for (final type in FaceContourType.values) {
        final indices = faceContourMeshIndices[type]!;
        final contour = face.getContour(type);
        expect(contour, isNotNull, reason: '$type');
        expect(contour!.length, indices.length, reason: '$type');
        for (int k = 0; k < indices.length; k++) {
          // point i == (i, i, i), so x recovers the source index.
          expect(contour[k].x, indices[k].toDouble(), reason: '$type point $k');
          expect(contour[k].y, indices[k].toDouble(), reason: '$type point $k');
        }
      }
    });

    test('cheeks return a single point', () {
      final face = faceWithIdentityMesh();
      expect(face.getContour(FaceContourType.leftCheek)!.length, 1);
      expect(face.getContour(FaceContourType.rightCheek)!.length, 1);
      expect(face.getContour(FaceContourType.leftCheek)!.first.x, 280);
      expect(face.getContour(FaceContourType.rightCheek)!.first.x, 50);
    });

    test('returns null in fast mode (no mesh)', () {
      final face = fastModeFace();
      for (final type in FaceContourType.values) {
        expect(face.getContour(type), isNull, reason: '$type');
      }
    });
  });

  group('Face.contours', () {
    test('returns all contour types keyed by type', () {
      final face = faceWithIdentityMesh();
      final all = face.contours;
      expect(all, isNotNull);
      expect(all!.keys.toSet(), FaceContourType.values.toSet());
      for (final type in FaceContourType.values) {
        expect(all[type], face.getContour(type), reason: '$type');
      }
    });

    test('returns null in fast mode (no mesh)', () {
      expect(fastModeFace().contours, isNull);
    });
  });
}
