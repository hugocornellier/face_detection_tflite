import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/src/native/face_native_lib.dart';
import 'package:face_detection_tflite/src/shared/face_geometry.dart'
    show
        computeFaceAlignment,
        eyeRoisFromMesh,
        transformIrisFlatToAbsolute,
        transformIrisNormToAbsolute,
        transformMeshToAbsolute;

import 'test_config.dart';

/// Unit tests for the ROI-alignment and landmark-transform half of
/// `face_geometry.dart`.
///
/// These functions are pure Dart and shared by the native and web pipelines,
/// but they only ran inside model-backed integration tests, which collect no
/// coverage. The tests below pin their contracts directly: the invariants that
/// must hold for any input (mapping the ROI center back to itself, preserving
/// point counts) and the differential agreement between the normalized-space
/// and pixel-space variants of the iris transform.
///
/// `headEulerAnglesFromMesh` and `transformMeshFlatToAbsolute` are covered by
/// `head_pose_test.dart`.
void main() {
  globalTestSetup();

  const double eps = 1e-9;

  // Builds a detection whose six keypoints are laid out as a level face:
  // eyes on a horizontal line, mouth centered below them.
  Detection makeDetection({
    required double leftEyeX,
    required double leftEyeY,
    required double rightEyeX,
    required double rightEyeY,
    required double mouthX,
    required double mouthY,
  }) {
    return Detection(
      boundingBox: const RectF(0.2, 0.2, 0.8, 0.8),
      score: 0.9,
      keypointsXY: <double>[
        leftEyeX, leftEyeY, // leftEye
        rightEyeX, rightEyeY, // rightEye
        0.5, 0.5, // noseTip (unused by the alignment math)
        mouthX, mouthY, // mouth
        0.1, 0.4, // leftEyeTragion (unused)
        0.9, 0.4, // rightEyeTragion (unused)
      ],
    );
  }

  group('computeFaceAlignment', () {
    test('a level face has zero rotation', () {
      final a = computeFaceAlignment(
        makeDetection(
          leftEyeX: 0.3,
          leftEyeY: 0.4,
          rightEyeX: 0.7,
          rightEyeY: 0.4,
          mouthX: 0.5,
          mouthY: 0.7,
        ),
        100,
        100,
      );

      expect(a.theta, closeTo(0.0, eps));
    });

    test('theta follows the eye line, not the mouth', () {
      // Eyes on a 45-degree line: dy == dx, so atan2 gives pi/4.
      final a = computeFaceAlignment(
        makeDetection(
          leftEyeX: 0.3,
          leftEyeY: 0.3,
          rightEyeX: 0.7,
          rightEyeY: 0.7,
          mouthX: 0.5,
          mouthY: 0.9,
        ),
        100,
        100,
      );

      expect(a.theta, closeTo(math.pi / 4, eps));
    });

    test('keypoints are denormalized against a non-square image', () {
      // Eye center is at normalized x 0.5, y 0.4; with the mouth directly
      // below the eye center the center only shifts along y.
      final a = computeFaceAlignment(
        makeDetection(
          leftEyeX: 0.3,
          leftEyeY: 0.4,
          rightEyeX: 0.7,
          rightEyeY: 0.4,
          mouthX: 0.5,
          mouthY: 0.9,
        ),
        640,
        480,
      );

      const double eyeCx = 0.5 * 640;
      const double eyeCy = 0.4 * 480;
      const double mouthY = 0.9 * 480;

      expect(a.cx, closeTo(eyeCx, 1e-6));
      // Center is pulled 10% of the way from the eye center to the mouth.
      expect(a.cy, closeTo(eyeCy + (mouthY - eyeCy) * 0.1, 1e-6));
    });

    test('the center is nudged toward the mouth on both axes', () {
      // An off-center mouth shifts cx as well as cy; a mouth directly below
      // the eyes leaves cx untouched and hides the horizontal term.
      final a = computeFaceAlignment(
        makeDetection(
          leftEyeX: 0.3,
          leftEyeY: 0.4,
          rightEyeX: 0.7,
          rightEyeY: 0.4,
          mouthX: 0.8,
          mouthY: 0.9,
        ),
        100,
        100,
      );

      const double eyeCx = 0.5 * 100;
      const double eyeCy = 0.4 * 100;
      expect(a.cx, closeTo(eyeCx + (0.8 * 100 - eyeCx) * 0.1, 1e-6));
      expect(a.cy, closeTo(eyeCy + (0.9 * 100 - eyeCy) * 0.1, 1e-6));
    });

    test('size takes the eye-distance branch on a wide, short face', () {
      // eyeDist = 0.6 * 100 = 60 -> 240; mouthDist = 0.1 * 100 = 10 -> 36.
      final a = computeFaceAlignment(
        makeDetection(
          leftEyeX: 0.2,
          leftEyeY: 0.4,
          rightEyeX: 0.8,
          rightEyeY: 0.4,
          mouthX: 0.5,
          mouthY: 0.5,
        ),
        100,
        100,
      );

      expect(a.size, closeTo(60.0 * 4.0, 1e-6));
    });

    test('size takes the mouth-distance branch on a narrow, long face', () {
      // eyeDist = 0.1 * 100 = 10 -> 40; mouthDist = 0.5 * 100 = 50 -> 180.
      final a = computeFaceAlignment(
        makeDetection(
          leftEyeX: 0.45,
          leftEyeY: 0.3,
          rightEyeX: 0.55,
          rightEyeY: 0.3,
          mouthX: 0.5,
          mouthY: 0.8,
        ),
        100,
        100,
      );

      expect(a.size, closeTo(50.0 * 3.6, 1e-6));
    });
  });

  group('transformMeshToAbsolute', () {
    // The normalized mesh center maps back to the ROI center regardless of
    // how the ROI is rotated or scaled. This is the property the whole crop
    // round-trip depends on.
    test('the normalized center maps to the ROI center for any pose', () {
      for (final double theta in <double>[
        0.0,
        math.pi / 6,
        math.pi / 2,
        -math.pi / 3,
        math.pi,
      ]) {
        final mesh = transformMeshToAbsolute(
          <List<double>>[
            <double>[0.5, 0.5, 0.0],
          ],
          120.0,
          80.0,
          200.0,
          theta,
        );

        expect(mesh.single.x, closeTo(120.0, 1e-9));
        expect(mesh.single.y, closeTo(80.0, 1e-9));
      }
    });

    test('an unrotated ROI maps the unit square onto the ROI box', () {
      final mesh = transformMeshToAbsolute(
        <List<double>>[
          <double>[0.0, 0.0, 0.0],
          <double>[1.0, 1.0, 0.0],
        ],
        100.0,
        100.0,
        50.0,
        0.0,
      );

      expect(mesh[0].x, closeTo(75.0, 1e-9));
      expect(mesh[0].y, closeTo(75.0, 1e-9));
      expect(mesh[1].x, closeTo(125.0, 1e-9));
      expect(mesh[1].y, closeTo(125.0, 1e-9));
    });

    test('a quarter turn rotates offsets from the center', () {
      // At theta = pi/2 the ROI's +x axis points along image +y.
      final mesh = transformMeshToAbsolute(
        <List<double>>[
          <double>[1.0, 0.5, 0.0],
        ],
        0.0,
        0.0,
        100.0,
        math.pi / 2,
      );

      expect(mesh.single.x, closeTo(0.0, 1e-9));
      expect(mesh.single.y, closeTo(50.0, 1e-9));
    });

    test('z is scaled by the ROI size', () {
      final mesh = transformMeshToAbsolute(
        <List<double>>[
          <double>[0.5, 0.5, 0.25],
        ],
        0.0,
        0.0,
        200.0,
        0.0,
      );

      expect(mesh.single.z, closeTo(0.25 * 200.0, 1e-9));
    });

    test('point count is preserved and an empty mesh stays empty', () {
      final many = transformMeshToAbsolute(
        List<List<double>>.generate(
          468,
          (int i) => <double>[i / 468, i / 468, 0.0],
        ),
        10.0,
        10.0,
        20.0,
        0.3,
      );
      expect(many, hasLength(468));

      expect(
        transformMeshToAbsolute(<List<double>>[], 0.0, 0.0, 1.0, 0.0),
        isEmpty,
      );
    });
  });

  group('transformIrisNormToAbsolute', () {
    const AlignedRoi roi = AlignedRoi(50.0, 60.0, 40.0, 0.0);

    test('the normalized center maps to the ROI center for both eyes', () {
      for (final bool isRight in <bool>[false, true]) {
        final out = transformIrisNormToAbsolute(
          <List<double>>[
            <double>[0.5, 0.5, 0.0],
          ],
          roi,
          isRight,
        );

        expect(out.single[0], closeTo(roi.cx, 1e-9));
        expect(out.single[1], closeTo(roi.cy, 1e-9));
      }
    });

    test('the right eye mirrors x about the ROI center', () {
      const List<List<double>> lm = <List<double>>[
        <double>[0.25, 0.5, 0.0],
      ];

      final left = transformIrisNormToAbsolute(lm, roi, false);
      final right = transformIrisNormToAbsolute(lm, roi, true);

      expect(left.single[0], closeTo(roi.cx - 0.25 * roi.size, 1e-9));
      expect(right.single[0], closeTo(roi.cx + 0.25 * roi.size, 1e-9));
      // Mirroring is horizontal only.
      expect(left.single[1], closeTo(right.single[1], 1e-9));
    });

    test('z passes through unscaled', () {
      final out = transformIrisNormToAbsolute(
        <List<double>>[
          <double>[0.5, 0.5, 0.75],
        ],
        roi,
        false,
      );

      expect(out.single[2], closeTo(0.75, 1e-9));
    });

    test('a rotated ROI rotates the offset', () {
      const AlignedRoi rotated = AlignedRoi(0.0, 0.0, 100.0, math.pi / 2);
      final out = transformIrisNormToAbsolute(
        <List<double>>[
          <double>[1.0, 0.5, 0.0],
        ],
        rotated,
        false,
      );

      expect(out.single[0], closeTo(0.0, 1e-9));
      expect(out.single[1], closeTo(50.0, 1e-9));
    });
  });

  group('transformIrisFlatToAbsolute', () {
    const AlignedRoi roi = AlignedRoi(50.0, 60.0, 40.0, 0.0);
    const int inW = 64;
    const int inH = 64;

    test('the center pixel maps to the ROI center for both eyes', () {
      for (final bool isRight in <bool>[false, true]) {
        final out = transformIrisFlatToAbsolute(
          Float32List.fromList(<double>[inW / 2, inH / 2, 0.0]),
          roi,
          isRight,
          inW,
          inH,
        );

        expect(out.single.x, closeTo(roi.cx, 1e-6));
        expect(out.single.y, closeTo(roi.cy, 1e-6));
      }
    });

    test('the right eye mirrors x about the ROI center', () {
      final flat = Float32List.fromList(<double>[16.0, 32.0, 0.0]);

      final left = transformIrisFlatToAbsolute(flat, roi, false, inW, inH);
      final right = transformIrisFlatToAbsolute(flat, roi, true, inW, inH);

      expect(left.single.x, closeTo(roi.cx - 0.25 * roi.size, 1e-6));
      expect(right.single.x, closeTo(roi.cx + 0.25 * roi.size, 1e-6));
      expect(left.single.y, closeTo(right.single.y, 1e-6));
    });

    // The two iris transforms feed the same downstream consumers from
    // different model output layouts, so on a square crop they must agree in
    // x and y for the same physical landmark.
    test('agrees with the normalized variant on a square crop', () {
      const List<List<double>> norm = <List<double>>[
        <double>[0.10, 0.20, 0.0],
        <double>[0.50, 0.50, 0.0],
        <double>[0.85, 0.30, 0.0],
        <double>[0.33, 0.90, 0.0],
      ];
      const AlignedRoi tilted = AlignedRoi(120.0, 90.0, 55.0, 0.4);

      final flat = Float32List.fromList(<double>[
        for (final List<double> p in norm) ...<double>[
          p[0] * inW,
          p[1] * inH,
          p[2],
        ],
      ]);

      for (final bool isRight in <bool>[false, true]) {
        final expected = transformIrisNormToAbsolute(norm, tilted, isRight);
        final actual = transformIrisFlatToAbsolute(
          flat,
          tilted,
          isRight,
          inW,
          inH,
        );

        expect(actual, hasLength(norm.length));
        for (int i = 0; i < norm.length; i++) {
          expect(actual[i].x, closeTo(expected[i][0], 1e-4));
          expect(actual[i].y, closeTo(expected[i][1], 1e-4));
        }
      }
    });

    test('z is scaled by the ROI size', () {
      final out = transformIrisFlatToAbsolute(
        Float32List.fromList(<double>[inW / 2, inH / 2, 0.5]),
        roi,
        false,
        inW,
        inH,
      );

      expect(out.single.z, closeTo(0.5 * roi.size, 1e-6));
    });

    test('a trailing partial point is ignored', () {
      // 7 floats is two whole (x, y, z) points plus a stray value.
      final out = transformIrisFlatToAbsolute(
        Float32List.fromList(<double>[0, 0, 0, 1, 1, 1, 2]),
        roi,
        false,
        inW,
        inH,
      );

      expect(out, hasLength(2));
    });
  });

  group('eyeRoisFromMesh', () {
    // Builds a mesh large enough to index the canonical eye corners, with the
    // four corner landmarks placed explicitly.
    List<Point> meshWithEyeCorners({
      required Point left0,
      required Point left1,
      required Point right0,
      required Point right1,
    }) {
      final mesh = List<Point>.filled(468, const Point(0, 0, 0));
      mesh[33] = left0;
      mesh[133] = left1;
      mesh[362] = right0;
      mesh[263] = right1;
      return mesh;
    }

    test('derives center, size, and rotation from the eye corners', () {
      final rois = eyeRoisFromMesh(
        meshWithEyeCorners(
          left0: const Point(10, 50, 0),
          left1: const Point(30, 50, 0),
          right0: const Point(70, 50, 0),
          right1: const Point(90, 50, 0),
        ),
      );

      expect(rois, hasLength(2));

      // Left eye: corners 20px apart, centered at x 20.
      expect(rois[0].cx, closeTo(20.0, 1e-9));
      expect(rois[0].cy, closeTo(50.0, 1e-9));
      expect(rois[0].size, closeTo(20.0 * 2.3, 1e-9));
      expect(rois[0].theta, closeTo(0.0, 1e-9));

      // Right eye: same span, centered at x 80.
      expect(rois[1].cx, closeTo(80.0, 1e-9));
      expect(rois[1].size, closeTo(20.0 * 2.3, 1e-9));
    });

    test('a tilted eye line produces a rotated ROI', () {
      final rois = eyeRoisFromMesh(
        meshWithEyeCorners(
          left0: const Point(0, 0, 0),
          left1: const Point(10, 10, 0),
          right0: const Point(0, 0, 0),
          right1: const Point(0, 10, 0),
        ),
      );

      expect(rois[0].theta, closeTo(math.pi / 4, 1e-9));
      expect(rois[0].size, closeTo(math.sqrt(200.0) * 2.3, 1e-9));

      // A vertical corner pair is a quarter turn.
      expect(rois[1].theta, closeTo(math.pi / 2, 1e-9));
    });

    test('a mesh too short to hold the eye indices throws', () {
      expect(
        () => eyeRoisFromMesh(List<Point>.filled(10, const Point(0, 0, 0))),
        throwsRangeError,
      );
    });
  });
}
