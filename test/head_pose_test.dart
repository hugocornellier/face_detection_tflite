import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/src/native/face_native_lib.dart';
import 'package:face_detection_tflite/src/shared/face_geometry.dart'
    show headEulerAnglesFromMesh, transformMeshFlatToAbsolute;

import 'test_config.dart';

/// Tests for head Euler angle estimation (Face.headEulerAngleX/Y/Z), including
/// the mesh-based path and the fast-mode (roll-only) fallback.
///
/// Sign conventions follow Google ML Kit:
/// - X (pitch): positive = face looking up.
/// - Y (yaw): positive = face turned toward the right side of the image.
/// - Z (roll): positive = counter-clockwise in-plane tilt.
void main() {
  globalTestSetup();

  const double deg = math.pi / 180.0;

  // A minimal frontal mesh in image coordinates (x right, y down, z away).
  // Only the four landmarks used by the estimator are meaningful; the rest are
  // placeholders. Centered on the origin so rotations are pure.
  List<List<double>> frontalMesh() {
    final m = List.generate(468, (_) => <double>[0.0, 0.0, 0.0]);
    m[10] = [0, -100, 0]; // forehead top (up)
    m[152] = [0, 100, 0]; // chin bottom (down)
    m[234] = [-100, 0, 0]; // left cheek (image-left)
    m[454] = [100, 0, 0]; // right cheek (image-right)
    return m;
  }

  List<List<double>> rotX(double a) => [
    [1, 0, 0],
    [0, math.cos(a), -math.sin(a)],
    [0, math.sin(a), math.cos(a)],
  ];
  List<List<double>> rotY(double a) => [
    [math.cos(a), 0, math.sin(a)],
    [0, 1, 0],
    [-math.sin(a), 0, math.cos(a)],
  ];
  List<List<double>> rotZ(double a) => [
    [math.cos(a), -math.sin(a), 0],
    [math.sin(a), math.cos(a), 0],
    [0, 0, 1],
  ];

  // Builds a Face whose mesh is the frontal mesh transformed by [r] (offset to
  // positive image coordinates so points are plausible pixels).
  Face faceWithRotatedMesh(List<List<double>> r) {
    final points = frontalMesh().map((p) {
      final x = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2];
      final y = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2];
      final z = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2];
      return Point(x + 320, y + 240, z);
    }).toList();
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

  group('Face.headEulerAngles - mesh path', () {
    test('frontal mesh yields ~0 on all axes', () {
      final a = faceWithRotatedMesh(rotZ(0));
      expect(a.headEulerAngleX, closeTo(0, 0.5));
      expect(a.headEulerAngleY, closeTo(0, 0.5));
      expect(a.headEulerAngleZ, closeTo(0, 0.5));
    });

    test('looking up gives positive pitch (X)', () {
      // rotX(-a) tips the forehead back / chin forward => looking up.
      final a = faceWithRotatedMesh(rotX(-25 * deg));
      expect(a.headEulerAngleX, closeTo(25, 0.5));
      expect(a.headEulerAngleY, closeTo(0, 0.5));
      expect(a.headEulerAngleZ, closeTo(0, 0.5));
    });

    test('looking down gives negative pitch (X)', () {
      final a = faceWithRotatedMesh(rotX(25 * deg));
      expect(a.headEulerAngleX, closeTo(-25, 0.5));
    });

    test('turning toward image-right gives positive yaw (Y)', () {
      // rotY(-a) sends the nose toward image-right (+x).
      final a = faceWithRotatedMesh(rotY(-25 * deg));
      expect(a.headEulerAngleY, closeTo(25, 0.5));
      expect(a.headEulerAngleX, closeTo(0, 0.5));
      expect(a.headEulerAngleZ, closeTo(0, 0.5));
    });

    test('turning toward image-left gives negative yaw (Y)', () {
      final a = faceWithRotatedMesh(rotY(25 * deg));
      expect(a.headEulerAngleY, closeTo(-25, 0.5));
    });

    test('counter-clockwise tilt gives positive roll (Z)', () {
      // rotZ(-a) rotates counter-clockwise in a y-down image.
      final a = faceWithRotatedMesh(rotZ(-25 * deg));
      expect(a.headEulerAngleZ, closeTo(25, 0.5));
      expect(a.headEulerAngleX, closeTo(0, 0.5));
      expect(a.headEulerAngleY, closeTo(0, 0.5));
    });

    test('clockwise tilt gives negative roll (Z)', () {
      final a = faceWithRotatedMesh(rotZ(25 * deg));
      expect(a.headEulerAngleZ, closeTo(-25, 0.5));
    });
  });

  group('Face.headEulerAngles - fast-mode fallback', () {
    // No mesh: only roll is estimated (from eye keypoints), pitch/yaw are 0.
    Face fastFace(Point leftEye, Point rightEye) {
      final w = TestConstants.mediumImage.width;
      final h = TestConstants.mediumImage.height;
      return Face(
        detection: Detection(
          boundingBox: RectF(0.2, 0.3, 0.8, 0.7),
          score: 0.9,
          keypointsXY: TestUtils.generateValidKeypoints(
            leftEye: Point(leftEye.x / w, leftEye.y / h),
            rightEye: Point(rightEye.x / w, rightEye.y / h),
          ),
          imageSize: TestConstants.mediumImage,
        ),
        mesh: null,
        irises: const [],
        originalSize: TestConstants.mediumImage,
      );
    }

    test('level eyes give ~0 roll and 0 pitch/yaw', () {
      final a = fastFace(const Point(200, 240), const Point(440, 240));
      expect(a.headEulerAngleX, 0.0);
      expect(a.headEulerAngleY, 0.0);
      expect(a.headEulerAngleZ, closeTo(0, 0.5));
    });

    test('right eye higher (counter-clockwise tilt) gives positive roll', () {
      // Right eye above left => CCW => positive Z.
      final a = fastFace(const Point(200, 260), const Point(440, 220));
      expect(a.headEulerAngleZ, greaterThan(5));
    });

    test('right eye lower (clockwise tilt) gives negative roll', () {
      final a = fastFace(const Point(200, 220), const Point(440, 260));
      expect(a.headEulerAngleZ, lessThan(-5));
    });
  });

  group('web mesh transform - z scale consistency', () {
    // Guards the web path (transformMeshFlatToAbsolute): its z must be scaled
    // like x/y, not by the ROI size, or head pose collapses to yaw ~= 90.
    const int inW = 256;
    const int inH = 256;
    const double c = 128; // input-space center

    Float32List flatFrontal({double rightCheekZ = 0}) {
      final f = Float32List(468 * 3);
      void set(int i, double x, double y, double z) {
        f[i * 3] = x;
        f[i * 3 + 1] = y;
        f[i * 3 + 2] = z;
      }

      set(10, c, c - 80, 0); // forehead up
      set(152, c, c + 80, 0); // chin down
      set(234, c - 80, c, 0); // left cheek
      set(454, c + 80, c, rightCheekZ); // right cheek
      return f;
    }

    test('planar frontal mesh yields ~0 on all axes', () {
      final pts = transformMeshFlatToAbsolute(
        flatFrontal(),
        300,
        300,
        400,
        0,
        inW,
        inH,
      );
      final a = HeadEulerAngles(0, 0, 0);
      final r = headEulerAnglesFromMesh(pts) ?? a;
      expect(r.x, closeTo(0, 1.0));
      expect(r.y, closeTo(0, 1.0));
      expect(r.z, closeTo(0, 1.0));
    });

    test('small out-of-plane depth gives a small (not ~90) yaw', () {
      // Right cheek 20px back over a 160px horizontal span -> ~7 deg true yaw.
      // With the old z*=size bug this collapses toward the asin(+-1) limit.
      final pts = transformMeshFlatToAbsolute(
        flatFrontal(rightCheekZ: 20),
        300,
        300,
        400,
        0,
        inW,
        inH,
      );
      final r = headEulerAnglesFromMesh(pts)!;
      expect(r.y, greaterThan(2));
      expect(r.y, lessThan(20));
    });
  });

  group('HeadEulerAngles serialization', () {
    test('round-trips through toMap/fromMap', () {
      const a = HeadEulerAngles(12.5, -7.25, 3.0);
      final b = HeadEulerAngles.fromMap(a.toMap());
      expect(b.x, a.x);
      expect(b.y, a.y);
      expect(b.z, a.z);
    });
  });
}
