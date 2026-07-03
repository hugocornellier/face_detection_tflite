/// Shared face geometry math used by both the native and web pipelines.
///
/// All functions here are pure Dart (no `dart:io`, no `cv.Mat`, no platform
/// imports) so they are safe to call from any platform.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart' show Point;

import 'face_types.dart'
    show AlignedRoi, Detection, FaceLandmarkType, HeadEulerAngles, RectF;

/// Computes the rotation, center, and size of the aligned face ROI from a
/// detection's eye/mouth keypoints.
({double theta, double cx, double cy, double size}) computeFaceAlignment(
  Detection det,
  double imgW,
  double imgH,
) {
  final lx = det.keypointsXY[FaceLandmarkType.leftEye.index * 2] * imgW;
  final ly = det.keypointsXY[FaceLandmarkType.leftEye.index * 2 + 1] * imgH;
  final rx = det.keypointsXY[FaceLandmarkType.rightEye.index * 2] * imgW;
  final ry = det.keypointsXY[FaceLandmarkType.rightEye.index * 2 + 1] * imgH;
  final mx = det.keypointsXY[FaceLandmarkType.mouth.index * 2] * imgW;
  final my = det.keypointsXY[FaceLandmarkType.mouth.index * 2 + 1] * imgH;

  final eyeCx = (lx + rx) * 0.5;
  final eyeCy = (ly + ry) * 0.5;
  final vEx = rx - lx;
  final vEy = ry - ly;
  final vMx = mx - eyeCx;
  final vMy = my - eyeCy;

  final theta = math.atan2(vEy, vEx);
  final eyeDist = math.sqrt(vEx * vEx + vEy * vEy);
  final mouthDist = math.sqrt(vMx * vMx + vMy * vMy);
  final size = math.max(mouthDist * 3.6, eyeDist * 4.0);

  final cx = eyeCx + vMx * 0.1;
  final cy = eyeCy + vMy * 0.1;

  return (theta: theta, cx: cx, cy: cy, size: size);
}

/// Transforms normalized 468pt mesh landmarks to absolute image coordinates.
List<Point> transformMeshToAbsolute(
  List<List<double>> lmNorm,
  double cx,
  double cy,
  double size,
  double theta,
) {
  final double ct = math.cos(theta);
  final double st = math.sin(theta);
  final double sct = size * ct;
  final double sst = size * st;
  final double tx = cx - 0.5 * sct + 0.5 * sst;
  final double ty = cy - 0.5 * sst - 0.5 * sct;

  final int n = lmNorm.length;
  final List<Point> mesh = List<Point>.filled(n, const Point(0, 0, 0));
  for (int i = 0; i < n; i++) {
    final List<double> p = lmNorm[i];
    mesh[i] = Point(
      tx + sct * p[0] - sst * p[1],
      ty + sst * p[0] + sct * p[1],
      p[2] * size,
    );
  }
  return mesh;
}

/// Transforms a flat Float32List of (x, y, z) mesh points (in the model's
/// input pixel space) back to the original image's absolute coordinates.
///
/// Used by the web mesh path which produces output in pixel space rather than
/// normalized space.
List<Point> transformMeshFlatToAbsolute(
  Float32List flat,
  double cx,
  double cy,
  double size,
  double theta,
  int inW,
  int inH,
) {
  final int n = flat.length ~/ 3;
  final List<Point> out = List<Point>.filled(n, const Point(0, 0, 0));
  final double ct = math.cos(theta);
  final double st = math.sin(theta);
  final double scale = size / inW;
  for (int i = 0; i < n; i++) {
    final double mx = flat[i * 3] - inW / 2.0;
    final double my = flat[i * 3 + 1] - inH / 2.0;
    final double mz = flat[i * 3 + 2];
    final double rx = ct * mx - st * my;
    final double ry = st * mx + ct * my;
    // Scale z the same way as x/y (by `scale`, not `size`) so depth is in the
    // same units as the 2D coordinates; needed for head-pose estimation.
    out[i] = Point(cx + rx * scale, cy + ry * scale, mz * scale);
  }
  return out;
}

/// Transforms iris landmarks from a model's normalized space back to absolute
/// pixel coordinates, undoing any horizontal flip applied to right-eye crops.
List<List<double>> transformIrisNormToAbsolute(
  List<List<double>> lmNorm,
  AlignedRoi roi,
  bool isRight,
) {
  final double ct = math.cos(roi.theta);
  final double st = math.sin(roi.theta);
  final double s = roi.size;
  final List<List<double>> out = <List<double>>[];
  for (final List<double> p in lmNorm) {
    final double px = isRight ? (1.0 - p[0]) : p[0];
    final double lx2 = (px - 0.5) * s;
    final double ly2 = (p[1] - 0.5) * s;
    out.add([roi.cx + lx2 * ct - ly2 * st, roi.cy + lx2 * st + ly2 * ct, p[2]]);
  }
  return out;
}

/// Transforms iris landmarks given as a flat Float32List in pixel space (used
/// by the web iris path).
List<Point> transformIrisFlatToAbsolute(
  Float32List flat,
  AlignedRoi roi,
  bool isRight,
  int inW,
  int inH,
) {
  final int n = flat.length ~/ 3;
  final List<Point> out = List<Point>.filled(n, const Point(0, 0, 0));
  final double ct = math.cos(roi.theta);
  final double st = math.sin(roi.theta);
  final double scale = roi.size / inW;
  for (int i = 0; i < n; i++) {
    double mx = flat[i * 3] - inW / 2.0;
    final double my = flat[i * 3 + 1] - inH / 2.0;
    final double mz = flat[i * 3 + 2];
    if (isRight) mx = -mx;
    final double rx = ct * mx - st * my;
    final double ry = st * mx + ct * my;
    out[i] = Point(roi.cx + rx * scale, roi.cy + ry * scale, mz * roi.size);
  }
  return out;
}

/// Computes the eye ROIs from the canonical mesh landmark indices used by
/// MediaPipe (left: 33/133, right: 362/263).
List<AlignedRoi> eyeRoisFromMesh(List<Point> meshAbs) {
  AlignedRoi fromCorners(int a, int b) {
    final Point p0 = meshAbs[a];
    final Point p1 = meshAbs[b];
    final double cx = (p0.x + p1.x) * 0.5;
    final double cy = (p0.y + p1.y) * 0.5;
    final double dx = p1.x - p0.x;
    final double dy = p1.y - p0.y;
    final double eyeDist = math.sqrt(dx * dx + dy * dy);
    return AlignedRoi(cx, cy, eyeDist * 2.3, math.atan2(dy, dx));
  }

  return [fromCorners(33, 133), fromCorners(362, 263)];
}

/// Canonical MediaPipe face-mesh indices used to derive head orientation.
///
/// These four landmarks span the two head axes: forehead->chin (vertical) and
/// left cheek->right cheek (horizontal). "Left"/"right" here are in image
/// space (left = smaller x).
const int kMeshForeheadTop = 10;
const int kMeshChinBottom = 152;
const int kMeshLeftCheek = 234;
const int kMeshRightCheek = 454;

double _degrees(double radians) => radians * 180.0 / math.pi;

/// Estimates head Euler angles (pitch, yaw, roll in degrees) from the 468-point
/// face mesh in absolute image coordinates.
///
/// Builds an orthonormal head reference frame from four canonical landmarks
/// (forehead, chin, and both cheeks) using their 3D positions, then reads the
/// Euler angles off the resulting rotation matrix. Returns null when the mesh
/// is too short or lacks depth information (mesh points always carry z, so this
/// only guards against malformed input).
HeadEulerAngles? headEulerAnglesFromMesh(List<Point> mesh) {
  if (mesh.length <= kMeshRightCheek) return null;
  final Point foreheadTop = mesh[kMeshForeheadTop];
  final Point chinBottom = mesh[kMeshChinBottom];
  final Point leftCheek = mesh[kMeshLeftCheek];
  final Point rightCheek = mesh[kMeshRightCheek];
  if (foreheadTop.z == null ||
      chinBottom.z == null ||
      leftCheek.z == null ||
      rightCheek.z == null) {
    return null;
  }

  // Image coordinate frame: x right, y down, z away from camera. The head
  // right axis runs left cheek -> right cheek; the head down axis runs
  // forehead -> chin.
  double rx = rightCheek.x - leftCheek.x;
  double ry = rightCheek.y - leftCheek.y;
  double rz = rightCheek.z! - leftCheek.z!;
  double dx = chinBottom.x - foreheadTop.x;
  double dy = chinBottom.y - foreheadTop.y;
  double dz = chinBottom.z! - foreheadTop.z!;

  double rLen = math.sqrt(rx * rx + ry * ry + rz * rz);
  double dLen = math.sqrt(dx * dx + dy * dy + dz * dz);
  if (rLen < 1e-6 || dLen < 1e-6) return null;
  rx /= rLen;
  ry /= rLen;
  rz /= rLen;
  dx /= dLen;
  dy /= dLen;
  dz /= dLen;

  // Gram-Schmidt: make the down axis orthogonal to the right axis.
  final double dDotR = dx * rx + dy * ry + dz * rz;
  dx -= dDotR * rx;
  dy -= dDotR * ry;
  dz -= dDotR * rz;
  dLen = math.sqrt(dx * dx + dy * dy + dz * dz);
  if (dLen < 1e-6) return null;
  dx /= dLen;
  dy /= dLen;
  dz /= dLen;

  // Head backward axis = right x down (right-handed), z-component only. Points
  // away from the face front (behind the head) for a frontal pose.
  final double bz = rx * dy - ry * dx;

  // The rotation matrix has columns [right, down, back] in world (image)
  // coordinates, so its bottom row is [rz, dz, bz]. Aerospace ZYX
  // (yaw-pitch-roll) extraction only needs that row plus (rx, ry).
  final double pitch = math.atan2(dz, bz);
  final double yaw = math.asin((-rz).clamp(-1.0, 1.0));
  final double roll = math.atan2(ry, rx);

  // Flip signs to match ML Kit semantics (see [HeadEulerAngles]).
  return HeadEulerAngles(_degrees(-pitch), _degrees(-yaw), _degrees(-roll));
}

/// Estimates head roll (in-plane tilt) in degrees from two eye keypoints in
/// absolute image coordinates. Used as a fast-mode fallback when no mesh is
/// available. Positive for a counter-clockwise tilt, matching [HeadEulerAngles].
double rollFromEyes(Point leftEye, Point rightEye) {
  final double dx = rightEye.x - leftEye.x;
  final double dy = rightEye.y - leftEye.y;
  return _degrees(-math.atan2(dy, dx));
}

/// Expands a face bounding box and turns it into a square ROI suitable for
/// face mesh alignment.
RectF faceDetectionToRoi(RectF boundingBox, {double expandFraction = 0.6}) {
  final RectF e = boundingBox.expand(expandFraction);
  final double cx = (e.xmin + e.xmax) * 0.5;
  final double cy = (e.ymin + e.ymax) * 0.5;
  final double s = math.max(e.w, e.h) * 0.5;
  return RectF(cx - s, cy - s, cx + s, cy + s);
}
