/// Detection-gate helpers shared by the native and web pipelines.
///
/// A "gate" filters which detections are returned:
///
/// - [minScore] drops detections whose confidence is below a threshold.
/// - [minFaceSize] drops faces that are small relative to the source image,
///   measured as face width divided by image width (matching Google ML Kit's
///   `minFaceSize` convention).
///
/// Gates are applied at two points with identical arithmetic:
/// [applyDetectionGates] runs right after the detector stage (post-NMS,
/// post-letterbox) so gated-out detections skip the per-face mesh, iris and
/// blendshape stages entirely, and [applyFaceGates] runs on the final [Face]
/// list as a safety net.
///
/// Keeping this logic in a single pure-Dart file (no `dart:io`, no `dart:ui`
/// beyond what [Face] already needs, no platform imports) guarantees that the
/// native and web pipelines filter results identically, and makes the math
/// trivially unit-testable with synthetic [Face] objects.
library;

import 'dart:math' as math;

import 'face_types.dart' show Detection, Face, RectF;

/// Validates detection-gate parameters.
///
/// Throws [ArgumentError] when either value is NaN or falls outside the
/// inclusive range `[0.0, 1.0]`. Called once at detector initialization so bad
/// configuration fails fast, before any model is loaded.
void validateFaceGates({
  required double minScore,
  required double minFaceSize,
  double minFacePresenceConfidence = 0.0,
}) {
  if (minScore.isNaN || minScore < 0.0 || minScore > 1.0) {
    throw ArgumentError.value(
      minScore,
      'minScore',
      'must be in the inclusive range [0.0, 1.0]',
    );
  }
  if (minFaceSize.isNaN || minFaceSize < 0.0 || minFaceSize > 1.0) {
    throw ArgumentError.value(
      minFaceSize,
      'minFaceSize',
      'must be in the inclusive range [0.0, 1.0]',
    );
  }
  if (minFacePresenceConfidence.isNaN ||
      minFacePresenceConfidence < 0.0 ||
      minFacePresenceConfidence > 1.0) {
    throw ArgumentError.value(
      minFacePresenceConfidence,
      'minFacePresenceConfidence',
      'must be in the inclusive range [0.0, 1.0]',
    );
  }
}

/// Returns the subset of [faces] that pass all gates, preserving input order.
///
/// A face passes when all of the following hold (comparisons inclusive, so a
/// face exactly at a threshold is kept):
///
/// - [Face.score] is greater than or equal to [minScore].
/// - [Face.widthFraction] is greater than or equal to [minFaceSize].
/// - [Face.meshScore] is greater than or equal to [minFacePresenceConfidence].
///   A face whose [Face.meshScore] is null (fast mode, or a mesh model with no
///   presence output) is always kept: absence of a presence score is treated
///   as "cannot evaluate", never as a rejection.
///
/// When all gates are at their no-op defaults (`<= 0.0`) the original list is
/// returned unchanged with no allocation or iteration, so the common
/// "no filtering" path stays free.
///
/// Note: the detector already discards candidates below its internal
/// confidence floor before faces reach this function, so a [minScore] at or
/// below that floor cannot surface additional faces; it can only tighten the
/// result set further. [minFacePresenceConfidence] is the MediaPipe
/// `min_face_presence_confidence` gate, applied to the mesh model's face-flag
/// output; it is the second-stage check that rejects first-stage detections
/// (e.g. a palm) the landmark model does not confirm as a face.
List<Face> applyFaceGates(
  List<Face> faces, {
  required double minScore,
  required double minFaceSize,
  double minFacePresenceConfidence = 0.0,
}) {
  if (minScore <= 0.0 &&
      minFaceSize <= 0.0 &&
      minFacePresenceConfidence <= 0.0) {
    return faces;
  }
  return faces
      .where(
        (f) =>
            f.score >= minScore &&
            f.widthFraction >= minFaceSize &&
            (minFacePresenceConfidence <= 0.0 ||
                (f.meshScore ?? double.infinity) >= minFacePresenceConfidence),
      )
      .toList(growable: false);
}

/// Visible width of a normalized detector [box] as a fraction of the image
/// width [imageWidth] (in pixels), clipped to the image bounds.
///
/// This is the single source of truth for the `minFaceSize` measurement:
/// [Face.widthFraction] delegates here, and [applyDetectionGates] uses it at
/// the detector stage, so early and late gating agree to the last bit. The
/// operation order (scale to pixels, clip, divide) is deliberately identical
/// to the original [Face.widthFraction] implementation; do not "simplify" it
/// to normalized-space clipping, which can differ by one ULP at a threshold.
double boxVisibleWidthFraction(RectF box, double imageWidth) {
  if (imageWidth <= 0) return 0.0;
  final double left = box.xmin * imageWidth;
  final double right = box.xmax * imageWidth;
  final double visible = math.min(right, imageWidth) - math.max(left, 0.0);
  return visible > 0 ? visible / imageWidth : 0.0;
}

/// Returns the subset of [detections] that pass both gates, preserving input
/// order. Detector-stage equivalent of [applyFaceGates]: same inclusive
/// comparisons, same width arithmetic via [boxVisibleWidthFraction].
///
/// [imageWidth] is the width in pixels of the image the detections came from
/// (the detector boxes themselves are normalized). When both gates are at
/// their no-op defaults the original list is returned unchanged.
List<Detection> applyDetectionGates(
  List<Detection> detections, {
  required double minScore,
  required double minFaceSize,
  required double imageWidth,
}) {
  if (minScore <= 0.0 && minFaceSize <= 0.0) return detections;
  return detections
      .where(
        (d) =>
            d.score >= minScore &&
            (minFaceSize <= 0.0 ||
                boxVisibleWidthFraction(d.boundingBox, imageWidth) >=
                    minFaceSize),
      )
      .toList(growable: false);
}
