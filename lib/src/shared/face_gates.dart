/// Detection-gate helpers shared by the native and web pipelines.
///
/// A "gate" is a post-detection filter applied to the list of [Face] results
/// before they are handed back to the caller:
///
/// - [minScore] drops detections whose confidence is below a threshold.
/// - [minFaceSize] drops faces that are small relative to the source image,
///   measured as face width divided by image width (matching Google ML Kit's
///   `minFaceSize` convention).
///
/// Keeping this logic in a single pure-Dart file (no `dart:io`, no `dart:ui`
/// beyond what [Face] already needs, no platform imports) guarantees that the
/// native and web pipelines filter results identically, and makes the math
/// trivially unit-testable with synthetic [Face] objects.
library;

import 'face_types.dart' show Face;

/// Validates detection-gate parameters.
///
/// Throws [ArgumentError] when either value is NaN or falls outside the
/// inclusive range `[0.0, 1.0]`. Called once at detector initialization so bad
/// configuration fails fast, before any model is loaded.
void validateFaceGates({
  required double minScore,
  required double minFaceSize,
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
}

/// Returns the subset of [faces] that pass both gates, preserving input order.
///
/// A face passes when both of the following hold (both comparisons inclusive,
/// so a face exactly at a threshold is kept):
///
/// - [Face.score] is greater than or equal to [minScore].
/// - [Face.widthFraction] is greater than or equal to [minFaceSize].
///
/// When both gates are at their no-op defaults (`<= 0.0`) the original list is
/// returned unchanged with no allocation or iteration, so the common
/// "no filtering" path stays free.
///
/// Note: the detector already discards candidates below its internal
/// confidence floor before faces reach this function, so a [minScore] at or
/// below that floor cannot surface additional faces; it can only tighten the
/// result set further.
List<Face> applyFaceGates(
  List<Face> faces, {
  required double minScore,
  required double minFaceSize,
}) {
  if (minScore <= 0.0 && minFaceSize <= 0.0) return faces;
  return faces
      .where((f) => f.score >= minScore && f.widthFraction >= minFaceSize)
      .toList(growable: false);
}
