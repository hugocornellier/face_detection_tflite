part of 'native/face_native_lib.dart';

/// Aligned face crop data holder for OpenCV-based processing.
///
/// Holds a [cv.Mat] for the aligned face crop. Lives in the native side
/// because it is tied to opencv_dart.
class AlignedFace {
  /// X coordinate of the face center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of the face center in absolute pixel coordinates.
  final double cy;

  /// Length of the square crop edge in absolute pixels.
  final double size;

  /// Rotation applied to align the face, in radians.
  final double theta;

  /// The aligned face crop as cv.Mat. Caller must dispose when done.
  final cv.Mat faceCrop;

  /// Creates an aligned face crop with a cv.Mat face crop image.
  AlignedFace({
    required this.cx,
    required this.cy,
    required this.size,
    required this.theta,
    required this.faceCrop,
  });
}
