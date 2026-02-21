import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Test helper to create a minimal 1x1 PNG image.
class TestUtils {
  static Uint8List createDummyImageBytes() {
    return ImageGenerator.create1x1Png();
  }
}

/// Helper to create test images.
class ImageGenerator {
  /// Creates a minimal 1x1 PNG image.
  static Uint8List create1x1Png() {
    return Uint8List.fromList([
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
      0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
      0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
      0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, // RGBA
      0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
      0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
      0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, // IEND chunk
      0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
      0x42, 0x60, 0x82
    ]);
  }

  /// Creates a solid color cv.Mat of specified size.
  static cv.Mat createSolidMat(int width, int height,
      {int r = 128, int g = 128, int b = 128}) {
    final mat = cv.Mat.zeros(height, width, cv.MatType.CV_8UC3);
    mat.setTo(cv.Scalar(b.toDouble(), g.toDouble(), r.toDouble(), 255));
    return mat;
  }

  /// Creates a large cv.Mat.
  static cv.Mat createLargeMat(int width, int height) {
    return createSolidMat(width, height, r: 200, g: 200, b: 200);
  }
}
