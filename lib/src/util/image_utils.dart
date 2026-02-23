import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Utility functions for image preprocessing and transformations using OpenCV.
///
/// Provides letterbox preprocessing, coordinate transformations, tensor
/// conversion utilities, and rotation-aware cropping for face detection.
/// Uses native OpenCV operations for 10-50x better performance than pure Dart.
class ImageUtils {
  ImageUtils._();

  /// Keeps aspect ratio while resizing and centers with padding.
  ///
  /// This matches the letterbox preprocessing used by MediaPipe models.
  /// Uses OpenCV's native resize for significantly better performance.
  ///
  /// Returns a tuple of (paddedImage, resizedImage) - both must be disposed.
  static (cv.Mat padded, cv.Mat resized) keepAspectResizeAndPad(
    cv.Mat image,
    int resizeWidth,
    int resizeHeight,
  ) {
    final imageHeight = image.rows;
    final imageWidth = image.cols;

    final ash = resizeHeight / imageHeight;
    final asw = resizeWidth / imageWidth;

    int newWidth, newHeight;
    if (asw < ash) {
      newWidth = (imageWidth * asw).toInt();
      newHeight = (imageHeight * asw).toInt();
    } else {
      newWidth = (imageWidth * ash).toInt();
      newHeight = (imageHeight * ash).toInt();
    }

    final resizedImage = cv.resize(
        image,
        (
          newWidth,
          newHeight,
        ),
        interpolation: cv.INTER_LINEAR);

    final padTop = (resizeHeight - newHeight) ~/ 2;
    final padBottom = resizeHeight - newHeight - padTop;
    final padLeft = (resizeWidth - newWidth) ~/ 2;
    final padRight = resizeWidth - newWidth - padLeft;

    final paddedImage = cv.copyMakeBorder(
      resizedImage,
      padTop,
      padBottom,
      padLeft,
      padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );

    return (paddedImage, resizedImage);
  }

  /// Crops a rotated square from an image using OpenCV's warpAffine.
  ///
  /// This is used to extract aligned face regions with proper rotation
  /// for the landmark model. Uses OpenCV's SIMD-optimized warpAffine which
  /// is 10-50x faster than pure Dart bilinear interpolation.
  ///
  /// Parameters:
  /// - [image]: Source image
  /// - [cx]: Center X coordinate in pixels
  /// - [cy]: Center Y coordinate in pixels
  /// - [size]: Output square size in pixels
  /// - [theta]: Rotation angle in radians (positive = counter-clockwise)
  ///
  /// Returns the cropped and rotated image, or null if the size is invalid.
  static cv.Mat? rotateAndCropSquare(
    cv.Mat image,
    double cx,
    double cy,
    double size,
    double theta,
  ) {
    final sizeInt = size.round();
    if (sizeInt <= 0) return null;

    final angleDegrees = -theta * 180.0 / math.pi;

    final rotMat = cv.getRotationMatrix2D(
      cv.Point2f(cx, cy),
      angleDegrees,
      1.0,
    );

    final outCenter = sizeInt / 2.0;

    final tx = rotMat.at<double>(0, 2) + outCenter - cx;
    final ty = rotMat.at<double>(1, 2) + outCenter - cy;
    rotMat.set<double>(0, 2, tx);
    rotMat.set<double>(1, 2, ty);

    final output = cv.warpAffine(
      image,
      rotMat,
      (sizeInt, sizeInt),
      borderMode: cv.BORDER_CONSTANT,
      borderValue: cv.Scalar.black,
    );

    rotMat.dispose();
    return output;
  }

  /// Crops a rectangular region from an image.
  ///
  /// Parameters:
  /// - [image]: Source image
  /// - [x1]: Left boundary in pixels
  /// - [y1]: Top boundary in pixels
  /// - [x2]: Right boundary in pixels
  /// - [y2]: Bottom boundary in pixels
  ///
  /// Returns the cropped region. Caller must dispose.
  static cv.Mat cropRect(cv.Mat image, int x1, int y1, int x2, int y2) {
    final clampedX1 = x1.clamp(0, image.cols - 1);
    final clampedY1 = y1.clamp(0, image.rows - 1);
    final clampedX2 = x2.clamp(clampedX1 + 1, image.cols);
    final clampedY2 = y2.clamp(clampedY1 + 1, image.rows);

    final width = clampedX2 - clampedX1;
    final height = clampedY2 - clampedY1;

    final rect = cv.Rect(clampedX1, clampedY1, width, height);
    return image.region(rect);
  }

  /// Crops a region from an image using normalized coordinates.
  ///
  /// Parameters:
  /// - [image]: Source image
  /// - [x1Norm]: Left boundary (0.0 to 1.0)
  /// - [y1Norm]: Top boundary (0.0 to 1.0)
  /// - [x2Norm]: Right boundary (0.0 to 1.0)
  /// - [y2Norm]: Bottom boundary (0.0 to 1.0)
  ///
  /// Returns the cropped region. Caller must dispose.
  static cv.Mat cropFromNormalizedRoi(
    cv.Mat image,
    double x1Norm,
    double y1Norm,
    double x2Norm,
    double y2Norm,
  ) {
    final x1 = (x1Norm * image.cols).round();
    final y1 = (y1Norm * image.rows).round();
    final x2 = (x2Norm * image.cols).round();
    final y2 = (y2Norm * image.rows).round();
    return cropRect(image, x1, y1, x2, y2);
  }

  /// Converts a cv.Mat to a flat Float32List tensor for MediaPipe TFLite models.
  ///
  /// Converts pixel values from 0-255 range to normalized [-1.0, 1.0] range
  /// as required by MediaPipe models.
  /// Also converts from BGR (OpenCV format) to RGB (TFLite expected format).
  ///
  /// Parameters:
  /// - [mat]: Source image in BGR format
  /// - [buffer]: Optional pre-allocated buffer to reuse
  ///
  /// Returns a flat Float32List with normalized RGB pixel values in [-1, 1].
  static Float32List matToFloat32TensorMediaPipe(
    cv.Mat mat, {
    Float32List? buffer,
  }) {
    final data = mat.data;
    final totalPixels = mat.rows * mat.cols;
    final size = totalPixels * 3;
    final tensor = buffer ?? Float32List(size);

    for (int i = 0, j = 0; i < totalPixels * 3 && j < size; i += 3, j += 3) {
      tensor[j] = (data[i + 2] / 127.5) - 1.0;
      tensor[j + 1] = (data[i + 1] / 127.5) - 1.0;
      tensor[j + 2] = (data[i] / 127.5) - 1.0;
    }
    return tensor;
  }

  /// Converts a cv.Mat to a flat Float32List tensor with [0, 1] normalization.
  ///
  /// Converts pixel values from 0-255 range to normalized 0.0-1.0 range.
  /// Also converts from BGR (OpenCV format) to RGB (TFLite expected format).
  ///
  /// Parameters:
  /// - [mat]: Source image in BGR format
  /// - [buffer]: Optional pre-allocated buffer to reuse
  ///
  /// Returns a flat Float32List with normalized RGB pixel values in [0, 1].
  static Float32List matToFloat32Tensor(cv.Mat mat, {Float32List? buffer}) {
    final data = mat.data;
    final totalPixels = mat.rows * mat.cols;
    final size = totalPixels * 3;
    final tensor = buffer ?? Float32List(size);
    const scale = 1.0 / 255.0;

    for (int i = 0, j = 0; i < totalPixels * 3 && j < size; i += 3, j += 3) {
      tensor[j] = data[i + 2] * scale;
      tensor[j + 1] = data[i + 1] * scale;
      tensor[j + 2] = data[i] * scale;
    }
    return tensor;
  }

  /// Applies letterbox preprocessing to fit an image into target dimensions.
  ///
  /// Scales the source image to fit within [tw]x[th] while maintaining aspect ratio,
  /// then pads with black to fill the target dimensions.
  ///
  /// Parameters:
  /// - [src]: Source image to preprocess
  /// - [tw]: Target width in pixels
  /// - [th]: Target height in pixels
  ///
  /// Returns a record with:
  /// - padded: The letterboxed image with dimensions [tw]x[th]
  /// - scale: The scale ratio used
  /// - padLeft: Left padding in pixels
  /// - padTop: Top padding in pixels
  static ({cv.Mat padded, double scale, int padLeft, int padTop}) letterbox(
    cv.Mat src,
    int tw,
    int th,
  ) {
    final int w = src.cols;
    final int h = src.rows;
    final double scale = math.min(th / h, tw / w);
    final int nw = (w * scale).round();
    final int nh = (h * scale).round();
    final int padLeft = (tw - nw) ~/ 2;
    final int padTop = (th - nh) ~/ 2;

    final resized = cv.resize(src, (nw, nh), interpolation: cv.INTER_LINEAR);

    final padRight = tw - nw - padLeft;
    final padBottom = th - nh - padTop;

    final padded = cv.copyMakeBorder(
      resized,
      padTop,
      padBottom,
      padLeft,
      padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );
    resized.dispose();

    return (padded: padded, scale: scale, padLeft: padLeft, padTop: padTop);
  }

  /// Converts a cv.Mat to 4D tensor in NHWC format for TensorFlow Lite.
  ///
  /// Converts pixel values from 0-255 range to normalized [-1.0, 1.0] range
  /// as required by MediaPipe models.
  /// The output format is [batch, height, width, channels] where batch=1 and channels=3.
  ///
  /// Parameters:
  /// - [mat]: Source image in BGR format
  /// - [width]: Target width (must match mat.cols)
  /// - [height]: Target height (must match mat.rows)
  /// - [reuse]: Optional tensor buffer to reuse (must match dimensions)
  ///
  /// Returns a 4D tensor [1, height, width, 3] with normalized pixel values.
  static List<List<List<List<double>>>> matToNHWC4DMediaPipe(
    cv.Mat mat,
    int width,
    int height, {
    List<List<List<List<double>>>>? reuse,
  }) {
    final List<List<List<List<double>>>> out = reuse ??
        List.generate(
          1,
          (_) => List.generate(
            height,
            (_) => List.generate(
              width,
              (_) => List<double>.filled(3, 0.0),
              growable: false,
            ),
            growable: false,
          ),
          growable: false,
        );

    final bytes = mat.data;
    int byteIndex = 0;

    for (int y = 0; y < height; y++) {
      final List<List<double>> row = out[0][y];
      for (int x = 0; x < width; x++) {
        final List<double> pixel = row[x];
        pixel[0] = (bytes[byteIndex + 2] / 127.5) - 1.0;
        pixel[1] = (bytes[byteIndex + 1] / 127.5) - 1.0;
        pixel[2] = (bytes[byteIndex] / 127.5) - 1.0;
        byteIndex += 3;
      }
    }
    return out;
  }

  /// Decodes image bytes to cv.Mat.
  ///
  /// Parameters:
  /// - [bytes]: Image bytes (JPEG, PNG, etc.)
  /// - [flags]: Decode flags (default: IMREAD_COLOR for BGR)
  ///
  /// Returns the decoded image. Caller must dispose.
  static cv.Mat decodeImage(Uint8List bytes, {int flags = cv.IMREAD_COLOR}) {
    return cv.imdecode(bytes, flags);
  }
}
