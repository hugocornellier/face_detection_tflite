import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/src/image_utils.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'test_config.dart';

void main() {
  globalTestSetup();

  /// Creates a solid-color BGR cv.Mat for testing.
  cv.Mat solidMat(int width, int height, int b, int g, int r) {
    return cv.Mat.fromList(
      height,
      width,
      cv.MatType.CV_8UC3,
      List<int>.generate(
        width * height * 3,
        (i) => [b, g, r][i % 3],
      ),
    );
  }

  group('ImageUtils.keepAspectResizeAndPad', () {
    test('should resize a square image to target square dimensions', () {
      final src = solidMat(100, 100, 128, 128, 128);
      final (padded, resized) = ImageUtils.keepAspectResizeAndPad(src, 50, 50);

      expect(padded.cols, 50);
      expect(padded.rows, 50);
      expect(resized.cols, 50);
      expect(resized.rows, 50);

      padded.dispose();
      resized.dispose();
      src.dispose();
    });

    test('should maintain aspect ratio for landscape image', () {
      final src = solidMat(200, 100, 0, 0, 255);
      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(src, 100, 100);

      expect(padded.cols, 100);
      expect(padded.rows, 100);
      // Width-constrained: new width = 100, new height = 50
      expect(resized.cols, 100);
      expect(resized.rows, 50);

      padded.dispose();
      resized.dispose();
      src.dispose();
    });

    test('should maintain aspect ratio for portrait image', () {
      final src = solidMat(100, 200, 255, 0, 0);
      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(src, 100, 100);

      expect(padded.cols, 100);
      expect(padded.rows, 100);
      // Height-constrained: new height = 100, new width = 50
      expect(resized.cols, 50);
      expect(resized.rows, 100);

      padded.dispose();
      resized.dispose();
      src.dispose();
    });
  });

  group('ImageUtils.rotateAndCropSquare', () {
    test('should return null for zero size', () {
      final src = solidMat(100, 100, 128, 128, 128);
      final result = ImageUtils.rotateAndCropSquare(src, 50, 50, 0, 0);
      expect(result, isNull);
      src.dispose();
    });

    test('should return null for negative size', () {
      final src = solidMat(100, 100, 128, 128, 128);
      final result = ImageUtils.rotateAndCropSquare(src, 50, 50, -10, 0);
      expect(result, isNull);
      src.dispose();
    });

    test('should return a square crop of the correct size', () {
      final src = solidMat(200, 200, 128, 128, 128);
      final result = ImageUtils.rotateAndCropSquare(src, 100, 100, 50, 0);

      expect(result, isNotNull);
      expect(result!.cols, 50);
      expect(result.rows, 50);

      result.dispose();
      src.dispose();
    });

    test('should handle rotation', () {
      final src = solidMat(200, 200, 128, 128, 128);
      final result = ImageUtils.rotateAndCropSquare(
        src,
        100,
        100,
        60,
        3.14159 / 4,
      );

      expect(result, isNotNull);
      expect(result!.cols, 60);
      expect(result.rows, 60);

      result.dispose();
      src.dispose();
    });
  });

  group('ImageUtils.cropRect', () {
    test('should crop a valid region', () {
      final src = solidMat(100, 100, 0, 0, 255);
      final cropped = ImageUtils.cropRect(src, 10, 10, 50, 50);

      expect(cropped.cols, 40);
      expect(cropped.rows, 40);

      cropped.dispose();
      src.dispose();
    });

    test('should clamp coordinates to image bounds', () {
      final src = solidMat(100, 100, 0, 0, 255);
      final cropped = ImageUtils.cropRect(src, -10, -10, 200, 200);

      expect(cropped.cols, 100);
      expect(cropped.rows, 100);

      cropped.dispose();
      src.dispose();
    });
  });

  group('ImageUtils.cropFromNormalizedRoi', () {
    test('should crop using normalized coordinates', () {
      final src = solidMat(100, 100, 0, 255, 0);
      final cropped = ImageUtils.cropFromNormalizedRoi(src, 0.1, 0.1, 0.5, 0.5);

      expect(cropped.cols, 40);
      expect(cropped.rows, 40);

      cropped.dispose();
      src.dispose();
    });

    test('should handle full image crop', () {
      final src = solidMat(100, 100, 0, 255, 0);
      final cropped = ImageUtils.cropFromNormalizedRoi(src, 0.0, 0.0, 1.0, 1.0);

      expect(cropped.cols, 100);
      expect(cropped.rows, 100);

      cropped.dispose();
      src.dispose();
    });
  });

  group('ImageUtils.matToFloat32TensorMediaPipe', () {
    test('should convert BGR Mat to normalized RGB tensor in [-1,1]', () {
      // Create a 2x2 BGR Mat with known values
      final src = cv.Mat.fromList(2, 2, cv.MatType.CV_8UC3, [
        255, 0, 0, // pixel (0,0): B=255, G=0, R=0
        0, 255, 0, // pixel (1,0): B=0, G=255, R=0
        0, 0, 255, // pixel (0,1): B=0, G=0, R=255
        127, 127, 127, // pixel (1,1): B=127, G=127, R=127
      ]);

      final tensor = ImageUtils.matToFloat32TensorMediaPipe(src);

      expect(tensor.length, 12); // 2x2x3

      // Pixel (0,0): BGR(255,0,0) -> RGB: R=0/127.5-1=-1, G=0/127.5-1=-1, B=255/127.5-1=1
      expect(tensor[0], closeTo(-1.0, 0.01)); // R
      expect(tensor[1], closeTo(-1.0, 0.01)); // G
      expect(tensor[2], closeTo(1.0, 0.01)); // B

      // Pixel (1,0): BGR(0,255,0) -> RGB: R=0/127.5-1=-1, G=255/127.5-1=1, B=0/127.5-1=-1
      expect(tensor[3], closeTo(-1.0, 0.01)); // R
      expect(tensor[4], closeTo(1.0, 0.01)); // G
      expect(tensor[5], closeTo(-1.0, 0.01)); // B

      src.dispose();
    });

    test('should reuse provided buffer', () {
      final src = solidMat(2, 2, 128, 128, 128);
      final buffer = Float32List(12);
      final tensor =
          ImageUtils.matToFloat32TensorMediaPipe(src, buffer: buffer);

      expect(identical(tensor, buffer), isTrue);
      src.dispose();
    });
  });

  group('ImageUtils.matToFloat32Tensor', () {
    test('should convert BGR Mat to normalized RGB tensor in [0,1]', () {
      final src = cv.Mat.fromList(2, 2, cv.MatType.CV_8UC3, [
        255, 0, 0, // BGR(255,0,0) -> RGB: R=0, G=0, B=1.0
        0, 0, 255, // BGR(0,0,255) -> RGB: R=1.0, G=0, B=0
        0, 255, 0, // BGR(0,255,0) -> RGB: R=0, G=1.0, B=0
        255, 255, 255,
      ]);

      final tensor = ImageUtils.matToFloat32Tensor(src);

      expect(tensor.length, 12);

      // Pixel (0,0): BGR(255,0,0) -> R=0, G=0, B=1.0
      expect(tensor[0], closeTo(0.0, 0.01)); // R
      expect(tensor[1], closeTo(0.0, 0.01)); // G
      expect(tensor[2], closeTo(1.0, 0.01)); // B

      // Pixel (1,0): BGR(0,0,255) -> R=1.0, G=0, B=0
      expect(tensor[3], closeTo(1.0, 0.01)); // R
      expect(tensor[4], closeTo(0.0, 0.01)); // G
      expect(tensor[5], closeTo(0.0, 0.01)); // B

      src.dispose();
    });

    test('should reuse provided buffer', () {
      final src = solidMat(2, 2, 128, 128, 128);
      final buffer = Float32List(12);
      final tensor = ImageUtils.matToFloat32Tensor(src, buffer: buffer);

      expect(identical(tensor, buffer), isTrue);
      src.dispose();
    });
  });

  group('ImageUtils.letterbox', () {
    test('should letterbox a square image to square target', () {
      final src = solidMat(100, 100, 128, 128, 128);
      final result = ImageUtils.letterbox(src, 50, 50);

      expect(result.padded.cols, 50);
      expect(result.padded.rows, 50);
      expect(result.scale, closeTo(0.5, 0.01));
      expect(result.padLeft, 0);
      expect(result.padTop, 0);

      result.padded.dispose();
      src.dispose();
    });

    test('should letterbox a landscape image with vertical padding', () {
      final src = solidMat(200, 100, 0, 0, 128);
      final result = ImageUtils.letterbox(src, 100, 100);

      expect(result.padded.cols, 100);
      expect(result.padded.rows, 100);
      expect(result.scale, closeTo(0.5, 0.01));
      expect(result.padLeft, 0);
      expect(result.padTop, 25);

      result.padded.dispose();
      src.dispose();
    });

    test('should letterbox a portrait image with horizontal padding', () {
      final src = solidMat(100, 200, 128, 0, 0);
      final result = ImageUtils.letterbox(src, 100, 100);

      expect(result.padded.cols, 100);
      expect(result.padded.rows, 100);
      expect(result.scale, closeTo(0.5, 0.01));
      expect(result.padLeft, 25);
      expect(result.padTop, 0);

      result.padded.dispose();
      src.dispose();
    });
  });

  group('ImageUtils.matToNHWC4DMediaPipe', () {
    test('should create 4D tensor with correct dimensions', () {
      final src = solidMat(3, 2, 128, 128, 128);
      final tensor = ImageUtils.matToNHWC4DMediaPipe(src, 3, 2);

      expect(tensor.length, 1); // batch
      expect(tensor[0].length, 2); // height
      expect(tensor[0][0].length, 3); // width
      expect(tensor[0][0][0].length, 3); // channels

      src.dispose();
    });

    test('should normalize values to [-1, 1]', () {
      final src = cv.Mat.fromList(1, 1, cv.MatType.CV_8UC3, [0, 0, 255]);
      final tensor = ImageUtils.matToNHWC4DMediaPipe(src, 1, 1);

      // BGR(0,0,255) -> R=255/127.5-1=1.0, G=0/127.5-1=-1.0, B=0/127.5-1=-1.0
      expect(tensor[0][0][0][0], closeTo(1.0, 0.01)); // R
      expect(tensor[0][0][0][1], closeTo(-1.0, 0.01)); // G
      expect(tensor[0][0][0][2], closeTo(-1.0, 0.01)); // B

      src.dispose();
    });

    test('should reuse provided buffer', () {
      final src = solidMat(2, 2, 128, 128, 128);
      final reuse = List.generate(
        1,
        (_) => List.generate(
          2,
          (_) => List.generate(
            2,
            (_) => List<double>.filled(3, 0.0),
            growable: false,
          ),
          growable: false,
        ),
        growable: false,
      );
      final tensor = ImageUtils.matToNHWC4DMediaPipe(src, 2, 2, reuse: reuse);
      expect(identical(tensor, reuse), isTrue);
      src.dispose();
    });
  });

  group('ImageUtils.decodeImage', () {
    test('should decode valid PNG bytes', () {
      final bytes = TestUtils.createDummyImageBytes();
      final mat = ImageUtils.decodeImage(bytes);

      expect(mat.isEmpty, isFalse);
      expect(mat.cols, 1);
      expect(mat.rows, 1);

      mat.dispose();
    });

    test('should return empty Mat for invalid bytes', () {
      final bytes = Uint8List.fromList([0, 1, 2, 3]);
      final mat = ImageUtils.decodeImage(bytes);

      expect(mat.isEmpty, isTrue);
      mat.dispose();
    });
  });
}
