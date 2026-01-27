// ignore_for_file: avoid_print

/// Integration tests for ImageUtils OpenCV operations.
///
/// Tests cover:
/// - keepAspectResizeAndPad: aspect-preserving resize with padding
/// - rotateAndCropSquare: rotation and cropping
/// - cropRect / cropFromNormalizedRoi: region cropping
/// - letterbox: letterbox preprocessing
/// - matToFloat32Tensor*: tensor conversion with normalization
/// - matToNHWC4DMediaPipe: 4D tensor conversion
library;

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testTimeout = Timeout(Duration(minutes: 2));

  /// Creates a solid color cv.Mat of specified size
  cv.Mat createSolidMat(int width, int height,
      {int r = 128, int g = 128, int b = 128}) {
    final mat = cv.Mat.zeros(height, width, cv.MatType.CV_8UC3);
    mat.setTo(cv.Scalar(b.toDouble(), g.toDouble(), r.toDouble(), 255));
    return mat;
  }

  /// Creates a gradient cv.Mat for testing transformations
  /// Uses setTo approach since per-pixel channel access is unreliable in opencv_dart
  cv.Mat createGradientMat(int width, int height) {
    final mat = cv.Mat.zeros(height, width, cv.MatType.CV_8UC3);
    mat.setTo(cv.Scalar(128, 64, 192, 255));
    return mat;
  }

  group('ImageUtils.keepAspectResizeAndPad', () {
    test('square image maintains aspect ratio', () {
      final mat = createSolidMat(100, 100, r: 255, g: 0, b: 0);

      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(mat, 200, 200);

      expect(padded.cols, 200);
      expect(padded.rows, 200);
      padded.dispose();
      resized.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('landscape image adds top/bottom padding', () {
      final mat = createSolidMat(200, 100, r: 0, g: 255, b: 0);

      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(mat, 200, 200);

      expect(padded.cols, 200);
      expect(padded.rows, 200);
      expect(resized.cols, 200);
      expect(resized.rows, 100);

      final topPixel = padded.at<cv.Vec3b>(0, 100);
      expect(topPixel.val1, 0);
      expect(topPixel.val2, 0);
      expect(topPixel.val3, 0);

      padded.dispose();
      resized.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('portrait image adds left/right padding', () {
      final mat = createSolidMat(100, 200, r: 0, g: 0, b: 255);

      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(mat, 200, 200);

      expect(padded.cols, 200);
      expect(padded.rows, 200);
      expect(resized.cols, 100);
      expect(resized.rows, 200);

      final leftPixel = padded.at<cv.Vec3b>(100, 0);
      expect(leftPixel.val1, 0);
      expect(leftPixel.val2, 0);
      expect(leftPixel.val3, 0);

      padded.dispose();
      resized.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('handles small image upscaling', () {
      final mat = createSolidMat(10, 10);

      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(mat, 100, 100);

      expect(padded.cols, 100);
      expect(padded.rows, 100);

      padded.dispose();
      resized.dispose();
      mat.dispose();
    }, timeout: testTimeout);
  });

  group('ImageUtils.rotateAndCropSquare', () {
    test('no rotation extracts centered square', () {
      final mat = createGradientMat(100, 100);

      final result = ImageUtils.rotateAndCropSquare(mat, 50.0, 50.0, 40.0, 0.0);

      expect(result, isNotNull);
      expect(result!.cols, 40);
      expect(result.rows, 40);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('45 degree rotation', () {
      final mat = createSolidMat(100, 100, r: 255, g: 128, b: 64);

      final result =
          ImageUtils.rotateAndCropSquare(mat, 50.0, 50.0, 30.0, 3.14159 / 4);

      expect(result, isNotNull);
      expect(result!.cols, 30);
      expect(result.rows, 30);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('90 degree rotation', () {
      final mat = createGradientMat(100, 100);

      final result =
          ImageUtils.rotateAndCropSquare(mat, 50.0, 50.0, 40.0, 3.14159 / 2);

      expect(result, isNotNull);
      expect(result!.cols, 40);
      expect(result.rows, 40);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('returns null for size <= 0', () {
      final mat = createSolidMat(100, 100);

      final result1 = ImageUtils.rotateAndCropSquare(mat, 50.0, 50.0, 0.0, 0.0);
      final result2 =
          ImageUtils.rotateAndCropSquare(mat, 50.0, 50.0, -10.0, 0.0);

      expect(result1, isNull);
      expect(result2, isNull);

      mat.dispose();
    }, timeout: testTimeout);

    test('handles extraction near image boundary', () {
      final mat = createSolidMat(100, 100, r: 200, g: 100, b: 50);

      final result = ImageUtils.rotateAndCropSquare(mat, 10.0, 10.0, 30.0, 0.0);

      expect(result, isNotNull);
      expect(result!.cols, 30);
      expect(result.rows, 30);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);
  });

  group('ImageUtils.cropRect', () {
    test('normal crop within bounds', () {
      final mat = createGradientMat(100, 100);

      final cropped = ImageUtils.cropRect(mat, 20, 30, 80, 70);

      expect(cropped.cols, 60);
      expect(cropped.rows, 40);

      cropped.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('clamps to image boundaries', () {
      final mat = createSolidMat(100, 100);

      final cropped = ImageUtils.cropRect(mat, -10, -10, 150, 150);

      expect(cropped.cols, greaterThan(0));
      expect(cropped.rows, greaterThan(0));
      expect(cropped.cols, lessThanOrEqualTo(100));
      expect(cropped.rows, lessThanOrEqualTo(100));

      cropped.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('crop at edges', () {
      final mat = createSolidMat(100, 100);

      final croppedTopLeft = ImageUtils.cropRect(mat, 0, 0, 50, 50);
      final croppedBottomRight = ImageUtils.cropRect(mat, 50, 50, 100, 100);

      expect(croppedTopLeft.cols, 50);
      expect(croppedTopLeft.rows, 50);
      expect(croppedBottomRight.cols, 50);
      expect(croppedBottomRight.rows, 50);

      croppedTopLeft.dispose();
      croppedBottomRight.dispose();
      mat.dispose();
    }, timeout: testTimeout);
  });

  group('ImageUtils.cropFromNormalizedRoi', () {
    test('converts normalized coords to pixels', () {
      final mat = createGradientMat(200, 100);

      final cropped =
          ImageUtils.cropFromNormalizedRoi(mat, 0.25, 0.25, 0.75, 0.75);

      expect(cropped.cols, 100);
      expect(cropped.rows, 50);

      cropped.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('full image (0,0,1,1) returns full size', () {
      final mat = createSolidMat(100, 80);

      final cropped = ImageUtils.cropFromNormalizedRoi(mat, 0.0, 0.0, 1.0, 1.0);

      expect(cropped.cols, 100);
      expect(cropped.rows, 80);

      cropped.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('small normalized region', () {
      final mat = createSolidMat(100, 100);

      final cropped = ImageUtils.cropFromNormalizedRoi(mat, 0.4, 0.4, 0.6, 0.6);

      expect(cropped.cols, 20);
      expect(cropped.rows, 20);

      cropped.dispose();
      mat.dispose();
    }, timeout: testTimeout);
  });

  group('ImageUtils.letterbox', () {
    test('returns correct scale and padding for landscape', () {
      final mat = createSolidMat(200, 100);

      final result = ImageUtils.letterbox(mat, 100, 100);

      expect(result.padded.cols, 100);
      expect(result.padded.rows, 100);
      expect(result.scale, closeTo(0.5, 0.01));
      expect(result.padLeft, 0);
      expect(result.padTop, greaterThan(0));

      result.padded.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('returns correct scale and padding for portrait', () {
      final mat = createSolidMat(100, 200);

      final result = ImageUtils.letterbox(mat, 100, 100);

      expect(result.padded.cols, 100);
      expect(result.padded.rows, 100);
      expect(result.scale, closeTo(0.5, 0.01));
      expect(result.padLeft, greaterThan(0));
      expect(result.padTop, 0);

      result.padded.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('already matching dimensions has minimal padding', () {
      final mat = createSolidMat(100, 100);

      final result = ImageUtils.letterbox(mat, 100, 100);

      expect(result.padded.cols, 100);
      expect(result.padded.rows, 100);
      expect(result.scale, closeTo(1.0, 0.01));
      expect(result.padLeft, 0);
      expect(result.padTop, 0);

      result.padded.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('upscales small image correctly', () {
      final mat = createSolidMat(50, 50);

      final result = ImageUtils.letterbox(mat, 100, 100);

      expect(result.padded.cols, 100);
      expect(result.padded.rows, 100);
      expect(result.scale, closeTo(2.0, 0.01));

      result.padded.dispose();
      mat.dispose();
    }, timeout: testTimeout);
  });

  group('ImageUtils.matToFloat32TensorMediaPipe', () {
    test('BGR to RGB conversion', () {
      final mat = createSolidMat(4, 4, r: 0, g: 0, b: 255);

      final tensor = ImageUtils.matToFloat32TensorMediaPipe(mat);

      expect(tensor[0], closeTo(-1.0, 0.01));
      expect(tensor[1], closeTo(-1.0, 0.01));
      expect(tensor[2], closeTo(1.0, 0.01));

      mat.dispose();
    }, timeout: testTimeout);

    test('normalizes to [-1, 1] range', () {
      final mat = createSolidMat(2, 2, r: 128, g: 128, b: 128);

      final tensor = ImageUtils.matToFloat32TensorMediaPipe(mat);

      for (int i = 0; i < tensor.length; i++) {
        expect(tensor[i], closeTo(0.0, 0.02));
      }

      mat.dispose();
    }, timeout: testTimeout);

    test('buffer reuse works correctly', () {
      final mat = createSolidMat(10, 10, r: 200, g: 100, b: 50);
      final buffer = Float32List(10 * 10 * 3);

      final tensor =
          ImageUtils.matToFloat32TensorMediaPipe(mat, buffer: buffer);

      expect(identical(tensor, buffer), isTrue);
      expect(tensor.length, 300);

      mat.dispose();
    }, timeout: testTimeout);

    test('handles larger image', () {
      final mat = createGradientMat(64, 64);

      final tensor = ImageUtils.matToFloat32TensorMediaPipe(mat);

      expect(tensor.length, 64 * 64 * 3);
      for (final v in tensor) {
        expect(v, greaterThanOrEqualTo(-1.0));
        expect(v, lessThanOrEqualTo(1.0));
      }

      mat.dispose();
    }, timeout: testTimeout);
  });

  group('ImageUtils.matToFloat32Tensor', () {
    test('normalizes to [0, 1] range', () {
      final mat = createSolidMat(4, 4, r: 255, g: 0, b: 0);

      final tensor = ImageUtils.matToFloat32Tensor(mat);

      expect(tensor[0], closeTo(1.0, 0.01));
      expect(tensor[1], closeTo(0.0, 0.01));
      expect(tensor[2], closeTo(0.0, 0.01));

      mat.dispose();
    }, timeout: testTimeout);

    test('black image produces zeros', () {
      final mat = createSolidMat(4, 4, r: 0, g: 0, b: 0);

      final tensor = ImageUtils.matToFloat32Tensor(mat);

      for (final v in tensor) {
        expect(v, closeTo(0.0, 0.001));
      }

      mat.dispose();
    }, timeout: testTimeout);

    test('white image produces ones', () {
      final mat = createSolidMat(4, 4, r: 255, g: 255, b: 255);

      final tensor = ImageUtils.matToFloat32Tensor(mat);

      for (final v in tensor) {
        expect(v, closeTo(1.0, 0.001));
      }

      mat.dispose();
    }, timeout: testTimeout);
  });

  group('ImageUtils.matToNHWC4DMediaPipe', () {
    test('produces correct 4D shape', () {
      final mat = createSolidMat(8, 8, r: 100, g: 150, b: 200);

      final tensor = ImageUtils.matToNHWC4DMediaPipe(mat, 8, 8);

      expect(tensor.length, 1);
      expect(tensor[0].length, 8);
      expect(tensor[0][0].length, 8);
      expect(tensor[0][0][0].length, 3);

      mat.dispose();
    }, timeout: testTimeout);

    test('tensor reuse works', () {
      final mat1 = createSolidMat(4, 4, r: 50, g: 100, b: 150);
      final mat2 = createSolidMat(4, 4, r: 200, g: 150, b: 100);

      final tensor1 = ImageUtils.matToNHWC4DMediaPipe(mat1, 4, 4);
      final firstValue = tensor1[0][0][0][0];

      final tensor2 =
          ImageUtils.matToNHWC4DMediaPipe(mat2, 4, 4, reuse: tensor1);

      expect(identical(tensor1, tensor2), isTrue);
      expect(tensor2[0][0][0][0], isNot(equals(firstValue)));

      mat1.dispose();
      mat2.dispose();
    }, timeout: testTimeout);

    test('values are normalized to [-1, 1]', () {
      final mat = createGradientMat(4, 4);

      final tensor = ImageUtils.matToNHWC4DMediaPipe(mat, 4, 4);

      for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
          for (int c = 0; c < 3; c++) {
            expect(tensor[0][y][x][c], greaterThanOrEqualTo(-1.0));
            expect(tensor[0][y][x][c], lessThanOrEqualTo(1.0));
          }
        }
      }

      mat.dispose();
    }, timeout: testTimeout);
  });

  group('ImageUtils.decodeImage', () {
    test('decodes valid JPEG bytes', () {
      final mat = createSolidMat(10, 10, r: 128, g: 64, b: 32);
      final encoded = cv.imencode('.jpg', mat);

      final decoded = ImageUtils.decodeImage(encoded.$2);

      expect(decoded.cols, 10);
      expect(decoded.rows, 10);

      decoded.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('decodes valid PNG bytes', () {
      final mat = createSolidMat(15, 20, r: 255, g: 0, b: 128);
      final encoded = cv.imencode('.png', mat);

      final decoded = ImageUtils.decodeImage(encoded.$2);

      expect(decoded.cols, 15);
      expect(decoded.rows, 20);

      decoded.dispose();
      mat.dispose();
    }, timeout: testTimeout);
  });
}
