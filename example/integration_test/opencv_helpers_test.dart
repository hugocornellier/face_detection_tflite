// ignore_for_file: avoid_print

/// Integration tests for OpenCV-based helper functions in helpers.dart.
///
/// Tests cover:
/// - convertImageToTensorFromMat: tensor conversion with letterboxing
/// - extractAlignedSquareFromMat: rotated square extraction via warpAffine
/// - cropFromRoiMat: region cropping with normalized coordinates
/// - AlignedFaceFromMat: cv.Mat-based aligned face container
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
  cv.Mat createGradientMat(int width, int height) {
    final mat = cv.Mat.zeros(height, width, cv.MatType.CV_8UC3);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final r = (x * 255 ~/ width);
        final g = (y * 255 ~/ height);
        final b = 128;
        mat.set<int>(y, x, 0, b);
        mat.set<int>(y, x, 1, g);
        mat.set<int>(y, x, 2, r);
      }
    }
    return mat;
  }

  group('convertImageToTensorFromMat', () {
    test('produces correct tensor with no padding for square image', () {
      final mat = createSolidMat(100, 100, r: 128, g: 128, b: 128);

      final result = convertImageToTensorFromMat(mat, outW: 100, outH: 100);

      expect(result.width, 100);
      expect(result.height, 100);
      expect(result.tensorNHWC.length, 100 * 100 * 3);
      // No padding for same aspect ratio
      expect(result.padding[0], closeTo(0.0, 0.01)); // padTop
      expect(result.padding[1], closeTo(0.0, 0.01)); // padBottom
      expect(result.padding[2], closeTo(0.0, 0.01)); // padLeft
      expect(result.padding[3], closeTo(0.0, 0.01)); // padRight

      mat.dispose();
    }, timeout: testTimeout);

    test('adds padding for landscape image', () {
      final mat = createSolidMat(200, 100, r: 255, g: 0, b: 0);

      final result = convertImageToTensorFromMat(mat, outW: 200, outH: 200);

      expect(result.width, 200);
      expect(result.height, 200);
      // Should have top/bottom padding
      expect(result.padding[0], greaterThan(0.0)); // padTop
      expect(result.padding[1], greaterThan(0.0)); // padBottom
      expect(result.padding[2], closeTo(0.0, 0.01)); // padLeft
      expect(result.padding[3], closeTo(0.0, 0.01)); // padRight

      mat.dispose();
    }, timeout: testTimeout);

    test('adds padding for portrait image', () {
      final mat = createSolidMat(100, 200, r: 0, g: 255, b: 0);

      final result = convertImageToTensorFromMat(mat, outW: 200, outH: 200);

      expect(result.width, 200);
      expect(result.height, 200);
      // Should have left/right padding
      expect(result.padding[0], closeTo(0.0, 0.01)); // padTop
      expect(result.padding[1], closeTo(0.0, 0.01)); // padBottom
      expect(result.padding[2], greaterThan(0.0)); // padLeft
      expect(result.padding[3], greaterThan(0.0)); // padRight

      mat.dispose();
    }, timeout: testTimeout);

    test('normalizes BGR to RGB in [-1, 1] range', () {
      // Create mat with known BGR values
      final mat = cv.Mat.zeros(2, 2, cv.MatType.CV_8UC3);
      // Set pixel (0,0) to BGR(255, 0, 0) = pure blue in BGR
      mat.set<int>(0, 0, 0, 255); // B
      mat.set<int>(0, 0, 1, 0); // G
      mat.set<int>(0, 0, 2, 0); // R

      final result = convertImageToTensorFromMat(mat, outW: 2, outH: 2);

      // After BGR->RGB conversion: R=0, G=0, B=255
      // Normalized: R=-1, G=-1, B=1
      expect(result.tensorNHWC[0], closeTo(-1.0, 0.01)); // R
      expect(result.tensorNHWC[1], closeTo(-1.0, 0.01)); // G
      expect(result.tensorNHWC[2], closeTo(1.0, 0.01)); // B

      mat.dispose();
    }, timeout: testTimeout);

    test('buffer reuse works correctly', () {
      final mat = createSolidMat(10, 10, r: 200, g: 100, b: 50);
      final buffer = Float32List(10 * 10 * 3);

      final result =
          convertImageToTensorFromMat(mat, outW: 10, outH: 10, buffer: buffer);

      expect(identical(result.tensorNHWC, buffer), isTrue);

      mat.dispose();
    }, timeout: testTimeout);

    test('handles upscaling small image', () {
      final mat = createSolidMat(10, 10);

      final result = convertImageToTensorFromMat(mat, outW: 100, outH: 100);

      expect(result.width, 100);
      expect(result.height, 100);
      expect(result.tensorNHWC.length, 100 * 100 * 3);

      mat.dispose();
    }, timeout: testTimeout);
  });

  group('extractAlignedSquareFromMat', () {
    test('extracts centered square with no rotation', () {
      final mat = createGradientMat(100, 100);

      final result = extractAlignedSquareFromMat(mat, 50.0, 50.0, 40.0, 0.0);

      expect(result, isNotNull);
      expect(result!.cols, 40);
      expect(result.rows, 40);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('extracts with 45 degree rotation', () {
      final mat = createSolidMat(100, 100, r: 128, g: 64, b: 32);

      final result = extractAlignedSquareFromMat(
          mat, 50.0, 50.0, 30.0, 3.14159 / 4); // 45 degrees

      expect(result, isNotNull);
      expect(result!.cols, 30);
      expect(result.rows, 30);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('extracts with 90 degree rotation', () {
      final mat = createGradientMat(100, 100);

      final result = extractAlignedSquareFromMat(
          mat, 50.0, 50.0, 40.0, 3.14159 / 2); // 90 degrees

      expect(result, isNotNull);
      expect(result!.cols, 40);
      expect(result.rows, 40);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('extracts with negative rotation', () {
      final mat = createSolidMat(100, 100);

      final result = extractAlignedSquareFromMat(
          mat, 50.0, 50.0, 30.0, -3.14159 / 6); // -30 degrees

      expect(result, isNotNull);
      expect(result!.cols, 30);
      expect(result.rows, 30);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('returns null for size <= 0', () {
      final mat = createSolidMat(100, 100);

      final result1 = extractAlignedSquareFromMat(mat, 50.0, 50.0, 0.0, 0.0);
      final result2 = extractAlignedSquareFromMat(mat, 50.0, 50.0, -10.0, 0.0);

      expect(result1, isNull);
      expect(result2, isNull);

      mat.dispose();
    }, timeout: testTimeout);

    test('handles extraction near image boundary', () {
      final mat = createSolidMat(100, 100);

      // Extract near corner
      final result = extractAlignedSquareFromMat(mat, 10.0, 10.0, 30.0, 0.0);

      expect(result, isNotNull);
      expect(result!.cols, 30);
      expect(result.rows, 30);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('handles extraction outside image boundary', () {
      final mat = createSolidMat(100, 100);

      // Extract partially outside image - should fill with black
      final result = extractAlignedSquareFromMat(mat, 90.0, 90.0, 40.0, 0.0);

      expect(result, isNotNull);
      expect(result!.cols, 40);
      expect(result.rows, 40);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);
  });

  group('cropFromRoiMat', () {
    test('crops with normalized coordinates', () {
      final mat = createGradientMat(200, 100);

      // Crop center 50%
      final result = cropFromRoiMat(mat, RectF(0.25, 0.25, 0.75, 0.75));

      expect(result.cols, 100); // 50% of 200
      expect(result.rows, 50); // 50% of 100

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('full ROI (0,0,1,1) returns full size', () {
      final mat = createSolidMat(100, 80);

      final result = cropFromRoiMat(mat, RectF(0.0, 0.0, 1.0, 1.0));

      expect(result.cols, 100);
      expect(result.rows, 80);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('small ROI region', () {
      final mat = createSolidMat(100, 100);

      final result = cropFromRoiMat(mat, RectF(0.4, 0.4, 0.6, 0.6));

      expect(result.cols, 20);
      expect(result.rows, 20);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('ROI at top-left corner', () {
      final mat = createSolidMat(100, 100);

      final result = cropFromRoiMat(mat, RectF(0.0, 0.0, 0.5, 0.5));

      expect(result.cols, 50);
      expect(result.rows, 50);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);

    test('ROI at bottom-right corner', () {
      final mat = createSolidMat(100, 100);

      final result = cropFromRoiMat(mat, RectF(0.5, 0.5, 1.0, 1.0));

      expect(result.cols, 50);
      expect(result.rows, 50);

      result.dispose();
      mat.dispose();
    }, timeout: testTimeout);
  });

  group('AlignedFaceFromMat', () {
    test('stores cv.Mat reference correctly', () {
      final faceCrop = createSolidMat(112, 112, r: 200, g: 150, b: 100);

      final aligned = AlignedFaceFromMat(
        cx: 100.0,
        cy: 150.0,
        size: 112.0,
        theta: 0.1,
        faceCrop: faceCrop,
      );

      expect(aligned.cx, 100.0);
      expect(aligned.cy, 150.0);
      expect(aligned.size, 112.0);
      expect(aligned.theta, 0.1);
      expect(aligned.faceCrop.cols, 112);
      expect(aligned.faceCrop.rows, 112);

      faceCrop.dispose();
    }, timeout: testTimeout);

    test('handles negative theta', () {
      final faceCrop = createSolidMat(96, 96);

      final aligned = AlignedFaceFromMat(
        cx: 50.0,
        cy: 60.0,
        size: 96.0,
        theta: -0.5,
        faceCrop: faceCrop,
      );

      expect(aligned.theta, -0.5);

      faceCrop.dispose();
    }, timeout: testTimeout);
  });
}
