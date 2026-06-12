// ignore_for_file: avoid_print

// Equivalence tests for the preprocessing fast paths:
//  1. SIMD tensor conversion (cvtColor + convertTo) vs the scalar Dart loop.
//  2. Single-resample extractAlignedSquare(outSize:) vs crop-then-cv.resize.
//     On a linear gradient bilinear interpolation is exact, so both must
//     agree to float rounding; any alignment-convention mismatch would show
//     up as a systematic offset.
//  3. convertImageToTensor letterbox output vs a reference implementation of
//     the pre-fast-path pipeline (resize + copyMakeBorder + Dart loop).
//
//   flutter test integration_test/preprocessing_equivalence_test.dart -d macos

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  Future<cv.Mat> loadSample() async {
    final bytes = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();
    return cv.imdecode(bytes, cv.IMREAD_COLOR);
  }

  test('SIMD bgrMatToSignedFloat32 matches scalar Dart loop', () async {
    final mat = await loadSample();
    final int totalPixels = mat.cols * mat.rows;

    final Float32List simd = bgrMatToSignedFloat32(
      mat,
      totalPixels: totalPixels,
    );
    final Float32List scalar = bgrBytesToSignedFloat32(
      bytes: mat.data,
      totalPixels: totalPixels,
    );

    double maxDiff = 0;
    for (int i = 0; i < simd.length; i++) {
      final d = (simd[i] - scalar[i]).abs();
      if (d > maxDiff) maxDiff = d;
    }
    print('SIMD vs scalar conversion max abs diff: $maxDiff');
    expect(maxDiff, lessThan(1e-5));
    mat.dispose();
  });

  test('extractAlignedSquare(outSize) matches crop-then-resize alignment', () {
    // Linear gradient: bilinear interpolation reproduces it exactly, so the
    // two resampling routes must agree pixel-for-pixel (within rounding).
    const int side = 600;
    final grad = cv.Mat.zeros(side, side, cv.MatType.CV_8UC3);
    for (int y = 0; y < side; y++) {
      for (int x = 0; x < side; x++) {
        grad.set(
          y,
          x,
          cv.Vec3b(
              (x * 255) ~/ side, (y * 255) ~/ side, ((x + y) * 127) ~/ side),
        );
      }
    }

    for (final (cx, cy, size, theta, outSize) in [
      (300.0, 300.0, 401.3, 0.0, 192),
      (250.0, 320.0, 333.7, 0.35, 192),
      (300.0, 280.0, 150.2, -0.6, 64),
      (310.0, 290.0, 48.0, 0.2, 64), // upscale case
    ]) {
      final direct = extractAlignedSquare(
        grad,
        cx,
        cy,
        size,
        theta,
        outSize: outSize,
      )!;
      final fullCrop = extractAlignedSquare(grad, cx, cy, size, theta)!;
      final resized = cv.resize(
          fullCrop,
          (
            outSize,
            outSize,
          ),
          interpolation: cv.INTER_LINEAR);

      final Uint8List a = direct.data;
      final Uint8List b = resized.data;
      expect(a.length, b.length);
      double sum = 0;
      int maxDiff = 0;
      for (int i = 0; i < a.length; i++) {
        final d = (a[i] - b[i]).abs();
        sum += d;
        if (d > maxDiff) maxDiff = d;
      }
      final mean = sum / a.length;
      print(
        'warp equivalence size=$size theta=$theta out=$outSize: '
        'mean=$mean max=$maxDiff',
      );
      // Gradient image: agreement should be within quantization noise.
      expect(mean, lessThan(1.0));
      expect(maxDiff, lessThanOrEqualTo(3));

      direct.dispose();
      fullCrop.dispose();
      resized.dispose();
    }
    grad.dispose();
  });

  test('extractAlignedSquare without outSize is byte-identical to before', () {
    // The outSize == null path must keep producing exactly the same pixels
    // (scale == 1 reduces the new math to the original placement).
    final rng = math.Random(7);
    final noise = cv.Mat.zeros(240, 240, cv.MatType.CV_8UC3);
    for (int y = 0; y < 240; y++) {
      for (int x = 0; x < 240; x++) {
        noise.set(
          y,
          x,
          cv.Vec3b(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256)),
        );
      }
    }
    final a = extractAlignedSquare(noise, 120, 118, 101.4, 0.3)!;
    final b = extractAlignedSquare(noise, 120, 118, 101.4, 0.3, outSize: 101)!;
    expect(a.data, b.data);
    a.dispose();
    b.dispose();
    noise.dispose();
  });

  test('convertImageToTensor matches pre-fast-path reference', () async {
    final mat = await loadSample();

    // Reference: the original implementation (unconditional resize +
    // copyMakeBorder + scalar conversion).
    ImageTensor reference(cv.Mat src, int outW, int outH) {
      final lbp = computeLetterboxParams(
        srcWidth: src.cols,
        srcHeight: src.rows,
        targetWidth: outW,
        targetHeight: outH,
      );
      final resized = cv.resize(
          src,
          (
            lbp.newWidth,
            lbp.newHeight,
          ),
          interpolation: cv.INTER_LINEAR);
      final padded = cv.copyMakeBorder(
        resized,
        lbp.padTop,
        lbp.padBottom,
        lbp.padLeft,
        lbp.padRight,
        cv.BORDER_CONSTANT,
        value: cv.Scalar.black,
      );
      resized.dispose();
      final tensor = bgrBytesToSignedFloat32(
        bytes: padded.data,
        totalPixels: outW * outH,
      );
      padded.dispose();
      return ImageTensor(
          tensor,
          [
            lbp.padTop / outH,
            lbp.padBottom / outH,
            lbp.padLeft / outW,
            lbp.padRight / outW,
          ],
          outW,
          outH);
    }

    // Letterboxed case (non-square source) and same-size fast-path case.
    final cases = <cv.Mat>[mat];
    final squareCrop = extractAlignedSquare(
      mat,
      400,
      400,
      500,
      0.1,
      outSize: 192,
    )!;
    cases.add(squareCrop);

    for (final src in cases) {
      final got = convertImageToTensor(src, outW: 192, outH: 192);
      final want = reference(src, 192, 192);
      expect(got.padding, want.padding);
      double maxDiff = 0;
      for (int i = 0; i < got.tensorNHWC.length; i++) {
        final d = (got.tensorNHWC[i] - want.tensorNHWC[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      print(
        'convertImageToTensor ${src.cols}x${src.rows} max abs diff: $maxDiff',
      );
      expect(maxDiff, lessThan(1e-5));
    }

    squareCrop.dispose();
    mat.dispose();
  });
}
