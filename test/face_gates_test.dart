import 'package:flutter/widgets.dart' show Size;
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/src/native/face_native_lib.dart';
import 'package:face_detection_tflite/src/shared/face_gates.dart';

import 'test_config.dart';

/// Unit tests for the detection gates (`minScore` / `minFaceSize`).
///
/// These are pure-Dart tests over synthetic [Face] objects: no models, no
/// isolates, no image decode. They pin down the exact filtering contract used
/// by both the native and web pipelines (which share [applyFaceGates]) plus the
/// [Face.widthFraction] math and [validateFaceGates] error handling.
void main() {
  globalTestSetup();

  // Builds a face with a chosen confidence and a normalized horizontal box
  // [xmin, xmax] over an [imgW] x [imgH] image. The vertical extent is fixed;
  // only width matters for these tests.
  Face makeFace({
    required double score,
    required double xmin,
    required double xmax,
    double imgW = 100,
    double imgH = 100,
  }) {
    final size = Size(imgW, imgH);
    return Face(
      detection: Detection(
        boundingBox: RectF(xmin, 0.2, xmax, 0.8),
        score: score,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: size,
      ),
      mesh: null,
      irises: const [],
      originalSize: size,
    );
  }

  group('validateFaceGates', () {
    test('accepts boundary and mid-range values', () {
      expect(
        () => validateFaceGates(minScore: 0.0, minFaceSize: 0.0),
        returnsNormally,
      );
      expect(
        () => validateFaceGates(minScore: 1.0, minFaceSize: 1.0),
        returnsNormally,
      );
      expect(
        () => validateFaceGates(minScore: 0.5, minFaceSize: 0.1),
        returnsNormally,
      );
    });

    test('rejects NaN on either parameter', () {
      expect(
        () => validateFaceGates(minScore: double.nan, minFaceSize: 0),
        throwsArgumentError,
      );
      expect(
        () => validateFaceGates(minScore: 0, minFaceSize: double.nan),
        throwsArgumentError,
      );
    });

    test('rejects out-of-range values and infinities', () {
      expect(
        () => validateFaceGates(minScore: -0.01, minFaceSize: 0),
        throwsArgumentError,
      );
      expect(
        () => validateFaceGates(minScore: 1.01, minFaceSize: 0),
        throwsArgumentError,
      );
      expect(
        () => validateFaceGates(minScore: 0, minFaceSize: -1),
        throwsArgumentError,
      );
      expect(
        () => validateFaceGates(minScore: 0, minFaceSize: 2),
        throwsArgumentError,
      );
      expect(
        () => validateFaceGates(minScore: double.infinity, minFaceSize: 0),
        throwsArgumentError,
      );
      expect(
        () => validateFaceGates(
          minScore: 0,
          minFaceSize: double.negativeInfinity,
        ),
        throwsArgumentError,
      );
    });
  });

  group('Face.widthFraction', () {
    test('in-frame box is width / imageWidth', () {
      expect(
        makeFace(score: 0.9, xmin: 0.1, xmax: 0.6).widthFraction,
        closeTo(0.5, 1e-9),
      );
    });

    test('box extending past the right edge is clipped to the image', () {
      // Raw width 0.6 of image, but only 0.2 is visible.
      expect(
        makeFace(score: 0.9, xmin: 0.8, xmax: 1.4).widthFraction,
        closeTo(0.2, 1e-9),
      );
    });

    test('box extending past the left edge is clipped to the image', () {
      // Raw width 0.5, visible 0.3.
      expect(
        makeFace(score: 0.9, xmin: -0.2, xmax: 0.3).widthFraction,
        closeTo(0.3, 1e-9),
      );
    });

    test('never exceeds 1.0 even for an oversized box', () {
      expect(
        makeFace(score: 0.9, xmin: -0.5, xmax: 1.5).widthFraction,
        closeTo(1.0, 1e-9),
      );
    });

    test('zero image width yields 0.0', () {
      expect(
        makeFace(score: 0.9, xmin: 0.1, xmax: 0.6, imgW: 0).widthFraction,
        0.0,
      );
    });

    test('box entirely outside the image yields 0.0', () {
      expect(makeFace(score: 0.9, xmin: 1.2, xmax: 1.5).widthFraction, 0.0);
    });
  });

  group('applyFaceGates', () {
    test('no-op defaults return the exact same list instance', () {
      final list = [makeFace(score: 0.9, xmin: 0.1, xmax: 0.6)];
      expect(
        identical(applyFaceGates(list, minScore: 0, minFaceSize: 0), list),
        isTrue,
      );
    });

    test('empty input yields empty output', () {
      expect(
        applyFaceGates(<Face>[], minScore: 0.6, minFaceSize: 0.1),
        isEmpty,
      );
    });

    test('score gate is inclusive at the threshold', () {
      final f = makeFace(score: 0.6, xmin: 0.1, xmax: 0.6);
      expect(applyFaceGates([f], minScore: 0.6, minFaceSize: 0), hasLength(1));
      expect(applyFaceGates([f], minScore: 0.61, minFaceSize: 0), isEmpty);
    });

    test('size gate is inclusive at the threshold', () {
      final f = makeFace(score: 0.9, xmin: 0.1, xmax: 0.6); // widthFraction 0.5
      expect(applyFaceGates([f], minScore: 0, minFaceSize: 0.5), hasLength(1));
      expect(applyFaceGates([f], minScore: 0, minFaceSize: 0.51), isEmpty);
    });

    test('combined gates require both conditions', () {
      final tooSmall = makeFace(score: 0.99, xmin: 0.1, xmax: 0.2); // frac 0.1
      final tooLow = makeFace(score: 0.55, xmin: 0.1, xmax: 0.9); // frac 0.8
      final good = makeFace(score: 0.9, xmin: 0.1, xmax: 0.9); // frac 0.8
      final out = applyFaceGates(
        [tooSmall, tooLow, good],
        minScore: 0.7,
        minFaceSize: 0.3,
      );
      expect(out, hasLength(1));
      expect(out.single.score, 0.9);
    });

    test('preserves input order', () {
      final a = makeFace(score: 0.90, xmin: 0.0, xmax: 0.9);
      final b = makeFace(score: 0.80, xmin: 0.0, xmax: 0.8);
      final c = makeFace(score: 0.95, xmin: 0.0, xmax: 0.7);
      final out = applyFaceGates([a, b, c], minScore: 0.5, minFaceSize: 0);
      expect(identical(out[0], a), isTrue);
      expect(identical(out[1], b), isTrue);
      expect(identical(out[2], c), isTrue);
    });

    test('degenerate image width is dropped only when size gating', () {
      final degen = makeFace(score: 0.9, xmin: 0.1, xmax: 0.6, imgW: 0);
      expect(applyFaceGates([degen], minScore: 0, minFaceSize: 0.1), isEmpty);
      expect(
        applyFaceGates([degen], minScore: 0, minFaceSize: 0),
        hasLength(1),
      );
    });

    test('minScore below the internal 0.5 floor never adds faces', () {
      // Detection never returns candidates below 0.5, so a sub-0.5 minScore is a
      // no-op: it cannot surface anything, only keep what is already present.
      final f = makeFace(score: 0.5, xmin: 0.1, xmax: 0.6);
      expect(applyFaceGates([f], minScore: 0.3, minFaceSize: 0), hasLength(1));
      expect(applyFaceGates([f], minScore: 0.5, minFaceSize: 0), hasLength(1));
    });
  });
}
