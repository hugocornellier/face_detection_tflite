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
    double? meshScore,
  }) {
    final size = Size(imgW, imgH);
    return Face(
      detection: Detection(
        boundingBox: RectF(xmin, 0.2, xmax, 0.8),
        score: score,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: size,
      ),
      mesh: meshScore == null
          ? null
          : FaceMesh(
              List<Point>.generate(kMeshPoints, (_) => const Point(0, 0, 0)),
              score: meshScore,
            ),
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

  group('applyFaceGates minFacePresenceConfidence', () {
    test('defaults to a no-op when left unspecified', () {
      // A face with a low mesh score is kept when the presence gate is not set,
      // preserving the pre-existing two-gate behavior for callers of the helper.
      final f = makeFace(score: 0.9, xmin: 0.1, xmax: 0.9, meshScore: 0.01);
      expect(applyFaceGates([f], minScore: 0, minFaceSize: 0), hasLength(1));
    });

    test('drops faces whose meshScore is below the threshold', () {
      final palm = makeFace(score: 0.9, xmin: 0.1, xmax: 0.9, meshScore: 0.05);
      expect(
        applyFaceGates(
          [palm],
          minScore: 0,
          minFaceSize: 0,
          minFacePresenceConfidence: 0.5,
        ),
        isEmpty,
      );
    });

    test('keeps faces whose meshScore meets the threshold (inclusive)', () {
      final real = makeFace(score: 0.9, xmin: 0.1, xmax: 0.9, meshScore: 0.5);
      expect(
        applyFaceGates(
          [real],
          minScore: 0,
          minFaceSize: 0,
          minFacePresenceConfidence: 0.5,
        ),
        hasLength(1),
      );
    });

    test('a null meshScore (fast mode / no presence output) always passes', () {
      final noMesh = makeFace(score: 0.9, xmin: 0.1, xmax: 0.9);
      expect(noMesh.meshScore, isNull);
      expect(
        applyFaceGates(
          [noMesh],
          minScore: 0,
          minFaceSize: 0,
          minFacePresenceConfidence: 0.9,
        ),
        hasLength(1),
      );
    });

    test('combines with the score and size gates', () {
      final faces = [
        makeFace(score: 0.9, xmin: 0.1, xmax: 0.9, meshScore: 1.0), // keep
        makeFace(score: 0.9, xmin: 0.1, xmax: 0.9, meshScore: 0.2), // low mesh
        makeFace(score: 0.9, xmin: 0.1, xmax: 0.2, meshScore: 1.0), // too small
      ];
      final kept = applyFaceGates(
        faces,
        minScore: 0.6,
        minFaceSize: 0.5,
        minFacePresenceConfidence: 0.5,
      );
      expect(kept, hasLength(1));
      expect(kept.single.meshScore, 1.0);
    });
  });

  group('validateFaceGates minFacePresenceConfidence', () {
    test('accepts boundary values', () {
      expect(
        () => validateFaceGates(
          minScore: 0,
          minFaceSize: 0,
          minFacePresenceConfidence: 0.0,
        ),
        returnsNormally,
      );
      expect(
        () => validateFaceGates(
          minScore: 0,
          minFaceSize: 0,
          minFacePresenceConfidence: 1.0,
        ),
        returnsNormally,
      );
    });

    test('rejects out-of-range and NaN', () {
      expect(
        () => validateFaceGates(
          minScore: 0,
          minFaceSize: 0,
          minFacePresenceConfidence: 1.5,
        ),
        throwsArgumentError,
      );
      expect(
        () => validateFaceGates(
          minScore: 0,
          minFaceSize: 0,
          minFacePresenceConfidence: -0.1,
        ),
        throwsArgumentError,
      );
      expect(
        () => validateFaceGates(
          minScore: 0,
          minFaceSize: 0,
          minFacePresenceConfidence: double.nan,
        ),
        throwsArgumentError,
      );
    });
  });

  group('boxVisibleWidthFraction', () {
    // The early (detector-stage) gate and the late (Face) gate must agree to
    // the last bit, so the helper must return exactly Face.widthFraction for
    // every box shape, including awkward widths where a normalized-space
    // reformulation could differ by one ULP.
    test('bit-identical to Face.widthFraction across edge boxes', () {
      final cases = <(double xmin, double xmax, double imgW)>[
        (0.1, 0.6, 100.0),
        (0.8, 1.4, 100.0), // past right edge
        (-0.2, 0.3, 100.0), // past left edge
        (-0.5, 1.5, 100.0), // oversized both sides
        (1.2, 1.5, 100.0), // entirely outside
        (0.1, 0.6, 0.0), // degenerate image width
        (0.0, 1.0, 640.0), // exact full width
        (1.0 / 3.0, 2.0 / 3.0, 1279.0), // non-terminating fractions
        (0.07717565415691328, 0.9, 1279.0), // threshold-like values
      ];
      for (final (xmin, xmax, imgW) in cases) {
        final f = makeFace(score: 0.9, xmin: xmin, xmax: xmax, imgW: imgW);
        final helper = boxVisibleWidthFraction(
          f.detectionData.boundingBox,
          imgW,
        );
        // Strict equality on purpose: closeTo() would hide ULP drift.
        expect(
          helper,
          f.widthFraction,
          reason: 'mismatch for box [$xmin, $xmax] over width $imgW',
        );
      }
    });
  });

  group('applyDetectionGates', () {
    Detection makeDetection({
      required double score,
      required double xmin,
      required double xmax,
    }) => Detection(
      boundingBox: RectF(xmin, 0.2, xmax, 0.8),
      score: score,
      keypointsXY: TestUtils.generateValidKeypoints(),
    );

    test('keeps exactly the detections whose Faces pass applyFaceGates', () {
      const imgW = 100.0;
      final specs = <(double score, double xmin, double xmax)>[
        (0.99, 0.1, 0.2), // small
        (0.55, 0.1, 0.9), // low score
        (0.90, 0.1, 0.9), // good
        (0.60, 0.8, 1.4), // clipped at right edge
        (0.70, -0.2, 0.3), // clipped at left edge
      ];
      final dets = [
        for (final (s, x0, x1) in specs)
          makeDetection(score: s, xmin: x0, xmax: x1),
      ];
      final faces = [
        for (final (s, x0, x1) in specs)
          makeFace(score: s, xmin: x0, xmax: x1, imgW: imgW),
      ];
      for (final (minScore, minFaceSize) in <(double, double)>[
        (0.0, 0.0),
        (0.7, 0.3),
        (0.6, 0.25),
        (0.9, 0.0),
        (0.0, 0.5),
        (1.0, 1.0),
      ]) {
        final keptDets = applyDetectionGates(
          dets,
          minScore: minScore,
          minFaceSize: minFaceSize,
          imageWidth: imgW,
        );
        final keptFaces = applyFaceGates(
          faces,
          minScore: minScore,
          minFaceSize: minFaceSize,
        );
        expect(
          keptDets.map((d) => d.score).toList(),
          keptFaces.map((f) => f.score).toList(),
          reason: 'gate ($minScore, $minFaceSize) diverged',
        );
      }
    });

    test('no-op defaults return the exact same list instance', () {
      final dets = [makeDetection(score: 0.9, xmin: 0.1, xmax: 0.6)];
      expect(
        identical(
          applyDetectionGates(
            dets,
            minScore: 0,
            minFaceSize: 0,
            imageWidth: 100.0,
          ),
          dets,
        ),
        isTrue,
      );
    });

    test('thresholds are inclusive, matching the late gate', () {
      final d = makeDetection(score: 0.6, xmin: 0.1, xmax: 0.6);
      final f = makeFace(score: 0.6, xmin: 0.1, xmax: 0.6);
      final frac = f.widthFraction;
      expect(
        applyDetectionGates(
          [d],
          minScore: 0.6,
          minFaceSize: frac,
          imageWidth: 100.0,
        ),
        hasLength(1),
      );
      expect(
        applyDetectionGates(
          [d],
          minScore: 0.6000000000000001,
          minFaceSize: 0,
          imageWidth: 100.0,
        ),
        isEmpty,
      );
    });

    test('preserves input order', () {
      final a = makeDetection(score: 0.90, xmin: 0.0, xmax: 0.9);
      final b = makeDetection(score: 0.80, xmin: 0.0, xmax: 0.8);
      final c = makeDetection(score: 0.95, xmin: 0.0, xmax: 0.7);
      final out = applyDetectionGates(
        [a, b, c],
        minScore: 0.5,
        minFaceSize: 0,
        imageWidth: 100.0,
      );
      expect(identical(out[0], a), isTrue);
      expect(identical(out[1], b), isTrue);
      expect(identical(out[2], c), isTrue);
    });
  });
}
