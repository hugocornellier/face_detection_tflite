import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart' show Point;
import 'package:face_detection_tflite/src/shared/blendshape_input.dart';

import 'test_config.dart';

/// Tests for the shared Blendshape V2 landmark packer, enum, and index tables.
///
/// These guard the two silent-failure risks the model is prone to: an incorrect
/// 146-index subset (wrong landmarks fed to the model) and mis-routed iris
/// points (left/right eye swap). Everything here is pure Dart, identical on
/// native and web.
void main() {
  globalTestSetup();

  // A mesh where point i is (i, i + 0.5) so any packed value reveals its source
  // index, and iris points are clearly disjoint from the mesh range.
  List<Point> buildMesh({int n = 468}) =>
      List<Point>.generate(n, (int i) => Point(i.toDouble(), i + 0.5, 0));
  List<Point> buildIris({int n = 152}) =>
      List<Point>.generate(n, (int i) => Point(1000.0 + i, 2000.0 + i, 0));

  group('kBlendshapeLandmarkSubset', () {
    test('has 146 unique, strictly increasing indices', () {
      expect(kBlendshapeLandmarkSubset.length, kBlendshapeLandmarkCount);
      expect(kBlendshapeLandmarkCount, 146);
      expect(kBlendshapeLandmarkSubset.toSet().length, 146, reason: 'unique');
      for (int i = 1; i < kBlendshapeLandmarkSubset.length; i++) {
        expect(
          kBlendshapeLandmarkSubset[i],
          greaterThan(kBlendshapeLandmarkSubset[i - 1]),
          reason: 'strictly increasing at $i',
        );
      }
    });

    test('first is 0, all in [0, 478), last ten are the iris slots', () {
      expect(kBlendshapeLandmarkSubset.first, 0);
      for (final int idx in kBlendshapeLandmarkSubset) {
        expect(idx, inInclusiveRange(0, 477));
      }
      expect(kBlendshapeLandmarkSubset.sublist(136), <int>[
        468,
        469,
        470,
        471,
        472,
        473,
        474,
        475,
        476,
        477,
      ]);
    });
  });

  group('Blendshape enum', () {
    test('has 52 values with _neutral at 0 and no tongueOut', () {
      expect(Blendshape.values.length, kBlendshapeCount);
      expect(kBlendshapeCount, 52);
      expect(Blendshape.neutral.index, 0);
      expect(Blendshape.neutral.label, '_neutral');
      expect(
        Blendshape.values.any((Blendshape b) => b.label == 'tongueOut'),
        isFalse,
      );
    });

    test('key ML Kit mapping indices are correct', () {
      expect(Blendshape.eyeBlinkLeft.index, 9);
      expect(Blendshape.eyeBlinkRight.index, 10);
      expect(Blendshape.mouthSmileLeft.index, 44);
      expect(Blendshape.mouthSmileRight.index, 45);
      expect(Blendshape.noseSneerRight.index, 51);
    });

    test('kBlendshapeNames mirrors the enum order and labels', () {
      expect(kBlendshapeNames.length, 52);
      for (final Blendshape b in Blendshape.values) {
        expect(kBlendshapeNames[b.index], b.label);
      }
      // Only _neutral carries the underscore convention.
      for (final Blendshape b in Blendshape.values) {
        if (b == Blendshape.neutral) continue;
        expect(b.label.startsWith('_'), isFalse, reason: b.name);
      }
    });

    test('kBlendshapeNames is unmodifiable', () {
      expect(() => kBlendshapeNames[0] = 'x', throwsUnsupportedError);
    });
  });

  group('packBlendshapeInput', () {
    test('returns a 292-float tensor for valid inputs', () {
      final Float32List? packed = packBlendshapeInput(buildMesh(), buildIris());
      expect(packed, isNotNull);
      expect(packed!.length, kBlendshapeInputFloats);
      expect(packed.length, 292);
    });

    test('emits x then y per landmark in subset order', () {
      final Float32List packed = packBlendshapeInput(buildMesh(), buildIris())!;
      // First subset index is 0 -> mesh[0] = (0, 0.5).
      expect(packed[0], 0.0);
      expect(packed[1], 0.5);
      // A non-eye mesh index routes straight from the mesh: 195 -> (195, 195.5).
      final int pos195 = kBlendshapeLandmarkSubset.indexOf(195);
      expect(pos195, greaterThanOrEqualTo(0));
      expect(kBlendshapeEyeRefineOffsets.containsKey(195), isFalse);
      expect(packed[pos195 * 2], 195.0);
      expect(packed[pos195 * 2 + 1], 195.5);
    });

    test(
      'routes slots 468-472 from the image-left eye iris (offsets 71-75)',
      () {
        final List<Point> mesh = buildMesh();
        final List<Point> iris = buildIris();
        // Distinct markers at the image-left eye's 5 iris keypoints.
        for (int k = 0; k < 5; k++) {
          iris[71 + k] = Point(70000.0 + k, 80000.0 + k, 0);
        }
        final Float32List packed = packBlendshapeInput(mesh, iris)!;
        for (int k = 0; k < 5; k++) {
          final int pos = kBlendshapeLandmarkSubset.indexOf(468 + k);
          expect(pos, 136 + k);
          expect(packed[pos * 2], 70000.0 + k, reason: 'slot ${468 + k} x');
          expect(packed[pos * 2 + 1], 80000.0 + k, reason: 'slot ${468 + k} y');
        }
      },
    );

    test(
      'routes slots 473-477 from the image-right eye iris (offsets 147-151)',
      () {
        final List<Point> mesh = buildMesh();
        final List<Point> iris = buildIris();
        for (int k = 0; k < 5; k++) {
          iris[147 + k] = Point(90000.0 + k, 95000.0 + k, 0);
        }
        final Float32List packed = packBlendshapeInput(mesh, iris)!;
        for (int k = 0; k < 5; k++) {
          final int pos = kBlendshapeLandmarkSubset.indexOf(473 + k);
          expect(pos, 141 + k);
          expect(packed[pos * 2], 90000.0 + k, reason: 'slot ${473 + k} x');
          expect(packed[pos * 2 + 1], 95000.0 + k, reason: 'slot ${473 + k} y');
        }
      },
    );

    test('the two eyes are not swapped', () {
      // Guards R2: the left-eye iris (offset 71) must NOT land in the
      // right-eye slots (473-477) and vice versa.
      final List<Point> mesh = buildMesh();
      final List<Point> iris = buildIris();
      iris[71] = const Point(11111.0, 22222.0, 0); // image-left iris center
      iris[147] = const Point(33333.0, 44444.0, 0); // image-right iris center
      final Float32List packed = packBlendshapeInput(mesh, iris)!;
      final int leftPos = kBlendshapeLandmarkSubset.indexOf(468) * 2;
      final int rightPos = kBlendshapeLandmarkSubset.indexOf(473) * 2;
      expect(packed[leftPos], 11111.0);
      expect(packed[leftPos + 1], 22222.0);
      expect(packed[rightPos], 33333.0);
      expect(packed[rightPos + 1], 44444.0);
    });

    test('returns null when the mesh is too short', () {
      expect(packBlendshapeInput(buildMesh(n: 467), buildIris()), isNull);
    });

    test('returns null when the iris stream is too short', () {
      expect(packBlendshapeInput(buildMesh(), buildIris(n: 151)), isNull);
    });

    test('accepts extra trailing points beyond the required minimums', () {
      // Some pipelines may carry 478 mesh points; only 0..467 are consumed.
      final Float32List? packed = packBlendshapeInput(
        buildMesh(n: 478),
        buildIris(n: 160),
      );
      expect(packed, isNotNull);
      expect(packed!.length, 292);
    });
  });

  group('eyelid-ring refinement (closed-eye fix)', () {
    test('maps 30 eyelid indices (15 per eye) to iris-contour offsets', () {
      expect(kBlendshapeEyeRefineOffsets.length, 30);
      // Every refined mesh index is part of the 146-subset the model reads.
      for (final int meshIdx in kBlendshapeEyeRefineOffsets.keys) {
        expect(
          kBlendshapeLandmarkSubset.contains(meshIdx),
          isTrue,
          reason: 'mesh index $meshIdx must be in the subset',
        );
      }
      // Offsets stay within each eye's 15-point eyelid ring (0..14 image-left,
      // 76..90 image-right), never reaching the iris keypoints (71-75/147-151).
      for (final int off in kBlendshapeEyeRefineOffsets.values) {
        final bool leftRing = off >= 0 && off <= 14;
        final bool rightRing = off >= 76 && off <= 90;
        expect(
          leftRing || rightRing,
          isTrue,
          reason: 'offset $off out of ring',
        );
      }
      expect(
        kBlendshapeEyeRefineOffsets.values.toSet().length,
        30,
        reason: 'each offset used once',
      );
    });

    test(
      'eyelid-ring indices route from the refined iris contour, not mesh',
      () {
        final List<Point> mesh = buildMesh();
        final List<Point> iris = buildIris();
        final Float32List packed = packBlendshapeInput(mesh, iris)!;
        // With mesh[i]=(i,i+.5) and iris[i]=(1000+i,2000+i), a refined slot must
        // carry the iris value (>=1000), proving it did not fall back to mesh.
        kBlendshapeEyeRefineOffsets.forEach((int meshIdx, int off) {
          final int pos = kBlendshapeLandmarkSubset.indexOf(meshIdx);
          expect(packed[pos * 2], 1000.0 + off, reason: 'x of mesh $meshIdx');
          expect(
            packed[pos * 2 + 1],
            2000.0 + off,
            reason: 'y of mesh $meshIdx',
          );
        });
      },
    );

    test(
      'upper/lower lid centers come from the vertically-tracking offsets',
      () {
        // The blink signal is the gap between upper-lid center (159 left / 386
        // right) and lower-lid center (145 left / 374 right); these must be the
        // refined points so closure registers.
        expect(kBlendshapeEyeRefineOffsets[159], 12); // left upper
        expect(kBlendshapeEyeRefineOffsets[145], 4); // left lower
        expect(kBlendshapeEyeRefineOffsets[386], 88); // right upper
        expect(kBlendshapeEyeRefineOffsets[374], 80); // right lower
      },
    );
  });
}
