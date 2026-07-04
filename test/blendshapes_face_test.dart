import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/src/native/face_native_lib.dart';

import 'test_config.dart';

/// Tests for the Face-level blendshape API (probability getters, serialization)
/// and the model's output sanitizer. These are pure Dart / host-runnable; the
/// model-inference goldens run in `example/integration_test` where TFLite is
/// available.
void main() {
  globalTestSetup();

  // A distinct value per coefficient so any mis-indexed getter is obvious.
  List<double> distinctScores() =>
      List<double>.generate(kBlendshapeCount, (int i) => i / 100.0);

  Face faceWithScores(List<double>? scores) => Face(
    detection: Detection(
      boundingBox: const RectF(0.2, 0.2, 0.8, 0.8),
      score: 0.9,
      keypointsXY: TestUtils.generateValidKeypoints(),
      imageSize: TestConstants.mediumImage,
    ),
    mesh: null,
    irises: const <Point>[],
    blendshapeScores: scores,
    originalSize: TestConstants.mediumImage,
  );

  group('Face blendshape probability getters', () {
    test('map to the correct ML Kit coefficients', () {
      final List<double> s = distinctScores();
      final Face f = faceWithScores(s);

      // smiling = mean(mouthSmileLeft=44, mouthSmileRight=45)
      expect(f.smilingProbability, closeTo((s[44] + s[45]) / 2.0, 1e-9));
      // leftEyeOpen = 1 - eyeBlinkLeft(9); rightEyeOpen = 1 - eyeBlinkRight(10)
      expect(f.leftEyeOpenProbability, closeTo(1.0 - s[9], 1e-9));
      expect(f.rightEyeOpenProbability, closeTo(1.0 - s[10], 1e-9));
    });

    test('blendshapes exposes all 52 via Blendshape indexing', () {
      final List<double> s = distinctScores();
      final FaceBlendshapes b = faceWithScores(s).blendshapes!;
      expect(b.scores.length, 52);
      expect(b[Blendshape.eyeBlinkLeft], s[9]);
      expect(b[Blendshape.mouthSmileRight], s[45]);
      expect(b[Blendshape.noseSneerRight], s[51]);
    });

    test('probabilities are clamped into [0, 1]', () {
      // eyeBlink of -0.5 would give open probability 1.5; smile of 2.0 -> >1.
      final List<double> s = List<double>.filled(kBlendshapeCount, 0.0);
      s[9] = -0.5; // -> leftEyeOpen would be 1.5 pre-clamp
      s[44] = 2.0;
      s[45] = 2.0; // -> smiling would be 2.0 pre-clamp
      final Face f = faceWithScores(s);
      expect(f.leftEyeOpenProbability, 1.0);
      expect(f.smilingProbability, 1.0);
    });

    test('are null when no scores were computed', () {
      final Face f = faceWithScores(null);
      expect(f.blendshapes, isNull);
      expect(f.smilingProbability, isNull);
      expect(f.leftEyeOpenProbability, isNull);
      expect(f.rightEyeOpenProbability, isNull);
    });

    test('are null when the score vector is the wrong length', () {
      final Face f = faceWithScores(List<double>.filled(10, 0.5));
      expect(f.blendshapes, isNull);
      expect(f.smilingProbability, isNull);
    });
  });

  group('Face serialization carries blendshape scores', () {
    test('toMap/fromMap roundtrips the 52 scores', () {
      final List<double> s = distinctScores();
      final Face restored = Face.fromMap(faceWithScores(s).toMap());
      expect(restored.blendshapes, isNotNull);
      expect(restored.blendshapes!.scores, s);
      expect(restored.smilingProbability, closeTo((s[44] + s[45]) / 2.0, 1e-9));
    });

    test('toMap omits scores when absent; fromMap yields null', () {
      final Map<String, dynamic> map = faceWithScores(null).toMap();
      expect(map.containsKey('blendshapeScores'), isFalse);
      expect(Face.fromMap(map).blendshapes, isNull);
    });
  });

  group('FaceBlendshapes wrapper', () {
    test('rejects a wrong-length vector', () {
      expect(() => FaceBlendshapes(<double>[0.1, 0.2]), throwsA(anything));
    });

    test('toMap/fromMap roundtrips', () {
      final FaceBlendshapes b = FaceBlendshapes(distinctScores());
      final FaceBlendshapes restored = FaceBlendshapes.fromMap(b.toMap());
      expect(restored.scores, b.scores);
    });
  });

  group('model output sanitizer', () {
    test('clamps out-of-range values into [0, 1]', () {
      final Float32List raw = Float32List.fromList(
        List<double>.generate(
          52,
          (int i) => i == 0 ? -3.0 : (i == 1 ? 4.0 : 0.5),
        ),
      );
      final Float32List? out = testSanitizeBlendshapes(raw);
      expect(out, isNotNull);
      expect(out![0], 0.0);
      expect(out[1], 1.0);
      expect(out[2], 0.5);
    });

    test('returns null when any value is NaN', () {
      final Float32List raw = Float32List.fromList(List<double>.filled(52, 0.5))
        ..[7] = double.nan;
      expect(testSanitizeBlendshapes(raw), isNull);
    });
  });
}
