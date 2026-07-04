// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';
// packBlendshapeInput is intentionally not part of the public surface; the
// closed-eye regression exercises it directly (as the unit tests do).
import 'package:face_detection_tflite/src/shared/blendshape_input.dart'
    show packBlendshapeInput;

import 'blendshapes_golden_data.dart';
import 'blendshapes_closed_eye_data.dart';

/// Integration tests for the MediaPipe Blendshape V2 classification feature
/// (smile / eye-open probabilities + the 52 raw coefficients), which run only
/// where the native TFLite runtime is available.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Golden A: model-only I/O vs official MediaPipe testdata', () {
    test('reproduces face_blendshapes_out from face_blendshapes_in', () async {
      final ByteData data = await rootBundle.load(
        'packages/face_detection_tflite/assets/models/face_blendshapes.tflite',
      );
      final FaceBlendshapesModel model =
          await FaceBlendshapesModel.createFromBuffer(
              data.buffer.asUint8List());
      try {
        final Float32List packed = Float32List.fromList(
          kGoldenBlendshapeInput,
        );
        final Float32List? out = await model.call(packed);
        expect(out, isNotNull);
        expect(out!.length, 52);
        // fp16 model; the reference interpreter reproduces the golden to
        // < 2e-6, so on-device deltas stay tiny. Allow generous slack for
        // delegate/hardware differences.
        double maxErr = 0.0;
        for (int i = 0; i < 52; i++) {
          final double e = (out[i] - kGoldenBlendshapeExpected[i]).abs();
          if (e > maxErr) maxErr = e;
        }
        print('Golden A max abs error: $maxErr');
        expect(maxErr, lessThan(2e-2));
      } finally {
        model.dispose();
      }
    });
  });

  group('Full-mode classification on a real image', () {
    late FaceDetector detector;

    setUpAll(() async {
      detector = FaceDetector();
      await detector.initialize();
    });

    tearDownAll(() => detector.dispose());

    test('landmark-ex1.jpg yields valid probabilities in full mode', () async {
      final ByteData data = await rootBundle.load(
        'assets/samples/landmark-ex1.jpg',
      );
      final List<Face> faces = await detector.detectFacesFromBytes(
        data.buffer.asUint8List(),
        mode: FaceDetectionMode.full,
      );
      expect(faces, isNotEmpty);

      for (final Face f in faces) {
        final FaceBlendshapes? b = f.blendshapes;
        expect(b, isNotNull, reason: 'full mode should populate blendshapes');
        expect(b!.scores.length, 52);
        for (final double s in b.scores) {
          expect(s, inInclusiveRange(0.0, 1.0));
        }
        expect(f.smilingProbability, isNotNull);
        expect(f.smilingProbability, inInclusiveRange(0.0, 1.0));
        expect(f.leftEyeOpenProbability, inInclusiveRange(0.0, 1.0));
        expect(f.rightEyeOpenProbability, inInclusiveRange(0.0, 1.0));
        // Named indexing agrees with the derived getters.
        expect(
          f.leftEyeOpenProbability,
          closeTo(1.0 - b[Blendshape.eyeBlinkLeft], 1e-6),
        );
      }
    });

    test('fast and standard modes leave classification null', () async {
      final ByteData data = await rootBundle.load(
        'assets/samples/landmark-ex1.jpg',
      );
      final Uint8List bytes = data.buffer.asUint8List();
      for (final FaceDetectionMode mode in <FaceDetectionMode>[
        FaceDetectionMode.fast,
        FaceDetectionMode.standard,
      ]) {
        final List<Face> faces = await detector.detectFacesFromBytes(
          bytes,
          mode: mode,
        );
        for (final Face f in faces) {
          expect(f.blendshapes, isNull, reason: '$mode must not classify');
          expect(f.smilingProbability, isNull);
          expect(f.leftEyeOpenProbability, isNull);
        }
      }
    });

    test('multiple faces each carry an independent score vector', () async {
      final ByteData data = await rootBundle.load(
        'assets/samples/group-shot-bounding-box-ex1.jpeg',
      );
      final List<Face> faces = await detector.detectFacesFromBytes(
        data.buffer.asUint8List(),
        mode: FaceDetectionMode.full,
      );
      final List<Face> classified =
          faces.where((Face f) => f.blendshapes != null).toList();
      expect(classified, isNotEmpty,
          reason: 'group shot should classify at least one face');
      // No cross-face buffer leakage: distinct list instances, and real
      // different faces should not produce byte-identical vectors. Only
      // meaningful when more than one face was classified.
      for (int i = 0; i < classified.length; i++) {
        for (int j = i + 1; j < classified.length; j++) {
          final List<double> a = classified[i].blendshapes!.scores;
          final List<double> b = classified[j].blendshapes!.scores;
          expect(identical(a, b), isFalse);
          bool allEqual = true;
          for (int k = 0; k < 52; k++) {
            if ((a[k] - b[k]).abs() > 1e-9) {
              allEqual = false;
              break;
            }
          }
          expect(allEqual, isFalse, reason: 'faces $i and $j share a vector');
        }
      }
    });
  });

  group('Closed-eye regression: eyelid refinement drives eyeBlink', () {
    // Guards the bug where leftEyeOpenProbability / rightEyeOpenProbability
    // could not detect a shut eye: the coarse 468-mesh keeps the eyelids in a
    // canonical open configuration, so the blendshape model read eyeBlink~=0.05
    // (eye-open ~0.95) even on fully closed eyes. packBlendshapeInput now routes
    // the eyelid ring from the iris model's refined contour, which collapses on
    // closure. The fixture is real landmarks captured from a verified
    // eyes-shut face; if the refinement is removed, packing falls back to the
    // open-looking coarse eyelids and these expectations fail.
    List<Point> pts(Float32List xy) => List<Point>.generate(
        xy.length ~/ 2, (int i) => Point(xy[i * 2], xy[i * 2 + 1], 0));

    test('shut-eye landmarks yield high eyeBlink (low eye-open)', () async {
      final ByteData data = await rootBundle.load(
        'packages/face_detection_tflite/assets/models/face_blendshapes.tflite',
      );
      final FaceBlendshapesModel model =
          await FaceBlendshapesModel.createFromBuffer(
              data.buffer.asUint8List());
      try {
        final Float32List? packed = packBlendshapeInput(
          pts(kClosedEyeMeshXY),
          pts(kClosedEyeIrisXY),
        );
        expect(packed, isNotNull, reason: 'fixture must pack');
        final Float32List? out = await model.call(packed!);
        expect(out, isNotNull);
        expect(out!.length, 52);
        final double blinkL = out[Blendshape.eyeBlinkLeft.index];
        final double blinkR = out[Blendshape.eyeBlinkRight.index];
        print('closed-eye blink L=$blinkL R=$blinkR '
            '(eye-open ${1 - blinkL} / ${1 - blinkR})');
        // Eyes are shut -> blink high, eye-open low. The pre-fix coarse path
        // produced blink ~0.05; a generous 0.4 bound cleanly separates the two
        // while tolerating delegate/hardware variation.
        expect(blinkL, greaterThan(0.4),
            reason: 'left eye shut: eyeBlinkLeft should fire');
        expect(blinkR, greaterThan(0.4),
            reason: 'right eye shut: eyeBlinkRight should fire');
        // Sanity: the mouth path is untouched by the eyelid refinement.
        expect(
            out[Blendshape.mouthSmileLeft.index], inInclusiveRange(0.0, 1.0));
      } finally {
        model.dispose();
      }
    });
  });

  group('CompiledModel vs Interpreter parity', () {
    test('produce close blendshape scores on the same image', () async {
      final ByteData data = await rootBundle.load(
        'assets/samples/landmark-ex1.jpg',
      );
      final Uint8List bytes = data.buffer.asUint8List();

      final FaceDetector interp = FaceDetector();
      await interp.initialize();
      final FaceDetector compiled = FaceDetector();
      try {
        await compiled.initialize(useCompiledModel: true);
      } catch (e) {
        // CompiledModel needs a supported delegate (e.g. Metal). If the runner
        // lacks it, skip rather than fail the classification parity check.
        interp.dispose();
        markTestSkipped('CompiledModel unavailable on this runner: $e');
        return;
      }
      try {
        final List<Face> a = await interp.detectFacesFromBytes(
          bytes,
          mode: FaceDetectionMode.full,
        );
        final List<Face> b = await compiled.detectFacesFromBytes(
          bytes,
          mode: FaceDetectionMode.full,
        );
        expect(a, isNotEmpty);
        expect(b, isNotEmpty);
        final FaceBlendshapes? ba = a.first.blendshapes;
        final FaceBlendshapes? bb = b.first.blendshapes;
        expect(ba, isNotNull);
        expect(bb, isNotNull);
        double maxErr = 0.0;
        for (int i = 0; i < 52; i++) {
          final double e = (ba!.scores[i] - bb!.scores[i]).abs();
          if (e > maxErr) maxErr = e;
        }
        print('Compiled-vs-interpreter max abs error: $maxErr');
        // The blendshape model is CPU-pinned on both paths, so on identical
        // input it would match to fp16 precision. But useCompiledModel runs the
        // upstream detector/mesh models on the GPU (Metal) rather than the CPU
        // interpreter, so their landmark outputs differ by a pixel or two, and
        // the blendshape model is sensitive to off-distribution landmarks (R1).
        // That upstream drift is what shows up here; a generous bound still
        // catches a genuinely broken compiled path (which would diverge far
        // more, e.g. an all-zero or constant vector).
        expect(maxErr, lessThan(0.25));
      } finally {
        interp.dispose();
        compiled.dispose();
      }
    });
  });
}
