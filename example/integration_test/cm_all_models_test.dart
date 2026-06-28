// Verifies the CompiledModel engine against every FaceDetectionModel variant:
// all variants except fullSparse must initialize and detect faces; fullSparse
// must fail with a catchable error (upstream LiteRT DENSIFY bug) rather than
// crash the process.
// Run: flutter test integration_test/cm_all_models_test.dart -d macos
// ignore_for_file: avoid_print
@TestOn('mac-os')
library;

import 'package:face_detection_tflite/face_detection_tflite_native.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('CompiledModel engine: every detection model variant', () async {
    final bytes = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();

    for (final model in FaceDetectionModel.values) {
      if (model == FaceDetectionModel.fullSparse) {
        final detector = FaceDetector();
        Object? error;
        try {
          await detector.initialize(model: model, useCompiledModel: true);
        } catch (e) {
          error = e;
        } finally {
          await detector.dispose();
        }
        print('$model: rejected with ${error.runtimeType}: $error');
        expect(
          error,
          isNotNull,
          reason: 'fullSparse must be rejected by the CompiledModel guard',
        );
        continue;
      }

      final detector = FaceDetector();
      await detector.initialize(model: model, useCompiledModel: true);
      try {
        final faces = await detector.detectFacesFromBytes(
          bytes,
          mode: FaceDetectionMode.full,
        );
        print('$model: ${faces.length} face(s) detected');
        expect(faces, isNotEmpty, reason: '$model found no faces');
      } finally {
        await detector.dispose();
      }
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}
