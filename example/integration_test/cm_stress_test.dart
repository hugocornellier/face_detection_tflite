// CompiledModel stability stress test: hammers the maximum-concurrency path
// (multi-face full mode → concurrent mesh pool + iris pair GPU dispatches)
// that previously exposed Metal tensor-buffer lock crashes.
// Run: flutter test integration_test/cm_stress_test.dart -d macos
// ignore_for_file: avoid_print
@TestOn('mac-os')
library;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const int _iterations = 300;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('CompiledModel full-mode stress (multi-face concurrency)', () async {
    final group = (await rootBundle.load(
      'assets/samples/group-shot-bounding-box-ex1.jpeg',
    ))
        .buffer
        .asUint8List();
    final single = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();

    final detector = FaceDetector();
    await detector.initialize(useCompiledModel: true);
    addTearDown(detector.dispose);

    for (var i = 0; i < _iterations; i++) {
      // Alternate images and modes to vary dispatch interleavings.
      final bytes = i.isEven ? group : single;
      final mode = i % 4 < 2 ? FaceDetectionMode.full : FaceDetectionMode.fast;
      final faces = await detector.detectFacesFromBytes(bytes, mode: mode);
      expect(faces, isNotEmpty, reason: 'iteration $i returned no faces');
      if (i % 50 == 0) {
        print('stress[$i/$_iterations] ${faces.length} faces OK');
      }
    }
    print('stress complete: $_iterations iterations, no crashes');
  }, timeout: const Timeout(Duration(minutes: 15)));
}
