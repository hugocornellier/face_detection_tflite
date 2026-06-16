// Verifies the accelerator / precision control params added to FaceDetector.
// Inits with useCompiledModel: true, accelerators: {Accelerator.cpu}, runs one
// detect, and asserts no throw.
//
// Run: flutter test integration_test/accelerator_control_test.dart -d macos
// ignore_for_file: avoid_print
@TestOn('mac-os')
library;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test(
    'FaceDetector: useCompiledModel=true with accelerators={cpu} does not throw',
    () async {
      final detector = FaceDetector();
      try {
        await detector.initialize(
          useCompiledModel: true,
          accelerators: {Accelerator.cpu},
          precision: Precision.fp16,
        );
      } on UnsupportedError {
        // fullSparse is the only model that intentionally throws; the default
        // (backCamera) must not. Re-throw to fail the test.
        rethrow;
      }
      expect(detector.isReady, isTrue);

      final imageBytes = (await rootBundle.load(
        'assets/samples/landmark-ex1.jpg',
      ))
          .buffer
          .asUint8List();

      final List<Face> faces = await detector.detectFacesFromBytes(
        imageBytes,
        mode: FaceDetectionMode.standard,
      );
      print(
        'accelerator_control_test: ${faces.length} face(s) detected '
        'with {Accelerator.cpu} + Precision.fp16',
      );
      await detector.dispose();
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );

  test(
    'FaceDetector.create: accelerators={gpu,cpu} default equals back-compat',
    () async {
      final detector = await FaceDetector.create(
        useCompiledModel: true,
        accelerators: {Accelerator.gpu, Accelerator.cpu},
        precision: Precision.fp16,
      );
      expect(detector.isReady, isTrue);
      await detector.dispose();
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );
}
