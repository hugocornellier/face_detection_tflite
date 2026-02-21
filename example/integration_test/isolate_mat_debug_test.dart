// ignore_for_file: avoid_print

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Test with disabled delegates (no XNNPACK, no GPU)
  test('Isolate cycles with disabled delegates', () async {
    final ByteData data =
        await rootBundle.load('assets/samples/landmark-ex1.jpg');
    final Uint8List bytes = data.buffer.asUint8List();

    for (int i = 0; i < 15; i++) {
      final d = await FaceDetectorIsolate.spawn(
        performanceConfig: const PerformanceConfig(
          mode: PerformanceMode.disabled,
          numThreads: 1,
        ),
      );
      final faces = await d.detectFaces(bytes, mode: FaceDetectionMode.fast);
      print('Disabled cycle $i: ${faces.length} faces');
      await d.dispose();
    }
  });

  // Test with auto mode (default - uses GPU on iOS)
  test('Isolate cycles with auto mode', () async {
    final ByteData data =
        await rootBundle.load('assets/samples/landmark-ex1.jpg');
    final Uint8List bytes = data.buffer.asUint8List();

    for (int i = 0; i < 15; i++) {
      final d = await FaceDetectorIsolate.spawn();
      final faces = await d.detectFaces(bytes, mode: FaceDetectionMode.fast);
      print('Auto cycle $i: ${faces.length} faces');
      await d.dispose();
    }
  });
}
