// ignore_for_file: avoid_print

// Does single-face full-mode detection favor GPU? Runs per-image GPU vs XNNPACK
// and prints the detected face count, to separate "face count" from "image
// cost" as the driver.
//
//   flutter test integration_test/face_singleface_gpu_test.dart -d macos

import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';

const int warmup = 20;
const int iterations = 100;
const List<String> images = [
  'assets/samples/iris-detection-ex1.jpg',
  'assets/samples/iris-detection-ex2.jpg',
  'assets/samples/landmark-ex1.jpg',
  'assets/samples/mesh-ex1.jpeg',
];

double _p50(List<int> us) {
  final s = List<int>.from(us)..sort();
  return s[((s.length - 1) * 0.50).floor()] / 1000.0;
}

double _std(List<int> us) {
  final m = us.reduce((a, b) => a + b) / us.length;
  final v =
      us.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b) / us.length;
  return sqrt(v) / 1000.0;
}

Future<(List<int>, int)> _run(PerformanceConfig cfg, Uint8List bytes) async {
  final d = FaceDetector();
  await d.initialize(performanceConfig: cfg);
  int faces = 0;
  for (int i = 0; i < warmup; i++) {
    final r = await d.detectFacesFromBytes(bytes, mode: FaceDetectionMode.full);
    faces = r.length;
  }
  final us = <int>[];
  for (int i = 0; i < iterations; i++) {
    final sw = Stopwatch()..start();
    await d.detectFacesFromBytes(bytes, mode: FaceDetectionMode.full);
    sw.stop();
    us.add(sw.elapsedMicroseconds);
  }
  await d.dispose();
  return (us, faces);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Single-face full-mode: GPU vs XNNPACK', () {
    test('compare', timeout: const Timeout(Duration(minutes: 15)), () async {
      print('\n${'=' * 78}');
      print('SINGLE-FACE FULL-MODE: GPU vs XNNPACK (p50±std ms)');
      print('=' * 78);
      print(
          '${'image'.padRight(28)} faces ${'xnnpack'.padLeft(12)} ${'gpu'.padLeft(12)}   speedup');
      print('-' * 78);
      for (final path in images) {
        final bytes = (await rootBundle.load(path)).buffer.asUint8List();
        final (xnn, xf) = await _run(const PerformanceConfig.xnnpack(), bytes);
        final (gpu, gf) = await _run(const PerformanceConfig.gpu(), bytes);
        final xp = _p50(xnn), gp = _p50(gpu);
        print('${path.split('/').last.padRight(28)} '
            '${xf.toString().padLeft(5)} '
            '${('${xp.toStringAsFixed(1)}±${_std(xnn).toStringAsFixed(1)}').padLeft(12)} '
            '${('${gp.toStringAsFixed(1)}±${_std(gpu).toStringAsFixed(1)}').padLeft(12)}   '
            '${(xp / gp).toStringAsFixed(2)}x  (faces gpu=$gf)');
      }
      print('=' * 78);
    });
  });
}
