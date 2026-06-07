// ignore_for_file: avoid_print

// End-to-end FaceDetector full-mode benchmark comparing XNNPACK vs GPU.
// Confirms whether the GPU default is actually faster for the *whole* face
// pipeline (detection + landmark + iris + embedding), not just raw invoke().
//
//   flutter test integration_test/face_gpu_vs_xnnpack_test.dart -d macos
//   (iOS: flutter test can't attach the VM service on a physical device; use
//    the ios_sweep_main.dart / devicectl approach for on-device numbers.)

import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

const int warmup = 30;
const int iterations = 100;
const List<String> sampleImages = [
  'assets/samples/landmark-ex1.jpg',
  'assets/samples/group-shot-bounding-box-ex1.jpeg',
  'assets/samples/mesh-ex1.jpeg',
];

double _p50(List<int> us) {
  final s = List<int>.from(us)..sort();
  return s[((s.length - 1) * 0.50).floor()] / 1000.0;
}

double _mean(List<int> us) => us.reduce((a, b) => a + b) / us.length / 1000.0;

double _std(List<int> us) {
  final m = us.reduce((a, b) => a + b) / us.length;
  final v =
      us.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b) / us.length;
  return sqrt(v) / 1000.0;
}

Future<Map<String, List<int>>> _run(PerformanceConfig config) async {
  final detector = FaceDetector();
  await detector.initialize(performanceConfig: config);
  final out = <String, List<int>>{};
  for (final path in sampleImages) {
    final bytes = (await rootBundle.load(path)).buffer.asUint8List();
    for (int i = 0; i < warmup; i++) {
      await detector.detectFacesFromBytes(bytes, mode: FaceDetectionMode.full);
    }
    final us = <int>[];
    for (int i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      await detector.detectFacesFromBytes(bytes, mode: FaceDetectionMode.full);
      sw.stop();
      us.add(sw.elapsedMicroseconds);
    }
    out[path] = us;
  }
  await detector.dispose();
  return out;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('FaceDetector full-mode: GPU vs XNNPACK', () {
    test('compare', timeout: const Timeout(Duration(minutes: 10)), () async {
      final xnn = await _run(const PerformanceConfig.xnnpack());
      final gpu = await _run(const PerformanceConfig.gpu());

      print('\n${'=' * 72}');
      print('FACE FULL-MODE: GPU vs XNNPACK (p50 ms, end-to-end detectFaces)');
      print('=' * 72);
      print(
          '${'image'.padRight(42)} ${'xnnpack'.padLeft(9)} ${'gpu'.padLeft(9)}   speedup');
      print('-' * 72);
      for (final path in sampleImages) {
        final name = path.split('/').last;
        final xp = _p50(xnn[path]!);
        final gp = _p50(gpu[path]!);
        final spd = (xp / gp);
        print('${name.padRight(42)} '
            '${xp.toStringAsFixed(1).padLeft(9)} '
            '${gp.toStringAsFixed(1).padLeft(9)}   '
            '${spd.toStringAsFixed(2)}x');
      }
      print('-' * 72);
      // overall
      final xnnAll = xnn.values.expand((e) => e).toList();
      final gpuAll = gpu.values.expand((e) => e).toList();
      double minMs(List<int> us) => us.reduce((a, b) => a < b ? a : b) / 1000.0;
      print('XNNPACK  overall p50=${_p50(xnnAll).toStringAsFixed(1)}ms '
          'mean=${_mean(xnnAll).toStringAsFixed(1)}ms std=${_std(xnnAll).toStringAsFixed(1)}ms');
      print('GPU      overall p50=${_p50(gpuAll).toStringAsFixed(1)}ms '
          'mean=${_mean(gpuAll).toStringAsFixed(1)}ms std=${_std(gpuAll).toStringAsFixed(1)}ms');
      print('per-image min (warmup-immune floor):');
      for (final path in sampleImages) {
        print('  ${path.split('/').last.padRight(40)} '
            'xnn=${minMs(xnn[path]!).toStringAsFixed(1)}ms  gpu=${minMs(gpu[path]!).toStringAsFixed(1)}ms');
      }
      print(
          'OVERALL SPEEDUP: ${(_p50(xnnAll) / _p50(gpuAll)).toStringAsFixed(2)}x (p50)');
      print('=' * 72);
    });
  });
}
