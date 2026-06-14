// ignore_for_file: avoid_print

// End-to-end multiclass segmentation benchmark: GPU vs XNNPACK.
// selfie_multiclass is a single large conv model (static shape): the case
// where the raw-invoke sweep showed ~9x for GPU. This checks whether that
// survives the SelfieSegmentation pipeline (decode/resize + invoke + mask).
//
//   flutter test integration_test/seg_gpu_vs_xnnpack_test.dart -d macos

import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int warmup = 10;
const int iterations = 100;
const String image = 'assets/samples/landmark-ex1.jpg';

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

Future<List<int>> _run(PerformanceConfig config, cv.Mat mat) async {
  final seg = await SelfieSegmentation.create(
    config: SegmentationConfig(
      model: SegmentationModel.multiclass,
      performanceConfig: config,
      useIsolate: false,
    ),
  );
  for (int i = 0; i < warmup; i++) {
    await seg.call(mat);
  }
  final us = <int>[];
  for (int i = 0; i < iterations; i++) {
    final sw = Stopwatch()..start();
    await seg.call(mat);
    sw.stop();
    us.add(sw.elapsedMicroseconds);
  }
  seg.dispose();
  return us;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Multiclass segmentation: GPU vs XNNPACK', () {
    test('compare', timeout: const Timeout(Duration(minutes: 10)), () async {
      final bytes = (await rootBundle.load(image)).buffer.asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final xnn = await _run(const PerformanceConfig.xnnpack(), mat);
      final gpu = await _run(const PerformanceConfig.gpu(), mat);
      mat.dispose();

      print('\n${'=' * 64}');
      print('MULTICLASS SEGMENTATION: GPU vs XNNPACK (end-to-end call(Mat))');
      print('=' * 64);
      print('XNNPACK  p50=${_p50(xnn).toStringAsFixed(1)}ms '
          'mean=${_mean(xnn).toStringAsFixed(1)}ms std=${_std(xnn).toStringAsFixed(1)}ms');
      print('GPU      p50=${_p50(gpu).toStringAsFixed(1)}ms '
          'mean=${_mean(gpu).toStringAsFixed(1)}ms std=${_std(gpu).toStringAsFixed(1)}ms');
      print('SPEEDUP: ${(_p50(xnn) / _p50(gpu)).toStringAsFixed(2)}x (p50)');
      print('=' * 64);
    });
  });
}
