// ignore_for_file: avoid_print
library;

import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('microbench: fillNHWC4D 192x192 x 10,000 iterations', () async {
    const int h = 192;
    const int w = 192;
    final Float32List flat = Float32List(h * w * 3);
    for (int i = 0; i < flat.length; i++) {
      flat[i] = (i % 255) / 255.0;
    }
    final cache = createNHWCTensor4D(h, w);

    const int warmup = 100;
    const int iters = 10000;
    for (int i = 0; i < warmup; i++) {
      fillNHWC4D(flat, cache, h, w);
    }
    final sw = Stopwatch()..start();
    for (int i = 0; i < iters; i++) {
      fillNHWC4D(flat, cache, h, w);
    }
    sw.stop();
    final double usPerCall = sw.elapsedMicroseconds / iters;
    print('\n[FILLNHWC4D 192x192] $iters calls: '
        '${sw.elapsedMilliseconds} ms total, '
        '${usPerCall.toStringAsFixed(1)} µs/call');
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('microbench: flattenDynamicTensor 2D [1,1404] x 50,000 iterations',
      () async {
    // Mesh output shape
    final Object tensor = allocTensorShape([1, 1404]);
    final list2d = tensor as List<List<double>>;
    for (int i = 0; i < 1404; i++) {
      list2d[0][i] = i.toDouble();
    }
    const int warmup = 500;
    const int iters = 50000;
    for (int i = 0; i < warmup; i++) {
      flattenDynamicTensor(tensor);
    }
    final sw = Stopwatch()..start();
    for (int i = 0; i < iters; i++) {
      flattenDynamicTensor(tensor);
    }
    sw.stop();
    final double usPerCall = sw.elapsedMicroseconds / iters;
    print(
        '\n[FLATTEN 2D 1x1404] $iters calls: ${sw.elapsedMilliseconds} ms total, '
        '${usPerCall.toStringAsFixed(2)} µs/call');
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('microbench: flattenDynamicTensor 3D [1,896,16] x 5,000 iterations',
      () async {
    // Detection boxes output shape
    final Object tensor = allocTensorShape([1, 896, 16]);
    final list3d = tensor as List<List<List<double>>>;
    for (int b = 0; b < 896; b++) {
      for (int k = 0; k < 16; k++) {
        list3d[0][b][k] = (b + k).toDouble();
      }
    }
    const int warmup = 100;
    const int iters = 5000;
    for (int i = 0; i < warmup; i++) {
      flattenDynamicTensor(tensor);
    }
    final sw = Stopwatch()..start();
    for (int i = 0; i < iters; i++) {
      flattenDynamicTensor(tensor);
    }
    sw.stop();
    final double usPerCall = sw.elapsedMicroseconds / iters;
    print(
        '\n[FLATTEN 3D 1x896x16] $iters calls: ${sw.elapsedMilliseconds} ms total, '
        '${usPerCall.toStringAsFixed(1)} µs/call');
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('microbench: fillNHWC4D 64x64 x 10,000 iterations', () async {
    const int h = 64;
    const int w = 64;
    final Float32List flat = Float32List(h * w * 3);
    for (int i = 0; i < flat.length; i++) {
      flat[i] = (i % 255) / 255.0;
    }
    final cache = createNHWCTensor4D(h, w);

    const int warmup = 100;
    const int iters = 10000;
    for (int i = 0; i < warmup; i++) {
      fillNHWC4D(flat, cache, h, w);
    }
    final sw = Stopwatch()..start();
    for (int i = 0; i < iters; i++) {
      fillNHWC4D(flat, cache, h, w);
    }
    sw.stop();
    final double usPerCall = sw.elapsedMicroseconds / iters;
    print('\n[FILLNHWC4D 64x64] $iters calls: '
        '${sw.elapsedMilliseconds} ms total, '
        '${usPerCall.toStringAsFixed(1)} µs/call');
  }, timeout: const Timeout(Duration(minutes: 5)));
}
