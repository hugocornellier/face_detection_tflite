// Diagnostic: time PURE inference in the real macOS app context, to separate
// inference cost from the face-detection pipeline/marshalling overhead.
//
// Per model (detection back / face mesh / iris), times:
//   - classic Interpreter (default options) invoke + output readback
//   - CompiledModel.run()      (sync)   on CPU and GPU|CPU
//   - CompiledModel.runAsync() (async)  on CPU and GPU|CPU
//
// Run: flutter test integration_test/cm_inference_only_test.dart -d macos
@TestOn('mac-os')
library;

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const int _warmup = 20;
const int _iterations = 100;
const String _assetBase = 'packages/face_detection_tflite/assets/models';
const List<String> _models = [
  'face_detection_back.tflite',
  'face_landmark.tflite',
  'iris_landmark.tflite',
];

typedef _Stats = ({double mean, double median, double min});

_Stats _stats(List<int> micros) {
  final s = [...micros]..sort();
  final sum = s.fold<int>(0, (a, b) => a + b);
  return (
    mean: sum / s.length / 1000,
    median: s[s.length ~/ 2] / 1000,
    min: s.first / 1000,
  );
}

String _fmt(_Stats r) => 'mean ${r.mean.toStringAsFixed(3)} ms, '
    'median ${r.median.toStringAsFixed(3)} ms, '
    'min ${r.min.toStringAsFixed(3)} ms';

Future<_Stats> _timeSync(void Function() once) async {
  for (var i = 0; i < _warmup; i++) {
    once();
  }
  final s = <int>[];
  final sw = Stopwatch();
  for (var i = 0; i < _iterations; i++) {
    sw
      ..reset()
      ..start();
    once();
    sw.stop();
    s.add(sw.elapsedMicroseconds);
  }
  return _stats(s);
}

Future<_Stats> _timeAsync(Future<void> Function() once) async {
  for (var i = 0; i < _warmup; i++) {
    await once();
  }
  final s = <int>[];
  final sw = Stopwatch();
  for (var i = 0; i < _iterations; i++) {
    sw
      ..reset()
      ..start();
    await once();
    sw.stop();
    s.add(sw.elapsedMicroseconds);
  }
  return _stats(s);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('pure inference latency in-app: Interpreter vs CompiledModel', () async {
    for (final model in _models) {
      final bytes = (await rootBundle.load(
        '$_assetBase/$model',
      ))
          .buffer
          .asUint8List();
      // ignore: avoid_print
      print('=== $model ===');

      // Classic Interpreter baseline (default options): input-write + invoke
      // + output readback per iteration, matching CompiledModel.run()'s
      // copy-in/copy-out semantics.
      {
        final itp = Interpreter.fromBuffer(bytes);
        itp.allocateTensors();
        final input = itp.getInputTensor(0);
        final inputBytes = Uint8List(input.data.length);
        final outputs = itp.getOutputTensors();
        final r = await _timeSync(() {
          input.data = inputBytes;
          itp.invoke();
          for (final t in outputs) {
            t.data;
          }
        });
        itp.close();
        // ignore: avoid_print
        print('  interpreter:        ${_fmt(r)}');
      }

      for (final entry in {
        'CPU    ': {Accelerator.cpu},
        'GPU|CPU': {Accelerator.gpu, Accelerator.cpu},
      }.entries) {
        CompiledModel cm;
        try {
          cm = CompiledModel.fromBuffer(bytes, accelerators: entry.value);
        } catch (e) {
          // ignore: avoid_print
          print('  cm ${entry.key} FAILED to compile: $e');
          continue;
        }
        final input = Float32List(cm.inputByteSizes[0] ~/ 4);
        final sync = await _timeSync(() => cm.run([input]));
        final async = await _timeAsync(() => cm.runAsync([input]));
        cm.close();
        // ignore: avoid_print
        print('  cm ${entry.key} run():      ${_fmt(sync)}');
        // ignore: avoid_print
        print('  cm ${entry.key} runAsync(): ${_fmt(async)}');
      }
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}
