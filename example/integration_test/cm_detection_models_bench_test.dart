// ignore_for_file: avoid_print

@TestOn('mac-os')
library;

import 'package:face_detection_tflite/face_detection_tflite_native.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int _warmup = 10;
const int _iterations = 80;

double _p50(List<int> us) {
  final sorted = [...us]..sort();
  return sorted[sorted.length ~/ 2] / 1000.0;
}

double _mean(List<int> us) => us.reduce((a, b) => a + b) / us.length / 1000.0;

Future<({double mean, double p50, int faces})> _bench(
  Future<List<Detection>> Function() run,
) async {
  for (var i = 0; i < _warmup; i++) {
    await run();
  }
  final samples = <int>[];
  var faceCount = 0;
  for (var i = 0; i < _iterations; i++) {
    final sw = Stopwatch()..start();
    final detections = await run();
    sw.stop();
    samples.add(sw.elapsedMicroseconds);
    if (i == 0) faceCount = detections.length;
  }
  return (mean: _mean(samples), p50: _p50(samples), faces: faceCount);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('all face detection models: Interpreter vs CompiledModel', () async {
    final imageBytes = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();
    final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);

    print('');
    print('FACE DETECTION MODEL BENCH - p50/mean ms, macOS');
    print('warmup=$_warmup iterations=$_iterations');

    try {
      for (final model in FaceDetectionModel.values) {
        if (model == FaceDetectionModel.fullSparse) {
          print('${model.name.padRight(12)} skipped: unsupported by guard');
          continue;
        }

        final modelBytes = (await rootBundle.load(
          'packages/face_detection_tflite/assets/models/${testNameFor(model)}',
        ))
            .buffer
            .asUint8List();

        final interpreter = await FaceDetection.createFromBuffer(
          modelBytes,
          model,
        );
        final compiled = await FaceDetection.createCompiledFromBuffer(
          modelBytes,
          model,
        );

        try {
          final interpreterTensor = convertImageToTensor(
            mat,
            outW: interpreter.inputWidth,
            outH: interpreter.inputHeight,
          );
          final compiledTensor = convertImageToTensor(
            mat,
            outW: compiled.inputWidth,
            outH: compiled.inputHeight,
          );

          final i = await _bench(
            () => interpreter.callWithTensor(interpreterTensor),
          );
          final c = await _bench(() => compiled.callWithTensor(compiledTensor));
          final speedup = i.p50 / c.p50;

          print(
            '${model.name.padRight(12)} '
            'interp p50=${i.p50.toStringAsFixed(3)} mean=${i.mean.toStringAsFixed(3)} '
            'compiled p50=${c.p50.toStringAsFixed(3)} mean=${c.mean.toStringAsFixed(3)} '
            'speedup=${speedup.toStringAsFixed(2)}x faces=${c.faces}',
          );
        } finally {
          interpreter.dispose();
          compiled.dispose();
        }
      }
    } finally {
      mat.dispose();
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}
