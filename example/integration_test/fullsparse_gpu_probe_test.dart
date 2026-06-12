// ignore_for_file: avoid_print

// One-off probe: does fullSparse run on the classic Interpreter's GPU (Metal)
// delegate? CompiledModel GPU SIGABRTs on this model's DENSIFY op (upstream
// LiteRT bug); compiled-CPU works. This fills in the remaining cell.
//
//   flutter test integration_test/fullsparse_gpu_probe_test.dart -d macos

@TestOn('mac-os')
library;

import 'package:face_detection_tflite/face_detection_tflite.dart';
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

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('fullSparse on Interpreter: XNNPACK vs GPU delegate', () async {
    final imageBytes = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();
    final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    final modelBytes = (await rootBundle.load(
      'packages/face_detection_tflite/assets/models/${testNameFor(FaceDetectionModel.fullSparse)}',
    ))
        .buffer
        .asUint8List();

    Future<void> bench(String label, PerformanceConfig config) async {
      final det = await FaceDetection.createFromBuffer(
        modelBytes,
        FaceDetectionModel.fullSparse,
        performanceConfig: config,
      );
      try {
        final tensor = convertImageToTensor(
          mat,
          outW: det.inputWidth,
          outH: det.inputHeight,
        );
        for (var i = 0; i < _warmup; i++) {
          await det.callWithTensor(tensor);
        }
        final samples = <int>[];
        var faces = 0;
        for (var i = 0; i < _iterations; i++) {
          final sw = Stopwatch()..start();
          final detections = await det.callWithTensor(tensor);
          sw.stop();
          samples.add(sw.elapsedMicroseconds);
          faces = detections.length;
        }
        print(
          'fullSparse $label p50=${_p50(samples).toStringAsFixed(2)}ms '
          'faces=$faces',
        );
        expect(faces, greaterThan(0), reason: '$label found no faces');
      } finally {
        det.dispose();
      }
    }

    await bench('xnnpack', const PerformanceConfig.xnnpack());
    await bench('gpu    ', const PerformanceConfig.gpu());
  });
}
