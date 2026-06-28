// ignore_for_file: avoid_print

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart'
    as mlkit;

const int _kIterations = 10;

const List<String> _kImages = [
  'assets/samples/landmark-ex1.jpg',
  'assets/samples/iris-detection-ex1.jpg',
  'assets/samples/group-shot-bounding-box-ex1.jpeg',
];

class _BenchResult {
  final List<double> timingsMs;
  final int faceCount;

  const _BenchResult({required this.timingsMs, required this.faceCount});

  double get avg => timingsMs.reduce((a, b) => a + b) / timingsMs.length;
  double get min => timingsMs.reduce((a, b) => a < b ? a : b);
  double get max => timingsMs.reduce((a, b) => a > b ? a : b);
}

Future<_BenchResult> _benchmarkTflite(
  FaceDetector detector,
  Uint8List bytes,
) async {
  await detector.detectFacesFromBytes(bytes, mode: FaceDetectionMode.fast);

  final timings = <double>[];
  int faceCount = 0;

  for (int i = 0; i < _kIterations; i++) {
    final sw = Stopwatch()..start();
    final results = await detector.detectFacesFromBytes(bytes,
        mode: FaceDetectionMode.fast);
    sw.stop();
    timings.add(sw.elapsedMicroseconds / 1000.0);
    if (i == 0) faceCount = results.length;
  }

  return _BenchResult(timingsMs: timings, faceCount: faceCount);
}

Future<_BenchResult> _benchmarkMlkit(
  mlkit.FaceDetector detector,
  Uint8List bytes,
  String imageName,
) async {
  final tempFile = File('${Directory.systemTemp.path}/benchmark_$imageName');
  await tempFile.writeAsBytes(bytes);

  await detector.processImage(mlkit.InputImage.fromFilePath(tempFile.path));

  final timings = <double>[];
  int faceCount = 0;

  for (int i = 0; i < _kIterations; i++) {
    final freshInput = mlkit.InputImage.fromFilePath(tempFile.path);
    final sw = Stopwatch()..start();
    final results = await detector.processImage(freshInput);
    sw.stop();
    timings.add(sw.elapsedMicroseconds / 1000.0);
    if (i == 0) faceCount = results.length;
  }

  return _BenchResult(timingsMs: timings, faceCount: faceCount);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('MLKit comparison: interpreter vs compiled-model', () {
    late FaceDetector tfliteInterp;
    late FaceDetector tfliteCompiled;
    late mlkit.FaceDetector mlkitDetector;

    final interpSpeedups = <double>[];
    final compiledSpeedups = <double>[];

    setUpAll(() async {
      tfliteInterp = FaceDetector();
      await tfliteInterp.initialize(useCompiledModel: false);

      tfliteCompiled = FaceDetector();
      await tfliteCompiled.initialize(useCompiledModel: true);

      mlkitDetector = mlkit.FaceDetector(
        options: mlkit.FaceDetectorOptions(
          enableLandmarks: true,
          performanceMode: mlkit.FaceDetectorMode.fast,
        ),
      );
    });

    tearDownAll(() async {
      tfliteInterp.dispose();
      tfliteCompiled.dispose();
      await mlkitDetector.close();
    });

    for (final imagePath in _kImages) {
      final imageName = imagePath.split('/').last;

      test('Benchmark: $imageName', () async {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();

        final interp = await _benchmarkTflite(tfliteInterp, bytes);
        final compiled = await _benchmarkTflite(tfliteCompiled, bytes);
        final ml = await _benchmarkMlkit(mlkitDetector, bytes, imageName);

        final interpSpeedup = ml.avg / interp.avg;
        final compiledSpeedup = ml.avg / compiled.avg;
        interpSpeedups.add(interpSpeedup);
        compiledSpeedups.add(compiledSpeedup);

        print('');
        print('=== Benchmark: $imageName ===');
        print('  tflite (interpreter): ${interp.avg.toStringAsFixed(1)}ms '
            '(min ${interp.min.toStringAsFixed(1)} / max ${interp.max.toStringAsFixed(1)}) '
            'faces=${interp.faceCount}');
        print('  tflite (compiled):    ${compiled.avg.toStringAsFixed(1)}ms '
            '(min ${compiled.min.toStringAsFixed(1)} / max ${compiled.max.toStringAsFixed(1)}) '
            'faces=${compiled.faceCount}');
        print('  google_mlkit:         ${ml.avg.toStringAsFixed(1)}ms '
            '(min ${ml.min.toStringAsFixed(1)} / max ${ml.max.toStringAsFixed(1)}) '
            'faces=${ml.faceCount}');
        print('  speedup vs mlkit -> interpreter: '
            '${interpSpeedup.toStringAsFixed(2)}x | compiled: '
            '${compiledSpeedup.toStringAsFixed(2)}x');
        print('');

        expect(interp.faceCount, greaterThanOrEqualTo(1));
        expect(compiled.faceCount, greaterThanOrEqualTo(1));
        expect(ml.faceCount, greaterThanOrEqualTo(1));
      });
    }

    test('Summary', () async {
      double mean(List<double> xs) => xs.reduce((a, b) => a + b) / xs.length;
      print('');
      print('=== OVERALL SUMMARY (Android) ===');
      print('  Avg speedup vs mlkit, interpreter: '
          '${mean(interpSpeedups).toStringAsFixed(2)}x');
      print('  Avg speedup vs mlkit, compiled:    '
          '${mean(compiledSpeedups).toStringAsFixed(2)}x');
    });
  });
}
