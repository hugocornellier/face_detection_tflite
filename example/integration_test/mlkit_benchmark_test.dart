// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:math' show sqrt;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart'
    as mlkit;

const int _kIterations = 10;

const List<String> _kImages = [
  'assets/samples/landmark-ex1.jpg',
  'assets/samples/iris-detection-ex1.jpg',
  'assets/samples/group-shot-bounding-box-ex1.jpeg',
];

/// Result of benchmarking one detector on one image.
class _BenchResult {
  final List<double> timingsMs;
  final int faceCount;

  const _BenchResult({required this.timingsMs, required this.faceCount});

  double get avg => timingsMs.reduce((a, b) => a + b) / timingsMs.length;
  double get min => timingsMs.reduce((a, b) => a < b ? a : b);
  double get max => timingsMs.reduce((a, b) => a > b ? a : b);
}

/// Run one detector for [_kIterations] and return timings + face count.
Future<_BenchResult> _benchmarkTflite(
  FaceDetector detector,
  Uint8List bytes,
) async {
  // Warmup
  await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);

  final timings = <double>[];
  int faceCount = 0;

  for (int i = 0; i < _kIterations; i++) {
    final sw = Stopwatch()..start();
    final results =
        await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
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
  final inputImage = mlkit.InputImage.fromFilePath(tempFile.path);

  // Warmup
  await detector.processImage(inputImage);

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

double _euclidean(double x1, double y1, double x2, double y2) {
  final dx = x1 - x2;
  final dy = y1 - y2;
  return sqrt(dx * dx + dy * dy);
}

void _printLandmarkComparison(
  Face tfliteFace,
  mlkit.Face mlkitFace,
) {
  final tl = tfliteFace.landmarks;

  final pairs = <({String label, Point? tflite, mlkit.FaceLandmark? mlkitLm})>[
    (
      label: 'Left eye',
      tflite: tl.leftEye,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.leftEye],
    ),
    (
      label: 'Right eye',
      tflite: tl.rightEye,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.rightEye],
    ),
    (
      label: 'Nose',
      tflite: tl.noseTip,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.noseBase],
    ),
    (
      label: 'Mouth',
      tflite: tl.mouth,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.bottomMouth],
    ),
    (
      label: 'Left ear',
      tflite: tl.leftEyeTragion,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.leftEar],
    ),
    (
      label: 'Right ear',
      tflite: tl.rightEyeTragion,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.rightEar],
    ),
  ];

  print('Landmark comparison (first face):');
  for (final p in pairs) {
    if (p.tflite == null || p.mlkitLm == null) {
      print(
          '  ${p.label.padRight(12)}: tflite=${p.tflite} | mlkit=${p.mlkitLm} | delta=N/A');
      continue;
    }
    final tx = p.tflite!.x;
    final ty = p.tflite!.y;
    final mx = p.mlkitLm!.position.x.toDouble();
    final my = p.mlkitLm!.position.y.toDouble();
    final delta = _euclidean(tx, ty, mx, my);
    print(
      '  ${p.label.padRight(12)}: tflite=(${tx.toStringAsFixed(1)}, ${ty.toStringAsFixed(1)})'
      ' | mlkit=(${mx.toStringAsFixed(0)}, ${my.toStringAsFixed(0)})'
      ' | delta=${delta.toStringAsFixed(1)}px',
    );
  }
}

/// Match tflite faces to mlkit faces for landmark comparison.
/// For single-face images this is trivial; for multi-face images sort by bbox center X.
List<(Face, mlkit.Face)> _matchFaces(
  List<Face> tfliteFaces,
  List<mlkit.Face> mlkitFaces,
) {
  if (tfliteFaces.isEmpty || mlkitFaces.isEmpty) return [];

  final sortedTflite = List<Face>.from(tfliteFaces)
    ..sort((a, b) => a.boundingBox.center.x.compareTo(b.boundingBox.center.x));
  final sortedMlkit = List<mlkit.Face>.from(mlkitFaces)
    ..sort(
        (a, b) => a.boundingBox.center.dx.compareTo(b.boundingBox.center.dx));

  final count = sortedTflite.length < sortedMlkit.length
      ? sortedTflite.length
      : sortedMlkit.length;

  return [
    for (int i = 0; i < count; i++) (sortedTflite[i], sortedMlkit[i]),
  ];
}

double _avgLandmarkDelta(Face tfliteFace, mlkit.Face mlkitFace) {
  final tl = tfliteFace.landmarks;
  final pairs = <({Point? tflite, mlkit.FaceLandmark? mlkitLm})>[
    (
      tflite: tl.leftEye,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.leftEye]
    ),
    (
      tflite: tl.rightEye,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.rightEye]
    ),
    (
      tflite: tl.noseTip,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.noseBase]
    ),
    (
      tflite: tl.mouth,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.bottomMouth]
    ),
    (
      tflite: tl.leftEyeTragion,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.leftEar]
    ),
    (
      tflite: tl.rightEyeTragion,
      mlkitLm: mlkitFace.landmarks[mlkit.FaceLandmarkType.rightEar]
    ),
  ];

  final deltas = <double>[];
  for (final p in pairs) {
    if (p.tflite == null || p.mlkitLm == null) continue;
    deltas.add(_euclidean(
      p.tflite!.x,
      p.tflite!.y,
      p.mlkitLm!.position.x.toDouble(),
      p.mlkitLm!.position.y.toDouble(),
    ));
  }
  if (deltas.isEmpty) return double.nan;
  return deltas.reduce((a, b) => a + b) / deltas.length;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group(
      'Performance Benchmark: face_detection_tflite vs google_mlkit_face_detection',
      () {
    late FaceDetector tfliteDetector;
    late mlkit.FaceDetector mlkitDetector;

    // Per-image speedups and landmark deltas for summary
    final speedups = <double>[];
    final avgDeltas = <double>[];

    setUpAll(() async {
      tfliteDetector = FaceDetector();
      await tfliteDetector.initialize();

      mlkitDetector = mlkit.FaceDetector(
        options: mlkit.FaceDetectorOptions(
          enableLandmarks: true,
          performanceMode: mlkit.FaceDetectorMode.fast,
        ),
      );
    });

    tearDownAll(() async {
      tfliteDetector.dispose();
      await mlkitDetector.close();
    });

    for (final imagePath in _kImages) {
      final imageName = imagePath.split('/').last;

      test('Benchmark: $imageName', () async {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();

        final tfliteResult = await _benchmarkTflite(tfliteDetector, bytes);
        final mlkitResult =
            await _benchmarkMlkit(mlkitDetector, bytes, imageName);

        final speedup = mlkitResult.avg / tfliteResult.avg;
        speedups.add(speedup);

        print('');
        print('=== Benchmark: $imageName ===');
        print('face_detection_tflite:');
        print('  Faces detected: ${tfliteResult.faceCount}');
        print(
            '  Avg latency: ${tfliteResult.avg.toStringAsFixed(1)}ms ($_kIterations runs)');
        print(
            '  Min: ${tfliteResult.min.toStringAsFixed(1)}ms | Max: ${tfliteResult.max.toStringAsFixed(1)}ms');
        print('google_mlkit_face_detection:');
        print('  Faces detected: ${mlkitResult.faceCount}');
        print(
            '  Avg latency: ${mlkitResult.avg.toStringAsFixed(1)}ms ($_kIterations runs)');
        print(
            '  Min: ${mlkitResult.min.toStringAsFixed(1)}ms | Max: ${mlkitResult.max.toStringAsFixed(1)}ms');

        if (speedup >= 1.0) {
          print(
              'Speedup: ${speedup.toStringAsFixed(1)}x faster (face_detection_tflite)');
        } else {
          print(
              'Speedup: ${(1.0 / speedup).toStringAsFixed(1)}x faster (google_mlkit_face_detection)');
        }
        print('');

        // Landmark comparison on first matched face pair
        final tfliteFaces = await tfliteDetector.detectFaces(bytes,
            mode: FaceDetectionMode.fast);
        final mlkitInput = mlkit.InputImage.fromFilePath(
          '${Directory.systemTemp.path}/benchmark_$imageName',
        );
        final mlkitFaces = await mlkitDetector.processImage(mlkitInput);

        final matched = _matchFaces(tfliteFaces, mlkitFaces);
        if (matched.isNotEmpty) {
          final (firstTflite, firstMlkit) = matched.first;
          _printLandmarkComparison(firstTflite, firstMlkit);
          final delta = _avgLandmarkDelta(firstTflite, firstMlkit);
          if (!delta.isNaN) avgDeltas.add(delta);
        } else {
          print('Landmark comparison: no matched face pairs available');
        }

        expect(tfliteResult.faceCount, greaterThanOrEqualTo(1),
            reason:
                'face_detection_tflite should detect at least one face in $imageName');
        expect(mlkitResult.faceCount, greaterThanOrEqualTo(1),
            reason: 'ML Kit should detect at least one face in $imageName');
      });
    }

    test('Summary', () async {
      print('');
      print('=== OVERALL SUMMARY ===');
      if (speedups.isNotEmpty) {
        final avgSpeedup = speedups.reduce((a, b) => a + b) / speedups.length;
        if (avgSpeedup >= 1.0) {
          print(
              'Average speedup across all images: ${avgSpeedup.toStringAsFixed(1)}x faster (face_detection_tflite)');
        } else {
          print(
              'Average speedup across all images: ${(1.0 / avgSpeedup).toStringAsFixed(1)}x faster (google_mlkit_face_detection)');
        }
      } else {
        print('Average speedup: N/A (no data)');
      }
      if (avgDeltas.isNotEmpty) {
        final overallAvgDelta =
            avgDeltas.reduce((a, b) => a + b) / avgDeltas.length;
        print(
            'Average landmark delta: ${overallAvgDelta.toStringAsFixed(1)}px');
      } else {
        print('Average landmark delta: N/A (no data)');
      }
    });
  });
}
