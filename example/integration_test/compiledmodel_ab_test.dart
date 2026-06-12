// ignore_for_file: avoid_print

import 'dart:math' as math;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const List<String> _sampleImages = [
  'assets/samples/landmark-ex1.jpg',
  'assets/samples/group-shot-bounding-box-ex1.jpeg',
];
const int _warmupIterations = 10;
const int _benchmarkIterations = 100;
const List<FaceDetectionMode> _modes = [
  FaceDetectionMode.fast,
  FaceDetectionMode.full,
];

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test(
    'CompiledModel A/B benchmark: face detection pipeline (fast + full)',
    () async {
      final images = <String, Uint8List>{
        for (final path in _sampleImages)
          path: (await rootBundle.load(path)).buffer.asUint8List(),
      };

      final interpreterStats = await _benchmarkEngine(
        images,
        useCompiledModel: false,
      );
      final compiledStats = await _benchmarkEngine(
        images,
        useCompiledModel: true,
      );

      print('');
      print('CompiledModel A/B benchmark');
      print('warmup: $_warmupIterations, measured: $_benchmarkIterations');
      for (final path in _sampleImages) {
        print('sample: $path');
        for (final mode in _modes) {
          final i = interpreterStats[(path, mode)]!;
          final c = compiledStats[(path, mode)]!;
          final double speedup = i.meanMs / c.meanMs;
          print('[${mode.name}]');
          print(
            '  interpreter:   mean=${i.meanMs.toStringAsFixed(3)} ms, '
            'median=${i.medianMs.toStringAsFixed(3)} ms, '
            'min=${i.minMs.toStringAsFixed(3)} ms, '
            'faces=${i.faceCount}',
          );
          print(
            '  compiledmodel: mean=${c.meanMs.toStringAsFixed(3)} ms, '
            'median=${c.medianMs.toStringAsFixed(3)} ms, '
            'min=${c.minMs.toStringAsFixed(3)} ms, '
            'faces=${c.faceCount}',
          );
          print('  speedup_mean=${speedup.toStringAsFixed(3)}x');

          expect(i.timingsMs, hasLength(_benchmarkIterations));
          expect(c.timingsMs, hasLength(_benchmarkIterations));
          expect(c.faceCount, i.faceCount);

          // Engines must agree on geometry, not just face count. Compared on
          // single-face samples only, where "first face" is unambiguous.
          if (i.faceCount != 1) continue;
          final iFace = i.firstFace!;
          final cFace = c.firstFace!;
          final double diag = _diagonal(iFace);
          expect(
            _dist(iFace.boundingBox.topLeft, cFace.boundingBox.topLeft),
            lessThan(diag * 0.05),
            reason: '[${mode.name}] bounding boxes diverge between engines',
          );
          if (iFace.mesh != null || cFace.mesh != null) {
            final iMesh = iFace.mesh!.points;
            final cMesh = cFace.mesh!.points;
            expect(cMesh.length, iMesh.length);
            double sum = 0;
            for (int p = 0; p < iMesh.length; p++) {
              sum += _dist(iMesh[p], cMesh[p]);
            }
            final double meshMad = sum / iMesh.length;
            print('  mesh mean point deviation: '
                '${meshMad.toStringAsFixed(2)} px (face diag '
                '${diag.toStringAsFixed(0)} px)');
            expect(
              meshMad,
              lessThan(diag * 0.02),
              reason: '[${mode.name}] mesh points diverge between engines',
            );
          }
        }
      }
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );
}

Future<Map<(String, FaceDetectionMode), _BenchStats>> _benchmarkEngine(
  Map<String, Uint8List> images, {
  required bool useCompiledModel,
}) async {
  final stats = <(String, FaceDetectionMode), _BenchStats>{};
  final detector = FaceDetector();
  await detector.initialize(useCompiledModel: useCompiledModel);
  try {
    for (final entry in images.entries) {
      for (final mode in _modes) {
        stats[(entry.key, mode)] = await _benchmarkDetector(
          detector,
          entry.value,
          mode,
        );
      }
    }
  } finally {
    await detector.dispose();
  }
  return stats;
}

Future<_BenchStats> _benchmarkDetector(
  FaceDetector detector,
  Uint8List bytes,
  FaceDetectionMode mode,
) async {
  for (int i = 0; i < _warmupIterations; i++) {
    await detector.detectFacesFromBytes(bytes, mode: mode);
  }

  final timings = <double>[];
  var faceCount = 0;
  Face? firstFace;
  for (int i = 0; i < _benchmarkIterations; i++) {
    final sw = Stopwatch()..start();
    final faces = await detector.detectFacesFromBytes(bytes, mode: mode);
    sw.stop();
    timings.add(sw.elapsedMicroseconds / 1000.0);
    if (i == 0) {
      faceCount = faces.length;
      firstFace = faces.isNotEmpty ? faces.first : null;
    }
  }

  return _BenchStats(
    timingsMs: timings,
    faceCount: faceCount,
    firstFace: firstFace,
  );
}

double _dist(Point a, Point b) {
  final dx = a.x - b.x;
  final dy = a.y - b.y;
  return math.sqrt((dx * dx + dy * dy).toDouble());
}

double _diagonal(Face f) =>
    _dist(f.boundingBox.topLeft, f.boundingBox.bottomRight);

class _BenchStats {
  _BenchStats({
    required this.timingsMs,
    required this.faceCount,
    required this.firstFace,
  });

  final List<double> timingsMs;
  final int faceCount;
  final Face? firstFace;

  double get meanMs => timingsMs.reduce((a, b) => a + b) / timingsMs.length;

  double get medianMs {
    final sorted = [...timingsMs]..sort();
    final mid = sorted.length ~/ 2;
    return sorted.length.isOdd
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2.0;
  }

  double get minMs => timingsMs.reduce((a, b) => a < b ? a : b);
}
