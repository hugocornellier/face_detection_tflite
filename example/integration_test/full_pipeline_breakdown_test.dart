// ignore_for_file: avoid_print

// Stage-by-stage accounting of the FULL-mode detection pipeline: times the
// OpenCV / transfer stages individually, then the end-to-end pipeline through
// both entry points (JPEG bytes vs raw Mat bytes) on both engines, so the
// wall clock can be attributed to inference vs everything else.
//
//   flutter test integration_test/full_pipeline_breakdown_test.dart -d macos

import 'dart:isolate' show TransferableTypedData;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int iterations = 40;
const int warmup = 8;

double _p50(List<int> us) {
  final s = List<int>.from(us)..sort();
  return s[s.length ~/ 2] / 1000.0;
}

Future<double> _bench(Future<void> Function() once) async {
  for (int i = 0; i < warmup; i++) {
    await once();
  }
  final us = <int>[];
  for (int i = 0; i < iterations; i++) {
    final sw = Stopwatch()..start();
    await once();
    sw.stop();
    us.add(sw.elapsedMicroseconds);
  }
  return _p50(us);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('FULL-mode pipeline breakdown (single face)', () async {
    final jpegBytes = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();
    final mat = cv.imdecode(jpegBytes, cv.IMREAD_COLOR);
    final w = mat.cols, h = mat.rows;
    final minDim = w < h ? w : h;
    print('image: ${w}x$h');

    // --- Stage costs (engine-independent) ---
    final tDecode = await _bench(() async {
      cv.imdecode(jpegBytes, cv.IMREAD_COLOR).dispose();
    });
    final tTensor = await _bench(() async {
      convertImageToTensor(mat, outW: 256, outH: 256);
    });
    final faceSize = minDim * 0.45;
    final tFaceCrop = await _bench(() async {
      extractAlignedSquare(mat, w / 2, h / 2, faceSize, 0.1)?.dispose();
    });
    final eyeSize = minDim * 0.10;
    final tEyeCrop = await _bench(() async {
      extractAlignedSquare(mat, w / 2, h / 2, eyeSize, 0.1)?.dispose();
    });
    final rawMatBytes = mat.data;
    final tTransfer = await _bench(() async {
      TransferableTypedData.fromList([rawMatBytes]);
    });

    print('--- stage p50 (engine-independent) ---');
    print('jpeg decode            ${tDecode.toStringAsFixed(2)} ms');
    print('tensor convert (256px) ${tTensor.toStringAsFixed(2)} ms');
    print('face crop warp         ${tFaceCrop.toStringAsFixed(2)} ms');
    print('eye crop warp (x2)     ${tEyeCrop.toStringAsFixed(2)} ms each');
    print('transferable wrap      ${tTransfer.toStringAsFixed(2)} ms');

    // --- End-to-end, both engines, both entry points ---
    for (final compiled in [false, true]) {
      final detector = await FaceDetector.create(useCompiledModel: compiled);
      try {
        final tBytes = await _bench(() async {
          await detector.detectFacesFromBytes(
            jpegBytes,
            mode: FaceDetectionMode.full,
          );
        });
        final tMat = await _bench(() async {
          await detector.detectFacesFromMatBytes(
            rawMatBytes,
            width: w,
            height: h,
            mode: FaceDetectionMode.full,
          );
        });
        final engine = compiled ? 'compiledmodel' : 'interpreter  ';
        print('--- end-to-end FULL, $engine ---');
        print(
            'fromBytes (jpeg decode in worker) ${tBytes.toStringAsFixed(2)} ms');
        print(
            'fromMatBytes (no jpeg decode)     ${tMat.toStringAsFixed(2)} ms');
      } finally {
        await detector.dispose();
      }
    }
    mat.dispose();
  }, timeout: const Timeout(Duration(minutes: 10)));
}
