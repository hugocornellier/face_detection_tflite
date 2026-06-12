// ignore_for_file: avoid_print

// Micro-benchmarks for the pipeline glue: (1) Mat reconstruction from raw
// bytes — the current Mat.fromList element-loop vs Mat.create + typed-data
// memcpy; (2) the isolate RPC floor, measured by running FULL mode on a tiny
// black image (no faces -> detection only, overhead-dominated).
//
//   flutter test integration_test/glue_micro_bench_test.dart -d macos

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

  test('Mat reconstruction: fromList vs create+setAll', () async {
    final jpegBytes = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();
    final mat = cv.imdecode(jpegBytes, cv.IMREAD_COLOR);
    final bytes = Uint8List.fromList(mat.data);
    final w = mat.cols, h = mat.rows;
    mat.dispose();

    final tFromList = await _bench(() async {
      cv.Mat.fromList(h, w, cv.MatType.CV_8UC3, bytes).dispose();
    });
    final tCreateCopy = await _bench(() async {
      final m = cv.Mat.create(rows: h, cols: w, type: cv.MatType.CV_8UC3);
      m.data.setAll(0, bytes);
      m.dispose();
    });
    print(
      'mat rebuild ${w}x$h (${(bytes.length / 1e6).toStringAsFixed(1)}MB): '
      'fromList=${tFromList.toStringAsFixed(2)}ms  '
      'create+setAll=${tCreateCopy.toStringAsFixed(2)}ms  '
      '(${(tFromList / tCreateCopy).toStringAsFixed(1)}x)',
    );

    // Sanity: identical pixels.
    final a = cv.Mat.fromList(h, w, cv.MatType.CV_8UC3, bytes);
    final b = cv.Mat.create(rows: h, cols: w, type: cv.MatType.CV_8UC3);
    b.data.setAll(0, bytes);
    expect(a.data, b.data);
    a.dispose();
    b.dispose();
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('non-continuous ROI Mat detects correctly (stride regression)',
      () async {
    final jpegBytes = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();
    final full = cv.imdecode(jpegBytes, cv.IMREAD_COLOR);
    // Pad by 40px on every side, then take an ROI view of the original area:
    // the view shares the padded buffer, so its rows are not contiguous.
    // Before the isContinuous guard in _extractMatFields this shipped
    // scrambled pixels to the isolate.
    final padded = cv.copyMakeBorder(
      full,
      40,
      40,
      40,
      40,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );
    final roi = padded.region(cv.Rect(40, 40, full.cols, full.rows));
    expect(roi.isContinuous, isFalse, reason: 'ROI must be non-continuous');

    final detector = await FaceDetector.create(useCompiledModel: true);
    try {
      final direct = await detector.detectFacesFromMat(
        full,
        mode: FaceDetectionMode.fast,
      );
      final viaRoi = await detector.detectFacesFromMat(
        roi,
        mode: FaceDetectionMode.fast,
      );
      expect(direct.length, 1);
      expect(viaRoi.length, direct.length);
      final a = direct.first.detectionData.boundingBox;
      final b = viaRoi.first.detectionData.boundingBox;
      expect((a.xmin - b.xmin).abs(), lessThan(0.01));
      expect((a.ymin - b.ymin).abs(), lessThan(0.01));
      print(
        'ROI regression: direct=(${a.xmin.toStringAsFixed(3)},'
        '${a.ymin.toStringAsFixed(3)}) roi=(${b.xmin.toStringAsFixed(3)},'
        '${b.ymin.toStringAsFixed(3)})',
      );
    } finally {
      await detector.dispose();
      roi.dispose();
      padded.dispose();
      full.dispose();
    }
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('isolate RPC floor (tiny no-face image, FULL mode)', () async {
    final tiny = cv.Mat.zeros(64, 64, cv.MatType.CV_8UC3);
    final tinyBytes = Uint8List.fromList(tiny.data);
    tiny.dispose();

    for (final compiled in [false, true]) {
      final detector = await FaceDetector.create(useCompiledModel: compiled);
      try {
        final t = await _bench(() async {
          await detector.detectFacesFromMatBytes(
            tinyBytes,
            width: 64,
            height: 64,
            mode: FaceDetectionMode.full,
          );
        });
        print(
          'rpc+detect floor (${compiled ? 'compiled' : 'interp  '}) '
          '${t.toStringAsFixed(2)} ms',
        );
      } finally {
        await detector.dispose();
      }
    }
  }, timeout: const Timeout(Duration(minutes: 5)));
}
