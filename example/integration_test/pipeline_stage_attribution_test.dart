// ignore_for_file: avoid_print

// Attributes FULL-mode wall time to individual pipeline stages using the
// CompiledModel engine (the FaceDetector default). Runs every stage with the
// real intermediate data (real detection -> real aligned crop -> real eye
// ROIs), so the per-stage p50s sum to approximately the in-isolate pipeline
// cost; the gap to the end-to-end FaceDetector number is isolate RPC +
// serialization overhead.
//
//   flutter test integration_test/pipeline_stage_attribution_test.dart -d macos

import 'dart:math' as math;

import 'package:face_detection_tflite/face_detection_tflite_native.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int iterations = 60;
const int warmup = 10;

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

  test('FULL pipeline stage attribution (compiledmodel)', () async {
    final jpegBytes = (await rootBundle.load(
      'assets/samples/landmark-ex1.jpg',
    ))
        .buffer
        .asUint8List();
    final mat = cv.imdecode(jpegBytes, cv.IMREAD_COLOR);

    Future<Uint8List> loadModel(String file) async => (await rootBundle.load(
          'packages/face_detection_tflite/assets/models/$file',
        ))
            .buffer
            .asUint8List();

    final detection = await FaceDetection.createCompiledFromBuffer(
      await loadModel(testNameFor(FaceDetectionModel.backCamera)),
      FaceDetectionModel.backCamera,
    );
    final mesh = await FaceLandmark.createCompiledFromBuffer(
      await loadModel(kFaceLandmarkModel),
    );
    final irisBytes = await loadModel(kIrisLandmarkModel);
    final irisLeft = await IrisLandmark.createCompiledFromBuffer(irisBytes);
    final irisRight = await IrisLandmark.createCompiledFromBuffer(irisBytes);

    final rows = <String>[];
    void report(String label, double ms) {
      rows.add('${label.padRight(34)} ${ms.toStringAsFixed(3)} ms');
    }

    // 1. Detection input tensor.
    final detPack = convertImageToTensor(
      mat,
      outW: detection.inputWidth,
      outH: detection.inputHeight,
    );
    report(
      'detect: convertImageToTensor',
      await _bench(() async {
        convertImageToTensor(
          mat,
          outW: detection.inputWidth,
          outH: detection.inputHeight,
        );
      }),
    );

    // 2. Detection inference + postprocess.
    final dets = await detection.callWithTensor(detPack);
    expect(dets, isNotEmpty);
    report(
      'detect: callWithTensor (inf+post)',
      await _bench(() => detection.callWithTensor(detPack)),
    );

    // 3. Aligned face crop from the real detection.
    final align = testComputeFaceAlignment(
      dets.first,
      mat.cols.toDouble(),
      mat.rows.toDouble(),
    );
    final faceCrop = extractAlignedSquare(
      mat,
      align.cx,
      align.cy,
      align.size,
      -align.theta,
      outSize: mesh.inputWidth,
    )!;
    print('face crop side: ${faceCrop.cols}px (roi ${align.size.round()}px)');
    report(
      'mesh: extractAlignedSquare',
      await _bench(() async {
        extractAlignedSquare(
          mat,
          align.cx,
          align.cy,
          align.size,
          -align.theta,
          outSize: mesh.inputWidth,
        )!
            .dispose();
      }),
    );

    // 4. Mesh model call (tensor conversion + inference + unpack).
    final lmNorm = await mesh.call(faceCrop);
    report('mesh: FaceLandmark.call', await _bench(() => mesh.call(faceCrop)));

    // 5. Mesh transform to absolute.
    final meshAbs = testTransformMeshToAbsolute(
      lmNorm,
      align.cx,
      align.cy,
      align.size,
      align.theta,
    );
    report(
      'mesh: transform to absolute',
      await _bench(() async {
        testTransformMeshToAbsolute(
          lmNorm,
          align.cx,
          align.cy,
          align.size,
          align.theta,
        );
      }),
    );

    // 6. Eye ROIs and crops from the real mesh.
    AlignedRoi roiFromCorners(int a, int b) {
      final p0 = meshAbs[a];
      final p1 = meshAbs[b];
      final cx = (p0.x + p1.x) * 0.5;
      final cy = (p0.y + p1.y) * 0.5;
      final dx = p1.x - p0.x;
      final dy = p1.y - p0.y;
      final eyeDist = math.sqrt(dx * dx + dy * dy);
      return AlignedRoi(cx, cy, eyeDist * 2.3, math.atan2(dy, dx));
    }

    final leftRoi = roiFromCorners(33, 133);
    final rightRoi = roiFromCorners(362, 263);

    final leftCrop = extractAlignedSquare(
      mat,
      leftRoi.cx,
      leftRoi.cy,
      leftRoi.size,
      leftRoi.theta,
      outSize: irisLeft.inputWidth,
    )!;
    final rightCropRaw = extractAlignedSquare(
      mat,
      rightRoi.cx,
      rightRoi.cy,
      rightRoi.size,
      rightRoi.theta,
      outSize: irisRight.inputWidth,
    )!;
    print(
      'eye crop side: ${leftCrop.cols}px (roi ${leftRoi.size.round()}px)',
    );
    report(
      'iris: extractAlignedSquare (x2)',
      await _bench(() async {
        extractAlignedSquare(
          mat,
          leftRoi.cx,
          leftRoi.cy,
          leftRoi.size,
          leftRoi.theta,
          outSize: irisLeft.inputWidth,
        )!
            .dispose();
        extractAlignedSquare(
          mat,
          rightRoi.cx,
          rightRoi.cy,
          rightRoi.size,
          rightRoi.theta,
          outSize: irisRight.inputWidth,
        )!
            .dispose();
      }),
    );

    final rightCrop = cv.flip(rightCropRaw, 1);
    report(
      'iris: cv.flip (right eye)',
      await _bench(() async {
        cv.flip(rightCropRaw, 1).dispose();
      }),
    );

    // 7. Iris model calls (both eyes, as the core runs them).
    report(
      'iris: IrisLandmark.call (x2)',
      await _bench(() async {
        await Future.wait([irisLeft.call(leftCrop), irisRight.call(rightCrop)]);
      }),
    );

    print('\nSTAGE ATTRIBUTION (p50, compiledmodel, landmark-ex1)');
    for (final r in rows) {
      print(r);
    }

    faceCrop.dispose();
    leftCrop.dispose();
    rightCropRaw.dispose();
    rightCrop.dispose();
    mat.dispose();
    detection.dispose();
    mesh.dispose();
    irisLeft.dispose();
    irisRight.dispose();
  }, timeout: const Timeout(Duration(minutes: 10)));
}
