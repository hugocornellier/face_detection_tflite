// ignore_for_file: avoid_print

// Per-component CompiledModel vs Interpreter benchmark on a real image:
// embedding (mobilefacenet) and all three segmentation variants. Decode
// happens once; each cell times pure inference (median of 50 after 10
// warmup).
//
//   flutter test integration_test/cm_component_bench_test.dart -d macos

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int iterations = 50;
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

  test('embedding + segmentation: CompiledModel vs Interpreter', () async {
    final imgBytes = (await rootBundle.load(
      'assets/samples/embedding_test/one_face.jpg',
    ))
        .buffer
        .asUint8List();
    final mat = cv.imdecode(imgBytes, cv.IMREAD_COLOR);
    final rows = <String>[];

    // Detection (back camera model, the FaceDetector default).
    {
      final detBytes = (await rootBundle.load(
        'packages/face_detection_tflite/assets/models/${testNameFor(FaceDetectionModel.backCamera)}',
      ))
          .buffer
          .asUint8List();
      final interp = await FaceDetection.createFromBuffer(
        detBytes,
        FaceDetectionModel.backCamera,
      );
      final compiled = await FaceDetection.createCompiledFromBuffer(
        detBytes,
        FaceDetectionModel.backCamera,
      );
      final iPack = convertImageToTensor(
        mat,
        outW: interp.inputWidth,
        outH: interp.inputHeight,
      );
      final cPack = convertImageToTensor(
        mat,
        outW: compiled.inputWidth,
        outH: compiled.inputHeight,
      );
      final i = await _bench(() => interp.callWithTensor(iPack));
      final c = await _bench(() => compiled.callWithTensor(cPack));
      rows.add(
        'detection (backCamera)        interp=${i.toStringAsFixed(2)}ms  '
        'compiled=${c.toStringAsFixed(2)}ms  ${(i / c).toStringAsFixed(2)}x',
      );
      interp.dispose();
      compiled.dispose();
    }

    // Mesh (468-point face landmark).
    {
      final meshBytes = (await rootBundle.load(
        'packages/face_detection_tflite/assets/models/$kFaceLandmarkModel',
      ))
          .buffer
          .asUint8List();
      final interp = await FaceLandmark.createFromBuffer(meshBytes);
      final compiled = await FaceLandmark.createCompiledFromBuffer(meshBytes);
      final i = await _bench(() => interp.call(mat));
      final c = await _bench(() => compiled.call(mat));
      rows.add(
        'mesh (face_landmark)          interp=${i.toStringAsFixed(2)}ms  '
        'compiled=${c.toStringAsFixed(2)}ms  ${(i / c).toStringAsFixed(2)}x',
      );
      interp.dispose();
      compiled.dispose();
    }

    // Iris.
    {
      final irisBytes = (await rootBundle.load(
        'packages/face_detection_tflite/assets/models/$kIrisLandmarkModel',
      ))
          .buffer
          .asUint8List();
      final interp = await IrisLandmark.createFromBuffer(irisBytes);
      final compiled = await IrisLandmark.createCompiledFromBuffer(irisBytes);
      final i = await _bench(() => interp.call(mat));
      final c = await _bench(() => compiled.call(mat));
      rows.add(
        'iris (iris_landmark)          interp=${i.toStringAsFixed(2)}ms  '
        'compiled=${c.toStringAsFixed(2)}ms  ${(i / c).toStringAsFixed(2)}x',
      );
      interp.dispose();
      compiled.dispose();
    }

    // Embedding.
    {
      final embBytes = (await rootBundle.load(
        'packages/face_detection_tflite/assets/models/mobilefacenet.tflite',
      ))
          .buffer
          .asUint8List();
      final interp = await FaceEmbedding.createFromBuffer(embBytes);
      final compiled = await FaceEmbedding.createCompiledFromBuffer(embBytes);
      final i = await _bench(() => interp.call(mat));
      final c = await _bench(() => compiled.call(mat));
      rows.add(
        'embedding (mobilefacenet)     interp=${i.toStringAsFixed(2)}ms  '
        'compiled=${c.toStringAsFixed(2)}ms  ${(i / c).toStringAsFixed(2)}x',
      );
      interp.dispose();
      compiled.dispose();
    }

    // Segmentation variants.
    for (final model in SegmentationModel.values) {
      final config = SegmentationConfig(model: model);
      final segBytes = (await rootBundle.load(
        'packages/face_detection_tflite/assets/models/${testModelFileFor(model)}',
      ))
          .buffer
          .asUint8List();
      final interp = await SelfieSegmentation.createFromBuffer(
        segBytes,
        config: config,
      );
      final compiled = await SelfieSegmentation.createCompiledFromBuffer(
        segBytes,
        config: config,
      );
      final i = await _bench(() => interp.call(mat));
      final c = await _bench(() => compiled.call(mat));
      rows.add(
        'segmentation ${model.name.padRight(15)} interp=${i.toStringAsFixed(2)}ms  '
        'compiled=${c.toStringAsFixed(2)}ms  ${(i / c).toStringAsFixed(2)}x',
      );
      interp.dispose();
      compiled.dispose();
    }

    mat.dispose();
    print('\nCOMPONENT BENCH: p50 ms, macOS (speedup = interp/compiled)');
    for (final r in rows) {
      print(r);
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}
