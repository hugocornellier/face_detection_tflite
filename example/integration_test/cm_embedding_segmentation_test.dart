// ignore_for_file: avoid_print

// Verifies the CompiledModel-backed FaceEmbedding and SelfieSegmentation
// paths: parity against the Interpreter engine on real images, plus the
// Interpreter fallback for the custom-op binary segmentation models.
//
//   flutter test integration_test/cm_embedding_segmentation_test.dart -d macos

import 'dart:math' as math;

import 'package:face_detection_tflite/face_detection_tflite_native.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const String _faceImage = 'assets/samples/embedding_test/one_face.jpg';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List faceBytes;

  setUpAll(() async {
    faceBytes = (await rootBundle.load(_faceImage)).buffer.asUint8List();
  });

  test('CompiledModel embedding matches Interpreter embedding', () async {
    final interp = await FaceDetector.create(useCompiledModel: false);
    final compiled = await FaceDetector.create(useCompiledModel: true);
    try {
      final interpFaces = await interp.detectFacesFromBytes(
        faceBytes,
        mode: FaceDetectionMode.fast,
      );
      final compiledFaces = await compiled.detectFacesFromBytes(
        faceBytes,
        mode: FaceDetectionMode.fast,
      );
      expect(interpFaces.length, 1);
      expect(compiledFaces.length, 1);

      final a = await interp.getFaceEmbedding(interpFaces.first, faceBytes);
      final b = await compiled.getFaceEmbedding(
        compiledFaces.first,
        faceBytes,
      );

      expect(b.length, a.length, reason: 'embedding dimensions must match');
      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      print(
        'embedding dims=${a.length}, '
        'interp-vs-compiled cosine=${similarity.toStringAsFixed(6)}',
      );
      expect(
        similarity,
        greaterThan(0.99),
        reason: 'engines must agree on the same face',
      );
    } finally {
      await interp.dispose();
      await compiled.dispose();
    }
  });

  test('CompiledModel multiclass segmentation matches Interpreter', () async {
    const config = SegmentationConfig(model: SegmentationModel.multiclass);

    final interp = await FaceDetector.create(
      withSegmentation: true,
      segmentationConfig: config,
      useCompiledModel: false,
    );
    final compiled = await FaceDetector.create(
      withSegmentation: true,
      segmentationConfig: config,
      useCompiledModel: true,
    );
    try {
      final maskA = await interp.getSegmentationMask(faceBytes);
      final maskB = await compiled.getSegmentationMask(faceBytes);

      expect(maskB.width, maskA.width);
      expect(maskB.height, maskA.height);
      expect(maskA, isA<MulticlassSegmentationMask>());
      expect(maskB, isA<MulticlassSegmentationMask>());

      double sumDiff = 0;
      double maxDiff = 0;
      for (int i = 0; i < maskA.data.length; i++) {
        final d = (maskA.data[i] - maskB.data[i]).abs();
        sumDiff += d;
        maxDiff = math.max(maxDiff, d);
      }
      final meanDiff = sumDiff / maskA.data.length;
      final fg = maskB.data.where((v) => v > 0.5).length / maskB.data.length;
      print(
        'multiclass mask ${maskB.width}x${maskB.height}, '
        'foreground=${(fg * 100).toStringAsFixed(1)}%, '
        'meanDiff=${meanDiff.toStringAsFixed(5)}, '
        'maxDiff=${maxDiff.toStringAsFixed(5)}',
      );
      expect(fg, greaterThan(0.05), reason: 'person must be segmented');
      expect(meanDiff, lessThan(0.02), reason: 'engines must agree');
    } finally {
      await interp.dispose();
      await compiled.dispose();
    }
  });

  test(
    'binary segmentation pipeline works under useCompiledModel',
    () async {
      // general/landscape use the Convolution2DTransposeBias custom op; the
      // segmentation isolate compiles it where the runtime supports it and
      // falls back to the Interpreter otherwise; either way the pipeline
      // must produce a usable mask.
      final detector = await FaceDetector.create(
        withSegmentation: true,
        segmentationConfig: const SegmentationConfig(
          model: SegmentationModel.general,
        ),
        useCompiledModel: true,
      );
      try {
        final mask = await detector.getSegmentationMask(faceBytes);
        final fg = mask.data.where((v) => v > 0.5).length / mask.data.length;
        print(
          'general (fallback) mask ${mask.width}x${mask.height}, '
          'foreground=${(fg * 100).toStringAsFixed(1)}%',
        );
        expect(mask.data.length, mask.width * mask.height);
        expect(fg, greaterThan(0.05), reason: 'person must be segmented');
      } finally {
        await detector.dispose();
      }
    },
  );

  test('binary (custom-op) model compiles and matches Interpreter', () async {
    // The general model uses the Convolution2DTransposeBias custom op, yet
    // LiteRT Next compiles it; verify the compiled output actually matches
    // the Interpreter rather than trusting compilation success.
    final segBytes = (await rootBundle.load(
      'packages/face_detection_tflite/assets/models/${testModelFileFor(SegmentationModel.general)}',
    ))
        .buffer
        .asUint8List();

    final interp = await SelfieSegmentation.createFromBuffer(segBytes);
    final compiled = await SelfieSegmentation.createCompiledFromBuffer(
      segBytes,
    );
    try {
      final maskA = await interp.callFromBytes(faceBytes);
      final maskB = await compiled.callFromBytes(faceBytes);

      expect(maskB.width, maskA.width);
      expect(maskB.height, maskA.height);
      double sumDiff = 0;
      for (int i = 0; i < maskA.data.length; i++) {
        sumDiff += (maskA.data[i] - maskB.data[i]).abs();
      }
      final meanDiff = sumDiff / maskA.data.length;
      print(
        'general compiled-vs-interp mask ${maskB.width}x${maskB.height}, '
        'meanDiff=${meanDiff.toStringAsFixed(5)}',
      );
      expect(meanDiff, lessThan(0.02), reason: 'engines must agree');
    } finally {
      interp.dispose();
      compiled.dispose();
    }
  });
}
