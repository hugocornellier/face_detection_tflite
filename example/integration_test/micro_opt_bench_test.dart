// ignore_for_file: avoid_print
library;

import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('microbench: _unpackLandmarks 1M iterations', () async {
    const int nPoints = 468;
    final Float32List flat = Float32List(nPoints * 3);
    for (int i = 0; i < flat.length; i++) {
      flat[i] = (i % 192).toDouble();
    }
    const List<double> padding = [0.1, 0.1, 0.05, 0.05];
    const int warmup = 10000;
    const int iters = 1000000;

    for (int i = 0; i < warmup; i++) {
      testUnpackLandmarks(flat, 192, 192, padding, clamp: true);
    }

    final sw = Stopwatch()..start();
    for (int i = 0; i < iters; i++) {
      testUnpackLandmarks(flat, 192, 192, padding, clamp: true);
    }
    sw.stop();
    final double nsPerCall = sw.elapsedMicroseconds * 1000 / iters;
    print('\n[UNPACK] $iters calls: ${sw.elapsedMilliseconds} ms total, '
        '${nsPerCall.toStringAsFixed(1)} ns/call');
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('microbench: FaceLandmark.call 1000 iterations', () async {
    final landmark = await FaceLandmark.create(
      performanceConfig: PerformanceConfig.xnnpack(),
    );
    final mat = cv.Mat.zeros(192, 192, cv.MatType.CV_8UC3);
    try {
      for (int y = 0; y < 192; y++) {
        for (int x = 0; x < 192; x++) {
          mat.set(y, x, cv.Vec3b(x & 0xFF, y & 0xFF, (x + y) & 0xFF));
        }
      }
      const int warmup = 20;
      const int iters = 1000;
      for (int i = 0; i < warmup; i++) {
        await landmark(mat);
      }
      final sw = Stopwatch()..start();
      for (int i = 0; i < iters; i++) {
        await landmark(mat);
      }
      sw.stop();
      final double usPerCall = sw.elapsedMicroseconds / iters;
      print(
          '\n[FACELANDMARK] $iters calls: ${sw.elapsedMilliseconds} ms total, '
          '${usPerCall.toStringAsFixed(1)} µs/call');
    } finally {
      mat.dispose();
      landmark.dispose();
    }
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('microbench: IrisLandmark.call 1000 iterations', () async {
    final iris = await IrisLandmark.create(
      performanceConfig: PerformanceConfig.xnnpack(),
    );
    final mat = cv.Mat.zeros(64, 64, cv.MatType.CV_8UC3);
    try {
      for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
          mat.set(y, x, cv.Vec3b(x & 0xFF, y & 0xFF, (x + y) & 0xFF));
        }
      }
      const int warmup = 20;
      const int iters = 1000;
      for (int i = 0; i < warmup; i++) {
        await iris(mat);
      }
      final sw = Stopwatch()..start();
      for (int i = 0; i < iters; i++) {
        await iris(mat);
      }
      sw.stop();
      final double usPerCall = sw.elapsedMicroseconds / iters;
      print(
          '\n[IRISLANDMARK] $iters calls: ${sw.elapsedMilliseconds} ms total, '
          '${usPerCall.toStringAsFixed(1)} µs/call');
    } finally {
      mat.dispose();
      iris.dispose();
    }
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('microbench: FaceDetection.callWithTensor 2000 iterations', () async {
    final detection = await FaceDetection.create(
      FaceDetectionModel.backCamera,
      performanceConfig: PerformanceConfig.xnnpack(),
    );
    final mat = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);
    try {
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          mat.set(y, x, cv.Vec3b(x & 0xFF, y & 0xFF, (x + y) & 0xFF));
        }
      }
      final pack = convertImageToTensor(mat, outW: 256, outH: 256);
      const int warmup = 50;
      const int iters = 2000;
      for (int i = 0; i < warmup; i++) {
        await detection.callWithTensor(pack);
      }
      final sw = Stopwatch()..start();
      for (int i = 0; i < iters; i++) {
        await detection.callWithTensor(pack);
      }
      sw.stop();
      final double usPerCall = sw.elapsedMicroseconds / iters;
      print(
          '\n[FACEDETECTION] $iters calls: ${sw.elapsedMilliseconds} ms total, '
          '${usPerCall.toStringAsFixed(1)} µs/call');
    } finally {
      mat.dispose();
      detection.dispose();
    }
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('microbench: FaceDetector.detectFacesFromMat 300 iterations', () async {
    final detector = FaceDetector();
    await detector.initialize(
      performanceConfig: PerformanceConfig.xnnpack(),
    );
    // Real face image so mesh + iris paths are exercised (to hit the big
    // isolate-serialization change: mesh points packed as Float32List).
    final ByteData data =
        await rootBundle.load('assets/samples/landmark-ex1.jpg');
    final bytes = data.buffer.asUint8List();
    final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
    try {
      const int warmup = 10;
      const int iters = 300;
      for (int i = 0; i < warmup; i++) {
        await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);
      }
      final sw = Stopwatch()..start();
      for (int i = 0; i < iters; i++) {
        await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);
      }
      sw.stop();
      final double usPerCall = sw.elapsedMicroseconds / iters;
      print(
          '\n[DETECTFACESFROMMAT] $iters calls: ${sw.elapsedMilliseconds} ms total, '
          '${usPerCall.toStringAsFixed(1)} µs/call');
    } finally {
      mat.dispose();
      detector.dispose();
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}
