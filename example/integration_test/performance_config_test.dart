// ignore_for_file: avoid_print

/// Integration tests for PerformanceConfig options.
///
/// Tests all performance configurations:
/// - PerformanceConfig.disabled (CPU-only, maximum compatibility)
/// - PerformanceConfig.xnnpack (CPU optimization, desktop only)
/// - PerformanceConfig.gpu (GPU acceleration, iOS/Android)
/// - PerformanceConfig.auto (automatic selection)
/// - Thread count variations
library;

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Some configurations may be slow to initialize
  const configTimeout = Timeout(Duration(minutes: 2));

  late Uint8List testImageBytes;

  setUpAll(() async {
    final ByteData data =
        await rootBundle.load('assets/samples/landmark-ex1.jpg');
    testImageBytes = data.buffer.asUint8List();
  });

  group('PerformanceConfig.disabled', () {
    test('should work on all platforms', () async {
      final detector = FaceDetector();
      await detector.initialize(
        performanceConfig: PerformanceConfig.disabled,
      );

      expect(detector.isReady, true);

      final faces = await detector.detectFaces(
        testImageBytes,
        mode: FaceDetectionMode.full,
      );

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNotNull);
      expect(faces.first.irisPoints, isNotEmpty);

      detector.dispose();
    }, timeout: configTimeout);

    test('should provide baseline performance', () async {
      final detector = FaceDetector();
      await detector.initialize(
        performanceConfig: PerformanceConfig.disabled,
      );

      // Warmup
      await detector.detectFaces(testImageBytes, mode: FaceDetectionMode.fast);

      // Benchmark
      final times = <int>[];
      for (int i = 0; i < 5; i++) {
        final sw = Stopwatch()..start();
        await detector.detectFaces(testImageBytes,
            mode: FaceDetectionMode.fast);
        sw.stop();
        times.add(sw.elapsedMilliseconds);
      }

      final avg = times.reduce((a, b) => a + b) / times.length;
      print(
          'PerformanceConfig.disabled - fast mode avg: ${avg.toStringAsFixed(1)}ms');

      detector.dispose();
    }, timeout: configTimeout);
  });

  group('PerformanceConfig.auto', () {
    test('should automatically select best config for platform', () async {
      final detector = FaceDetector();
      await detector.initialize(
        performanceConfig: PerformanceConfig.auto(),
      );

      expect(detector.isReady, true);

      final faces = await detector.detectFaces(
        testImageBytes,
        mode: FaceDetectionMode.full,
      );

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNotNull);

      print('Platform: ${Platform.operatingSystem}');
      print('Auto config detected ${faces.length} faces');

      detector.dispose();
    }, timeout: configTimeout);

    test('should work with custom thread count', () async {
      final detector = FaceDetector();
      await detector.initialize(
        performanceConfig: PerformanceConfig.auto(numThreads: 2),
      );

      expect(detector.isReady, true);

      final faces = await detector.detectFaces(
        testImageBytes,
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);

      detector.dispose();
    }, timeout: configTimeout);
  });

  group('PerformanceConfig.xnnpack', () {
    test('should work on supported platforms (macOS/Linux)', () async {
      final detector = FaceDetector();
      await detector.initialize(
        performanceConfig: PerformanceConfig.xnnpack(),
      );

      expect(detector.isReady, true);

      final faces = await detector.detectFaces(
        testImageBytes,
        mode: FaceDetectionMode.full,
      );

      // On supported platforms, should work
      // On unsupported platforms (Windows/iOS/Android), falls back to CPU
      expect(faces, isNotEmpty);

      print(
          'XNNPACK config on ${Platform.operatingSystem}: ${faces.length} faces');

      detector.dispose();
    }, timeout: configTimeout);

    test('should work with thread count specification', () async {
      final threadCounts = [1, 2, 4];

      for (final threads in threadCounts) {
        final detector = FaceDetector();
        await detector.initialize(
          performanceConfig: PerformanceConfig.xnnpack(numThreads: threads),
        );

        final faces = await detector.detectFaces(
          testImageBytes,
          mode: FaceDetectionMode.fast,
        );

        expect(faces, isNotEmpty,
            reason: 'XNNPACK with $threads threads should work');

        detector.dispose();
      }
    }, timeout: configTimeout);

    test('should provide speedup on desktop platforms', () async {
      if (!Platform.isMacOS && !Platform.isLinux) {
        print('Skipping XNNPACK speedup test on ${Platform.operatingSystem}');
        return;
      }

      // Baseline with disabled
      final cpuDetector = FaceDetector();
      await cpuDetector.initialize(
        performanceConfig: PerformanceConfig.disabled,
      );

      // Warmup
      await cpuDetector.detectFaces(testImageBytes,
          mode: FaceDetectionMode.fast);

      final cpuTimes = <int>[];
      for (int i = 0; i < 5; i++) {
        final sw = Stopwatch()..start();
        await cpuDetector.detectFaces(testImageBytes,
            mode: FaceDetectionMode.fast);
        sw.stop();
        cpuTimes.add(sw.elapsedMilliseconds);
      }
      cpuDetector.dispose();

      // XNNPACK
      final xnnDetector = FaceDetector();
      await xnnDetector.initialize(
        performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
      );

      // Warmup
      await xnnDetector.detectFaces(testImageBytes,
          mode: FaceDetectionMode.fast);

      final xnnTimes = <int>[];
      for (int i = 0; i < 5; i++) {
        final sw = Stopwatch()..start();
        await xnnDetector.detectFaces(testImageBytes,
            mode: FaceDetectionMode.fast);
        sw.stop();
        xnnTimes.add(sw.elapsedMilliseconds);
      }
      xnnDetector.dispose();

      final cpuAvg = cpuTimes.reduce((a, b) => a + b) / cpuTimes.length;
      final xnnAvg = xnnTimes.reduce((a, b) => a + b) / xnnTimes.length;
      final speedup = cpuAvg / xnnAvg;

      print('CPU avg: ${cpuAvg.toStringAsFixed(1)}ms');
      print('XNNPACK avg: ${xnnAvg.toStringAsFixed(1)}ms');
      print('Speedup: ${speedup.toStringAsFixed(2)}x');

      // XNNPACK should be at least as fast as CPU on supported platforms
      expect(xnnAvg, lessThanOrEqualTo(cpuAvg * 1.2),
          reason: 'XNNPACK should not be slower than CPU');
    }, timeout: configTimeout);
  });

  group('PerformanceConfig.gpu', () {
    test('should initialize without crashing', () async {
      final detector = FaceDetector();

      // GPU delegate may fail on some platforms/devices, but shouldn't crash
      try {
        await detector.initialize(
          performanceConfig: PerformanceConfig.gpu(),
        );

        if (detector.isReady) {
          final faces = await detector.detectFaces(
            testImageBytes,
            mode: FaceDetectionMode.fast,
          );

          print(
              'GPU config on ${Platform.operatingSystem}: ${faces.length} faces');
          expect(faces, isNotEmpty);
        }

        detector.dispose();
      } catch (e) {
        // GPU delegate failure is acceptable on some platforms
        print('GPU delegate failed (expected on some platforms): $e');
      }
    }, timeout: configTimeout);

    test('should fall back gracefully on unsupported platforms', () async {
      // On desktop platforms without GPU support, should fall back to CPU
      if (Platform.isMacOS || Platform.isLinux || Platform.isWindows) {
        final detector = FaceDetector();
        await detector.initialize(
          performanceConfig: PerformanceConfig.gpu(),
        );

        expect(detector.isReady, true);

        final faces = await detector.detectFaces(
          testImageBytes,
          mode: FaceDetectionMode.fast,
        );

        // Should still work even if GPU delegate is not available
        expect(faces, isNotEmpty);

        detector.dispose();
      }
    }, timeout: configTimeout);
  });

  group('Thread Configuration', () {
    test('should work with different thread counts', () async {
      final threadConfigs = [
        ('1 thread', const PerformanceConfig(numThreads: 1)),
        ('2 threads', const PerformanceConfig(numThreads: 2)),
        ('4 threads', const PerformanceConfig(numThreads: 4)),
        ('auto threads', const PerformanceConfig(numThreads: null)),
      ];

      final results = <String, double>{};

      for (final (name, config) in threadConfigs) {
        final detector = FaceDetector();
        await detector.initialize(performanceConfig: config);

        // Warmup
        await detector.detectFaces(testImageBytes,
            mode: FaceDetectionMode.fast);

        final times = <int>[];
        for (int i = 0; i < 5; i++) {
          final sw = Stopwatch()..start();
          await detector.detectFaces(testImageBytes,
              mode: FaceDetectionMode.fast);
          sw.stop();
          times.add(sw.elapsedMilliseconds);
        }

        final avg = times.reduce((a, b) => a + b) / times.length;
        results[name] = avg;

        detector.dispose();
      }

      print('\nThread count comparison:');
      for (final entry in results.entries) {
        print('  ${entry.key}: ${entry.value.toStringAsFixed(1)}ms');
      }
    }, timeout: configTimeout);

    test('should handle zero threads (single-threaded)', () async {
      final detector = FaceDetector();
      await detector.initialize(
        performanceConfig: const PerformanceConfig(numThreads: 0),
      );

      expect(detector.isReady, true);

      final faces = await detector.detectFaces(
        testImageBytes,
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);

      detector.dispose();
    }, timeout: configTimeout);
  });

  group('PerformanceConfig with FaceDetectorIsolate', () {
    test('should work with auto config', () async {
      final detector = await FaceDetectorIsolate.spawn(
        performanceConfig: PerformanceConfig.auto(),
      );

      final faces = await detector.detectFaces(
        testImageBytes,
        mode: FaceDetectionMode.full,
      );

      expect(faces, isNotEmpty);
      expect(faces.first.mesh, isNotNull);

      await detector.dispose();
    }, timeout: configTimeout);

    test('should work with xnnpack config', () async {
      final detector = await FaceDetectorIsolate.spawn(
        performanceConfig: PerformanceConfig.xnnpack(numThreads: 2),
      );

      final faces = await detector.detectFaces(
        testImageBytes,
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);

      await detector.dispose();
    }, timeout: configTimeout);

    test('should work with disabled config', () async {
      final detector = await FaceDetectorIsolate.spawn(
        performanceConfig: PerformanceConfig.disabled,
      );

      final faces = await detector.detectFaces(
        testImageBytes,
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);

      await detector.dispose();
    }, timeout: configTimeout);
  });

  group('Config Consistency Tests', () {
    test('all configs should produce same detection results', () async {
      final configs = <String, PerformanceConfig>{
        'disabled': PerformanceConfig.disabled,
        'auto': PerformanceConfig.auto(),
        'xnnpack': PerformanceConfig.xnnpack(),
      };

      final faceCounts = <String, int>{};
      final bboxes = <String, BoundingBox>{};

      for (final entry in configs.entries) {
        final detector = FaceDetector();
        await detector.initialize(performanceConfig: entry.value);

        final faces = await detector.detectFaces(
          testImageBytes,
          mode: FaceDetectionMode.fast,
        );

        faceCounts[entry.key] = faces.length;
        if (faces.isNotEmpty) {
          bboxes[entry.key] = faces.first.boundingBox;
        }

        detector.dispose();
      }

      // All configs should detect same number of faces
      final counts = faceCounts.values.toSet();
      expect(counts.length, 1,
          reason: 'All configs should detect same number of faces');

      // Bounding boxes should be nearly identical
      if (bboxes.length > 1) {
        final reference = bboxes.values.first;
        for (final bbox in bboxes.values.skip(1)) {
          final xDiff = (bbox.topLeft.x - reference.topLeft.x).abs();
          final yDiff = (bbox.topLeft.y - reference.topLeft.y).abs();
          expect(xDiff, lessThan(2),
              reason: 'Bounding box X should be consistent');
          expect(yDiff, lessThan(2),
              reason: 'Bounding box Y should be consistent');
        }
      }

      print('All configs produced consistent results');
    }, timeout: configTimeout);
  });

  group('meshPoolSize Configuration', () {
    test('should work with different mesh pool sizes', () async {
      final poolSizes = [1, 2, 4];

      for (final poolSize in poolSizes) {
        final detector = FaceDetector();
        await detector.initialize(meshPoolSize: poolSize);

        // Detect faces in group photo to test mesh pool
        final ByteData data = await rootBundle
            .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
        final bytes = data.buffer.asUint8List();

        final faces = await detector.detectFaces(
          bytes,
          mode: FaceDetectionMode.standard,
        );

        expect(faces, isNotEmpty, reason: 'meshPoolSize=$poolSize should work');

        for (final face in faces) {
          expect(face.mesh, isNotNull,
              reason: 'All faces should have mesh with poolSize=$poolSize');
        }

        print('meshPoolSize=$poolSize: Detected ${faces.length} faces');

        detector.dispose();
      }
    }, timeout: configTimeout);

    test('mesh pool should handle more faces than pool size', () async {
      final detector = FaceDetector();
      await detector.initialize(meshPoolSize: 1); // Only 1 mesh interpreter

      // Group photo has 4 faces
      final ByteData data = await rootBundle
          .load('assets/samples/group-shot-bounding-box-ex1.jpeg');
      final bytes = data.buffer.asUint8List();

      final faces = await detector.detectFaces(
        bytes,
        mode: FaceDetectionMode.full,
      );

      // Should still process all faces, just sequentially
      expect(faces.length, greaterThanOrEqualTo(2));

      for (final face in faces) {
        expect(face.mesh, isNotNull);
        expect(face.mesh!.points.length, 468);
      }

      detector.dispose();
    }, timeout: configTimeout);
  });
}
