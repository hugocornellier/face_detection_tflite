// ignore_for_file: avoid_print

import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int iterations = 20;
const List<String> sampleImages = [
  'assets/samples/landmark-ex1.jpg',
  'assets/samples/iris-detection-ex1.jpg',
  'assets/samples/iris-detection-ex2.jpg',
  'assets/samples/group-shot-bounding-box-ex1.jpeg',
  'assets/samples/mesh-ex1.jpeg',
];

/// Statistics for a single image benchmark
class BenchmarkStats {
  final String imagePath;
  final List<int> timings;
  final int imageSize;
  final int detectionCount;

  BenchmarkStats({
    required this.imagePath,
    required this.timings,
    required this.imageSize,
    required this.detectionCount,
  });

  double get mean => timings.reduce((a, b) => a + b) / timings.length;

  double get median {
    final sorted = List<int>.from(timings)..sort();
    final middle = sorted.length ~/ 2;
    if (sorted.length % 2 == 1) {
      return sorted[middle].toDouble();
    } else {
      return (sorted[middle - 1] + sorted[middle]) / 2.0;
    }
  }

  int get min => timings.reduce((a, b) => a < b ? a : b);
  int get max => timings.reduce((a, b) => a > b ? a : b);

  double get stdDev {
    final m = mean;
    final variance =
        timings.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b) /
            timings.length;
    return variance > 0 ? variance : 0.0;
  }

  void printResults(String label) {
    print('\n$label:');
    print('  Image size: ${(imageSize / 1024).toStringAsFixed(1)} KB');
    print('  Detections: $detectionCount face(s)');
    print('  Mean:   ${mean.toStringAsFixed(2)} ms');
    print('  Median: ${median.toStringAsFixed(2)} ms');
    print('  Min:    $min ms');
    print('  Max:    $max ms');
    print('  StdDev: ${stdDev.toStringAsFixed(2)} ms');
  }

  Map<String, dynamic> toJson() => {
        'image_path': imagePath,
        'image_size_kb': (imageSize / 1024),
        'detection_count': detectionCount,
        'iterations': timings.length,
        'timings_ms': timings,
        'mean_ms': mean,
        'median_ms': median,
        'min_ms': min,
        'max_ms': max,
        'stddev_ms': stdDev,
      };
}

/// Aggregated benchmark results
class BenchmarkResults {
  final String timestamp;
  final String testName;
  final Map<String, dynamic> configuration;
  final List<BenchmarkStats> results;

  BenchmarkResults({
    required this.timestamp,
    required this.testName,
    required this.configuration,
    required this.results,
  });

  double get overallMean {
    final allTimings = results.expand((r) => r.timings).toList();
    return allTimings.reduce((a, b) => a + b) / allTimings.length;
  }

  void printSummary() {
    print('\n${'=' * 60}');
    print('BENCHMARK SUMMARY');
    print('=' * 60);
    print('Test: $testName');
    print('Timestamp: $timestamp');
    print('Configuration:');
    configuration.forEach((key, value) {
      print('  $key: $value');
    });
    print('\nOverall mean: ${overallMean.toStringAsFixed(2)} ms');
    print('Total iterations: ${results.length * iterations}');
    print('=' * 60);
  }

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp,
        'test_name': testName,
        'configuration': configuration,
        'overall_mean_ms': overallMean,
        'results': results.map((r) => r.toJson()).toList(),
      };

  void printJson(String filename) {
    print('\n📊 BENCHMARK_JSON_START:$filename');
    print(const JsonEncoder.withIndent('  ').convert(toJson()));
    print('📊 BENCHMARK_JSON_END:$filename');
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('SelfieSegmentation - Performance Benchmarks', () {
    test(
      'Benchmark segmentation with OpenCV decode (optimized path)',
      () async {
        final segmenter = await SelfieSegmentation.create(
          config: SegmentationConfig(
            performanceConfig: PerformanceConfig.xnnpack(),
          ),
        );

        print('\n${'=' * 60}');
        print('BENCHMARK: Segmentation with OpenCV decode (call) + XNNPACK');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];

          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await segmenter.call(bytes);
            stopwatch.stop();
            timings.add(stopwatch.elapsedMilliseconds);
          }

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: 1, // Segmentation always produces 1 mask
          );
          stats.printResults(imagePath);
          allStats.add(stats);
        }

        segmenter.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Segmentation with OpenCV decode (call) + XNNPACK',
          configuration: {
            'performance_config': 'xnnpack',
            'api': 'call (OpenCV decode)',
            'iterations': iterations,
            'sample_images': sampleImages.length,
          },
          results: allStats,
        );
        benchmarkResults.printSummary();
        benchmarkResults
            .printJson('benchmark_segmentation_opencv_$timestamp.json');
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'Benchmark segmentation FAST mode (no isolate overhead)',
      () async {
        final segmenter = await SelfieSegmentation.create(
          config: SegmentationConfig.fast, // useIsolate: false
        );

        print('\n${'=' * 60}');
        print(
            'BENCHMARK: Segmentation FAST mode (useIsolate: false) + XNNPACK');
        print('Direct interpreter invoke - no isolate serialization overhead');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];

          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await segmenter.call(bytes);
            stopwatch.stop();
            timings.add(stopwatch.elapsedMilliseconds);
          }

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: 1,
          );
          stats.printResults('$imagePath (FAST)');
          allStats.add(stats);
        }

        segmenter.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Segmentation FAST mode (useIsolate: false) + XNNPACK',
          configuration: {
            'performance_config': 'xnnpack',
            'api': 'call (OpenCV decode)',
            'useIsolate': false,
            'iterations': iterations,
            'sample_images': sampleImages.length,
          },
          results: allStats,
        );
        benchmarkResults.printSummary();
        benchmarkResults
            .printJson('benchmark_segmentation_fast_$timestamp.json');
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'Benchmark SegmentationWorker (background isolate, non-blocking)',
      () async {
        final worker = SegmentationWorker();
        await worker.initialize(
          performanceConfig: PerformanceConfig.xnnpack(),
        );

        print('\n${'=' * 60}');
        print('BENCHMARK: SegmentationWorker (background isolate)');
        print('Non-blocking inference with TransferableTypedData transfer');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];

          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await worker.segment(bytes);
            stopwatch.stop();
            timings.add(stopwatch.elapsedMilliseconds);
          }

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: 1,
          );
          stats.printResults('$imagePath (Worker)');
          allStats.add(stats);
        }

        worker.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'SegmentationWorker (background isolate)',
          configuration: {
            'performance_config': 'xnnpack',
            'api': 'SegmentationWorker.segment',
            'blocking': false,
            'iterations': iterations,
            'sample_images': sampleImages.length,
          },
          results: allStats,
        );
        benchmarkResults.printSummary();
        benchmarkResults
            .printJson('benchmark_segmentation_worker_$timestamp.json');
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'Benchmark segmentation with cv.Mat input (callFromMat)',
      () async {
        final segmenter = await SelfieSegmentation.create(
          config: SegmentationConfig(
            performanceConfig: PerformanceConfig.xnnpack(),
          ),
        );

        print('\n${'=' * 60}');
        print('BENCHMARK: Segmentation with cv.Mat (callFromMat) + XNNPACK');
        print('Note: Each iteration decodes fresh Mat');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];

          for (int i = 0; i < iterations; i++) {
            final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

            final stopwatch = Stopwatch()..start();
            await segmenter.callFromMat(mat);
            stopwatch.stop();

            mat.dispose();
            timings.add(stopwatch.elapsedMilliseconds);
          }

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: 1,
          );
          stats.printResults('$imagePath (cv.Mat)');
          allStats.add(stats);
        }

        segmenter.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Segmentation with cv.Mat (callFromMat) + XNNPACK',
          configuration: {
            'performance_config': 'xnnpack',
            'api': 'callFromMat',
            'note': 'Excludes cv.imdecode time',
            'iterations': iterations,
            'sample_images': sampleImages.length,
          },
          results: allStats,
        );
        benchmarkResults.printSummary();
        benchmarkResults
            .printJson('benchmark_segmentation_mat_$timestamp.json');
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'Benchmark segmentation with pure Dart decode (callWithImagePackage)',
      () async {
        final segmenter = await SelfieSegmentation.create(
          config: SegmentationConfig(
            performanceConfig: PerformanceConfig.xnnpack(),
          ),
        );

        print('\n${'=' * 60}');
        print(
            'BENCHMARK: Segmentation with pure Dart decode (callWithImagePackage)');
        print('This is the SLOW path for comparison');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];

          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await segmenter.callWithImagePackage(bytes);
            stopwatch.stop();
            timings.add(stopwatch.elapsedMilliseconds);
          }

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: 1,
          );
          stats.printResults('$imagePath (Dart decode)');
          allStats.add(stats);
        }

        segmenter.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Segmentation with pure Dart decode (callWithImagePackage)',
          configuration: {
            'performance_config': 'xnnpack',
            'api': 'callWithImagePackage (pure Dart)',
            'note': 'SLOW path - uses img.decodeImage()',
            'iterations': iterations,
            'sample_images': sampleImages.length,
          },
          results: allStats,
        );
        benchmarkResults.printSummary();
        benchmarkResults
            .printJson('benchmark_segmentation_dart_$timestamp.json');
      },
      timeout: const Timeout(Duration(minutes: 5)),
    );

    test(
      'Benchmark comparison: Segmentation decode overhead',
      () async {
        final segmenter = await SelfieSegmentation.create(
          config: SegmentationConfig(
            performanceConfig: PerformanceConfig.xnnpack(),
          ),
        );

        print('\n${'=' * 60}');
        print('BENCHMARK: Segmentation Decode Overhead Comparison');
        print(
            'Compares: callFromMat (no decode) vs call (OpenCV) vs callWithImagePackage (Dart)');
        print('=' * 60);

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          // Warmup
          final warmupMat = cv.imdecode(bytes, cv.IMREAD_COLOR);
          await segmenter.callFromMat(warmupMat);
          warmupMat.dispose();

          // 1. callFromMat (pipeline only, no decode)
          final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
          final List<int> matTimings = [];
          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await segmenter.callFromMat(mat);
            stopwatch.stop();
            matTimings.add(stopwatch.elapsedMilliseconds);
          }
          mat.dispose();

          // 2. call (OpenCV decode)
          final List<int> opencvTimings = [];
          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await segmenter.call(bytes);
            stopwatch.stop();
            opencvTimings.add(stopwatch.elapsedMilliseconds);
          }

          // 3. callWithImagePackage (Dart decode)
          final List<int> dartTimings = [];
          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await segmenter.callWithImagePackage(bytes);
            stopwatch.stop();
            dartTimings.add(stopwatch.elapsedMilliseconds);
          }

          final matMean =
              matTimings.reduce((a, b) => a + b) / matTimings.length;
          final opencvMean =
              opencvTimings.reduce((a, b) => a + b) / opencvTimings.length;
          final dartMean =
              dartTimings.reduce((a, b) => a + b) / dartTimings.length;

          print('\n$imagePath:');
          print(
              '  callFromMat (no decode):      ${matMean.toStringAsFixed(1)} ms');
          print(
              '  call (OpenCV decode):         ${opencvMean.toStringAsFixed(1)} ms');
          print(
              '  callWithImagePackage (Dart):  ${dartMean.toStringAsFixed(1)} ms');
          print('  ---');
          print(
              '  OpenCV decode overhead:       ${(opencvMean - matMean).toStringAsFixed(1)} ms');
          print(
              '  Dart decode overhead:         ${(dartMean - matMean).toStringAsFixed(1)} ms');
          print(
              '  OpenCV vs Dart speedup:       ${(dartMean / opencvMean).toStringAsFixed(2)}x');
        }

        segmenter.dispose();
        print('\n${'=' * 60}');
      },
      timeout: const Timeout(Duration(minutes: 5)),
    );
  });

  group('FaceDetector - Performance Benchmarks', () {
    test(
      'Benchmark full mode with OpenCV (native SIMD)',
      () async {
        final detector = FaceDetector();
        await detector.initialize(
          performanceConfig: PerformanceConfig.xnnpack(),
        );

        print('\n${'=' * 60}');
        print('BENCHMARK: Full Mode with OpenCV (detectFaces) + XNNPACK');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];
          int detectionCount = 0;

          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            final results = await detector.detectFaces(
              bytes,
              mode: FaceDetectionMode.full,
            );
            stopwatch.stop();

            timings.add(stopwatch.elapsedMilliseconds);
            if (i == 0) detectionCount = results.length;
          }

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: detectionCount,
          );
          stats.printResults(imagePath);
          allStats.add(stats);
        }

        print('\n${'=' * 60}');
        print('Iris Detection Statistics:');
        print('  Successful: ${detector.irisOkCount}');
        print('  Failed: ${detector.irisFailCount}');
        final totalIris = detector.irisOkCount + detector.irisFailCount;
        if (totalIris > 0) {
          print(
              '  Success rate: ${(detector.irisOkCount / totalIris * 100).toStringAsFixed(1)}%');
        }
        print('=' * 60);

        detector.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Full Mode with OpenCV (detectFaces) + XNNPACK',
          configuration: {
            'mode': 'full',
            'model': 'backCamera (default)',
            'performance_config': 'xnnpack',
            'api': 'opencv (detectFaces)',
            'iterations': iterations,
            'sample_images': sampleImages.length,
          },
          results: allStats,
        );
        benchmarkResults.printSummary();
        benchmarkResults.printJson('benchmark_$timestamp.json');
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'Benchmark full mode with cv.Mat input (fresh decode each iteration)',
      () async {
        final detector = FaceDetector();
        await detector.initialize(
          performanceConfig: PerformanceConfig.xnnpack(),
        );

        print('\n${'=' * 60}');
        print(
            'BENCHMARK: Full Mode with cv.Mat (detectFacesFromMat) + XNNPACK');
        print(
            'Note: Each iteration decodes fresh Mat to avoid opencv_dart state issues');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];
          int detectionCount = 0;

          for (int i = 0; i < iterations; i++) {
            final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

            final stopwatch = Stopwatch()..start();
            final results = await detector.detectFacesFromMat(
              mat,
              mode: FaceDetectionMode.full,
            );
            stopwatch.stop();

            mat.dispose();

            timings.add(stopwatch.elapsedMilliseconds);
            if (i == 0) detectionCount = results.length;
          }

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: detectionCount,
          );
          stats.printResults('$imagePath (cv.Mat)');
          allStats.add(stats);
        }

        print('\n${'=' * 60}');
        print('Iris Detection Statistics:');
        print('  Successful: ${detector.irisOkCount}');
        print('  Failed: ${detector.irisFailCount}');
        final totalIris = detector.irisOkCount + detector.irisFailCount;
        if (totalIris > 0) {
          print(
              '  Success rate: ${(detector.irisOkCount / totalIris * 100).toStringAsFixed(1)}%');
        }
        print('=' * 60);

        detector.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Full Mode with cv.Mat (detectFacesFromMat) + XNNPACK',
          configuration: {
            'mode': 'full',
            'model': 'backCamera (default)',
            'performance_config': 'xnnpack',
            'api': 'opencv mat (detectFacesFromMat)',
            'note': 'Includes cv.imdecode time',
            'iterations': iterations,
            'sample_images': sampleImages.length,
          },
          results: allStats,
        );
        benchmarkResults.printSummary();
        benchmarkResults.printJson('benchmark_mat_$timestamp.json');
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'Benchmark comparison: Pipeline only (excludes decode)',
      () async {
        final detector = FaceDetector();
        await detector.initialize(
          performanceConfig: PerformanceConfig.xnnpack(),
        );

        print('\n${'=' * 60}');
        print('BENCHMARK: Pipeline Performance (excludes image decode)');
        print('This simulates live camera where frames are already decoded');
        print('=' * 60);

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

          final List<int> opencvTimings = [];
          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await detector.detectFacesFromMat(mat,
                mode: FaceDetectionMode.full);
            stopwatch.stop();
            opencvTimings.add(stopwatch.elapsedMilliseconds);
          }

          mat.dispose();

          final List<int> fullTimings = [];
          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
            stopwatch.stop();
            fullTimings.add(stopwatch.elapsedMilliseconds);
          }

          final opencvMean =
              opencvTimings.reduce((a, b) => a + b) / opencvTimings.length;
          final fullMean =
              fullTimings.reduce((a, b) => a + b) / fullTimings.length;
          final decodeSavings = fullMean - opencvMean;

          print('\n$imagePath:');
          print(
              '  OpenCV pipeline only:     ${opencvMean.toStringAsFixed(1)} ms');
          print(
              '  Full API (with decode):   ${fullMean.toStringAsFixed(1)} ms');
          print(
              '  Decode overhead:          ${decodeSavings.toStringAsFixed(1)} ms');
          print(
              '  Pipeline is ${(fullMean / opencvMean).toStringAsFixed(1)}x of total time');
        }

        detector.dispose();
        print('\n${'=' * 60}');
      },
      timeout: const Timeout(Duration(minutes: 5)),
    );

    test(
      'Benchmark: Live camera simulation (fresh Mat each frame)',
      () async {
        final detector = FaceDetector();
        await detector.initialize(
          performanceConfig: PerformanceConfig.xnnpack(),
        );

        print('\n${'=' * 60}');
        print('BENCHMARK: Live Camera Simulation');
        print('Fresh cv.Mat each frame (simulates camera frame processing)');
        print('=' * 60);

        final ByteData data = await rootBundle.load(sampleImages[0]);
        final Uint8List bytes = data.buffer.asUint8List();

        for (int i = 0; i < 5; i++) {
          final warmupMat = cv.imdecode(bytes, cv.IMREAD_COLOR);
          await detector.detectFacesFromMat(warmupMat,
              mode: FaceDetectionMode.full);
          warmupMat.dispose();
        }

        const int frames = 30;
        final stopwatch = Stopwatch()..start();
        for (int i = 0; i < frames; i++) {
          final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
          await detector.detectFacesFromMat(mat, mode: FaceDetectionMode.full);
          mat.dispose();
        }
        stopwatch.stop();

        final totalMs = stopwatch.elapsedMilliseconds;
        final avgMs = totalMs / frames;
        final fps = 1000 / avgMs;

        print('\nSustained throughput test ($frames frames):');
        print('  Total time:    $totalMs ms');
        print(
            '  Avg per frame: ${avgMs.toStringAsFixed(1)} ms (includes decode)');
        print('  Throughput:    ${fps.toStringAsFixed(1)} FPS');
        print('=' * 60);

        detector.dispose();
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );
  });
}
