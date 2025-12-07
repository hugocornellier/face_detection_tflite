// ignore_for_file: avoid_print

// Benchmark tests for FaceDetector.
//
// This test measures the performance of face detection across multiple iterations
// and sample images. Results are printed to the console with special markers that
// the runBenchmark.sh script extracts and saves to benchmark_results/*.json files.
//
// To run:
// - Use the runBenchmark.sh script in the project root (recommended)
// - Or run directly: flutter test integration_test/benchmark_test.dart -d macos

import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

// Benchmark configuration
const int iterations = 20;
const List<String> sampleImages = [
  '../assets/samples/landmark-ex1.jpg',
  '../assets/samples/iris-detection-ex1.jpg',
  '../assets/samples/iris-detection-ex2.jpg',
  '../assets/samples/group-shot-bounding-box-ex1.jpeg',
  '../assets/samples/mesh-ex1.jpeg',
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

  group('FaceDetector - Performance Benchmarks', () {
    test(
      'Benchmark full mode with backCamera model + XNNPACK',
      () async {
        final detector = FaceDetector();
        await detector.initialize(
          performanceConfig:
              PerformanceConfig.xnnpack(), // Enable XNNPACK for 2-5x speedup
        );

        print('\n${'=' * 60}');
        print(
            'BENCHMARK: Full Mode (FaceDetectionMode.full) with backCamera + XNNPACK');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];
          int detectionCount = 0;

          // Run iterations
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

        // Print iris statistics
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

        // Write results to file
        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName:
              'Full Mode (FaceDetectionMode.full) with backCamera + XNNPACK',
          configuration: {
            'mode': 'full',
            'model': 'backCamera (default)',
            'performance_config': 'xnnpack',
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
  });
}
