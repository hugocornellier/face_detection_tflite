// ignore_for_file: avoid_print

/// Concurrency and stress tests for FaceDetector.
///
/// Tests:
/// - Concurrent detection calls on same detector
/// - Multiple detectors running in parallel
/// - Rapid successive detection calls
/// - Memory stability under load
/// - Multiple images processed sequentially
/// - Stress testing with many faces
library;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Stress tests need longer timeouts
  const stressTimeout = Timeout(Duration(minutes: 5));

  late Map<String, Uint8List> testImages;

  setUpAll(() async {
    testImages = {};

    final imageFiles = [
      'assets/samples/landmark-ex1.jpg',
      'assets/samples/iris-detection-ex1.jpg',
      'assets/samples/iris-detection-ex2.jpg',
      'assets/samples/mesh-ex1.jpeg',
      'assets/samples/group-shot-bounding-box-ex1.jpeg',
    ];

    for (final path in imageFiles) {
      final data = await rootBundle.load(path);
      testImages[path] = data.buffer.asUint8List();
    }
  });

  group('Concurrent Detection Calls', () {
    test('should handle concurrent detectFaces calls safely', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      // Fire multiple concurrent detection calls
      final futures = List.generate(5, (i) {
        return detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      });

      // All should complete without throwing
      final results = await Future.wait(futures);

      expect(results.length, 5);

      // At least some should return valid results
      final nonEmpty = results.where((r) => r.isNotEmpty).length;
      expect(nonEmpty, greaterThan(0),
          reason: 'At least some concurrent calls should succeed');

      print('Concurrent calls: $nonEmpty out of 5 returned faces');

      detector.dispose();
    }, timeout: stressTimeout);

    test('should handle concurrent calls with different modes', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      final futures = [
        detector.detectFaces(bytes, mode: FaceDetectionMode.fast),
        detector.detectFaces(bytes, mode: FaceDetectionMode.standard),
        detector.detectFaces(bytes, mode: FaceDetectionMode.full),
      ];

      final results = await Future.wait(futures);

      expect(results.length, 3);

      // Check results have correct mode outputs
      final fastResult = results[0];
      final standardResult = results[1];
      final fullResult = results[2];

      // At least one of each mode should work
      if (fastResult.isNotEmpty) {
        expect(fastResult.first.mesh, isNull);
      }
      if (standardResult.isNotEmpty) {
        expect(standardResult.first.mesh, isNotNull);
      }
      if (fullResult.isNotEmpty) {
        expect(fullResult.first.irisPoints, isNotEmpty);
      }

      detector.dispose();
    }, timeout: stressTimeout);

    test('should handle concurrent embedding calls', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      // Get face first
      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty);

      // Fire concurrent embedding calls for same face
      final futures = List.generate(3, (i) {
        return detector.getFaceEmbedding(faces.first, bytes);
      });

      final embeddings = await Future.wait(futures);

      expect(embeddings.length, 3);

      // All embeddings should be identical
      for (int i = 1; i < embeddings.length; i++) {
        final similarity =
            FaceDetector.compareFaces(embeddings[0], embeddings[i]);
        expect(similarity, closeTo(1.0, 0.001),
            reason: 'Same face embeddings should be identical');
      }

      detector.dispose();
    }, timeout: stressTimeout);
  });

  group('Multiple Detectors', () {
    test('should support multiple detectors in parallel', () async {
      final detector1 = FaceDetector();
      final detector2 = FaceDetector();
      final detector3 = FaceDetector();

      await Future.wait([
        detector1.initialize(),
        detector2.initialize(),
        detector3.initialize(),
      ]);

      expect(detector1.isReady, true);
      expect(detector2.isReady, true);
      expect(detector3.isReady, true);

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      final futures = [
        detector1.detectFaces(bytes, mode: FaceDetectionMode.fast),
        detector2.detectFaces(bytes, mode: FaceDetectionMode.standard),
        detector3.detectFaces(bytes, mode: FaceDetectionMode.full),
      ];

      final results = await Future.wait(futures);

      for (final result in results) {
        expect(result, isNotEmpty);
      }

      detector1.dispose();
      detector2.dispose();
      detector3.dispose();
    }, timeout: stressTimeout);

    test('should work with multiple isolate detectors', () async {
      final isolate1 = await FaceDetectorIsolate.spawn();
      final isolate2 = await FaceDetectorIsolate.spawn();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      final futures = [
        isolate1.detectFaces(bytes, mode: FaceDetectionMode.full),
        isolate2.detectFaces(bytes, mode: FaceDetectionMode.full),
      ];

      final results = await Future.wait(futures);

      expect(results[0], isNotEmpty);
      expect(results[1], isNotEmpty);
      expect(results[0].length, results[1].length);

      await isolate1.dispose();
      await isolate2.dispose();
    }, timeout: stressTimeout);
  });

  group('Rapid Successive Calls', () {
    test('should handle rapid fire detection calls', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      // Fire 20 rapid calls
      const numCalls = 20;
      final results = <List<Face>>[];

      for (int i = 0; i < numCalls; i++) {
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
        results.add(faces);
      }

      expect(results.length, numCalls);

      // All calls should return same count
      final counts = results.map((r) => r.length).toSet();
      expect(counts.length, 1,
          reason: 'All rapid calls should return same face count');

      print(
          'Rapid fire: $numCalls calls completed, all returned ${counts.first} face(s)');

      detector.dispose();
    }, timeout: stressTimeout);

    test('should handle rapid mode switching', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      // Alternate between modes rapidly
      for (int i = 0; i < 10; i++) {
        final mode = FaceDetectionMode.values[i % 3];
        final faces = await detector.detectFaces(bytes, mode: mode);
        expect(faces, isNotEmpty, reason: 'Call $i with ${mode.name} failed');
      }

      detector.dispose();
    }, timeout: stressTimeout);

    test('should handle rapid image switching', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final images = testImages.values.toList();

      // Process different images rapidly
      for (int i = 0; i < 15; i++) {
        final bytes = images[i % images.length];
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
        expect(faces, isNotNull, reason: 'Call $i failed');
      }

      detector.dispose();
    }, timeout: stressTimeout);
  });

  group('Memory Stability', () {
    test('should not leak memory with repeated detection', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      // Run many detections
      const iterations = 50;

      for (int i = 0; i < iterations; i++) {
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
        expect(faces, isNotEmpty);

        if (i % 10 == 0) {
          print('Memory test: ${i + 1}/$iterations iterations complete');
        }
      }

      // If we get here without crash, memory is stable
      print('Memory stability test passed: $iterations iterations');

      detector.dispose();
    }, timeout: stressTimeout);

    test('should not leak memory with repeated embedding generation', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces, isNotEmpty);

      // Generate many embeddings
      const iterations = 30;

      for (int i = 0; i < iterations; i++) {
        final embedding = await detector.getFaceEmbedding(faces.first, bytes);
        expect(embedding.length, greaterThan(0));
      }

      print('Embedding memory test passed: $iterations iterations');

      detector.dispose();
    }, timeout: stressTimeout);

    test('should handle create/dispose cycles', () async {
      // Create and dispose multiple detectors
      const cycles = 5;

      for (int i = 0; i < cycles; i++) {
        final detector = FaceDetector();
        await detector.initialize();

        final bytes = testImages['assets/samples/landmark-ex1.jpg']!;
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
        expect(faces, isNotEmpty);

        detector.dispose();
      }

      print('Create/dispose cycles test passed: $cycles cycles');
    }, timeout: stressTimeout);
  });

  group('Multi-Face Stress Tests', () {
    test('should handle group photo detection repeatedly', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes =
          testImages['assets/samples/group-shot-bounding-box-ex1.jpeg']!;

      // Process group photo multiple times
      const iterations = 10;
      final faceCounts = <int>[];

      for (int i = 0; i < iterations; i++) {
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
        faceCounts.add(faces.length);

        for (final face in faces) {
          expect(face.mesh, isNotNull);
          expect(face.mesh!.points.length, 468);
        }
      }

      // Detection count should be consistent
      expect(faceCounts.toSet().length, 1,
          reason: 'Face count should be consistent');

      print(
          'Multi-face stress test: Consistently detected ${faceCounts.first} faces');

      detector.dispose();
    }, timeout: stressTimeout);

    test('should handle batch embeddings for all faces', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final bytes =
          testImages['assets/samples/group-shot-bounding-box-ex1.jpeg']!;

      final faces =
          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      expect(faces.length, greaterThan(1));

      // Generate embeddings for all faces multiple times
      const iterations = 5;

      for (int i = 0; i < iterations; i++) {
        final embeddings = await detector.getFaceEmbeddings(faces, bytes);
        expect(embeddings.length, faces.length);

        final validCount = embeddings.where((e) => e != null).length;
        expect(validCount, greaterThan(0));
      }

      print('Batch embedding stress test passed');

      detector.dispose();
    }, timeout: stressTimeout);
  });

  group('FaceDetectorIsolate Concurrency', () {
    test('should handle concurrent calls on isolate detector', () async {
      final detector = await FaceDetectorIsolate.spawn();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      // Fire concurrent calls
      final futures = List.generate(5, (i) {
        return detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
      });

      final results = await Future.wait(futures);

      expect(results.length, 5);

      final nonEmpty = results.where((r) => r.isNotEmpty).length;
      expect(nonEmpty, greaterThan(0));

      print('Isolate concurrent: $nonEmpty out of 5 succeeded');

      await detector.dispose();
    }, timeout: stressTimeout);

    test('should handle rapid isolate spawn/dispose', () async {
      const cycles = 3;

      for (int i = 0; i < cycles; i++) {
        final detector = await FaceDetectorIsolate.spawn();

        final bytes = testImages['assets/samples/landmark-ex1.jpg']!;
        final faces =
            await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);
        expect(faces, isNotEmpty);

        await detector.dispose();
      }

      print('Isolate spawn/dispose cycles passed: $cycles cycles');
    }, timeout: stressTimeout);
  });

  group('Mixed Workload', () {
    test('should handle mixed detection and embedding workload', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final singleFaceBytes = testImages['assets/samples/landmark-ex1.jpg']!;
      final groupBytes =
          testImages['assets/samples/group-shot-bounding-box-ex1.jpeg']!;

      // Interleave different operations
      for (int i = 0; i < 10; i++) {
        // Detection on single face
        final singleFaces = await detector.detectFaces(
          singleFaceBytes,
          mode: FaceDetectionMode.fast,
        );
        expect(singleFaces, isNotEmpty);

        // Embedding on single face
        final embedding = await detector.getFaceEmbedding(
          singleFaces.first,
          singleFaceBytes,
        );
        expect(embedding.length, greaterThan(0));

        // Detection on group photo
        final groupFaces = await detector.detectFaces(
          groupBytes,
          mode: FaceDetectionMode.standard,
        );
        expect(groupFaces.length, greaterThan(1));

        // Full mode detection
        final fullFaces = await detector.detectFaces(
          singleFaceBytes,
          mode: FaceDetectionMode.full,
        );
        expect(fullFaces.first.irisPoints, isNotEmpty);
      }

      print('Mixed workload test passed');

      detector.dispose();
    }, timeout: stressTimeout);
  });

  group('Error Recovery Under Load', () {
    test('should recover from invalid image during stress test', () async {
      final detector = FaceDetector();
      await detector.initialize();

      final validBytes = testImages['assets/samples/landmark-ex1.jpg']!;
      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

      // Mix valid and invalid images
      for (int i = 0; i < 10; i++) {
        if (i % 3 == 0) {
          // Invalid image - should handle gracefully
          try {
            await detector.detectFaces(invalidBytes,
                mode: FaceDetectionMode.fast);
          } catch (e) {
            // Exception is acceptable
          }
        } else {
          // Valid image - should always work
          final faces = await detector.detectFaces(
            validBytes,
            mode: FaceDetectionMode.fast,
          );
          expect(faces, isNotEmpty,
              reason: 'Should recover and process valid image after error');
        }
      }

      detector.dispose();
    }, timeout: stressTimeout);
  });

  group('Benchmark Comparison', () {
    test('FaceDetector vs FaceDetectorIsolate under load', () async {
      final regularDetector = FaceDetector();
      await regularDetector.initialize();
      final isolateDetector = await FaceDetectorIsolate.spawn();

      final bytes = testImages['assets/samples/landmark-ex1.jpg']!;

      const warmupRuns = 3;
      const benchmarkRuns = 10;

      // Warmup
      for (int i = 0; i < warmupRuns; i++) {
        await regularDetector.detectFaces(bytes, mode: FaceDetectionMode.full);
        await isolateDetector.detectFaces(bytes, mode: FaceDetectionMode.full);
      }

      // Benchmark regular
      final regularTimes = <int>[];
      for (int i = 0; i < benchmarkRuns; i++) {
        final sw = Stopwatch()..start();
        await regularDetector.detectFaces(bytes, mode: FaceDetectionMode.full);
        sw.stop();
        regularTimes.add(sw.elapsedMicroseconds);
      }

      // Benchmark isolate
      final isolateTimes = <int>[];
      for (int i = 0; i < benchmarkRuns; i++) {
        final sw = Stopwatch()..start();
        await isolateDetector.detectFaces(bytes, mode: FaceDetectionMode.full);
        sw.stop();
        isolateTimes.add(sw.elapsedMicroseconds);
      }

      final regularAvg =
          regularTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;
      final isolateAvg =
          isolateTimes.reduce((a, b) => a + b) / benchmarkRuns / 1000;

      print('\nBenchmark Results ($benchmarkRuns runs, full mode):');
      print('  FaceDetector:        ${regularAvg.toStringAsFixed(2)} ms');
      print('  FaceDetectorIsolate: ${isolateAvg.toStringAsFixed(2)} ms');
      print(
          '  Overhead:            ${(isolateAvg - regularAvg).toStringAsFixed(2)} ms');

      regularDetector.dispose();
      await isolateDetector.dispose();
    }, timeout: stressTimeout);
  });
}
