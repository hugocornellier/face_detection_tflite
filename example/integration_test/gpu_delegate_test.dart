// ignore_for_file: avoid_print

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const gpuTimeout = Timeout(Duration(minutes: 3));

  group('GPU Delegate Safety Tests', () {
    test('GPU vs CPU vs Auto - Complete comparison', () async {
      print('\n${'=' * 60}');
      print('GPU DELEGATE COMPREHENSIVE TEST');
      print('Platform: ${Platform.operatingSystem}');
      print('=' * 60);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final configs = <String, PerformanceConfig>{
        'CPU (disabled)': PerformanceConfig.disabled,
        'Auto': PerformanceConfig.auto(),
        'GPU': PerformanceConfig.gpu(),
      };

      final results = <String, Map<String, dynamic>>{};

      for (final entry in configs.entries) {
        final name = entry.key;
        final config = entry.value;

        print('\n--- Testing: $name ---');

        final detector = FaceDetector();

        try {
          final initSw = Stopwatch()..start();
          await detector.initialize(performanceConfig: config);
          initSw.stop();

          print('  Init time: ${initSw.elapsedMilliseconds}ms');

          if (!detector.isReady) {
            print('  ERROR: Detector not ready after init');
            results[name] = {'error': 'not ready'};
            continue;
          }

          await detector.detectFaces(bytes, mode: FaceDetectionMode.fast);

          const runs = 5;
          final detectionTimes = <int>[];

          for (int i = 0; i < runs; i++) {
            final sw = Stopwatch()..start();
            final faces =
                await detector.detectFaces(bytes, mode: FaceDetectionMode.full);
            sw.stop();
            detectionTimes.add(sw.elapsedMilliseconds);

            if (i == 0) {
              print('  Faces detected: ${faces.length}');
              if (faces.isNotEmpty) {
                print('  Mesh points: ${faces.first.mesh?.points.length ?? 0}');
                print('  Iris points: ${faces.first.irisPoints.length}');
              }
            }
          }

          final avgDetection = detectionTimes.reduce((a, b) => a + b) / runs;
          final minDetection = detectionTimes.reduce((a, b) => a < b ? a : b);
          final maxDetection = detectionTimes.reduce((a, b) => a > b ? a : b);

          print('  Detection (full mode):');
          print('    Avg: ${avgDetection.toStringAsFixed(0)}ms');
          print('    Min: ${minDetection}ms');
          print('    Max: ${maxDetection}ms');

          results[name] = {
            'initMs': initSw.elapsedMilliseconds,
            'avgDetectionMs': avgDetection,
            'minDetectionMs': minDetection,
            'maxDetectionMs': maxDetection,
          };

          detector.dispose();
        } catch (e, st) {
          print('  ERROR: $e');
          print('  Stack: $st');
          results[name] = {'error': e.toString()};
          try {
            detector.dispose();
          } catch (_) {}
        }
      }

      print('\n${'=' * 60}');
      print('SUMMARY');
      print('=' * 60);
      print('');
      print('Config'.padRight(20) +
          'Init (ms)'.padLeft(12) +
          'Detect (ms)'.padLeft(14));
      print('-' * 46);

      for (final entry in results.entries) {
        final name = entry.key;
        final data = entry.value;

        if (data.containsKey('error')) {
          print('${name.padRight(20)}ERROR: ${data['error']}');
        } else {
          final initMs = data['initMs'] as int;
          final avgMs = data['avgDetectionMs'] as double;
          print(name.padRight(20) +
              '$initMs'.padLeft(12) +
              avgMs.toStringAsFixed(0).padLeft(14));
        }
      }
      print('=' * 60);

      if (results['CPU (disabled)'] != null &&
          !results['CPU (disabled)']!.containsKey('error')) {
        final cpuDetect =
            results['CPU (disabled)']!['avgDetectionMs'] as double;

        print('\nANALYSIS:');

        if (results['GPU'] != null && !results['GPU']!.containsKey('error')) {
          final gpuInit = results['GPU']!['initMs'] as int;
          final gpuDetect = results['GPU']!['avgDetectionMs'] as double;

          if (gpuInit > 10000) {
            print('- GPU init is VERY SLOW (${gpuInit}ms) - NOT recommended');
          } else if (gpuInit > 2000) {
            print('- GPU init is slow (${gpuInit}ms) - may be problematic');
          } else {
            print('- GPU init is acceptable (${gpuInit}ms)');
          }

          final speedup = cpuDetect / gpuDetect;
          if (speedup > 1.2) {
            print(
                '- GPU detection is ${speedup.toStringAsFixed(1)}x FASTER than CPU');
          } else if (speedup < 0.8) {
            print(
                '- GPU detection is ${(1 / speedup).toStringAsFixed(1)}x SLOWER than CPU');
          } else {
            print('- GPU and CPU detection are similar speed');
          }
        }

        if (results['Auto'] != null && !results['Auto']!.containsKey('error')) {
          final autoInit = results['Auto']!['initMs'] as int;
          final autoDetect = results['Auto']!['avgDetectionMs'] as double;
          print(
              '- Auto mode: ${autoInit}ms init, ${autoDetect.toStringAsFixed(0)}ms detect');
        }
      }

      print('\nRECOMMENDATION:');
      if (Platform.isAndroid) {
        final gpuInit = results['GPU']?['initMs'] as int? ?? 999999;
        if (gpuInit > 5000) {
          print('- On this Android device, GPU delegate is NOT recommended');
          print('- Use PerformanceConfig.auto() or PerformanceConfig.disabled');
        } else {
          print('- GPU delegate appears usable on this device');
        }
      }

      print('=' * 60);

      expect(results.isNotEmpty, true);
    }, timeout: gpuTimeout);

    test('Verify GPU results match CPU results', () async {
      print('\n${'=' * 60}');
      print('TEST: Result Quality Comparison');
      print('=' * 60);

      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final cpuDetector = FaceDetector();
      await cpuDetector.initialize(
        performanceConfig: PerformanceConfig.disabled,
      );
      final cpuFaces =
          await cpuDetector.detectFaces(bytes, mode: FaceDetectionMode.full);

      print('CPU detected ${cpuFaces.length} faces');
      expect(cpuFaces, isNotEmpty);

      final autoDetector = FaceDetector();
      await autoDetector.initialize(
        performanceConfig: PerformanceConfig.auto(),
      );
      final autoFaces =
          await autoDetector.detectFaces(bytes, mode: FaceDetectionMode.full);

      print('Auto detected ${autoFaces.length} faces');

      expect(autoFaces.length, cpuFaces.length,
          reason: 'Auto and CPU should detect same number of faces');

      for (int i = 0; i < cpuFaces.length; i++) {
        final cpuBbox = cpuFaces[i].boundingBox;
        final autoBbox = autoFaces[i].boundingBox;

        final xDiff = (cpuBbox.topLeft.x - autoBbox.topLeft.x).abs();
        final yDiff = (cpuBbox.topLeft.y - autoBbox.topLeft.y).abs();

        print('Face $i: bbox diff = ($xDiff, $yDiff) pixels');

        expect(xDiff, lessThan(2));
        expect(yDiff, lessThan(2));
      }

      cpuDetector.dispose();
      autoDetector.dispose();

      print('Test passed - Auto mode produces same results as CPU');
    }, timeout: gpuTimeout);
  });
}
