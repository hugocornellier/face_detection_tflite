// Scratch helper (not committed): fast-mode single-face screening for demo clip
// selection. For every .mp4 in FACE_CLIPS_DIR it samples every Nth frame, runs
// the backCamera detector in fast mode, and prints a one-line score per clip:
//   FILTER <name> frames=<sampled> single=<frac 1-face> mfw=<median face-width
//   fraction> mscore=<median score>
// Used to auto-pick clearly-visible single-face portrait heroes.
//
//   cd example && FACE_CLIPS_DIR=/path/to/clips \
//     flutter test integration_test/gen_face_filter_test.dart -d macos
//
// ignore_for_file: avoid_print
@TestOn('mac-os')
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:face_detection_tflite/face_detection_tflite_native.dart';

double _median(List<double> xs) {
  if (xs.isEmpty) return 0;
  xs.sort();
  final n = xs.length;
  return n.isOdd ? xs[n ~/ 2] : (xs[n ~/ 2 - 1] + xs[n ~/ 2]) / 2;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('screen clips for a single clearly-visible face', (tester) async {
    final clipsDir = Platform.environment['FACE_CLIPS_DIR'] ?? 'assets/samples';
    final step = int.parse(Platform.environment['FACE_STEP'] ?? '4');

    final detector =
        await FaceDetector.create(model: FaceDetectionModel.backCamera);

    final clips = Directory(clipsDir)
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.mp4'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));

    for (final clip in clips) {
      final name = clip.uri.pathSegments.last.replaceAll('.mp4', '');
      final cap = cv.VideoCapture.fromFile(clip.path);
      if (!cap.isOpened) {
        print('FILTER $name OPEN_FAILED');
        continue;
      }
      cv.Mat? frame;
      int idx = 0, sampled = 0, single = 0;
      final widths = <double>[];
      final scores = <double>[];
      while (true) {
        final res = cap.read(m: frame);
        if (!res.$1) break;
        frame = res.$2;
        if (frame.isEmpty) break;
        if (idx % step == 0) {
          final faces = await detector.detectFacesFromMat(frame,
              mode: FaceDetectionMode.fast);
          sampled++;
          if (faces.length == 1) single++;
          if (faces.isNotEmpty) {
            faces.sort((a, b) {
              final aw = a.boundingBox.right - a.boundingBox.left;
              final bw = b.boundingBox.right - b.boundingBox.left;
              return bw.compareTo(aw);
            });
            final f = faces.first;
            widths.add((f.boundingBox.right - f.boundingBox.left) /
                frame.cols.toDouble());
            scores.add(f.score);
          }
        }
        idx++;
      }
      frame?.dispose();
      cap.release();
      final singleFrac = sampled == 0 ? 0.0 : single / sampled;
      print('FILTER $name frames=$sampled '
          'single=${singleFrac.toStringAsFixed(2)} '
          'mfw=${_median(widths).toStringAsFixed(3)} '
          'mscore=${_median(scores).toStringAsFixed(2)}');
    }

    await detector.dispose();
  }, timeout: const Timeout(Duration(minutes: 30)));
}
