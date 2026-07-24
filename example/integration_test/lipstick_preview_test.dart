// ignore_for_file: avoid_print

// Renders the demo lipstick overlay over a sample photo and writes a PNG so
// the result can be inspected by eye. Not an assertion-based test; it exists
// because the geometry tests prove the mask is correct without proving it
// looks like makeup.

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';

import 'package:face_detection_tflite_example/lipstick_painter.dart';

const String kOutDir = String.fromEnvironment('LIPSTICK_OUT', defaultValue: '');

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('render lipstick preview PNGs', () async {
    if (kOutDir.isEmpty) {
      // Opt-in tool, not a suite test: runAllTests.sh globs this directory, so
      // without an explicit output directory this must not run (or fail).
      //   flutter test integration_test/lipstick_preview_test.dart -d macos \
      //     --dart-define=LIPSTICK_OUT=/some/dir
      markTestSkipped('set --dart-define=LIPSTICK_OUT=<dir> to render');
      return;
    }
    final FaceDetector detector = FaceDetector();
    await detector.initialize();
    addTearDown(detector.dispose);

    const String sample = String.fromEnvironment('LIPSTICK_SAMPLE',
        defaultValue: 'assets/samples/landmark-ex1.jpg');
    final ByteData data = await rootBundle.load(sample);
    final Uint8List bytes = data.buffer.asUint8List();

    final List<Face> faces = await detector.detectFacesFromBytes(
      bytes,
      mode: FaceDetectionMode.full,
    );
    expect(faces, isNotEmpty);

    final ui.Codec codec = await ui.instantiateImageCodec(bytes);
    final ui.Image src = (await codec.getNextFrame()).image;
    final ui.Size size = ui.Size(src.width.toDouble(), src.height.toDouble());

    // Crop to the mouth so the written PNG is worth looking at.
    final Path? probe =
        buildLipPath(faces.first, map: (double x, double y) => ui.Offset(x, y));
    expect(probe, isNotNull);
    final ui.Rect lips = probe!.getBounds();
    final ui.Rect crop = ui.Rect.fromCenter(
      center: lips.center,
      width: lips.width * 2.2,
      height: lips.height * 4.0,
    );

    Future<void> render(String name, LipstickPainter? painter) async {
      final ui.PictureRecorder rec = ui.PictureRecorder();
      final Canvas canvas = Canvas(rec);
      canvas.drawImage(src, ui.Offset.zero, Paint());
      painter?.paint(canvas, size);
      final ui.Image out =
          await rec.endRecording().toImage(src.width, src.height);

      // Second pass: crop.
      final ui.PictureRecorder rec2 = ui.PictureRecorder();
      final Canvas c2 = Canvas(rec2);
      c2.drawImageRect(
        out,
        crop,
        ui.Rect.fromLTWH(0, 0, crop.width, crop.height),
        Paint(),
      );
      final ui.Image cropped = await rec2.endRecording().toImage(
            crop.width.round(),
            crop.height.round(),
          );
      final ByteData? png =
          await cropped.toByteData(format: ui.ImageByteFormat.png);
      final String path = '$kOutDir/$name.png';
      await File(path).writeAsBytes(png!.buffer.asUint8List());
      print('WROTE $path');
    }

    await render('lipstick_before', null);
    await render(
      'lipstick_after',
      LipstickPainter(
        faces: faces,
        originalImageSize: size,
        color: const Color(0xFFB3123C),
      ),
    );
    await render(
      'lipstick_flat_srcover',
      LipstickPainter(
        faces: faces,
        originalImageSize: size,
        color: const Color(0xFFB3123C),
        blendMode: BlendMode.srcOver,
      ),
    );
    // Tuned: a desaturated berry rather than a primary, no outward dilation
    // (the smoothing spline already bulges past convex vertices), and a wider
    // feather so the coarse-mesh boundary error is not a visible edge.
    await render(
      'lipstick_tuned',
      LipstickPainter(
        faces: faces,
        originalImageSize: size,
        color: const Color(0xFF8C3A4A),
        strength: 0.75,
        dilatePixels: -1.0,
        featherPixels: 3.0,
        smoothing: 0.6,
      ),
    );
  });
}
