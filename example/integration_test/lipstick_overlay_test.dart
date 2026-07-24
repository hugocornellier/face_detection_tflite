// ignore_for_file: avoid_print

import 'dart:ui' show Offset, Path, Rect, Size;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';
import 'package:flutter_litert/flutter_litert.dart' show CoverFitTransform;

import 'package:face_detection_tflite_example/lipstick_painter.dart';

/// Verifies the demo lipstick mask (`example/lib/lipstick_painter.dart`) as
/// geometry rather than as pixels: the filled region must cover lip flesh,
/// exclude the mouth opening, and exclude surrounding skin.
///
/// Path containment is used instead of rasterizing so the test says something
/// about the mask itself rather than about the blend mode or the renderer.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Lipstick mask geometry', () {
    late FaceDetector detector;
    late Face face;

    setUpAll(() async {
      detector = FaceDetector();
      await detector.initialize();
      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final faces = await detector.detectFacesFromBytes(
        data.buffer.asUint8List(),
        mode: FaceDetectionMode.full,
      );
      expect(faces, isNotEmpty);
      face = faces.first;
    });

    tearDownAll(() => detector.dispose());

    // Identity transform so path coordinates are source-image pixels.
    Path? build({double minOpenFraction = 0.06, double dilate = 0}) =>
        buildLipPath(
          face,
          map: (double x, double y) => Offset(x, y),
          dilatePixels: dilate,
          minOpenFraction: minOpenFraction,
        );

    Offset mid(Point a, Point b) => Offset((a.x + b.x) / 2, (a.y + b.y) / 2);

    // Contour index 5 is the mid-line vertex of each lip arc: mesh 0 (upper
    // lip top), 13 (top of the mouth opening), 14 (bottom of the opening) and
    // 17 (lower lip bottom).
    Point arcMid(FaceContourType t) => face.getContour(t)![5];

    test('covers upper and lower lip flesh', () {
      final Path path = build()!;
      final Offset upperFlesh = mid(
        arcMid(FaceContourType.upperLipTop),
        arcMid(FaceContourType.upperLipBottom),
      );
      final Offset lowerFlesh = mid(
        arcMid(FaceContourType.lowerLipTop),
        arcMid(FaceContourType.lowerLipBottom),
      );
      expect(path.contains(upperFlesh), isTrue,
          reason: 'upper lip flesh must be painted');
      expect(path.contains(lowerFlesh), isTrue,
          reason: 'lower lip flesh must be painted');
    });

    test('excludes the surrounding skin', () {
      final Path path = build()!;
      final outer = path.getBounds();
      // Sample well outside the mouth on all four sides.
      for (final Offset p in <Offset>[
        Offset(outer.left - outer.width * 0.3, outer.center.dy),
        Offset(outer.right + outer.width * 0.3, outer.center.dy),
        Offset(outer.center.dx, outer.top - outer.height * 0.8),
        Offset(outer.center.dx, outer.bottom + outer.height * 0.8),
      ]) {
        expect(path.contains(p), isFalse, reason: 'skin at $p must be clean');
      }
    });

    test('the open-fraction gate controls whether the mouth is cut out', () {
      // The reference face measures ~0.107 (see the contour geometry test), so
      // a threshold below that cuts the hole and one above it suppresses it.
      final Offset opening = mid(
        arcMid(FaceContourType.upperLipBottom),
        arcMid(FaceContourType.lowerLipTop),
      );

      final Path cut = build(minOpenFraction: 0.05)!;
      final Path filled = build(minOpenFraction: 0.5)!;
      print('LIPSTICK opening=$opening '
          'holeCut=${!cut.contains(opening)} '
          'holeSuppressed=${filled.contains(opening)}');

      expect(cut.contains(opening), isFalse,
          reason: 'below threshold the mouth opening must be cut out');
      expect(filled.contains(opening), isTrue,
          reason: 'above threshold the mouth fills solid (closed-mouth case)');
    });

    test('dilation grows the region without moving its centre', () {
      final Rect tight = build(dilate: 0)!.getBounds();
      final Rect grown = build(dilate: 3)!.getBounds();
      print('LIPSTICK tight=$tight grown=$grown');
      expect(grown.width, greaterThan(tight.width));
      expect(grown.height, greaterThan(tight.height));
      expect((grown.center - tight.center).distance,
          lessThan(tight.shortestSide * 0.25),
          reason: 'dilation must not translate the mask');
    });

    test('live cover-fit mapping places the mask, and mirroring reflects it',
        () {
      // The live-camera clipper maps through CoverFitTransform rather than a
      // plain scale, and gets it wrong invisibly if mirroring is mishandled.
      const Size view = Size(800, 600);
      final Size img = face.originalSize;

      Path build({required bool mirror}) {
        final t = CoverFitTransform.cover(
          sourceWidth: img.width,
          sourceHeight: img.height,
          viewWidth: view.width,
          viewHeight: view.height,
          mirror: mirror,
        );
        return buildAllLipPaths(<Face>[face], map: t.map)!;
      }

      final Rect plain = build(mirror: false).getBounds();
      final Rect mirrored = build(mirror: true).getBounds();
      print('LIVE MAP plain=$plain mirrored=$mirrored view=$view');

      // Same shape, reflected about the view's vertical centre line.
      expect(mirrored.width, closeTo(plain.width, 0.5));
      expect(mirrored.height, closeTo(plain.height, 0.5));
      expect(mirrored.top, closeTo(plain.top, 0.5));
      expect(mirrored.center.dx + plain.center.dx, closeTo(view.width, 1.0),
          reason: 'mirrored mask must reflect about the view centre');
    });

    test('returns null in fast mode (no mesh, so no contours)', () async {
      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final faces = await detector.detectFacesFromBytes(
        data.buffer.asUint8List(),
        mode: FaceDetectionMode.fast,
      );
      expect(faces, isNotEmpty);
      expect(
        buildLipPath(faces.first, map: (double x, double y) => Offset(x, y)),
        isNull,
      );
    });
  });
}
