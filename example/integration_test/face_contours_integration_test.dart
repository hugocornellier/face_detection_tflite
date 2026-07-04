// ignore_for_file: avoid_print

import 'dart:math' show sqrt;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart' as fdt;
import 'package:face_detection_tflite/face_detection_tflite_native.dart'
    show FaceContourType, faceContourMeshIndices;

/// Validates ML Kit-style named contours (`Face.getContour` / `Face.contours`)
/// against a real detected face, checking that each canonical MediaPipe index
/// group lands where its anatomy says it should: the oval encloses the
/// features, eyebrows sit above the eyes, lips are ordered top-to-bottom, the
/// nose bridge is vertical, and the subject-relative left/right sides match the
/// image-relative eye landmarks.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Centroid of a list of contour points.
  fdt.Point centroid(List<fdt.Point> pts) {
    double sx = 0, sy = 0;
    for (final p in pts) {
      sx += p.x;
      sy += p.y;
    }
    return fdt.Point(sx / pts.length, sy / pts.length);
  }

  double meanY(List<fdt.Point> pts) =>
      pts.map((p) => p.y).reduce((a, b) => a + b) / pts.length;

  double dist(fdt.Point a, fdt.Point b) {
    final dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
  }

  group('Face contours - real image geometry', () {
    late fdt.FaceDetector detector;
    late fdt.Face face;

    setUpAll(() async {
      detector = fdt.FaceDetector();
      await detector.initialize();
      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final faces = await detector.detectFacesFromBytes(
        data.buffer.asUint8List(),
        mode: fdt.FaceDetectionMode.full,
      );
      expect(faces, isNotEmpty);
      face = faces.first;
    });

    tearDownAll(() => detector.dispose());

    test('all contours are present, non-empty, and inside the image', () {
      final contours = face.contours;
      expect(contours, isNotNull);
      expect(contours!.keys.toSet(), FaceContourType.values.toSet());

      final w = face.originalSize.width, h = face.originalSize.height;
      for (final type in FaceContourType.values) {
        final pts = face.getContour(type);
        expect(pts, isNotNull, reason: '$type');
        expect(pts, contours[type]); // getContour matches contours (deep eq)
        expect(pts!, isNotEmpty, reason: '$type');
        expect(pts.length, faceContourMeshIndices[type]!.length);
        for (final p in pts) {
          expect(p.x, inInclusiveRange(0, w), reason: '$type x');
          expect(p.y, inInclusiveRange(0, h), reason: '$type y');
        }
      }
    });

    test('the face oval encloses the eyes, nose, and lips', () {
      final oval = face.getContour(FaceContourType.face)!;
      final minX = oval.map((p) => p.x).reduce((a, b) => a < b ? a : b);
      final maxX = oval.map((p) => p.x).reduce((a, b) => a > b ? a : b);
      final minY = oval.map((p) => p.y).reduce((a, b) => a < b ? a : b);
      final maxY = oval.map((p) => p.y).reduce((a, b) => a > b ? a : b);

      for (final type in [
        FaceContourType.leftEye,
        FaceContourType.rightEye,
        FaceContourType.noseBridge,
        FaceContourType.noseBottom,
        FaceContourType.upperLipTop,
        FaceContourType.lowerLipBottom,
        FaceContourType.leftCheek,
        FaceContourType.rightCheek,
      ]) {
        final c = centroid(face.getContour(type)!);
        expect(c.x, inInclusiveRange(minX, maxX),
            reason: '$type inside oval x');
        expect(c.y, inInclusiveRange(minY, maxY),
            reason: '$type inside oval y');
      }
    });

    test('eyebrows sit above the eyes, tops above bottoms', () {
      // Smaller y is higher in image space.
      expect(meanY(face.getContour(FaceContourType.leftEyebrowTop)!),
          lessThan(meanY(face.getContour(FaceContourType.leftEyebrowBottom)!)));
      expect(
          meanY(face.getContour(FaceContourType.rightEyebrowTop)!),
          lessThan(
              meanY(face.getContour(FaceContourType.rightEyebrowBottom)!)));

      expect(meanY(face.getContour(FaceContourType.leftEyebrowBottom)!),
          lessThan(meanY(face.getContour(FaceContourType.leftEye)!)));
      expect(meanY(face.getContour(FaceContourType.rightEyebrowBottom)!),
          lessThan(meanY(face.getContour(FaceContourType.rightEye)!)));
    });

    test('lips are ordered top-to-bottom', () {
      final upperTop = meanY(face.getContour(FaceContourType.upperLipTop)!);
      final upperBottom =
          meanY(face.getContour(FaceContourType.upperLipBottom)!);
      final lowerTop = meanY(face.getContour(FaceContourType.lowerLipTop)!);
      final lowerBottom =
          meanY(face.getContour(FaceContourType.lowerLipBottom)!);

      // The outer arcs bound the mouth; both inner arcs fall strictly between.
      final mouthHeight = lowerBottom - upperTop;
      expect(mouthHeight, greaterThan(0));
      for (final inner in [upperBottom, lowerTop]) {
        expect(inner, greaterThan(upperTop), reason: 'inner below outer-top');
        expect(inner, lessThan(lowerBottom),
            reason: 'inner above outer-bottom');
      }
      // upperLipBottom and lowerLipTop are the single inner mouth line: on a
      // closed mouth they nearly coincide (and may cross by a hair), so assert
      // proximity, not a strict order.
      expect((upperBottom - lowerTop).abs(), lessThan(0.4 * mouthHeight));

      // The whole mouth is below the nose base.
      expect(meanY(face.getContour(FaceContourType.noseBottom)!),
          lessThan(upperTop));
    });

    test('nose bridge runs vertically from between the eyes to the tip', () {
      final bridge = face.getContour(FaceContourType.noseBridge)!;
      // 168 (top, between the eyes) is above 4 (tip).
      expect(bridge.first.y, lessThan(bridge.last.y));
      // Bridge is much taller than it is wide (near-vertical midline).
      final xs = bridge.map((p) => p.x);
      final ys = bridge.map((p) => p.y);
      final xSpread = xs.reduce((a, b) => a > b ? a : b) -
          xs.reduce((a, b) => a < b ? a : b);
      final ySpread = ys.reduce((a, b) => a > b ? a : b) -
          ys.reduce((a, b) => a < b ? a : b);
      expect(ySpread, greaterThan(xSpread));
    });

    test('subject-relative left/right matches the image-relative eye landmarks',
        () {
      // Package landmarks are image-relative: leftEye is the image-left eye.
      final imgLeftEye = face.landmarks.leftEye!;
      final imgRightEye = face.landmarks.rightEye!;

      final subjectLeftEye =
          centroid(face.getContour(FaceContourType.leftEye)!);
      final subjectRightEye =
          centroid(face.getContour(FaceContourType.rightEye)!);

      // Subject's left eye is on the RIGHT of the image (larger x).
      expect(subjectLeftEye.x, greaterThan(subjectRightEye.x));

      // The subject-right eye contour must be nearer the image-left landmark,
      // and the subject-left eye contour nearer the image-right landmark.
      expect(dist(subjectRightEye, imgLeftEye),
          lessThan(dist(subjectRightEye, imgRightEye)));
      expect(dist(subjectLeftEye, imgRightEye),
          lessThan(dist(subjectLeftEye, imgLeftEye)));

      // Cheeks follow the same convention.
      final leftCheek = face.getContour(FaceContourType.leftCheek)!.first;
      final rightCheek = face.getContour(FaceContourType.rightCheek)!.first;
      expect(leftCheek.x, greaterThan(rightCheek.x));
    });

    test('cheeks lie below the eyes and above the mouth', () {
      final leftCheekY = face.getContour(FaceContourType.leftCheek)!.first.y;
      final rightCheekY = face.getContour(FaceContourType.rightCheek)!.first.y;
      final eyeY = meanY(face.getContour(FaceContourType.leftEye)!);
      final mouthY = meanY(face.getContour(FaceContourType.lowerLipBottom)!);
      for (final cy in [leftCheekY, rightCheekY]) {
        expect(cy, greaterThan(eyeY), reason: 'cheek below eye');
        expect(cy, lessThan(mouthY), reason: 'cheek above mouth bottom');
      }
    });
  });

  group('Face contours - mode gating', () {
    late fdt.FaceDetector detector;

    setUpAll(() async {
      detector = fdt.FaceDetector();
      await detector.initialize();
    });

    tearDownAll(() => detector.dispose());

    test('fast mode has no mesh so contours are null', () async {
      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final faces = await detector.detectFacesFromBytes(
        data.buffer.asUint8List(),
        mode: fdt.FaceDetectionMode.fast,
      );
      expect(faces, isNotEmpty);
      expect(faces.first.contours, isNull);
      expect(faces.first.getContour(FaceContourType.face), isNull);
    });

    test('standard mode computes a mesh so contours are available', () async {
      final ByteData data =
          await rootBundle.load('assets/samples/landmark-ex1.jpg');
      final faces = await detector.detectFacesFromBytes(
        data.buffer.asUint8List(),
        mode: fdt.FaceDetectionMode.standard,
      );
      expect(faces, isNotEmpty);
      final contours = faces.first.contours;
      expect(contours, isNotNull);
      expect(contours!.length, FaceContourType.values.length);
    });
  });
}
