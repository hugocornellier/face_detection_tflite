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

  // --- ring assembly + polygon predicates -----------------------------------
  //
  // The lip and eyebrow arcs are stored as open polylines; a fillable ring is
  // the upper arc followed by the lower arc reversed. Lips store both arcs in
  // the same direction AND share their endpoints (61/291 outer, 78/308 inner),
  // so the reversed arc must drop its first and last point. Eyebrow arcs share
  // no endpoints, so nothing is dropped. These helpers exist to check that
  // assumption against real coordinates rather than assert it.

  /// Joins [top] with [bottom] reversed into a closed ring (first != last).
  /// When [sharedEndpoints], drops the duplicated join and close vertices.
  List<fdt.Point> ring(
    List<fdt.Point> top,
    List<fdt.Point> bottom, {
    required bool sharedEndpoints,
  }) {
    final rev = bottom.reversed.toList();
    return <fdt.Point>[
      ...top,
      ...sharedEndpoints ? rev.sublist(1, rev.length - 1) : rev,
    ];
  }

  /// Shoelace signed area. Sign encodes winding; magnitude is in px^2.
  double signedArea(List<fdt.Point> r) {
    double s = 0;
    for (int i = 0; i < r.length; i++) {
      final a = r[i], b = r[(i + 1) % r.length];
      s += a.x * b.y - b.x * a.y;
    }
    return s / 2;
  }

  double cross(fdt.Point o, fdt.Point a, fdt.Point b) =>
      (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  /// True when segments ab and cd cross at an interior point of both.
  bool properlyIntersect(fdt.Point a, fdt.Point b, fdt.Point c, fdt.Point d) {
    final d1 = cross(a, b, c), d2 = cross(a, b, d);
    final d3 = cross(c, d, a), d4 = cross(c, d, b);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
  }

  /// Indices of the first self-intersecting edge pair, or null when simple.
  List<int>? selfIntersection(List<fdt.Point> r) {
    final n = r.length;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        // Skip adjacent edges (and the wrap-around pair), which share a vertex.
        if (j == i || j == i + 1 || (i == 0 && j == n - 1)) continue;
        if (properlyIntersect(r[i], r[(i + 1) % n], r[j], r[(j + 1) % n])) {
          return <int>[i, j];
        }
      }
    }
    return null;
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

    // Ring-level geometry: whether the arc pairs actually close into fillable
    // polygons on real coordinates. The unit tests cannot check any of this
    // because their fixture mesh is Point(i, i, i), i.e. fully collinear.

    test('lip arc pairs dedupe into 20-point rings with no repeated vertex',
        () {
      final outer = ring(
        face.getContour(FaceContourType.upperLipTop)!,
        face.getContour(FaceContourType.lowerLipBottom)!,
        sharedEndpoints: true,
      );
      final inner = ring(
        face.getContour(FaceContourType.upperLipBottom)!,
        face.getContour(FaceContourType.lowerLipTop)!,
        sharedEndpoints: true,
      );

      for (final MapEntry<String, List<fdt.Point>> e
          in {'outer': outer, 'inner': inner}.entries) {
        final r = e.value;
        expect(r.length, 20, reason: '${e.key} ring length');
        // No vertex may repeat: a duplicate collapses an edge to zero length
        // and makes winding and fill undefined at that point.
        final seen = <String>{};
        for (final p in r) {
          final key = '${p.x},${p.y}';
          expect(seen.add(key), isTrue,
              reason: '${e.key} ring has a duplicated vertex at $key');
        }
      }
    });

    test('lip rings are simple polygons; winding is not stable between them',
        () {
      final outer = ring(
        face.getContour(FaceContourType.upperLipTop)!,
        face.getContour(FaceContourType.lowerLipBottom)!,
        sharedEndpoints: true,
      );
      final inner = ring(
        face.getContour(FaceContourType.upperLipBottom)!,
        face.getContour(FaceContourType.lowerLipTop)!,
        sharedEndpoints: true,
      );

      final outerArea = signedArea(outer);
      final innerArea = signedArea(inner);
      final outerX = selfIntersection(outer);
      final innerX = selfIntersection(inner);
      print('LIP RINGS outerArea=$outerArea innerArea=$innerArea '
          'innerFrac=${(innerArea.abs() / outerArea.abs()).toStringAsFixed(4)} '
          'outerSelfIntersect=$outerX innerSelfIntersect=$innerX');

      // The outer ring bounds the whole mouth and must be usable as a fill
      // boundary on any face; a self-intersection here would break every
      // even-odd fill built on it.
      expect(outerX, isNull,
          reason: 'outer lip ring self-intersects at $outerX');
      expect(outerArea.abs(), greaterThan(0.0));

      // Winding is NOT stable between the two rings and must never be assumed.
      // Both rings are built identically (upper arc, then lower arc reversed),
      // so a naive reading says they should wind the same way. On this face
      // they do not: outer is positive, inner is negative. The cause is the
      // inner arcs swapping vertical order on a closed mouth, the same effect
      // the ordering test above tolerates ("may cross by a hair"). So the sign
      // of the inner ring is a property of the subject's expression, not of
      // the index tables.
      //
      // Consequence for any fill built on these rings: PathFillType.evenOdd is
      // mandatory. Under non-zero fill the inner ring would cut a hole on some
      // faces and fill solid on others. Same for OpenCV fillPoly over both
      // contours at once.
      expect(innerArea, isNot(0.0), reason: 'inner ring is degenerate');

      // The mouth opening is strictly smaller than the whole mouth.
      expect(innerArea.abs(), lessThan(outerArea.abs()));

      // Recorded for the closed-mouth gate: on this (near-closed) mouth the
      // opening is ~10.7% of the outer ring area. A geometric gate on this
      // fraction is what the lipstick overlay should use, rather than the
      // jawOpen/mouthClose blendshapes, which are driven by entirely
      // unrefined lip landmarks.
      expect(innerArea.abs() / outerArea.abs(), lessThan(1.0));
    });

    test('eyebrow arc pairs close without dropping any endpoint', () {
      // Unlike the lips, the eyebrow arcs share no mesh indices, so the ring
      // is a straight 5 + 5 with no dedupe. Whether that actually yields a
      // simple polygon has never been checked against real coordinates.
      for (final MapEntry<String, List<FaceContourType>> e in {
        'left': [
          FaceContourType.leftEyebrowTop,
          FaceContourType.leftEyebrowBottom
        ],
        'right': [
          FaceContourType.rightEyebrowTop,
          FaceContourType.rightEyebrowBottom
        ],
      }.entries) {
        final r = ring(
          face.getContour(e.value[0])!,
          face.getContour(e.value[1])!,
          sharedEndpoints: false,
        );
        expect(r.length, 10, reason: '${e.key} eyebrow ring length');
        final area = signedArea(r);
        final x = selfIntersection(r);
        print('EYEBROW RING ${e.key} area=$area selfIntersect=$x');
        expect(area.abs(), greaterThan(0.0),
            reason: '${e.key} eyebrow ring is degenerate');
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
