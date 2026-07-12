import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart' show sigmoidClipped;
import 'package:face_detection_tflite/src/shared/face_model_config.dart'
    show kMinScore, kRawScoreLimit;
import 'package:face_detection_tflite/src/shared/face_types.dart' show RectF;
import 'package:face_detection_tflite/src/web/detection_decode.dart';

/// Unit tests for the web pipeline's pure BlazeFace decode.
///
/// Two properties are pinned:
/// 1. On inputs with no degenerate boxes, the output is value-identical to
///    the previous inline implementation (replicated below as a reference).
/// 2. When a candidate's decoded box is degenerate and skipped, every
///    remaining candidate keeps its OWN score. The previous parallel-array
///    implementation shifted the score lists relative to the boxes in that
///    case, pairing later boxes with earlier candidates' scores.
void main() {
  const int k = 16; // 4 box values + 6 keypoints x 2

  /// Builds a boxes tensor row: center x/y, w/h, then 6 keypoint pairs,
  /// all in model-input pixel units (pre scale-division).
  List<double> boxRow({
    required double xc,
    required double yc,
    required double w,
    required double h,
  }) => [
    xc,
    yc,
    w,
    h,
    for (int j = 0; j < 6; j++) ...[xc + j, yc - j],
  ];

  /// The previous inline implementation (parallel index-aligned lists),
  /// reproduced verbatim as a reference for the no-degenerate-box case.
  ({List<RectF> boxes, List<double> scores, List<List<double>> kps})
  referenceOldDecode({
    required Float32List scoresRaw,
    required Float32List boxesRaw,
    required List<List<double>> anchors,
    required int n,
    required double scale,
  }) {
    final List<int> candIndices = <int>[];
    final List<double> candScores = <double>[];
    for (int i = 0; i < n; i++) {
      final double s = sigmoidClipped(scoresRaw[i], limit: kRawScoreLimit);
      if (s >= kMinScore) {
        candIndices.add(i);
        candScores.add(s);
      }
    }
    final List<RectF> boxes = <RectF>[];
    final List<List<double>> kps = <List<double>>[];
    final Float32List tmp = Float32List(k);
    for (final int i in candIndices) {
      final int base = i * k;
      for (int j = 0; j < k; j++) {
        tmp[j] = boxesRaw[base + j] / scale;
      }
      final double ax = anchors[i][0];
      final double ay = anchors[i][1];
      tmp[0] += ax;
      tmp[1] += ay;
      for (int j = 4; j < k; j += 2) {
        tmp[j] += ax;
        tmp[j + 1] += ay;
      }
      final double xc = tmp[0], yc = tmp[1], w = tmp[2], h = tmp[3];
      if (w <= 0 || h <= 0) continue;
      boxes.add(RectF(xc - w * 0.5, yc - h * 0.5, xc + w * 0.5, yc + h * 0.5));
      kps.add([
        for (int j = 4; j < k; j += 2) ...[tmp[j], tmp[j + 1]],
      ]);
    }
    // Note: candScores is intentionally NOT compacted on skip; that was the
    // bug. Callers below only use this reference when nothing is skipped.
    return (boxes: boxes, scores: candScores, kps: kps);
  }

  group('decodeBlazeFaceCandidates', () {
    final anchors = [
      for (int i = 0; i < 4; i++) [0.1 * (i + 1), 0.2 * (i + 1)],
    ];

    test('matches the previous implementation when no box is degenerate', () {
      const double scale = 128.0;
      // Anchor 2 falls below the 0.5 score floor; the rest pass.
      final scoresRaw = Float32List.fromList([2.0, 3.0, -5.0, 2.5]);
      final boxesRaw = Float32List.fromList([
        ...boxRow(xc: 10, yc: 12, w: 30, h: 32),
        ...boxRow(xc: 20, yc: 22, w: 40, h: 42),
        ...boxRow(xc: 30, yc: 32, w: 50, h: 52),
        ...boxRow(xc: 40, yc: 42, w: 60, h: 62),
      ]);

      final decoded = decodeBlazeFaceCandidates(
        scoresRaw: scoresRaw,
        boxesRaw: boxesRaw,
        anchors: anchors,
        anchorCount: 4,
        valuesPerBox: k,
        scale: scale,
      );
      final ref = referenceOldDecode(
        scoresRaw: scoresRaw,
        boxesRaw: boxesRaw,
        anchors: anchors,
        n: 4,
        scale: scale,
      );

      expect(decoded.length, ref.boxes.length);
      for (int i = 0; i < decoded.length; i++) {
        expect(decoded[i].box.xmin, ref.boxes[i].xmin);
        expect(decoded[i].box.ymin, ref.boxes[i].ymin);
        expect(decoded[i].box.xmax, ref.boxes[i].xmax);
        expect(decoded[i].box.ymax, ref.boxes[i].ymax);
        expect(decoded[i].score, ref.scores[i]);
        expect(decoded[i].keypointsXY, ref.kps[i]);
      }
    });

    test('keeps each box paired with its own score across a skipped box', () {
      const double scale = 128.0;
      // Three candidates pass the score floor; the middle one decodes to a
      // zero-width box and must be skipped without shifting scores.
      final scoresRaw = Float32List.fromList([2.0, 3.0, 2.5, -5.0]);
      final boxesRaw = Float32List.fromList([
        ...boxRow(xc: 10, yc: 12, w: 30, h: 32),
        ...boxRow(xc: 20, yc: 22, w: 0, h: 42), // degenerate: w == 0
        ...boxRow(xc: 30, yc: 32, w: 50, h: 52),
        ...boxRow(xc: 40, yc: 42, w: 60, h: 62),
      ]);

      final decoded = decodeBlazeFaceCandidates(
        scoresRaw: scoresRaw,
        boxesRaw: boxesRaw,
        anchors: anchors,
        anchorCount: 4,
        valuesPerBox: k,
        scale: scale,
      );

      expect(decoded, hasLength(2));
      final s0 = sigmoidClipped(2.0, limit: kRawScoreLimit);
      final s2 = sigmoidClipped(2.5, limit: kRawScoreLimit);
      final sSkipped = sigmoidClipped(3.0, limit: kRawScoreLimit);

      expect(decoded[0].score, s0);
      // The surviving second candidate must carry the anchor-2 score, not the
      // skipped anchor-1 score the old parallel-list pairing would give it.
      expect(decoded[1].score, s2);
      expect(decoded[1].score, isNot(sSkipped));
      // And its box really is the anchor-2 box (center 30/128 + anchor x).
      // Tolerance sized for float32: the pipeline's intermediate buffer is a
      // Float32List, so expectations computed in doubles differ by ~1e-8.
      expect(
        decoded[1].box.xmin,
        closeTo(30 / scale + anchors[2][0] - 50 / scale / 2, 1e-6),
      );
    });

    test('all candidates below the floor yields an empty list', () {
      final decoded = decodeBlazeFaceCandidates(
        scoresRaw: Float32List.fromList([-5.0, -4.0, -3.0, -6.0]),
        boxesRaw: Float32List(4 * k),
        anchors: anchors,
        anchorCount: 4,
        valuesPerBox: k,
        scale: 128.0,
      );
      expect(decoded, isEmpty);
    });

    test('NaN scores are rejected instead of reaching NMS', () {
      final decoded = decodeBlazeFaceCandidates(
        scoresRaw: Float32List.fromList([double.nan, -5.0, -5.0, -5.0]),
        boxesRaw: Float32List.fromList([
          ...boxRow(xc: 10, yc: 12, w: 30, h: 32),
          ...boxRow(xc: 20, yc: 22, w: 40, h: 42),
          ...boxRow(xc: 30, yc: 32, w: 50, h: 52),
          ...boxRow(xc: 40, yc: 42, w: 60, h: 62),
        ]),
        anchors: anchors,
        anchorCount: 4,
        valuesPerBox: k,
        scale: 128.0,
      );
      expect(decoded, isEmpty);
    });
  });
}
