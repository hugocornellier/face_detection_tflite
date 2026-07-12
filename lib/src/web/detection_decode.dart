/// Pure-Dart decode of raw BlazeFace output tensors for the web pipeline.
///
/// Kept free of `dart:js_interop` / `package:web` imports so the decode and
/// score/box pairing logic is unit-testable on the host VM; the web detector
/// (`FaceDetectionModelWeb`) is otherwise only constructible in a browser.
library;

import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart' show sigmoidClipped;

import '../shared/face_model_config.dart' show kMinScore, kRawScoreLimit;
import '../shared/face_types.dart' show RectF;

/// One decoded BlazeFace candidate with its box, keypoints and score kept
/// together.
///
/// Pairing the score with the box in a single record (instead of parallel
/// index-aligned lists) is load-bearing: candidates whose decoded box is
/// degenerate are skipped, and with parallel lists that skip used to shift
/// every later box onto the wrong score, corrupting NMS weighting and the
/// reported `Face.score`.
class DecodedCandidate {
  /// Normalized candidate box in model-input space.
  final RectF box;

  /// Flattened normalized keypoints `[x0, y0, x1, y1, ...]`.
  final List<double> keypointsXY;

  /// Sigmoid-activated confidence for this candidate.
  final double score;

  /// Creates a decoded candidate.
  DecodedCandidate(this.box, this.keypointsXY, this.score);
}

/// Decodes raw BlazeFace score/box tensors into the candidates at or above
/// the internal [kMinScore] floor, skipping degenerate boxes (nonpositive
/// width or height), with each candidate's score correctly attached.
///
/// [scoresRaw] holds one raw logit per anchor; [boxesRaw] holds
/// [valuesPerBox] values per anchor (center x/y, width/height, then keypoint
/// pairs), all divided by [scale] and offset by that anchor's center.
List<DecodedCandidate> decodeBlazeFaceCandidates({
  required Float32List scoresRaw,
  required Float32List boxesRaw,
  required List<List<double>> anchors,
  required int anchorCount,
  required int valuesPerBox,
  required double scale,
}) {
  final List<DecodedCandidate> out = <DecodedCandidate>[];
  final Float32List tmp = Float32List(valuesPerBox);
  for (int i = 0; i < anchorCount; i++) {
    final double s = sigmoidClipped(scoresRaw[i], limit: kRawScoreLimit);
    // Keep the old `s >= kMinScore` acceptance semantics exactly: NaN makes
    // both ordered comparisons false and must be rejected, not sent into NMS.
    if (!(s >= kMinScore)) continue;

    final int base = i * valuesPerBox;
    for (int j = 0; j < valuesPerBox; j++) {
      tmp[j] = boxesRaw[base + j] / scale;
    }
    final double ax = anchors[i][0];
    final double ay = anchors[i][1];
    tmp[0] += ax;
    tmp[1] += ay;
    for (int j = 4; j < valuesPerBox; j += 2) {
      tmp[j] += ax;
      tmp[j + 1] += ay;
    }
    final double xc = tmp[0], yc = tmp[1], w = tmp[2], h = tmp[3];
    if (w <= 0 || h <= 0) continue;
    final List<double> kp = <double>[];
    for (int j = 4; j < valuesPerBox; j += 2) {
      kp.add(tmp[j]);
      kp.add(tmp[j + 1]);
    }
    out.add(
      DecodedCandidate(
        RectF(xc - w * 0.5, yc - h * 0.5, xc + w * 0.5, yc + h * 0.5),
        kp,
        s,
      ),
    );
  }
  return out;
}
