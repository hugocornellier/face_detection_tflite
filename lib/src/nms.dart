part of 'face_core.dart';

double _iou(RectF a, RectF b) {
  final x1 = math.max(a.xmin, b.xmin);
  final y1 = math.max(a.ymin, b.ymin);
  final x2 = math.min(a.xmax, b.xmax);
  final y2 = math.min(a.ymax, b.ymax);
  final iw = math.max(0.0, x2 - x1);
  final ih = math.max(0.0, y2 - y1);
  final inter = iw * ih;
  final areaA = math.max(0.0, a.w) * math.max(0.0, a.h);
  final areaB = math.max(0.0, b.w) * math.max(0.0, b.h);
  final uni = areaA + areaB - inter;
  return uni <= 0 ? 0.0 : inter / uni;
}

List<Detection> _nms(List<Detection> dets, double iouThresh, double scoreThresh, {bool weighted = true}) {
  final kept = <Detection>[];
  final cand = dets.where((d) => d.score >= scoreThresh).toList()
    ..sort((a, b) => b.score.compareTo(a.score));
  while (cand.isNotEmpty) {
    final base = cand.removeAt(0);
    final merged = <Detection>[base];
    cand.removeWhere((d) {
      if (_iou(base.bbox, d.bbox) >= iouThresh) {
        merged.add(d);
        return true;
      }
      return false;
    });
    if (!weighted || merged.length == 1) {
      kept.add(base);
    } else {
      double sw = 0, xmin = 0, ymin = 0, xmax = 0, ymax = 0;
      for (final m in merged) {
        sw += m.score;
        xmin += m.bbox.xmin * m.score;
        ymin += m.bbox.ymin * m.score;
        xmax += m.bbox.xmax * m.score;
        ymax += m.bbox.ymax * m.score;
      }
      kept.add(Detection(
        bbox: RectF(xmin / sw, ymin / sw, xmax / sw, ymax / sw),
        score: base.score,
        keypointsXY: base.keypointsXY,
      ));
    }
  }
  return kept;
}