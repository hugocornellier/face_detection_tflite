part of 'face_core.dart';

RectF faceDetectionToRoi(RectF bbox, {double expandFraction = 0.5}) {
  final e = bbox.expand(expandFraction);
  final cx = (e.xmin + e.xmax) * 0.5;
  final cy = (e.ymin + e.ymax) * 0.5;
  final s = math.max(e.w, e.h) * 0.5;
  return RectF(cx - s, cy - s, cx + s, cy + s);
}

img.Image cropFromRoi(img.Image src, RectF roi) {
  final w = src.width.toDouble(), h = src.height.toDouble();
  final x0 = (roi.xmin * w).clamp(0.0, w - 1).toInt();
  final y0 = (roi.ymin * h).clamp(0.0, h - 1).toInt();
  final x1 = (roi.xmax * w).clamp(0.0, w.toDouble()).toInt();
  final y1 = (roi.ymax * h).clamp(0.0, h.toDouble()).toInt();
  final cw = math.max(1, x1 - x0);
  final ch = math.max(1, y1 - y0);
  return img.copyCrop(src, x: x0, y: y0, width: cw, height: ch);
}