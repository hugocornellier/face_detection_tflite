part of 'face_core.dart';

class RectF {
  final double xmin, ymin, xmax, ymax;
  const RectF(this.xmin, this.ymin, this.xmax, this.ymax);
  double get w => xmax - xmin;
  double get h => ymax - ymin;
  RectF scale(double sx, double sy) => RectF(xmin * sx, ymin * sy, xmax * sx, ymax * sy);
  RectF expand(double frac) {
    final cx = (xmin + xmax) * 0.5;
    final cy = (ymin + ymax) * 0.5;
    final hw = (w * (1.0 + frac)) * 0.5;
    final hh = (h * (1.0 + frac)) * 0.5;
    return RectF(cx - hw, cy - hh, cx + hw, cy + hh);
  }
}