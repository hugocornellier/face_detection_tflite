part of 'face_core.dart';

class Detection {
  final RectF bbox;
  final double score;
  final List<double> keypointsXY;
  Detection({required this.bbox, required this.score, required this.keypointsXY});
  double operator [](int i) => keypointsXY[i];
}

class _DecodedBox {
  final RectF bbox;
  final List<double> keypointsXY;
  _DecodedBox(this.bbox, this.keypointsXY);
}