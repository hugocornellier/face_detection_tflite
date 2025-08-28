part of 'face_core.dart';

class ImageTensor {
  final Float32List tensorNHWC;
  final List<double> padding;
  final int width, height;
  ImageTensor(this.tensorNHWC, this.padding, this.width, this.height);
}

double _clip(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

double _sigmoidClipped(double x, {double limit = _rawScoreLimit}) {
  final v = _clip(x, -limit, limit);
  return 1.0 / (1.0 + math.exp(-v));
}

ImageTensor _imageToTensor(img.Image src, {required int outW, required int outH}) {
  final inW = src.width, inH = src.height;
  final scale = (outW / inW < outH / inH) ? outW / inW : outH / inH;
  final newW = (inW * scale).round();
  final newH = (inH * scale).round();

  final resized = img.copyResize(
    src,
    width: newW,
    height: newH,
    interpolation: img.Interpolation.average,
  );

  final dx = (outW - newW) ~/ 2;
  final dy = (outH - newH) ~/ 2;

  final canvas = img.Image(width: outW, height: outH);
  img.fill(canvas, color: img.ColorRgb8(0, 0, 0));

  for (var y = 0; y < resized.height; y++) {
    for (var x = 0; x < resized.width; x++) {
      final px = resized.getPixel(x, y);
      canvas.setPixel(x + dx, y + dy, px);
    }
  }

  final t = Float32List(outW * outH * 3);
  var k = 0;
  for (var y = 0; y < outH; y++) {
    for (var x = 0; x < outW; x++) {
      final px = canvas.getPixel(x, y) as img.Pixel;
      t[k++] = (px.r / 127.5) - 1.0;
      t[k++] = (px.g / 127.5) - 1.0;
      t[k++] = (px.b / 127.5) - 1.0;
    }
  }

  final padTop = dy / outH;
  final padBottom = (outH - dy - newH) / outH;
  final padLeft = dx / outW;
  final padRight = (outW - dx - newW) / outW;

  return ImageTensor(t, [padTop, padBottom, padLeft, padRight], outW, outH);
}

List<Detection> _detectionLetterboxRemoval(List<Detection> dets, List<double> padding) {
  final pt = padding[0], pb = padding[1], pl = padding[2], pr = padding[3];
  final sx = 1.0 - (pl + pr);
  final sy = 1.0 - (pt + pb);

  RectF unpad(RectF r) => RectF((r.xmin - pl) / sx, (r.ymin - pt) / sy, (r.xmax - pl) / sx, (r.ymax - pt) / sy);

  List<double> unpadKp(List<double> kps) {
    final out = List<double>.from(kps);
    for (var i = 0; i < out.length; i += 2) {
      out[i] = (out[i] - pl) / sx;
      out[i + 1] = (out[i + 1] - pt) / sy;
    }
    return out;
  }

  return dets
      .map((d) => Detection(bbox: unpad(d.bbox), score: d.score, keypointsXY: unpadKp(d.keypointsXY)))
      .toList();
}