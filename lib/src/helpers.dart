part of face_detection_tflite;

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
    interpolation: img.Interpolation.linear,
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
      final px = canvas.getPixel(x, y);
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

double _clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);

List<List<double>> _unpackLandmarks(Float32List flat, int inW, int inH, List<double> padding, {bool clamp = true}) {
  final pt = padding[0], pb = padding[1], pl = padding[2], pr = padding[3];
  final sx = 1.0 - (pl + pr);
  final sy = 1.0 - (pt + pb);
  final n = (flat.length / 3).floor();
  final out = <List<double>>[];
  for (var i = 0; i < n; i++) {
    var x = flat[i * 3 + 0] / inW;
    var y = flat[i * 3 + 1] / inH;
    final z = flat[i * 3 + 2];
    x = (x - pl) / sx;
    y = (y - pt) / sy;
    if (clamp) {
      x = _clamp01(x);
      y = _clamp01(y);
    }
    out.add([x, y, z]);
  }
  return out;
}

Offset _centralPoint(List<Offset> pts) {
  if (pts.isEmpty) return const Offset(0, 0);
  var bestIdx = 0;
  var best = double.infinity;
  for (var k = 0; k < pts.length; k++) {
    var s = 0.0;
    for (var j = 0; j < pts.length; j++) {
      if (j == k) continue;
      final dx = pts[j].dx - pts[k].dx;
      final dy = pts[j].dy - pts[k].dy;
      s += dx * dx + dy * dy;
    }
    if (s < best) {
      best = s;
      bestIdx = k;
    }
  }
  return pts[bestIdx];
}

Future<List<Offset>> _computeIrisCenters(IrisLandmark ir, img.Image decoded, List<AlignedRoi> rois, {Offset? leftFallback, Offset? rightFallback}) async {
  final centers = <Offset>[];
  for (var i = 0; i < rois.length; i++) {
    final isRight = (i == 1);
    final raw = await ir.runOnImageAlignedIris(decoded, rois[i], isRight: isRight);
    if (raw.isEmpty) {
      centers.add(isRight ? (rightFallback ?? Offset.zero) : (leftFallback ?? Offset.zero));
      continue;
    }
    final pts = raw.map((p) => Offset(p[0].toDouble(), p[1].toDouble())).toList();
    centers.add(_centralPoint(pts));
  }
  return centers;
}

Detection _mapDetectionToRoi(Detection d, RectF roi) {
  final dx = roi.xmin, dy = roi.ymin, sx = roi.w, sy = roi.h;
  RectF mapRect(RectF r) => RectF(dx + r.xmin * sx, dy + r.ymin * sy, dx + r.xmax * sx, dy + r.ymax * sy);
  List<double> mapKp(List<double> k) {
    final o = List<double>.from(k);
    for (int i = 0; i < o.length; i += 2) {
      o[i] = _clamp01(dx + o[i] * sx);
      o[i + 1] = _clamp01(dy + o[i + 1] * sy);
    }
    return o;
  }
  return Detection(bbox: mapRect(d.bbox), score: d.score, keypointsXY: mapKp(d.keypointsXY), imageSize: d.imageSize);
}

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

Float32List _ssdGenerateAnchors(Map<String, Object> opts) {
  final numLayers = opts['num_layers'] as int;
  final strides = (opts['strides'] as List).cast<int>();
  final inputH = opts['input_size_height'] as int;
  final inputW = opts['input_size_width'] as int;
  final ax = (opts['anchor_offset_x'] as num).toDouble();
  final ay = (opts['anchor_offset_y'] as num).toDouble();
  final interp = (opts['interpolated_scale_aspect_ratio'] as num).toDouble();
  final anchors = <double>[];
  var layerId = 0;
  while (layerId < numLayers) {
    var lastSameStride = layerId;
    var repeats = 0;
    while (lastSameStride < numLayers && strides[lastSameStride] == strides[layerId]) {
      lastSameStride++;
      repeats += (interp == 1.0) ? 2 : 1;
    }
    final stride = strides[layerId];
    final fmH = inputH ~/ stride;
    final fmW = inputW ~/ stride;
    for (var y = 0; y < fmH; y++) {
      final yCenter = (y + ay) / fmH;
      for (var x = 0; x < fmW; x++) {
        final xCenter = (x + ax) / fmW;
        for (var r = 0; r < repeats; r++) {
          anchors.add(xCenter);
          anchors.add(yCenter);
        }
      }
    }
    layerId = lastSameStride;
  }
  return Float32List.fromList(anchors);
}

Map<String, Object> _optsFor(FaceDetectionModel m) {
  switch (m) {
    case FaceDetectionModel.frontCamera:
      return _ssdFront;
    case FaceDetectionModel.backCamera:
      return _ssdBack;
    case FaceDetectionModel.shortRange:
      return _ssdShort;
    case FaceDetectionModel.full:
      return _ssdFull;
    case FaceDetectionModel.fullSparse:
      return _ssdFull;
  }
}

String _nameFor(FaceDetectionModel m) {
  switch (m) {
    case FaceDetectionModel.frontCamera:
      return _modelNameFront;
    case FaceDetectionModel.backCamera:
      return _modelNameBack;
    case FaceDetectionModel.shortRange:
      return _modelNameShort;
    case FaceDetectionModel.full:
      return _modelNameFull;
    case FaceDetectionModel.fullSparse:
      return _modelNameFullSparse;
  }
}

RectF faceDetectionToRoi(RectF bbox, {double expandFraction = 0.6}) {
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

img.Image extractAlignedSquare(img.Image src, double cx, double cy, double size, double theta) {
  final side = math.max(1, size.round());
  final ct = math.cos(theta);
  final st = math.sin(theta);
  final out = img.Image(width: side, height: side);
  for (int y = 0; y < side; y++) {
    final vy = ((y + 0.5) / side - 0.5) * size;
    for (int x = 0; x < side; x++) {
      final vx = ((x + 0.5) / side - 0.5) * size;
      final sx = cx + vx * ct - vy * st;
      final sy = cy + vx * st + vy * ct;
      final px = _bilinearSampleRgb8(src, sx, sy);
      out.setPixel(x, y, px);
    }
  }
  return out;
}

img.ColorRgb8 _bilinearSampleRgb8(img.Image src, double fx, double fy) {
  final x0 = fx.floor();
  final y0 = fy.floor();
  final x1 = x0 + 1;
  final y1 = y0 + 1;
  final ax = fx - x0;
  final ay = fy - y0;

  int cx0 = x0.clamp(0, src.width - 1);
  int cx1 = x1.clamp(0, src.width - 1);
  int cy0 = y0.clamp(0, src.height - 1);
  int cy1 = y1.clamp(0, src.height - 1);

  final p00 = src.getPixel(cx0, cy0);
  final p10 = src.getPixel(cx1, cy0);
  final p01 = src.getPixel(cx0, cy1);
  final p11 = src.getPixel(cx1, cy1);

  final r0 = p00.r * (1 - ax) + p10.r * ax;
  final g0 = p00.g * (1 - ax) + p10.g * ax;
  final b0 = p00.b * (1 - ax) + p10.b * ax;

  final r1 = p01.r * (1 - ax) + p11.r * ax;
  final g1 = p01.g * (1 - ax) + p11.g * ax;
  final b1 = p01.b * (1 - ax) + p11.b * ax;

  final r = (r0 * (1 - ay) + r1 * ay).round().clamp(0, 255);
  final g = (g0 * (1 - ay) + g1 * ay).round().clamp(0, 255);
  final b = (b0 * (1 - ay) + b1 * ay).round().clamp(0, 255);

  return img.ColorRgb8(r, g, b);
}
