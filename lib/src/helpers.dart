part of face_detection_tflite;

double _clip(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

double _sigmoidClipped(double x, {double limit = _rawScoreLimit}) {
  final v = _clip(x, -limit, limit);
  return 1.0 / (1.0 + math.exp(-v));
}

Future<ImageTensor> _imageToTensor(img.Image src,
    {required int outW, required int outH}) async {
  final rp = ReceivePort();
  final rgb = src.getBytes(order: img.ChannelOrder.rgb);
  final params = {
    'sendPort': rp.sendPort,
    'inW': src.width,
    'inH': src.height,
    'outW': outW,
    'outH': outH,
    'rgb': TransferableTypedData.fromList([rgb]),
  };
  await Isolate.spawn(_imageToTensorIsolate, params);
  final Map msg = await rp.first as Map;
  rp.close();

  final ByteBuffer tBB = (msg['tensor'] as TransferableTypedData).materialize();
  final Float32List tensor = tBB.asUint8List().buffer.asFloat32List();

  final List paddingRaw = msg['padding'] as List;
  final List<double> padding =
  paddingRaw.map((e) => (e as num).toDouble()).toList();
  final int ow = msg['outW'] as int;
  final int oh = msg['outH'] as int;

  return ImageTensor(tensor, padding, ow, oh);
}

@pragma('vm:entry-point')
Future<void> _imageToTensorIsolate(Map<String, dynamic> params) async {
  final SendPort sp = params['sendPort'] as SendPort;
  final int inW = params['inW'] as int;
  final int inH = params['inH'] as int;
  final int outW = params['outW'] as int;
  final int outH = params['outH'] as int;
  final ByteBuffer rgbBB = (params['rgb'] as TransferableTypedData).materialize();
  final Uint8List rgb = rgbBB.asUint8List();

  final src = img.Image.fromBytes(
    width: inW,
    height: inH,
    bytes: rgb.buffer,
    order: img.ChannelOrder.rgb,
  );

  final double s1 = outW / inW;
  final double s2 = outH / inH;
  final double scale = s1 < s2 ? s1 : s2;
  final int newW = (inW * scale).round();
  final int newH = (inH * scale).round();

  final resized = img.copyResize(
    src,
    width: newW,
    height: newH,
    interpolation: img.Interpolation.linear,
  );

  final int dx = (outW - newW) ~/ 2;
  final int dy = (outH - newH) ~/ 2;

  final canvas = img.Image(width: outW, height: outH);
  img.fill(canvas, color: img.ColorRgb8(0, 0, 0));

  for (int y = 0; y < resized.height; y++) {
    for (int x = 0; x < resized.width; x++) {
      final px = resized.getPixel(x, y);
      canvas.setPixel(x + dx, y + dy, px);
    }
  }

  final t = Float32List(outW * outH * 3);
  int k = 0;
  for (int y = 0; y < outH; y++) {
    for (int x = 0; x < outW; x++) {
      final px = canvas.getPixel(x, y);
      t[k++] = (px.r / 127.5) - 1.0;
      t[k++] = (px.g / 127.5) - 1.0;
      t[k++] = (px.b / 127.5) - 1.0;
    }
  }

  final double padTop = dy / outH;
  final double padBottom = (outH - dy - newH) / outH;
  final double padLeft = dx / outW;
  final double padRight = (outW - dx - newW) / outW;

  sp.send({
    'tensor': TransferableTypedData.fromList([t.buffer.asUint8List()]),
    'padding': [padTop, padBottom, padLeft, padRight],
    'outW': outW,
    'outH': outH,
  });
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

Future<img.Image> cropFromRoi(img.Image src, RectF roi) async {
  final rgb = src.getBytes(order: img.ChannelOrder.rgb);
  final rp = ReceivePort();
  final params = {
    'sendPort': rp.sendPort,
    'op': 'crop',
    'w': src.width,
    'h': src.height,
    'rgb': TransferableTypedData.fromList([rgb]),
    'roi': {
      'xmin': roi.xmin,
      'ymin': roi.ymin,
      'xmax': roi.xmax,
      'ymax': roi.ymax,
    },
  };
  await Isolate.spawn(_imageTransformIsolate, params);
  final Map msg = await rp.first as Map;
  rp.close();
  final ByteBuffer outBB = (msg['rgb'] as TransferableTypedData).materialize();
  final Uint8List outRgb = outBB.asUint8List();
  final int ow = msg['w'] as int;
  final int oh = msg['h'] as int;
  return img.Image.fromBytes(width: ow, height: oh, bytes: outRgb.buffer, order: img.ChannelOrder.rgb);
}

Future<img.Image> extractAlignedSquare(img.Image src, double cx, double cy, double size, double theta) async {
  final rgb = src.getBytes(order: img.ChannelOrder.rgb);
  final rp = ReceivePort();
  final params = {
    'sendPort': rp.sendPort,
    'op': 'extract',
    'w': src.width,
    'h': src.height,
    'rgb': TransferableTypedData.fromList([rgb]),
    'cx': cx,
    'cy': cy,
    'size': size,
    'theta': theta,
  };
  await Isolate.spawn(_imageTransformIsolate, params);
  final Map msg = await rp.first as Map;
  rp.close();
  final ByteBuffer outBB = (msg['rgb'] as TransferableTypedData).materialize();
  final Uint8List outRgb = outBB.asUint8List();
  final int ow = msg['w'] as int;
  final int oh = msg['h'] as int;
  return img.Image.fromBytes(width: ow, height: oh, bytes: outRgb.buffer, order: img.ChannelOrder.rgb);
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

class _DecodedRgb {
  final int width;
  final int height;
  final Uint8List rgb; // RGB, 3 bytes per pixel
  const _DecodedRgb(this.width, this.height, this.rgb);
}

Future<_DecodedRgb> _decodeImageOffUi(Uint8List bytes) async {
  final rp = ReceivePort();
  final params = {
    'sendPort': rp.sendPort,
    'bytes': TransferableTypedData.fromList([bytes]),
  };
  await Isolate.spawn(_decodeImageIsolate, params);
  final Map msg = await rp.first as Map;
  rp.close();

  if (msg['ok'] != true) {
    throw const FormatException('Could not decode image bytes (unsupported or corrupt).');
  }

  final ByteBuffer rgbBB = (msg['rgb'] as TransferableTypedData).materialize();
  final Uint8List rgb = rgbBB.asUint8List();
  final int w = msg['w'] as int;
  final int h = msg['h'] as int;
  return _DecodedRgb(w, h, rgb);
}

img.Image _imageFromDecodedRgb(_DecodedRgb d) {
  return img.Image.fromBytes(
    width: d.width,
    height: d.height,
    bytes: d.rgb.buffer,
    order: img.ChannelOrder.rgb,
  );
}

@pragma('vm:entry-point')
Future<void> _decodeImageIsolate(Map<String, dynamic> params) async {
  final SendPort sp = params['sendPort'] as SendPort;
  final ByteBuffer bb = (params['bytes'] as TransferableTypedData).materialize();
  final Uint8List inBytes = bb.asUint8List();

  final img.Image? decoded = img.decodeImage(inBytes);
  if (decoded == null) {
    sp.send({'ok': false});
    return;
  }

  final Uint8List rgb = decoded.getBytes(order: img.ChannelOrder.rgb);
  sp.send({
    'ok': true,
    'w': decoded.width,
    'h': decoded.height,
    'rgb': TransferableTypedData.fromList([rgb]),
  });
}

@pragma('vm:entry-point')
Future<void> _imageTransformIsolate(Map<String, dynamic> params) async {
  final SendPort sp = params['sendPort'] as SendPort;
  final String op = params['op'] as String;
  final int w = params['w'] as int;
  final int h = params['h'] as int;
  final ByteBuffer inBB = (params['rgb'] as TransferableTypedData).materialize();
  final Uint8List inRgb = inBB.asUint8List();

  final src = img.Image.fromBytes(width: w, height: h, bytes: inRgb.buffer, order: img.ChannelOrder.rgb);

  img.Image out;

  if (op == 'crop') {
    final m = params['roi'] as Map;
    final xmin = (m['xmin'] as num).toDouble();
    final ymin = (m['ymin'] as num).toDouble();
    final xmax = (m['xmax'] as num).toDouble();
    final ymax = (m['ymax'] as num).toDouble();

    final W = src.width.toDouble(), H = src.height.toDouble();
    final x0 = (xmin * W).clamp(0.0, W - 1).toInt();
    final y0 = (ymin * H).clamp(0.0, H - 1).toInt();
    final x1 = (xmax * W).clamp(0.0, W).toInt();
    final y1 = (ymax * H).clamp(0.0, H).toInt();
    final cw = math.max(1, x1 - x0);
    final ch = math.max(1, y1 - y0);
    out = img.copyCrop(src, x: x0, y: y0, width: cw, height: ch);
  } else if (op == 'extract') {
    final cx = (params['cx'] as num).toDouble();
    final cy = (params['cy'] as num).toDouble();
    final size = (params['size'] as num).toDouble();
    final theta = (params['theta'] as num).toDouble();

    final side = math.max(1, size.round());
    final ct = math.cos(theta);
    final st = math.sin(theta);
    out = img.Image(width: side, height: side);
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
  } else if (op == 'flipH') {
    out = img.Image(width: src.width, height: src.height);
    for (int y = 0; y < src.height; y++) {
      for (int x = 0; x < src.width; x++) {
        out.setPixel(src.width - 1 - x, y, src.getPixel(x, y));
      }
    }
  } else {
    sp.send({'ok': false});
    return;
  }

  final outRgb = out.getBytes(order: img.ChannelOrder.rgb);
  sp.send({
    'ok': true,
    'w': out.width,
    'h': out.height,
    'rgb': TransferableTypedData.fromList([outRgb]),
  });
}
