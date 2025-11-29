part of '../face_detection_tflite.dart';

double _clip(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

double _sigmoidClipped(double x, {double limit = _rawScoreLimit}) {
  final double v = _clip(x, -limit, limit);
  return 1.0 / (1.0 + math.exp(-v));
}

Future<ImageTensor> _imageToTensor(img.Image src,
    {required int outW, required int outH}) async {
  final ReceivePort rp = ReceivePort();
  final Uint8List rgb = src.getBytes(order: img.ChannelOrder.rgb);
  await Isolate.spawn(_imageToTensorIsolate, {
    'sendPort': rp.sendPort,
    'inW': src.width,
    'inH': src.height,
    'outW': outW,
    'outH': outH,
    'rgb': TransferableTypedData.fromList([rgb]),
  });
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
  final ByteBuffer rgbBB =
      (params['rgb'] as TransferableTypedData).materialize();
  final Uint8List rgb = rgbBB.asUint8List();

  final img.Image src = img.Image.fromBytes(
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

  final img.Image canvas = img.Image(width: outW, height: outH);
  img.fill(canvas, color: img.ColorRgb8(0, 0, 0));

  for (int y = 0; y < resized.height; y++) {
    for (int x = 0; x < resized.width; x++) {
      final img.Pixel px = resized.getPixel(x, y);
      canvas.setPixel(x + dx, y + dy, px);
    }
  }

  final Float32List t = Float32List(outW * outH * 3);
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

List<Detection> _detectionLetterboxRemoval(
    List<Detection> dets, List<double> padding) {
  final double pt = padding[0],
      pb = padding[1],
      pl = padding[2],
      pr = padding[3];
  final double sx = 1.0 - (pl + pr);
  final double sy = 1.0 - (pt + pb);

  RectF unpad(RectF r) => RectF((r.xmin - pl) / sx, (r.ymin - pt) / sy,
      (r.xmax - pl) / sx, (r.ymax - pt) / sy);
  List<double> unpadKp(List<double> kps) {
    final List<double> out = List<double>.from(kps);
    for (int i = 0; i < out.length; i += 2) {
      out[i] = (out[i] - pl) / sx;
      out[i + 1] = (out[i + 1] - pt) / sy;
    }
    return out;
  }

  return dets
      .map((d) => Detection(
          bbox: unpad(d.bbox),
          score: d.score,
          keypointsXY: unpadKp(d.keypointsXY)))
      .toList();
}

double _clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);

List<List<double>> _unpackLandmarks(
    Float32List flat, int inW, int inH, List<double> padding,
    {bool clamp = true}) {
  final double pt = padding[0],
      pb = padding[1],
      pl = padding[2],
      pr = padding[3];
  final double sx = 1.0 - (pl + pr);
  final double sy = 1.0 - (pt + pb);

  final int n = (flat.length / 3).floor();
  final List<List<double>> out = <List<double>>[];
  for (var i = 0; i < n; i++) {
    double x = flat[i * 3 + 0] / inW;
    double y = flat[i * 3 + 1] / inH;
    final double z = flat[i * 3 + 2];
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
  final double dx = roi.xmin, dy = roi.ymin, sx = roi.w, sy = roi.h;
  RectF mapRect(RectF r) => RectF(
      dx + r.xmin * sx, dy + r.ymin * sy, dx + r.xmax * sx, dy + r.ymax * sy);
  List<double> mapKp(List<double> k) {
    final List<double> o = List<double>.from(k);
    for (int i = 0; i < o.length; i += 2) {
      o[i] = _clamp01(dx + o[i] * sx);
      o[i + 1] = _clamp01(dy + o[i + 1] * sy);
    }
    return o;
  }

  return Detection(
      bbox: mapRect(d.bbox),
      score: d.score,
      keypointsXY: mapKp(d.keypointsXY),
      imageSize: d.imageSize);
}

double _iou(RectF a, RectF b) {
  final double x1 = math.max(a.xmin, b.xmin);
  final double y1 = math.max(a.ymin, b.ymin);
  final double x2 = math.min(a.xmax, b.xmax);
  final double y2 = math.min(a.ymax, b.ymax);
  final double iw = math.max(0.0, x2 - x1);
  final double ih = math.max(0.0, y2 - y1);
  final double inter = iw * ih;
  final double areaA = math.max(0.0, a.w) * math.max(0.0, a.h);
  final double areaB = math.max(0.0, b.w) * math.max(0.0, b.h);
  final double uni = areaA + areaB - inter;
  return uni <= 0 ? 0.0 : inter / uni;
}

List<Detection> _nms(List<Detection> dets, double iouThresh, double scoreThresh,
    {bool weighted = true}) {
  final List<Detection> kept = <Detection>[];
  final List<Detection> cand = dets
      .where((d) => d.score >= scoreThresh)
      .toList()
    ..sort((a, b) => b.score.compareTo(a.score));
  while (cand.isNotEmpty) {
    final Detection base = cand.removeAt(0);
    final List<Detection> merged = <Detection>[base];
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
      for (final Detection m in merged) {
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
  final List<int> strides = (opts['strides'] as List).cast<int>();
  final int numLayers = opts['num_layers'] as int;
  final int inputH = opts['input_size_height'] as int;
  final int inputW = opts['input_size_width'] as int;

  final double ax = (opts['anchor_offset_x'] as num).toDouble();
  final double ay = (opts['anchor_offset_y'] as num).toDouble();
  final double interp =
      (opts['interpolated_scale_aspect_ratio'] as num).toDouble();
  final List<double> anchors = <double>[];
  int layerId = 0;
  while (layerId < numLayers) {
    int lastSameStride = layerId;
    int repeats = 0;
    while (lastSameStride < numLayers &&
        strides[lastSameStride] == strides[layerId]) {
      lastSameStride++;
      repeats += (interp == 1.0) ? 2 : 1;
    }
    final int stride = strides[layerId];
    final int fmH = inputH ~/ stride;
    final int fmW = inputW ~/ stride;
    for (var y = 0; y < fmH; y++) {
      final double yCenter = (y + ay) / fmH;
      for (int x = 0; x < fmW; x++) {
        final double xCenter = (x + ax) / fmW;
        for (int r = 0; r < repeats; r++) {
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

/// Converts a face detection bounding box to a square region of interest (ROI).
///
/// This function takes a face bounding box and generates a square ROI suitable
/// for face mesh alignment. The process involves:
/// 1. Expanding the bounding box by [expandFraction] (default 0.6 = 60% larger)
/// 2. Finding the center of the expanded box
/// 3. Creating a square ROI centered on that point
///
/// The [bbox] parameter is the face bounding box in normalized coordinates (0.0 to 1.0).
///
/// The [expandFraction] controls how much to expand the bounding box before
/// computing the square ROI. Default is 0.6 (60% expansion).
///
/// Returns a square [RectF] in normalized coordinates centered on the face,
/// with dimensions based on the larger of the expanded width or height.
///
/// This is typically used to prepare face regions for mesh landmark detection,
/// which requires a square input with some padding around the face.
RectF faceDetectionToRoi(RectF bbox, {double expandFraction = 0.6}) {
  final e = bbox.expand(expandFraction);
  final cx = (e.xmin + e.xmax) * 0.5;
  final cy = (e.ymin + e.ymax) * 0.5;
  final s = math.max(e.w, e.h) * 0.5;
  return RectF(cx - s, cy - s, cx + s, cy + s);
}

/// Crops a region of interest from an image using normalized coordinates.
///
/// This function extracts a rectangular region from [src] based on the normalized
/// [roi] coordinates. The cropping operation runs in a separate isolate to avoid
/// blocking the main thread.
///
/// The [src] parameter is the source image to crop from.
///
/// The [roi] parameter defines the crop region with normalized coordinates where
/// (0, 0) is the top-left corner and (1, 1) is the bottom-right corner of the
/// source image. All coordinates must be in the range [0.0, 1.0].
///
/// Returns a cropped [img.Image] containing the specified region.
///
/// Throws [ArgumentError] if:
/// - Any ROI coordinate is outside the [0.0, 1.0] range
/// - The ROI has invalid dimensions (min >= max)
///
/// Throws [StateError] if the cropping operation fails in the isolate.
///
/// Example:
/// ```dart
/// final roi = RectF(0.2, 0.3, 0.8, 0.7); // Crop center region
/// final cropped = await cropFromRoi(sourceImage, roi);
/// ```
Future<img.Image> cropFromRoi(img.Image src, RectF roi) async {
  if (roi.xmin < 0 || roi.ymin < 0 || roi.xmax > 1 || roi.ymax > 1) {
    throw ArgumentError(
        'ROI coordinates must be normalized [0,1], got: (${roi.xmin}, ${roi.ymin}, ${roi.xmax}, ${roi.ymax})');
  }
  if (roi.xmin >= roi.xmax || roi.ymin >= roi.ymax) {
    throw ArgumentError('Invalid ROI: min coordinates must be less than max');
  }
  final Uint8List rgb = src.getBytes(order: img.ChannelOrder.rgb);
  final ReceivePort rp = ReceivePort();
  await Isolate.spawn(_imageTransformIsolate, {
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
  });
  final Map msg = await rp.first as Map;
  rp.close();
  if (msg['ok'] != true) {
    final error = msg['error'];
    throw StateError('Image crop failed: ${error ?? "unknown error"}');
  }
  final ByteBuffer outBB = (msg['rgb'] as TransferableTypedData).materialize();
  final Uint8List outRgb = outBB.asUint8List();
  final int ow = msg['w'] as int;
  final int oh = msg['h'] as int;
  return img.Image.fromBytes(
      width: ow, height: oh, bytes: outRgb.buffer, order: img.ChannelOrder.rgb);
}

/// Extracts a rotated square region from an image with bilinear sampling.
///
/// This function extracts a square image patch centered at ([cx], [cy]) with
/// the specified [size] and rotation angle [theta]. The extraction uses bilinear
/// interpolation for smooth results and runs in a separate isolate to avoid
/// blocking the main thread.
///
/// This is commonly used to align faces to a canonical orientation before
/// running face mesh detection, where the rotation angle is computed from
/// eye positions.
///
/// The [src] parameter is the source image to extract from.
///
/// The [cx] and [cy] parameters specify the center point in absolute pixel
/// coordinates within the source image.
///
/// The [size] parameter is the side length of the output square in pixels.
/// Must be positive.
///
/// The [theta] parameter is the rotation angle in radians. Positive values
/// rotate counter-clockwise. The extracted region is rotated by this angle
/// around the center point before sampling.
///
/// Returns a square [img.Image] of dimensions [size] × [size] pixels containing
/// the rotated region with bilinear interpolation for smooth edges.
///
/// Throws [ArgumentError] if [size] is not positive.
///
/// Throws [StateError] if the extraction operation fails in the isolate.
///
/// Example:
/// ```dart
/// // Extract 192x192 face aligned to horizontal
/// final aligned = await extractAlignedSquare(
///   image,
///   faceCenterX,
///   faceCenterY,
///   192.0,
///   -rotationAngle, // Negative to upright the face
/// );
/// ```
Future<img.Image> extractAlignedSquare(
    img.Image src, double cx, double cy, double size, double theta) async {
  if (size <= 0) {
    throw ArgumentError('Size must be positive, got: $size');
  }
  final Uint8List rgb = src.getBytes(order: img.ChannelOrder.rgb);
  final ReceivePort rp = ReceivePort();
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
  if (msg['ok'] != true) {
    final error = msg['error'];
    throw StateError('Image extraction failed: ${error ?? "unknown error"}');
  }
  final ByteBuffer outBB = (msg['rgb'] as TransferableTypedData).materialize();
  final Uint8List outRgb = outBB.asUint8List();
  final int ow = msg['w'] as int;
  final int oh = msg['h'] as int;
  return img.Image.fromBytes(
      width: ow, height: oh, bytes: outRgb.buffer, order: img.ChannelOrder.rgb);
}

img.ColorRgb8 _bilinearSampleRgb8(img.Image src, double fx, double fy) {
  final int x0 = fx.floor();
  final int y0 = fy.floor();
  final int x1 = x0 + 1;
  final int y1 = y0 + 1;
  final double ax = fx - x0;
  final double ay = fy - y0;

  int cx0 = x0.clamp(0, src.width - 1);
  int cx1 = x1.clamp(0, src.width - 1);
  int cy0 = y0.clamp(0, src.height - 1);
  int cy1 = y1.clamp(0, src.height - 1);

  final img.Pixel p00 = src.getPixel(cx0, cy0);
  final img.Pixel p10 = src.getPixel(cx1, cy0);
  final img.Pixel p01 = src.getPixel(cx0, cy1);
  final img.Pixel p11 = src.getPixel(cx1, cy1);

  final double r0 = p00.r * (1 - ax) + p10.r * ax;
  final double g0 = p00.g * (1 - ax) + p10.g * ax;
  final double b0 = p00.b * (1 - ax) + p10.b * ax;

  final double r1 = p01.r * (1 - ax) + p11.r * ax;
  final double g1 = p01.g * (1 - ax) + p11.g * ax;
  final double b1 = p01.b * (1 - ax) + p11.b * ax;

  final int r = (r0 * (1 - ay) + r1 * ay).round().clamp(0, 255);
  final int g = (g0 * (1 - ay) + g1 * ay).round().clamp(0, 255);
  final int b = (b0 * (1 - ay) + b1 * ay).round().clamp(0, 255);

  return img.ColorRgb8(r, g, b);
}

/// RGB image payload decoded off the UI thread.
class DecodedRgb {
  /// Width of the decoded image in pixels.
  final int width;

  /// Height of the decoded image in pixels.
  final int height;

  /// Raw RGB bytes in row-major order.
  final Uint8List rgb;

  /// Creates a decoded RGB payload with explicit [width], [height], and
  /// row-major [rgb] bytes that can be converted back to an [img.Image].
  const DecodedRgb(this.width, this.height, this.rgb);
}

Future<DecodedRgb> _decodeImageOffUi(Uint8List bytes) async {
  if (bytes.isEmpty) {
    throw ArgumentError('Image bytes cannot be empty');
  }
  final ReceivePort rp = ReceivePort();
  await Isolate.spawn(_decodeImageIsolate, {
    'sendPort': rp.sendPort,
    'bytes': TransferableTypedData.fromList([bytes]),
  });
  final Map msg = await rp.first as Map;
  rp.close();

  if (msg['ok'] != true) {
    final error = msg['error'];
    throw FormatException(
        'Could not decode image bytes: ${error ?? "unsupported or corrupt"}');
  }

  final ByteBuffer rgbBB = (msg['rgb'] as TransferableTypedData).materialize();
  final Uint8List rgb = rgbBB.asUint8List();
  final int w = msg['w'] as int;
  final int h = msg['h'] as int;
  return DecodedRgb(w, h, rgb);
}

img.Image _imageFromDecodedRgb(DecodedRgb d) {
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
  try {
    final ByteBuffer bb =
        (params['bytes'] as TransferableTypedData).materialize();
    final Uint8List inBytes = bb.asUint8List();

    final img.Image? decoded = img.decodeImage(inBytes);
    if (decoded == null) {
      sp.send({'ok': false, 'error': 'Failed to decode image format'});
      return;
    }

    final Uint8List rgb = decoded.getBytes(order: img.ChannelOrder.rgb);
    sp.send({
      'ok': true,
      'w': decoded.width,
      'h': decoded.height,
      'rgb': TransferableTypedData.fromList([rgb]),
    });
  } catch (e) {
    sp.send({'ok': false, 'error': e.toString()});
  }
}

@pragma('vm:entry-point')
Future<void> _imageTransformIsolate(Map<String, dynamic> params) async {
  final SendPort sp = params['sendPort'] as SendPort;
  try {
    final String op = params['op'] as String;
    final int w = params['w'] as int;
    final int h = params['h'] as int;
    final ByteBuffer inBB =
        (params['rgb'] as TransferableTypedData).materialize();
    final Uint8List inRgb = inBB.asUint8List();

    final img.Image src = img.Image.fromBytes(
        width: w, height: h, bytes: inRgb.buffer, order: img.ChannelOrder.rgb);

    img.Image out;

    if (op == 'crop') {
      final Map<dynamic, dynamic> m = params['roi'] as Map;
      final double xmin = (m['xmin'] as num).toDouble();
      final double ymin = (m['ymin'] as num).toDouble();
      final double xmax = (m['xmax'] as num).toDouble();
      final double ymax = (m['ymax'] as num).toDouble();

      final double W = src.width.toDouble(), H = src.height.toDouble();
      final int x0 = (xmin * W).clamp(0.0, W - 1).toInt();
      final int y0 = (ymin * H).clamp(0.0, H - 1).toInt();
      final int x1 = (xmax * W).clamp(0.0, W).toInt();
      final int y1 = (ymax * H).clamp(0.0, H).toInt();
      final int cw = math.max(1, x1 - x0);
      final int ch = math.max(1, y1 - y0);
      out = img.copyCrop(src, x: x0, y: y0, width: cw, height: ch);
    } else if (op == 'extract') {
      final double cx = (params['cx'] as num).toDouble();
      final double cy = (params['cy'] as num).toDouble();
      final double size = (params['size'] as num).toDouble();
      final double theta = (params['theta'] as num).toDouble();

      final int side = math.max(1, size.round());
      final double ct = math.cos(theta);
      final double st = math.sin(theta);
      out = img.Image(width: side, height: side);
      for (int y = 0; y < side; y++) {
        final double vy = ((y + 0.5) / side - 0.5) * size;
        for (int x = 0; x < side; x++) {
          final double vx = ((x + 0.5) / side - 0.5) * size;
          final double sx = cx + vx * ct - vy * st;
          final double sy = cy + vx * st + vy * ct;
          final img.ColorRgb8 px = _bilinearSampleRgb8(src, sx, sy);
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
      sp.send({'ok': false, 'error': 'Unknown operation: $op'});
      return;
    }

    final Uint8List outRgb = out.getBytes(order: img.ChannelOrder.rgb);
    sp.send({
      'ok': true,
      'w': out.width,
      'h': out.height,
      'rgb': TransferableTypedData.fromList([outRgb]),
    });
  } catch (e) {
    sp.send({'ok': false, 'error': e.toString()});
  }
}

// =============================================================================
// WORKER-AWARE HELPER FUNCTIONS (Phase 2 Optimization)
// =============================================================================
//
// These functions provide optimized paths when an ImageProcessingWorker is
// available. They fall back to the original isolate-spawning implementations
// when no worker is provided, ensuring full backwards compatibility.
//
// Usage: Internal to FaceDetector and related classes.
// =============================================================================

/// Decodes an image using a worker if provided, otherwise spawns a new isolate.
///
/// This is an optimized variant of [_decodeImageOffUi] that uses a long-lived
/// worker isolate when available, avoiding the overhead of spawning fresh
/// isolates for each decode operation.
///
/// When [worker] is null, falls back to [_decodeImageOffUi] for backwards
/// compatibility.
Future<DecodedRgb> decodeImageWithWorker(
  Uint8List bytes,
  ImageProcessingWorker? worker,
) async {
  if (worker != null) {
    return await worker.decodeImage(bytes);
  } else {
    return await _decodeImageOffUi(bytes);
  }
}

/// Converts an image to a tensor using a worker if provided.
///
/// This is an optimized variant of [_imageToTensor] that uses a long-lived
/// worker isolate when available, avoiding the overhead of spawning fresh
/// isolates for each conversion.
///
/// When [worker] is null, falls back to [_imageToTensor] for backwards
/// compatibility.
Future<ImageTensor> imageToTensorWithWorker(
  img.Image src, {
  required int outW,
  required int outH,
  ImageProcessingWorker? worker,
}) async {
  if (worker != null) {
    return await worker.imageToTensor(src, outW: outW, outH: outH);
  } else {
    return await _imageToTensor(src, outW: outW, outH: outH);
  }
}

/// Crops a region from an image using a worker if provided.
///
/// This is an optimized variant of [cropFromRoi] that uses a long-lived
/// worker isolate when available, avoiding the overhead of spawning fresh
/// isolates for each crop operation.
///
/// When [worker] is null, falls back to [cropFromRoi] for backwards
/// compatibility.
Future<img.Image> cropFromRoiWithWorker(
  img.Image src,
  RectF roi,
  ImageProcessingWorker? worker,
) async {
  if (worker != null) {
    return await worker.cropFromRoi(src, roi);
  } else {
    return await cropFromRoi(src, roi);
  }
}

/// Extracts an aligned square from an image using a worker if provided.
///
/// This is an optimized variant of [extractAlignedSquare] that uses a long-lived
/// worker isolate when available, avoiding the overhead of spawning fresh
/// isolates for each extraction.
///
/// When [worker] is null, falls back to [extractAlignedSquare] for backwards
/// compatibility.
Future<img.Image> extractAlignedSquareWithWorker(
  img.Image src,
  double cx,
  double cy,
  double size,
  double theta,
  ImageProcessingWorker? worker,
) async {
  if (worker != null) {
    return await worker.extractAlignedSquare(src, cx, cy, size, theta);
  } else {
    return await extractAlignedSquare(src, cx, cy, size, theta);
  }
}
