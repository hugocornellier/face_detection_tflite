part of '../face_detection_tflite.dart';

/// Holds metadata for an output tensor (shape plus its writable buffer).
class OutputTensorInfo {
  /// Creates an [OutputTensorInfo] with the given [shape] and [buffer].
  ///
  /// The [shape] describes the tensor dimensions and [buffer] provides
  /// direct access to the tensor's underlying Float32 data.
  OutputTensorInfo(this.shape, this.buffer);

  /// The dimensions of the tensor (e.g., [1, 896, 1] for a 1D output with 896 elements).
  final List<int> shape;

  /// The underlying Float32 buffer containing the tensor's raw data.
  ///
  /// This provides direct access to the tensor output values without copying.
  final Float32List buffer;
}

/// Creates a 4D tensor in NHWC format (batch=1, height, width, channels=3).
///
/// This pre-allocates a nested list structure used as input buffers for
/// TensorFlow Lite models to avoid repeated allocations during inference.
///
/// - [height]: Tensor height dimension (H)
/// - [width]: Tensor width dimension (W)
/// - Returns: [1][height][width][3] nested list structure with double values
List<List<List<List<double>>>> createNHWCTensor4D(int height, int width) {
  return List<List<List<List<double>>>>.generate(
    1,
    (_) => List.generate(
      height,
      (_) => List.generate(
        width,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    ),
    growable: false,
  );
}

/// Fills a 4D NHWC tensor cache from a flat Float32List.
///
/// Converts a flat array of pixel data into the nested list structure
/// expected by TensorFlow Lite models in NHWC format:
/// - N: Batch dimension (assumed to be 1, index 0)
/// - H: Height dimension (inH)
/// - W: Width dimension (inW)
/// - C: Channel dimension (3 for RGB)
///
/// [flat] - Flattened pixel data (length must be inH * inW * 3)
/// [input4dCache] - Pre-allocated 4D structure [1][H][W][3]
/// [inH] - Input tensor height
/// [inW] - Input tensor width
@pragma('vm:prefer-inline')
void fillNHWC4D(
  Float32List flat,
  List<List<List<List<double>>>> input4dCache,
  int inH,
  int inW,
) {
  double sanitize(double value) => (value * 1e6).roundToDouble() / 1e6;

  int k = 0;
  for (int y = 0; y < inH; y++) {
    for (int x = 0; x < inW; x++) {
      final List<double> px = input4dCache[0][y][x];
      px[0] = sanitize(flat[k++]);
      px[1] = sanitize(flat[k++]);
      px[2] = sanitize(flat[k++]);
    }
  }
}

/// Allocates a nested list structure matching the given tensor shape.
///
/// Recursively builds nested lists where the innermost dimension contains
/// doubles initialized to 0.0. Used for pre-allocating output buffers for
/// TensorFlow Lite inference.
///
/// The [shape] parameter defines the dimensions of the tensor.
///
/// Returns an [Object] that is a nested list structure matching the shape.
/// The actual type depends on the number of dimensions (e.g., `List<double>`
/// for 1D, `List<List<double>>` for 2D, etc.).
///
/// Example:
/// ```dart
/// allocTensorShape([3]) → [0.0, 0.0, 0.0]
/// allocTensorShape([2, 3]) → [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
/// allocTensorShape([1, 2, 3]) → [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
/// ```
Object allocTensorShape(List<int> shape) {
  if (shape.isEmpty) return <double>[];

  Object build(int depth) {
    final int size = shape[depth];

    if (depth == shape.length - 1) {
      return List<double>.filled(size, 0.0, growable: false);
    }

    switch (shape.length - depth) {
      case 2:
        return List<List<double>>.generate(
          size,
          (_) => build(depth + 1) as List<double>,
          growable: false,
        );
      case 3:
        return List<List<List<double>>>.generate(
          size,
          (_) => build(depth + 1) as List<List<double>>,
          growable: false,
        );
      case 4:
        return List<List<List<List<double>>>>.generate(
          size,
          (_) => build(depth + 1) as List<List<List<double>>>,
          growable: false,
        );
      default:
        return List.generate(size, (_) => build(depth + 1), growable: false);
    }
  }

  return build(0);
}

/// Flattens a nested numeric tensor (dynamic output) into a Float32List.
///
/// Supports arbitrarily nested `List` structures containing `num` leaves.
/// Throws a [StateError] if an unexpected type is encountered or if [out] is null.
Float32List flattenDynamicTensor(Object? out) {
  if (out == null) {
    throw TypeError();
  }
  final List<double> flat = <double>[];
  void walk(dynamic x) {
    if (x is num) {
      flat.add(x.toDouble());
    } else if (x is List) {
      for (final e in x) {
        walk(e);
      }
    } else {
      throw StateError('Unexpected output element type: ${x.runtimeType}');
    }
  }

  walk(out);
  return Float32List.fromList(flat);
}

/// Collects output tensor shapes (and their backing buffers) for an interpreter.
///
/// Iterates output indices until `getOutputTensor` throws, mirroring existing
/// try/break loops in model constructors.
Map<int, OutputTensorInfo> collectOutputTensorInfo(Interpreter itp) {
  final Map<int, OutputTensorInfo> outputs = <int, OutputTensorInfo>{};
  for (int i = 0;; i++) {
    try {
      final Tensor t = itp.getOutputTensor(i);
      outputs[i] = OutputTensorInfo(t.shape, t.data.buffer.asFloat32List());
    } catch (_) {
      break;
    }
  }
  return outputs;
}

/// Test-only access to [collectOutputTensorInfo] for verifying output tensor collection.
///
/// This function exposes the private [collectOutputTensorInfo] for unit testing.
/// It collects all output tensor metadata from the given [itp] interpreter.
@visibleForTesting
Map<int, OutputTensorInfo> testCollectOutputTensorInfo(Interpreter itp) =>
    collectOutputTensorInfo(itp);

double _clip(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

double _sigmoidClipped(double x, {double limit = _rawScoreLimit}) {
  final double v = _clip(x, -limit, limit);
  return 1.0 / (1.0 + math.exp(-v));
}

Future<ImageTensor> _imageToTensor(
  img.Image src, {
  required int outW,
  required int outH,
}) async {
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

  final ImageTensor result = convertImageToTensor(src, outW: outW, outH: outH);

  sp.send({
    'tensor': TransferableTypedData.fromList([
      result.tensorNHWC.buffer.asUint8List(),
    ]),
    'padding': result.padding,
    'outW': result.width,
    'outH': result.height,
  });
}

/// Converts an RGB image to a normalized tensor with letterboxing.
///
/// This function performs aspect-preserving resize with black padding
/// and normalizes pixel values to the [-1.0, 1.0] range expected by
/// MediaPipe TensorFlow Lite models.
///
/// The [src] image will be resized to fit within [outW]×[outH] dimensions
/// while preserving its aspect ratio. Black padding is added to fill
/// the remaining space.
///
/// Returns an [ImageTensor] containing:
/// - Normalized float32 tensor in NHWC format
/// - Padding information needed to reverse the letterbox transformation
///
/// Example:
/// ```dart
/// final img.Image source = img.decodeImage(bytes)!;
/// final result = convertImageToTensor(source, outW: 192, outH: 192);
/// // result.tensorNHWC is ready for TFLite inference
/// // result.padding can be used to unpad output coordinates
/// ```
@Deprecated(
  'Will be removed in 5.0.0. Use convertImageToTensorFromMat instead.',
)
ImageTensor convertImageToTensor(
  img.Image src, {
  required int outW,
  required int outH,
}) {
  final int inW = src.width;
  final int inH = src.height;

  final double s1 = outW / inW;
  final double s2 = outH / inH;
  final double scale = s1 < s2 ? s1 : s2;
  final int newW = (inW * scale).round();
  final int newH = (inH * scale).round();

  final img.Image resized = img.copyResize(
    src,
    width: newW,
    height: newH,
    interpolation: img.Interpolation.linear,
  );

  final int dx = (outW - newW) ~/ 2;
  final int dy = (outH - newH) ~/ 2;

  final Float32List tensor = Float32List(outW * outH * 3);
  tensor.fillRange(0, tensor.length, -1.0);

  final Uint8List resizedRgb = resized.getBytes(order: img.ChannelOrder.rgb);
  int srcIdx = 0;
  for (int y = 0; y < resized.height; y++) {
    int dstIdx = ((y + dy) * outW + dx) * 3;
    for (int x = 0; x < resized.width; x++) {
      final int r = resizedRgb[srcIdx++];
      final int g = resizedRgb[srcIdx++];
      final int b = resizedRgb[srcIdx++];
      tensor[dstIdx++] = (r / 127.5) - 1.0;
      tensor[dstIdx++] = (g / 127.5) - 1.0;
      tensor[dstIdx++] = (b / 127.5) - 1.0;
    }
  }

  final double padTop = dy / outH;
  final double padBottom = (outH - dy - newH) / outH;
  final double padLeft = dx / outW;
  final double padRight = (outW - dx - newW) / outW;

  return ImageTensor(
    tensor,
    [padTop, padBottom, padLeft, padRight],
    outW,
    outH,
  );
}

List<Detection> _detectionLetterboxRemoval(
  List<Detection> dets,
  List<double> padding,
) {
  final double pt = padding[0],
      pb = padding[1],
      pl = padding[2],
      pr = padding[3];
  final double sx = 1.0 - (pl + pr);
  final double sy = 1.0 - (pt + pb);

  RectF unpad(RectF r) => RectF(
        (r.xmin - pl) / sx,
        (r.ymin - pt) / sy,
        (r.xmax - pl) / sx,
        (r.ymax - pt) / sy,
      );
  List<double> unpadKp(List<double> kps) {
    final List<double> out = List<double>.from(kps);
    for (int i = 0; i < out.length; i += 2) {
      out[i] = (out[i] - pl) / sx;
      out[i + 1] = (out[i + 1] - pt) / sy;
    }
    return out;
  }

  return dets
      .map(
        (d) => Detection(
          boundingBox: unpad(d.boundingBox),
          score: d.score,
          keypointsXY: unpadKp(d.keypointsXY),
        ),
      )
      .toList();
}

double _clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);

List<List<double>> _unpackLandmarks(
  Float32List flat,
  int inW,
  int inH,
  List<double> padding, {
  bool clamp = true,
}) {
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

/// Optimized NMS using spatial indexing to reduce IoU comparisons.
///
/// Uses a grid-based spatial index to skip comparisons between boxes that
/// cannot possibly overlap, reducing complexity from O(M²) to O(M × k) where
/// k is the average number of candidates per overlapping cell region.
///
/// For small candidate counts (≤8), falls back to simple O(M²) approach
/// since grid construction overhead exceeds the savings.
List<Detection> _nms(
  List<Detection> dets,
  double iouThresh,
  double scoreThresh, {
  bool weighted = true,
}) {
  final List<Detection> sorted = <Detection>[];
  for (final d in dets) {
    if (d.score >= scoreThresh) {
      sorted.add(d);
    }
  }
  if (sorted.isEmpty) return const <Detection>[];

  sorted.sort((a, b) => b.score.compareTo(a.score));

  final int n = sorted.length;

  if (n <= 8) {
    return _nmsCore(sorted, iouThresh, weighted, null);
  }

  const int gridSize = 10;
  const double cellSize = 1.0 / gridSize;

  final Map<int, List<int>> cellToIndices = <int, List<int>>{};

  for (int i = 0; i < n; i++) {
    final RectF box = sorted[i].boundingBox;
    final int minCol = (box.xmin / cellSize).floor().clamp(0, gridSize - 1);
    final int maxCol = (box.xmax / cellSize).floor().clamp(0, gridSize - 1);
    final int minRow = (box.ymin / cellSize).floor().clamp(0, gridSize - 1);
    final int maxRow = (box.ymax / cellSize).floor().clamp(0, gridSize - 1);

    for (int row = minRow; row <= maxRow; row++) {
      for (int col = minCol; col <= maxCol; col++) {
        final int cellKey = row * gridSize + col;
        (cellToIndices[cellKey] ??= <int>[]).add(i);
      }
    }
  }

  return _nmsCore(sorted, iouThresh, weighted, cellToIndices);
}

/// Core NMS logic shared between spatial and non-spatial paths.
///
/// When [cellToIndices] is null, performs O(M²) pairwise comparison.
/// When provided, uses spatial index to reduce comparisons.
List<Detection> _nmsCore(
  List<Detection> sorted,
  double iouThresh,
  bool weighted,
  Map<int, List<int>>? cellToIndices,
) {
  const int gridSize = 10;
  const double cellSize = 1.0 / gridSize;

  final int n = sorted.length;
  final List<bool> active = List<bool>.filled(n, true);
  final List<Detection> kept = <Detection>[];

  for (int i = 0; i < n; i++) {
    if (!active[i]) continue;

    final Detection base = sorted[i];
    final RectF baseBox = base.boundingBox;

    Iterable<int> candidateIndices;
    if (cellToIndices == null) {
      candidateIndices = Iterable<int>.generate(n - i - 1, (k) => i + 1 + k);
    } else {
      final int minCol = (baseBox.xmin / cellSize).floor().clamp(
            0,
            gridSize - 1,
          );
      final int maxCol = (baseBox.xmax / cellSize).floor().clamp(
            0,
            gridSize - 1,
          );
      final int minRow = (baseBox.ymin / cellSize).floor().clamp(
            0,
            gridSize - 1,
          );
      final int maxRow = (baseBox.ymax / cellSize).floor().clamp(
            0,
            gridSize - 1,
          );

      final Set<int> candidates = <int>{};
      for (int row = minRow; row <= maxRow; row++) {
        for (int col = minCol; col <= maxCol; col++) {
          final int cellKey = row * gridSize + col;
          final List<int>? indices = cellToIndices[cellKey];
          if (indices != null) {
            for (final int idx in indices) {
              if (idx > i) candidates.add(idx);
            }
          }
        }
      }
      candidateIndices = candidates;
    }

    if (!weighted) {
      kept.add(base);
      for (final int j in candidateIndices) {
        if (active[j] && _iou(baseBox, sorted[j].boundingBox) >= iouThresh) {
          active[j] = false;
        }
      }
    } else {
      double sw = base.score;
      double xmin = baseBox.xmin * base.score;
      double ymin = baseBox.ymin * base.score;
      double xmax = baseBox.xmax * base.score;
      double ymax = baseBox.ymax * base.score;

      for (final int j in candidateIndices) {
        if (!active[j]) continue;
        final Detection d = sorted[j];
        if (_iou(baseBox, d.boundingBox) >= iouThresh) {
          active[j] = false;
          sw += d.score;
          xmin += d.boundingBox.xmin * d.score;
          ymin += d.boundingBox.ymin * d.score;
          xmax += d.boundingBox.xmax * d.score;
          ymax += d.boundingBox.ymax * d.score;
        }
      }

      if (sw == base.score) {
        kept.add(base);
      } else {
        kept.add(
          Detection(
            boundingBox: RectF(xmin / sw, ymin / sw, xmax / sw, ymax / sw),
            score: base.score,
            keypointsXY: base.keypointsXY,
          ),
        );
      }
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
    for (int y = 0; y < fmH; y++) {
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
/// The [boundingBox] parameter is the face bounding box in normalized coordinates (0.0 to 1.0).
///
/// The [expandFraction] controls how much to expand the bounding box before
/// computing the square ROI. Default is 0.6 (60% expansion).
///
/// Returns a square [RectF] in normalized coordinates centered on the face,
/// with dimensions based on the larger of the expanded width or height.
///
/// This is typically used to prepare face regions for mesh landmark detection,
/// which requires a square input with some padding around the face.
RectF faceDetectionToRoi(RectF boundingBox, {double expandFraction = 0.6}) {
  final RectF e = boundingBox.expand(expandFraction);
  final double cx = (e.xmin + e.xmax) * 0.5;
  final double cy = (e.ymin + e.ymax) * 0.5;
  final double s = math.max(e.w, e.h) * 0.5;
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
@Deprecated('Will be removed in 5.0.0. Use cropFromRoiMat instead.')
Future<img.Image> cropFromRoi(img.Image src, RectF roi) async {
  if (roi.xmin < 0 || roi.ymin < 0 || roi.xmax > 1 || roi.ymax > 1) {
    throw ArgumentError(
      'ROI coordinates must be normalized [0,1], got: (${roi.xmin}, ${roi.ymin}, ${roi.xmax}, ${roi.ymax})',
    );
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
    width: ow,
    height: oh,
    bytes: outRgb.buffer,
    order: img.ChannelOrder.rgb,
  );
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
@Deprecated(
  'Will be removed in 5.0.0. Use extractAlignedSquareFromMat instead.',
)
Future<img.Image> extractAlignedSquare(
  img.Image src,
  double cx,
  double cy,
  double size,
  double theta,
) async {
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
    width: ow,
    height: oh,
    bytes: outRgb.buffer,
    order: img.ChannelOrder.rgb,
  );
}

/// Optimized bilinear sampling that writes directly to an RGB buffer.
///
/// Instead of creating intermediate ColorRgb8 objects, this writes the
/// sampled RGB values directly to [outRgb] at position [outIdx].
/// Uses direct buffer access for both reading source pixels and writing output.
@pragma('vm:prefer-inline')
void _bilinearSampleToBuffer(
  Uint8List srcRgb,
  int srcW,
  int srcH,
  double fx,
  double fy,
  Uint8List outRgb,
  int outIdx,
) {
  final int x0 = fx.floor();
  final int y0 = fy.floor();
  final double ax = fx - x0;
  final double ay = fy - y0;

  final int cx0 = x0 < 0 ? 0 : (x0 >= srcW ? srcW - 1 : x0);
  final int cx1 = x0 + 1 < 0 ? 0 : (x0 + 1 >= srcW ? srcW - 1 : x0 + 1);
  final int cy0 = y0 < 0 ? 0 : (y0 >= srcH ? srcH - 1 : y0);
  final int cy1 = y0 + 1 < 0 ? 0 : (y0 + 1 >= srcH ? srcH - 1 : y0 + 1);

  final int i00 = (cy0 * srcW + cx0) * 3;
  final int i10 = (cy0 * srcW + cx1) * 3;
  final int i01 = (cy1 * srcW + cx0) * 3;
  final int i11 = (cy1 * srcW + cx1) * 3;

  final int r00 = srcRgb[i00], g00 = srcRgb[i00 + 1], b00 = srcRgb[i00 + 2];
  final int r10 = srcRgb[i10], g10 = srcRgb[i10 + 1], b10 = srcRgb[i10 + 2];
  final int r01 = srcRgb[i01], g01 = srcRgb[i01 + 1], b01 = srcRgb[i01 + 2];
  final int r11 = srcRgb[i11], g11 = srcRgb[i11 + 1], b11 = srcRgb[i11 + 2];

  final double oneMinusAx = 1.0 - ax;
  final double oneMinusAy = 1.0 - ay;

  final double r0 = r00 * oneMinusAx + r10 * ax;
  final double g0 = g00 * oneMinusAx + g10 * ax;
  final double b0 = b00 * oneMinusAx + b10 * ax;
  final double r1 = r01 * oneMinusAx + r11 * ax;
  final double g1 = g01 * oneMinusAx + g11 * ax;
  final double b1 = b01 * oneMinusAx + b11 * ax;

  int r = (r0 * oneMinusAy + r1 * ay).round();
  int g = (g0 * oneMinusAy + g1 * ay).round();
  int b = (b0 * oneMinusAy + b1 * ay).round();
  outRgb[outIdx] = r < 0 ? 0 : (r > 255 ? 255 : r);
  outRgb[outIdx + 1] = g < 0 ? 0 : (g > 255 ? 255 : g);
  outRgb[outIdx + 2] = b < 0 ? 0 : (b > 255 ? 255 : b);
}

/// RGB image payload decoded off the UI thread.
@Deprecated('Will be removed in 5.0.0. Use cv.imdecode instead.')
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
      'Could not decode image bytes: ${error ?? "unsupported or corrupt"}',
    );
  }

  final ByteBuffer rgbBB = (msg['rgb'] as TransferableTypedData).materialize();
  final Uint8List rgb = rgbBB.asUint8List();
  final int w = msg['w'] as int;
  final int h = msg['h'] as int;
  return DecodedRgb(w, h, rgb);
}

@visibleForTesting
@Deprecated('Will be removed in 5.0.0. Use cv.imdecode instead.')
Future<DecodedRgb> testDecodeImageOffUi(Uint8List bytes) =>
    _decodeImageOffUi(bytes);

img.Image _imageFromDecodedRgb(DecodedRgb d) {
  return img.Image.fromBytes(
    width: d.width,
    height: d.height,
    bytes: d.rgb.buffer,
    order: img.ChannelOrder.rgb,
  );
}

@visibleForTesting
@Deprecated(
  'Will be removed in 5.0.0. The image package dependency will be removed.',
)
img.Image testImageFromDecodedRgb(DecodedRgb d) => _imageFromDecodedRgb(d);

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
      width: w,
      height: h,
      bytes: inRgb.buffer,
      order: img.ChannelOrder.rgb,
    );

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

      final Uint8List outRgbBuf = Uint8List(side * side * 3);
      final double invSide = 1.0 / side;

      int outIdx = 0;
      for (int y = 0; y < side; y++) {
        final double vy = ((y + 0.5) * invSide - 0.5) * size;
        final double vyct = vy * ct;
        final double vyst = vy * st;
        for (int x = 0; x < side; x++) {
          final double vx = ((x + 0.5) * invSide - 0.5) * size;
          final double sx = cx + vx * ct - vyst;
          final double sy = cy + vx * st + vyct;
          _bilinearSampleToBuffer(inRgb, w, h, sx, sy, outRgbBuf, outIdx);
          outIdx += 3;
        }
      }

      out = img.Image.fromBytes(
        width: side,
        height: side,
        bytes: outRgbBuf.buffer,
        order: img.ChannelOrder.rgb,
      );
    } else if (op == 'flipH') {
      final Uint8List outRgbBuf = Uint8List(w * h * 3);
      for (int y = 0; y < h; y++) {
        final int rowStart = y * w * 3;
        for (int x = 0; x < w; x++) {
          final int srcIdx = rowStart + x * 3;
          final int dstIdx = rowStart + (w - 1 - x) * 3;
          outRgbBuf[dstIdx] = inRgb[srcIdx];
          outRgbBuf[dstIdx + 1] = inRgb[srcIdx + 1];
          outRgbBuf[dstIdx + 2] = inRgb[srcIdx + 2];
        }
      }
      out = img.Image.fromBytes(
        width: w,
        height: h,
        bytes: outRgbBuf.buffer,
        order: img.ChannelOrder.rgb,
      );
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

//
//

/// Decodes an image using a worker if provided, otherwise spawns a new isolate.
///
/// This is an optimized variant of [_decodeImageOffUi] that uses a long-lived
/// worker isolate when available, avoiding the overhead of spawning fresh
/// isolates for each decode operation.
///
/// When [worker] is null, falls back to [_decodeImageOffUi] for backwards
/// compatibility.
@Deprecated('Will be removed in 5.0.0. Use cv.imdecode instead.')
Future<DecodedRgb> decodeImageWithWorker(
  Uint8List bytes,
  IsolateWorker? worker,
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
@Deprecated(
  'Will be removed in 5.0.0. Use convertImageToTensorFromMat instead.',
)
Future<ImageTensor> imageToTensorWithWorker(
  img.Image src, {
  required int outW,
  required int outH,
  IsolateWorker? worker,
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
@Deprecated('Will be removed in 5.0.0. Use cropFromRoiMat instead.')
Future<img.Image> cropFromRoiWithWorker(
  img.Image src,
  RectF roi,
  IsolateWorker? worker,
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
@Deprecated(
  'Will be removed in 5.0.0. Use extractAlignedSquareFromMat instead.',
)
Future<img.Image> extractAlignedSquareWithWorker(
  img.Image src,
  double cx,
  double cy,
  double size,
  double theta,
  IsolateWorker? worker,
) async {
  if (worker != null) {
    return await worker.extractAlignedSquare(src, cx, cy, size, theta);
  } else {
    return await extractAlignedSquare(src, cx, cy, size, theta);
  }
}

/// Test-only access to [_clip] for verifying value clamping behavior.
///
/// This function exposes the private [_clip] for unit testing.
/// It clamps [v] to the range [[lo], [hi]].
@visibleForTesting
double testClip(double v, double lo, double hi) => _clip(v, lo, hi);

@visibleForTesting
double testSigmoidClipped(double x, {double limit = _rawScoreLimit}) =>
    _sigmoidClipped(x, limit: limit);

@visibleForTesting
List<Detection> testDetectionLetterboxRemoval(
  List<Detection> dets,
  List<double> padding,
) =>
    _detectionLetterboxRemoval(dets, padding);

@visibleForTesting
List<List<double>> testUnpackLandmarks(
  Float32List flat,
  int inW,
  int inH,
  List<double> padding, {
  bool clamp = true,
}) =>
    _unpackLandmarks(flat, inW, inH, padding, clamp: clamp);

@visibleForTesting
List<Detection> testNms(
  List<Detection> dets,
  double iouThresh,
  double scoreThresh, {
  bool weighted = true,
}) =>
    _nms(dets, iouThresh, scoreThresh, weighted: weighted);

/// Converts a cv.Mat image to a normalized tensor with letterboxing.
///
/// This function performs aspect-preserving resize with black padding
/// and normalizes pixel values to the [-1.0, 1.0] range expected by
/// MediaPipe TensorFlow Lite models.
///
/// The [src] cv.Mat will be resized to fit within [outW]×[outH] dimensions
/// while preserving its aspect ratio. Black padding is added to fill
/// the remaining space.
///
/// Returns an [ImageTensor] containing:
/// - Normalized float32 tensor in NHWC format
/// - Padding information needed to reverse the letterbox transformation
///
/// Note: The input cv.Mat is NOT disposed by this function.
ImageTensor convertImageToTensorFromMat(
  cv.Mat src, {
  required int outW,
  required int outH,
  Float32List? buffer,
}) {
  final int inW = src.cols;
  final int inH = src.rows;

  final double s1 = outW / inW;
  final double s2 = outH / inH;
  final double scale = s1 < s2 ? s1 : s2;
  final int newW = (inW * scale).round();
  final int newH = (inH * scale).round();

  final cv.Mat resized = cv.resize(
      src,
      (
        newW,
        newH,
      ),
      interpolation: cv.INTER_LINEAR);

  final int dx = (outW - newW) ~/ 2;
  final int dy = (outH - newH) ~/ 2;

  final int padRight = outW - newW - dx;
  final int padBottom = outH - newH - dy;

  final cv.Mat padded = cv.copyMakeBorder(
    resized,
    dy,
    padBottom,
    dx,
    padRight,
    cv.BORDER_CONSTANT,
    value: cv.Scalar.black,
  );
  resized.dispose();

  final Float32List tensor = buffer ?? Float32List(outW * outH * 3);

  final Uint8List data = padded.data;
  final int totalPixels = outW * outH;
  for (int i = 0, j = 0;
      i < totalPixels * 3 && j < totalPixels * 3;
      i += 3, j += 3) {
    tensor[j] = (data[i + 2] / 127.5) - 1.0;
    tensor[j + 1] = (data[i + 1] / 127.5) - 1.0;
    tensor[j + 2] = (data[i] / 127.5) - 1.0;
  }
  padded.dispose();

  final double padTop = dy / outH;
  final double padBottomNorm = (outH - dy - newH) / outH;
  final double padLeft = dx / outW;
  final double padRightNorm = (outW - dx - newW) / outW;

  return ImageTensor(
    tensor,
    [padTop, padBottomNorm, padLeft, padRightNorm],
    outW,
    outH,
  );
}

/// Extracts a rotated square region from a cv.Mat using OpenCV's warpAffine.
///
/// This function uses SIMD-accelerated warpAffine which is 10-50x faster
/// than pure Dart bilinear interpolation.
///
/// Parameters:
/// - [src]: Source cv.Mat image
/// - [cx]: Center X coordinate in pixels
/// - [cy]: Center Y coordinate in pixels
/// - [size]: Output square size in pixels
/// - [theta]: Rotation angle in radians (positive = counter-clockwise)
///
/// Returns the cropped and rotated cv.Mat. Caller must dispose.
/// Returns null if size is invalid.
cv.Mat? extractAlignedSquareFromMat(
  cv.Mat src,
  double cx,
  double cy,
  double size,
  double theta,
) {
  final int sizeInt = size.round();
  if (sizeInt <= 0) return null;

  final double angleDegrees = -theta * 180.0 / math.pi;

  final cv.Mat rotMat = cv.getRotationMatrix2D(
    cv.Point2f(cx, cy),
    angleDegrees,
    1.0,
  );

  final double outCenter = sizeInt / 2.0;

  final double tx = rotMat.at<double>(0, 2) + outCenter - cx;
  final double ty = rotMat.at<double>(1, 2) + outCenter - cy;
  rotMat.set<double>(0, 2, tx);
  rotMat.set<double>(1, 2, ty);

  final cv.Mat output = cv.warpAffine(
    src,
    rotMat,
    (sizeInt, sizeInt),
    borderMode: cv.BORDER_CONSTANT,
    borderValue: cv.Scalar.black,
  );

  rotMat.dispose();
  return output;
}

/// Crops a rectangular region from a [cv.Mat] using normalized coordinates.
///
/// This is the OpenCV-based equivalent of [cropFromRoi], operating directly
/// on [cv.Mat] objects for better performance when working with OpenCV
/// pipelines.
///
/// The [src] parameter is the source image to crop from.
///
/// The [roi] parameter defines the crop region with normalized coordinates
/// where (0, 0) is the top-left corner and (1, 1) is the bottom-right corner
/// of the source image. Coordinates are clamped to valid image bounds.
///
/// Returns a cropped [cv.Mat] containing the specified region. The returned
/// Mat shares memory with [src] via [cv.Mat.region], so [src] must remain
/// valid while the result is in use. **Caller is responsible for disposing
/// the returned Mat.**
///
/// Example:
/// ```dart
/// final roi = RectF(0.2, 0.3, 0.8, 0.7);
/// final cropped = cropFromRoiMat(sourceMat, roi);
/// // Use cropped...
/// cropped.dispose();
/// ```
cv.Mat cropFromRoiMat(cv.Mat src, RectF roi) {
  final int w = src.cols;
  final int h = src.rows;

  final int x1 = (roi.xmin * w).round().clamp(0, w - 1);
  final int y1 = (roi.ymin * h).round().clamp(0, h - 1);
  final int x2 = (roi.xmax * w).round().clamp(x1 + 1, w);
  final int y2 = (roi.ymax * h).round().clamp(y1 + 1, h);

  final int cropW = x2 - x1;
  final int cropH = y2 - y1;

  final cv.Rect rect = cv.Rect(x1, y1, cropW, cropH);
  return src.region(rect);
}
