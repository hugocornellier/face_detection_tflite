part of '../../face_detection_tflite.dart';

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
  for (final Detection d in dets) {
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

@visibleForTesting
Float32List testSsdGenerateAnchors(Map<String, Object> opts) =>
    _ssdGenerateAnchors(opts);

@visibleForTesting
Map<String, Object> testOptsFor(FaceDetectionModel m) => _optsFor(m);

@visibleForTesting
String testNameFor(FaceDetectionModel m) => _nameFor(m);

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
ImageTensor convertImageToTensor(
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
cv.Mat? extractAlignedSquare(
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
/// Operates directly on [cv.Mat] objects for efficient OpenCV pipeline
/// integration.
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
