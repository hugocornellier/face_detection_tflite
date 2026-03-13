part of '../../face_detection_tflite.dart';

/// Holds metadata for an output tensor (shape plus its writable buffer).
class OutputTensorInfo {
  /// Creates an [OutputTensorInfo] with the given [shape] and [buffer].
  ///
  /// The [shape] describes the tensor dimensions and [buffer] provides
  /// direct access to the tensor's underlying Float32 data.
  OutputTensorInfo(this.shape, this.buffer);

  /// The dimensions of the tensor (e.g., `[1, 896, 1]` for a 1D output with 896 elements).
  final List<int> shape;

  /// The underlying Float32 buffer containing the tensor's raw data.
  ///
  /// This provides direct access to the tensor output values without copying.
  final Float32List buffer;
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

/// Shared dispose logic for TFLite model classes.
///
/// Provides [_doDispose] (synchronous) and [_doDisposeAsync] (asynchronous)
/// helpers that handle the [_disposed] guard, delegate cleanup, isolate
/// shutdown, and interpreter close. Consuming classes provide [_itp] as a
/// `final` field, which satisfies the abstract getter.
mixin _TfliteModelDisposable {
  IsolateInterpreter? _iso;
  Delegate? _delegate;
  bool _disposed = false;

  Interpreter get _itp;

  void _doDispose() {
    if (_disposed) return;
    _disposed = true;
    _delegate?.delete();
    _delegate = null;
    _iso?.close();
    _itp.close();
  }

  Future<void> _doDisposeAsync() async {
    if (_disposed) return;
    _disposed = true;
    final IsolateInterpreter? iso = _iso;
    if (iso != null) await iso.close();
    _itp.close();
    _delegate?.delete();
    _delegate = null;
  }
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
      x = clamp01(x);
      y = clamp01(y);
    }
    out.add([x, y, z]);
  }
  return out;
}

/// Weighted NMS over [Detection] objects using [weightedNms] from flutter_litert.
///
/// Pre-filters by [scoreThresh] (>= threshold), then delegates to
/// [weightedNms] which uses a strict IoU comparison (> [iouThresh]).
/// Note: the old local NMS used >= iouThresh; this is a one-ULP difference
/// unlikely to affect results in practice.
///
/// Keypoints from the highest-scoring detection in each cluster are preserved
/// via [r.index], which refers to the index in the pre-filtered sorted list.
List<Detection> _weightedNmsDetections(
  List<Detection> dets,
  double iouThresh,
  double scoreThresh, {
  int maxDetections = 100,
}) {
  final List<Detection> filtered = dets
      .where((d) => d.score >= scoreThresh)
      .toList()
    ..sort((a, b) => b.score.compareTo(a.score));
  if (filtered.isEmpty) return const <Detection>[];

  final boxes = filtered
      .map((d) => [
            d.boundingBox.xmin,
            d.boundingBox.ymin,
            d.boundingBox.xmax,
            d.boundingBox.ymax
          ])
      .toList();
  final scores = filtered.map((d) => d.score).toList();

  final results =
      weightedNms(boxes, scores, iouThres: iouThresh, maxDet: maxDetections);

  return results.map((r) {
    final Detection src = filtered[r.index];
    return Detection(
      boundingBox: RectF(r.box[0], r.box[1], r.box[2], r.box[3]),
      score: r.score,
      keypointsXY: src.keypointsXY,
    );
  }).toList();
}

SSDAnchorOptions _optsFor(FaceDetectionModel m) {
  switch (m) {
    case FaceDetectionModel.frontCamera:
      return _ssdFront;
    case FaceDetectionModel.backCamera:
      return _ssdBack;
    case FaceDetectionModel.shortRange:
      return _ssdFront;
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
@visibleForTesting
RectF faceDetectionToRoi(RectF boundingBox, {double expandFraction = 0.6}) {
  final RectF e = boundingBox.expand(expandFraction);
  final double cx = (e.xmin + e.xmax) * 0.5;
  final double cy = (e.ymin + e.ymax) * 0.5;
  final double s = math.max(e.w, e.h) * 0.5;
  return RectF(cx - s, cy - s, cx + s, cy + s);
}

/// Test-only access to [clip] for verifying value clamping behavior.
///
/// This function exposes [clip] for unit testing.
/// It clamps [v] to the range [[lo], [hi]].
@visibleForTesting
double testClip(double v, double lo, double hi) => clip(v, lo, hi);

@visibleForTesting
double testSigmoidClipped(double x, {double limit = _rawScoreLimit}) =>
    sigmoidClipped(x, limit: limit);

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
  double scoreThresh,
) =>
    _weightedNmsDetections(dets, iouThresh, scoreThresh);

@visibleForTesting
Float32List testSsdGenerateAnchors(SSDAnchorOptions opts) {
  final anchors = generateAnchors(opts);
  final result = Float32List(anchors.length * 2);
  for (int i = 0; i < anchors.length; i++) {
    result[i * 2] = anchors[i][0];
    result[i * 2 + 1] = anchors[i][1];
  }
  return result;
}

@visibleForTesting
SSDAnchorOptions testOptsFor(FaceDetectionModel m) => _optsFor(m);

@visibleForTesting
String testNameFor(FaceDetectionModel m) => _nameFor(m);

/// Converts a cv.Mat image to a normalized tensor with letterboxing.
///
/// This function performs aspect-preserving resize with black padding
/// and normalizes pixel values to the `[-1.0, 1.0]` range expected by
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

  final LetterboxParams lbp = computeLetterboxParams(
    srcWidth: inW,
    srcHeight: inH,
    targetWidth: outW,
    targetHeight: outH,
  );

  final cv.Mat resized = cv.resize(
    src,
    (lbp.newWidth, lbp.newHeight),
    interpolation: cv.INTER_LINEAR,
  );

  final cv.Mat padded = cv.copyMakeBorder(
    resized,
    lbp.padTop,
    lbp.padBottom,
    lbp.padLeft,
    lbp.padRight,
    cv.BORDER_CONSTANT,
    value: cv.Scalar.black,
  );
  resized.dispose();

  final Float32List tensor = bgrBytesToSignedFloat32(
    bytes: padded.data,
    totalPixels: outW * outH,
    buffer: buffer,
  );
  padded.dispose();

  final double padTopNorm = lbp.padTop / outH;
  final double padBottomNorm = lbp.padBottom / outH;
  final double padLeftNorm = lbp.padLeft / outW;
  final double padRightNorm = lbp.padRight / outW;

  return ImageTensor(
    tensor,
    [padTopNorm, padBottomNorm, padLeftNorm, padRightNorm],
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
@visibleForTesting
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

/// Shared interpreter setup used by [FaceLandmark], [IrisLandmark], and
/// [FaceEmbedding] factory methods.
///
/// Creates [InterpreterOptions] (via [InterpreterFactory] unless [options] is
/// provided), loads the interpreter with [load], reads the NHWC input shape,
/// resizes the input tensor, and calls [allocateTensors].
///
/// Returns a record of `(interpreter, inputWidth, inputHeight, delegate)`.
Future<(Interpreter, int, int, Delegate?)> _loadModelInterpreter({
  required FutureOr<Interpreter> Function(InterpreterOptions) load,
  InterpreterOptions? options,
  PerformanceConfig? performanceConfig,
}) async {
  Delegate? delegate;
  final InterpreterOptions opts;
  if (options != null) {
    opts = options;
  } else {
    final result = InterpreterFactory.create(performanceConfig);
    opts = result.$1;
    delegate = result.$2;
  }

  final Interpreter itp = await load(opts);
  final List<int> ishape = itp.getInputTensor(0).shape;
  final int inH = ishape[1];
  final int inW = ishape[2];
  itp.resizeInputTensor(0, [1, inH, inW, 3]);
  itp.allocateTensors();

  return (itp, inW, inH, delegate);
}

/// Shared factory body used by [FaceLandmark], [IrisLandmark], and
/// [FaceEmbedding]'s `_createWithLoader` methods.
///
/// Calls [_loadModelInterpreter], constructs the model via [construct],
/// sets the delegate, and calls [initTensors].
Future<T> _buildModel<T extends _TfliteModelDisposable>({
  required FutureOr<Interpreter> Function(InterpreterOptions) load,
  InterpreterOptions? options,
  PerformanceConfig? performanceConfig,
  bool useIsolateInterpreter = true,
  required T Function(Interpreter, int, int) construct,
  required Future<void> Function(T, bool) initTensors,
}) async {
  final (itp, inW, inH, delegate) = await _loadModelInterpreter(
    load: load,
    options: options,
    performanceConfig: performanceConfig,
  );
  final T obj = construct(itp, inW, inH);
  obj._delegate = delegate;
  await initTensors(obj, useIsolateInterpreter);
  return obj;
}
