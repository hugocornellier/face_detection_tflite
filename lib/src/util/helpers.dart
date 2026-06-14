part of '../native/face_native_lib.dart';

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
  for (int i = 0; ; i++) {
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

  Interpreter? get _itp;

  void _doDispose() {
    if (_disposed) return;
    _disposed = true;
    _delegate?.delete();
    _delegate = null;
    _iso?.close();
    _itp?.close();
  }

  Future<void> _doDisposeAsync() async {
    if (_disposed) return;
    _disposed = true;
    final IsolateInterpreter? iso = _iso;
    if (iso != null) await iso.close();
    _itp?.close();
    _delegate?.delete();
    _delegate = null;
  }
}

/// Logs the GPU->CPU CompiledModel fallback. Passed as the `onFallback`
/// callback to [CompiledModel.fromBufferWithGpuFallback] so the shared litert
/// helper handles the {gpu,cpu} fp32 attempt + CPU retry while preserving this
/// package's diagnostic message.
void _onGpuFallback(Object e) {
  debugPrint(
    'face_detection_tflite: GPU CompiledModel compilation failed, '
    'falling back to CPU only: $e',
  );
}

/// Returns the float32 element count for a CompiledModel tensor of
/// [byteSize] bytes, throwing if the size is not float32-aligned.
int _compiledFloatCount(int byteSize, String label) {
  if (byteSize % Float32List.bytesPerElement != 0) {
    throw StateError('$label byte size $byteSize is not float32-aligned.');
  }
  return byteSize ~/ Float32List.bytesPerElement;
}

/// Derives the square input side for a CompiledModel whose single input is
/// `[1, side, side, 3]` float32 (the shape of all landmark-style models).
int _compiledSquareInputSide(CompiledModel compiledModel, String label) {
  if (compiledModel.inputCount != 1) {
    throw UnsupportedError(
      '$label expects one input tensor; got ${compiledModel.inputCount}.',
    );
  }
  final int floats = _compiledFloatCount(
    compiledModel.inputByteSizes.single,
    '$label input[0]',
  );
  final int side = math.sqrt(floats / 3).round();
  if (side * side * 3 != floats) {
    throw UnsupportedError(
      '$label input has $floats floats; expected [1, side, side, 3].',
    );
  }
  return side;
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
  final double invSx = 1.0 / (1.0 - (pl + pr));
  final double invSy = 1.0 / (1.0 - (pt + pb));
  final double invW = 1.0 / inW;
  final double invH = 1.0 / inH;

  final int n = flat.length ~/ 3;
  return List<List<double>>.generate(n, (i) {
    final int i3 = i * 3;
    double x = (flat[i3] * invW - pl) * invSx;
    double y = (flat[i3 + 1] * invH - pt) * invSy;
    final double z = flat[i3 + 2];
    if (clamp) {
      x = clamp01(x);
      y = clamp01(y);
    }
    return <double>[x, y, z];
  }, growable: false);
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
  final List<Detection> filtered =
      dets.where((d) => d.score >= scoreThresh).toList()
        ..sort((a, b) => b.score.compareTo(a.score));
  if (filtered.isEmpty) return const <Detection>[];

  final boxes = filtered
      .map(
        (d) => [
          d.boundingBox.xmin,
          d.boundingBox.ymin,
          d.boundingBox.xmax,
          d.boundingBox.ymax,
        ],
      )
      .toList();
  final scores = filtered.map((d) => d.score).toList();

  final results = weightedNms(
    boxes,
    scores,
    iouThres: iouThresh,
    maxDet: maxDetections,
  );

  return results.map((r) {
    final Detection src = filtered[r.index];
    return Detection(
      boundingBox: RectF(r.box[0], r.box[1], r.box[2], r.box[3]),
      score: r.score,
      keypointsXY: src.keypointsXY,
    );
  }).toList();
}

/// Test-only access to [clip] for verifying value clamping behavior.
///
/// This function exposes [clip] for unit testing.
/// It clamps [v] to the range [[lo], [hi]].
@visibleForTesting
double testClip(double v, double lo, double hi) => clip(v, lo, hi);

/// Test-only: exposes [sigmoidClipped] for unit tests.
@visibleForTesting
double testSigmoidClipped(double x, {double limit = kRawScoreLimit}) =>
    sigmoidClipped(x, limit: limit);

/// Test-only: exposes the private letterbox-removal logic for unit tests.
@visibleForTesting
List<Detection> testDetectionLetterboxRemoval(
  List<Detection> dets,
  List<double> padding,
) => _detectionLetterboxRemoval(dets, padding);

/// Test-only: exposes the private landmark-unpacking logic for unit tests.
@visibleForTesting
List<List<double>> testUnpackLandmarks(
  Float32List flat,
  int inW,
  int inH,
  List<double> padding, {
  bool clamp = true,
}) => _unpackLandmarks(flat, inW, inH, padding, clamp: clamp);

/// Test-only: exposes the private weighted-NMS logic for unit tests.
@visibleForTesting
List<Detection> testNms(
  List<Detection> dets,
  double iouThresh,
  double scoreThresh,
) => _weightedNmsDetections(dets, iouThresh, scoreThresh);

/// Test-only: flattens generated SSD anchors into a `Float32List` for unit tests.
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

/// Test-only: exposes the private model-to-[SSDAnchorOptions] mapping for unit tests.
@visibleForTesting
SSDAnchorOptions testOptsFor(FaceDetectionModel m) => ssdOptionsFor(m);

/// Test-only: exposes the private model-name mapping for unit tests.
@visibleForTesting
String testNameFor(FaceDetectionModel m) => faceDetectionModelFile(m);

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

  // Skip the resize when the source already matches the target geometry
  // (crops warped directly to model input size hit this every call). A
  // non-continuous source still goes through cv.resize so the conversion
  // below always reads tightly packed rows, as it did before this fast path.
  final bool needsResize =
      inW != lbp.newWidth || inH != lbp.newHeight || !src.isContinuous;
  final cv.Mat resized = needsResize
      ? cv.resize(src, (
          lbp.newWidth,
          lbp.newHeight,
        ), interpolation: cv.INTER_LINEAR)
      : src;

  final bool needsPad =
      lbp.padTop != 0 ||
      lbp.padBottom != 0 ||
      lbp.padLeft != 0 ||
      lbp.padRight != 0;
  final cv.Mat padded = needsPad
      ? cv.copyMakeBorder(
          resized,
          lbp.padTop,
          lbp.padBottom,
          lbp.padLeft,
          lbp.padRight,
          cv.BORDER_CONSTANT,
          value: cv.Scalar.black,
        )
      : resized;
  if (needsResize && needsPad) resized.dispose();

  final Float32List tensor = bgrMatToSignedFloat32(
    padded,
    totalPixels: outW * outH,
    buffer: buffer,
  );
  if (needsResize || needsPad) padded.dispose();

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

/// Converts a continuous BGR `CV_8UC3` [cv.Mat] to a `[-1, 1]`-normalized
/// RGB float tensor using OpenCV SIMD kernels (BGR→RGB swap + scaled float
/// conversion) and a single memcpy into [buffer].
///
/// Falls back to the scalar Dart loop ([bgrBytesToSignedFloat32]) for
/// non-`CV_8UC3` or non-continuous inputs.
@visibleForTesting
Float32List bgrMatToSignedFloat32(
  cv.Mat mat, {
  required int totalPixels,
  Float32List? buffer,
}) {
  if (mat.type != cv.MatType.CV_8UC3 || !mat.isContinuous) {
    return bgrBytesToSignedFloat32(
      bytes: mat.data,
      totalPixels: totalPixels,
      buffer: buffer,
    );
  }
  final cv.Mat rgb = cv.cvtColor(mat, cv.COLOR_BGR2RGB);
  final cv.Mat f32 = rgb.convertTo(
    cv.MatType.CV_32FC3,
    alpha: 1.0 / 127.5,
    beta: -1.0,
  );
  rgb.dispose();
  final Uint8List raw = f32.data;
  final Float32List view = Float32List.view(
    raw.buffer,
    raw.offsetInBytes,
    totalPixels * 3,
  );
  final Float32List tensor = buffer ?? Float32List(totalPixels * 3);
  tensor.setAll(0, view);
  f32.dispose();
  return tensor;
}

/// Rebuilds a tightly packed [cv.Mat] from raw [bytes] with a single memcpy.
///
/// [cv.Mat.fromList] routes every byte through a lazy `cast<int>` view and
/// copies element-by-element (~8 ms for a 1080p BGR frame); allocating the
/// Mat and copying into its native buffer is two orders of magnitude faster.
/// Only valid for tightly packed (continuous) pixel data.
///
/// Throws [ArgumentError] if [bytes] does not match rows x cols x elemSize
/// for [type].
cv.Mat matFromPackedBytes(
  int rows,
  int cols,
  cv.MatType type,
  Uint8List bytes,
) {
  final cv.Mat mat = cv.Mat.create(rows: rows, cols: cols, type: type);
  final Uint8List dst = mat.data;
  if (dst.length != bytes.length) {
    final int expected = dst.length;
    mat.dispose();
    throw ArgumentError(
      'bytes.length ${bytes.length} does not match a '
      '$rows x $cols Mat of type $type ($expected bytes)',
    );
  }
  dst.setAll(0, bytes);
  return mat;
}

/// Rebuilds a [cv.Mat] from a backend-neutral packed image [layout].
cv.Mat matFromPackedLayout(
  PackedImageLayout layout,
  Uint8List bytes,
  cv.MatType type,
) {
  final cv.Mat mat = cv.Mat.create(
    rows: layout.rows,
    cols: layout.cols,
    type: type,
  );
  try {
    layout.copyTo(mat.data, bytes);
    return mat;
  } catch (_) {
    mat.dispose();
    rethrow;
  }
}

/// Decodes a [CameraFrame] into a 3-channel BGR [cv.Mat].
///
/// The layout and safe operation order come from `flutter_litert`
/// ([CameraFrame.decodePlan]); this only maps that backend-neutral plan onto
/// OpenCV primitives. Equivalent to the previous inline conversion: for
/// 4-channel frames it resizes/rotates before dropping alpha; for packed YUV it
/// colour-converts first (the packed layout can't be resized).
cv.Mat cameraFrameToBgrMat(CameraFrame frame, {int? maxDim}) {
  final CameraFrameDecodePlan plan = frame.decodePlan();
  final cv.Mat source = matFromPackedLayout(
    plan.sourceLayout,
    frame.bytes,
    plan.sourceLayout.channels == 4 ? cv.MatType.CV_8UC4 : cv.MatType.CV_8UC1,
  );

  cv.Mat maybeResize(cv.Mat m) {
    if (maxDim == null || (m.cols <= maxDim && m.rows <= maxDim)) return m;
    final double scale = maxDim / (m.cols > m.rows ? m.cols : m.rows);
    final cv.Mat resized = cv.resize(m, (
      (m.cols * scale).toInt(),
      (m.rows * scale).toInt(),
    ), interpolation: cv.INTER_LINEAR);
    m.dispose();
    return resized;
  }

  int? rotateFlag() {
    return switch (plan.rotation) {
      CameraFrameRotation.cw90 => cv.ROTATE_90_CLOCKWISE,
      CameraFrameRotation.cw180 => cv.ROTATE_180,
      CameraFrameRotation.cw270 => cv.ROTATE_90_COUNTERCLOCKWISE,
      null => null,
    };
  }

  cv.Mat maybeRotate(cv.Mat m) {
    final int? flag = rotateFlag();
    if (flag == null) return m;
    final cv.Mat rotated = cv.rotate(m, flag);
    m.dispose();
    return rotated;
  }

  final int cvtCode = switch (plan.conversion) {
    CameraFrameConversion.bgra2bgr => cv.COLOR_BGRA2BGR,
    CameraFrameConversion.rgba2bgr => cv.COLOR_RGBA2BGR,
    CameraFrameConversion.yuv2bgrNv12 => cv.COLOR_YUV2BGR_NV12,
    CameraFrameConversion.yuv2bgrNv21 => cv.COLOR_YUV2BGR_NV21,
    CameraFrameConversion.yuv2bgrI420 => cv.COLOR_YUV2BGR_I420,
  };

  switch (plan.order) {
    case CameraFrameDecodeOrder.resizeRotateThenColorConvert:
      cv.Mat current = plan.hasStridePadding
          ? source.region(cv.Rect(0, 0, plan.visibleWidth, plan.visibleHeight))
          : source;

      if (maxDim != null && (current.cols > maxDim || current.rows > maxDim)) {
        final double scale =
            maxDim /
            (current.cols > current.rows ? current.cols : current.rows);
        final cv.Mat resized = cv.resize(current, (
          (current.cols * scale).toInt(),
          (current.rows * scale).toInt(),
        ), interpolation: cv.INTER_LINEAR);
        if (!identical(current, source)) current.dispose();
        current = resized;
      }

      final int? flag = rotateFlag();
      if (flag != null) {
        final cv.Mat rotated = cv.rotate(current, flag);
        if (!identical(current, source)) current.dispose();
        current = rotated;
      }

      final cv.Mat bgr = cv.cvtColor(current, cvtCode);
      if (!identical(current, source)) current.dispose();
      source.dispose();
      return bgr;

    case CameraFrameDecodeOrder.colorConvertThenResizeRotate:
      cv.Mat current = cv.cvtColor(source, cvtCode);
      source.dispose();
      current = maybeResize(current);
      current = maybeRotate(current);
      return current;
  }
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
/// - [size]: Side length of the square region in source pixels
/// - [theta]: Rotation angle in radians (positive = counter-clockwise)
/// - [outSize]: Optional output side length in pixels. When set, the warp
///   scales the `size`-by-`size` source region directly to
///   `outSize`-by-`outSize` in a single resample, equivalent to cropping at
///   `size` and then `cv.resize`-ing to `outSize` (the scaled matrix uses the
///   same pixel-center alignment as cv.resize). Use this to extract crops at
///   a model's input resolution without paying for a full-size warp plus a
///   second resample.
///
/// Returns the cropped and rotated cv.Mat. Caller must dispose.
/// Returns null if size is invalid.
cv.Mat? extractAlignedSquare(
  cv.Mat src,
  double cx,
  double cy,
  double size,
  double theta, {
  int? outSize,
}) {
  final int sizeInt = size.round();
  if (sizeInt <= 0) return null;
  final int dstSize = outSize ?? sizeInt;
  final double scale = dstSize / sizeInt;

  final double angleDegrees = -theta * 180.0 / math.pi;

  final cv.Mat rotMat = cv.getRotationMatrix2D(
    cv.Point2f(cx, cy),
    angleDegrees,
    scale,
  );

  // Pixel-center alignment matching crop-then-cv.resize: a resize by factor
  // s samples src((x + 0.5) / s - 0.5), so the source center must land at
  // dstSize / 2 + 0.5 * (s - 1) rather than dstSize / 2. Reduces to the
  // plain crop placement when outSize is null (s == 1).
  final double outCenter = dstSize / 2.0 + 0.5 * (scale - 1.0);

  final double tx = rotMat.at<double>(0, 2) + outCenter - cx;
  final double ty = rotMat.at<double>(1, 2) + outCenter - cy;
  rotMat.set<double>(0, 2, tx);
  rotMat.set<double>(1, 2, ty);

  final cv.Mat output = cv.warpAffine(
    src,
    rotMat,
    (dstSize, dstSize),
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
