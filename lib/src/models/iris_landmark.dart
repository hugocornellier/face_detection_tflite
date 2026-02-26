part of '../../face_detection_tflite.dart';

/// Transforms normalized iris landmarks to absolute pixel coordinates
/// using the alignment parameters from an [AlignedRoi].
///
/// When [isRight] is true, the x-coordinate is flipped (mirrored) before
/// transforming, which undoes the horizontal flip applied to right eye crops.
List<List<double>> _transformIrisToAbsolute(
  List<List<double>> lmNorm,
  AlignedRoi roi,
  bool isRight,
) {
  final double ct = math.cos(roi.theta);
  final double st = math.sin(roi.theta);
  final double s = roi.size;
  final List<List<double>> out = <List<double>>[];
  for (final List<double> p in lmNorm) {
    final double px = isRight ? (1.0 - p[0]) : p[0];
    final double lx2 = (px - 0.5) * s;
    final double ly2 = (p[1] - 0.5) * s;
    out.add([roi.cx + lx2 * ct - ly2 * st, roi.cy + lx2 * st + ly2 * ct, p[2]]);
  }
  return out;
}

/// Estimates dense iris keypoints within cropped eye regions and lets callers
/// derive a robust iris center (with fallback if inference fails).
///
/// The underlying TFLite model (`iris_landmark.tflite`) is sourced from Google's MediaPipe
/// framework. See the official model card for architecture details, training data, and
/// intended use cases: https://mediapipe.page.link/iris-mc
/// (local copy: `doc/model_cards/iris_landmark_model_card.pdf`)
class IrisLandmark {
  IsolateInterpreter? _iso;
  final Interpreter _itp;
  final int _inW, _inH;
  Delegate? _delegate;
  late final Tensor _inputTensor;
  late final Float32List _inputBuf;
  late final Map<int, List<int>> _outShapes;
  late final Map<int, Float32List> _outBuffers;
  late final List<List<List<List<double>>>> _input4dCache;
  late final Map<int, Object> _outputsCache;

  IrisLandmark._(this._itp, this._inW, this._inH);

  /// Creates and initializes an iris landmark model instance.
  ///
  /// This factory method loads the iris landmark TensorFlow Lite model from
  /// package assets and prepares it for inference. The model predicts 5 keypoints
  /// per iris plus eye contour points.
  ///
  /// The [options] parameter allows you to customize the TFLite interpreter
  /// configuration (e.g., number of threads, use of GPU delegate).
  ///
  /// The [performanceConfig] parameter enables hardware acceleration delegates.
  /// Use [PerformanceConfig.xnnpack()] for 2-5x speedup on CPU. If both [options]
  /// and [performanceConfig] are provided, [options] takes precedence.
  ///
  /// Returns a fully initialized [IrisLandmark] instance ready to detect irises.
  ///
  /// **Note:** This model expects a cropped eye region as input. For full pipeline
  /// processing, use the high-level [FaceDetector] class with [FaceDetectionMode.full].
  ///
  /// Example:
  /// ```dart
  /// // Default (no acceleration)
  /// final irisModel = await IrisLandmark.create();
  /// final irisPoints = await irisModel.call(eyeCropMat);
  ///
  /// // With XNNPACK acceleration
  /// final irisModel = await IrisLandmark.create(
  ///   performanceConfig: PerformanceConfig.xnnpack(),
  /// );
  /// ```
  ///
  /// See also:
  /// - [createFromFile] for loading a model from a custom file path
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<IrisLandmark> create({
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
  }) =>
      _createWithLoader(
        load: (opts) => Interpreter.fromAsset(
          'packages/face_detection_tflite/assets/models/$_irisLandmarkModel',
          options: opts,
        ),
        options: options,
        performanceConfig: performanceConfig,
      );

  /// Creates an iris landmark model from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetectorIsolate] to initialize models
  /// in a background isolate where asset loading is not available.
  ///
  /// The [modelBytes] parameter should contain the raw TFLite model file contents.
  static Future<IrisLandmark> createFromBuffer(
    Uint8List modelBytes, {
    PerformanceConfig? performanceConfig,
  }) =>
      _createWithLoader(
        load: (opts) => Interpreter.fromBuffer(modelBytes, options: opts),
        performanceConfig: performanceConfig,
        useIsolateInterpreter: false,
      );

  /// Shared tensor initialization logic.
  ///
  /// When [useIsolateInterpreter] is false, inference runs directly via
  /// `_itp.invoke()` instead of spawning a nested isolate. This should be
  /// used when the model is already running inside a background isolate.
  Future<void> _initializeTensors({bool useIsolateInterpreter = true}) async {
    _inputTensor = _itp.getInputTensor(0);
    _inputBuf = _inputTensor.data.buffer.asFloat32List();

    final Map<int, OutputTensorInfo> outputInfo = collectOutputTensorInfo(_itp);
    _outShapes = outputInfo.map(
      (int k, OutputTensorInfo v) => MapEntry(k, v.shape),
    );
    _outBuffers = outputInfo.map(
      (int k, OutputTensorInfo v) => MapEntry(k, v.buffer),
    );
    _input4dCache = createNHWCTensor4D(_inH, _inW);

    _outputsCache = <int, Object>{};
    _outShapes.forEach((i, shape) {
      _outputsCache[i] = allocTensorShape(shape);
    });

    if (useIsolateInterpreter && _delegate == null) {
      _iso = await IsolateInterpreter.create(address: _itp.address);
    }
  }

  /// Creates and initializes an iris landmark model from a custom file path.
  ///
  /// This factory method loads a TensorFlow Lite model from the specified
  /// [modelPath] on the filesystem instead of from package assets. This is
  /// useful for advanced users who want to use custom-trained or alternative
  /// iris tracking models.
  ///
  /// The [options] parameter allows you to customize the TFLite interpreter
  /// configuration (e.g., number of threads, use of GPU delegate).
  ///
  /// The [performanceConfig] parameter enables hardware acceleration delegates.
  /// Use [PerformanceConfig.xnnpack()] for 2-5x speedup on CPU. If both [options]
  /// and [performanceConfig] are provided, [options] takes precedence.
  ///
  /// Returns a fully initialized [IrisLandmark] instance ready to detect irises.
  ///
  /// Example:
  /// ```dart
  /// // Default (no acceleration)
  /// final customModel = await IrisLandmark.createFromFile(
  ///   '/path/to/custom_iris_model.tflite',
  /// );
  /// final irisPoints = await customModel(eyeCropImage);
  /// customModel.dispose(); // Clean up when done
  ///
  /// // With XNNPACK acceleration
  /// final customModel = await IrisLandmark.createFromFile(
  ///   '/path/to/custom_iris_model.tflite',
  ///   performanceConfig: PerformanceConfig.xnnpack(),
  /// );
  /// ```
  ///
  /// See also:
  /// - [create] for loading the default bundled model from assets
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<IrisLandmark> createFromFile(
    String modelPath, {
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
  }) =>
      _createWithLoader(
        load: (opts) => Interpreter.fromFile(File(modelPath), options: opts),
        options: options,
        performanceConfig: performanceConfig,
      );

  /// Shared factory logic for [create], [createFromBuffer], and [createFromFile].
  static Future<IrisLandmark> _createWithLoader({
    required FutureOr<Interpreter> Function(InterpreterOptions) load,
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
    bool useIsolateInterpreter = true,
  }) async {
    Delegate? delegate;
    final InterpreterOptions opts;
    if (options != null) {
      opts = options;
    } else {
      final result = _createInterpreterOptions(performanceConfig);
      opts = result.$1;
      delegate = result.$2;
    }

    final Interpreter itp = await load(opts);
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final IrisLandmark obj = IrisLandmark._(itp, inW, inH);
    obj._delegate = delegate;
    await obj._initializeTensors(useIsolateInterpreter: useIsolateInterpreter);
    return obj;
  }

  /// Runs iris detection in a separate isolate for non-blocking inference.
  ///
  /// This static method spawns a dedicated isolate to perform iris landmark
  /// detection on encoded eye crop image bytes. This is useful for running
  /// iris detection without blocking the main UI thread, especially for
  /// one-off detections or background processing.
  ///
  /// The [eyeCropBytes] parameter should contain encoded image data (JPEG, PNG)
  /// of a cropped eye region.
  ///
  /// The [modelPath] parameter specifies the filesystem path to the iris model
  /// (.tflite file).
  ///
  /// Returns a list of 3D landmark points in normalized coordinates (0.0 to 1.0)
  /// relative to the eye crop, where each point is `[x, y, z]`.
  ///
  /// **Performance:** Creates a new isolate for each call. For repeated detections,
  /// prefer creating a long-lived [IrisLandmark] instance.
  ///
  /// Example:
  /// ```dart
  /// final irisPoints = await IrisLandmark.callWithIsolate(
  ///   eyeCropBytes,
  ///   '/path/to/iris_landmark.tflite',
  /// );
  /// ```
  ///
  /// Throws [StateError] if the model cannot be loaded or inference fails.
  ///
  /// See also:
  /// - [create] for persistent isolate inference
  /// - [call] for the instance method alternative
  static Future<List<List<double>>> callWithIsolate(
    Uint8List eyeCropBytes,
    String modelPath,
  ) async {
    final ReceivePort rp = ReceivePort();
    final Isolate iso = await Isolate.spawn(IrisLandmark._isolateEntry, {
      'sendPort': rp.sendPort,
      'modelPath': modelPath,
      'eyeCropBytes': eyeCropBytes,
    });
    final Map<dynamic, dynamic> msg = await rp.first as Map;
    rp.close();
    iso.kill(priority: Isolate.immediate);
    if (msg['ok'] == true) {
      final List pts = msg['points'] as List;
      return pts
          .map<List<double>>(
            (e) => (e as List).map((n) => (n as num).toDouble()).toList(),
          )
          .toList();
    } else {
      throw StateError(msg['err'] as String);
    }
  }

  @pragma('vm:entry-point')
  static Future<void> _isolateEntry(Map<String, dynamic> params) async {
    final SendPort sendPort = params['sendPort'] as SendPort;
    final String modelPath = params['modelPath'] as String;
    final Uint8List eyeCropBytes = params['eyeCropBytes'] as Uint8List;

    try {
      final IrisLandmark iris = await IrisLandmark.createFromFile(modelPath);
      final cv.Mat eye = cv.imdecode(eyeCropBytes, cv.IMREAD_COLOR);
      if (eye.isEmpty) {
        sendPort.send({'ok': false, 'err': 'decode_failed'});
        return;
      }
      final List<List<double>> res = await iris.call(eye);
      eye.dispose();
      iris.dispose();
      sendPort.send({'ok': true, 'points': res});
    } catch (e) {
      sendPort.send({'ok': false, 'err': e.toString()});
    }
  }

  /// Predicts iris and eye contour landmarks from a cv.Mat eye crop.
  ///
  /// Accepts a cv.Mat directly, providing better performance by avoiding
  /// image format conversions.
  ///
  /// The [eyeCrop] parameter should contain a tight crop around a single eye as cv.Mat.
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
  ///
  /// The optional [buffer] parameter allows reusing a pre-allocated Float32List
  /// for the tensor conversion to reduce GC pressure.
  ///
  /// Returns a list of 3D landmark points in normalized coordinates.
  ///
  /// Example:
  /// ```dart
  /// final eyeCropMat = cv.imdecode(bytes, cv.IMREAD_COLOR);
  /// final irisPoints = await irisLandmark.call(eyeCropMat);
  /// eyeCropMat.dispose();
  /// ```
  Future<List<List<double>>> call(cv.Mat eyeCrop, {Float32List? buffer}) async {
    final ImageTensor pack = convertImageToTensor(
      eyeCrop,
      outW: _inW,
      outH: _inH,
      buffer: buffer,
    );
    return _inferAndUnpack(pack);
  }

  /// Runs inference on pre-computed tensor and unpacks all landmarks.
  Future<List<List<double>>> _inferAndUnpack(ImageTensor pack) async {
    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();

      final List<List<double>> lm = <List<double>>[];
      for (final Float32List flat in _outBuffers.values) {
        lm.addAll(
          _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false),
        );
      }
      return lm;
    } else {
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [_input4dCache];
      await _iso!.runForMultipleInputs(inputs, _outputsCache);

      final List<List<double>> lm = <List<double>>[];
      _outShapes.forEach((i, _) {
        final Float32List flat = flattenDynamicTensor(_outputsCache[i]);
        lm.addAll(
          _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false),
        );
      });
      return lm;
    }
  }

  /// Runs iris detection on a cv.Mat using an aligned eye ROI.
  ///
  /// Uses SIMD-accelerated warpAffine for the rotation crop, providing 10-50x
  /// better performance than pure Dart bilinear interpolation.
  ///
  /// The [src] parameter is the full image as cv.Mat.
  /// The [roi] parameter defines the eye region with center, size, and rotation.
  /// When [isRight] is true, the eye crop is flipped before processing.
  ///
  /// Returns iris landmarks in absolute pixel coordinates.
  ///
  /// Note: The input cv.Mat is NOT disposed by this method.
  Future<List<List<double>>> runOnImageAlignedIris(
    cv.Mat src,
    AlignedRoi roi, {
    bool isRight = false,
    Float32List? buffer,
  }) async {
    final cv.Mat? crop = extractAlignedSquare(
      src,
      roi.cx,
      roi.cy,
      roi.size,
      roi.theta,
    );
    if (crop == null) {
      return const <List<double>>[];
    }

    cv.Mat eye;
    if (isRight) {
      eye = cv.flip(crop, 1);
      crop.dispose();
    } else {
      eye = crop;
    }

    final List<List<double>> lmNorm = await call(eye, buffer: buffer);
    eye.dispose();
    return _transformIrisToAbsolute(lmNorm, roi, isRight);
  }

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the iris landmark model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// **Note:** Most users should call [FaceDetector.dispose] instead, which
  /// automatically disposes all internal models (detection, mesh, and iris).
  void dispose() {
    _delegate?.delete();
    _delegate = null;
    _iso?.close();
    _itp.close();
  }

  /// Creates interpreter options with delegates based on performance configuration.
  ///
  /// Delegates to [FaceDetection._createInterpreterOptions] for consistent
  /// platform-aware delegate selection across all model types.
  static (InterpreterOptions, Delegate?) _createInterpreterOptions(
    PerformanceConfig? config,
  ) {
    return FaceDetection._createInterpreterOptions(config);
  }
}

@visibleForTesting
List<List<double>> testTransformIrisToAbsolute(
  List<List<double>> lmNorm,
  AlignedRoi roi,
  bool isRight,
) =>
    _transformIrisToAbsolute(lmNorm, roi, isRight);
