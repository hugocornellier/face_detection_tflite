part of '../native/face_native_lib.dart';

/// Runs face box detection and predicts a small set of facial keypoints
/// (eyes, nose, mouth, tragions) on the detected face(s).
///
/// The underlying TFLite models are sourced from Google's MediaPipe framework
/// (BlazeFace). See the official model cards for architecture details, training
/// data, and intended use cases:
/// - Front camera / short range: https://mediapipe.page.link/blazeface-mc
///   (local copy: `doc/model_cards/blazeface_short_range_model_card.pdf`)
/// - Back camera / full range: https://mediapipe.page.link/blazeface-back-mc
///   (local copy: `doc/model_cards/blazeface_full_range_model_card.pdf`)
/// - Full range sparse: https://mediapipe.page.link/blazeface-back-sparse-mc
///   (local copy: `doc/model_cards/blazeface_full_range_sparse_model_card.pdf`)
class FaceDetection {
  final Interpreter? _itp;
  final CompiledModel? _compiledModel;
  final int _inW, _inH;
  final int _boundingBoxIndex = 0, _scoreIndex = 1;
  final List<List<double>> _anchors;
  IsolateInterpreter? _iso;
  Delegate? _delegate;
  bool _disposed = false;
  late final int _inputIdx;
  late final List<int> _boxesShape;
  late final List<int> _scoresShape;
  late final TensorFloat32Views _views;
  late final List<List<List<List<double>>>> _input4dCache;
  late final List<List<List<double>>> _boxesOutCache;
  late final Object _scoresOutCache;

  FaceDetection._interpreter(
    Interpreter itp,
    this._inW,
    this._inH,
    this._anchors,
  ) : _itp = itp,
      _compiledModel = null;

  FaceDetection._compiled(
    CompiledModel compiledModel,
    this._inW,
    this._inH,
    this._anchors,
  ) : _itp = null,
      _compiledModel = compiledModel;

  /// The model input width in pixels.
  int get inputWidth => _inW;

  /// The model input height in pixels.
  int get inputHeight => _inH;

  /// Creates and initializes a face detection model instance.
  ///
  /// This factory method loads the specified TensorFlow Lite [model] from package
  /// assets and prepares it for inference. Different model variants are optimized
  /// for different use cases (see [FaceDetectionModel] for details).
  ///
  /// The [options] parameter allows you to customize the TFLite interpreter
  /// configuration (e.g., number of threads, use of GPU delegate).
  ///
  /// The [performanceConfig] parameter enables hardware acceleration delegates.
  /// Use [PerformanceConfig.xnnpack()] for 2-5x speedup on CPU. If both [options]
  /// and [performanceConfig] are provided, [options] takes precedence.
  ///
  /// Returns a fully initialized [FaceDetection] instance ready to detect faces.
  ///
  /// **Note:** Most users should use the high-level [FaceDetector] class instead
  /// of working with this low-level API directly.
  ///
  /// Example:
  /// ```dart
  /// // Default (auto mode)
  /// final detector = await FaceDetection.create(
  ///   FaceDetectionModel.frontCamera,
  /// );
  ///
  /// // With XNNPACK acceleration
  /// final detector = await FaceDetection.create(
  ///   FaceDetectionModel.backCamera,
  ///   performanceConfig: PerformanceConfig.xnnpack(),
  /// );
  /// ```
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<FaceDetection> create(
    FaceDetectionModel model, {
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
  }) => _createWithLoader(
    model: model,
    load: (opts) => Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/${faceDetectionModelFile(model)}',
      options: opts,
    ),
    options: options,
    performanceConfig: performanceConfig,
  );

  /// Creates a face detection model from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetector] to initialize models
  /// in a background isolate where asset loading is not available.
  ///
  /// The [modelBytes] parameter should contain the raw TFLite model file contents.
  /// The [model] parameter specifies which model variant this is (for anchor generation).
  static Future<FaceDetection> createFromBuffer(
    Uint8List modelBytes,
    FaceDetectionModel model, {
    PerformanceConfig? performanceConfig,
  }) => _createWithLoader(
    model: model,
    load: (opts) => Interpreter.fromBuffer(modelBytes, options: opts),
    performanceConfig: performanceConfig,
    useIsolateInterpreter: false,
  );

  /// Creates a face detection model backed by LiteRT CompiledModel.
  static Future<FaceDetection> createCompiledFromBuffer(
    Uint8List modelBytes,
    FaceDetectionModel model, {
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    if (model == FaceDetectionModel.fullSparse) {
      // Upstream LiteRT bug (reproduced in Google's own Python API): GPU
      // compilation of this model's DENSIFY op aborts the process with an
      // uncatchable SIGABRT, even with CPU fallback in the accelerator set.
      throw UnsupportedError(
        'FaceDetectionModel.fullSparse is not supported with the '
        'CompiledModel engine; use the Interpreter engine for this model.',
      );
    }
    final SSDAnchorOptions opts = ssdOptionsFor(model);
    final int inW = opts.inputSizeWidth;
    final int inH = opts.inputSizeHeight;
    final List<List<double>> anchors = generateAnchors(opts);

    final CompiledModel compiledModel = _isDefaultAccelerators(accelerators)
        ? CompiledModel.fromBufferWithGpuFallback(
            modelBytes,
            precision: precision,
            onFallback: _onGpuFallback,
          )
        : CompiledModel.fromBuffer(
            modelBytes,
            accelerators: accelerators,
            precision: precision,
          );
    final obj = FaceDetection._compiled(compiledModel, inW, inH, anchors);
    try {
      obj._initializeCompiledModel();
      return obj;
    } catch (_) {
      obj.dispose();
      rethrow;
    }
  }

  static Future<FaceDetection> _createWithLoader({
    required FaceDetectionModel model,
    required FutureOr<Interpreter> Function(InterpreterOptions) load,
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
    bool useIsolateInterpreter = true,
  }) async {
    final SSDAnchorOptions opts = ssdOptionsFor(model);
    final int inW = opts.inputSizeWidth;
    final int inH = opts.inputSizeHeight;

    Delegate? delegate;
    final InterpreterOptions interpreterOptions;
    if (options != null) {
      interpreterOptions = options;
    } else {
      final result = InterpreterFactory.create(performanceConfig);
      interpreterOptions = result.$1;
      delegate = result.$2;
    }

    final Interpreter itp = await load(interpreterOptions);
    final List<List<double>> anchors = generateAnchors(opts);
    final FaceDetection obj = FaceDetection._interpreter(
      itp,
      inW,
      inH,
      anchors,
    );
    obj._delegate = delegate;

    await obj._initializeInterpreterTensors(
      useIsolateInterpreter: useIsolateInterpreter,
    );
    return obj;
  }

  /// Shared tensor initialization logic.
  ///
  /// When [useIsolateInterpreter] is false, inference runs directly via
  /// `_itp.invoke()` instead of spawning a nested isolate. This should be
  /// used when the model is already running inside a background isolate.
  Future<void> _initializeInterpreterTensors({
    bool useIsolateInterpreter = true,
  }) async {
    final Interpreter itp = _requireInterpreter();
    int foundIdx = -1;
    for (int i = 0; i < 10; i++) {
      try {
        final List<int> s = itp.getInputTensor(i).shape;
        if (s.length == 4 && s.last == 3) {
          foundIdx = i;
          break;
        }
      } catch (_) {
        break;
      }
    }
    if (foundIdx == -1) {
      itp.close();
      throw StateError(
        'No valid input tensor found with shape [batch, height, width, 3]',
      );
    }
    _inputIdx = foundIdx;

    itp.resizeInputTensor(_inputIdx, [1, _inH, _inW, 3]);
    itp.allocateTensors();

    _boxesShape = itp.getOutputTensor(_boundingBoxIndex).shape;
    _scoresShape = itp.getOutputTensor(_scoreIndex).shape;
    _views = TensorFloat32Views.capture(itp);

    _input4dCache = createNHWCTensor4D(_inH, _inW);

    final int b0 = _boxesShape[0], b1 = _boxesShape[1], b2 = _boxesShape[2];
    _boxesOutCache = List.generate(
      b0,
      (_) => List.generate(
        b1,
        (_) => List<double>.filled(b2, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    if (_scoresShape.length == 3) {
      final int s0 = _scoresShape[0],
          s1 = _scoresShape[1],
          s2 = _scoresShape[2];
      _scoresOutCache = List.generate(
        s0,
        (_) => List.generate(
          s1,
          (_) => List<double>.filled(s2, 0.0, growable: false),
          growable: false,
        ),
        growable: false,
      );
    } else {
      final int s0 = _scoresShape[0], s1 = _scoresShape[1];
      _scoresOutCache = List.generate(
        s0,
        (_) => List<double>.filled(s1, 0.0, growable: false),
        growable: false,
      );
    }

    if (useIsolateInterpreter) {
      _iso = await InterpreterFactory.createIsolateIfNeeded(itp, _delegate);
    }
  }

  void _initializeCompiledModel() {
    final CompiledModel compiledModel = _requireCompiledModel();
    if (compiledModel.inputCount != 1) {
      throw UnsupportedError(
        'Compiled face detection expects one input tensor; got '
        '${compiledModel.inputCount}.',
      );
    }
    if (compiledModel.outputCount <= _scoreIndex) {
      throw UnsupportedError(
        'Compiled face detection expects at least two outputs; got '
        '${compiledModel.outputCount}.',
      );
    }

    final int inputFloats = _compiledFloatCount(
      compiledModel.inputByteSizes.single,
      'input[0]',
    );
    final int expectedInputFloats = _inH * _inW * 3;
    if (inputFloats != expectedInputFloats) {
      throw UnsupportedError(
        'Compiled face detection input has $inputFloats floats; expected '
        '$expectedInputFloats for [1, $_inH, $_inW, 3].',
      );
    }

    final int anchorCount = _anchors.length;
    final int boxFloats = _compiledFloatCount(
      compiledModel.outputByteSizes[_boundingBoxIndex],
      'output[$_boundingBoxIndex]',
    );
    final int scoreFloats = _compiledFloatCount(
      compiledModel.outputByteSizes[_scoreIndex],
      'output[$_scoreIndex]',
    );

    if (boxFloats % anchorCount != 0) {
      throw UnsupportedError(
        'Compiled face detection boxes output has $boxFloats floats, which '
        'does not align with $anchorCount anchors.',
      );
    }
    final int boxValues = boxFloats ~/ anchorCount;
    if (boxValues < 4 || boxValues.isOdd) {
      throw UnsupportedError(
        'Compiled face detection boxes output has $boxValues values per '
        'anchor; expected box coordinates plus keypoint pairs.',
      );
    }

    if (scoreFloats % anchorCount != 0) {
      throw UnsupportedError(
        'Compiled face detection scores output has $scoreFloats floats, which '
        'does not align with $anchorCount anchors.',
      );
    }
    final int scoreValues = scoreFloats ~/ anchorCount;
    if (scoreValues < 1) {
      throw UnsupportedError('Compiled face detection scores output is empty.');
    }

    _inputIdx = 0;
    _boxesShape = [1, anchorCount, boxValues];
    _scoresShape = scoreValues == 1
        ? [1, anchorCount]
        : [1, anchorCount, scoreValues];
  }

  /// Runs face detection on a pre-computed tensor.
  ///
  /// This is used by the OpenCV pipeline which computes the tensor directly
  /// from cv.Mat without going through the isolate worker.
  Future<List<Detection>> callWithTensor(ImageTensor pack) async {
    return _runInference(pack);
  }

  /// Runs the actual inference on a tensor.
  Future<List<Detection>> _runInference(ImageTensor pack) async {
    Float32List boxesBuf;
    Float32List scoresBuf;

    final CompiledModel? compiledModel = _compiledModel;
    if (compiledModel != null) {
      // Copying runAsync is the official LiteRT pattern for host-side data
      // (the C++ Write/Read API is lock+memcpy+unlock); the Metal accelerator
      // only supports MetalBufferPacked tensor buffers, so host zero-copy is
      // not available on the GPU path.
      final List<Float32List> outputs = await compiledModel.runAsync([
        pack.tensorNHWC,
      ]);
      return _postprocess(
        outputs[_boundingBoxIndex],
        outputs[_scoreIndex],
        pack.padding,
      );
    } else if (_iso != null) {
      final Interpreter itp = _requireInterpreter();
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final int inputCount = itp.getInputTensors().length;
      final List<Object?> inputs = List<Object?>.filled(
        inputCount,
        null,
        growable: false,
      );
      inputs[_inputIdx] = _input4dCache;

      final Map<int, Object> outputs = <int, Object>{
        _boundingBoxIndex: _boxesOutCache,
        _scoreIndex: _scoresOutCache,
      };

      await _iso!.runForMultipleInputs(inputs.cast<Object>(), outputs);

      final Float32List outBoxes = flattenDynamicTensor(_boxesOutCache);
      final Float32List outScores = flattenDynamicTensor(_scoresOutCache);

      boxesBuf = outBoxes;
      scoresBuf = outScores;
    } else {
      final Interpreter itp = _requireInterpreter();
      _views.inputs[_inputIdx].setAll(0, pack.tensorNHWC);
      itp.invoke();
      boxesBuf = _views.outputs[_boundingBoxIndex];
      scoresBuf = _views.outputs[_scoreIndex];
    }

    return _postprocess(boxesBuf, scoresBuf, pack.padding);
  }

  /// Decodes raw box/score buffers into NMS-pruned, letterbox-corrected
  /// detections.
  List<Detection> _postprocess(
    Float32List boxesBuf,
    Float32List scoresBuf,
    List<double> padding,
  ) {
    final (:candidateIndices, :candidateScores) = _collectCandidateScores(
      scoresBuf,
      _scoresShape,
    );

    final List<DecodedBox> boxes = _decodeBoxesForIndices(
      boxesBuf,
      _boxesShape,
      candidateIndices,
    );

    final List<Detection> dets = _toDetectionsFiltered(boxes, candidateScores);

    final List<Detection> pruned = _weightedNmsDetections(
      dets,
      kMinSuppressionThreshold,
      kMinScore,
    );
    return _detectionLetterboxRemoval(pruned, padding);
  }

  /// Decodes boxes only for the specified anchor indices.
  ///
  /// This is an optimized variant that skips anchors with low scores,
  /// reducing unnecessary computation by ~17x for typical images.
  List<DecodedBox> _decodeBoxesForIndices(
    Float32List raw,
    List<int> shape,
    List<int> indices,
  ) {
    final int k = shape[2];
    final double scale = _inH.toDouble();
    final List<DecodedBox> out = <DecodedBox>[];
    final Float32List tmp = Float32List(k);

    for (final int i in indices) {
      final int base = i * k;
      for (int j = 0; j < k; j++) {
        tmp[j] = raw[base + j] / scale;
      }
      final double ax = _anchors[i][0];
      final double ay = _anchors[i][1];
      tmp[0] += ax;
      tmp[1] += ay;
      for (int j = 4; j < k; j += 2) {
        tmp[j + 0] += ax;
        tmp[j + 1] += ay;
      }
      final double xc = tmp[0], yc = tmp[1], w = tmp[2], h = tmp[3];
      final double xmin = xc - w * 0.5,
          ymin = yc - h * 0.5,
          xmax = xc + w * 0.5,
          ymax = yc + h * 0.5;
      final List<double> kp = <double>[];
      for (int j = 4; j < k; j += 2) {
        kp.add(tmp[j + 0]);
        kp.add(tmp[j + 1]);
      }
      out.add(DecodedBox(RectF(xmin, ymin, xmax, ymax), kp));
    }
    return out;
  }

  /// Raw-logit equivalent of `kMinScore`: `sigmoidClipped` is monotonic, so
  /// `sigmoidClipped(x) >= kMinScore` iff `x >= logit(kMinScore)`. Comparing
  /// raw logits skips the sigmoid for the vast majority of anchors that fall
  /// below threshold.
  static final double _rawScoreThreshold = math.log(
    kMinScore / (1.0 - kMinScore),
  );

  ({List<int> candidateIndices, List<double> candidateScores})
  _collectCandidateScores(Float32List raw, List<int> shape) {
    final int n = shape[1];
    final List<int> candidateIndices = <int>[];
    final List<double> candidateScores = <double>[];
    for (int i = 0; i < n; i++) {
      if (raw[i] >= _rawScoreThreshold) {
        candidateIndices.add(i);
        candidateScores.add(sigmoidClipped(raw[i]));
      }
    }
    return (
      candidateIndices: candidateIndices,
      candidateScores: candidateScores,
    );
  }

  /// Creates detections from pre-filtered boxes and scores.
  ///
  /// Unlike `_toDetections`, this method expects boxes and scores to already
  /// be filtered and matched 1:1 (same length, corresponding indices).
  List<Detection> _toDetectionsFiltered(
    List<DecodedBox> boxes,
    List<double> filteredScores,
  ) {
    final List<Detection> res = <Detection>[];
    final int n = boxes.length;
    for (int i = 0; i < n; i++) {
      final RectF b = boxes[i].boundingBox;
      if (b.xmax <= b.xmin || b.ymax <= b.ymin) continue;
      res.add(
        Detection(
          boundingBox: b,
          score: filteredScores[i],
          keypointsXY: boxes[i].keypointsXY,
        ),
      );
    }
    return res;
  }

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the face detection model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// **Note:** Most users should call [FaceDetector.dispose] instead, which
  /// automatically disposes all internal models (detection, mesh, and iris).
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _delegate?.delete();
    _delegate = null;
    _iso?.close();
    _compiledModel?.close();
    _itp?.close();
  }

  Interpreter _requireInterpreter() {
    final Interpreter? itp = _itp;
    if (itp == null) {
      throw StateError(
        'FaceDetection is using CompiledModel, not Interpreter.',
      );
    }
    return itp;
  }

  CompiledModel _requireCompiledModel() {
    final CompiledModel? compiledModel = _compiledModel;
    if (compiledModel == null) {
      throw StateError(
        'FaceDetection is using Interpreter, not CompiledModel.',
      );
    }
    return compiledModel;
  }
}
