part of '../native/face_native_lib.dart';

/// Predicts the full 468-point face mesh (x, y, z per point) for an aligned face crop.
/// Coordinates are normalized before later mapping back to image space.
///
/// The underlying TFLite model (`face_landmark.tflite`) is sourced from Google's MediaPipe
/// framework. See the official model card for architecture details, training data, and
/// intended use cases: https://mediapipe.page.link/facemesh-mc
/// (local copy: `doc/model_cards/face_landmark_model_card.pdf`)
class FaceLandmark with _TfliteModelDisposable {
  @override
  final Interpreter? _itp;
  final CompiledModel? _compiledModel;
  final int _inW, _inH;
  late final int _bestIdx;
  late final TensorFloat32Views _views;
  late final Float32List _scratchBuf;
  late final List<List<int>> _outShapes;
  late final List<List<List<List<double>>>> _input4dCache;
  late final Map<int, Object> _outputsCache;

  /// The model input width in pixels.
  int get inputWidth => _inW;

  /// The model input height in pixels.
  int get inputHeight => _inH;

  FaceLandmark._(this._itp, this._inW, this._inH) : _compiledModel = null;

  FaceLandmark._compiled(CompiledModel compiledModel, this._inW, this._inH)
    : _itp = null,
      _compiledModel = compiledModel;

  /// Creates and initializes a face landmark (mesh) model instance.
  ///
  /// This factory method loads the 468-point face mesh TensorFlow Lite model
  /// from package assets and prepares it for inference. The face mesh provides
  /// detailed 3D geometry of facial features.
  ///
  /// The [options] parameter allows you to customize the TFLite interpreter
  /// configuration (e.g., number of threads, use of GPU delegate).
  ///
  /// The [performanceConfig] parameter enables hardware acceleration delegates.
  /// Use [PerformanceConfig.xnnpack()] for 2-5x speedup on CPU. If both [options]
  /// and [performanceConfig] are provided, [options] takes precedence.
  ///
  /// Returns a fully initialized [FaceLandmark] instance ready to predict face meshes.
  ///
  /// **Note:** This model expects an aligned face crop as input. For full pipeline
  /// processing, use the high-level [FaceDetector] class instead.
  ///
  /// Example:
  /// ```dart
  /// // Default (auto mode)
  /// final landmarkModel = await FaceLandmark.create();
  /// final meshPoints = await landmarkModel.call(alignedFaceCropMat);
  ///
  /// // With XNNPACK acceleration
  /// final landmarkModel = await FaceLandmark.create(
  ///   performanceConfig: PerformanceConfig.xnnpack(),
  /// );
  /// ```
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<FaceLandmark> create({
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
  }) => _createWithLoader(
    load: (opts) => Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$kFaceLandmarkModel',
      options: opts,
    ),
    options: options,
    performanceConfig: performanceConfig,
  );

  /// Creates a face landmark model from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetector] to initialize models
  /// in a background isolate where asset loading is not available.
  ///
  /// The [modelBytes] parameter should contain the raw TFLite model file contents.
  static Future<FaceLandmark> createFromBuffer(
    Uint8List modelBytes, {
    PerformanceConfig? performanceConfig,
  }) => _createWithLoader(
    load: (opts) => Interpreter.fromBuffer(modelBytes, options: opts),
    performanceConfig: performanceConfig,
    useIsolateInterpreter: false,
  );

  /// Creates a face landmark model backed by LiteRT CompiledModel.
  static Future<FaceLandmark> createCompiledFromBuffer(
    Uint8List modelBytes, {
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
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
    final int side;
    try {
      side = _compiledSquareInputSide(compiledModel, 'Compiled face landmark');
    } catch (_) {
      compiledModel.close();
      rethrow;
    }
    final obj = FaceLandmark._compiled(compiledModel, side, side);
    try {
      obj._initializeCompiledModel();
      return obj;
    } catch (_) {
      obj.dispose();
      rethrow;
    }
  }

  static Future<FaceLandmark> _createWithLoader({
    required FutureOr<Interpreter> Function(InterpreterOptions) load,
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
    bool useIsolateInterpreter = true,
  }) => _buildModel(
    load: load,
    options: options,
    performanceConfig: performanceConfig,
    useIsolateInterpreter: useIsolateInterpreter,
    construct: FaceLandmark._,
    initTensors: (obj, iso) =>
        obj._initializeTensors(useIsolateInterpreter: iso),
  );

  /// Shared tensor initialization logic.
  ///
  /// When [useIsolateInterpreter] is false, inference runs directly via
  /// `_itp.invoke()` instead of spawning a nested isolate. This should be
  /// used when the model is already running inside a background isolate.
  Future<void> _initializeTensors({bool useIsolateInterpreter = true}) async {
    final Interpreter itp = _itp!;
    int numElements(List<int> s) => s.fold(1, (a, b) => a * b);

    final Map<int, OutputTensorInfo> outputInfo = collectOutputTensorInfo(itp);
    final Map<int, List<int>> shapes = outputInfo.map(
      (int k, OutputTensorInfo v) => MapEntry(k, v.shape),
    );

    int bestIdx = -1;
    int bestLen = -1;
    for (final MapEntry<int, List<int>> e in shapes.entries) {
      final int len = numElements(e.value);
      if (len > bestLen && len % 3 == 0) {
        bestLen = len;
        bestIdx = e.key;
      }
    }
    _bestIdx = bestIdx;
    _views = TensorFloat32Views.capture(itp);
    _scratchBuf = Float32List(_inH * _inW * 3);

    final int maxIndex = shapes.keys.isEmpty
        ? -1
        : shapes.keys.reduce((a, b) => a > b ? a : b);
    _outShapes = List<List<int>>.generate(
      maxIndex + 1,
      (i) => shapes[i] ?? const <int>[],
    );

    _input4dCache = createNHWCTensor4D(_inH, _inW);

    _outputsCache = <int, Object>{};
    for (int i = 0; i < _outShapes.length; i++) {
      final List<int> s = _outShapes[i];
      if (s.isNotEmpty) {
        _outputsCache[i] = allocTensorShape(s);
      }
    }

    if (useIsolateInterpreter) {
      _iso = await InterpreterFactory.createIsolateIfNeeded(itp, _delegate);
    }
  }

  void _initializeCompiledModel() {
    final CompiledModel compiledModel = _compiledModel!;
    _scratchBuf = Float32List(_inH * _inW * 3);

    int bestIdx = -1;
    int bestLen = -1;
    for (int i = 0; i < compiledModel.outputCount; i++) {
      final int len = _compiledFloatCount(
        compiledModel.outputByteSizes[i],
        'Compiled face landmark output[$i]',
      );
      if (len > bestLen && len % 3 == 0) {
        bestLen = len;
        bestIdx = i;
      }
    }
    if (bestIdx == -1) {
      throw UnsupportedError(
        'Compiled face landmark has no output with a float count divisible '
        'by 3.',
      );
    }
    _bestIdx = bestIdx;
  }

  /// Predicts the 468-point face mesh for an aligned face crop using cv.Mat.
  ///
  /// Accepts a cv.Mat directly, providing better performance by avoiding
  /// image format conversions.
  ///
  /// The [faceCrop] parameter should contain an aligned, cropped face as cv.Mat.
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
  ///
  /// The optional [buffer] parameter allows reusing a pre-allocated Float32List
  /// for the tensor conversion to reduce GC pressure.
  ///
  /// Returns a list of 468 3D landmark points in normalized coordinates.
  ///
  /// Example:
  /// ```dart
  /// final faceCropMat = cv.imdecode(bytes, cv.IMREAD_COLOR);
  /// final meshPoints = await faceLandmark.call(faceCropMat);
  /// faceCropMat.dispose();
  /// ```
  Future<List<List<double>>> call(
    cv.Mat faceCrop, {
    Float32List? buffer,
  }) async {
    final CompiledModel? compiledModel = _compiledModel;
    if (compiledModel != null) {
      // Copying runAsync is the official LiteRT pattern for host-side data
      // (the C++ Write/Read API is lock+memcpy+unlock); the Metal accelerator
      // only supports MetalBufferPacked tensor buffers, so host zero-copy is
      // not available on the GPU path.
      final ImageTensor pack = convertImageToTensor(
        faceCrop,
        outW: _inW,
        outH: _inH,
        buffer: buffer ?? _scratchBuf,
      );
      final List<Float32List> outputs = await compiledModel.runAsync([
        pack.tensorNHWC,
      ]);
      return _unpackLandmarks(
        outputs[_bestIdx],
        _inW,
        _inH,
        pack.padding,
        clamp: true,
      );
    }

    final ImageTensor pack = convertImageToTensor(
      faceCrop,
      outW: _inW,
      outH: _inH,
      buffer: buffer ?? _scratchBuf,
    );

    if (_iso == null) {
      _views.inputs[0].setAll(0, pack.tensorNHWC);
      _itp!.invoke();
      return _unpackLandmarks(
        _views.outputs[_bestIdx],
        _inW,
        _inH,
        pack.padding,
        clamp: true,
      );
    } else {
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [_input4dCache];
      await _iso!.runForMultipleInputs(inputs, _outputsCache);

      final dynamic best = _outputsCache[_bestIdx];

      final Float32List bestFlat = flattenDynamicTensor(best);
      return _unpackLandmarks(bestFlat, _inW, _inH, pack.padding, clamp: true);
    }
  }

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the face landmark model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// **Note:** Most users should call [FaceDetector.dispose] instead, which
  /// automatically disposes all internal models (detection, mesh, and iris).
  void dispose() {
    if (_disposed) return;
    _compiledModel?.close();
    _doDispose();
  }
}
