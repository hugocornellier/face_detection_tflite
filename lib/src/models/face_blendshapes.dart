part of '../native/face_native_lib.dart';

/// Runs the MediaPipe **Blendshape V2** model, which classifies a packed set of
/// 146 facial landmarks into 52 expression coefficients (blendshapes).
///
/// Unlike the other models in this package, this is **not an image model**: it
/// never sees pixels. Its input is a `[1, 146, 2]` tensor of landmark
/// coordinates produced by [packBlendshapeInput], and its output is 52
/// coefficients in `[0, 1]` (a terminal sigmoid keeps them in range). Because
/// the input is tiny and dense, the model runs in well under a millisecond on
/// CPU, so it is pinned to the CPU on every path (no GPU dispatch payoff).
///
/// The underlying TFLite model (`face_blendshapes.tflite`, MLP-Mixer, float16)
/// is extracted from Google's MediaPipe Face Landmarker task bundle. See the
/// official model card for architecture, training data, and intended use:
/// https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
/// (local copy: `doc/model_cards/blendshape_v2_model_card.pdf`)
///
/// Used internally by [FaceDetector] in [FaceDetectionMode.full]; results are
/// exposed via [Face.blendshapes], [Face.smilingProbability], and the
/// eye-open probability getters.
class FaceBlendshapesModel with _TfliteModelDisposable {
  @override
  final Interpreter? _itp;
  final CompiledModel? _compiledModel;
  late final TensorFloat32Views _views;

  FaceBlendshapesModel._(this._itp) : _compiledModel = null;

  FaceBlendshapesModel._compiled(CompiledModel compiledModel)
    : _itp = null,
      _compiledModel = compiledModel;

  /// Creates and initializes the model from package assets.
  ///
  /// For full pipeline use, prefer the high-level [FaceDetector] with
  /// [FaceDetectionMode.full], which loads and runs this model for you.
  static Future<FaceBlendshapesModel> create({
    InterpreterOptions? options,
  }) async {
    final Interpreter itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$kFaceBlendshapesModel',
      options: options,
    );
    return _finishInterpreter(itp);
  }

  /// Creates the model from pre-loaded model bytes.
  ///
  /// Used by [FaceDetector] to initialize the model inside its background
  /// isolate, where asset loading is unavailable. Inference runs directly via
  /// `_itp.invoke()` (no nested isolate) since the caller is already off the
  /// UI thread and the model is trivially cheap.
  static Future<FaceBlendshapesModel> createFromBuffer(
    Uint8List modelBytes, {
    InterpreterOptions? options,
  }) async {
    final Interpreter itp = Interpreter.fromBuffer(
      modelBytes,
      options: options,
    );
    return _finishInterpreter(itp);
  }

  /// Creates the model from a custom `.tflite` file on disk.
  static Future<FaceBlendshapesModel> createFromFile(
    String modelPath, {
    InterpreterOptions? options,
  }) async {
    final Interpreter itp = Interpreter.fromFile(
      File(modelPath),
      options: options,
    );
    return _finishInterpreter(itp);
  }

  static FaceBlendshapesModel _finishInterpreter(Interpreter itp) {
    final obj = FaceBlendshapesModel._(itp);
    try {
      obj._initializeTensors();
      return obj;
    } catch (_) {
      obj.dispose();
      rethrow;
    }
  }

  void _initializeTensors() {
    final Interpreter itp = _itp!;
    // The input shape ([1, 146, 2]) is fully static, so no resize is needed;
    // just ensure the tensor buffers exist before capturing views.
    itp.allocateTensors();
    _views = TensorFloat32Views.capture(itp);
    _validateShapes(
      inputFloats: _views.inputs.isNotEmpty ? _views.inputs.first.length : -1,
      outputFloats: _views.outputs.isNotEmpty
          ? _views.outputs.first.length
          : -1,
      inputCount: _views.inputs.length,
      outputCount: _views.outputs.length,
    );
  }

  /// Creates the model backed by a LiteRT [CompiledModel], pinned to the CPU.
  ///
  /// A 292-in / 52-out MLP is far below the size where GPU dispatch pays off,
  /// and CPU pinning sidesteps Metal tensor-format questions for a non-image
  /// input (mirrors [IrisLandmark.createCompiledFromBuffer]).
  static Future<FaceBlendshapesModel> createCompiledFromBuffer(
    Uint8List modelBytes,
  ) async {
    final CompiledModel compiledModel = CompiledModel.fromBuffer(
      modelBytes,
      accelerators: {Accelerator.cpu},
    );
    final obj = FaceBlendshapesModel._compiled(compiledModel);
    try {
      obj._initializeCompiledModel();
      return obj;
    } catch (_) {
      obj.dispose();
      rethrow;
    }
  }

  void _initializeCompiledModel() {
    final CompiledModel cm = _compiledModel!;
    _validateShapes(
      inputFloats: cm.inputCount == 1
          ? _compiledFloatCount(
              cm.inputByteSizes.single,
              'Compiled face blendshapes input[0]',
            )
          : -1,
      outputFloats: cm.outputCount >= 1
          ? _compiledFloatCount(
              cm.outputByteSizes.first,
              'Compiled face blendshapes output[0]',
            )
          : -1,
      inputCount: cm.inputCount,
      outputCount: cm.outputCount,
    );
  }

  static void _validateShapes({
    required int inputFloats,
    required int outputFloats,
    required int inputCount,
    required int outputCount,
  }) {
    if (inputCount != 1 || inputFloats != kBlendshapeInputFloats) {
      throw UnsupportedError(
        'Face blendshapes model expects one input of $kBlendshapeInputFloats '
        'floats ([1, $kBlendshapeLandmarkCount, 2]); got $inputCount input(s) '
        'with $inputFloats floats.',
      );
    }
    if (outputCount < 1 || outputFloats != kBlendshapeCount) {
      throw UnsupportedError(
        'Face blendshapes model expects an output of $kBlendshapeCount floats; '
        'got $outputCount output(s) with $outputFloats floats.',
      );
    }
  }

  /// Runs the model on a packed `[1, 146, 2]` landmark tensor and returns the
  /// 52 blendshape coefficients, each clamped to `[0, 1]`.
  ///
  /// [packed] must have length [kBlendshapeInputFloats] (use
  /// [packBlendshapeInput]). Returns `null` when the input is the wrong size or
  /// the model produces a NaN (delegate/fp16 quirk), so the caller leaves that
  /// face's scores null rather than surfacing garbage.
  Future<Float32List?> call(Float32List packed) async {
    if (packed.length != kBlendshapeInputFloats) return null;

    final CompiledModel? cm = _compiledModel;
    if (cm != null) {
      final List<Float32List> outputs = await cm.runAsync([packed]);
      if (outputs.isEmpty) return null;
      return _sanitize(outputs.first);
    }

    _views.inputs[0].setAll(0, packed);
    _itp!.invoke();
    return _sanitize(_views.outputs.first);
  }

  /// Copies [raw] into a fresh list, clamping to `[0, 1]`; returns null if any
  /// value is NaN (the whole vector is treated as failed).
  static Float32List? _sanitize(Float32List raw) {
    final int n = raw.length < kBlendshapeCount ? raw.length : kBlendshapeCount;
    final Float32List out = Float32List(kBlendshapeCount);
    for (int i = 0; i < n; i++) {
      final double v = raw[i];
      if (v.isNaN) return null;
      out[i] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
    return out;
  }

  /// Releases all TensorFlow Lite resources held by this model.
  void dispose() {
    if (_disposed) return;
    _compiledModel?.close();
    _doDispose();
  }
}

/// Test-only: exposes the private NaN-sanitize/clamp applied to model output.
@visibleForTesting
Float32List? testSanitizeBlendshapes(Float32List raw) =>
    FaceBlendshapesModel._sanitize(raw);
