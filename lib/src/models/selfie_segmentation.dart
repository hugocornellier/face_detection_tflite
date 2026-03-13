part of '../../face_detection_tflite.dart';

/// Minimum input image size (smaller images rejected).
const int kMinSegmentationInputSize = 16;

/// Returns the model filename for a given [SegmentationModel] variant.
String _modelFileFor(SegmentationModel model) => switch (model) {
      SegmentationModel.general => 'selfie_segmenter.tflite',
      SegmentationModel.landscape => 'selfie_segmenter_landscape.tflite',
      SegmentationModel.multiclass => 'selfie_multiclass.tflite',
    };

/// Model input width for all segmentation variants.
const int _segmentationInputWidth = 256;

/// Returns the model input height for a given [SegmentationModel] variant.
int _inputHeightFor(SegmentationModel model) => switch (model) {
      SegmentationModel.landscape => 144,
      _ => 256,
    };

/// Returns the expected output channels for a given [SegmentationModel] variant.
int _expectedOutputChannels(SegmentationModel model) => switch (model) {
      SegmentationModel.multiclass => 6,
      _ => 1,
    };

/// Segmentation class indices in the multiclass model output.
/// The model outputs 6 channels representing probabilities for each class.
class SegmentationClass {
  /// Background (not a person).
  static const int background = 0;

  /// Hair.
  static const int hair = 1;

  /// Body skin (arms, legs, etc.).
  static const int bodySkin = 2;

  /// Face skin.
  static const int faceSkin = 3;

  /// Clothes.
  static const int clothes = 4;

  /// Other (accessories, etc.).
  static const int other = 5;

  /// All person classes (everything except background).
  static const List<int> allPerson = [hair, bodySkin, faceSkin, clothes, other];
}

/// Performs selfie/person segmentation using MediaPipe TFLite models.
///
/// Generates a per-pixel probability mask separating foreground (person)
/// from background. Works on full images - no face detection required.
///
/// See the official model cards for architecture details, training data, and
/// intended use cases:
/// - Selfie segmentation: https://mediapipe.page.link/selfiesegmentation-mc
///   (local copy: `doc/model_cards/selfie_segmentation_model_card.pdf`)
/// - Multiclass segmentation:
///   (local copy: `doc/model_cards/multiclass_segmentation_model_card.pdf`)
///
/// ## Model Variants
///
/// Three model variants are available via [SegmentationConfig.model]:
///
/// - **[SegmentationModel.general]** (default): Binary person/background,
///   ~244KB, 256x256 input, single-channel sigmoid output.
/// - **[SegmentationModel.landscape]**: Binary person/background optimized
///   for 16:9 video, ~244KB, 144x256 input.
/// - **[SegmentationModel.multiclass]**: 6-class body part segmentation,
///   ~16MB, 256x256 input. Returns [MulticlassSegmentationMask] with
///   per-class accessors (hair, face, body, clothes, etc.).
///
/// ## Platform Support
///
/// All model variants work on all platforms. Binary models (general/landscape)
/// use the `Convolution2DTransposeBias` custom op which is bundled for each platform.
///
/// | Platform | Delegate | Notes |
/// |----------|----------|-------|
/// | iOS | Metal GPU | Custom ops statically linked via CocoaPods |
/// | Android | GPU/CPU | Custom ops built via CMake, loaded at runtime |
/// | macOS | CPU | Custom ops bundled as dylib |
/// | Linux | CPU | Custom ops bundled as .so |
/// | Windows | CPU | Custom ops bundled as dll |
///
/// ## Example
///
/// ```dart
/// // Default: fast binary segmentation (~244KB model)
/// final segmenter = await SelfieSegmentation.create();
/// final mask = await segmenter(imageBytes);
/// final binary = mask.toBinary(threshold: 0.5);
/// segmenter.dispose();
///
/// // Multiclass: per-class body part segmentation (~16MB model)
/// final multiSeg = await SelfieSegmentation.create(
///   config: SegmentationConfig(model: SegmentationModel.multiclass),
/// );
/// final multiMask = await multiSeg(imageBytes);
/// if (multiMask is MulticlassSegmentationMask) {
///   final hair = multiMask.hairMask;
/// }
/// multiSeg.dispose();
/// ```
///
/// ## Memory Considerations (varies by model)
///
/// - **General/Landscape**: ~244KB model + ~768KB buffers
/// - **Multiclass**: ~16MB model + ~2.3MB buffers
///
/// ## Integration with FaceDetector
///
/// For combined face detection and segmentation, use [FaceDetector]:
///
/// ```dart
/// final detector = FaceDetector();
/// await detector.initialize();
/// await detector.initializeSegmentation();
///
/// final faces = await detector.detectFaces(imageBytes);
/// final mask = await detector.getSegmentationMask(imageBytes);
/// ```
class SelfieSegmentation with _TfliteModelDisposable {
  @override
  final Interpreter _itp;
  final int _inW;
  final int _inH;
  final int _outChannels;
  final SegmentationConfig _config;
  final SegmentationModel _model;
  bool _delegateFailed = false;

  late final Tensor _inputTensor;
  late final Tensor _outputTensor;
  late final Float32List _inputBuf;
  late final Float32List _outputBuf;
  late final List<List<List<List<double>>>> _input4dCache;
  late final Object _output4dCache;
  late final Float32List _matTensorBuffer;
  late final int _outW;
  late final int _outH;

  SelfieSegmentation._(
    this._itp,
    this._inW,
    this._inH,
    this._outChannels,
    this._config,
    this._model,
  );

  /// Creates and initializes a selfie segmentation model instance.
  ///
  /// This factory method loads the selected segmentation TensorFlow Lite model
  /// from package assets and prepares it for inference.
  ///
  /// [config]: Configuration for model selection, delegates, and output limits.
  /// The [SegmentationConfig.model] field selects which model variant to load.
  ///
  /// Returns a fully initialized [SelfieSegmentation] instance.
  ///
  /// Throws [SegmentationException] if:
  /// - Model file cannot be loaded ([SegmentationError.modelNotFound])
  /// - Interpreter creation fails ([SegmentationError.interpreterCreationFailed])
  /// - Model validation fails ([SegmentationError.unexpectedTensorShape])
  ///
  /// Example:
  /// ```dart
  /// // Default: fast binary model (~244KB)
  /// final segmenter = await SelfieSegmentation.create();
  ///
  /// // Multiclass model with per-class masks (~16MB)
  /// final segmenter = await SelfieSegmentation.create(
  ///   config: SegmentationConfig(model: SegmentationModel.multiclass),
  /// );
  /// ```
  static Future<SelfieSegmentation> create({
    SegmentationConfig config = const SegmentationConfig(),
  }) async {
    final effectiveModel = config.model;
    final modelFile = _modelFileFor(effectiveModel);

    final obj = await _createWithLoader(
      config: config,
      loadInterpreter: (options) => Interpreter.fromAsset(
        'packages/face_detection_tflite/assets/models/$modelFile',
        options: options,
      ),
      loadErrorCode: SegmentationError.modelNotFound,
      loadErrorPrefix: 'Failed to load model $modelFile',
    );
    await obj._initializeTensors();
    return obj;
  }

  /// Creates a selfie segmentation model from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetectorIsolate] to initialize models
  /// in a background isolate where asset loading is not available.
  /// The [IsolateInterpreter] is skipped since the model is already running
  /// inside a background isolate.
  ///
  /// [modelBytes]: Raw TFLite model file contents.
  /// [config]: Configuration for model selection, delegates, and output limits.
  /// The [SegmentationConfig.model] field must match the model contained in [modelBytes].
  ///
  /// Example:
  /// ```dart
  /// final bytes = await rootBundle.load('assets/models/selfie_segmenter.tflite');
  /// final segmenter = await SelfieSegmentation.createFromBuffer(
  ///   bytes.buffer.asUint8List(),
  /// );
  /// ```
  static Future<SelfieSegmentation> createFromBuffer(
    Uint8List modelBytes, {
    SegmentationConfig config = const SegmentationConfig(),
  }) async {
    final obj = await _createWithLoader(
      config: config,
      loadInterpreter: (options) =>
          Interpreter.fromBuffer(modelBytes, options: options),
      loadErrorCode: SegmentationError.interpreterCreationFailed,
      loadErrorPrefix: 'Failed to create interpreter from buffer',
    );
    await obj._initializeTensors(useIsolateInterpreter: false);
    return obj;
  }

  /// Shared factory logic for [create] and [createFromBuffer].
  ///
  /// Handles delegate setup, fallback retry, validation, and tensor resize.
  /// Returns a [SelfieSegmentation] instance ready for [_initializeTensors].
  static Future<SelfieSegmentation> _createWithLoader({
    required SegmentationConfig config,
    required FutureOr<Interpreter> Function(InterpreterOptions) loadInterpreter,
    required SegmentationError loadErrorCode,
    required String loadErrorPrefix,
  }) async {
    final effectiveModel = config.model;
    final inW = _segmentationInputWidth;
    final inH = _inputHeightFor(effectiveModel);
    final outChannels = _expectedOutputChannels(effectiveModel);

    Delegate? delegate;
    InterpreterOptions options;
    bool delegateFailed = false;

    try {
      final result = InterpreterFactory.create(
        config.performanceConfig,
        addMediaPipeCustomOps: true,
      );
      options = result.$1;
      delegate = result.$2;
    } catch (e) {
      throw SegmentationException(
        SegmentationError.interpreterCreationFailed,
        'Failed to create interpreter options: $e',
        e,
      );
    }

    Interpreter itp;
    try {
      itp = await loadInterpreter(options);
    } catch (e) {
      if (delegate != null) {
        delegate.delete();
        delegate = null;
        delegateFailed = true;

        options = InterpreterOptions();
        options.addMediaPipeCustomOps();
        options.threads = config.performanceConfig.numThreads ?? 4;

        try {
          itp = await loadInterpreter(options);
        } catch (retryError) {
          throw SegmentationException(
            loadErrorCode,
            '$loadErrorPrefix (even after delegate fallback): $retryError',
            retryError,
          );
        }
      } else {
        throw SegmentationException(loadErrorCode, '$loadErrorPrefix: $e', e);
      }
    }

    if (config.validateModel) {
      _validateModel(itp, effectiveModel);
    }

    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final obj = SelfieSegmentation._(
      itp,
      inW,
      inH,
      outChannels,
      config,
      effectiveModel,
    );
    obj._delegate = delegate;
    obj._delegateFailed = delegateFailed;
    return obj;
  }

  /// Shared tensor initialization logic.
  ///
  /// When [useIsolateInterpreter] is false, inference runs directly via
  /// `_itp.invoke()` instead of spawning a nested isolate. This should be
  /// used when the model is already running inside a background isolate.
  Future<void> _initializeTensors({bool useIsolateInterpreter = true}) async {
    _inputTensor = _itp.getInputTensor(0);
    _outputTensor = _itp.getOutputTensor(0);
    _inputBuf = _inputTensor.data.buffer.asFloat32List();
    _outputBuf = _outputTensor.data.buffer.asFloat32List();

    final outShape = _outputTensor.shape;
    _outH = outShape.length >= 2 ? outShape[1] : _inH;
    _outW = outShape.length >= 3 ? outShape[2] : _inW;

    _input4dCache = createNHWCTensor4D(_inH, _inW);

    _output4dCache = allocTensorShape(outShape);

    _matTensorBuffer = Float32List(_inW * _inH * 3);

    if (useIsolateInterpreter && _config.useIsolate) {
      _iso = await InterpreterFactory.createIsolateIfNeeded(_itp, _delegate);
    }
  }

  /// Validates model tensor shapes match expected shapes for the given model variant.
  static void _validateModel(Interpreter itp, SegmentationModel model) {
    final inputShape = itp.getInputTensor(0).shape;
    final outputShape = itp.getOutputTensor(0).shape;
    final expectedChannels = _expectedOutputChannels(model);

    if (inputShape.length != 4 || inputShape[3] != 3) {
      throw SegmentationException(
        SegmentationError.unexpectedTensorShape,
        'Expected input shape [1, H, W, 3] for ${model.name} model, got $inputShape',
      );
    }

    if (outputShape.length != 4 || outputShape[3] != expectedChannels) {
      throw SegmentationException(
        SegmentationError.unexpectedTensorShape,
        'Expected output channels=$expectedChannels for ${model.name} model, got $outputShape',
      );
    }
  }

  /// Number of output channels (1 for binary models, 6 for multiclass).
  int get outputChannels => _outChannels;

  /// The segmentation model variant in use.
  SegmentationModel get model => _model;

  /// Model input width in pixels.
  int get inputWidth => _inW;

  /// Model input height in pixels.
  int get inputHeight => _inH;

  /// Output mask width (may differ from input due to model architecture).
  int get outputWidth => _outW;

  /// Output mask height.
  int get outputHeight => _outH;

  /// Whether this instance has been disposed.
  ///
  /// After disposal, inference methods will throw [StateError].
  bool get isDisposed => _disposed;

  /// Whether GPU delegate failed during inference.
  ///
  /// Check this after inference to detect GPU compatibility issues.
  /// Note: This does not indicate automatic fallback - the model continues
  /// to use whatever delegate was configured at creation time.
  bool get hasGpuDelegateFailed => _delegateFailed;

  /// Current configuration.
  SegmentationConfig get config => _config;

  /// Segments an image to separate foreground (person) from background.
  ///
  /// The [image] parameter is a [cv.Mat] in BGR or BGRA format.
  /// It is NOT disposed by this method -- caller is responsible for disposal.
  ///
  /// The optional [buffer] parameter allows reusing a pre-allocated Float32List
  /// for the tensor conversion to reduce GC pressure.
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities at model output resolution.
  ///
  /// Throws [SegmentationException] on:
  /// - Empty Mat ([SegmentationError.imageDecodeFailed])
  /// - Image smaller than 16x16 ([SegmentationError.imageTooSmall])
  /// - Inference failure ([SegmentationError.inferenceFailed])
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
  /// final mask = await segmenter.call(mat);
  /// mat.dispose();
  /// ```
  Future<SegmentationMask> call(
    cv.Mat image, {
    Float32List? buffer,
  }) async {
    if (_disposed) {
      throw StateError('Cannot use SelfieSegmentation after dispose()');
    }

    if (image.isEmpty) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Input Mat is empty',
      );
    }

    if (image.cols < kMinSegmentationInputSize ||
        image.rows < kMinSegmentationInputSize) {
      throw SegmentationException(
        SegmentationError.imageTooSmall,
        'Mat ${image.cols}x${image.rows} is smaller than minimum '
        '${kMinSegmentationInputSize}x$kMinSegmentationInputSize',
      );
    }

    final ImageTensor pack = convertImageToTensor(
      image,
      outW: _inW,
      outH: _inH,
      buffer: buffer ?? _matTensorBuffer,
    );

    final Float32List rawOutput;
    try {
      if (_iso == null) {
        _inputBuf.setAll(0, pack.tensorNHWC);
        _itp.invoke();
        rawOutput = Float32List.fromList(_outputBuf);
      } else {
        fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
        final List<List<List<List<List<double>>>>> inputs = [_input4dCache];

        final Map<int, Object> outputs = <int, Object>{0: _output4dCache};

        await _iso!.runForMultipleInputs(inputs, outputs);
        rawOutput = flattenDynamicTensor(outputs[0]);
      }
    } catch (e) {
      if (!_delegateFailed && _delegate != null) {
        _delegateFailed = true;
      }
      throw SegmentationException(
        SegmentationError.inferenceFailed,
        'Inference failed: $e',
        e,
      );
    }

    return _buildMask(rawOutput, image.cols, image.rows, pack.padding);
  }

  /// Segments an image from encoded bytes (JPEG, PNG, etc.).
  ///
  /// Decodes [imageBytes] to a [cv.Mat], runs segmentation, and disposes
  /// the intermediate Mat automatically.
  ///
  /// Example:
  /// ```dart
  /// final mask = await segmenter.callFromBytes(imageBytes);
  /// ```
  Future<SegmentationMask> callFromBytes(
    Uint8List imageBytes, {
    Float32List? buffer,
  }) async {
    final cv.Mat mat;
    try {
      mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    } catch (e) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to decode image bytes with OpenCV (length: ${imageBytes.length}): $e',
        e,
      );
    }

    if (mat.isEmpty) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to decode image bytes (length: ${imageBytes.length})',
      );
    }

    try {
      return await call(mat, buffer: buffer);
    } finally {
      mat.dispose();
    }
  }

  /// Builds the appropriate mask type based on the active model.
  ///
  /// For binary models (general/landscape): copies single-channel sigmoid output.
  /// For multiclass: computes softmax, returns [MulticlassSegmentationMask].
  SegmentationMask _buildMask(
    Float32List rawOutput,
    int originalWidth,
    int originalHeight,
    List<double> padding,
  ) {
    if (_model == SegmentationModel.multiclass) {
      final classProbs = _computeClassProbabilities(rawOutput, _outW, _outH);
      final int numPixels = _outW * _outH;
      final personMask = _buildPersonMask(rawOutput, numPixels, classProbs);
      return MulticlassSegmentationMask(
        data: personMask,
        width: _outW,
        height: _outH,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
        padding: padding,
        classData: classProbs,
      );
    } else {
      final int numPixels = _outW * _outH;
      final personMask = _buildPersonMask(rawOutput, numPixels, null);
      return SegmentationMask(
        data: personMask,
        width: _outW,
        height: _outH,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
        padding: padding,
      );
    }
  }

  /// Computes per-pixel softmax probabilities for all 6 multiclass channels.
  ///
  /// Returns Float32List of length width * height * 6, with per-pixel
  /// probabilities in channel order: background, hair, bodySkin, faceSkin, clothes, other.
  static Float32List _computeClassProbabilities(
    Float32List rawOutput,
    int width,
    int height,
  ) {
    final int numPixels = width * height;
    final result = Float32List(numPixels * 6);

    for (int i = 0; i < numPixels; i++) {
      final int base = i * 6;

      final double l0 = rawOutput[base];
      final double l1 = rawOutput[base + 1];
      final double l2 = rawOutput[base + 2];
      final double l3 = rawOutput[base + 3];
      final double l4 = rawOutput[base + 4];
      final double l5 = rawOutput[base + 5];

      double maxLogit = l0;
      if (l1 > maxLogit) maxLogit = l1;
      if (l2 > maxLogit) maxLogit = l2;
      if (l3 > maxLogit) maxLogit = l3;
      if (l4 > maxLogit) maxLogit = l4;
      if (l5 > maxLogit) maxLogit = l5;

      final double e0 = math.exp(l0 - maxLogit);
      final double e1 = math.exp(l1 - maxLogit);
      final double e2 = math.exp(l2 - maxLogit);
      final double e3 = math.exp(l3 - maxLogit);
      final double e4 = math.exp(l4 - maxLogit);
      final double e5 = math.exp(l5 - maxLogit);

      final double sumExp = e0 + e1 + e2 + e3 + e4 + e5;

      result[base] = e0 / sumExp;
      result[base + 1] = e1 / sumExp;
      result[base + 2] = e2 / sumExp;
      result[base + 3] = e3 / sumExp;
      result[base + 4] = e4 / sumExp;
      result[base + 5] = e5 / sumExp;
    }

    return result;
  }

  /// Builds the per-pixel person mask from raw output.
  ///
  /// If [classProbs] is provided (multiclass), uses `1 - background` probability.
  /// Otherwise (binary), copies the raw sigmoid output directly.
  static Float32List _buildPersonMask(
    Float32List rawOutput,
    int numPixels,
    Float32List? classProbs,
  ) {
    final personMask = Float32List(numPixels);
    if (classProbs != null) {
      for (int i = 0; i < numPixels; i++) {
        personMask[i] = 1.0 - classProbs[i * 6];
      }
    } else {
      for (int i = 0; i < numPixels; i++) {
        personMask[i] = rawOutput[i];
      }
    }
    return personMask;
  }

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the segmentation model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// It is safe to call dispose multiple times.
  ///
  /// **Important:** If you are switching models (disposing one and immediately
  /// creating another), prefer [disposeAsync] which ensures the background
  /// isolate is fully terminated before freeing the native interpreter.
  void dispose() => _doDispose();

  /// Asynchronously releases all resources, ensuring the background isolate
  /// is fully terminated before freeing the native interpreter.
  ///
  /// Prefer this over [dispose] when switching models or when you can await
  /// the cleanup. This prevents native crashes caused by the interpreter
  /// being freed while the background isolate still references it.
  ///
  /// It is safe to call this multiple times.
  Future<void> disposeAsync() => _doDisposeAsync();
}

@visibleForTesting
Float32List testComputeClassProbabilities(
  Float32List rawOutput,
  int width,
  int height,
) =>
    SelfieSegmentation._computeClassProbabilities(rawOutput, width, height);

@visibleForTesting
String testModelFileFor(SegmentationModel model) => _modelFileFor(model);

@visibleForTesting
int testInputWidthFor(SegmentationModel model) => _segmentationInputWidth;

@visibleForTesting
int testInputHeightFor(SegmentationModel model) => _inputHeightFor(model);

@visibleForTesting
int testExpectedOutputChannels(SegmentationModel model) =>
    _expectedOutputChannels(model);
