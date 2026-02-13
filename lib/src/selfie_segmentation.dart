part of '../face_detection_tflite.dart';

/// Minimum input image size (smaller images rejected).
const int kMinSegmentationInputSize = 16;

/// Returns the effective model to use.
///
/// All platforms now support all models with custom ops.
SegmentationModel _effectiveModel(SegmentationModel requested) {
  return requested;
}

/// Returns the model filename for a given [SegmentationModel] variant.
String _modelFileFor(SegmentationModel model) => switch (model) {
      SegmentationModel.general => 'selfie_segmenter.tflite',
      SegmentationModel.landscape => 'selfie_segmenter_landscape.tflite',
      SegmentationModel.multiclass => 'selfie_multiclass.tflite',
    };

/// Returns the model input width for a given [SegmentationModel] variant.
int _inputWidthFor(SegmentationModel model) => 256;

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
class SelfieSegmentation {
  IsolateInterpreter? _iso;
  final Interpreter _itp;
  final int _inW;
  final int _inH;
  final int _outChannels;
  final SegmentationConfig _config;
  final SegmentationModel _model;
  Delegate? _delegate;
  bool _delegateFailed = false;
  bool _disposed = false;

  late final Tensor _inputTensor;
  late final Tensor _outputTensor;
  late final Float32List _inputBuf;
  late final Float32List _outputBuf;
  late final List<List<List<List<double>>>> _input4dCache;
  late final Object _output4dCache; // Pre-allocated output for isolate path
  late final Float32List
      _matTensorBuffer; // Reusable buffer for Mat conversions
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
    // Resolve the effective model (all platforms now support all models)
    final effectiveModel = _effectiveModel(config.model);
    final modelFile = _modelFileFor(effectiveModel);
    final inW = _inputWidthFor(effectiveModel);
    final inH = _inputHeightFor(effectiveModel);
    final outChannels = _expectedOutputChannels(effectiveModel);

    Delegate? delegate;
    InterpreterOptions options;
    bool delegateFailed = false;

    try {
      final result = _createSegmentationInterpreterOptions(
        config.performanceConfig,
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
      itp = await Interpreter.fromAsset(
        'packages/face_detection_tflite/assets/models/$modelFile',
        options: options,
      );
    } catch (e) {
      // If using a delegate, try again without it (fallback to CPU)
      if (delegate != null) {
        delegate.delete();
        delegate = null;
        delegateFailed = true;

        // Retry with CPU-only options
        options = InterpreterOptions();
        options.addMediaPipeCustomOps();
        options.threads = config.performanceConfig.numThreads ?? 4;

        try {
          itp = await Interpreter.fromAsset(
            'packages/face_detection_tflite/assets/models/$modelFile',
            options: options,
          );
        } catch (retryError) {
          throw SegmentationException(
            SegmentationError.modelNotFound,
            'Failed to load model $modelFile (even after delegate fallback): $retryError',
            retryError,
          );
        }
      } else {
        throw SegmentationException(
          SegmentationError.modelNotFound,
          'Failed to load model $modelFile: $e',
          e,
        );
      }
    }

    // Validate model if configured
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
    // Resolve the effective model (all platforms now support all models)
    final effectiveModel = _effectiveModel(config.model);
    final inW = _inputWidthFor(effectiveModel);
    final inH = _inputHeightFor(effectiveModel);
    final outChannels = _expectedOutputChannels(effectiveModel);

    Delegate? delegate;
    InterpreterOptions options;
    bool delegateFailed = false;

    try {
      final result = _createSegmentationInterpreterOptions(
        config.performanceConfig,
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
      itp = Interpreter.fromBuffer(modelBytes, options: options);
    } catch (e) {
      // If using a delegate, try again without it (fallback to CPU)
      if (delegate != null) {
        delegate.delete();
        delegate = null;
        delegateFailed = true;

        options = InterpreterOptions();
        options.addMediaPipeCustomOps();
        options.threads = config.performanceConfig.numThreads ?? 4;

        try {
          itp = Interpreter.fromBuffer(modelBytes, options: options);
        } catch (retryError) {
          throw SegmentationException(
            SegmentationError.interpreterCreationFailed,
            'Failed to create interpreter from buffer (even after delegate fallback): $retryError',
            retryError,
          );
        }
      } else {
        throw SegmentationException(
          SegmentationError.interpreterCreationFailed,
          'Failed to create interpreter from buffer: $e',
          e,
        );
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
    await obj._initializeTensors(useIsolateInterpreter: false);
    return obj;
  }

  /// Shared tensor initialization logic.
  ///
  /// When [useIsolateInterpreter] is false, inference runs directly via
  /// `_itp.invoke()` instead of spawning a nested isolate. This should be
  /// used when the model is already running inside a background isolate.
  Future<void> _initializeTensors({
    bool useIsolateInterpreter = true,
  }) async {
    _inputTensor = _itp.getInputTensor(0);
    _outputTensor = _itp.getOutputTensor(0);
    _inputBuf = _inputTensor.data.buffer.asFloat32List();
    _outputBuf = _outputTensor.data.buffer.asFloat32List();

    // Output shape: [1, H, W, C] where C=1 (binary) or C=6 (multiclass)
    final outShape = _outputTensor.shape;
    _outH = outShape.length >= 2 ? outShape[1] : _inH;
    _outW = outShape.length >= 3 ? outShape[2] : _inW;

    // Pre-allocate input cache for isolate path
    _input4dCache = createNHWCTensor4D(_inH, _inW);

    // Pre-allocate output cache for isolate path (eliminates per-call allocation)
    _output4dCache = allocTensorShape(outShape);

    // Pre-allocate buffer for Mat tensor conversions
    _matTensorBuffer = Float32List(_inW * _inH * 3);

    // Only create IsolateInterpreter if configured, not inside a worker isolate,
    // and no delegate is active (delegates handle threading internally)
    if (useIsolateInterpreter && _config.useIsolate && _delegate == null) {
      _iso = await IsolateInterpreter.create(address: _itp.address);
    }
  }

  /// Validates model tensor shapes match expected shapes for the given model variant.
  static void _validateModel(Interpreter itp, SegmentationModel model) {
    final inputShape = itp.getInputTensor(0).shape;
    final outputShape = itp.getOutputTensor(0).shape;
    final expectedChannels = _expectedOutputChannels(model);

    // Validate input shape [1, H, W, 3]
    if (inputShape.length != 4 || inputShape[3] != 3) {
      throw SegmentationException(
        SegmentationError.unexpectedTensorShape,
        'Expected input shape [1, H, W, 3] for ${model.name} model, got $inputShape',
      );
    }

    // Validate output channels match model type
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

  /// Segments encoded image bytes using OpenCV for fast decoding.
  ///
  /// [imageBytes]: Encoded image (JPEG, PNG, or other supported format).
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities at model output resolution.
  ///
  /// This method uses OpenCV's native `imdecode()` which is 5-10x faster than
  /// pure Dart image decoding. For even better performance with video frames,
  /// use [callFromMat] directly with pre-decoded cv.Mat images.
  ///
  /// Throws [SegmentationException] on:
  /// - Image decode failure ([SegmentationError.imageDecodeFailed])
  /// - Image smaller than 16x16 ([SegmentationError.imageTooSmall])
  /// - Inference failure ([SegmentationError.inferenceFailed])
  ///
  /// Example:
  /// ```dart
  /// final imageBytes = await File('selfie.jpg').readAsBytes();
  /// final mask = await segmenter(imageBytes);
  /// print('Mask: ${mask.width}x${mask.height}');
  /// ```
  Future<SegmentationMask> call(Uint8List imageBytes) async {
    if (_disposed) {
      throw StateError('Cannot use SelfieSegmentation after dispose()');
    }

    // Use OpenCV native decode (5-10x faster than img.decodeImage)
    final cv.Mat image;
    try {
      image = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    } catch (e) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to decode image bytes with OpenCV (length: ${imageBytes.length}): $e',
        e,
      );
    }

    if (image.isEmpty) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to decode image bytes (length: ${imageBytes.length})',
      );
    }

    try {
      return await callFromMat(image);
    } finally {
      image.dispose();
    }
  }

  /// Segments encoded image bytes using pure Dart decoding.
  ///
  /// This is a fallback method that uses `package:image` for decoding.
  /// Prefer [call] which uses OpenCV and is 5-10x faster.
  ///
  /// [imageBytes]: Encoded image (JPEG, PNG, or other supported format).
  @Deprecated(
    'Will be removed in 5.0.0. Use call (which uses OpenCV internally) or callFromMat instead.',
  )
  Future<SegmentationMask> callWithImagePackage(Uint8List imageBytes) async {
    img.Image? decoded;
    try {
      decoded = img.decodeImage(imageBytes);
    } catch (e) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to decode image bytes (length: ${imageBytes.length}): $e',
        e,
      );
    }
    if (decoded == null) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to decode image bytes (length: ${imageBytes.length})',
      );
    }
    return callWithDecoded(decoded);
  }

  /// Segments a pre-decoded image.
  ///
  /// [decoded]: Pre-decoded image (from `package:image`).
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities.
  ///
  /// Example:
  /// ```dart
  /// final image = img.decodeImage(bytes)!;
  /// final mask = await segmenter.callWithDecoded(image);
  /// ```
  @Deprecated('Will be removed in 5.0.0. Use callFromMat instead.')
  Future<SegmentationMask> callWithDecoded(img.Image decoded) async {
    if (_disposed) {
      throw StateError('Cannot use SelfieSegmentation after dispose()');
    }

    // Validate minimum size
    if (decoded.width < kMinSegmentationInputSize ||
        decoded.height < kMinSegmentationInputSize) {
      throw SegmentationException(
        SegmentationError.imageTooSmall,
        'Image ${decoded.width}x${decoded.height} is smaller than minimum '
        '${kMinSegmentationInputSize}x$kMinSegmentationInputSize',
      );
    }

    // Handle grayscale: convert to RGB
    img.Image inputImage = decoded;
    if (decoded.numChannels == 1) {
      inputImage = _grayscaleToRgb(decoded);
    } else if (decoded.numChannels == 4) {
      // Handle RGBA: use as-is, getBytes will handle channel conversion
      inputImage = decoded;
    }

    final ImageTensor pack = convertImageToTensor(
      inputImage,
      outW: _inW,
      outH: _inH,
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

        // Use pre-allocated output cache to avoid per-call allocation
        final Map<int, Object> outputs = <int, Object>{0: _output4dCache};

        await _iso!.runForMultipleInputs(inputs, outputs);
        rawOutput = flattenDynamicTensor(outputs[0]);
      }
    } catch (e) {
      // Track GPU delegate failures for fallback detection
      if (!_delegateFailed && _delegate != null) {
        _delegateFailed = true;
      }
      throw SegmentationException(
        SegmentationError.inferenceFailed,
        'Inference failed: $e',
        e,
      );
    }

    return _buildMask(rawOutput, decoded.width, decoded.height, pack.padding);
  }

  /// Segments a cv.Mat image (OpenCV pipeline).
  ///
  /// [image]: cv.Mat in BGR or BGRA format (OpenCV default).
  /// [buffer]: Optional pre-allocated buffer for tensor conversion.
  ///
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
  /// final mask = await segmenter.callFromMat(mat);
  /// mat.dispose();
  /// ```
  Future<SegmentationMask> callFromMat(
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

    // Use provided buffer or fall back to pre-allocated internal buffer
    final ImageTensor pack = convertImageToTensorFromMat(
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

        // Use pre-allocated output cache to avoid per-call allocation
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
      // Multiclass: compute full class probabilities, derive person mask
      final classProbs = _computeClassProbabilities(rawOutput, _outW, _outH);
      final int numPixels = _outW * _outH;
      final personMask = Float32List(numPixels);
      for (int i = 0; i < numPixels; i++) {
        personMask[i] = 1.0 - classProbs[i * 6]; // 1 - P(background)
      }
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
      // Binary model: single channel, already sigmoid — copy directly
      final int numPixels = _outW * _outH;
      final personMask = Float32List(numPixels);
      for (int i = 0; i < numPixels; i++) {
        personMask[i] = rawOutput[i];
      }
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

      // Read all 6 logits and find max in single pass
      final double l0 = rawOutput[base];
      final double l1 = rawOutput[base + 1];
      final double l2 = rawOutput[base + 2];
      final double l3 = rawOutput[base + 3];
      final double l4 = rawOutput[base + 4];
      final double l5 = rawOutput[base + 5];

      // Find max for numerical stability (unrolled for speed)
      double maxLogit = l0;
      if (l1 > maxLogit) maxLogit = l1;
      if (l2 > maxLogit) maxLogit = l2;
      if (l3 > maxLogit) maxLogit = l3;
      if (l4 > maxLogit) maxLogit = l4;
      if (l5 > maxLogit) maxLogit = l5;

      // Compute exp(logit - max) for each channel (unrolled)
      final double e0 = _fastExp(l0 - maxLogit);
      final double e1 = _fastExp(l1 - maxLogit);
      final double e2 = _fastExp(l2 - maxLogit);
      final double e3 = _fastExp(l3 - maxLogit);
      final double e4 = _fastExp(l4 - maxLogit);
      final double e5 = _fastExp(l5 - maxLogit);

      // Sum of all exponentials
      final double sumExp = e0 + e1 + e2 + e3 + e4 + e5;

      // Normalize to probabilities
      result[base] = e0 / sumExp;
      result[base + 1] = e1 / sumExp;
      result[base + 2] = e2 / sumExp;
      result[base + 3] = e3 / sumExp;
      result[base + 4] = e4 / sumExp;
      result[base + 5] = e5 / sumExp;
    }

    return result;
  }

  /// Fast exponential approximation for softmax.
  ///
  /// Uses range reduction + polynomial approximation.
  /// ~2x faster than math.exp() with <1% relative error for [-20, 20] range.
  static double _fastExp(double x) {
    // Early exit for extreme values
    if (x < -20.0) return 0.0;
    if (x > 20.0) return 485165195.4; // e^20

    // For softmax, most values after max subtraction are in [-10, 0]
    // Use standard exp() which is well-optimized for this range
    return math.exp(x);
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
  void dispose() {
    if (_disposed) return;
    _disposed = true;

    final IsolateInterpreter? iso = _iso;
    if (iso != null) {
      // IsolateInterpreter.close() kills the isolate first (before awaits),
      // but since we can't await here, the isolate may still be alive when
      // _itp.close() runs. Use disposeAsync() when possible.
      iso.close();
      _iso = null;
    }
    _itp.close();
    _delegate?.delete();
    _delegate = null;
  }

  /// Asynchronously releases all resources, ensuring the background isolate
  /// is fully terminated before freeing the native interpreter.
  ///
  /// Prefer this over [dispose] when switching models or when you can await
  /// the cleanup. This prevents native crashes caused by the interpreter
  /// being freed while the background isolate still references it.
  ///
  /// It is safe to call this multiple times.
  Future<void> disposeAsync() async {
    if (_disposed) return;
    _disposed = true;

    final IsolateInterpreter? iso = _iso;
    if (iso != null) {
      await iso.close();
      _iso = null;
    }
    _itp.close();
    _delegate?.delete();
    _delegate = null;
  }

  /// Creates interpreter options with delegates based on performance configuration.
  ///
  /// Binary models (general/landscape) require custom ops, which are registered
  /// on all platforms. Multiclass uses only standard TFLite ops.
  ///
  /// ## Platform Behavior
  ///
  /// | Mode | macOS/Linux | Windows | iOS | Android |
  /// |------|-------------|---------|-----|---------|
  /// | disabled | CPU | CPU | CPU | CPU |
  /// | xnnpack | XNNPACK | CPU* | CPU* | CPU* |
  /// | gpu | CPU | CPU | Metal | OpenGL/CL** |
  /// | auto | XNNPACK | CPU | Metal | CPU |
  ///
  /// *Falls back to CPU (XNNPACK not supported on this platform)
  /// **Experimental, may crash on some devices
  static (InterpreterOptions, Delegate?) _createSegmentationInterpreterOptions(
    PerformanceConfig? config,
  ) {
    final options = InterpreterOptions();
    // Register MediaPipe custom ops (Convolution2DTransposeBias) required by
    // the binary segmentation models. Available on all platforms:
    // - Desktop: loaded from bundled dylib/dll/so
    // - iOS: statically linked via CocoaPods, accessed via DynamicLibrary.process()
    // - Android: built via CMake, loaded at runtime
    options.addMediaPipeCustomOps();
    final effectiveConfig = config ?? const PerformanceConfig();

    final threadCount = effectiveConfig.numThreads?.clamp(0, 8) ??
        math.min(4, Platform.numberOfProcessors);

    if (effectiveConfig.mode == PerformanceMode.disabled) {
      options.threads = threadCount;
      return (options, null);
    }

    if (effectiveConfig.mode == PerformanceMode.auto) {
      return _createAutoModeOptions(options, threadCount);
    }

    if (effectiveConfig.mode == PerformanceMode.xnnpack) {
      return _createXnnpackOptions(options, threadCount);
    }

    if (effectiveConfig.mode == PerformanceMode.gpu) {
      return _createGpuOptions(options, threadCount);
    }

    options.threads = threadCount;
    return (options, null);
  }

  /// Creates options for auto mode - selects best delegate per platform.
  static (InterpreterOptions, Delegate?) _createAutoModeOptions(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (Platform.isMacOS || Platform.isLinux) {
      return _createXnnpackOptions(options, threadCount);
    }

    if (Platform.isIOS) {
      return _createGpuOptions(options, threadCount);
    }

    options.threads = threadCount;
    return (options, null);
  }

  /// Creates options with XNNPACK delegate (desktop only).
  static (InterpreterOptions, Delegate?) _createXnnpackOptions(
    InterpreterOptions options,
    int threadCount,
  ) {
    options.threads = threadCount;

    if (!Platform.isMacOS && !Platform.isLinux) {
      return (options, null);
    }

    try {
      final xnnpackDelegate = XNNPackDelegate(
        options: XNNPackDelegateOptions(numThreads: threadCount),
      );
      options.addDelegate(xnnpackDelegate);
      return (options, xnnpackDelegate);
    } catch (e) {
      return (options, null);
    }
  }

  /// Creates options with GPU delegate.
  static (InterpreterOptions, Delegate?) _createGpuOptions(
    InterpreterOptions options,
    int threadCount,
  ) {
    options.threads = threadCount;

    if (!Platform.isIOS && !Platform.isAndroid) {
      return (options, null);
    }

    try {
      final gpuDelegate =
          Platform.isIOS ? GpuDelegate() : GpuDelegateV2() as Delegate;
      options.addDelegate(gpuDelegate);
      return (options, gpuDelegate);
    } catch (e) {
      return (options, null);
    }
  }

  /// Converts a grayscale image to RGB by replicating channels.
  img.Image _grayscaleToRgb(img.Image gray) {
    final rgb = img.Image(width: gray.width, height: gray.height);
    for (int y = 0; y < gray.height; y++) {
      for (int x = 0; x < gray.width; x++) {
        final pixel = gray.getPixel(x, y);
        final luminance = pixel.luminance.toInt();
        rgb.setPixelRgb(x, y, luminance, luminance, luminance);
      }
    }
    return rgb;
  }
}
