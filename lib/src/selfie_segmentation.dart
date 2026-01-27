part of '../face_detection_tflite.dart';

/// Model file name for selfie segmentation (multiclass model).
const _segmentationModel = 'selfie_multiclass.tflite';

/// Minimum input image size (smaller images rejected).
const int kMinSegmentationInputSize = 16;

/// Model input/output dimensions for the multiclass model.
const int _segmentationInputSize = 256;
const int _segmentationOutputChannels = 6;

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

/// Performs selfie/person segmentation using MediaPipe's multiclass model.
///
/// Generates a per-pixel probability mask separating foreground (person)
/// from background. Works on full images - no face detection required.
///
/// The multiclass model outputs 6 classes: background, hair, body skin,
/// face skin, clothes, and other. By default, all non-background classes
/// are combined into a single "person" mask.
///
/// ## Platform Support
///
/// Uses only standard TFLite ops - works on all platforms without custom ops.
///
/// | Platform | Delegate | Notes |
/// |----------|----------|-------|
/// | iOS | Metal GPU | Reliable, recommended |
/// | Android | GPU/CPU | Works with both delegates |
/// | macOS | CPU | Stable |
/// | Linux | CPU | Stable |
/// | Windows | CPU | Stable |
///
/// ## Example
///
/// ```dart
/// final segmenter = await SelfieSegmentation.create();
/// final mask = await segmenter(imageBytes);
/// final binary = mask.toBinary(threshold: 0.5);
/// segmenter.dispose();
/// ```
///
/// ## Memory Considerations
///
/// - Model memory: ~16MB
/// - Input buffer: ~768KB (256x256x3 float32)
/// - Output buffer: ~1.5MB (256x256x6 float32)
/// - Combined person mask: ~256KB (256x256 float32)
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
  Delegate? _delegate;
  bool _delegateFailed = false;
  bool _disposed = false;

  late final Tensor _inputTensor;
  late final Tensor _outputTensor;
  late final Float32List _inputBuf;
  late final Float32List _outputBuf;
  late final List<List<List<List<double>>>> _input4dCache;
  late final int _outW;
  late final int _outH;

  SelfieSegmentation._(
      this._itp, this._inW, this._inH, this._outChannels, this._config);

  /// Creates and initializes a selfie segmentation model instance.
  ///
  /// This factory method loads the MediaPipe multiclass segmentation TensorFlow
  /// Lite model from package assets and prepares it for inference.
  ///
  /// [config]: Configuration for delegates and output limits.
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
  /// // Default configuration
  /// final segmenter = await SelfieSegmentation.create();
  ///
  /// // With custom configuration
  /// final segmenter = await SelfieSegmentation.create(
  ///   config: SegmentationConfig.performance,
  /// );
  /// ```
  static Future<SelfieSegmentation> create({
    SegmentationConfig config = const SegmentationConfig(),
  }) async {
    Delegate? delegate;
    InterpreterOptions options;
    bool delegateFailed = false;

    try {
      final result =
          _createSegmentationInterpreterOptions(config.performanceConfig);
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
        'packages/face_detection_tflite/assets/models/$_segmentationModel',
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
        options.threads = config.performanceConfig.numThreads ?? 4;

        try {
          itp = await Interpreter.fromAsset(
            'packages/face_detection_tflite/assets/models/$_segmentationModel',
            options: options,
          );
        } catch (retryError) {
          throw SegmentationException(
            SegmentationError.modelNotFound,
            'Failed to load model $_segmentationModel (even after delegate fallback): $retryError',
            retryError,
          );
        }
      } else {
        throw SegmentationException(
          SegmentationError.modelNotFound,
          'Failed to load model $_segmentationModel: $e',
          e,
        );
      }
    }

    // Validate model if configured
    if (config.validateModel) {
      _validateModel(itp);
    }

    itp.resizeInputTensor(
        0, [1, _segmentationInputSize, _segmentationInputSize, 3]);
    itp.allocateTensors();

    final obj = SelfieSegmentation._(itp, _segmentationInputSize,
        _segmentationInputSize, _segmentationOutputChannels, config);
    obj._delegate = delegate;
    obj._delegateFailed = delegateFailed;
    await obj._initializeTensors();
    return obj;
  }

  /// Creates a selfie segmentation model from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetectorIsolate] to initialize models
  /// in a background isolate where asset loading is not available.
  ///
  /// [modelBytes]: Raw TFLite model file contents.
  /// [config]: Configuration for delegates and output limits.
  ///
  /// Example:
  /// ```dart
  /// final bytes = await rootBundle.load('assets/models/selfie_multiclass.tflite');
  /// final segmenter = await SelfieSegmentation.createFromBuffer(
  ///   bytes.buffer.asUint8List(),
  /// );
  /// ```
  static Future<SelfieSegmentation> createFromBuffer(
    Uint8List modelBytes, {
    SegmentationConfig config = const SegmentationConfig(),
  }) async {
    Delegate? delegate;
    InterpreterOptions options;
    bool delegateFailed = false;

    try {
      final result =
          _createSegmentationInterpreterOptions(config.performanceConfig);
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
      _validateModel(itp);
    }

    itp.resizeInputTensor(
        0, [1, _segmentationInputSize, _segmentationInputSize, 3]);
    itp.allocateTensors();

    final obj = SelfieSegmentation._(itp, _segmentationInputSize,
        _segmentationInputSize, _segmentationOutputChannels, config);
    obj._delegate = delegate;
    obj._delegateFailed = delegateFailed;
    await obj._initializeTensors();
    return obj;
  }

  /// Shared tensor initialization logic.
  Future<void> _initializeTensors() async {
    _inputTensor = _itp.getInputTensor(0);
    _outputTensor = _itp.getOutputTensor(0);
    _inputBuf = _inputTensor.data.buffer.asFloat32List();
    _outputBuf = _outputTensor.data.buffer.asFloat32List();

    // Output shape is [1, H, W, 6] for multiclass segmentation
    final outShape = _outputTensor.shape;
    _outH = outShape.length >= 2 ? outShape[1] : _inH;
    _outW = outShape.length >= 3 ? outShape[2] : _inW;

    _input4dCache = createNHWCTensor4D(_inH, _inW);
    _iso = await IsolateInterpreter.create(address: _itp.address);
  }

  /// Validates model tensor shapes match expected multiclass model.
  static void _validateModel(Interpreter itp) {
    final inputShape = itp.getInputTensor(0).shape;
    final outputShape = itp.getOutputTensor(0).shape;

    // Validate input shape [1, 256, 256, 3]
    if (inputShape.length != 4 || inputShape[3] != 3) {
      throw SegmentationException(
        SegmentationError.unexpectedTensorShape,
        'Expected input shape [1, 256, 256, 3], got $inputShape',
      );
    }

    // Validate output shape [1, 256, 256, 6]
    if (outputShape.length != 4 || outputShape[3] != _segmentationOutputChannels) {
      throw SegmentationException(
        SegmentationError.unexpectedTensorShape,
        'Expected output shape [1, 256, 256, $_segmentationOutputChannels], got $outputShape',
      );
    }
  }

  /// Number of output channels (6 for multiclass model).
  int get outputChannels => _outChannels;

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

  /// Segments encoded image bytes.
  ///
  /// [imageBytes]: Encoded image (JPEG, PNG, or other supported format).
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities at model output resolution.
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

    Float32List rawOutput;
    try {
      if (_iso == null) {
        _inputBuf.setAll(0, pack.tensorNHWC);
        _itp.invoke();
        rawOutput = Float32List.fromList(_outputBuf);
      } else {
        fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
        final List<List<List<List<List<double>>>>> inputs = [_input4dCache];

        final List<int> outShape = _outputTensor.shape;
        final Map<int, Object> outputs = <int, Object>{
          0: allocTensorShape(outShape),
        };

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

    // Combine all person classes (non-background) into a single mask
    final maskData = _combinePersonClasses(rawOutput, _outW, _outH);

    return SegmentationMask(
      data: maskData,
      width: _outW,
      height: _outH,
      originalWidth: decoded.width,
      originalHeight: decoded.height,
      padding: pack.padding,
    );
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
  Future<SegmentationMask> callFromMat(cv.Mat image,
      {Float32List? buffer}) async {
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

    final ImageTensor pack = convertImageToTensorFromMat(
      image,
      outW: _inW,
      outH: _inH,
      buffer: buffer,
    );

    Float32List rawOutput;
    try {
      if (_iso == null) {
        _inputBuf.setAll(0, pack.tensorNHWC);
        _itp.invoke();
        rawOutput = Float32List.fromList(_outputBuf);
      } else {
        fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
        final List<List<List<List<List<double>>>>> inputs = [_input4dCache];

        final List<int> outShape = _outputTensor.shape;
        final Map<int, Object> outputs = <int, Object>{
          0: allocTensorShape(outShape),
        };

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

    // Combine all person classes (non-background) into a single mask
    final maskData = _combinePersonClasses(rawOutput, _outW, _outH);

    return SegmentationMask(
      data: maskData,
      width: _outW,
      height: _outH,
      originalWidth: image.cols,
      originalHeight: image.rows,
      padding: pack.padding,
    );
  }

  /// Combines multiclass output into a single person probability mask.
  ///
  /// The model outputs 6 channels: [background, hair, body, face, clothes, other].
  /// This method computes person probability as (1 - softmax(background)).
  static Float32List _combinePersonClasses(
      Float32List rawOutput, int width, int height) {
    final int numPixels = width * height;
    final result = Float32List(numPixels);

    for (int i = 0; i < numPixels; i++) {
      final int baseIdx = i * _segmentationOutputChannels;

      // Apply softmax to get proper probabilities
      double maxLogit = rawOutput[baseIdx];
      for (int c = 1; c < _segmentationOutputChannels; c++) {
        final v = rawOutput[baseIdx + c];
        if (v > maxLogit) maxLogit = v;
      }

      double sumExp = 0.0;
      double bgExp = 0.0;
      for (int c = 0; c < _segmentationOutputChannels; c++) {
        final exp = _fastExp(rawOutput[baseIdx + c] - maxLogit);
        sumExp += exp;
        if (c == 0) bgExp = exp;
      }

      // Person probability = 1 - background probability
      final bgProb = bgExp / sumExp;
      result[i] = (1.0 - bgProb).clamp(0.0, 1.0);
    }

    return result;
  }

  /// Fast exponential approximation for softmax.
  static double _fastExp(double x) {
    // Clamp to avoid overflow/underflow
    if (x < -20.0) return 0.0;
    if (x > 20.0) return 485165195.4; // e^20
    return math.exp(x);
  }

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the segmentation model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// It is safe to call dispose multiple times.
  void dispose() {
    if (_disposed) return;
    _disposed = true;

    _delegate?.delete();
    _delegate = null;
    final IsolateInterpreter? iso = _iso;
    if (iso != null) {
      iso.close();
      _iso = null;
    }
    _itp.close();
  }

  /// Creates interpreter options with delegates based on performance configuration.
  ///
  /// The multiclass model uses only standard TFLite ops, so no custom ops needed.
  static (InterpreterOptions, Delegate?) _createSegmentationInterpreterOptions(
      PerformanceConfig? config) {
    final options = InterpreterOptions();
    final effectiveConfig = config ?? const PerformanceConfig();
    options.threads = effectiveConfig.numThreads ?? 4;

    // No custom ops needed - multiclass model uses standard TFLite ops only

    return (options, null);
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
