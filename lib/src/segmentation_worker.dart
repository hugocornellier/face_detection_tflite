part of '../face_detection_tflite.dart';

/// A dedicated background isolate for selfie segmentation inference.
///
/// This worker runs TFLite inference in a background isolate to avoid blocking
/// the main UI thread, while using [TransferableTypedData] for efficient
/// zero-copy data transfer (unlike [IsolateInterpreter] which uses slow
/// nested list serialization).
///
/// ## Performance
///
/// - **No UI blocking**: Inference runs entirely in background isolate
/// - **Fast data transfer**: ~1ms via TransferableTypedData vs ~60-80ms with IsolateInterpreter
/// - **Overall**: ~30-50ms per inference (same as direct invoke, but non-blocking)
///
/// ## Usage
///
/// ```dart
/// final worker = SegmentationWorker();
/// await worker.initialize();
///
/// // Process frames without blocking UI
/// final mask = await worker.segment(imageBytes);
/// print('Foreground: ${mask.data.where((v) => v > 0.5).length} pixels');
///
/// worker.dispose();
/// ```
///
/// ## Memory Management
///
/// The worker holds model memory (~16MB) in the background isolate.
/// Call [dispose] when done to free resources.
class SegmentationWorker {
  /// Creates an uninitialized worker; call [initialize] before use.
  SegmentationWorker();

  Isolate? _isolate;
  SendPort? _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  final Map<int, Completer<dynamic>> _pending = {};
  int _nextId = 0;
  bool _initialized = false;
  int _inputWidth = 256;
  int _inputHeight = 256;
  int _outputWidth = 256;
  int _outputHeight = 256;

  /// Returns true if the worker has been initialized and is ready for inference.
  bool get isInitialized => _initialized;

  /// Model input width in pixels.
  int get inputWidth => _inputWidth;

  /// Model input height in pixels.
  int get inputHeight => _inputHeight;

  /// Output mask width in pixels.
  int get outputWidth => _outputWidth;

  /// Output mask height in pixels.
  int get outputHeight => _outputHeight;

  /// Initializes the worker isolate and loads the segmentation model.
  ///
  /// This spawns a background isolate, transfers the model bytes, and creates
  /// the TFLite interpreter inside the isolate.
  ///
  /// [config] controls delegate selection and other options.
  ///
  /// Throws [StateError] if already initialized.
  /// Throws [SegmentationException] if model loading fails.
  Future<void> initialize({
    PerformanceConfig performanceConfig = const PerformanceConfig.auto(),
  }) async {
    if (_initialized) {
      throw StateError('SegmentationWorker already initialized');
    }

    // Load model bytes in main isolate (has access to assets)
    final ByteData modelData = await rootBundle.load(
      'packages/face_detection_tflite/assets/models/$_segmentationModel',
    );
    final Uint8List modelBytes = modelData.buffer.asUint8List();

    try {
      _isolate = await Isolate.spawn(
        _isolateEntry,
        _receivePort.sendPort,
        debugName: 'SegmentationWorker',
      );

      final Completer<SendPort> initCompleter = Completer<SendPort>();
      late final StreamSubscription subscription;

      subscription = _receivePort.listen((message) {
        if (!initCompleter.isCompleted) {
          if (message is SendPort) {
            initCompleter.complete(message);
          } else {
            initCompleter.completeError(
              StateError('Expected SendPort, got ${message.runtimeType}'),
            );
          }
          return;
        }
        _handleResponse(message);
      });

      _sendPort = await initCompleter.future.timeout(
        const Duration(seconds: 10),
        onTimeout: () {
          subscription.cancel();
          throw TimeoutException('Worker initialization timed out');
        },
      );

      // Initialize interpreter in the worker isolate
      final Map result = await _sendRequest<Map>('init', {
        'modelBytes': TransferableTypedData.fromList([modelBytes]),
        'numThreads': performanceConfig.numThreads ?? 4,
        'mode': performanceConfig.mode.index,
      });

      _inputWidth = result['inputWidth'] as int;
      _inputHeight = result['inputHeight'] as int;
      _outputWidth = result['outputWidth'] as int;
      _outputHeight = result['outputHeight'] as int;

      _initialized = true;
    } catch (e) {
      _isolate?.kill(priority: Isolate.immediate);
      _receivePort.close();
      _initialized = false;
      if (e is SegmentationException) rethrow;
      throw SegmentationException(
        SegmentationError.interpreterCreationFailed,
        'Failed to initialize SegmentationWorker: $e',
        e,
      );
    }
  }

  void _handleResponse(dynamic message) {
    if (message is! Map) return;

    final int? id = message['id'] as int?;
    if (id == null) return;

    final Completer? completer = _pending.remove(id);
    if (completer == null) return;

    if (message['error'] != null) {
      completer.completeError(
        SegmentationException(
          SegmentationError.inferenceFailed,
          message['error'] as String,
        ),
      );
    } else {
      completer.complete(message['result']);
    }
  }

  Future<T> _sendRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) async {
    if (!_initialized && operation != 'init') {
      throw StateError(
        'SegmentationWorker not initialized. Call initialize() first.',
      );
    }
    if (_sendPort == null) {
      throw StateError('Worker not ready');
    }

    final int id = _nextId++;
    final Completer<T> completer = Completer<T>();
    _pending[id] = completer;

    try {
      _sendPort!.send({'id': id, 'op': operation, ...params});
      return await completer.future;
    } catch (e) {
      _pending.remove(id);
      rethrow;
    }
  }

  /// Segments encoded image bytes.
  ///
  /// [imageBytes] should contain encoded image data (JPEG, PNG, etc.).
  /// The image is decoded, preprocessed, and segmented in the worker isolate.
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities.
  ///
  /// Throws [SegmentationException] on decode or inference failure.
  Future<SegmentationMask> segment(Uint8List imageBytes) async {
    final Map result = await _sendRequest<Map>('segment', {
      'imageBytes': TransferableTypedData.fromList([imageBytes]),
    });

    final ByteBuffer maskBB =
        (result['mask'] as TransferableTypedData).materialize();
    final Float32List maskData = maskBB.asUint8List().buffer.asFloat32List();

    return SegmentationMask(
      data: maskData,
      width: result['width'] as int,
      height: result['height'] as int,
      originalWidth: result['originalWidth'] as int,
      originalHeight: result['originalHeight'] as int,
      padding: (result['padding'] as List).cast<double>(),
    );
  }

  /// Segments a pre-decoded cv.Mat image.
  ///
  /// [mat] should be a BGR or BGRA cv.Mat (OpenCV format).
  /// The Mat data is transferred efficiently to the worker isolate.
  ///
  /// Note: The Mat is NOT disposed by this method.
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities.
  Future<SegmentationMask> segmentMat(cv.Mat mat) async {
    if (mat.isEmpty) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Input Mat is empty',
      );
    }

    // Get raw bytes from Mat
    final Uint8List matBytes = mat.data;
    final int channels = mat.channels;

    final Map result = await _sendRequest<Map>('segmentMat', {
      'matBytes': TransferableTypedData.fromList([matBytes]),
      'width': mat.cols,
      'height': mat.rows,
      'channels': channels,
    });

    final ByteBuffer maskBB =
        (result['mask'] as TransferableTypedData).materialize();
    final Float32List maskData = maskBB.asUint8List().buffer.asFloat32List();

    return SegmentationMask(
      data: maskData,
      width: result['width'] as int,
      height: result['height'] as int,
      originalWidth: result['originalWidth'] as int,
      originalHeight: result['originalHeight'] as int,
      padding: (result['padding'] as List).cast<double>(),
    );
  }

  /// Releases all resources held by the worker.
  ///
  /// This kills the background isolate and frees model memory.
  /// After calling dispose, this instance cannot be used for inference.
  void dispose() {
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('Worker disposed'));
      }
    }
    _pending.clear();

    _isolate?.kill(priority: Isolate.immediate);
    _receivePort.close();

    _isolate = null;
    _sendPort = null;
    _initialized = false;
  }

  // ========== Isolate Entry Point ==========

  @pragma('vm:entry-point')
  static void _isolateEntry(SendPort mainSendPort) {
    final ReceivePort workerReceivePort = ReceivePort();
    mainSendPort.send(workerReceivePort.sendPort);

    _SegmentationWorkerState? state;

    workerReceivePort.listen((message) async {
      if (message is! Map) return;

      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;
      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'init':
            state = await _initInterpreter(message);
            mainSendPort.send({
              'id': id,
              'result': {
                'inputWidth': state!.inputWidth,
                'inputHeight': state!.inputHeight,
                'outputWidth': state!.outputWidth,
                'outputHeight': state!.outputHeight,
              },
            });
            break;

          case 'segment':
            if (state == null) {
              mainSendPort.send({
                'id': id,
                'error': 'Interpreter not initialized',
              });
              return;
            }
            final result = _runSegmentation(state!, message);
            mainSendPort.send({'id': id, 'result': result});
            break;

          case 'segmentMat':
            if (state == null) {
              mainSendPort.send({
                'id': id,
                'error': 'Interpreter not initialized',
              });
              return;
            }
            final result = _runSegmentationFromMat(state!, message);
            mainSendPort.send({'id': id, 'result': result});
            break;

          default:
            mainSendPort.send({
              'id': id,
              'error': 'Unknown operation: $op',
            });
        }
      } catch (e, stackTrace) {
        mainSendPort.send({
          'id': id,
          'error': 'Worker error: $e\n$stackTrace',
        });
      }
    });
  }

  static Future<_SegmentationWorkerState> _initInterpreter(
    Map<dynamic, dynamic> params,
  ) async {
    final ByteBuffer modelBB =
        (params['modelBytes'] as TransferableTypedData).materialize();
    final Uint8List modelBytes = modelBB.asUint8List();
    final int numThreads = params['numThreads'] as int;
    final int modeIndex = params['mode'] as int;
    final PerformanceMode mode = PerformanceMode.values[modeIndex];

    // Create interpreter options with appropriate delegate
    final options = InterpreterOptions();
    options.threads = numThreads;

    Delegate? delegate;

    // Apply delegate based on mode (same logic as SelfieSegmentation)
    if (mode == PerformanceMode.auto) {
      if (Platform.isMacOS || Platform.isLinux) {
        try {
          delegate = XNNPackDelegate(
            options: XNNPackDelegateOptions(numThreads: numThreads),
          );
          options.addDelegate(delegate);
        } catch (_) {}
      } else if (Platform.isIOS) {
        try {
          delegate = GpuDelegate();
          options.addDelegate(delegate);
        } catch (_) {}
      }
    } else if (mode == PerformanceMode.xnnpack) {
      if (Platform.isMacOS || Platform.isLinux) {
        try {
          delegate = XNNPackDelegate(
            options: XNNPackDelegateOptions(numThreads: numThreads),
          );
          options.addDelegate(delegate);
        } catch (_) {}
      }
    } else if (mode == PerformanceMode.gpu) {
      if (Platform.isIOS) {
        try {
          delegate = GpuDelegate();
          options.addDelegate(delegate);
        } catch (_) {}
      } else if (Platform.isAndroid) {
        try {
          delegate = GpuDelegateV2();
          options.addDelegate(delegate);
        } catch (_) {}
      }
    }

    final Interpreter interpreter = Interpreter.fromBuffer(
      modelBytes,
      options: options,
    );

    interpreter.resizeInputTensor(
      0,
      [1, _segmentationInputSize, _segmentationInputSize, 3],
    );
    interpreter.allocateTensors();

    final inputTensor = interpreter.getInputTensor(0);
    final outputTensor = interpreter.getOutputTensor(0);
    final outShape = outputTensor.shape;

    return _SegmentationWorkerState(
      interpreter: interpreter,
      delegate: delegate,
      inputWidth: _segmentationInputSize,
      inputHeight: _segmentationInputSize,
      outputWidth: outShape.length >= 3 ? outShape[2] : _segmentationInputSize,
      outputHeight: outShape.length >= 2 ? outShape[1] : _segmentationInputSize,
      inputBuffer: inputTensor.data.buffer.asFloat32List(),
      outputBuffer: outputTensor.data.buffer.asFloat32List(),
    );
  }

  static Map<String, dynamic> _runSegmentation(
    _SegmentationWorkerState state,
    Map<dynamic, dynamic> params,
  ) {
    final ByteBuffer imageBB =
        (params['imageBytes'] as TransferableTypedData).materialize();
    final Uint8List imageBytes = imageBB.asUint8List();

    // Decode image using OpenCV (fast native decode)
    final cv.Mat image = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    if (image.isEmpty) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to decode image',
      );
    }

    try {
      return _runInference(state, image);
    } finally {
      image.dispose();
    }
  }

  static Map<String, dynamic> _runSegmentationFromMat(
    _SegmentationWorkerState state,
    Map<dynamic, dynamic> params,
  ) {
    final ByteBuffer matBB =
        (params['matBytes'] as TransferableTypedData).materialize();
    final Uint8List matBytes = matBB.asUint8List();
    final int width = params['width'] as int;
    final int height = params['height'] as int;
    final int channels = params['channels'] as int;

    // Reconstruct Mat from bytes
    final cv.Mat image =
        cv.Mat.fromList(height, width, cv.MatType.CV_8UC(channels), matBytes);
    if (image.isEmpty) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to reconstruct Mat',
      );
    }

    try {
      return _runInference(state, image);
    } finally {
      image.dispose();
    }
  }

  static Map<String, dynamic> _runInference(
    _SegmentationWorkerState state,
    cv.Mat image,
  ) {
    final int originalWidth = image.cols;
    final int originalHeight = image.rows;

    // Convert Mat to tensor using OpenCV pipeline
    final ImageTensor pack = convertImageToTensorFromMat(
      image,
      outW: state.inputWidth,
      outH: state.inputHeight,
    );

    // Copy tensor data to input buffer
    state.inputBuffer.setAll(0, pack.tensorNHWC);

    // Run inference
    state.interpreter.invoke();

    // Copy output and run softmax
    final Float32List rawOutput = Float32List.fromList(state.outputBuffer);
    final Float32List maskData = _combinePersonClasses(
      rawOutput,
      state.outputWidth,
      state.outputHeight,
    );

    // Transfer mask data back via TransferableTypedData
    return {
      'mask': TransferableTypedData.fromList([maskData.buffer.asUint8List()]),
      'width': state.outputWidth,
      'height': state.outputHeight,
      'originalWidth': originalWidth,
      'originalHeight': originalHeight,
      'padding': pack.padding,
    };
  }

  /// Combines multiclass output into a single person probability mask.
  /// (Same optimized implementation as SelfieSegmentation)
  static Float32List _combinePersonClasses(
    Float32List rawOutput,
    int width,
    int height,
  ) {
    final int numPixels = width * height;
    final result = Float32List(numPixels);

    for (int i = 0; i < numPixels; i++) {
      final int baseIdx = i * _segmentationOutputChannels;

      // Read all 6 logits and find max in single pass
      final double l0 = rawOutput[baseIdx];
      final double l1 = rawOutput[baseIdx + 1];
      final double l2 = rawOutput[baseIdx + 2];
      final double l3 = rawOutput[baseIdx + 3];
      final double l4 = rawOutput[baseIdx + 4];
      final double l5 = rawOutput[baseIdx + 5];

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

      // Person probability = 1 - background probability
      result[i] = 1.0 - e0 / sumExp;
    }

    return result;
  }

  static double _fastExp(double x) {
    if (x < -20.0) return 0.0;
    if (x > 20.0) return 485165195.4;
    return math.exp(x);
  }
}

/// Internal state for the segmentation worker isolate.
class _SegmentationWorkerState {
  final Interpreter interpreter;
  final Delegate? delegate;
  final int inputWidth;
  final int inputHeight;
  final int outputWidth;
  final int outputHeight;
  final Float32List inputBuffer;
  final Float32List outputBuffer;

  _SegmentationWorkerState({
    required this.interpreter,
    required this.delegate,
    required this.inputWidth,
    required this.inputHeight,
    required this.outputWidth,
    required this.outputHeight,
    required this.inputBuffer,
    required this.outputBuffer,
  });
}
