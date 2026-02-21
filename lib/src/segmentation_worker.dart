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
/// Model memory usage depends on the selected model variant
/// (see [SegmentationModel]). Call [dispose] when done to free resources.
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
  SegmentationModel _model = SegmentationModel.general;

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
  /// [config] controls model selection, delegate selection, and other options.
  ///
  /// Throws [StateError] if already initialized.
  /// Throws [SegmentationException] if model loading fails.
  Future<void> initialize({
    SegmentationConfig config = const SegmentationConfig(),
  }) async {
    if (_initialized) {
      throw StateError('SegmentationWorker already initialized');
    }

    _model = _effectiveModel(config.model);
    final modelFile = _modelFileFor(_model);

    final ByteData modelData = await rootBundle.load(
      'packages/face_detection_tflite/assets/models/$modelFile',
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

      final Map result = await _sendRequest<Map>('init', {
        'modelBytes': TransferableTypedData.fromList([modelBytes]),
        'numThreads': config.performanceConfig.numThreads ?? 4,
        'mode': config.performanceConfig.mode.index,
        'modelIndex': config.model.index,
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
  /// Returns a [SegmentationMask] (or [MulticlassSegmentationMask] for
  /// multiclass model) with per-pixel probabilities.
  ///
  /// Throws [SegmentationException] on decode or inference failure.
  Future<SegmentationMask> segment(Uint8List imageBytes) async {
    final Map result = await _sendRequest<Map>('segment', {
      'imageBytes': TransferableTypedData.fromList([imageBytes]),
    });

    return _rebuildMask(result);
  }

  /// Segments a pre-decoded cv.Mat image.
  ///
  /// [mat] should be a BGR or BGRA cv.Mat (OpenCV format).
  /// The Mat data is transferred efficiently to the worker isolate.
  ///
  /// Note: The Mat is NOT disposed by this method.
  ///
  /// Returns a [SegmentationMask] (or [MulticlassSegmentationMask] for
  /// multiclass model) with per-pixel probabilities.
  Future<SegmentationMask> segmentMat(cv.Mat mat) async {
    if (mat.isEmpty) {
      throw SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Input Mat is empty',
      );
    }

    final Uint8List matBytes = mat.data;
    final int channels = mat.channels;

    final Map result = await _sendRequest<Map>('segmentMat', {
      'matBytes': TransferableTypedData.fromList([matBytes]),
      'width': mat.cols,
      'height': mat.rows,
      'channels': channels,
    });

    return _rebuildMask(result);
  }

  /// Rebuilds a [SegmentationMask] or [MulticlassSegmentationMask] from
  /// isolate transfer data.
  SegmentationMask _rebuildMask(Map result) {
    final ByteBuffer maskBB =
        (result['mask'] as TransferableTypedData).materialize();
    final Float32List maskData = maskBB.asUint8List().buffer.asFloat32List();
    final int width = result['width'] as int;
    final int height = result['height'] as int;
    final int originalWidth = result['originalWidth'] as int;
    final int originalHeight = result['originalHeight'] as int;
    final List<double> padding = (result['padding'] as List).cast<double>();

    if (_model == SegmentationModel.multiclass && result['classData'] != null) {
      final ByteBuffer classBB =
          (result['classData'] as TransferableTypedData).materialize();
      final Float32List classData =
          classBB.asUint8List().buffer.asFloat32List();
      return MulticlassSegmentationMask(
        data: maskData,
        width: width,
        height: height,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
        padding: padding,
        classData: classData,
      );
    }

    return SegmentationMask(
      data: maskData,
      width: width,
      height: height,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      padding: padding,
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
            mainSendPort.send({'id': id, 'error': 'Unknown operation: $op'});
        }
      } catch (e, stackTrace) {
        mainSendPort.send({'id': id, 'error': 'Worker error: $e\n$stackTrace'});
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
    final int modelIndex = params['modelIndex'] as int;
    final SegmentationModel model = SegmentationModel.values[modelIndex];

    final int inW = _inputWidthFor(model);
    final int inH = _inputHeightFor(model);

    final options = InterpreterOptions();
    options.addMediaPipeCustomOps();
    options.threads = numThreads;

    Delegate? delegate;

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

    interpreter.resizeInputTensor(0, [1, inH, inW, 3]);
    interpreter.allocateTensors();

    final inputTensor = interpreter.getInputTensor(0);
    final outputTensor = interpreter.getOutputTensor(0);
    final outShape = outputTensor.shape;

    return _SegmentationWorkerState(
      interpreter: interpreter,
      delegate: delegate,
      model: model,
      inputWidth: inW,
      inputHeight: inH,
      outputWidth: outShape.length >= 3 ? outShape[2] : inW,
      outputHeight: outShape.length >= 2 ? outShape[1] : inH,
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

    final cv.Mat image = cv.Mat.fromList(
      height,
      width,
      cv.MatType.CV_8UC(channels),
      matBytes,
    );
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

    final ImageTensor pack = convertImageToTensorFromMat(
      image,
      outW: state.inputWidth,
      outH: state.inputHeight,
    );

    state.inputBuffer.setAll(0, pack.tensorNHWC);

    state.interpreter.invoke();

    final Float32List rawOutput = Float32List.fromList(state.outputBuffer);

    final Map<String, dynamic> result = {
      'width': state.outputWidth,
      'height': state.outputHeight,
      'originalWidth': originalWidth,
      'originalHeight': originalHeight,
      'padding': pack.padding,
    };

    if (state.model == SegmentationModel.multiclass) {
      final classProbs = SelfieSegmentation._computeClassProbabilities(
        rawOutput,
        state.outputWidth,
        state.outputHeight,
      );
      final int numPixels = state.outputWidth * state.outputHeight;
      final personMask = Float32List(numPixels);
      for (int i = 0; i < numPixels; i++) {
        personMask[i] = 1.0 - classProbs[i * 6];
      }
      result['mask'] = TransferableTypedData.fromList([
        personMask.buffer.asUint8List(),
      ]);
      result['classData'] = TransferableTypedData.fromList([
        classProbs.buffer.asUint8List(),
      ]);
    } else {
      final int numPixels = state.outputWidth * state.outputHeight;
      final personMask = Float32List(numPixels);
      for (int i = 0; i < numPixels; i++) {
        personMask[i] = rawOutput[i];
      }
      result['mask'] = TransferableTypedData.fromList([
        personMask.buffer.asUint8List(),
      ]);
    }

    return result;
  }
}

/// Internal state for the segmentation worker isolate.
class _SegmentationWorkerState {
  final Interpreter interpreter;
  final Delegate? delegate;
  final SegmentationModel model;
  final int inputWidth;
  final int inputHeight;
  final int outputWidth;
  final int outputHeight;
  final Float32List inputBuffer;
  final Float32List outputBuffer;

  _SegmentationWorkerState({
    required this.interpreter,
    required this.delegate,
    required this.model,
    required this.inputWidth,
    required this.inputHeight,
    required this.outputWidth,
    required this.outputHeight,
    required this.inputBuffer,
    required this.outputBuffer,
  });
}
