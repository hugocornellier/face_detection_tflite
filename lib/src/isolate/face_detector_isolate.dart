part of '../../face_detection_tflite.dart';

/// Data passed to the detection isolate during startup.
class _DetectionIsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData faceDetectionBytes;
  final TransferableTypedData faceLandmarkBytes;
  final TransferableTypedData irisLandmarkBytes;
  final TransferableTypedData embeddingBytes;
  final String modelName;
  final String performanceModeName;
  final int? numThreads;
  final int meshPoolSize;

  _DetectionIsolateStartupData({
    required this.sendPort,
    required this.faceDetectionBytes,
    required this.faceLandmarkBytes,
    required this.irisLandmarkBytes,
    required this.embeddingBytes,
    required this.modelName,
    required this.performanceModeName,
    required this.numThreads,
    required this.meshPoolSize,
  });
}

/// Data passed to the segmentation isolate during startup.
class _SegmentationIsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData modelBytes;
  final String performanceModeName;
  final int? numThreads;
  final int maxOutputSize;
  final bool validateModel;
  final String modelName;
  final int modelIndex;

  _SegmentationIsolateStartupData({
    required this.sendPort,
    required this.modelBytes,
    required this.performanceModeName,
    required this.numThreads,
    required this.maxOutputSize,
    required this.validateModel,
    required this.modelName,
    required this.modelIndex,
  });
}

/// A wrapper that runs the entire face detection pipeline in a background isolate.
///
/// This class spawns a dedicated isolate containing a full [FaceDetector] instance,
/// ensuring that all face detection processing (image decoding, tensor conversion,
/// face detection, mesh computation, and iris tracking) runs completely off the
/// main UI thread.
///
/// ## Why Use This?
///
/// While [FaceDetector] already uses isolates internally for some operations,
/// [FaceDetectorIsolate] guarantees that **all** processing happens in the
/// background. This is ideal for:
///
/// - **Live camera processing** where any main-thread blocking causes dropped frames
/// - **Applications requiring smooth 60fps UI** during detection
/// - **Processing multiple images in parallel** without UI jank
///
/// ## Usage
///
/// ```dart
/// // Spawn isolate (loads models in background)
/// final detector = await FaceDetectorIsolate.spawn();
///
/// // All detection runs in background isolate - UI never blocked
/// final faces = await detector.detectFaces(imageBytes);
///
/// for (final face in faces) {
///   print('Face at: ${face.boundingBox.center}');
///   print('Mesh points: ${face.mesh?.length ?? 0}');
/// }
///
/// // Cleanup when done
/// await detector.dispose();
/// ```
///
/// ## Memory Considerations
///
/// The background isolate holds all TFLite models (~26-40MB for full pipeline).
/// Call [dispose] when finished to release these resources.
///
/// Image data is transferred to the worker using zero-copy
/// [TransferableTypedData], minimizing memory overhead.
///
/// See also:
/// - [FaceDetector] for the underlying detection implementation
/// - [FaceDetectionMode] for controlling detection features
/// - [FaceDetectionModel] for model selection
class FaceDetectorIsolate {
  FaceDetectorIsolate._();

  Isolate? _detectionIsolate;
  SendPort? _detectionSendPort;
  final ReceivePort _detectionReceivePort = ReceivePort();
  final Map<int, Completer<dynamic>> _detectionPending = {};
  int _detectionNextId = 0;

  Isolate? _segmentationIsolate;
  SendPort? _segmentationSendPort;
  final ReceivePort _segmentationReceivePort = ReceivePort();
  final Map<int, Completer<dynamic>> _segmentationPending = {};
  int _segmentationNextId = 0;

  bool _initialized = false;
  bool _segmentationInitialized = false;

  /// Returns true if the isolate is initialized and ready for detection.
  bool get isReady => _initialized;

  /// Returns true if segmentation model is loaded and ready.
  bool get isSegmentationReady => _segmentationInitialized;

  /// Spawns a new isolate with an initialized [FaceDetector].
  ///
  /// The isolate loads all TFLite models during spawn, so this operation
  /// may take 100-500ms depending on the device.
  ///
  /// Parameters:
  /// - [model]: Face detection model variant (default: [FaceDetectionModel.backCamera])
  /// - [performanceConfig]: Hardware acceleration settings (default: auto mode)
  /// - [meshPoolSize]: Number of mesh model instances for parallel face processing
  /// - [withSegmentation]: Whether to initialize segmentation model (default: false)
  /// - [segmentationConfig]: Configuration for segmentation (uses safe defaults if null)
  ///
  /// Example:
  /// ```dart
  /// // Default configuration (auto mode - optimal for each platform)
  /// final detector = await FaceDetectorIsolate.spawn();
  ///
  /// // With segmentation enabled
  /// final detector = await FaceDetectorIsolate.spawn(
  ///   model: FaceDetectionModel.frontCamera,
  ///   performanceConfig: PerformanceConfig.auto(),
  ///   meshPoolSize: 2,
  ///   withSegmentation: true,
  /// );
  /// ```
  static Future<FaceDetectorIsolate> spawn({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
    bool withSegmentation = false,
    SegmentationConfig? segmentationConfig,
  }) async {
    final instance = FaceDetectorIsolate._();
    await instance._initialize(
      model,
      performanceConfig,
      meshPoolSize,
      withSegmentation,
      segmentationConfig,
    );
    return instance;
  }

  Future<void> _initialize(
    FaceDetectionModel model,
    PerformanceConfig performanceConfig,
    int meshPoolSize,
    bool withSegmentation,
    SegmentationConfig? segmentationConfig,
  ) async {
    if (_initialized) {
      throw StateError('FaceDetectorIsolate already initialized');
    }

    try {
      final faceDetectionPath =
          'packages/face_detection_tflite/assets/models/${_nameFor(model)}';
      const faceLandmarkPath =
          'packages/face_detection_tflite/assets/models/$_faceLandmarkModel';
      const irisLandmarkPath =
          'packages/face_detection_tflite/assets/models/$_irisLandmarkModel';
      const embeddingPath =
          'packages/face_detection_tflite/assets/models/$_embeddingModel';

      final assetFutures = [
        rootBundle.load(faceDetectionPath),
        rootBundle.load(faceLandmarkPath),
        rootBundle.load(irisLandmarkPath),
        rootBundle.load(embeddingPath),
      ];

      if (withSegmentation) {
        final effectiveSegModel = _effectiveModel(
          segmentationConfig?.model ?? SegmentationModel.general,
        );
        final segModelFile = _modelFileFor(effectiveSegModel);
        final segmentationModelPath =
            'packages/face_detection_tflite/assets/models/$segModelFile';
        assetFutures.add(rootBundle.load(segmentationModelPath));
      }

      final results = await Future.wait(assetFutures);

      final faceDetectionBytes = results[0].buffer.asUint8List();
      final faceLandmarkBytes = results[1].buffer.asUint8List();
      final irisLandmarkBytes = results[2].buffer.asUint8List();
      final embeddingBytes = results[3].buffer.asUint8List();

      await _spawnDetectionIsolate(
        faceDetectionBytes: faceDetectionBytes,
        faceLandmarkBytes: faceLandmarkBytes,
        irisLandmarkBytes: irisLandmarkBytes,
        embeddingBytes: embeddingBytes,
        model: model,
        performanceConfig: performanceConfig,
        meshPoolSize: meshPoolSize,
      );

      if (withSegmentation && results.length > 4) {
        final segBytes = results[4].buffer.asUint8List();
        final config = segmentationConfig ?? SegmentationConfig.safe;
        await _spawnSegmentationIsolate(modelBytes: segBytes, config: config);
        _segmentationInitialized = true;
      }

      _initialized = true;
    } catch (e) {
      _detectionIsolate?.kill(priority: Isolate.immediate);
      _segmentationIsolate?.kill(priority: Isolate.immediate);
      _detectionReceivePort.close();
      _segmentationReceivePort.close();
      _initialized = false;
      _segmentationInitialized = false;
      rethrow;
    }
  }

  /// Spawns the detection isolate with face detection models.
  Future<void> _spawnDetectionIsolate({
    required Uint8List faceDetectionBytes,
    required Uint8List faceLandmarkBytes,
    required Uint8List irisLandmarkBytes,
    required Uint8List embeddingBytes,
    required FaceDetectionModel model,
    required PerformanceConfig performanceConfig,
    required int meshPoolSize,
  }) async {
    _detectionIsolate = await Isolate.spawn(
      _detectionIsolateEntry,
      _DetectionIsolateStartupData(
        sendPort: _detectionReceivePort.sendPort,
        faceDetectionBytes: TransferableTypedData.fromList([
          faceDetectionBytes,
        ]),
        faceLandmarkBytes: TransferableTypedData.fromList([faceLandmarkBytes]),
        irisLandmarkBytes: TransferableTypedData.fromList([irisLandmarkBytes]),
        embeddingBytes: TransferableTypedData.fromList([embeddingBytes]),
        modelName: model.name,
        performanceModeName: performanceConfig.mode.name,
        numThreads: performanceConfig.numThreads,
        meshPoolSize: meshPoolSize,
      ),
      debugName: 'FaceDetectorIsolate.detection',
    );

    _detectionSendPort = await _setupIsolateListener(
      receivePort: _detectionReceivePort,
      responseHandler: _handleDetectionResponse,
      timeout: const Duration(seconds: 30),
      timeoutMsg: 'Detection isolate initialization timed out',
    );
  }

  /// Spawns the segmentation isolate with segmentation model.
  Future<void> _spawnSegmentationIsolate({
    required Uint8List modelBytes,
    required SegmentationConfig config,
  }) async {
    final effectiveModel = _effectiveModel(config.model);
    _segmentationIsolate = await Isolate.spawn(
      _segmentationIsolateEntry,
      _SegmentationIsolateStartupData(
        sendPort: _segmentationReceivePort.sendPort,
        modelBytes: TransferableTypedData.fromList([modelBytes]),
        performanceModeName: config.performanceConfig.mode.name,
        numThreads: config.performanceConfig.numThreads,
        maxOutputSize: config.maxOutputSize,
        validateModel: config.validateModel,
        modelName: _modelFileFor(effectiveModel),
        modelIndex: effectiveModel.index,
      ),
      debugName: 'FaceDetectorIsolate.segmentation',
    );

    _segmentationSendPort = await _setupIsolateListener(
      receivePort: _segmentationReceivePort,
      responseHandler: _handleSegmentationResponse,
      timeout: const Duration(seconds: 15),
      timeoutMsg: 'Segmentation isolate initialization timed out',
    );
  }

  /// Sets up init handshake and message routing for an isolate.
  static Future<SendPort> _setupIsolateListener({
    required ReceivePort receivePort,
    required void Function(dynamic) responseHandler,
    required Duration timeout,
    required String timeoutMsg,
  }) async {
    final Completer<SendPort> initCompleter = Completer<SendPort>();
    late final StreamSubscription<dynamic> subscription;

    subscription = receivePort.listen((message) {
      if (!initCompleter.isCompleted) {
        if (message is SendPort) {
          initCompleter.complete(message);
        } else if (message is Map && message['error'] != null) {
          initCompleter.completeError(StateError(message['error'] as String));
        } else {
          initCompleter.completeError(
            StateError('Expected SendPort, got ${message.runtimeType}'),
          );
        }
        return;
      }
      responseHandler(message);
    });

    return initCompleter.future.timeout(
      timeout,
      onTimeout: () {
        subscription.cancel();
        throw TimeoutException(timeoutMsg);
      },
    );
  }

  void _handleDetectionResponse(dynamic msg) =>
      _handleIsolateResponse(msg, _detectionPending);

  void _handleSegmentationResponse(dynamic msg) =>
      _handleIsolateResponse(msg, _segmentationPending);

  static void _handleIsolateResponse(
    dynamic message,
    Map<int, Completer<dynamic>> pending,
  ) {
    if (message is! Map) return;

    final int? id = message['id'] as int?;
    if (id == null) return;

    final Completer<dynamic>? completer = pending.remove(id);
    if (completer == null) return;

    if (message['error'] != null) {
      completer.completeError(StateError(message['error'] as String));
    } else {
      completer.complete(message['result']);
    }
  }

  Future<T> _sendDetectionRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) async {
    if (!_initialized) {
      throw StateError(
        'FaceDetectorIsolate not initialized. Use FaceDetectorIsolate.spawn().',
      );
    }
    if (_detectionSendPort == null) {
      throw StateError('Detection isolate SendPort not available.');
    }

    final int id = _detectionNextId++;
    final Completer<T> completer = Completer<T>();
    _detectionPending[id] = completer;

    try {
      _detectionSendPort!.send({'id': id, 'op': operation, ...params});
      return await completer.future;
    } catch (e) {
      _detectionPending.remove(id);
      rethrow;
    }
  }

  Future<T> _sendSegmentationRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) async {
    if (!_segmentationInitialized) {
      throw StateError(
        'Segmentation not initialized. Use spawn(withSegmentation: true).',
      );
    }
    if (_segmentationSendPort == null) {
      throw StateError('Segmentation isolate SendPort not available.');
    }

    final int id = _segmentationNextId++;
    final Completer<T> completer = Completer<T>();
    _segmentationPending[id] = completer;

    try {
      _segmentationSendPort!.send({'id': id, 'op': operation, ...params});
      return await completer.future;
    } catch (e) {
      _segmentationPending.remove(id);
      rethrow;
    }
  }

  /// Detects faces in the given image entirely within the background isolate.
  ///
  /// All processing (image decoding, tensor conversion, face detection, mesh
  /// computation, and iris tracking) runs in the background isolate. The main
  /// thread only handles sending the image data and receiving the results.
  ///
  /// Parameters:
  /// - [imageBytes]: Encoded image data (JPEG, PNG, etc.)
  /// - [mode]: Detection mode controlling which features to compute
  ///   - [FaceDetectionMode.fast]: Bounding boxes + 6 landmarks only
  ///   - [FaceDetectionMode.standard]: Adds 468-point face mesh
  ///   - [FaceDetectionMode.full]: Adds iris tracking (default)
  ///
  /// Returns a list of [Face] objects, one per detected face.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes);
  /// ```
  Future<List<Face>> detectFaces(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'mode': mode.name,
      },
    );

    return result
        .map((map) => Face.fromMap(Map<String, dynamic>.from(map as Map)))
        .toList();
  }

  /// Detects faces in a pre-decoded [cv.Mat] image in the background isolate.
  ///
  /// The raw pixel data is extracted and transferred using zero-copy
  /// [TransferableTypedData]. The original Mat is NOT disposed by this method.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.Mat.fromList(height, width, cv.MatType.CV_8UC3, bgrBytes);
  /// final faces = await detector.detectFacesFromMat(mat);
  /// mat.dispose();
  /// ```
  Future<List<Face>> detectFacesFromMat(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    final int rows = image.rows;
    final int cols = image.cols;
    final int type = image.type.value;
    final Uint8List data = image.data;

    return detectFacesFromMatBytes(
      data,
      width: cols,
      height: rows,
      matType: type,
      mode: mode,
    );
  }

  /// Detects faces from raw pixel bytes in the background isolate.
  ///
  /// This is a lower-level API for when you already have raw pixel data
  /// (e.g., BGR bytes from a camera frame) without an existing cv.Mat.
  ///
  /// The bytes are transferred using zero-copy [TransferableTypedData],
  /// a cv.Mat is constructed in the background isolate, detection runs,
  /// and results are serialized back.
  ///
  /// Parameters:
  /// - [bytes]: Raw pixel data (typically BGR format, 3 bytes per pixel)
  /// - [width]: Image width in pixels
  /// - [height]: Image height in pixels
  /// - [matType]: OpenCV MatType value (default: CV_8UC3 = 16 for BGR)
  /// - [mode]: Detection mode controlling which features to compute
  ///
  /// Returns a list of [Face] objects, one per detected face.
  ///
  /// Example:
  /// ```dart
  /// // From camera YUV conversion
  /// final bgrBytes = convertYuvToBgr(cameraImage);
  ///
  /// final faces = await detector.detectFacesFromMatBytes(
  ///   bgrBytes,
  ///   width: cameraImage.width,
  ///   height: cameraImage.height,
  /// );
  /// ```
  Future<List<Face>> detectFacesFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'detectMat',
      {
        'bytes': TransferableTypedData.fromList([bytes]),
        'width': width,
        'height': height,
        'matType': matType,
        'mode': mode.name,
      },
    );

    return result
        .map((map) => Face.fromMap(Map<String, dynamic>.from(map as Map)))
        .toList();
  }

  /// Generates a face embedding for a detected face in the background isolate.
  ///
  /// This method runs the entire embedding pipeline (face alignment, crop,
  /// and embedding inference) in the background isolate, ensuring the UI
  /// thread is never blocked.
  ///
  /// Parameters:
  /// - [face]: A face detection result from [detectFaces]
  /// - [imageBytes]: Encoded image data (same image used for detection)
  ///
  /// Returns a [Float32List] containing the L2-normalized embedding vector.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes);
  /// final embedding = await detector.getFaceEmbedding(faces.first, imageBytes);
  ///
  /// // Compare with a reference
  /// final similarity = FaceDetector.compareFaces(embedding, referenceEmbedding);
  /// ```
  Future<Float32List> getFaceEmbedding(Face face, Uint8List imageBytes) async {
    final List<double> result = await _sendDetectionRequest<List<double>>(
      'embedding',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'face': face.toMap(),
      },
    );

    return Float32List.fromList(result);
  }

  /// Generates face embeddings for multiple detected faces in the background isolate.
  ///
  /// This is more efficient than calling [getFaceEmbedding] multiple times
  /// as it decodes the image only once in the isolate.
  ///
  /// Parameters:
  /// - [faces]: List of face detection results from [detectFaces]
  /// - [imageBytes]: Encoded image data (same image used for detection)
  ///
  /// Returns a list of [Float32List] embeddings in the same order as [faces].
  /// Faces that fail to produce embeddings will have null entries.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes);
  /// final embeddings = await detector.getFaceEmbeddings(faces, imageBytes);
  ///
  /// for (int i = 0; i < faces.length; i++) {
  ///   if (embeddings[i] != null) {
  ///     print('Face $i: ${embeddings[i]!.length} dimensions');
  ///   }
  /// }
  /// ```
  Future<List<Float32List?>> getFaceEmbeddings(
    List<Face> faces,
    Uint8List imageBytes,
  ) async {
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'embeddings',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'faces': faces.map((f) => f.toMap()).toList(),
      },
    );

    return result.map((dynamic item) {
      if (item == null) return null;
      final List<double> values = (item as List).cast<double>();
      return Float32List.fromList(values);
    }).toList();
  }

  /// Segments an image to separate foreground (person) from background.
  ///
  /// This method runs the segmentation pipeline entirely in the background
  /// isolate, ensuring the UI thread is never blocked.
  ///
  /// The [imageBytes] parameter should contain encoded image data (JPEG, PNG, etc.).
  /// For pre-decoded [cv.Mat] input, use [getSegmentationMaskFromMat] instead.
  ///
  /// Parameters:
  /// - [imageBytes]: Encoded image data (JPEG, PNG, etc.)
  /// - [outputFormat]: Controls the output format for transfer efficiency
  ///   - [IsolateOutputFormat.float32]: Full precision mask (largest transfer)
  ///   - [IsolateOutputFormat.uint8]: 8-bit grayscale (4x smaller)
  ///   - [IsolateOutputFormat.binary]: Binary mask at threshold (smallest)
  /// - [binaryThreshold]: Threshold for binary output (default 0.5)
  ///
  /// Returns a [SegmentationMask] containing per-pixel foreground probabilities.
  ///
  /// Throws [StateError] if segmentation was not initialized during [spawn].
  /// Throws [SegmentationException] on inference failure.
  ///
  /// Example:
  /// ```dart
  /// final detector = await FaceDetectorIsolate.spawn(withSegmentation: true);
  /// final mask = await detector.getSegmentationMask(imageBytes);
  /// final binary = mask.toBinary(threshold: 0.5);
  /// ```
  Future<SegmentationMask> getSegmentationMask(
    Uint8List imageBytes, {
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) async {
    if (!_segmentationInitialized) {
      throw StateError(
        'Segmentation not initialized. Use FaceDetectorIsolate.spawn(withSegmentation: true).',
      );
    }

    final Map<String, dynamic> result =
        await _sendSegmentationRequest<Map<String, dynamic>>('segment', {
      'bytes': TransferableTypedData.fromList([imageBytes]),
      'outputFormat': outputFormat.index,
      'binaryThreshold': binaryThreshold,
    });

    return _deserializeMask(result);
  }

  /// Segments a pre-decoded [cv.Mat] image in the background isolate.
  ///
  /// This is the cv.Mat variant of [getSegmentationMask]. The raw pixel data
  /// is extracted and transferred using zero-copy [TransferableTypedData].
  /// The original Mat is NOT disposed by this method.
  ///
  /// Throws [StateError] if segmentation was not initialized during [spawn].
  /// Throws [SegmentationException] on inference failure.
  Future<SegmentationMask> getSegmentationMaskFromMat(
    cv.Mat image, {
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) async {
    if (!_segmentationInitialized) {
      throw StateError(
        'Segmentation not initialized. Use FaceDetectorIsolate.spawn(withSegmentation: true).',
      );
    }

    final int rows = image.rows;
    final int cols = image.cols;
    final int type = image.type.value;
    final Uint8List data = image.data;

    final Map<String, dynamic> result =
        await _sendSegmentationRequest<Map<String, dynamic>>('segmentMat', {
      'bytes': TransferableTypedData.fromList([data]),
      'width': cols,
      'height': rows,
      'matType': type,
      'outputFormat': outputFormat.index,
      'binaryThreshold': binaryThreshold,
    });

    return _deserializeMask(result);
  }

  /// Detects faces and generates segmentation mask in parallel.
  ///
  /// This method runs face detection and segmentation simultaneously in
  /// separate isolates, returning results as soon as both complete. This
  /// provides optimal performance when both features are needed.
  ///
  /// Requires [withSegmentation: true] during [spawn].
  ///
  /// Parameters:
  /// - [imageBytes]: Encoded image data (JPEG, PNG, etc.)
  /// - [mode]: Detection mode controlling which features to compute
  /// - [outputFormat]: Controls the segmentation mask output format
  /// - [binaryThreshold]: Threshold for binary output format
  ///
  /// Returns a [DetectionWithSegmentationResult] containing both faces and mask.
  ///
  /// ## Performance
  ///
  /// Processing time is approximately `max(detectionTime, segmentationTime)`
  /// rather than their sum, typically 40-50% faster than sequential calls.
  ///
  /// Example:
  /// ```dart
  /// final detector = await FaceDetectorIsolate.spawn(withSegmentation: true);
  /// final result = await detector.detectFacesWithSegmentation(imageBytes);
  ///
  /// print('Found ${result.faces.length} faces');
  /// print('Mask: ${result.segmentationMask?.width}x${result.segmentationMask?.height}');
  /// print('Total time: ${result.totalTimeMs}ms (parallel processing)');
  /// ```
  Future<DetectionWithSegmentationResult> detectFacesWithSegmentation(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) async {
    if (!_initialized) {
      throw StateError('FaceDetectorIsolate not initialized');
    }
    if (!_segmentationInitialized) {
      throw StateError(
        'Segmentation not initialized. Use spawn(withSegmentation: true).',
      );
    }

    final detectionStopwatch = Stopwatch()..start();
    final segmentationStopwatch = Stopwatch()..start();

    final results = await Future.wait([
      _sendDetectionRequest<List<dynamic>>('detect', {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'mode': mode.name,
      }).then((result) {
        detectionStopwatch.stop();
        return result
            .map((m) => Face.fromMap(Map<String, dynamic>.from(m as Map)))
            .toList();
      }),
      _sendSegmentationRequest<Map<String, dynamic>>('segment', {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'outputFormat': outputFormat.index,
        'binaryThreshold': binaryThreshold,
      }).then((result) {
        segmentationStopwatch.stop();
        return _deserializeMask(result);
      }),
    ]);

    return DetectionWithSegmentationResult(
      faces: results[0] as List<Face>,
      segmentationMask: results[1] as SegmentationMask,
      detectionTimeMs: detectionStopwatch.elapsedMilliseconds,
      segmentationTimeMs: segmentationStopwatch.elapsedMilliseconds,
    );
  }

  /// Detects faces and generates segmentation mask in parallel from a [cv.Mat].
  ///
  /// This is the cv.Mat variant of [detectFacesWithSegmentation]. The raw pixel
  /// data is extracted and transferred to both isolates. The original Mat is
  /// NOT disposed by this method.
  ///
  /// Example:
  /// ```dart
  /// final detector = await FaceDetectorIsolate.spawn(withSegmentation: true);
  /// final result = await detector.detectFacesWithSegmentationFromMat(mat);
  /// mat.dispose();
  /// ```
  Future<DetectionWithSegmentationResult> detectFacesWithSegmentationFromMat(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) async {
    if (!_initialized) {
      throw StateError('FaceDetectorIsolate not initialized');
    }
    if (!_segmentationInitialized) {
      throw StateError(
        'Segmentation not initialized. Use spawn(withSegmentation: true).',
      );
    }

    final int rows = image.rows;
    final int cols = image.cols;
    final int type = image.type.value;
    final Uint8List data = image.data;

    final detectionStopwatch = Stopwatch()..start();
    final segmentationStopwatch = Stopwatch()..start();

    final results = await Future.wait([
      _sendDetectionRequest<List<dynamic>>('detectMat', {
        'bytes': TransferableTypedData.fromList([data]),
        'width': cols,
        'height': rows,
        'matType': type,
        'mode': mode.name,
      }).then((result) {
        detectionStopwatch.stop();
        return result
            .map((m) => Face.fromMap(Map<String, dynamic>.from(m as Map)))
            .toList();
      }),
      _sendSegmentationRequest<Map<String, dynamic>>('segmentMat', {
        'bytes': TransferableTypedData.fromList([data]),
        'width': cols,
        'height': rows,
        'matType': type,
        'outputFormat': outputFormat.index,
        'binaryThreshold': binaryThreshold,
      }).then((result) {
        segmentationStopwatch.stop();
        return _deserializeMask(result);
      }),
    ]);

    return DetectionWithSegmentationResult(
      faces: results[0] as List<Face>,
      segmentationMask: results[1] as SegmentationMask,
      detectionTimeMs: detectionStopwatch.elapsedMilliseconds,
      segmentationTimeMs: segmentationStopwatch.elapsedMilliseconds,
    );
  }

  /// Disposes both background isolates and releases all resources.
  ///
  /// This method:
  /// 1. Fails any pending detection and segmentation requests
  /// 2. Disposes the [FaceDetector] and [SelfieSegmentation] inside isolates
  /// 3. Kills both background isolates
  /// 4. Closes all communication ports
  ///
  /// After calling dispose, the instance cannot be reused. Create a new
  /// instance with [spawn] if needed.
  Future<void> dispose() async {
    for (final completer in _detectionPending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('FaceDetectorIsolate disposed'));
      }
    }
    _detectionPending.clear();

    for (final completer in _segmentationPending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('FaceDetectorIsolate disposed'));
      }
    }
    _segmentationPending.clear();

    if (_detectionSendPort != null) {
      try {
        _detectionSendPort!.send({'id': -1, 'op': 'dispose'});
      } catch (_) {}
    }
    if (_segmentationSendPort != null) {
      try {
        _segmentationSendPort!.send({'id': -1, 'op': 'dispose'});
      } catch (_) {}
    }

    _detectionIsolate?.kill(priority: Isolate.immediate);
    _segmentationIsolate?.kill(priority: Isolate.immediate);

    _detectionReceivePort.close();
    _segmentationReceivePort.close();

    _detectionIsolate = null;
    _segmentationIsolate = null;
    _detectionSendPort = null;
    _segmentationSendPort = null;
    _initialized = false;
    _segmentationInitialized = false;
  }

  /// Detection isolate entry point - handles face detection and embeddings.
  @pragma('vm:entry-point')
  static void _detectionIsolateEntry(_DetectionIsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    FaceDetector? detector;

    try {
      final faceDetectionBytes =
          data.faceDetectionBytes.materialize().asUint8List();
      final faceLandmarkBytes =
          data.faceLandmarkBytes.materialize().asUint8List();
      final irisLandmarkBytes =
          data.irisLandmarkBytes.materialize().asUint8List();
      final embeddingBytes = data.embeddingBytes.materialize().asUint8List();

      final model = FaceDetectionModel.values.firstWhere(
        (m) => m.name == data.modelName,
      );
      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );

      detector = FaceDetector();
      await detector.initializeFromBuffers(
        faceDetectionBytes: faceDetectionBytes,
        faceLandmarkBytes: faceLandmarkBytes,
        irisLandmarkBytes: irisLandmarkBytes,
        embeddingBytes: embeddingBytes,
        model: model,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
        meshPoolSize: data.meshPoolSize,
      );

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({
        'error': 'Detection isolate initialization failed: $e\n$st',
      });
      return;
    }

    workerReceivePort.listen((message) async {
      if (message is! Map) return;

      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;

      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'detect':
            if (detector == null || !detector!.isReady) {
              mainSendPort.send({
                'id': id,
                'error': 'FaceDetector not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List imageBytes = bb.asUint8List();
            final modeName = message['mode'] as String;
            final mode = FaceDetectionMode.values.firstWhere(
              (m) => m.name == modeName,
            );

            final faces = await detector!.detectFaces(imageBytes, mode: mode);
            final serialized = faces.map((f) => f.toMap()).toList();

            mainSendPort.send({'id': id, 'result': serialized});

          case 'detectMat':
            if (detector == null || !detector!.isReady) {
              mainSendPort.send({
                'id': id,
                'error': 'FaceDetector not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List matBytes = bb.asUint8List();
            final int width = message['width'] as int;
            final int height = message['height'] as int;
            final int matTypeValue = message['matType'] as int;
            final modeName = message['mode'] as String;
            final mode = FaceDetectionMode.values.firstWhere(
              (m) => m.name == modeName,
            );

            final matType = cv.MatType(matTypeValue);
            final mat = cv.Mat.fromList(height, width, matType, matBytes);

            try {
              final faces = await detector!.detectFacesFromMat(mat, mode: mode);
              final serialized = faces.map((f) => f.toMap()).toList();
              mainSendPort.send({'id': id, 'result': serialized});
            } finally {
              mat.dispose();
            }

          case 'embedding':
            if (detector == null || !detector!.isEmbeddingReady) {
              mainSendPort.send({
                'id': id,
                'error': 'FaceDetector embedding not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List imageBytes = bb.asUint8List();
            final Map<String, dynamic> faceMap = Map<String, dynamic>.from(
              message['face'] as Map,
            );
            final Face face = Face.fromMap(faceMap);

            final embedding = await detector!.getFaceEmbedding(
              face,
              imageBytes,
            );
            mainSendPort.send({'id': id, 'result': embedding.toList()});

          case 'embeddings':
            if (detector == null || !detector!.isEmbeddingReady) {
              mainSendPort.send({
                'id': id,
                'error': 'FaceDetector embedding not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List imageBytes = bb.asUint8List();
            final List<dynamic> faceMaps = message['faces'] as List;
            final List<Face> faces = faceMaps
                .map((m) => Face.fromMap(Map<String, dynamic>.from(m as Map)))
                .toList();

            final embeddings = await detector!.getFaceEmbeddings(
              faces,
              imageBytes,
            );
            final serialized = embeddings.map((e) => e?.toList()).toList();
            mainSendPort.send({'id': id, 'result': serialized});

          case 'dispose':
            detector?.dispose();
            detector = null;
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }

  /// Segmentation isolate entry point - handles selfie segmentation.
  @pragma('vm:entry-point')
  static void _segmentationIsolateEntry(
    _SegmentationIsolateStartupData data,
  ) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    SelfieSegmentation? segmenter;

    try {
      final modelBytes = data.modelBytes.materialize().asUint8List();

      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );

      final SegmentationModel model = SegmentationModel.values[data.modelIndex];

      final config = SegmentationConfig(
        model: model,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
        maxOutputSize: data.maxOutputSize,
        validateModel: data.validateModel,
        useIsolate: false,
      );

      segmenter = await SelfieSegmentation.createFromBuffer(
        modelBytes,
        config: config,
      );

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({
        'error': 'Segmentation isolate initialization failed: $e\n$st',
      });
      return;
    }

    workerReceivePort.listen((message) async {
      if (message is! Map) return;

      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;

      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'segment':
            if (segmenter == null) {
              mainSendPort.send({
                'id': id,
                'error': 'Segmentation not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List imageBytes = bb.asUint8List();
            final int outputFormatIndex = message['outputFormat'] as int;
            final double binaryThreshold = message['binaryThreshold'] as double;

            try {
              final mask = await segmenter!.callFromBytes(imageBytes);
              final serialized = _serializeMask(
                mask,
                IsolateOutputFormat.values[outputFormatIndex],
                binaryThreshold,
              );
              mainSendPort.send({'id': id, 'result': serialized});
            } on SegmentationException catch (e) {
              mainSendPort.send({
                'id': id,
                'error': 'SegmentationException(${e.code}): ${e.message}',
              });
            }

          case 'segmentMat':
            if (segmenter == null) {
              mainSendPort.send({
                'id': id,
                'error': 'Segmentation not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List matBytes = bb.asUint8List();
            final int width = message['width'] as int;
            final int height = message['height'] as int;
            final int matTypeValue = message['matType'] as int;
            final int outputFormatIndex = message['outputFormat'] as int;
            final double binaryThreshold = message['binaryThreshold'] as double;

            final matType = cv.MatType(matTypeValue);
            final mat = cv.Mat.fromList(height, width, matType, matBytes);

            try {
              final mask = await segmenter!.call(mat);
              final serialized = _serializeMask(
                mask,
                IsolateOutputFormat.values[outputFormatIndex],
                binaryThreshold,
              );
              mainSendPort.send({'id': id, 'result': serialized});
            } on SegmentationException catch (e) {
              mainSendPort.send({
                'id': id,
                'error': 'SegmentationException(${e.code}): ${e.message}',
              });
            } finally {
              mat.dispose();
            }

          case 'dispose':
            segmenter?.dispose();
            segmenter = null;
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }

  /// Serializes a segmentation mask for isolate transfer.
  ///
  /// Converts the mask to the requested output format to reduce transfer overhead.
  /// For [MulticlassSegmentationMask], also serializes per-class probability data.
  static Map<String, dynamic> _serializeMask(
    SegmentationMask mask,
    IsolateOutputFormat format,
    double binaryThreshold,
  ) {
    final result = <String, dynamic>{
      'width': mask.width,
      'height': mask.height,
      'originalWidth': mask.originalWidth,
      'originalHeight': mask.originalHeight,
      'padding': mask.padding,
    };

    switch (format) {
      case IsolateOutputFormat.float32:
        result['data'] = mask.data.toList();
        result['dataFormat'] = 'float32';
        break;

      case IsolateOutputFormat.uint8:
        final uint8Data = mask.toUint8();
        result['data'] = uint8Data.toList();
        result['dataFormat'] = 'uint8';
        break;

      case IsolateOutputFormat.binary:
        final binaryData = mask.toBinary(threshold: binaryThreshold);
        result['data'] = binaryData.toList();
        result['dataFormat'] = 'binary';
        result['binaryThreshold'] = binaryThreshold;
        break;
    }

    if (mask is MulticlassSegmentationMask) {
      result['classData'] = mask._classData.toList();
    }

    return result;
  }

  /// Deserializes a segmentation mask from isolate transfer data.
  ///
  /// Reconstructs a [MulticlassSegmentationMask] when classData is present,
  /// otherwise returns a standard [SegmentationMask].
  static SegmentationMask _deserializeMask(Map<String, dynamic> map) {
    final baseMask = SegmentationMask.fromMap(map);

    if (map['classData'] != null) {
      final List rawClassData = map['classData'] as List;
      final Float32List classData = Float32List.fromList(
        rawClassData.cast<double>(),
      );
      return MulticlassSegmentationMask(
        data: baseMask.data,
        width: baseMask.width,
        height: baseMask.height,
        originalWidth: baseMask.originalWidth,
        originalHeight: baseMask.originalHeight,
        padding: baseMask.padding,
        classData: classData,
      );
    }

    return baseMask;
  }
}

@visibleForTesting
Map<String, dynamic> testSerializeMask(
  SegmentationMask mask,
  IsolateOutputFormat format,
  double binaryThreshold,
) =>
    FaceDetectorIsolate._serializeMask(mask, format, binaryThreshold);

@visibleForTesting
SegmentationMask testDeserializeMask(Map<String, dynamic> map) =>
    FaceDetectorIsolate._deserializeMask(map);
