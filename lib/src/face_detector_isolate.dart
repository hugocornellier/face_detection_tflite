part of '../face_detection_tflite.dart';

/// Data passed to the background isolate during startup.
class _IsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData faceDetectionBytes;
  final TransferableTypedData faceLandmarkBytes;
  final TransferableTypedData irisLandmarkBytes;
  final TransferableTypedData embeddingBytes;
  final String modelName;
  final String performanceModeName;
  final int? numThreads;
  final int meshPoolSize;

  /// Optional segmentation model bytes.
  final TransferableTypedData? segmentationBytes;

  /// Segmentation configuration (if segmentation enabled).
  final Map<String, dynamic>? segmentationConfigMap;

  _IsolateStartupData({
    required this.sendPort,
    required this.faceDetectionBytes,
    required this.faceLandmarkBytes,
    required this.irisLandmarkBytes,
    required this.embeddingBytes,
    required this.modelName,
    required this.performanceModeName,
    required this.numThreads,
    required this.meshPoolSize,
    this.segmentationBytes,
    this.segmentationConfigMap,
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

  Isolate? _isolate;
  SendPort? _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  final Map<int, Completer<dynamic>> _pending = {};
  int _nextId = 0;
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

      // Build list of asset loads
      final assetFutures = [
        rootBundle.load(faceDetectionPath),
        rootBundle.load(faceLandmarkPath),
        rootBundle.load(irisLandmarkPath),
        rootBundle.load(embeddingPath),
      ];

      // Optionally load segmentation model
      if (withSegmentation) {
        const segmentationModelPath =
            'packages/face_detection_tflite/assets/models/$_segmentationModel';
        assetFutures.add(rootBundle.load(segmentationModelPath));
      }

      final results = await Future.wait(assetFutures);

      final faceDetectionBytes = results[0].buffer.asUint8List();
      final faceLandmarkBytes = results[1].buffer.asUint8List();
      final irisLandmarkBytes = results[2].buffer.asUint8List();
      final embeddingBytes = results[3].buffer.asUint8List();

      TransferableTypedData? segmentationBytesTransfer;
      if (withSegmentation && results.length > 4) {
        final segBytes = results[4].buffer.asUint8List();
        segmentationBytesTransfer = TransferableTypedData.fromList([segBytes]);
      }

      // Build segmentation config map for transfer
      Map<String, dynamic>? segConfigMap;
      if (withSegmentation) {
        final config = segmentationConfig ?? SegmentationConfig.safe;
        segConfigMap = {
          'performanceModeName': config.performanceConfig.mode.name,
          'numThreads': config.performanceConfig.numThreads,
          'maxOutputSize': config.maxOutputSize,
          'resizeStrategyName': config.resizeStrategy.name,
          'validateModel': config.validateModel,
        };
      }

      _isolate = await Isolate.spawn(
        _isolateEntry,
        _IsolateStartupData(
          sendPort: _receivePort.sendPort,
          faceDetectionBytes:
              TransferableTypedData.fromList([faceDetectionBytes]),
          faceLandmarkBytes:
              TransferableTypedData.fromList([faceLandmarkBytes]),
          irisLandmarkBytes:
              TransferableTypedData.fromList([irisLandmarkBytes]),
          embeddingBytes: TransferableTypedData.fromList([embeddingBytes]),
          modelName: model.name,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
          meshPoolSize: meshPoolSize,
          segmentationBytes: segmentationBytesTransfer,
          segmentationConfigMap: segConfigMap,
        ),
        debugName: 'FaceDetectorIsolate',
      );

      _segmentationInitialized = withSegmentation;

      final Completer<SendPort> initCompleter = Completer<SendPort>();
      late final StreamSubscription<dynamic> subscription;

      subscription = _receivePort.listen((message) {
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
        _handleResponse(message);
      });

      _sendPort = await initCompleter.future.timeout(
        const Duration(seconds: 30),
        onTimeout: () {
          subscription.cancel();
          throw TimeoutException(
              'FaceDetectorIsolate initialization timed out');
        },
      );

      _initialized = true;
    } catch (e) {
      _isolate?.kill(priority: Isolate.immediate);
      _receivePort.close();
      _initialized = false;
      rethrow;
    }
  }

  void _handleResponse(dynamic message) {
    if (message is! Map) return;

    final int? id = message['id'] as int?;
    if (id == null) return;

    final Completer<dynamic>? completer = _pending.remove(id);
    if (completer == null) return;

    if (message['error'] != null) {
      completer.completeError(StateError(message['error'] as String));
    } else {
      completer.complete(message['result']);
    }
  }

  Future<T> _sendRequest<T>(
      String operation, Map<String, dynamic> params) async {
    if (!_initialized && operation != 'init') {
      throw StateError(
        'FaceDetectorIsolate not initialized. Use FaceDetectorIsolate.spawn().',
      );
    }
    if (_sendPort == null) {
      throw StateError('FaceDetectorIsolate SendPort not available.');
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

  /// Detects faces in the given image entirely within the background isolate.
  ///
  /// All processing (image decoding, tensor conversion, face detection, mesh
  /// computation, and iris tracking) runs in the background isolate. The main
  /// thread only handles sending the image bytes and receiving the results.
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
  /// for (final face in faces) {
  ///   print('Bounding box: ${face.boundingBox}');
  ///   print('Left eye: ${face.landmarks.leftEye}');
  ///   if (face.mesh != null) {
  ///     print('Mesh has ${face.mesh!.length} points');
  ///   }
  /// }
  /// ```
  Future<List<Face>> detectFaces(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    final List<dynamic> result = await _sendRequest<List<dynamic>>('detect', {
      'bytes': TransferableTypedData.fromList([imageBytes]),
      'mode': mode.name,
    });

    return result
        .map((map) => Face.fromMap(Map<String, dynamic>.from(map as Map)))
        .toList();
  }

  /// Detects faces from an OpenCV [cv.Mat] in the background isolate.
  ///
  /// This method extracts the raw pixel data from the Mat, transfers it to the
  /// background isolate using zero-copy [TransferableTypedData], reconstructs
  /// the Mat in the isolate, and runs the full detection pipeline.
  ///
  /// This is ideal for live camera processing where you already have a cv.Mat
  /// from camera frame conversion, and need guaranteed non-blocking UI.
  ///
  /// Parameters:
  /// - [image]: An OpenCV Mat containing the image (typically BGR format)
  /// - [mode]: Detection mode controlling which features to compute
  ///
  /// Returns a list of [Face] objects, one per detected face.
  ///
  /// Example:
  /// ```dart
  /// // Convert camera frame to Mat
  /// final mat = cv.Mat.fromList(height, width, cv.MatType.CV_8UC3, bgrBytes);
  ///
  /// // Run detection in background isolate
  /// final faces = await detector.detectFacesFromMat(mat);
  ///
  /// // Don't forget to dispose the original Mat
  /// mat.dispose();
  /// ```
  ///
  /// Note: The Mat is reconstructed in the isolate and disposed there after
  /// detection. You are still responsible for disposing the original Mat
  /// in the main isolate.
  Future<List<Face>> detectFacesFromMat(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
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
    int matType = 16, // CV_8UC3
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    final List<dynamic> result =
        await _sendRequest<List<dynamic>>('detectMat', {
      'bytes': TransferableTypedData.fromList([bytes]),
      'width': width,
      'height': height,
      'matType': matType,
      'mode': mode.name,
    });

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
    final List<double> result = await _sendRequest<List<double>>('embedding', {
      'bytes': TransferableTypedData.fromList([imageBytes]),
      'face': face.toMap(),
    });

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
    final List<dynamic> result =
        await _sendRequest<List<dynamic>>('embeddings', {
      'bytes': TransferableTypedData.fromList([imageBytes]),
      'faces': faces.map((f) => f.toMap()).toList(),
    });

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
        await _sendRequest<Map<String, dynamic>>('segment', {
      'bytes': TransferableTypedData.fromList([imageBytes]),
      'outputFormat': outputFormat.index,
      'binaryThreshold': binaryThreshold,
    });

    return SegmentationMask.fromMap(result);
  }

  /// Segments an OpenCV [cv.Mat] image in the background isolate.
  ///
  /// This method extracts the raw pixel data from the Mat, transfers it to the
  /// background isolate using zero-copy [TransferableTypedData], reconstructs
  /// the Mat in the isolate, and runs the segmentation pipeline.
  ///
  /// Parameters:
  /// - [image]: An OpenCV Mat containing the image (typically BGR format)
  /// - [outputFormat]: Controls the output format for transfer efficiency
  /// - [binaryThreshold]: Threshold for binary output (default 0.5)
  ///
  /// Returns a [SegmentationMask] containing per-pixel foreground probabilities.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.Mat.fromList(height, width, cv.MatType.CV_8UC3, bgrBytes);
  /// final mask = await detector.getSegmentationMaskFromMat(mat);
  /// mat.dispose();
  /// ```
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
        await _sendRequest<Map<String, dynamic>>('segmentMat', {
      'bytes': TransferableTypedData.fromList([data]),
      'width': cols,
      'height': rows,
      'matType': type,
      'outputFormat': outputFormat.index,
      'binaryThreshold': binaryThreshold,
    });

    return SegmentationMask.fromMap(result);
  }

  /// Disposes the background isolate and releases all resources.
  ///
  /// This method:
  /// 1. Fails any pending detection requests
  /// 2. Disposes the [FaceDetector] inside the isolate
  /// 3. Kills the background isolate
  /// 4. Closes communication ports
  ///
  /// After calling dispose, the instance cannot be reused. Create a new
  /// instance with [spawn] if needed.
  Future<void> dispose() async {
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('FaceDetectorIsolate disposed'));
      }
    }
    _pending.clear();

    if (_initialized && _sendPort != null) {
      try {
        _sendPort!.send({'id': -1, 'op': 'dispose'});
      } catch (_) {}
    }

    _isolate?.kill(priority: Isolate.immediate);
    _receivePort.close();

    _isolate = null;
    _sendPort = null;
    _initialized = false;
    _segmentationInitialized = false;
  }

  /// Isolate entry point - handles initialization and message processing.
  @pragma('vm:entry-point')
  static void _isolateEntry(_IsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    FaceDetector? detector;
    SelfieSegmentation? segmenter;

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

      // Initialize segmentation if enabled
      if (data.segmentationBytes != null) {
        final segBytes = data.segmentationBytes!.materialize().asUint8List();

        // Reconstruct SegmentationConfig from map
        SegmentationConfig segConfig = SegmentationConfig.safe;
        if (data.segmentationConfigMap != null) {
          final cfgMap = data.segmentationConfigMap!;
          final perfMode = PerformanceMode.values.firstWhere(
            (m) => m.name == cfgMap['performanceModeName'],
          );
          final resizeStrategy = ResizeStrategy.values.firstWhere(
            (s) => s.name == cfgMap['resizeStrategyName'],
          );
          segConfig = SegmentationConfig(
            performanceConfig: PerformanceConfig(
              mode: perfMode,
              numThreads: cfgMap['numThreads'] as int?,
            ),
            maxOutputSize: cfgMap['maxOutputSize'] as int,
            resizeStrategy: resizeStrategy,
            validateModel: cfgMap['validateModel'] as bool,
          );
        }

        segmenter = await SelfieSegmentation.createFromBuffer(
          segBytes,
          config: segConfig,
        );
      }

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({'error': 'Isolate initialization failed: $e\n$st'});
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
            final Map<String, dynamic> faceMap =
                Map<String, dynamic>.from(message['face'] as Map);
            final Face face = Face.fromMap(faceMap);

            final embedding =
                await detector!.getFaceEmbedding(face, imageBytes);
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

            final embeddings =
                await detector!.getFaceEmbeddings(faces, imageBytes);
            final serialized = embeddings.map((e) => e?.toList()).toList();
            mainSendPort.send({'id': id, 'result': serialized});

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
              final mask = await segmenter!.call(imageBytes);
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
              final mask = await segmenter!.callFromMat(mat);
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
            detector?.dispose();
            detector = null;
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
  static Map<String, dynamic> _serializeMask(
    SegmentationMask mask,
    IsolateOutputFormat format,
    double binaryThreshold,
  ) {
    // Base metadata always included
    final result = <String, dynamic>{
      'width': mask.width,
      'height': mask.height,
      'originalWidth': mask.originalWidth,
      'originalHeight': mask.originalHeight,
      'padding': mask.padding,
    };

    switch (format) {
      case IsolateOutputFormat.float32:
        // Full precision - use toMap() data directly
        result['data'] = mask.data.toList();
        result['dataFormat'] = 'float32';
        break;

      case IsolateOutputFormat.uint8:
        // 8-bit grayscale - 4x smaller than float32
        final uint8Data = mask.toUint8();
        result['data'] = uint8Data.toList();
        result['dataFormat'] = 'uint8';
        break;

      case IsolateOutputFormat.binary:
        // Binary at threshold - smallest transfer
        final binaryData = mask.toBinary(threshold: binaryThreshold);
        result['data'] = binaryData.toList();
        result['dataFormat'] = 'binary';
        result['binaryThreshold'] = binaryThreshold;
        break;
    }

    return result;
  }
}
