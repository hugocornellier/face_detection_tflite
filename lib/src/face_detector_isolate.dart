part of '../face_detection_tflite.dart';

/// Data passed to the background isolate during startup.
class _IsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData faceDetectionBytes;
  final TransferableTypedData faceLandmarkBytes;
  final TransferableTypedData irisLandmarkBytes;
  final String modelName;
  final String performanceModeName;
  final int? numThreads;
  final int meshPoolSize;

  _IsolateStartupData({
    required this.sendPort,
    required this.faceDetectionBytes,
    required this.faceLandmarkBytes,
    required this.irisLandmarkBytes,
    required this.modelName,
    required this.performanceModeName,
    required this.numThreads,
    required this.meshPoolSize,
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

  /// Returns true if the isolate is initialized and ready for detection.
  bool get isReady => _initialized;

  /// Spawns a new isolate with an initialized [FaceDetector].
  ///
  /// The isolate loads all TFLite models during spawn, so this operation
  /// may take 100-500ms depending on the device.
  ///
  /// Parameters:
  /// - [model]: Face detection model variant (default: [FaceDetectionModel.backCamera])
  /// - [performanceConfig]: Hardware acceleration settings (default: XNNPACK enabled)
  /// - [meshPoolSize]: Number of mesh model instances for parallel face processing
  ///
  /// Example:
  /// ```dart
  /// // Default configuration
  /// final detector = await FaceDetectorIsolate.spawn();
  ///
  /// // Custom configuration
  /// final detector = await FaceDetectorIsolate.spawn(
  ///   model: FaceDetectionModel.frontCamera,
  ///   performanceConfig: PerformanceConfig.xnnpack(numThreads: 2),
  ///   meshPoolSize: 2,
  /// );
  /// ```
  static Future<FaceDetectorIsolate> spawn({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    PerformanceConfig performanceConfig = const PerformanceConfig.xnnpack(),
    int meshPoolSize = 3,
  }) async {
    final instance = FaceDetectorIsolate._();
    await instance._initialize(model, performanceConfig, meshPoolSize);
    return instance;
  }

  Future<void> _initialize(
    FaceDetectionModel model,
    PerformanceConfig performanceConfig,
    int meshPoolSize,
  ) async {
    if (_initialized) {
      throw StateError('FaceDetectorIsolate already initialized');
    }

    try {
      // Pre-load all model bytes in the main isolate (where rootBundle is available)
      final faceDetectionPath =
          'packages/face_detection_tflite/assets/models/${_nameFor(model)}';
      const faceLandmarkPath =
          'packages/face_detection_tflite/assets/models/$_faceLandmarkModel';
      const irisLandmarkPath =
          'packages/face_detection_tflite/assets/models/$_irisLandmarkModel';

      final results = await Future.wait([
        rootBundle.load(faceDetectionPath),
        rootBundle.load(faceLandmarkPath),
        rootBundle.load(irisLandmarkPath),
      ]);

      final faceDetectionBytes = results[0].buffer.asUint8List();
      final faceLandmarkBytes = results[1].buffer.asUint8List();
      final irisLandmarkBytes = results[2].buffer.asUint8List();

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
          modelName: model.name,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
          meshPoolSize: meshPoolSize,
        ),
        debugName: 'FaceDetectorIsolate',
      );

      final Completer<SendPort> initCompleter = Completer<SendPort>();
      late final StreamSubscription<dynamic> subscription;

      subscription = _receivePort.listen((message) {
        if (!initCompleter.isCompleted) {
          if (message is SendPort) {
            initCompleter.complete(message);
          } else if (message is Map && message['error'] != null) {
            // Handle initialization errors from the worker
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
          throw TimeoutException('FaceDetectorIsolate initialization timed out');
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

  Future<T> _sendRequest<T>(String operation, Map<String, dynamic> params) async {
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
    // Extract Mat properties for reconstruction in isolate
    final int rows = image.rows;
    final int cols = image.cols;
    final int type = image.type.value;

    // Get raw pixel data - this creates a copy
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
    final List<dynamic> result = await _sendRequest<List<dynamic>>('detectMat', {
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
    // Fail all pending requests
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('FaceDetectorIsolate disposed'));
      }
    }
    _pending.clear();

    // Tell the isolate to dispose its detector
    if (_initialized && _sendPort != null) {
      try {
        _sendPort!.send({'id': -1, 'op': 'dispose'});
      } catch (_) {
        // Isolate may already be dead
      }
    }

    // Kill isolate and close ports
    _isolate?.kill(priority: Isolate.immediate);
    _receivePort.close();

    _isolate = null;
    _sendPort = null;
    _initialized = false;
  }

  /// Isolate entry point - handles initialization and message processing.
  @pragma('vm:entry-point')
  static void _isolateEntry(_IsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    FaceDetector? detector;

    try {
      // Materialize the model bytes from TransferableTypedData
      final faceDetectionBytes =
          data.faceDetectionBytes.materialize().asUint8List();
      final faceLandmarkBytes =
          data.faceLandmarkBytes.materialize().asUint8List();
      final irisLandmarkBytes =
          data.irisLandmarkBytes.materialize().asUint8List();

      final model = FaceDetectionModel.values.firstWhere(
        (m) => m.name == data.modelName,
      );
      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );

      // Initialize FaceDetector with pre-loaded model bytes
      detector = FaceDetector();
      await detector.initializeFromBuffers(
        faceDetectionBytes: faceDetectionBytes,
        faceLandmarkBytes: faceLandmarkBytes,
        irisLandmarkBytes: irisLandmarkBytes,
        model: model,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
        meshPoolSize: data.meshPoolSize,
      );

      // Send the worker's send port to signal ready
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

            // Reconstruct cv.Mat from raw bytes
            final matType = cv.MatType(matTypeValue);
            final mat = cv.Mat.fromList(height, width, matType, matBytes);

            try {
              final faces = await detector!.detectFacesFromMat(mat, mode: mode);
              final serialized = faces.map((f) => f.toMap()).toList();
              mainSendPort.send({'id': id, 'result': serialized});
            } finally {
              // Always dispose the reconstructed Mat
              mat.dispose();
            }

          case 'dispose':
            detector?.dispose();
            detector = null;
            workerReceivePort.close();
            // Don't send response for dispose (id is -1)
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }
}
