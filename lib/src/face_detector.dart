part of '../face_detection_tflite.dart';

/// A complete face detection and analysis system using TensorFlow Lite models.
///
/// This class orchestrates four TensorFlow Lite models to provide comprehensive
/// facial analysis:
/// - Face detection with 6 keypoints (eyes, nose, mouth, tragions)
/// - 468-point face mesh for detailed facial geometry
/// - Iris landmark detection with 76 points per eye (71 eye mesh + 5 iris keypoints)
/// - Face embedding for identity vectors (192-dimensional)
///
/// All inference runs in a background isolate, ensuring the UI thread is never
/// blocked during detection.
///
/// ## Usage
///
/// ```dart
/// final detector = FaceDetector();
/// await detector.initialize();
///
/// // Detect faces with full mesh and iris tracking
/// final faces = await detector.detectFaces(
///   imageBytes,
///   mode: FaceDetectionMode.full,
/// );
///
/// // Clean up when done
/// await detector.dispose();
/// ```
///
/// ## Detection Modes
///
/// - [FaceDetectionMode.fast]: Basic detection with 6 keypoints only
/// - [FaceDetectionMode.standard]: Adds 468-point face mesh
/// - [FaceDetectionMode.full]: Adds iris tracking (default, most detailed)
///
/// ## Lifecycle
///
/// 1. Create instance with `FaceDetector()`
/// 2. Call [initialize] to load TensorFlow Lite models
/// 3. Check [isReady] to verify models are loaded
/// 4. Call [detectFaces] to analyze images
/// 5. Call [dispose] when done to free resources
///
/// See also:
/// - [initialize] for model loading options
/// - [detectFaces] for the main detection API
/// - [Face] for the structure of detection results
class FaceDetector {
  /// Creates a new face detector instance.
  ///
  /// The detector is not ready for use until you call [initialize].
  FaceDetector();

  _FaceDetectorWorker? _worker;
  IsolateRpcClient? _segmentationRpc;
  bool _segmentationInitialized = false;

  /// Returns true if all models are loaded and ready for inference.
  ///
  /// You must call [initialize] before this returns true.
  bool get isReady => _worker?.isReady ?? false;

  bool get isEmbeddingReady => isReady;

  /// Returns true if the segmentation model is loaded and ready.
  bool get isSegmentationReady => _segmentationInitialized;

  /// Loads the face detection, face mesh, iris landmark, and embedding models
  /// and prepares the interpreters for inference in a background isolate.
  ///
  /// This must be called before running any detections.
  /// Calling [initialize] twice without [dispose] throws [StateError].
  ///
  /// The [model] argument specifies which detection model variant to load
  /// (for example, `FaceDetectionModel.backCamera`).
  ///
  /// The [meshPoolSize] parameter controls how many face mesh model instances
  /// to create for parallel processing. Default is 3, which allows up to 3 faces
  /// to have their meshes computed in parallel. Increase for better multi-face
  /// performance (at the cost of ~7-10MB memory per additional instance).
  ///
  /// The [performanceConfig] parameter controls hardware acceleration delegates.
  /// By default, auto mode selects the optimal delegate per platform:
  /// - iOS: Metal GPU delegate
  /// - Android/macOS/Linux/Windows: XNNPACK (2-5x SIMD acceleration)
  ///
  /// The [withSegmentation] flag initializes the segmentation model at the same
  /// time, avoiding an extra model-load step later. Pass [segmentationConfig]
  /// to customize segmentation behaviour.
  ///
  /// Example:
  /// ```dart
  /// // Default (auto mode - optimal for each platform)
  /// final detector = FaceDetector();
  /// await detector.initialize();
  ///
  /// // Force CPU-only execution
  /// await detector.initialize(
  ///   performanceConfig: PerformanceConfig.disabled,
  /// );
  ///
  /// // Include segmentation from the start
  /// await detector.initialize(withSegmentation: true);
  /// ```
  Future<void> initialize({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
    bool withSegmentation = false,
    SegmentationConfig? segmentationConfig,
  }) async {
    if (isReady) {
      throw StateError('FaceDetector already initialized');
    }

    final worker = _FaceDetectorWorker();
    IsolateRpcClient? segmentationRpc;

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
        final effectiveSegModel =
            segmentationConfig?.model ?? SegmentationModel.general;
        final segModelFile = _modelFileFor(effectiveSegModel);
        assetFutures.add(rootBundle.load(
          'packages/face_detection_tflite/assets/models/$segModelFile',
        ));
      }

      final results = await Future.wait(assetFutures);

      await worker.initialize(
        faceDetectionBytes: results[0].buffer.asUint8List(),
        faceLandmarkBytes: results[1].buffer.asUint8List(),
        irisLandmarkBytes: results[2].buffer.asUint8List(),
        embeddingBytes: results[3].buffer.asUint8List(),
        model: model,
        performanceConfig: performanceConfig,
        meshPoolSize: meshPoolSize,
      );

      if (withSegmentation && results.length > 4) {
        final segBytes = results[4].buffer.asUint8List();
        final config = segmentationConfig ?? SegmentationConfig.safe;
        segmentationRpc = IsolateRpcClient();
        await _spawnSegmentationIsolate(
          rpc: segmentationRpc,
          modelBytes: segBytes,
          config: config,
        );
        _segmentationInitialized = true;
      }

      _worker = worker;
      _segmentationRpc = segmentationRpc;
    } catch (e) {
      segmentationRpc?.failAllAndDispose(disposeOp: 'dispose');
      if (worker.isReady) {
        await worker.dispose();
      }
      _segmentationInitialized = false;
      rethrow;
    }
  }

  /// Initializes the optional segmentation model.
  ///
  /// Call this after [initialize] to enable segmentation features.
  /// Does nothing if segmentation is already initialized.
  ///
  /// [config]: Segmentation configuration. If null, uses [SegmentationConfig.safe].
  ///
  /// Throws [SegmentationException] on model load failure.
  ///
  /// Example:
  /// ```dart
  /// final detector = FaceDetector();
  /// await detector.initialize();
  /// await detector.initializeSegmentation();
  ///
  /// final mask = await detector.getSegmentationMask(imageBytes);
  /// ```
  Future<void> initializeSegmentation({SegmentationConfig? config}) async {
    _requireReady();
    if (_segmentationInitialized) return;
    final effectiveConfig = config ?? SegmentationConfig.safe;
    final segModelFile = _modelFileFor(effectiveConfig.model);
    final segmentationRpc = IsolateRpcClient();
    final data = await rootBundle.load(
      'packages/face_detection_tflite/assets/models/$segModelFile',
    );
    try {
      await _spawnSegmentationIsolate(
        rpc: segmentationRpc,
        modelBytes: data.buffer.asUint8List(),
        config: effectiveConfig,
      );
      _segmentationRpc = segmentationRpc;
      _segmentationInitialized = true;
    } catch (_) {
      segmentationRpc.failAllAndDispose(disposeOp: 'dispose');
      rethrow;
    }
  }

  /// Detects faces in encoded image bytes and returns detailed results.
  ///
  /// The [imageBytes] parameter should contain encoded image data (JPEG, PNG, etc.).
  /// For pre-decoded [cv.Mat] input, use [detectFacesFromMat] instead.
  ///
  /// The [mode] parameter controls which features are computed:
  /// - [FaceDetectionMode.fast]: Only detection and landmarks
  /// - [FaceDetectionMode.standard]: Adds 468-point face mesh
  /// - [FaceDetectionMode.full]: Adds iris tracking (152 points: 76 per eye)
  ///
  /// Returns a [List] of [Face] objects, one per detected face.
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  /// Throws [FormatException] if the image bytes cannot be decoded.
  Future<List<Face>> detectFaces(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    _requireReady();
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'mode': mode.name,
      },
    );
    return _deserializeFacesFast(result);
  }

  /// Detects faces in a pre-decoded [cv.Mat] image.
  ///
  /// The Mat is NOT disposed by this method — caller is responsible for disposal.
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  Future<List<Face>> detectFacesFromMat(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    _requireReady();
    final f = _extractMatFields(image);
    return detectFacesFromMatBytes(
      f.data,
      width: f.width,
      height: f.height,
      matType: f.matType,
      mode: mode,
    );
  }

  /// Detects faces from raw pixel bytes without constructing a [cv.Mat] first.
  ///
  /// This avoids the overhead of building a Mat on the calling thread —
  /// the bytes are transferred via zero-copy [TransferableTypedData] and the
  /// Mat is reconstructed inside the background isolate.
  ///
  /// Parameters:
  /// - [bytes]: Raw pixel data (e.g. BGR, BGRA)
  /// - [width]: Image width in pixels
  /// - [height]: Image height in pixels
  /// - [matType]: OpenCV MatType value (default: CV_8UC3 = 16 for BGR)
  /// - [mode]: Detection mode controlling which features to compute
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  Future<List<Face>> detectFacesFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    _requireReady();
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
    return _deserializeFacesFast(result);
  }

  /// Generates a face embedding (identity vector) for a detected face.
  ///
  /// The [face] parameter should be a face detection result from [detectFaces].
  ///
  /// The [imageBytes] parameter should contain the encoded image data.
  /// For pre-decoded [cv.Mat] input, use [getFaceEmbeddingFromMat] instead.
  ///
  /// Returns a [Float32List] containing the L2-normalized embedding vector.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes);
  /// final embedding = await detector.getFaceEmbedding(faces.first, imageBytes);
  ///
  /// final similarity = FaceDetector.compareFaces(embedding, referenceEmbedding);
  /// if (similarity > 0.6) print('Same person!');
  /// ```
  Future<Float32List> getFaceEmbedding(
    Face face,
    Uint8List imageBytes,
  ) async {
    _requireReady();
    final List<double> result = await _sendDetectionRequest<List<double>>(
      'embedding',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'face': face.toMap(),
      },
    );
    return Float32List.fromList(result);
  }

  /// Generates a face embedding from raw pixel bytes without constructing a [cv.Mat].
  ///
  /// Mirrors [detectFacesFromMatBytes] — use this when you already have decoded
  /// pixel data to avoid the overhead of building a Mat on the calling thread.
  ///
  /// Parameters:
  /// - [bytes]: Raw pixel data (e.g. BGR, BGRA)
  /// - [width]: Image width in pixels
  /// - [height]: Image height in pixels
  /// - [matType]: OpenCV MatType value (default: CV_8UC3 = 16 for BGR)
  Future<Float32List> getFaceEmbeddingFromMatBytes(
    Face face,
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) async {
    _requireReady();
    final List<double> result = await _sendDetectionRequest<List<double>>(
      'embeddingMat',
      {
        'bytes': TransferableTypedData.fromList([bytes]),
        'width': width,
        'height': height,
        'matType': matType,
        'face': face.toMap(),
      },
    );
    return Float32List.fromList(result);
  }

  /// Generates a face embedding from a pre-decoded [cv.Mat] image.
  ///
  /// The Mat is NOT disposed by this method — caller is responsible for disposal.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
  /// final faces = await detector.detectFacesFromMat(mat);
  /// final embedding = await detector.getFaceEmbeddingFromMat(faces.first, mat);
  /// mat.dispose();
  /// ```
  Future<Float32List> getFaceEmbeddingFromMat(
    Face face,
    cv.Mat image,
  ) {
    _requireReady();
    final f = _extractMatFields(image);
    return getFaceEmbeddingFromMatBytes(
      face,
      f.data,
      width: f.width,
      height: f.height,
      matType: f.matType,
    );
  }

  /// Generates face embeddings for multiple detected faces.
  ///
  /// More efficient than calling [getFaceEmbedding] multiple times because it
  /// decodes the image only once in the isolate.
  ///
  /// Returns a list of [Float32List] embeddings in the same order as [faces].
  /// Faces that fail to produce embeddings will have null entries.
  Future<List<Float32List?>> getFaceEmbeddings(
    List<Face> faces,
    Uint8List imageBytes,
  ) async {
    _requireReady();
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'embeddings',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'faces': faces.map((f) => f.toMap()).toList(),
      },
    );
    return result.map((dynamic item) {
      if (item == null) return null;
      return Float32List.fromList((item as List).cast<double>());
    }).toList();
  }

  /// Compares two face embeddings and returns a cosine similarity score.
  ///
  /// Result ranges from -1 (completely different) to 1 (identical).
  ///
  /// Typical thresholds:
  /// - > 0.6: Very likely the same person
  /// - > 0.5: Probably the same person
  /// - < 0.3: Different people
  static double compareFaces(Float32List a, Float32List b) =>
      FaceEmbedding.cosineSimilarity(a, b);

  /// Computes the Euclidean distance between two face embeddings.
  ///
  /// Lower distance means more similar faces.
  ///
  /// Typical thresholds for normalized embeddings:
  /// - < 0.6: Very likely the same person
  /// - > 1.0: Different people
  static double faceDistance(Float32List a, Float32List b) =>
      FaceEmbedding.euclideanDistance(a, b);

  /// Segments an image to separate foreground (people) from background.
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities indicating
  /// foreground vs background.
  ///
  /// Parameters:
  /// - [outputFormat]: Controls the output format for transfer efficiency
  ///   - [IsolateOutputFormat.float32]: Full precision mask (largest transfer)
  ///   - [IsolateOutputFormat.uint8]: 8-bit grayscale (4x smaller)
  ///   - [IsolateOutputFormat.binary]: Binary mask at threshold (smallest)
  /// - [binaryThreshold]: Threshold for binary output (default 0.5)
  ///
  /// Throws [StateError] if [initializeSegmentation] hasn't been called.
  /// Throws [SegmentationException] on inference failure.
  Future<SegmentationMask> getSegmentationMask(
    Uint8List imageBytes, {
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) async {
    _requireReady();
    _requireSegmentationReady();
    final Map<String, dynamic> result =
        await _sendSegmentationRequest<Map<String, dynamic>>('segment', {
      'bytes': TransferableTypedData.fromList([imageBytes]),
      'outputFormat': outputFormat.index,
      'binaryThreshold': binaryThreshold,
    });
    return _deserializeMask(result);
  }

  /// Segments a pre-decoded [cv.Mat] image to separate foreground from background.
  ///
  /// The Mat is NOT disposed by this method — caller is responsible for disposal.
  ///
  /// Throws [StateError] if [initializeSegmentation] hasn't been called.
  /// Throws [SegmentationException] on inference failure.
  Future<SegmentationMask> getSegmentationMaskFromMat(
    cv.Mat image, {
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) async {
    _requireReady();
    _requireSegmentationReady();
    final f = _extractMatFields(image);
    final Map<String, dynamic> result =
        await _sendSegmentationRequest<Map<String, dynamic>>('segmentMat', {
      'bytes': TransferableTypedData.fromList([f.data]),
      'width': f.width,
      'height': f.height,
      'matType': f.matType,
      'outputFormat': outputFormat.index,
      'binaryThreshold': binaryThreshold,
    });
    return _deserializeMask(result);
  }

  /// Detects faces and generates segmentation mask in parallel.
  ///
  /// Requires [initializeSegmentation] or `initialize(withSegmentation: true)`.
  ///
  /// Returns a [DetectionWithSegmentationResult] containing both faces and mask.
  ///
  /// Processing time is approximately `max(detectionTime, segmentationTime)`
  /// rather than their sum, typically 40-50% faster than sequential calls.
  Future<DetectionWithSegmentationResult> detectFacesWithSegmentation(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) {
    _requireReady();
    _requireSegmentationReady();
    return _detectAndSegmentImpl(
      detectOp: 'detect',
      detectFields: {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'mode': mode.name,
      },
      segmentOp: 'segment',
      segmentFields: {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'outputFormat': outputFormat.index,
        'binaryThreshold': binaryThreshold,
      },
    );
  }

  /// Detects faces and generates segmentation mask in parallel from a [cv.Mat].
  ///
  /// The original Mat is NOT disposed by this method.
  Future<DetectionWithSegmentationResult> detectFacesWithSegmentationFromMat(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) {
    _requireReady();
    _requireSegmentationReady();
    final f = _extractMatFields(image);
    return _detectAndSegmentImpl(
      detectOp: 'detectMat',
      detectFields: {
        'bytes': TransferableTypedData.fromList([f.data]),
        'width': f.width,
        'height': f.height,
        'matType': f.matType,
        'mode': mode.name,
      },
      segmentOp: 'segmentMat',
      segmentFields: {
        'bytes': TransferableTypedData.fromList([f.data]),
        'width': f.width,
        'height': f.height,
        'matType': f.matType,
        'outputFormat': outputFormat.index,
        'binaryThreshold': binaryThreshold,
      },
    );
  }

  /// Extracts aligned eye regions of interest from face mesh landmarks.
  ///
  /// Uses specific mesh landmark points corresponding to the eye corners to
  /// compute aligned regions of interest (ROIs) for iris detection.
  ///
  /// Eye corner indices used:
  /// - Left eye: points 33 (inner corner) and 133 (outer corner)
  /// - Right eye: points 362 (inner corner) and 263 (outer corner)
  ///
  /// Returns a list of two [AlignedRoi] objects: `[left eye, right eye]`.
  List<AlignedRoi> eyeRoisFromMesh(List<Point> meshAbs) {
    AlignedRoi fromCorners(int a, int b) {
      final Point p0 = meshAbs[a];
      final Point p1 = meshAbs[b];
      final double cx = (p0.x + p1.x) * 0.5;
      final double cy = (p0.y + p1.y) * 0.5;
      final double dx = p1.x - p0.x;
      final double dy = p1.y - p0.y;
      final double eyeDist = math.sqrt(dx * dx + dy * dy);
      return AlignedRoi(cx, cy, eyeDist * 2.3, math.atan2(dy, dx));
    }

    return [fromCorners(33, 133), fromCorners(362, 263)];
  }

  /// Splits a concatenated list of mesh points into individual face meshes.
  ///
  /// If the list length is a multiple of 468, splits into sublists of 468 points.
  /// Otherwise returns the list unchanged (wrapped in a list).
  ///
  /// Example:
  /// ```dart
  /// // 936 points from 2 faces → [[face1 468 pts], [face2 468 pts]]
  /// final meshes = detector.splitMeshesIfConcatenated(allPoints);
  /// ```
  List<List<Point>> splitMeshesIfConcatenated(List<Point> meshPts) {
    if (meshPts.isEmpty) return const <List<Point>>[];
    if (meshPts.length % kMeshPoints != 0) return [meshPts];
    final int faces = meshPts.length ~/ kMeshPoints;
    return [
      for (int i = 0; i < faces; i++)
        meshPts.sublist(i * kMeshPoints, (i + 1) * kMeshPoints),
    ];
  }

  /// Releases all resources held by the detector.
  ///
  /// After calling dispose, you must call [initialize] again before
  /// running any detections.
  Future<void> dispose() async {
    final segmentationRpc = _segmentationRpc;
    _segmentationRpc = null;
    segmentationRpc?.failAllAndDispose(disposeOp: 'dispose');
    _segmentationInitialized = false;

    final worker = _worker;
    _worker = null;
    if (worker != null && worker.isReady) {
      await worker.dispose();
    }
  }

  void _requireReady() {
    if (!isReady) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before using.',
      );
    }
  }

  void _requireSegmentationReady() {
    if (!_segmentationInitialized) {
      throw StateError(
        'Segmentation not initialized. Call initializeSegmentation() or '
        'initialize(withSegmentation: true).',
      );
    }
  }

  Future<void> _spawnSegmentationIsolate({
    required IsolateRpcClient rpc,
    required Uint8List modelBytes,
    required SegmentationConfig config,
  }) async {
    final effectiveModel = config.model;
    rpc.isolate = await Isolate.spawn(
      _segmentationIsolateEntry,
      _SegmentationIsolateStartupData(
        sendPort: rpc.receivePort.sendPort,
        modelBytes: TransferableTypedData.fromList([modelBytes]),
        performanceModeName: config.performanceConfig.mode.name,
        numThreads: config.performanceConfig.numThreads,
        maxOutputSize: config.maxOutputSize,
        validateModel: config.validateModel,
        modelName: _modelFileFor(effectiveModel),
        modelIndex: effectiveModel.index,
      ),
      debugName: 'FaceDetector.segmentation',
    );

    rpc.sendPort = await setupIsolateHandshake(
      receivePort: rpc.receivePort,
      onResponse: (msg) => rpc.handleResponse(msg),
      timeout: const Duration(seconds: 15),
      timeoutMessage: 'Segmentation isolate initialization timed out',
    );
  }

  Future<T> _sendDetectionRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) =>
      _worker!.sendRequest<T>(operation, params);

  Future<T> _sendSegmentationRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) =>
      _segmentationRpc!.sendRequest<T>(operation, params);

  static List<Face> _deserializeFaces(List<dynamic> result) => result
      .map((map) => Face.fromMap(Map<String, dynamic>.from(map as Map)))
      .toList();

  static Float32List _packPoints(List<Point> points) {
    final buf = Float32List(points.length * 3);
    for (int i = 0; i < points.length; i++) {
      buf[i * 3] = points[i].x;
      buf[i * 3 + 1] = points[i].y;
      buf[i * 3 + 2] = points[i].z ?? 0.0;
    }
    return buf;
  }

  static List<Point> _unpackPoints(Float32List buf) {
    final n = buf.length ~/ 3;
    return List<Point>.generate(
        n, (i) => Point(buf[i * 3], buf[i * 3 + 1], buf[i * 3 + 2]));
  }

  static Map<String, dynamic> _faceToFastMap(Face f) {
    final bb = f._detection.boundingBox;
    return {
      'xmin': bb.xmin,
      'ymin': bb.ymin,
      'xmax': bb.xmax,
      'ymax': bb.ymax,
      'score': f._detection.score,
      'kp': f._detection.keypointsXY,
      'imgW': f.originalSize.width,
      'imgH': f.originalSize.height,
      if (f.mesh != null)
        'mesh': TransferableTypedData.fromList([_packPoints(f.mesh!.points)]),
      if (f.irisPoints.isNotEmpty)
        'iris': TransferableTypedData.fromList([_packPoints(f.irisPoints)]),
    };
  }

  static List<Face> _deserializeFacesFast(List<dynamic> result) {
    return result.map((raw) {
      final map = raw as Map;
      final imgW = (map['imgW'] as num).toDouble();
      final imgH = (map['imgH'] as num).toDouble();
      final imgSize = Size(imgW, imgH);
      final detection = Detection(
        boundingBox: RectF(
          (map['xmin'] as num).toDouble(),
          (map['ymin'] as num).toDouble(),
          (map['xmax'] as num).toDouble(),
          (map['ymax'] as num).toDouble(),
        ),
        score: (map['score'] as num).toDouble(),
        keypointsXY: (map['kp'] as List).cast<double>(),
        imageSize: imgSize,
      );
      FaceMesh? mesh;
      final meshTd = map['mesh'] as TransferableTypedData?;
      if (meshTd != null) {
        mesh = FaceMesh(_unpackPoints(meshTd.materialize().asFloat32List()));
      }
      List<Point> irisPoints = const [];
      final irisTd = map['iris'] as TransferableTypedData?;
      if (irisTd != null) {
        irisPoints = _unpackPoints(irisTd.materialize().asFloat32List());
      }
      return Face(
        detection: detection,
        mesh: mesh,
        irises: irisPoints,
        originalSize: imgSize,
      );
    }).toList();
  }

  static Uint8List _extractBytes(dynamic message) =>
      (message['bytes'] as TransferableTypedData).materialize().asUint8List();

  static cv.Mat _matFromMessage(Map message, Uint8List bytes) {
    final int width = message['width'] as int;
    final int height = message['height'] as int;
    final int matTypeValue = message['matType'] as int;
    return cv.Mat.fromList(height, width, cv.MatType(matTypeValue), bytes);
  }

  ({Uint8List data, int width, int height, int matType}) _extractMatFields(
    cv.Mat image,
  ) =>
      (
        data: image.data,
        width: image.cols,
        height: image.rows,
        matType: image.type.value,
      );

  Future<DetectionWithSegmentationResult> _detectAndSegmentImpl({
    required String detectOp,
    required Map<String, dynamic> detectFields,
    required String segmentOp,
    required Map<String, dynamic> segmentFields,
  }) async {
    final detectionStopwatch = Stopwatch()..start();
    final segmentationStopwatch = Stopwatch()..start();
    final results = await Future.wait([
      _sendDetectionRequest<List<dynamic>>(detectOp, detectFields).then((r) {
        detectionStopwatch.stop();
        return _deserializeFacesFast(r);
      }),
      _sendSegmentationRequest<Map<String, dynamic>>(
        segmentOp,
        segmentFields,
      ).then((r) {
        segmentationStopwatch.stop();
        return _deserializeMask(r);
      }),
    ]);
    return DetectionWithSegmentationResult(
      faces: results[0] as List<Face>,
      segmentationMask: results[1] as SegmentationMask,
      detectionTimeMs: detectionStopwatch.elapsedMilliseconds,
      segmentationTimeMs: segmentationStopwatch.elapsedMilliseconds,
    );
  }

  /// Detection isolate entry point — handles face detection and embeddings.
  @pragma('vm:entry-point')
  static void _detectionIsolateEntry(_DetectionIsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    _FaceDetectorCore? core;

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

      core = _FaceDetectorCore();
      await core.initializeFromBuffers(
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
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'FaceDetectorCore not initialized in isolate',
              });
              return;
            }
            final Uint8List imageBytes = _extractBytes(message);
            final mode = FaceDetectionMode.values.firstWhere(
              (m) => m.name == message['mode'] as String,
            );
            final cv.Mat mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
            try {
              final faces = await core!.detectFacesDirect(mat, mode: mode);
              mainSendPort.send({
                'id': id,
                'result': faces.map(_faceToFastMap).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'detectMat':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'FaceDetectorCore not initialized in isolate',
              });
              return;
            }
            final Uint8List matBytes = _extractBytes(message);
            final mode = FaceDetectionMode.values.firstWhere(
              (m) => m.name == message['mode'] as String,
            );
            final mat = _matFromMessage(message, matBytes);
            try {
              final faces = await core!.detectFacesDirect(mat, mode: mode);
              mainSendPort.send({
                'id': id,
                'result': faces.map(_faceToFastMap).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'embedding':
            if (core == null || !core!.isEmbeddingReady) {
              mainSendPort.send({
                'id': id,
                'error': 'Embedding not initialized in isolate',
              });
              return;
            }
            final Uint8List imageBytes = _extractBytes(message);
            final face = Face.fromMap(
              Map<String, dynamic>.from(message['face'] as Map),
            );
            final cv.Mat mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
            try {
              final embedding = await core!.getFaceEmbeddingDirect(face, mat);
              mainSendPort.send({'id': id, 'result': embedding.toList()});
            } finally {
              mat.dispose();
            }

          case 'embeddingMat':
            if (core == null || !core!.isEmbeddingReady) {
              mainSendPort.send({
                'id': id,
                'error': 'Embedding not initialized in isolate',
              });
              return;
            }
            final Uint8List embMatBytes = _extractBytes(message);
            final embMatFace = Face.fromMap(
              Map<String, dynamic>.from(message['face'] as Map),
            );
            final embMat = _matFromMessage(message, embMatBytes);
            try {
              final embedding =
                  await core!.getFaceEmbeddingDirect(embMatFace, embMat);
              mainSendPort.send({'id': id, 'result': embedding.toList()});
            } finally {
              embMat.dispose();
            }

          case 'embeddings':
            if (core == null || !core!.isEmbeddingReady) {
              mainSendPort.send({
                'id': id,
                'error': 'Embedding not initialized in isolate',
              });
              return;
            }
            final Uint8List imageBytes = _extractBytes(message);
            final List<Face> faces =
                _deserializeFaces(message['faces'] as List);
            final cv.Mat image = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
            try {
              final embeddings = <Float32List?>[];
              for (final face in faces) {
                try {
                  embeddings
                      .add(await core!.getFaceEmbeddingDirect(face, image));
                } catch (_) {
                  embeddings.add(null);
                }
              }
              mainSendPort.send({
                'id': id,
                'result': embeddings.map((e) => e?.toList()).toList(),
              });
            } finally {
              image.dispose();
            }

          case 'dispose':
            core?.dispose();
            core = null;
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }

  /// Segmentation isolate entry point — handles selfie segmentation.
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
            final Uint8List imageBytes = _extractBytes(message);
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
            final Uint8List matBytes = _extractBytes(message);
            final int outputFormatIndex = message['outputFormat'] as int;
            final double binaryThreshold = message['binaryThreshold'] as double;
            final mat = _matFromMessage(message, matBytes);
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
        result['data'] = TransferableTypedData.fromList([mask._data]);
        result['dataFormat'] = 'float32';
      case IsolateOutputFormat.uint8:
        result['data'] = TransferableTypedData.fromList([mask.toUint8()]);
        result['dataFormat'] = 'uint8';
      case IsolateOutputFormat.binary:
        result['data'] = TransferableTypedData.fromList(
            [mask.toBinary(threshold: binaryThreshold)]);
        result['dataFormat'] = 'binary';
        result['binaryThreshold'] = binaryThreshold;
    }

    if (mask is MulticlassSegmentationMask) {
      result['classData'] = TransferableTypedData.fromList([mask._classData]);
    }

    return result;
  }

  /// Deserializes a segmentation mask from isolate transfer data.
  static SegmentationMask _deserializeMask(Map<String, dynamic> map) {
    final width = map['width'] as int;
    final height = map['height'] as int;
    final originalWidth = map['originalWidth'] as int;
    final originalHeight = map['originalHeight'] as int;
    final padding =
        List<double>.unmodifiable((map['padding'] as List).cast<double>());

    final dataFormat = map['dataFormat'] as String? ?? 'float32';
    final rawData = map['data'] as TransferableTypedData;

    Float32List data;
    switch (dataFormat) {
      case 'float32':
        data = rawData.materialize().asFloat32List();
      case 'uint8':
        final uint8 = rawData.materialize().asUint8List();
        data = Float32List(uint8.length);
        for (int i = 0; i < uint8.length; i++) {
          data[i] = uint8[i] / 255.0;
        }
      case 'binary':
        final binary = rawData.materialize().asUint8List();
        data = Float32List(binary.length);
        for (int i = 0; i < binary.length; i++) {
          data[i] = binary[i] == 255 ? 1.0 : 0.0;
        }
      default:
        throw ArgumentError('Unknown data format: $dataFormat');
    }

    final classDataTd = map['classData'] as TransferableTypedData?;
    if (classDataTd != null) {
      final classData = classDataTd.materialize().asFloat32List();
      return MulticlassSegmentationMask._(
        data: data,
        width: width,
        height: height,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
        padding: padding,
        classData: classData,
      );
    }

    return SegmentationMask._(
      data: data,
      width: width,
      height: height,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      padding: padding,
    );
  }
}

class _FaceDetectorWorker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> initialize({
    required Uint8List faceDetectionBytes,
    required Uint8List faceLandmarkBytes,
    required Uint8List irisLandmarkBytes,
    required Uint8List embeddingBytes,
    required FaceDetectionModel model,
    required PerformanceConfig performanceConfig,
    required int meshPoolSize,
  }) async {
    await initWorker(
      (sendPort) => Isolate.spawn(
        FaceDetector._detectionIsolateEntry,
        _DetectionIsolateStartupData(
          sendPort: sendPort,
          faceDetectionBytes: TransferableTypedData.fromList([
            faceDetectionBytes,
          ]),
          faceLandmarkBytes: TransferableTypedData.fromList([
            faceLandmarkBytes,
          ]),
          irisLandmarkBytes: TransferableTypedData.fromList([
            irisLandmarkBytes,
          ]),
          embeddingBytes: TransferableTypedData.fromList([embeddingBytes]),
          modelName: model.name,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
          meshPoolSize: meshPoolSize,
        ),
        debugName: 'FaceDetector.detection',
      ),
      timeout: const Duration(seconds: 30),
      timeoutMessage: 'Detection isolate initialization timed out',
    );
  }
}

/// Iris point slice indices within the concatenated iris landmark output.
/// Left eye:  indices [_kLeftIrisStart, _kLeftIrisEnd)  → 5 iris keypoints at positions 71-75
/// Right eye: indices [_kRightIrisStart, _kRightIrisEnd) → 5 iris keypoints at positions 147-151
const int _kLeftIrisStart = 71;
const int _kLeftIrisEnd = 76;
const int _kRightIrisStart = 147;
const int _kRightIrisEnd = 152;

/// A lightweight sequential lock that chains futures to prevent concurrent access
/// to shared mutable buffers (e.g., TFLite interpreter buffers).
class _InferenceLock {
  Future<void> _lock = Future.value();

  Future<T> run<T>(Future<T> Function() fn) async {
    final previous = _lock;
    final completer = Completer<void>();
    _lock = completer.future;
    try {
      await previous;
      return await fn();
    } finally {
      completer.complete();
    }
  }
}

/// Computes face alignment geometry from detection keypoints.
({double theta, double cx, double cy, double size}) _computeFaceAlignment(
  Detection det,
  double imgW,
  double imgH,
) {
  final lx = det.keypointsXY[FaceLandmarkType.leftEye.index * 2] * imgW;
  final ly = det.keypointsXY[FaceLandmarkType.leftEye.index * 2 + 1] * imgH;
  final rx = det.keypointsXY[FaceLandmarkType.rightEye.index * 2] * imgW;
  final ry = det.keypointsXY[FaceLandmarkType.rightEye.index * 2 + 1] * imgH;
  final mx = det.keypointsXY[FaceLandmarkType.mouth.index * 2] * imgW;
  final my = det.keypointsXY[FaceLandmarkType.mouth.index * 2 + 1] * imgH;

  final eyeCx = (lx + rx) * 0.5;
  final eyeCy = (ly + ry) * 0.5;
  final vEx = rx - lx;
  final vEy = ry - ly;
  final vMx = mx - eyeCx;
  final vMy = my - eyeCy;

  final theta = math.atan2(vEy, vEx);
  final eyeDist = math.sqrt(vEx * vEx + vEy * vEy);
  final mouthDist = math.sqrt(vMx * vMx + vMy * vMy);
  final size = math.max(mouthDist * 3.6, eyeDist * 4.0);

  final cx = eyeCx + vMx * 0.1;
  final cy = eyeCy + vMy * 0.1;

  return (theta: theta, cx: cx, cy: cy, size: size);
}

/// Transforms normalized mesh landmarks to absolute image coordinates
/// using an affine transformation defined by center, size, and rotation.
List<Point> _transformMeshToAbsolute(
  List<List<double>> lmNorm,
  double cx,
  double cy,
  double size,
  double theta,
) {
  final double ct = math.cos(theta);
  final double st = math.sin(theta);
  final double sct = size * ct;
  final double sst = size * st;
  final double tx = cx - 0.5 * sct + 0.5 * sst;
  final double ty = cy - 0.5 * sst - 0.5 * sct;

  final int n = lmNorm.length;
  final List<Point> mesh = List<Point>.filled(n, const Point(0, 0, 0));

  for (int i = 0; i < n; i++) {
    final List<double> p = lmNorm[i];
    mesh[i] = Point(
      tx + sct * p[0] - sst * p[1],
      ty + sst * p[0] + sct * p[1],
      p[2] * size,
    );
  }
  return mesh;
}

@visibleForTesting
({double theta, double cx, double cy, double size}) testComputeFaceAlignment(
  Detection det,
  double imgW,
  double imgH,
) =>
    _computeFaceAlignment(det, imgW, imgH);

@visibleForTesting
List<Point> testTransformMeshToAbsolute(
  List<List<double>> lmNorm,
  double cx,
  double cy,
  double size,
  double theta,
) =>
    _transformMeshToAbsolute(lmNorm, cx, cy, size, theta);

@visibleForTesting
Future<T> Function<T>(Future<T> Function() fn) testCreateInferenceLockRunner() {
  final lock = _InferenceLock();
  return lock.run;
}

@visibleForTesting
Point testFindIrisCenterFromPoints(List<Point> irisPoints) =>
    _irisCenterFromPoints(irisPoints);

@visibleForTesting
Map<String, dynamic> testSerializeMask(
  SegmentationMask mask,
  IsolateOutputFormat format,
  double binaryThreshold,
) =>
    FaceDetector._serializeMask(mask, format, binaryThreshold);

@visibleForTesting
SegmentationMask testDeserializeMask(Map<String, dynamic> map) =>
    FaceDetector._deserializeMask(map);
