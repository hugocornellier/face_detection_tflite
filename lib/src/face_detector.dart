part of 'native/face_native_lib.dart';

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
/// // One-step construction
/// final detector = await FaceDetector.create();
///
/// // Or two-step, if you need to configure between construction and init
/// final detector = FaceDetector();
/// await detector.initialize();
///
/// // Detect faces with full mesh and iris tracking
/// final faces = await detector.detectFacesFromBytes(
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
/// 4. Call [detectFacesFromBytes] to analyze images
/// 5. Call [dispose] when done to free resources
///
/// See also:
/// - [initialize] for model loading options
/// - [detectFacesFromBytes] for the main detection API
/// - [Face] for the structure of detection results
class FaceDetector {
  /// Cache-invalidation key for consumers that persist detection results.
  ///
  /// Bump this when detection output could change for the same input bytes -
  /// i.e. on model file swaps (retraining, new weights), NMS threshold
  /// changes, preprocessing changes, or postprocessing / coordinate-space
  /// changes. Do NOT bump for pure refactors, doc updates, API additions,
  /// or anything else that leaves inference output identical.
  ///
  /// Downstream caches should include this value in their lookup key so
  /// stored results are ignored after an upgrade that changes behavior.
  static const String modelVersion = '1.0.1';

  /// Creates a new face detector instance.
  ///
  /// The detector is not ready for use until you call [initialize].
  FaceDetector();

  /// Creates and initializes a face detector in one step.
  ///
  /// Convenience factory that combines [FaceDetector.new] and [initialize].
  /// Accepts the same parameters as [initialize].
  ///
  /// Example:
  /// ```dart
  /// final detector = await FaceDetector.create();
  ///
  /// // Equivalent to:
  /// final detector = FaceDetector();
  /// await detector.initialize();
  /// ```
  static Future<FaceDetector> create({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
    bool withSegmentation = false,
    SegmentationConfig? segmentationConfig,
    bool useCompiledModel = false,
    // Web-only knobs; accepted here for API parity but ignored on native.
    bool useLiteRt = false,
    String liteRtAccelerator = 'auto',
  }) async {
    final detector = FaceDetector();
    await detector.initialize(
      model: model,
      performanceConfig: performanceConfig,
      meshPoolSize: meshPoolSize,
      withSegmentation: withSegmentation,
      segmentationConfig: segmentationConfig,
      useCompiledModel: useCompiledModel,
    );
    return detector;
  }

  _FaceDetectorWorker? _worker;
  IsolateRpcClient? _segmentationRpc;
  bool _segmentationInitialized = false;
  bool _useCompiledModel = false;

  /// Returns true if all models are loaded and ready for inference.
  ///
  /// You must call [initialize] before this returns true.
  bool get isReady => _worker?.isReady ?? false;

  /// Returns true if the embedding model is loaded and ready.
  ///
  /// All models load together, so this is equivalent to [isReady].
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
  /// The [performanceConfig] parameter controls classic Interpreter hardware
  /// acceleration delegates when [useCompiledModel] is false.
  ///
  /// The [withSegmentation] flag initializes the segmentation model at the same
  /// time, avoiding an extra model-load step later. Pass [segmentationConfig]
  /// to customize segmentation behaviour.
  ///
  /// The [useCompiledModel] flag defaults to true on this branch and runs the
  /// face detection, mesh, iris, embedding, and segmentation models through
  /// LiteRT CompiledModel (GPU with CPU fallback, async dispatch). If a
  /// segmentation model fails to compile on the current platform's LiteRT
  /// runtime, the segmentation isolate falls back to the Interpreter and
  /// prints a debug message.
  ///
  /// Example:
  /// ```dart
  /// // Default: LiteRT CompiledModel.
  /// final detector = FaceDetector();
  /// await detector.initialize();
  ///
  /// // Force the classic Interpreter CPU path.
  /// await detector.initialize(
  ///   useCompiledModel: false,
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
    bool useCompiledModel = false,
    // Web-only knobs; accepted here for API parity but ignored on native.
    bool useLiteRt = false,
    String liteRtAccelerator = 'auto',
  }) async {
    if (isReady) {
      throw StateError('FaceDetector already initialized');
    }
    _useCompiledModel = useCompiledModel;

    final worker = _FaceDetectorWorker();
    IsolateRpcClient? segmentationRpc;

    try {
      final faceDetectionPath =
          'packages/face_detection_tflite/assets/models/${faceDetectionModelFile(model)}';
      const faceLandmarkPath =
          'packages/face_detection_tflite/assets/models/$kFaceLandmarkModel';
      const irisLandmarkPath =
          'packages/face_detection_tflite/assets/models/$kIrisLandmarkModel';
      const embeddingPath =
          'packages/face_detection_tflite/assets/models/$kEmbeddingModel';

      final assetFutures = [
        rootBundle.load(faceDetectionPath),
        rootBundle.load(faceLandmarkPath),
        rootBundle.load(irisLandmarkPath),
        rootBundle.load(embeddingPath),
      ];

      if (withSegmentation) {
        final effectiveSegModel =
            segmentationConfig?.model ?? SegmentationModel.general;
        final segModelFile = segmentationModelFile(effectiveSegModel);
        assetFutures.add(
          rootBundle.load(
            'packages/face_detection_tflite/assets/models/$segModelFile',
          ),
        );
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
        useCompiledModel: useCompiledModel,
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
      await segmentationRpc?.disposeGracefully(disposeOp: 'dispose');
      if (worker.isReady) {
        // Graceful: let the detection isolate free its native models instead of
        // being force-killed mid-cleanup (the init-failure path leaked them).
        await worker.disposeGracefully();
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
    final segModelFile = segmentationModelFile(effectiveConfig.model);
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
      await segmentationRpc.disposeGracefully(disposeOp: 'dispose');
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
  Future<List<Face>> detectFacesFromBytes(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    _requireReady();
    final List<dynamic> result;
    try {
      result = await _sendDetectionRequest<List<dynamic>>('detect', {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'mode': mode.name,
      });
    } catch (e) {
      rethrowOrFormatException(e, imageBytes);
    }
    return _deserializeFacesFast(result);
  }

  /// Deprecated alias for [detectFacesFromBytes].
  ///
  /// Renamed for clarity: the input is encoded image bytes (JPEG/PNG/...),
  /// as opposed to the raw pixel bytes taken by [detectFacesFromMatBytes].
  @Deprecated(
    'Use detectFacesFromBytes instead. Will be removed in a future release.',
  )
  Future<List<Face>> detectFaces(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) => detectFacesFromBytes(imageBytes, mode: mode);

  /// Detects faces directly from a live `<video>` element. Web-only.
  ///
  /// Native platforms throw [UnsupportedError]. Provided here for API parity
  /// so cross-platform code can compile against either build.
  Future<List<Face>> detectFacesFromVideo(
    Object video, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    throw UnsupportedError(
      'detectFacesFromVideo is web-only. On native, use detectFacesFromCameraImage.',
    );
  }

  /// Runs segmentation on a live `<video>` frame. Web-only.
  Future<SegmentationMask> getSegmentationMaskFromVideo(Object video) {
    throw UnsupportedError('getSegmentationMaskFromVideo is web-only.');
  }

  /// Detects faces in an image file at [path].
  ///
  /// Convenience wrapper that reads the file and calls [detectFacesFromBytes].
  /// Not available on Flutter Web (uses `dart:io`).
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  /// Throws [FileSystemException] if the file cannot be read.
  /// Throws [FormatException] if the image cannot be decoded.
  Future<List<Face>> detectFacesFromFilepath(
    String path, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    final bytes = await File(path).readAsBytes();
    return detectFacesFromBytes(bytes, mode: mode);
  }

  /// Detects faces in a pre-decoded [cv.Mat] image.
  ///
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
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
  /// This avoids the overhead of building a Mat on the calling thread -
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

  /// Detects faces directly from a [CameraFrame] produced by
  /// [prepareCameraFrame].
  ///
  /// The frame's YUV→BGR colour conversion and any optional rotation happen
  /// inside the detection isolate, not on the calling thread. Use this from a
  /// `CameraController.startImageStream` callback to keep the UI thread free
  /// of OpenCV work.
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  Future<List<Face>> detectFacesFromCameraFrame(
    CameraFrame frame, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    int? maxDim,
  }) async {
    _requireReady();
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'detectCameraFrame',
      _cameraFrameFields(frame, {'mode': mode.name, 'maxDim': maxDim}),
    );
    return _deserializeFacesFast(result);
  }

  /// One-call wrapper for live camera streams: takes a `CameraImage`-shaped
  /// object directly (any object exposing `width`, `height`, and `planes` with
  /// `bytes` / `bytesPerRow` / `bytesPerPixel`) and runs YUV packing, colour
  /// conversion, rotation, and downscale in the detection isolate - all off
  /// the UI thread.
  ///
  /// Returns an empty list (not an error) when the plane shape can't be
  /// decoded. Throws at runtime if [cameraImage] doesn't expose the expected
  /// shape.
  ///
  /// [isBgra] selects BGRA vs. RGBA for the desktop single-plane path; ignored
  /// for YUV input (Android/iOS). Defaults to `true` on macOS (BGRA) and
  /// `false` on Windows/Linux (RGBA). Only pass this explicitly if you are
  /// using a non-standard camera plugin that delivers a different format.
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  Future<List<Face>> detectFacesFromCameraImage(
    Object cameraImage, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    CameraFrameRotation? rotation,
    bool? isBgra,
    int? maxDim,
  }) async {
    _requireReady();
    final frame = prepareCameraFrameFromImage(
      cameraImage,
      rotation: rotation,
      isBgra: isBgra ?? Platform.isMacOS,
    );
    if (frame == null) return const <Face>[];
    return detectFacesFromCameraFrame(frame, mode: mode, maxDim: maxDim);
  }

  /// Generates a face embedding (identity vector) for a detected face.
  ///
  /// The [face] parameter should be a face detection result from [detectFacesFromBytes].
  ///
  /// The [imageBytes] parameter should contain the encoded image data.
  /// For pre-decoded [cv.Mat] input, use [getFaceEmbeddingFromMat] instead.
  ///
  /// Returns a [Float32List] containing the L2-normalized embedding vector.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFacesFromBytes(imageBytes);
  /// final embedding = await detector.getFaceEmbedding(faces.first, imageBytes);
  ///
  /// final similarity = FaceDetector.compareFaces(embedding, referenceEmbedding);
  /// if (similarity > 0.6) print('Same person!');
  /// ```
  Future<Float32List> getFaceEmbedding(Face face, Uint8List imageBytes) async {
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

  /// Generates a face embedding from an image file at [path].
  ///
  /// Convenience wrapper that reads the file and calls [getFaceEmbedding].
  /// Not available on Flutter Web (uses `dart:io`).
  Future<Float32List> getFaceEmbeddingFromFilepath(
    Face face,
    String path,
  ) async {
    final bytes = await File(path).readAsBytes();
    return getFaceEmbedding(face, bytes);
  }

  /// Generates a face embedding from raw pixel bytes without constructing a [cv.Mat].
  ///
  /// Mirrors [detectFacesFromMatBytes] - use this when you already have decoded
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
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
  /// final faces = await detector.detectFacesFromMat(mat);
  /// final embedding = await detector.getFaceEmbeddingFromMat(faces.first, mat);
  /// mat.dispose();
  /// ```
  Future<Float32List> getFaceEmbeddingFromMat(Face face, cv.Mat image) {
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
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
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

  /// Segments a [CameraFrame] to separate foreground from background,
  /// deferring YUV→BGR colour conversion and rotation to the segmentation
  /// isolate.
  ///
  /// Throws [StateError] if [initializeSegmentation] hasn't been called.
  /// Throws [SegmentationException] on inference failure.
  Future<SegmentationMask> getSegmentationMaskFromCameraFrame(
    CameraFrame frame, {
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
    int? maxDim,
  }) async {
    _requireReady();
    _requireSegmentationReady();
    final Map<String, dynamic> result =
        await _sendSegmentationRequest<Map<String, dynamic>>(
          'segmentCameraFrame',
          _cameraFrameFields(frame, {
            'outputFormat': outputFormat.index,
            'binaryThreshold': binaryThreshold,
            'maxDim': maxDim,
          }),
        );
    return _deserializeMask(result);
  }

  /// Detects faces and generates a segmentation mask in parallel from a
  /// [CameraFrame], with all OpenCV work off the UI thread.
  ///
  /// The frame's bytes are transferred to both the detection and segmentation
  /// isolates; each does its own `cvtColor` + optional `rotate` before
  /// inference. Total wall time is still ~`max(detect, segment)` since the
  /// two run in parallel.
  Future<DetectionWithSegmentationResult>
  detectFacesWithSegmentationFromCameraFrame(
    CameraFrame frame, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
    int? maxDim,
  }) {
    _requireReady();
    _requireSegmentationReady();
    return _detectAndSegmentImpl(
      detectOp: 'detectCameraFrame',
      detectFields: _cameraFrameFields(frame, {
        'mode': mode.name,
        'maxDim': maxDim,
      }),
      segmentOp: 'segmentCameraFrame',
      segmentFields: _cameraFrameFields(frame, {
        'outputFormat': outputFormat.index,
        'binaryThreshold': binaryThreshold,
        'maxDim': maxDim,
      }),
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
    // Flip ALL ready-state synchronously (before any await) so an un-awaited
    // dispose() immediately reports not-ready: a later call throws StateError
    // via the _require* guards instead of a null-check TypeError on a
    // half-torn-down handle, and a second dispose() is a no-op rather than
    // disposing the same worker twice.
    final segmentationRpc = _segmentationRpc;
    final worker = _worker;
    _segmentationRpc = null;
    _segmentationInitialized = false;
    _worker = null;

    await segmentationRpc?.disposeGracefully(disposeOp: 'dispose');
    if (worker != null && worker.isReady) {
      // Graceful shutdown: await the isolate's dispose ack before force-killing
      // so it can free its native TFLite interpreters instead of being reaped
      // mid-cleanup by Isolate.kill(priority: immediate).
      await worker.disposeGracefully();
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
        modelName: segmentationModelFile(effectiveModel),
        modelIndex: effectiveModel.index,
        useCompiledModel: _useCompiledModel,
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
  ) => _worker!.sendRequest<T>(operation, params);

  Future<T> _sendSegmentationRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) => _segmentationRpc!.sendRequest<T>(operation, params);

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
      n,
      (i) => Point(buf[i * 3], buf[i * 3 + 1], buf[i * 3 + 2]),
    );
  }

  static Map<String, dynamic> _faceToFastMap(Face f) {
    final bb = f.detectionData.boundingBox;
    return {
      'xmin': bb.xmin,
      'ymin': bb.ymin,
      'xmax': bb.xmax,
      'ymax': bb.ymax,
      'score': f.detectionData.score,
      'kp': f.detectionData.keypointsXY,
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
    return matFromPackedBytes(height, width, cv.MatType(matTypeValue), bytes);
  }

  /// Builds the isolate-request field map for a [CameraFrame] payload, merged
  /// with any [extra] per-op fields (e.g. `mode`, `outputFormat`).
  Map<String, dynamic> _cameraFrameFields(
    CameraFrame frame,
    Map<String, dynamic> extra,
  ) => cameraFrameRpcFields(frame, extra);

  /// Decodes a [CameraFrame] message into a 3-channel BGR [cv.Mat]. Runs
  /// inside the detection / segmentation isolate - all OpenCV work happens off
  /// the UI thread.
  ///
  /// Op ordering is tuned to keep the big allocations tiny: for BGRA frames we
  /// resize and rotate on the 4-channel buffer and defer `cvtColor` to the end
  /// (so it converts the post-resize ~640px buffer, not full-res). For YUV we
  /// must `cvtColor` first because the packed layout isn't resizable, but we
  /// then resize before rotating so the rotate runs on the small BGR buffer.
  /// Output is byte-identical to the rotate→resize→cvtColor order because
  /// `cv.rotate` 90/180/270 is a lossless permutation, `cv.resize`
  /// (`INTER_LINEAR`) interpolates each channel independently, and the
  /// BGRA→BGR conversion is a per-pixel alpha drop.
  static cv.Mat _matFromCameraFrameMessage(Map message, Uint8List bytes) {
    return cameraFrameToBgrMat(
      cameraFrameFromRpcMessage(message, bytes),
      maxDim: message['maxDim'] as int?,
    );
  }

  ({Uint8List data, int width, int height, int matType}) _extractMatFields(
    cv.Mat image,
  ) {
    // Mat.data ignores row stride, so a non-continuous Mat (e.g. an ROI view
    // from region()) would ship scrambled pixels - pack it into a continuous
    // copy first.
    if (!image.isContinuous) {
      final cv.Mat packed = image.clone();
      try {
        final Uint8List view = packed.data;
        final Uint8List copy = Uint8List(view.length)..setAll(0, view);
        return (
          data: copy,
          width: packed.cols,
          height: packed.rows,
          matType: packed.type.value,
        );
      } finally {
        packed.dispose();
      }
    }
    return (
      data: image.data,
      width: image.cols,
      height: image.rows,
      matType: image.type.value,
    );
  }

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

  /// Detection isolate entry point - handles face detection and embeddings.
  @pragma('vm:entry-point')
  static void _detectionIsolateEntry(_DetectionIsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    _FaceDetectorCore? core;

    try {
      final faceDetectionBytes = data.faceDetectionBytes
          .materialize()
          .asUint8List();
      final faceLandmarkBytes = data.faceLandmarkBytes
          .materialize()
          .asUint8List();
      final irisLandmarkBytes = data.irisLandmarkBytes
          .materialize()
          .asUint8List();
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
        useCompiledModel: data.useCompiledModel,
      );

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({
        'error': 'Detection isolate initialization failed: $e\n$st',
      });
      return;
    }

    FaceDetectionMode parseMode(Map message) => FaceDetectionMode.values
        .firstWhere((m) => m.name == message['mode'] as String);

    Future<Object?> runDetect(
      _FaceDetectorCore c,
      cv.Mat mat,
      FaceDetectionMode mode,
    ) async {
      try {
        final faces = await c.detectFacesDirect(mat, mode: mode);
        return faces.map(_faceToFastMap).toList();
      } finally {
        mat.dispose();
      }
    }

    Future<Object?> runEmbed(_FaceDetectorCore c, cv.Mat mat, Face face) async {
      try {
        final embedding = await c.getFaceEmbeddingDirect(face, mat);
        return embedding.toList();
      } finally {
        mat.dispose();
      }
    }

    serveIsolateRpc(
      mainSendPort: mainSendPort,
      receivePort: workerReceivePort,
      handlers: {
        'detect': (message) {
          final c = core;
          if (c == null) {
            throw IsolateRpcExactError(
              'FaceDetectorCore not initialized in isolate',
            );
          }
          final mode = parseMode(message);
          final mat = cv.imdecode(_extractBytes(message), cv.IMREAD_COLOR);
          if (mat.isEmpty) {
            mat.dispose();
            throwDecodeFailure();
          }
          return runDetect(c, mat, mode);
        },
        'detectMat': (message) {
          final c = core;
          if (c == null) {
            throw IsolateRpcExactError(
              'FaceDetectorCore not initialized in isolate',
            );
          }
          final mode = parseMode(message);
          return runDetect(
            c,
            _matFromMessage(message, _extractBytes(message)),
            mode,
          );
        },
        'detectCameraFrame': (message) {
          final c = core;
          if (c == null) {
            throw IsolateRpcExactError(
              'FaceDetectorCore not initialized in isolate',
            );
          }
          final mode = parseMode(message);
          return runDetect(
            c,
            _matFromCameraFrameMessage(message, _extractBytes(message)),
            mode,
          );
        },
        'embedding': (message) {
          final c = core;
          if (c == null || !c.isEmbeddingReady) {
            throw IsolateRpcExactError('Embedding not initialized in isolate');
          }
          final face = Face.fromMap(
            Map<String, dynamic>.from(message['face'] as Map),
          );
          return runEmbed(
            c,
            cv.imdecode(_extractBytes(message), cv.IMREAD_COLOR),
            face,
          );
        },
        'embeddingMat': (message) {
          final c = core;
          if (c == null || !c.isEmbeddingReady) {
            throw IsolateRpcExactError('Embedding not initialized in isolate');
          }
          final face = Face.fromMap(
            Map<String, dynamic>.from(message['face'] as Map),
          );
          return runEmbed(
            c,
            _matFromMessage(message, _extractBytes(message)),
            face,
          );
        },
        'embeddings': (message) async {
          final c = core;
          if (c == null || !c.isEmbeddingReady) {
            throw IsolateRpcExactError('Embedding not initialized in isolate');
          }
          final List<Face> faces = _deserializeFaces(message['faces'] as List);
          final cv.Mat image = cv.imdecode(
            _extractBytes(message),
            cv.IMREAD_COLOR,
          );
          try {
            final embeddings = <Float32List?>[];
            for (final face in faces) {
              try {
                embeddings.add(await c.getFaceEmbeddingDirect(face, image));
              } catch (_) {
                embeddings.add(null);
              }
            }
            return embeddings.map((e) => e?.toList()).toList();
          } finally {
            image.dispose();
          }
        },
      },
      onDispose: () async {
        core?.dispose();
        core = null;
      },
    );
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

      if (data.useCompiledModel) {
        try {
          segmenter = await SelfieSegmentation.createCompiledFromBuffer(
            modelBytes,
            config: config,
          );
        } catch (e) {
          debugPrint(
            'face_detection_tflite: CompiledModel segmentation unavailable '
            'for ${data.modelName}; falling back to Interpreter: $e',
          );
        }
      }
      segmenter ??= await SelfieSegmentation.createFromBuffer(
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

    Future<Object?> runSegment(
      Future<SegmentationMask> Function() produce,
      int outputFormatIndex,
      double binaryThreshold, {
      cv.Mat? mat,
    }) async {
      try {
        final mask = await produce();
        return _serializeMask(
          mask,
          IsolateOutputFormat.values[outputFormatIndex],
          binaryThreshold,
        );
      } on SegmentationException catch (e) {
        // Preserve the exact wire format (no "Bad state:"/stack wrapping).
        throw IsolateRpcExactError(
          'SegmentationException(${e.code}): ${e.message}',
        );
      } finally {
        mat?.dispose();
      }
    }

    serveIsolateRpc(
      mainSendPort: mainSendPort,
      receivePort: workerReceivePort,
      handlers: {
        'segment': (message) {
          final s = segmenter;
          if (s == null) {
            throw IsolateRpcExactError(
              'Segmentation not initialized in isolate',
            );
          }
          final bytes = _extractBytes(message);
          return runSegment(
            () => s.callFromBytes(bytes),
            message['outputFormat'] as int,
            message['binaryThreshold'] as double,
          );
        },
        'segmentMat': (message) {
          final s = segmenter;
          if (s == null) {
            throw IsolateRpcExactError(
              'Segmentation not initialized in isolate',
            );
          }
          final outputFormat = message['outputFormat'] as int;
          final binaryThreshold = message['binaryThreshold'] as double;
          final mat = _matFromMessage(message, _extractBytes(message));
          return runSegment(
            () => s.call(mat),
            outputFormat,
            binaryThreshold,
            mat: mat,
          );
        },
        'segmentCameraFrame': (message) {
          final s = segmenter;
          if (s == null) {
            throw IsolateRpcExactError(
              'Segmentation not initialized in isolate',
            );
          }
          final outputFormat = message['outputFormat'] as int;
          final binaryThreshold = message['binaryThreshold'] as double;
          final frameMat = _matFromCameraFrameMessage(
            message,
            _extractBytes(message),
          );
          return runSegment(
            () => s.call(frameMat),
            outputFormat,
            binaryThreshold,
            mat: frameMat,
          );
        },
      },
      onDispose: () async {
        segmenter?.dispose();
        segmenter = null;
      },
    );
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
        result['data'] = TransferableTypedData.fromList([mask.internalData]);
        result['dataFormat'] = 'float32';
      case IsolateOutputFormat.uint8:
        result['data'] = TransferableTypedData.fromList([mask.toUint8()]);
        result['dataFormat'] = 'uint8';
      case IsolateOutputFormat.binary:
        result['data'] = TransferableTypedData.fromList([
          mask.toBinary(threshold: binaryThreshold),
        ]);
        result['dataFormat'] = 'binary';
        result['binaryThreshold'] = binaryThreshold;
    }

    if (mask is MulticlassSegmentationMask) {
      result['classData'] = TransferableTypedData.fromList([
        mask.internalClassData,
      ]);
    }

    return result;
  }

  /// Deserializes a segmentation mask from isolate transfer data.
  static SegmentationMask _deserializeMask(Map<String, dynamic> map) {
    final width = map['width'] as int;
    final height = map['height'] as int;
    final originalWidth = map['originalWidth'] as int;
    final originalHeight = map['originalHeight'] as int;
    final padding = List<double>.unmodifiable(
      (map['padding'] as List).cast<double>(),
    );

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
      return MulticlassSegmentationMask.internal(
        internalData: data,
        width: width,
        height: height,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
        padding: padding,
        internalClassData: classData,
      );
    }

    return SegmentationMask.internal(
      internalData: data,
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
    required bool useCompiledModel,
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
          useCompiledModel: useCompiledModel,
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

/// Test-only: exposes the internal face alignment computation for unit tests.
@visibleForTesting
({double theta, double cx, double cy, double size}) testComputeFaceAlignment(
  Detection det,
  double imgW,
  double imgH,
) => _computeFaceAlignment(det, imgW, imgH);

/// Test-only: exposes the internal mesh-to-absolute transform for unit tests.
@visibleForTesting
List<Point> testTransformMeshToAbsolute(
  List<List<double>> lmNorm,
  double cx,
  double cy,
  double size,
  double theta,
) => _transformMeshToAbsolute(lmNorm, cx, cy, size, theta);

/// Test-only: returns a fresh inference-lock `run` function for unit tests.
@visibleForTesting
Future<T> Function<T>(Future<T> Function() fn) testCreateInferenceLockRunner() {
  final lock = AsyncLock();
  return lock.run;
}

/// Test-only: exposes the internal iris-center computation for unit tests.
@visibleForTesting
Point testFindIrisCenterFromPoints(List<Point> irisPoints) =>
    irisCenterFromPoints(irisPoints);

/// Test-only: exposes the private mask-serialization logic for unit tests.
@visibleForTesting
Map<String, dynamic> testSerializeMask(
  SegmentationMask mask,
  IsolateOutputFormat format,
  double binaryThreshold,
) => FaceDetector._serializeMask(mask, format, binaryThreshold);

/// Test-only: exposes the private mask-deserialization logic for unit tests.
@visibleForTesting
SegmentationMask testDeserializeMask(Map<String, dynamic> map) =>
    FaceDetector._deserializeMask(map);
