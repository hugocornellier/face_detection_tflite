part of '../face_detection_tflite.dart';

// Iris point slice indices within the concatenated iris landmark output.
// Left eye:  indices [_kLeftIrisStart, _kLeftIrisEnd)  → 5 iris keypoints at positions 71-75
// Right eye: indices [_kRightIrisStart, _kRightIrisEnd) → 5 iris keypoints at positions 147-151
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
/// detector.dispose();
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
// TODO(v6.0.0): Remove this delegation wrapper. Rename FaceDetectorIsolate
// to FaceDetector and make the old _FaceDetectorCore fully private.
class FaceDetector {
  /// Creates a new face detector instance.
  ///
  /// The detector is not ready for use until you call [initialize].
  /// After initialization, use [detectFaces] to analyze images.
  ///
  /// Example:
  /// ```dart
  /// final detector = FaceDetector();
  /// await detector.initialize();
  /// final faces = await detector.detectFaces(imageBytes);
  /// detector.dispose(); // Clean up when done
  /// ```
  FaceDetector();

  // TODO(v6.0.0): Remove this internal worker. FaceDetector should become
  // FaceDetectorIsolate directly (renamed), not a wrapper around it.
  FaceDetectorIsolate? _worker;
  FaceDetectionModel _model = FaceDetectionModel.backCamera;
  PerformanceConfig _performanceConfig = const PerformanceConfig();
  int _meshPoolSize = 3;

  /// Counts successful iris landmark detections since initialization.
  ///
  /// Incremented each time iris center computation completes successfully
  /// with valid landmark points. Useful for monitoring detection reliability.
  @Deprecated(
    'irisOkCount cannot be tracked across isolate boundaries and will be '
    'removed in v6.0.0.',
  )
  int irisOkCount = 0;

  /// Counts failed iris landmark detections since initialization.
  ///
  /// Incremented when iris center computation fails to produce valid
  /// landmark points. Useful for monitoring detection reliability.
  @Deprecated(
    'irisFailCount cannot be tracked across isolate boundaries and will be '
    'removed in v6.0.0.',
  )
  int irisFailCount = 0;

  /// Returns true if all models are loaded and ready for inference.
  ///
  /// You must call [initialize] before this returns true.
  bool get isReady => _worker != null && _worker!.isReady;

  /// Loads the face detection, face mesh, and iris landmark models and prepares
  /// the interpreters for inference.
  ///
  /// This must be called before running any detections.
  /// The [model] argument specifies which detection model variant to load
  /// (for example, `FaceDetectionModel.backCamera`).
  ///
  /// The [meshPoolSize] parameter controls how many face mesh model instances
  /// to create for parallel processing. Default is 3, which allows up to 3 faces
  /// to have their meshes computed in parallel. Increase for better multi-face
  /// performance (at the cost of ~7-10MB memory per additional instance).
  ///
  /// Optionally, you can pass [options] to configure interpreter settings
  /// such as the number of threads or delegate type.
  ///
  /// The [performanceConfig] parameter controls hardware acceleration delegates.
  /// By default, auto mode selects the optimal delegate per platform:
  /// - iOS: Metal GPU delegate
  /// - Android/macOS/Linux/Windows: XNNPACK (2-5x SIMD acceleration)
  /// If both [options] and [performanceConfig] are provided, [options] takes
  /// precedence.
  ///
  /// Example:
  /// ```dart
  /// // Default (auto mode - optimal for each platform)
  /// final detector = FaceDetector();
  /// await detector.initialize();
  ///
  /// // Force CPU-only execution
  /// final detector = FaceDetector();
  /// await detector.initialize(
  ///   performanceConfig: PerformanceConfig.disabled,
  /// );
  /// ```
  Future<void> initialize({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    InterpreterOptions? options,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
  }) async {
    _model = model;
    _performanceConfig = performanceConfig;
    _meshPoolSize = meshPoolSize;
    // TODO(v6.0.0): Remove the options parameter. FaceDetectorIsolate.spawn()
    // only accepts PerformanceConfig. InterpreterOptions are ignored when
    // running in isolate mode.
    _worker = await FaceDetectorIsolate.spawn(
      model: model,
      performanceConfig: performanceConfig,
      meshPoolSize: meshPoolSize,
    );
  }

  /// Initializes the face detector from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetectorIsolate] to initialize models
  /// in a background isolate where asset loading is not available.
  ///
  /// @internal
  // TODO(v6.0.0): Move to _FaceDetectorCore as a private method.
  Future<void> initializeFromBuffers({
    required Uint8List faceDetectionBytes,
    required Uint8List faceLandmarkBytes,
    required Uint8List irisLandmarkBytes,
    Uint8List? embeddingBytes,
    required FaceDetectionModel model,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
  }) =>
      _initializeFromBuffersDirect(
        faceDetectionBytes: faceDetectionBytes,
        faceLandmarkBytes: faceLandmarkBytes,
        irisLandmarkBytes: irisLandmarkBytes,
        embeddingBytes: embeddingBytes,
        model: model,
        performanceConfig: performanceConfig,
        meshPoolSize: meshPoolSize,
      );

  /// Whether the face embedding model is loaded and ready.
  ///
  /// Returns true if [initialize] has been called successfully and the
  /// embedding model is ready to generate face embeddings.
  bool get isEmbeddingReady => isReady;

  /// Whether the segmentation model is loaded and ready.
  ///
  /// Returns true only after [initializeSegmentation] has been called
  /// successfully.
  bool get isSegmentationReady =>
      _worker != null && _worker!.isSegmentationReady;

  /// Extracts aligned eye regions of interest from face mesh landmarks.
  ///
  /// This method uses specific mesh landmark points corresponding to the eye
  /// corners to compute aligned regions of interest (ROIs) for iris detection.
  /// The ROIs are sized and oriented based on the distance between eye corners.
  ///
  /// The eye corner indices used are:
  /// - Left eye: points 33 (inner corner) and 133 (outer corner)
  /// - Right eye: points 362 (inner corner) and 263 (outer corner)
  ///
  /// For each eye, the ROI is:
  /// - Centered at the midpoint between the corners
  /// - Sized at 2.3× the distance between corners
  /// - Rotated to align with the eye's natural orientation
  ///
  /// The [meshAbs] parameter is a list of 468 face mesh points in absolute
  /// pixel coordinates, typically from [detectFaces].
  ///
  /// Returns a list of two [AlignedRoi] objects: `[left eye, right eye]`.
  /// Each ROI contains center coordinates, size, and rotation angle suitable
  /// for extracting aligned eye crops for iris landmark detection.
  List<AlignedRoi> eyeRoisFromMesh(List<Point> meshAbs) {
    AlignedRoi fromCorners(int a, int b) {
      final Point p0 = meshAbs[a];
      final Point p1 = meshAbs[b];
      final double cx = (p0.x + p1.x) * 0.5;
      final double cy = (p0.y + p1.y) * 0.5;
      final double dx = p1.x - p0.x;
      final double dy = p1.y - p0.y;
      final double eyeDist = math.sqrt(dx * dx + dy * dy);
      final double size = eyeDist * 2.3;
      return AlignedRoi(cx, cy, size, math.atan2(dy, dx));
    }

    final AlignedRoi left = fromCorners(33, 133);
    final AlignedRoi right = fromCorners(362, 263);
    return [left, right];
  }

  /// Splits a concatenated list of mesh points into individual face meshes.
  ///
  /// This utility method detects if a list of mesh points represents multiple
  /// faces concatenated together and splits them into separate lists. Each face
  /// mesh should contain exactly [kMeshPoints] (468) points.
  ///
  /// The splitting logic:
  /// - If the list is empty, returns an empty list
  /// - If the length is not a multiple of 468, returns the list unchanged (wrapped in a list)
  /// - Otherwise, splits into sublists of 468 points each
  ///
  /// The [meshPts] parameter is a potentially concatenated list of mesh points.
  ///
  /// Returns a list of mesh point lists, where each inner list represents one
  /// face with 468 points. If [meshPts] contains exactly 468 points, returns
  /// a list with one element. If it contains 936 points (468 × 2), returns two
  /// meshes, and so on.
  ///
  /// Example:
  /// ```dart
  /// // Input: 936 points from 2 faces
  /// final meshes = detector.splitMeshesIfConcatenated(allPoints);
  /// // Output: [[face1 468 points], [face2 468 points]]
  /// ```
  ///
  /// This is useful when processing batch results or when mesh data from
  /// multiple faces has been concatenated for efficiency.
  List<List<Point>> splitMeshesIfConcatenated(List<Point> meshPts) {
    if (meshPts.isEmpty) return const <List<Point>>[];
    if (meshPts.length % kMeshPoints != 0) return [meshPts];
    final int faces = meshPts.length ~/ kMeshPoints;
    final List<List<Point>> out = <List<Point>>[];
    for (int i = 0; i < faces; i++) {
      final int start = i * kMeshPoints;
      out.add(meshPts.sublist(start, start + kMeshPoints));
    }
    return out;
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
  /// Returns a [List] of [Face] objects, one per detected face. Each [Face] includes
  /// bounding box corners, facial landmarks, and optionally mesh and iris data
  /// depending on the mode.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes);
  /// ```
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  /// Throws [FormatException] if the image bytes cannot be decoded.
  Future<List<Face>> detectFaces(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    _requireReady();
    return _worker!.detectFaces(imageBytes, mode: mode);
  }

  /// Detects faces in a pre-decoded [cv.Mat] image.
  ///
  /// This is the cv.Mat variant of [detectFaces]. The Mat is NOT disposed
  /// by this method -- caller is responsible for disposal.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
  /// final faces = await detector.detectFacesFromMat(mat);
  /// mat.dispose();
  /// ```
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  Future<List<Face>> detectFacesFromMat(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    _requireReady();
    return _worker!.detectFacesFromMat(image, mode: mode);
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
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFacesFromMatBytes(
  ///   rawPixels,
  ///   width: 1920,
  ///   height: 1080,
  /// );
  /// ```
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  Future<List<Face>> detectFacesFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    _requireReady();
    return _worker!.detectFacesFromMatBytes(
      bytes,
      width: width,
      height: height,
      matType: matType,
      mode: mode,
    );
  }

  /// Generates a face embedding (identity vector) for a detected face.
  ///
  /// This method extracts the face region from the image, aligns it to a
  /// canonical pose, and generates a 192-dimensional embedding vector that
  /// represents the face's identity.
  ///
  /// The [face] parameter should be a face detection result from [detectFaces].
  ///
  /// The [imageBytes] parameter should contain the encoded image data.
  /// For pre-decoded [cv.Mat] input, use [getFaceEmbeddingFromMat] instead.
  ///
  /// Returns a [Float32List] containing the L2-normalized embedding vector.
  /// The embedding can be compared with other embeddings using [compareFaces].
  ///
  /// Throws [StateError] if the embedding model has not been initialized.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes);
  /// final embedding = await detector.getFaceEmbedding(faces.first, imageBytes);
  ///
  /// // Compare with a reference embedding
  /// final similarity = FaceDetector.compareFaces(embedding, referenceEmbedding);
  /// if (similarity > 0.6) {
  ///   print('Same person!');
  /// }
  /// ```
  Future<Float32List> getFaceEmbedding(
    Face face,
    Uint8List imageBytes,
  ) {
    _requireReady();
    return _worker!.getFaceEmbedding(face, imageBytes);
  }

  /// Generates a face embedding from a pre-decoded [cv.Mat] image.
  ///
  /// This is the cv.Mat variant of [getFaceEmbedding]. The Mat is NOT disposed
  /// by this method -- caller is responsible for disposal.
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
  ) async {
    _requireReady();
    // Encode the Mat data and delegate to the isolate worker, which
    // reconstructs the Mat on its side via TransferableTypedData.
    final (_, bytes) = cv.imencode('.bmp', image);
    return _worker!.getFaceEmbedding(face, bytes);
  }

  /// Generates face embeddings for multiple detected faces.
  ///
  /// This is more efficient than calling [getFaceEmbedding] multiple times
  /// because it decodes the image only once.
  ///
  /// The [faces] parameter should be a list of face detection results from
  /// [detectFaces].
  ///
  /// The [imageBytes] parameter should contain the encoded image data.
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
  ///     print('Face $i embedding: ${embeddings[i]!.length} dimensions');
  ///   }
  /// }
  /// ```
  Future<List<Float32List?>> getFaceEmbeddings(
    List<Face> faces,
    Uint8List imageBytes,
  ) {
    _requireReady();
    return _worker!.getFaceEmbeddings(faces, imageBytes);
  }

  /// Compares two face embeddings and returns a similarity score.
  ///
  /// Uses cosine similarity to measure how similar two faces are.
  /// The result ranges from -1 (completely different) to 1 (identical).
  ///
  /// Typical thresholds:
  /// - > 0.6: Very likely the same person
  /// - > 0.5: Probably the same person
  /// - > 0.4: Possibly the same person
  /// - < 0.3: Different people
  ///
  /// Both [a] and [b] should be embeddings from [getFaceEmbedding].
  ///
  /// Example:
  /// ```dart
  /// // Compare a face to a reference
  /// final similarity = FaceDetector.compareFaces(faceEmbedding, referenceEmbedding);
  /// if (similarity > 0.6) {
  ///   print('Match found!');
  /// }
  /// ```
  static double compareFaces(Float32List a, Float32List b) {
    return FaceEmbedding.cosineSimilarity(a, b);
  }

  /// Computes the Euclidean distance between two face embeddings.
  ///
  /// Lower distance means more similar faces.
  ///
  /// Typical thresholds for normalized embeddings:
  /// - < 0.6: Very likely the same person
  /// - < 0.8: Probably the same person
  /// - > 1.0: Different people
  ///
  /// Example:
  /// ```dart
  /// final distance = FaceDetector.faceDistance(embedding1, embedding2);
  /// if (distance < 0.6) {
  ///   print('Same person!');
  /// }
  /// ```
  static double faceDistance(Float32List a, Float32List b) {
    return FaceEmbedding.euclideanDistance(a, b);
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
  /// // Now segmentation is available
  /// final mask = await detector.getSegmentationMask(imageBytes);
  /// ```
  Future<void> initializeSegmentation({SegmentationConfig? config}) async {
    _requireReady();
    if (_worker!.isSegmentationReady) return;
    // Dispose the current worker and respawn with segmentation enabled.
    final oldWorker = _worker!;
    // We can't add segmentation to an existing isolate, so we need to
    // read the current configuration and respawn.
    // TODO(v6.0.0): Consider exposing a way to add segmentation to an
    // existing FaceDetectorIsolate without respawning.
    _worker = await FaceDetectorIsolate.spawn(
      model: _model,
      performanceConfig: _performanceConfig,
      meshPoolSize: _meshPoolSize,
      withSegmentation: true,
      segmentationConfig: config,
    );
    await oldWorker.dispose();
  }

  /// Segments an image to separate foreground (people) from background.
  ///
  /// Returns a [SegmentationMask] with per-pixel probabilities indicating
  /// foreground vs background.
  ///
  /// Throws [StateError] if [initializeSegmentation] hasn't been called.
  /// Throws [SegmentationException] on inference failure.
  ///
  /// Example:
  /// ```dart
  /// await detector.initializeSegmentation();
  ///
  /// final mask = await detector.getSegmentationMask(imageBytes);
  /// final binary = mask.toBinary(threshold: 0.5);
  /// ```
  Future<SegmentationMask> getSegmentationMask(
    Uint8List imageBytes,
  ) {
    _requireReady();
    return _worker!.getSegmentationMask(imageBytes);
  }

  /// Segments a pre-decoded [cv.Mat] image to separate foreground from background.
  ///
  /// This is the cv.Mat variant of [getSegmentationMask]. The Mat is NOT disposed
  /// by this method -- caller is responsible for disposal.
  ///
  /// Throws [StateError] if [initializeSegmentation] hasn't been called.
  /// Throws [SegmentationException] on inference failure.
  Future<SegmentationMask> getSegmentationMaskFromMat(
    cv.Mat image,
  ) {
    _requireReady();
    return _worker!.getSegmentationMaskFromMat(image);
  }

  /// Detects faces and generates segmentation mask in parallel.
  ///
  /// This method runs face detection and segmentation simultaneously,
  /// returning results as soon as both complete. This provides optimal
  /// performance when both features are needed.
  ///
  /// Requires [initializeSegmentation] to have been called first, or
  /// [initialize] with segmentation support.
  ///
  /// Returns a [DetectionWithSegmentationResult] containing both faces and mask.
  Future<DetectionWithSegmentationResult> detectFacesWithSegmentation(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) {
    _requireReady();
    return _worker!.detectFacesWithSegmentation(
      imageBytes,
      mode: mode,
      outputFormat: outputFormat,
      binaryThreshold: binaryThreshold,
    );
  }

  /// Detects faces and generates segmentation mask in parallel from a [cv.Mat].
  ///
  /// This is the cv.Mat variant of [detectFacesWithSegmentation]. The original
  /// Mat is NOT disposed by this method.
  Future<DetectionWithSegmentationResult> detectFacesWithSegmentationFromMat(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) {
    _requireReady();
    return _worker!.detectFacesWithSegmentationFromMat(
      image,
      mode: mode,
      outputFormat: outputFormat,
      binaryThreshold: binaryThreshold,
    );
  }

  /// Releases all resources held by the detector.
  ///
  /// Call this when you're done using the detector to free up memory.
  /// After calling dispose, you must call [initialize] again before
  /// running any detections.
  Future<void> dispose() async {
    await _worker?.dispose();
    _worker = null;
  }

  void _requireReady() {
    if (_worker == null || !_worker!.isReady) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before using.',
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Internal direct-initialization path used by FaceDetectorIsolate.
  //
  // When FaceDetectorIsolate spawns a background isolate, it creates a
  // FaceDetector inside that isolate and calls initializeFromBuffers().
  // In that context we must NOT spawn another nested isolate — we need the
  // old direct-invocation behavior. These fields and methods support that path.
  //
  // TODO(v6.0.0): Extract this into a private _FaceDetectorCore class.
  // ---------------------------------------------------------------------------

  FaceDetection? _detector;
  RoundRobinPool<FaceLandmark>? _meshPool;
  List<FaceLandmark> _meshItems = [];
  IrisLandmark? _irisLeft;
  IrisLandmark? _irisRight;
  FaceEmbedding? _embedding;
  SelfieSegmentation? _segmenter;

  final _irisLeftLock = _InferenceLock();
  final _irisRightLock = _InferenceLock();
  final _embeddingLock = _InferenceLock();

  /// Returns true when using the direct (non-isolate) initialization path.
  /// This is used by FaceDetectorIsolate's internal entry point.
  bool get _isDirectMode => _detector != null;

  Future<void> _initializeFromBuffersDirect({
    required Uint8List faceDetectionBytes,
    required Uint8List faceLandmarkBytes,
    required Uint8List irisLandmarkBytes,
    Uint8List? embeddingBytes,
    required FaceDetectionModel model,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
  }) async {
    try {
      _detector = await FaceDetection.createFromBuffer(
        faceDetectionBytes,
        model,
        performanceConfig: performanceConfig,
      );

      _meshItems = [];
      for (int i = 0; i < meshPoolSize; i++) {
        _meshItems.add(await FaceLandmark.createFromBuffer(
          faceLandmarkBytes,
          performanceConfig: performanceConfig,
        ));
      }
      _meshPool = RoundRobinPool(_meshItems);

      _irisLeft = await IrisLandmark.createFromBuffer(
        irisLandmarkBytes,
        performanceConfig: performanceConfig,
      );
      _irisRight = await IrisLandmark.createFromBuffer(
        irisLandmarkBytes,
        performanceConfig: performanceConfig,
      );

      if (embeddingBytes != null) {
        _embedding = await FaceEmbedding.createFromBuffer(
          embeddingBytes,
          performanceConfig: performanceConfig,
        );
      }
    } catch (e) {
      _cleanupOnInitError();
      rethrow;
    }
  }

  /// Direct-mode detection used inside the FaceDetectorIsolate's background
  /// isolate. Not used when FaceDetector is in delegation mode.
  Future<List<Face>> _detectFacesDirect(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    if (_detector == null) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before detectFaces().',
      );
    }

    final int width = image.cols;
    final int height = image.rows;
    final Size imgSize = Size(width.toDouble(), height.toDouble());

    final bool computeIris = mode == FaceDetectionMode.full;
    final bool computeMesh =
        mode == FaceDetectionMode.standard || mode == FaceDetectionMode.full;

    final List<Detection> dets = await _detectDetections(image);
    if (dets.isEmpty) return <Face>[];

    final List<(Detection, AlignedFace)?> alignedFaces =
        <(Detection, AlignedFace)?>[];
    for (final Detection det in dets) {
      try {
        final aligned = await _estimateAlignedFace(image, det);
        alignedFaces.add((det, aligned));
      } catch (e) {
        alignedFaces.add(null);
      }
    }

    final List<List<Point>?> meshResults;
    if (computeMesh) {
      meshResults = await Future.wait(
        alignedFaces.map((data) async {
          if (data == null) return null;
          try {
            return await _meshFromAlignedFace(
              data.$2.faceCrop,
              data.$2.cx,
              data.$2.cy,
              data.$2.size,
              data.$2.theta,
            );
          } catch (e) {
            return null;
          }
        }),
      );
    } else {
      meshResults = List<List<Point>?>.filled(alignedFaces.length, null);
    }

    for (final data in alignedFaces) {
      data?.$2.faceCrop.dispose();
    }

    final List<List<Point>?> irisResults = List<List<Point>?>.filled(
      dets.length,
      null,
    );
    if (computeIris) {
      for (int i = 0; i < meshResults.length; i++) {
        final meshPx = meshResults[i];
        if (meshPx == null || meshPx.isEmpty) continue;
        try {
          irisResults[i] = await _irisFromMesh(image, meshPx);
        } catch (_) {
          // Iris detection failed for this face; skip.
        }
      }
    }

    final List<Face> faces = <Face>[];
    for (int i = 0; i < dets.length; i++) {
      final aligned = alignedFaces[i];
      if (aligned == null) continue;

      final Detection det = aligned.$1;
      final List<Point> meshPx = meshResults[i] ?? <Point>[];
      final List<Point> irisPx = irisResults[i] ?? <Point>[];

      List<double> kp = det.keypointsXY;
      if (computeIris && irisPx.isNotEmpty) {
        kp = List<double>.from(det.keypointsXY);
        if (irisPx.length >= _kLeftIrisEnd) {
          final leftIrisPoints = irisPx.sublist(_kLeftIrisStart, _kLeftIrisEnd);
          final leftCenter = _irisCenterFromPoints(leftIrisPoints);
          kp[FaceLandmarkType.leftEye.index * 2] = leftCenter.x / width;
          kp[FaceLandmarkType.leftEye.index * 2 + 1] = leftCenter.y / height;
        }
        if (irisPx.length >= _kRightIrisEnd) {
          final rightIrisPoints =
              irisPx.sublist(_kRightIrisStart, _kRightIrisEnd);
          final rightCenter = _irisCenterFromPoints(rightIrisPoints);
          kp[FaceLandmarkType.rightEye.index * 2] = rightCenter.x / width;
          kp[FaceLandmarkType.rightEye.index * 2 + 1] = rightCenter.y / height;
        }
      }

      final Detection refinedDet = Detection(
        boundingBox: det.boundingBox,
        score: det.score,
        keypointsXY: kp,
        imageSize: imgSize,
      );

      final FaceMesh? faceMesh = meshPx.isNotEmpty ? FaceMesh(meshPx) : null;

      faces.add(
        Face(
          detection: refinedDet,
          mesh: faceMesh,
          irises: irisPx,
          originalSize: imgSize,
        ),
      );
    }

    return faces;
  }

  /// Internal: Detect raw detections from a cv.Mat.
  Future<List<Detection>> _detectDetections(cv.Mat image) async {
    final FaceDetection? d = _detector;
    if (d == null) {
      throw StateError('FaceDetector not initialized.');
    }

    final ImageTensor tensor = convertImageToTensor(
      image,
      outW: d.inputWidth,
      outH: d.inputHeight,
    );

    return await d.callWithTensor(tensor);
  }

  /// Internal: Aligned face data holder for Mat-based processing.
  Future<AlignedFace> _estimateAlignedFace(cv.Mat image, Detection det) async {
    final (:theta, :cx, :cy, :size) = _computeFaceAlignment(
      det,
      image.cols.toDouble(),
      image.rows.toDouble(),
    );

    final cv.Mat? faceCrop = extractAlignedSquare(image, cx, cy, size, -theta);

    if (faceCrop == null) {
      throw StateError('Failed to extract aligned face crop');
    }

    return AlignedFace(
      cx: cx,
      cy: cy,
      size: size,
      theta: theta,
      faceCrop: faceCrop,
    );
  }

  /// Internal: Generate mesh from aligned face cv.Mat.
  Future<List<Point>> _meshFromAlignedFace(
    cv.Mat faceCrop,
    double cx,
    double cy,
    double size,
    double theta,
  ) async {
    if (_meshPool == null || _meshPool!.isEmpty) return <Point>[];

    final lmNorm = await _meshPool!.withItem((fl) => fl.call(faceCrop));

    return _transformMeshToAbsolute(lmNorm, cx, cy, size, theta);
  }

  /// Internal: Get iris landmarks from mesh using cv.Mat source.
  ///
  /// Eye crop extraction (warpAffine) is done serially to avoid opencv_dart
  /// freeze issues, but TFLite inference runs in parallel for performance.
  Future<List<Point>> _irisFromMesh(cv.Mat image, List<Point> meshAbs) async {
    if (_irisLeft == null || _irisRight == null) {
      return <Point>[];
    }
    if (meshAbs.length < 468) {
      return <Point>[];
    }

    final List<AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
    if (rois.length < 2) {
      return <Point>[];
    }

    final cv.Mat? leftCrop = extractAlignedSquare(
      image,
      rois[0].cx,
      rois[0].cy,
      rois[0].size,
      rois[0].theta,
    );
    final cv.Mat? rightCropRaw = extractAlignedSquare(
      image,
      rois[1].cx,
      rois[1].cy,
      rois[1].size,
      rois[1].theta,
    );

    if (leftCrop == null || rightCropRaw == null) {
      leftCrop?.dispose();
      rightCropRaw?.dispose();
      // ignore: deprecated_member_use_from_same_package
      irisFailCount++;
      return <Point>[];
    }

    final cv.Mat rightCrop = cv.flip(rightCropRaw, 1);
    rightCropRaw.dispose();

    final List<List<List<double>>> results;
    try {
      results = await Future.wait([
        _irisLeftLock.run(() => _irisLeft!.call(leftCrop)),
        _irisRightLock.run(() => _irisRight!.call(rightCrop)),
      ]);
    } finally {
      leftCrop.dispose();
      rightCrop.dispose();
    }

    final leftAbs = _transformIrisToAbsolute(results[0], rois[0], false);
    final rightAbs = _transformIrisToAbsolute(results[1], rois[1], true);

    final List<Point> pts = <Point>[
      for (final p in leftAbs) Point(p[0], p[1], p[2]),
      for (final p in rightAbs) Point(p[0], p[1], p[2]),
    ];

    if (pts.isNotEmpty) {
      // ignore: deprecated_member_use_from_same_package
      irisOkCount++;
    } else {
      // ignore: deprecated_member_use_from_same_package
      irisFailCount++;
    }

    return pts;
  }

  /// Direct-mode embedding generation used inside FaceDetectorIsolate.
  Future<Float32List> _getFaceEmbeddingDirect(
    Face face,
    cv.Mat image,
  ) async {
    if (_embedding == null) {
      throw StateError(
        'Embedding model not initialized.',
      );
    }

    final landmarks = face.landmarks;
    final leftEye = landmarks.leftEye;
    final rightEye = landmarks.rightEye;

    if (leftEye == null || rightEye == null) {
      throw StateError('Face must have left and right eye landmarks');
    }

    final alignment = computeEmbeddingAlignment(
      leftEye: leftEye,
      rightEye: rightEye,
    );

    final cv.Mat? faceCrop = extractAlignedSquare(
      image,
      alignment.cx,
      alignment.cy,
      alignment.size,
      -alignment.theta,
    );

    if (faceCrop == null) {
      throw StateError('Failed to extract aligned face crop for embedding');
    }

    try {
      return await _embeddingLock.run(() => _embedding!.call(faceCrop));
    } finally {
      faceCrop.dispose();
    }
  }

  /// Disposes all model fields and clears references (direct mode only).
  void _disposeFields({bool safe = false}) {
    void d(void Function() fn) {
      if (safe) {
        try {
          fn();
        } on StateError catch (_) {}
      } else {
        fn();
      }
    }

    d(() => _detector?.dispose());
    for (final mesh in _meshItems) {
      d(() => mesh.dispose());
    }
    _meshItems = [];
    _meshPool = null;
    d(() => _irisLeft?.dispose());
    d(() => _irisRight?.dispose());
    d(() => _embedding?.dispose());
    d(() => _segmenter?.dispose());
    _detector = null;
    _irisLeft = null;
    _irisRight = null;
    _embedding = null;
    _segmenter = null;
  }

  /// Disposes all partially-initialized resources after a failed initialization.
  void _cleanupOnInitError() => _disposeFields(safe: true);

  /// Disposes direct-mode fields only.
  void _disposeDirect() => _disposeFields();
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
