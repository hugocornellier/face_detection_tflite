part of '../face_detection_tflite.dart';

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

  FaceDetection? _detector;
  final List<FaceLandmark> _meshPool = [];
  IrisLandmark? _irisLeft;
  IrisLandmark? _irisRight;
  FaceEmbedding? _embedding;
  SelfieSegmentation? _segmenter;

  final List<Future<void>> _meshInferenceLocks = [];
  final _irisLeftLock = _InferenceLock();
  final _irisRightLock = _InferenceLock();
  final _embeddingLock = _InferenceLock();
  final _segmentationLock = _InferenceLock();

  /// Counts successful iris landmark detections since initialization.
  ///
  /// Incremented each time iris center computation completes successfully
  /// with valid landmark points. Useful for monitoring detection reliability.
  int irisOkCount = 0;

  /// Counts failed iris landmark detections since initialization.
  ///
  /// Incremented when iris center computation fails to produce valid
  /// landmark points. Useful for monitoring detection reliability.
  int irisFailCount = 0;

  /// Counts how many times fallback eye positions were used.
  ///
  /// Incremented when iris detection falls back to original eye keypoint
  /// positions due to invalid ROI size (e.g., when eye region collapses).
  int irisUsedFallbackCount = 0;

  /// Duration of the most recent iris landmark detection operation.
  ///
  /// Updated after each iris center computation.
  /// Useful for profiling iris detection performance. Initialized to [Duration.zero].
  Duration lastIrisTime = Duration.zero;

  /// Returns true if all models are loaded and ready for inference.
  ///
  /// You must call [initialize] before this returns true.
  bool get isReady =>
      _detector != null &&
      _meshPool.isNotEmpty &&
      _irisLeft != null &&
      _irisRight != null;

  /// Loads the face detection, face mesh, and iris landmark models and prepares the interpreters for inference.
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
  /// - macOS/Linux: XNNPACK (2-5x CPU speedup)
  /// - iOS: Metal GPU delegate
  /// - Windows/Android: CPU-only (for stability)
  /// If both [options] and [performanceConfig] are provided, [options] takes precedence.
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
    try {
      _detector = await FaceDetection.create(
        model,
        options: options,
        performanceConfig: performanceConfig,
      );

      _meshPool.clear();
      _meshInferenceLocks.clear();
      for (int i = 0; i < meshPoolSize; i++) {
        _meshPool.add(
          await FaceLandmark.create(
            options: options,
            performanceConfig: performanceConfig,
          ),
        );
        _meshInferenceLocks.add(Future.value());
      }

      _irisLeft = await IrisLandmark.create(
        options: options,
        performanceConfig: performanceConfig,
      );
      _irisRight = await IrisLandmark.create(
        options: options,
        performanceConfig: performanceConfig,
      );

      _embedding = await FaceEmbedding.create(
        options: options,
        performanceConfig: performanceConfig,
      );
    } catch (e) {
      _cleanupOnInitError();
      rethrow;
    }
  }

  /// Initializes the face detector from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetectorIsolate] to initialize models
  /// in a background isolate where asset loading is not available.
  ///
  /// The [faceDetectionBytes] parameter should contain the face detection model.
  /// The [faceLandmarkBytes] parameter should contain the face mesh model.
  /// The [irisLandmarkBytes] parameter should contain the iris landmark model.
  /// The [embeddingBytes] parameter should contain the face embedding model (optional).
  ///
  /// @internal
  Future<void> initializeFromBuffers({
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

      _meshPool.clear();
      _meshInferenceLocks.clear();
      for (int i = 0; i < meshPoolSize; i++) {
        _meshPool.add(
          await FaceLandmark.createFromBuffer(
            faceLandmarkBytes,
            performanceConfig: performanceConfig,
          ),
        );
        _meshInferenceLocks.add(Future.value());
      }

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

  /// Disposes all partially-initialized resources after a failed initialization.
  ///
  /// Uses try-catch around each disposal to ensure all fields are cleaned up
  /// even if one model was already disposed (e.g. dispose→reinitialize failure).
  void _cleanupOnInitError() {
    try {
      _detector?.dispose();
    } on StateError catch (_) {}
    for (final mesh in _meshPool) {
      try {
        mesh.dispose();
      } on StateError catch (_) {}
    }
    _meshPool.clear();
    _meshInferenceLocks.clear();
    try {
      _irisLeft?.dispose();
    } on StateError catch (_) {}
    try {
      _irisRight?.dispose();
    } on StateError catch (_) {}
    try {
      _embedding?.dispose();
    } on StateError catch (_) {}
    _detector = null;
    _irisLeft = null;
    _irisRight = null;
    _embedding = null;
  }

  /// Serializes mesh model inference using a pool of models for parallel processing.
  ///
  /// Each FaceLandmark model uses shared mutable buffers that cannot be safely
  /// accessed concurrently. This method selects an available model from the pool
  /// and ensures only one inference runs per model instance at a time, while
  /// allowing different faces to use different models in parallel.
  ///
  /// Uses round-robin selection to distribute load across the pool.
  int _meshPoolCounter = 0;
  Future<T> _withMeshLock<T>(Future<T> Function(FaceLandmark) fn) async {
    if (_meshPool.isEmpty) {
      throw StateError('Mesh pool is empty. Call initialize() first.');
    }

    final int poolIndex = _meshPoolCounter % _meshPool.length;
    _meshPoolCounter = (_meshPoolCounter + 1) % _meshPool.length;

    final previous = _meshInferenceLocks[poolIndex];
    final completer = Completer<void>();
    _meshInferenceLocks[poolIndex] = completer.future;

    try {
      await previous;
      return await fn(_meshPool[poolIndex]);
    } finally {
      completer.complete();
    }
  }

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
  /// Returns a list of two [AlignedRoi] objects: [left eye, right eye].
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
  }) async {
    final cv.Mat mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    if (mat.isEmpty) {
      throw FormatException('Could not decode image bytes');
    }
    try {
      return await detectFacesFromMat(mat, mode: mode);
    } finally {
      mat.dispose();
    }
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
          // Iris detection is best-effort; failures are silently ignored.
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
        if (irisPx.length >= 76) {
          final leftIrisPoints = irisPx.sublist(71, 76);
          final leftCenter = _findIrisCenterFromPoints(leftIrisPoints);
          kp[FaceLandmarkType.leftEye.index * 2] = leftCenter.x / width;
          kp[FaceLandmarkType.leftEye.index * 2 + 1] = leftCenter.y / height;
        }
        if (irisPx.length >= 152) {
          final rightIrisPoints = irisPx.sublist(147, 152);
          final rightCenter = _findIrisCenterFromPoints(rightIrisPoints);
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
    if (_meshPool.isEmpty) return <Point>[];

    final lmNorm = await _withMeshLock((fl) => fl.call(faceCrop));

    return _transformMeshToAbsolute(lmNorm, cx, cy, size, theta);
  }

  /// Internal: Get iris landmarks from mesh using cv.Mat source.
  ///
  /// Eye crop extraction (warpAffine) is done serially to avoid opencv_dart
  /// freeze issues, but TFLite inference runs in parallel for performance.
  Future<List<Point>> _irisFromMesh(cv.Mat image, List<Point> meshAbs) async {
    if (_irisLeft == null || _irisRight == null) return <Point>[];
    if (meshAbs.length < 468) return <Point>[];

    final List<AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
    if (rois.length < 2) return <Point>[];

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
      irisOkCount++;
    } else {
      irisFailCount++;
    }

    return pts;
  }

  /// Finds the iris center from a list of iris contour points.
  ///
  /// Finds the point closest to the centroid of all iris points, which
  /// geometrically identifies the center of the iris contour.
  /// This can be computed in O(n) instead of O(n²).
  Point _findIrisCenterFromPoints(List<Point> irisPoints) {
    if (irisPoints.isEmpty) return const Point(0, 0, 0);
    if (irisPoints.length == 1) return irisPoints[0];

    double cx = 0, cy = 0;
    for (final Point p in irisPoints) {
      cx += p.x;
      cy += p.y;
    }
    cx /= irisPoints.length;
    cy /= irisPoints.length;

    int bestIdx = 0;
    double bestDist = double.infinity;
    for (int i = 0; i < irisPoints.length; i++) {
      final double dx = irisPoints[i].x - cx;
      final double dy = irisPoints[i].y - cy;
      final double dist = dx * dx + dy * dy;
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    return irisPoints[bestIdx];
  }

  /// Whether the face embedding model is loaded and ready.
  ///
  /// Returns true if [initialize] has been called successfully and the
  /// embedding model is ready to generate face embeddings.
  bool get isEmbeddingReady => _embedding != null;

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
  ) async {
    if (_embedding == null) {
      throw StateError(
        'Embedding model not initialized. Call initialize() before getFaceEmbedding().',
      );
    }
    final cv.Mat mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    if (mat.isEmpty) {
      throw FormatException('Could not decode image bytes');
    }
    try {
      return await getFaceEmbeddingFromMat(face, mat);
    } finally {
      mat.dispose();
    }
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
    if (_embedding == null) {
      throw StateError(
        'Embedding model not initialized. Call initialize() before getFaceEmbedding().',
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
      final cv.Mat resized;
      if (faceCrop.cols != _embedding!.inputWidth ||
          faceCrop.rows != _embedding!.inputHeight) {
        resized = cv.resize(
            faceCrop,
            (
              _embedding!.inputWidth,
              _embedding!.inputHeight,
            ),
            interpolation: cv.INTER_LINEAR);
        faceCrop.dispose();
      } else {
        resized = faceCrop;
      }

      try {
        return await _embeddingLock.run(() => _embedding!.call(resized));
      } finally {
        resized.dispose();
      }
    } catch (e) {
      faceCrop.dispose();
      rethrow;
    }
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
  ) async {
    if (_embedding == null) {
      throw StateError(
        'Embedding model not initialized. Call initialize() before getFaceEmbeddings().',
      );
    }

    if (faces.isEmpty) {
      return <Float32List?>[];
    }

    final cv.Mat image = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    if (image.isEmpty) {
      throw FormatException('Could not decode image bytes');
    }

    try {
      final List<Float32List?> embeddings = <Float32List?>[];
      for (final face in faces) {
        try {
          final embedding = await getFaceEmbeddingFromMat(face, image);
          embeddings.add(embedding);
        } catch (e) {
          embeddings.add(null);
        }
      }
      return embeddings;
    } finally {
      image.dispose();
    }
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

  /// Whether the segmentation model is loaded and ready.
  ///
  /// Returns true only after [initializeSegmentation] has been called successfully.
  bool get isSegmentationReady => _segmenter != null;

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
    if (_segmenter != null) return;
    _segmenter = await SelfieSegmentation.create(
      config: config ?? SegmentationConfig.safe,
    );
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
  ) async {
    if (_segmenter == null) {
      throw StateError(
        'Segmentation not initialized. Call initializeSegmentation() first.',
      );
    }
    final cv.Mat mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    if (mat.isEmpty) {
      throw FormatException('Could not decode image bytes');
    }
    try {
      return await _segmentationLock.run(() => _segmenter!.call(mat));
    } finally {
      mat.dispose();
    }
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
  ) async {
    if (_segmenter == null) {
      throw StateError(
        'Segmentation not initialized. Call initializeSegmentation() first.',
      );
    }
    return _segmentationLock.run(() => _segmenter!.call(image));
  }

  /// Releases all resources held by the detector.
  ///
  /// Call this when you're done using the detector to free up memory.
  /// After calling dispose, you must call [initialize] again before
  /// running any detections.
  void dispose() {
    _detector?.dispose();
    _detector = null;
    for (final mesh in _meshPool) {
      mesh.dispose();
    }
    _meshPool.clear();
    _meshInferenceLocks.clear();
    _irisLeft?.dispose();
    _irisLeft = null;
    _irisRight?.dispose();
    _irisRight = null;
    _embedding?.dispose();
    _embedding = null;
    _segmenter?.dispose();
    _segmenter = null;
  }
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
    FaceDetector()._findIrisCenterFromPoints(irisPoints);
