part of '../face_detection_tflite.dart';

class _DetectionFeatures {
  const _DetectionFeatures({
    required this.detection,
    required this.alignedFace,
    required this.mesh,
    required this.iris,
  });

  final Detection detection;
  final AlignedFace alignedFace;
  final List<Point> mesh;
  final List<Point> iris;
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

  IsolateWorker? _worker;
  FaceDetection? _detector;
  final List<FaceLandmark> _meshPool = [];
  IrisLandmark? _irisLeft;
  IrisLandmark? _irisRight;
  FaceEmbedding? _embedding;
  SelfieSegmentation? _segmenter;

  final List<Future<void>> _meshInferenceLocks = [];
  Future<void> _irisLeftInferenceLock = Future.value();
  Future<void> _irisRightInferenceLock = Future.value();
  Future<void> _embeddingInferenceLock = Future.value();
  Future<void> _segmentationInferenceLock = Future.value();

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
    await _ensureTFLiteLoaded();
    try {
      _worker = IsolateWorker();
      await _worker!.initialize();

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
      _worker?.dispose();
      _detector?.dispose();
      for (final mesh in _meshPool) {
        mesh.dispose();
      }
      _meshPool.clear();
      _meshInferenceLocks.clear();
      _irisLeft?.dispose();
      _irisRight?.dispose();
      _embedding?.dispose();
      _worker = null;
      _detector = null;
      _irisLeft = null;
      _irisRight = null;
      _embedding = null;
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
    await _ensureTFLiteLoaded();
    try {
      _worker = IsolateWorker();
      await _worker!.initialize();

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
      _worker?.dispose();
      _detector?.dispose();
      for (final mesh in _meshPool) {
        mesh.dispose();
      }
      _meshPool.clear();
      _meshInferenceLocks.clear();
      _irisLeft?.dispose();
      _irisRight?.dispose();
      _embedding?.dispose();
      _worker = null;
      _detector = null;
      _irisLeft = null;
      _irisRight = null;
      _embedding = null;
      rethrow;
    }
  }

  static ffi.DynamicLibrary? _tfliteLib;
  static Future<void> _ensureTFLiteLoaded() async {
    if (_tfliteLib != null) return;

    if (!Platform.isWindows && !Platform.isLinux) return;

    final File exe = File(Platform.resolvedExecutable);
    final Directory exeDir = exe.parent;
    late final List<String> candidates;
    late final String hint;

    if (Platform.isWindows) {
      candidates = [
        p.join(exeDir.path, 'libtensorflowlite_c-win.dll'),
        'libtensorflowlite_c-win.dll',
      ];
      hint = 'Make sure your Windows plugin CMakeLists.txt sets:\n'
          '  set(PLUGIN_NAME_bundled_libraries ".../libtensorflowlite_c-win.dll" PARENT_SCOPE)\n'
          'so Flutter copies it next to the app EXE.';
    } else {
      candidates = [
        p.join(exeDir.path, 'lib', 'libtensorflowlite_c-linux.so'),
        'libtensorflowlite_c-linux.so',
      ];
      hint = 'Ensure linux/CMakeLists.txt sets:\n'
          '  set(PLUGIN_NAME_bundled_libraries "../assets/bin/libtensorflowlite_c-linux.so" PARENT_SCOPE)\n'
          'so Flutter copies it into bundle/lib/.';
    }

    final List<String> tried = <String>[];
    for (final String c in candidates) {
      try {
        if (c.contains(p.separator)) {
          if (!File(c).existsSync()) {
            tried.add(c);
            continue;
          }
        }
        _tfliteLib = ffi.DynamicLibrary.open(c);
        return;
      } catch (_) {
        tried.add(c);
      }
    }

    throw ArgumentError(
      'Failed to locate TensorFlow Lite C library.\n'
      'Tried:\n - ${tried.join('\n - ')}\n\n$hint',
    );
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

  /// Serializes left iris inference calls to prevent buffer race conditions.
  ///
  /// The left IrisLandmark model uses shared mutable buffers that cannot be safely
  /// accessed concurrently. This method ensures only one inference runs
  /// at a time on the left model by chaining futures, while allowing the right
  /// model to run in parallel.
  Future<T> _withIrisLeftLock<T>(Future<T> Function() fn) async {
    final previous = _irisLeftInferenceLock;
    final completer = Completer<void>();
    _irisLeftInferenceLock = completer.future;

    try {
      await previous;
      return await fn();
    } finally {
      completer.complete();
    }
  }

  /// Serializes right iris inference calls to prevent buffer race conditions.
  ///
  /// The right IrisLandmark model uses shared mutable buffers that cannot be safely
  /// accessed concurrently. This method ensures only one inference runs
  /// at a time on the right model by chaining futures, while allowing the left
  /// model to run in parallel.
  Future<T> _withIrisRightLock<T>(Future<T> Function() fn) async {
    final previous = _irisRightInferenceLock;
    final completer = Completer<void>();
    _irisRightInferenceLock = completer.future;

    try {
      await previous;
      return await fn();
    } finally {
      completer.complete();
    }
  }

  /// Returns face detections with eye keypoints refined by iris center detection.
  ///
  /// This method performs face detection, generates the face mesh for all
  /// detected faces, and uses iris landmark detection to compute precise iris centers.
  /// These iris centers replace the original eye keypoints for improved accuracy.
  ///
  /// The iris center refinement process for each face:
  /// 1. Detects all faces in the image
  /// 2. For each face, generates 468-point face mesh to locate eye regions
  /// 3. Runs iris detection on both eyes
  /// 4. Replaces eye keypoints with computed iris centers
  /// 5. Falls back to original eye positions if iris detection fails
  ///
  /// The [imageBytes] parameter should contain encoded image data (JPEG, PNG, etc.).
  ///
  /// Returns a list of detections with iris-refined keypoints for all faces.
  /// Returns an empty list if no faces are detected.
  ///
  /// **Performance:** This method is computationally intensive as it runs the full
  /// pipeline (detection + mesh + iris) for all detected faces.
  ///
  /// Throws [StateError] if the iris model has not been initialized.
  ///
  /// See also:
  /// - [detectFaces] with [FaceDetectionMode.full] for multi-face iris tracking
  /// - [getDetections] for basic detection without iris refinement
  Future<List<Detection>> getDetectionsWithIrisCenters(
    Uint8List imageBytes,
  ) async {
    if (_irisLeft == null || _irisRight == null) {
      throw StateError(
        'Iris models not initialized. Call initialize() before getDetectionsWithIrisCenters().',
      );
    }

    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final List<Detection> dets = await _detectDetectionsWithDecoded(decoded);
    return dets;
  }

  Future<List<Offset>> _computeIrisCentersWithDecoded(
    img.Image decoded,
    List<AlignedRoi> rois, {
    Offset? leftFallback,
    Offset? rightFallback,
    List<Point>? allLandmarks,
  }) async {
    final Stopwatch sw = Stopwatch()..start();
    final IrisLandmark irisLeft = _irisLeft!;
    final IrisLandmark irisRight = _irisRight!;
    allLandmarks?.clear();

    Offset pickCenter(List<List<double>> lm, Offset? fallback) {
      if (lm.isEmpty) return fallback ?? const Offset(0, 0);

      double cx = 0, cy = 0;
      for (final p in lm) {
        cx += p[0];
        cy += p[1];
      }
      cx /= lm.length;
      cy /= lm.length;

      int bestIdx = 0;
      double bestDist = double.infinity;
      for (int i = 0; i < lm.length; i++) {
        final double dx = lm[i][0] - cx;
        final double dy = lm[i][1] - cy;
        final double dist = dx * dx + dy * dy;
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }
      return Offset(lm[bestIdx][0], lm[bestIdx][1]);
    }

    final results = await Future.wait([
      if (rois.isNotEmpty && rois[0].size > 0)
        _withIrisLeftLock(
          () => irisLeft.runOnImageAlignedIris(
            decoded,
            rois[0],
            isRight: false,
            worker: _worker,
          ),
        )
      else
        Future.value(<List<double>>[]),
      if (rois.length > 1 && rois[1].size > 0)
        _withIrisRightLock(
          () => irisRight.runOnImageAlignedIris(
            decoded,
            rois[1],
            isRight: true,
            worker: _worker,
          ),
        )
      else
        Future.value(<List<double>>[]),
    ]);

    final List<List<double>> leftLm = results[0];
    final List<List<double>> rightLm = results[1];

    final List<Offset> centers = <Offset>[];
    bool usedFallback = false;

    if (rois.isEmpty || rois[0].size <= 0) {
      centers.add(leftFallback ?? const Offset(0, 0));
      usedFallback = true;
      irisUsedFallbackCount++;
    } else {
      if (allLandmarks != null) {
        allLandmarks.addAll(
          leftLm.map(
            (p) => Point(
              p[0].toDouble(),
              p[1].toDouble(),
              p.length > 2 ? p[2].toDouble() : 0.0,
            ),
          ),
        );
      }
      centers.add(pickCenter(leftLm, leftFallback));
    }

    if (rois.length <= 1 || rois[1].size <= 0) {
      centers.add(rightFallback ?? const Offset(0, 0));
      usedFallback = true;
      irisUsedFallbackCount++;
    } else {
      if (allLandmarks != null) {
        allLandmarks.addAll(
          rightLm.map(
            (p) => Point(
              p[0].toDouble(),
              p[1].toDouble(),
              p.length > 2 ? p[2].toDouble() : 0.0,
            ),
          ),
        );
      }
      centers.add(pickCenter(rightLm, rightFallback));
    }

    sw.stop();
    lastIrisTime = sw.elapsed;

    if (centers.isNotEmpty && !usedFallback) {
      irisOkCount++;
    } else {
      irisFailCount++;
    }

    return centers;
  }

  Future<List<Detection>> _detectDetections(
    Uint8List imageBytes, {
    RectF? roi,
    bool refineEyesWithIris = true,
  }) async {
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    return _detectDetectionsWithDecoded(
      decoded,
      roi: roi,
      refineEyesWithIris: refineEyesWithIris,
    );
  }

  Future<List<Detection>> _detectDetectionsWithDecoded(
    img.Image decoded, {
    RectF? roi,
    bool refineEyesWithIris = true,
    List<_DetectionFeatures>? featuresOut,
  }) async {
    final FaceDetection? d = _detector;
    if (d == null) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before detectDetections().',
      );
    }
    if (_irisLeft == null || _irisRight == null) {
      throw StateError(
        'Iris models not initialized. initialize() must succeed before detectDetections().',
      );
    }

    final List<Detection> dets = await d.callWithDecoded(
      decoded,
      roi: roi,
      worker: _worker,
    );
    if (dets.isEmpty) return dets;
    featuresOut?.clear();

    if (!refineEyesWithIris) {
      final double imgW = decoded.width.toDouble();
      final double imgH = decoded.height.toDouble();
      return dets
          .map(
            (det) => Detection(
              boundingBox: det.boundingBox,
              score: det.score,
              keypointsXY: det.keypointsXY,
              imageSize: Size(imgW, imgH),
            ),
          )
          .toList();
    }

    final results = await Future.wait(
      dets.map((det) async {
        final AlignedFace aligned = await estimateAlignedFace(decoded, det);
        final List<Point> mesh = await meshFromAlignedFace(
          aligned.faceCrop,
          aligned,
        );
        final List<AlignedRoi> rois = eyeRoisFromMesh(mesh);

        final double imgW = decoded.width.toDouble();
        final double imgH = decoded.height.toDouble();
        final Offset lf = Offset(
          det.keypointsXY[FaceLandmarkType.leftEye.index * 2] * imgW,
          det.keypointsXY[FaceLandmarkType.leftEye.index * 2 + 1] * imgH,
        );
        final Offset rf = Offset(
          det.keypointsXY[FaceLandmarkType.rightEye.index * 2] * imgW,
          det.keypointsXY[FaceLandmarkType.rightEye.index * 2 + 1] * imgH,
        );

        final List<Point> iris = <Point>[];
        final List<Offset> centers = await _computeIrisCentersWithDecoded(
          decoded,
          rois,
          leftFallback: lf,
          rightFallback: rf,
          allLandmarks: iris,
        );

        final List<double> kp = List<double>.from(det.keypointsXY);
        kp[FaceLandmarkType.leftEye.index * 2] = centers[0].dx / imgW;
        kp[FaceLandmarkType.leftEye.index * 2 + 1] = centers[0].dy / imgH;
        kp[FaceLandmarkType.rightEye.index * 2] = centers[1].dx / imgW;
        kp[FaceLandmarkType.rightEye.index * 2 + 1] = centers[1].dy / imgH;

        final detection = Detection(
          boundingBox: det.boundingBox,
          score: det.score,
          keypointsXY: kp,
          imageSize: Size(imgW, imgH),
        );

        final features = featuresOut != null
            ? _DetectionFeatures(
                detection: detection,
                alignedFace: aligned,
                mesh: mesh,
                iris: iris,
              )
            : null;

        return (detection, features);
      }),
    );

    final List<Detection> updated = <Detection>[];
    for (final (detection, features) in results) {
      updated.add(detection);
      if (features != null) {
        featuresOut!.add(features);
      }
    }
    return updated;
  }

  /// Computes face alignment parameters and extracts an aligned face crop.
  ///
  /// This method analyzes the eye and mouth positions from a face detection
  /// to calculate the face's rotation angle and appropriate crop region. It then
  /// extracts a rotated square crop centered on the face, aligned to a canonical
  /// pose suitable for mesh landmark detection.
  ///
  /// The alignment process:
  /// 1. Computes the angle between the eyes to determine face rotation
  /// 2. Calculates the distance between eyes and between eye-center and mouth
  /// 3. Determines an appropriate crop size based on these distances
  /// 4. Extracts a rotated square crop using [extractAlignedSquare]
  ///
  /// The [decoded] parameter is the source image containing the face.
  ///
  /// The [det] parameter is the face detection containing keypoint positions
  /// for both eyes and mouth.
  ///
  /// Returns an [AlignedFace] object containing:
  /// - Center coordinates ([cx], [cy]) in absolute pixels
  /// - Crop size in pixels
  /// - Rotation angle [theta] in radians
  /// - The extracted aligned face crop image
  ///
  /// The crop size is calculated as the maximum of:
  /// - Eye distance × 4.0
  /// - Mouth distance × 3.6
  ///
  /// This ensures the full face fits within the crop with appropriate padding.
  Future<AlignedFace> estimateAlignedFace(
    img.Image decoded,
    Detection det,
  ) async {
    final imgW = decoded.width.toDouble();
    final imgH = decoded.height.toDouble();

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

    final faceCrop = await extractAlignedSquareWithWorker(
      decoded,
      cx,
      cy,
      size,
      -theta,
      _worker,
    );

    return AlignedFace(
      cx: cx,
      cy: cy,
      size: size,
      theta: theta,
      faceCrop: faceCrop,
    );
  }

  /// Generates 468-point face mesh landmarks from an aligned face crop.
  ///
  /// This method runs the face landmark model on an aligned face crop and
  /// transforms the resulting normalized landmark coordinates back to absolute
  /// pixel coordinates in the original image space using the alignment parameters.
  ///
  /// The transformation process:
  /// 1. Runs the face landmark model on [faceCrop] to get normalized landmarks (0.0 to 1.0)
  /// 2. Converts normalized coordinates to the crop's pixel space
  /// 3. Applies inverse rotation using the angle from [aligned]
  /// 4. Translates points back to original image coordinates
  ///
  /// The [faceCrop] parameter is the aligned face image extracted by [estimateAlignedFace].
  ///
  /// The [aligned] parameter contains the alignment transformation parameters
  /// (center, size, rotation angle) needed to map landmarks back to the original
  /// image coordinates.
  ///
  /// Returns a list of 468 [Point] objects in absolute pixel coordinates
  /// relative to the original decoded image. Each point includes x, y, and z
  /// coordinates where z represents relative depth. Returns an empty list if the
  /// face landmark model is not initialized.
  ///
  /// Each point represents a specific facial feature as defined by MediaPipe's
  /// canonical face mesh topology (eyes, eyebrows, nose, mouth, face contours).
  Future<List<Point>> meshFromAlignedFace(
    img.Image faceCrop,
    AlignedFace aligned,
  ) async {
    if (_meshPool.isEmpty) {
      return const <Point>[];
    }
    final lmNorm = await _withMeshLock(
      (fl) => fl.call(faceCrop, worker: _worker),
    );

    final double ct = math.cos(aligned.theta);
    final double st = math.sin(aligned.theta);
    final double s = aligned.size;
    final double cx = aligned.cx;
    final double cy = aligned.cy;

    final double sct = s * ct;
    final double sst = s * st;
    final double tx = cx - 0.5 * sct + 0.5 * sst;
    final double ty = cy - 0.5 * sst - 0.5 * sct;

    final int n = lmNorm.length;
    final List<Point> mesh = List<Point>.filled(n, const Point(0, 0, 0));

    for (int i = 0; i < n; i++) {
      final List<double> p = lmNorm[i];
      final double nx = p[0];
      final double ny = p[1];
      final double nz = p[2];
      mesh[i] = Point(
        tx + sct * nx - sst * ny,
        ty + sst * nx + sct * ny,
        nz * s,
      );
    }
    return mesh;
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
  /// pixel coordinates, typically from [meshFromAlignedFace].
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

  /// Detects iris landmarks for both eyes using aligned eye regions.
  ///
  /// This method runs iris landmark detection on aligned eye region crops
  /// and returns all iris and eye mesh keypoints in absolute pixel coordinates.
  /// Each eye produces 76 points (71 eye mesh landmarks + 5 iris keypoints).
  ///
  /// The detection process:
  /// 1. For each eye ROI in [rois], extracts an aligned eye crop from [decoded]
  /// 2. Runs iris landmark detection (right eyes are horizontally flipped first)
  /// 3. Transforms the normalized coordinates back to absolute pixels
  /// 4. Returns combined results from both eyes
  ///
  /// The [decoded] parameter is the source image containing the face.
  ///
  /// The [rois] parameter is a list of aligned eye regions, typically from
  /// [eyeRoisFromMesh]. The first ROI should be the left eye, the second
  /// should be the right eye.
  ///
  /// Returns a list of [Point] objects with x, y, and z coordinates in absolute
  /// pixel coordinates. Typically returns 152 total points (76 per eye × 2 eyes)
  /// when both eyes are detected. Returns an empty list if the iris model is not
  /// initialized.
  ///
  /// Each eye's 76 points include:
  /// - 71 eye mesh landmarks (eyelid, eyebrow, and tracking halos)
  /// - 5 iris keypoints (1 center + 4 contour points)
  Future<List<Point>> irisFromEyeRois(
    img.Image decoded,
    List<AlignedRoi> rois,
  ) async {
    if (_irisLeft == null || _irisRight == null) {
      return const <Point>[];
    }
    if (rois.length < 2) {
      return const <Point>[];
    }

    final IrisLandmark irisLeft = _irisLeft!;
    final IrisLandmark irisRight = _irisRight!;

    final results = await Future.wait([
      _withIrisLeftLock(
        () => irisLeft.runOnImageAlignedIris(
          decoded,
          rois[0],
          isRight: false,
          worker: _worker,
        ),
      ),
      _withIrisRightLock(
        () => irisRight.runOnImageAlignedIris(
          decoded,
          rois[1],
          isRight: true,
          worker: _worker,
        ),
      ),
    ]);

    final List<Point> pts = <Point>[];
    for (final List<List<double>> irisLm in results) {
      for (final List<double> p in irisLm) {
        final double x = p[0].toDouble();
        final double y = p[1].toDouble();
        final double z = p.length > 2 ? p[2].toDouble() : 0.0;
        pts.add(Point(x, y, z));
      }
    }
    return pts;
  }

  /// Returns raw face detections with iris-refined eye keypoint positions.
  ///
  /// This method performs face detection and optionally refines the eye keypoints
  /// (left and right eye positions) using iris center estimation for improved
  /// accuracy. This is a lower-level alternative to [detectFaces] that returns
  /// internal detection objects instead of the public [Face] API.
  ///
  /// The [imageBytes] parameter should contain encoded image data (JPEG, PNG, etc.).
  ///
  /// Returns a list of internal [Detection] objects containing:
  /// - Bounding box coordinates
  /// - Confidence score
  /// - 6 facial keypoints (eyes, nose, mouth) with iris-refined eye positions
  ///
  /// **Note:** Most users should prefer [detectFaces] for the high-level API.
  /// This method is primarily for internal use and advanced integration scenarios.
  ///
  /// Throws [StateError] if the detector has not been initialized via [initialize].
  ///
  /// See also:
  /// - [detectFaces] for the main public API with mesh and iris support
  /// - [getDetectionsWithIrisCenters] for explicit iris center refinement
  Future<List<Detection>> getDetections(Uint8List imageBytes) async {
    return await _detectDetections(imageBytes);
  }

  /// Predicts 468 facial landmarks for the given image region.
  ///
  /// The input [imageBytes] is an encoded image (e.g., JPEG/PNG).
  /// Returns an empty list when no face is found.
  Future<List<Point>> getFaceMesh(Uint8List imageBytes) async {
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final List<Detection> dets = await _detectDetectionsWithDecoded(decoded);
    if (dets.isEmpty) return const <Point>[];

    final AlignedFace aligned = await estimateAlignedFace(decoded, dets.first);
    final List<Point> mesh = await meshFromAlignedFace(
      aligned.faceCrop,
      aligned,
    );
    return mesh;
  }

  /// Detects iris landmarks for the first detected face in a full image.
  ///
  /// The [imageBytes] should contain encoded image data (e.g., JPEG/PNG).
  /// Returns up to 152 points (76 per eye: 71 eye mesh + 5 iris keypoints) in absolute pixels.
  /// If no face or iris is found, returns an empty list.
  Future<List<Point>> getEyeMeshWithIris(Uint8List imageBytes) async {
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);

    final List<Detection> dets = await _detectDetectionsWithDecoded(decoded);
    if (dets.isEmpty) return const <Point>[];
    final AlignedFace aligned = await estimateAlignedFace(decoded, dets.first);
    final List<Point> mesh = await meshFromAlignedFace(
      aligned.faceCrop,
      aligned,
    );
    final List<AlignedRoi> rois = eyeRoisFromMesh(mesh);
    return await irisFromEyeRois(decoded, rois);
  }

  /// Returns the dimensions of the decoded image.
  ///
  /// This utility method decodes the provided [imageBytes] and returns the
  /// original width and height as a [Size] object.
  ///
  /// Useful for determining image dimensions before running detection, or for
  /// coordinate calculations when working with raw detection data.
  ///
  /// Example:
  /// ```dart
  /// final size = await detector.getOriginalSize(imageBytes);
  /// print('Image is ${size.width}x${size.height} pixels');
  /// ```
  Future<Size> getOriginalSize(Uint8List imageBytes) async {
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);

    return Size(decoded.width.toDouble(), decoded.height.toDouble());
  }

  /// Generates 468-point face meshes from existing face detections.
  ///
  /// This method takes a list of face detections (typically from [getDetections])
  /// and computes the detailed 468-point facial mesh for each detected face. This
  /// is useful when you want to process detections in stages or cache detection
  /// results before computing the mesh.
  ///
  /// The mesh generation process:
  /// 1. For each detection, aligns the face to a canonical pose
  /// 2. Runs the face mesh model on the aligned face crop
  /// 3. Transforms mesh points back to original image coordinates
  ///
  /// The [imageBytes] parameter should contain the same encoded image data used
  /// for the detections.
  ///
  /// The [dets] parameter should be a list of detections from [getDetections].
  ///
  /// Returns a list of face meshes, where each mesh is a list of 468 points in
  /// absolute image coordinates. Returns an empty list if [dets] is empty.
  ///
  /// Example:
  /// ```dart
  /// final detections = await detector.getDetections(imageBytes);
  /// final meshes = await detector.getFaceMeshFromDetections(imageBytes, detections);
  /// // meshes[0] contains 468 points for the first face
  /// ```
  ///
  /// See also:
  /// - [getFaceMesh] for single-face mesh detection
  /// - [detectFaces] for the complete pipeline in one call
  Future<List<List<Point>>> getFaceMeshFromDetections(
    Uint8List imageBytes,
    List<Detection> dets,
  ) async {
    if (dets.isEmpty) return const <List<Point>>[];
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final List<List<Point>> out = await Future.wait(
      dets.map((Detection det) async {
        final AlignedFace aligned = await estimateAlignedFace(decoded, det);
        return meshFromAlignedFace(aligned.faceCrop, aligned);
      }),
    );
    return out;
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

  /// Detects faces in the provided image and returns detailed results.
  ///
  /// The [imageBytes] parameter should contain encoded image data (JPEG, PNG, etc.).
  /// The [mode] parameter controls which features are computed:
  /// - [FaceDetectionMode.fast]: Only detection and landmarks
  /// - [FaceDetectionMode.standard]: Adds 468-point face mesh
  /// - [FaceDetectionMode.full]: Adds iris tracking (152 points: 76 per eye)
  ///
  /// Returns a [List] of [Face] objects, one per detected face. Each [Face] includes
  /// bounding box corners, facial landmarks, and optionally mesh and iris data
  /// depending on the mode.
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  Future<List<Face>> detectFaces(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    final cv.Mat image = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    if (image.isEmpty) {
      throw FormatException('Could not decode image bytes');
    }
    try {
      return await detectFacesFromMat(image, mode: mode);
    } finally {
      image.dispose();
    }
  }

  /// Estimates an aligned face crop using a native frame ID.
  ///
  /// This is a variant of [estimateAlignedFace] optimized for native image
  /// pipelines. Instead of passing a decoded image, this method references
  /// an image already stored in native memory via its [frameId].
  ///
  /// The alignment process:
  /// 1. Computes the angle between the eyes to determine face rotation
  /// 2. Calculates the distance between eyes and between eye-center and mouth
  /// 3. Determines an appropriate crop size based on these distances
  /// 4. Extracts a rotated square crop from the native frame
  ///
  /// The [frameId] parameter references a native image previously registered
  /// with the worker.
  ///
  /// The [imgW] and [imgH] parameters specify the dimensions of the source image.
  ///
  /// The [det] parameter is the face detection containing keypoint positions
  /// for both eyes and mouth.
  ///
  /// Returns an [AlignedFace] object containing center coordinates, crop size,
  /// rotation angle, and the extracted aligned face crop image.
  Future<AlignedFace> estimateAlignedFaceWithFrameId(
    int frameId,
    int imgW,
    int imgH,
    Detection det,
  ) async {
    final double imgWd = imgW.toDouble();
    final double imgHd = imgH.toDouble();

    final lx = det.keypointsXY[FaceLandmarkType.leftEye.index * 2] * imgWd;
    final ly = det.keypointsXY[FaceLandmarkType.leftEye.index * 2 + 1] * imgHd;
    final rx = det.keypointsXY[FaceLandmarkType.rightEye.index * 2] * imgWd;
    final ry = det.keypointsXY[FaceLandmarkType.rightEye.index * 2 + 1] * imgHd;
    final mx = det.keypointsXY[FaceLandmarkType.mouth.index * 2] * imgWd;
    final my = det.keypointsXY[FaceLandmarkType.mouth.index * 2 + 1] * imgHd;

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

    final faceCrop = await _worker!.extractAlignedSquareWithFrameId(
      frameId,
      cx,
      cy,
      size,
      -theta,
    );

    return AlignedFace(
      cx: cx,
      cy: cy,
      size: size,
      theta: theta,
      faceCrop: faceCrop,
    );
  }

  /// Detects iris landmarks from eye ROIs using a native frame ID.
  ///
  /// This is a variant of [irisFromEyeRois] optimized for native image
  /// pipelines. Instead of passing a decoded image, this method references
  /// an image already stored in native memory via its [frameId].
  ///
  /// The [frameId] parameter references a native image previously registered
  /// with the worker.
  ///
  /// The [rois] parameter is a list of aligned eye regions, typically from
  /// [eyeRoisFromMesh]. The first ROI should be the left eye, the second
  /// should be the right eye.
  ///
  /// Returns a list of [Point] objects with x, y, and z coordinates in absolute
  /// pixel coordinates. Typically returns 152 total points (76 per eye × 2 eyes)
  /// when both eyes are detected. Returns an empty list if the iris model is not
  /// initialized or if fewer than 2 ROIs are provided.
  ///
  /// Each eye's 76 points include:
  /// - 71 eye mesh landmarks (eyelid, eyebrow, and tracking halos)
  /// - 5 iris keypoints (1 center + 4 contour points)
  Future<List<Point>> irisFromEyeRoisWithFrameId(
    int frameId,
    List<AlignedRoi> rois,
  ) async {
    if (_irisLeft == null || _irisRight == null) return <Point>[];
    if (rois.length < 2) return <Point>[];

    final IrisLandmark irisLeft = _irisLeft!;
    final IrisLandmark irisRight = _irisRight!;

    final results = await Future.wait([
      _withIrisLeftLock(
        () => irisLeft.runOnImageAlignedIrisWithFrameId(
          frameId,
          rois[0],
          isRight: false,
          worker: _worker,
        ),
      ),
      _withIrisRightLock(
        () => irisRight.runOnImageAlignedIrisWithFrameId(
          frameId,
          rois[1],
          isRight: true,
          worker: _worker,
        ),
      ),
    ]);

    final List<Point> pts = <Point>[];
    for (final List<List<double>> lm in results) {
      for (final List<double> p in lm) {
        final double z = p.length > 2 ? p[2].toDouble() : 0.0;
        pts.add(Point(p[0], p[1], z));
      }
    }

    return pts;
  }

  /// Detects faces in a cv.Mat image using OpenCV-accelerated processing.
  ///
  /// This is the OpenCV-based variant of [detectFaces] that accepts a cv.Mat
  /// directly and uses SIMD-accelerated operations for image transformations.
  /// This provides 10-50x faster rotation/crop operations compared to pure Dart.
  ///
  /// The [image] parameter should be a cv.Mat in BGR format (as returned by
  /// cv.imdecode or cv.imread). The Mat is NOT disposed by this method.
  ///
  /// The [mode] parameter controls which features are computed:
  /// - [FaceDetectionMode.fast]: Only detection and landmarks
  /// - [FaceDetectionMode.standard]: Adds 468-point face mesh
  /// - [FaceDetectionMode.full]: Adds iris tracking (152 points: 76 per eye)
  ///
  /// Returns a [List] of [Face] objects, one per detected face.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
  /// final faces = await detector.detectFacesFromMat(mat);
  /// mat.dispose(); // Don't forget to dispose the Mat!
  /// ```
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  Future<List<Face>> detectFacesFromMat(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    if (_detector == null) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before detectFacesFromMat().',
      );
    }

    final int width = image.cols;
    final int height = image.rows;
    final Size imgSize = Size(width.toDouble(), height.toDouble());

    final bool computeIris = mode == FaceDetectionMode.full;
    final bool computeMesh =
        mode == FaceDetectionMode.standard || mode == FaceDetectionMode.full;

    final List<Detection> dets = await _detectDetectionsFromMat(image);
    if (dets.isEmpty) return <Face>[];

    final List<(Detection, AlignedFaceFromMat)?> alignedFaces =
        <(Detection, AlignedFaceFromMat)?>[];
    for (final Detection det in dets) {
      try {
        final aligned = await _estimateAlignedFaceFromMat(image, det);
        alignedFaces.add((det, aligned));
      } catch (e) {
        alignedFaces.add(null); // Mark failed extraction
      }
    }

    final List<List<Point>?> meshResults;
    if (computeMesh) {
      meshResults = await Future.wait(
        alignedFaces.map((data) async {
          if (data == null) return null;
          try {
            return await _meshFromAlignedFaceFromMat(
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
          irisResults[i] = await _irisFromMeshFromMat(image, meshPx);
        } catch (e) {
          // Iris detection failure shouldn't block face detection
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
  Future<List<Detection>> _detectDetectionsFromMat(cv.Mat image) async {
    final FaceDetection? d = _detector;
    if (d == null) {
      throw StateError('FaceDetector not initialized.');
    }

    final ImageTensor tensor = convertImageToTensorFromMat(
      image,
      outW: d.inputWidth,
      outH: d.inputHeight,
    );

    return await d.callWithTensor(tensor);
  }

  /// Internal: Aligned face data holder for Mat-based processing.
  Future<AlignedFaceFromMat> _estimateAlignedFaceFromMat(
    cv.Mat image,
    Detection det,
  ) async {
    final double imgW = image.cols.toDouble();
    final double imgH = image.rows.toDouble();

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

    final cv.Mat? faceCrop = extractAlignedSquareFromMat(
      image,
      cx,
      cy,
      size,
      -theta,
    );

    if (faceCrop == null) {
      throw StateError('Failed to extract aligned face crop');
    }

    return AlignedFaceFromMat(
      cx: cx,
      cy: cy,
      size: size,
      theta: theta,
      faceCrop: faceCrop,
    );
  }

  /// Internal: Generate mesh from aligned face cv.Mat.
  Future<List<Point>> _meshFromAlignedFaceFromMat(
    cv.Mat faceCrop,
    double cx,
    double cy,
    double size,
    double theta,
  ) async {
    if (_meshPool.isEmpty) return <Point>[];

    final lmNorm = await _withMeshLock((fl) => fl.callFromMat(faceCrop));

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

  /// Internal: Get iris landmarks from mesh using cv.Mat source.
  ///
  /// Eye crop extraction (warpAffine) is done serially to avoid opencv_dart
  /// freeze issues, but TFLite inference runs in parallel for performance.
  Future<List<Point>> _irisFromMeshFromMat(
    cv.Mat image,
    List<Point> meshAbs,
  ) async {
    if (_irisLeft == null || _irisRight == null) return <Point>[];
    if (meshAbs.length < 468) return <Point>[];

    final List<AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
    if (rois.length < 2) return <Point>[];

    final cv.Mat? leftCrop = extractAlignedSquareFromMat(
      image,
      rois[0].cx,
      rois[0].cy,
      rois[0].size,
      rois[0].theta,
    );
    final cv.Mat? rightCropRaw = extractAlignedSquareFromMat(
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
        _withIrisLeftLock(() => _irisLeft!.callFromMat(leftCrop)),
        _withIrisRightLock(() => _irisRight!.callFromMat(rightCrop)),
      ]);
    } finally {
      leftCrop.dispose();
      rightCrop.dispose();
    }

    final List<List<double>> leftLmNorm = results[0];
    final List<List<double>> rightLmNorm = results[1];

    final double ctL = math.cos(rois[0].theta);
    final double stL = math.sin(rois[0].theta);
    final double sL = rois[0].size;

    final List<Point> pts = <Point>[];
    for (final List<double> p in leftLmNorm) {
      final double lx2 = (p[0] - 0.5) * sL;
      final double ly2 = (p[1] - 0.5) * sL;
      final double x = rois[0].cx + lx2 * ctL - ly2 * stL;
      final double y = rois[0].cy + lx2 * stL + ly2 * ctL;
      pts.add(Point(x, y, p.length > 2 ? p[2] : 0.0));
    }

    final double ctR = math.cos(rois[1].theta);
    final double stR = math.sin(rois[1].theta);
    final double sR = rois[1].size;

    for (final List<double> p in rightLmNorm) {
      final double px = 1.0 - p[0];
      final double py = p[1];
      final double lx2 = (px - 0.5) * sR;
      final double ly2 = (py - 0.5) * sR;
      final double x = rois[1].cx + lx2 * ctR - ly2 * stR;
      final double y = rois[1].cy + lx2 * stR + ly2 * ctR;
      pts.add(Point(x, y, p.length > 2 ? p[2] : 0.0));
    }

    if (pts.isNotEmpty) {
      irisOkCount++;
    } else {
      irisFailCount++;
    }

    return pts;
  }

  /// Finds the iris center from a list of iris contour points.
  ///
  /// Uses the same algorithm as [_computeIrisCentersWithDecoded]: finds the
  /// point with minimum sum of squared distances to all other points, which
  /// geometrically identifies the center of the iris contour.
  ///
  /// This is equivalent to finding the point closest to the centroid, which
  /// can be computed in O(n) instead of O(n²).
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

  /// Serializes embedding inference calls to prevent buffer race conditions.
  ///
  /// The FaceEmbedding model uses shared mutable buffers that cannot be safely
  /// accessed concurrently. This method ensures only one inference runs
  /// at a time by chaining futures.
  Future<T> _withEmbeddingLock<T>(Future<T> Function() fn) async {
    final previous = _embeddingInferenceLock;
    final completer = Completer<void>();
    _embeddingInferenceLock = completer.future;

    try {
      await previous;
      return await fn();
    } finally {
      completer.complete();
    }
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
  /// The [imageBytes] parameter should contain the same encoded image data
  /// that was used for detection.
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
  Future<Float32List> getFaceEmbedding(Face face, Uint8List imageBytes) async {
    if (_embedding == null) {
      throw StateError(
        'Embedding model not initialized. Call initialize() before getFaceEmbedding().',
      );
    }

    final cv.Mat image = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    if (image.isEmpty) {
      throw FormatException('Could not decode image bytes');
    }

    try {
      return await getFaceEmbeddingFromMat(face, image);
    } finally {
      image.dispose();
    }
  }

  /// Generates a face embedding from a cv.Mat image.
  ///
  /// This is the OpenCV-based variant of [getFaceEmbedding] that accepts a
  /// cv.Mat directly, providing better performance when you already have
  /// the image in Mat format.
  ///
  /// The [face] parameter should be a face detection result.
  ///
  /// The [image] parameter should contain the source image as cv.Mat.
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
  ///
  /// Returns a [Float32List] containing the L2-normalized embedding vector.
  ///
  /// Throws [StateError] if the embedding model has not been initialized.
  Future<Float32List> getFaceEmbeddingFromMat(Face face, cv.Mat image) async {
    if (_embedding == null) {
      throw StateError(
        'Embedding model not initialized. Call initialize() before getFaceEmbeddingFromMat().',
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

    final cv.Mat? faceCrop = extractAlignedSquareFromMat(
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
        return await _withEmbeddingLock(() => _embedding!.callFromMat(resized));
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

  // ============================================================================
  // Selfie Segmentation
  // ============================================================================

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
  /// [imageBytes]: Encoded image (JPEG, PNG, etc.)
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
  Future<SegmentationMask> getSegmentationMask(Uint8List imageBytes) async {
    if (_segmenter == null) {
      throw StateError(
        'Segmentation not initialized. Call initializeSegmentation() first.',
      );
    }
    return _withSegmentationLock(() => _segmenter!.call(imageBytes));
  }

  /// Segments a cv.Mat image.
  ///
  /// This is the OpenCV-based variant of [getSegmentationMask] that accepts
  /// a cv.Mat directly, providing better performance when you already have
  /// the image in Mat format.
  ///
  /// The [image] parameter should contain the source image as cv.Mat.
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
  ///
  /// Throws [StateError] if [initializeSegmentation] hasn't been called.
  /// Throws [SegmentationException] on inference failure.
  Future<SegmentationMask> getSegmentationMaskFromMat(cv.Mat image) async {
    if (_segmenter == null) {
      throw StateError(
        'Segmentation not initialized. Call initializeSegmentation() first.',
      );
    }
    return _withSegmentationLock(() => _segmenter!.callFromMat(image));
  }

  /// Concurrency lock for segmentation - only one at a time to prevent memory thrashing.
  Future<T> _withSegmentationLock<T>(Future<T> Function() fn) async {
    final previous = _segmentationInferenceLock;
    final completer = Completer<void>();
    _segmentationInferenceLock = completer.future;
    try {
      await previous;
      return await fn();
    } finally {
      completer.complete();
    }
  }

  /// Releases all resources held by the detector.
  ///
  /// Call this when you're done using the detector to free up memory.
  /// After calling dispose, you must call [initialize] again before
  /// running any detections.
  void dispose() {
    _worker?.dispose();
    _detector?.dispose();
    for (final mesh in _meshPool) {
      mesh.dispose();
    }
    _meshPool.clear();
    _meshInferenceLocks.clear();
    _irisLeft?.dispose();
    _irisRight?.dispose();
    _embedding?.dispose();
    _segmenter?.dispose();
    _segmenter = null;
  }

  @pragma('vm:entry-point')
  static Future<void> _irisCentersIsolate(Map<String, dynamic> params) async {
    final RootIsolateToken token = params['rootToken'] as RootIsolateToken;
    DartPluginRegistrant.ensureInitialized();
    BackgroundIsolateBinaryMessenger.ensureInitialized(token);

    final SendPort sp = params['sendPort'] as SendPort;
    final Uint8List bytes = params['imageBytes'] as Uint8List;
    final List roisData = params['rois'] as List;

    try {
      final IrisLandmark iris = await IrisLandmark.create();
      final img.Image? decoded = img.decodeImage(bytes);
      if (decoded == null) {
        sp.send({'ok': false});
        return;
      }

      final List<Map<String, double>> centers = <Map<String, double>>[];
      for (int i = 0; i < roisData.length; i++) {
        final Map<dynamic, dynamic> m = roisData[i] as Map;
        final AlignedRoi roi = AlignedRoi(
          (m['cx'] as num).toDouble(),
          (m['cy'] as num).toDouble(),
          (m['size'] as num).toDouble(),
          (m['theta'] as num).toDouble(),
        );
        final List<List<double>> lm = await iris.runOnImageAlignedIris(
          decoded,
          roi,
          isRight: i == 1,
        );
        if (lm.isEmpty) {
          final double? fx = i == 0
              ? (params['leftFx'] as double?)
              : (params['rightFx'] as double?);
          final double? fy = i == 0
              ? (params['leftFy'] as double?)
              : (params['rightFy'] as double?);
          centers.add({'x': (fx ?? 0.0), 'y': (fy ?? 0.0)});
        } else {
          double sx = 0.0, sy = 0.0;
          for (final List<double> p in lm) {
            sx += (p[0] as num).toDouble();
            sy += (p[1] as num).toDouble();
          }
          final double cx = sx / lm.length;
          final double cy = sy / lm.length;
          centers.add({'x': cx, 'y': cy});
        }
      }

      iris.dispose();
      sp.send({'ok': true, 'centers': centers});
    } catch (e) {
      sp.send({'ok': false, 'err': e.toString()});
    }
  }
}
