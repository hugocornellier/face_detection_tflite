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
/// This class orchestrates three TensorFlow Lite models to provide comprehensive
/// facial analysis:
/// - Face detection with 6 keypoints (eyes, nose, mouth corners)
/// - 468-point face mesh for detailed facial geometry
/// - Iris landmark detection with 76 points per eye (71 eye mesh + 5 iris keypoints)
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

  // Serialization chains to prevent concurrent model inference (race conditions)
  // Mesh pool uses multiple locks for parallel inference across multiple faces
  final List<Future<void>> _meshInferenceLocks = [];
  Future<void> _irisLeftInferenceLock = Future.value();
  Future<void> _irisRightInferenceLock = Future.value();

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
  Future<void> initialize({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    InterpreterOptions? options,
    int meshPoolSize = 3,
  }) async {
    await _ensureTFLiteLoaded();
    try {
      _worker = IsolateWorker();
      await _worker!.initialize();

      _detector = await FaceDetection.create(
        model,
        options: options,
      );

      // Create pool of mesh models for parallel multi-face inference
      // Each model has its own buffers to prevent race conditions
      _meshPool.clear();
      _meshInferenceLocks.clear();
      for (int i = 0; i < meshPoolSize; i++) {
        _meshPool.add(await FaceLandmark.create(options: options));
        _meshInferenceLocks.add(Future.value());
      }

      // Create separate iris models for parallel left/right eye inference
      // Each model has its own buffers to prevent race conditions
      _irisLeft = await IrisLandmark.create(options: options);
      _irisRight = await IrisLandmark.create(options: options);
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
      _worker = null;
      _detector = null;
      _irisLeft = null;
      _irisRight = null;
      rethrow;
    }
  }

  static ffi.DynamicLibrary? _tfliteLib;
  static Future<void> _ensureTFLiteLoaded() async {
    if (_tfliteLib != null) return;

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
    } else if (Platform.isLinux) {
      candidates = [
        p.join(exeDir.path, 'lib', 'libtensorflowlite_c-linux.so'),
        'libtensorflowlite_c-linux.so',
      ];
      hint = 'Ensure linux/CMakeLists.txt sets:\n'
          '  set(PLUGIN_NAME_bundled_libraries "../assets/bin/libtensorflowlite_c-linux.so" PARENT_SCOPE)\n'
          'so Flutter copies it into bundle/lib/.';
    } else if (Platform.isMacOS) {
      final contents = exeDir.parent;
      candidates = [
        p.join(contents.path, 'Resources', 'libtensorflowlite_c-mac.dylib'),
      ];
      hint = 'Expected in app bundle Resources, or resolvable by name.';
    } else {
      _tfliteLib = ffi.DynamicLibrary.process();
      return;
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

    // Round-robin selection to distribute load
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
      final List<Offset> pts =
          lm.map((p) => Offset(p[0].toDouble(), p[1].toDouble())).toList();
      int bestIdx = 0;
      double bestScore = double.infinity;
      for (int k = 0; k < pts.length; k++) {
        double s = 0;
        for (int j = 0; j < pts.length; j++) {
          if (j == k) continue;
          final double dx = pts[j].dx - pts[k].dx;
          final double dy = pts[j].dy - pts[k].dy;
          s += dx * dx + dy * dy;
        }
        if (s < bestScore) {
          bestScore = s;
          bestIdx = k;
        }
      }
      return pts[bestIdx];
    }

    // Run left and right eye iris detection in parallel using separate models
    final results = await Future.wait([
      // Left eye (index 0)
      if (rois.isNotEmpty && rois[0].size > 0)
        _withIrisLeftLock(() => irisLeft.runOnImageAlignedIris(
              decoded,
              rois[0],
              isRight: false,
              worker: _worker,
            ))
      else
        Future.value(<List<double>>[]),
      // Right eye (index 1)
      if (rois.length > 1 && rois[1].size > 0)
        _withIrisRightLock(() => irisRight.runOnImageAlignedIris(
              decoded,
              rois[1],
              isRight: true,
              worker: _worker,
            ))
      else
        Future.value(<List<double>>[]),
    ]);

    final List<List<double>> leftLm = results[0];
    final List<List<double>> rightLm = results[1];

    // Build centers list and allLandmarks
    final List<Offset> centers = <Offset>[];
    bool usedFallback = false;

    // Process left eye
    if (rois.isEmpty || rois[0].size <= 0) {
      centers.add(leftFallback ?? const Offset(0, 0));
      usedFallback = true;
      irisUsedFallbackCount++;
    } else {
      if (allLandmarks != null) {
        allLandmarks.addAll(
          leftLm.map((p) => Point(
                p[0].toDouble(),
                p[1].toDouble(),
                p.length > 2 ? p[2].toDouble() : 0.0,
              )),
        );
      }
      centers.add(pickCenter(leftLm, leftFallback));
    }

    // Process right eye
    if (rois.length <= 1 || rois[1].size <= 0) {
      centers.add(rightFallback ?? const Offset(0, 0));
      usedFallback = true;
      irisUsedFallbackCount++;
    } else {
      if (allLandmarks != null) {
        allLandmarks.addAll(
          rightLm.map((p) => Point(
                p[0].toDouble(),
                p[1].toDouble(),
                p.length > 2 ? p[2].toDouble() : 0.0,
              )),
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
    // Use pool to allow parallel mesh inference for multiple faces
    final lmNorm =
        await _withMeshLock((fl) => fl.call(faceCrop, worker: _worker));
    final ct = math.cos(aligned.theta);
    final st = math.sin(aligned.theta);
    final s = aligned.size;
    final cx = aligned.cx;
    final cy = aligned.cy;
    final mesh = <Point>[];

    for (final p in lmNorm) {
      final lx2 = (p[0] - 0.5) * s;
      final ly2 = (p[1] - 0.5) * s;
      final x = cx + lx2 * ct - ly2 * st;
      final y = cy + lx2 * st + ly2 * ct;
      final z = p[2] * s;
      mesh.add(Point(x, y, z));
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

    // Run left and right eye iris detection in parallel using separate models
    final results = await Future.wait([
      // Left eye (index 0)
      _withIrisLeftLock(() => irisLeft.runOnImageAlignedIris(
            decoded,
            rois[0],
            isRight: false,
            worker: _worker,
          )),
      // Right eye (index 1)
      _withIrisRightLock(() => irisRight.runOnImageAlignedIris(
            decoded,
            rois[1],
            isRight: true,
            worker: _worker,
          )),
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
  /// - [getIrisFromMesh] to generate iris landmarks from these meshes
  /// - [detectFaces] for the complete pipeline in one call
  Future<List<List<Point>>> getFaceMeshFromDetections(
    Uint8List imageBytes,
    List<Detection> dets,
  ) async {
    if (dets.isEmpty) return const <List<Point>>[];
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final List<List<Point>> out = <List<Point>>[];
    for (final Detection det in dets) {
      final AlignedFace aligned = await estimateAlignedFace(decoded, det);
      final List<Point> mesh = await meshFromAlignedFace(
        aligned.faceCrop,
        aligned,
      );
      out.add(mesh);
    }
    return out;
  }

  /// Generates iris landmarks from face mesh data.
  ///
  /// **DEPRECATED:** Use [detectFaces] with [FaceDetectionMode.full] and access
  /// `face.eyes` for structured iris data instead. This method is superseded by
  /// the high-level API which provides better structured access to iris landmarks
  /// via `face.eyes.leftEye.irisCenter` and `face.eyes.leftEye.irisContour`.
  ///
  /// This method takes pre-computed 468-point face meshes and extracts iris
  /// landmarks for each face. The face mesh is used to locate eye regions,
  /// which are then processed by the iris landmark model to detect 152 keypoints
  /// (76 per eye: 71 eye mesh + 5 iris keypoints) total per face.
  ///
  /// The iris detection process:
  /// 1. Extracts eye corner landmarks from the mesh (indices 33, 133, 362, 263)
  /// 2. Computes aligned eye region crops
  /// 3. Runs iris landmark detection on each eye
  /// 4. Transforms iris points back to original image coordinates
  ///
  /// The [imageBytes] parameter should contain the same encoded image data used
  /// for the mesh generation.
  ///
  /// The [meshesPerFace] parameter should be a list of face meshes, typically
  /// from [getFaceMeshFromDetections].
  ///
  /// Returns a list of iris landmark sets, where each set contains points for
  /// both eyes (left and right). Returns an empty list if [meshesPerFace] is empty.
  /// If a face mesh is empty, returns an empty landmark set for that face.
  ///
  /// Example:
  /// ```dart
  /// final meshes = await detector.getFaceMeshFromDetections(imageBytes, detections);
  /// final irises = await detector.getIrisFromMesh(imageBytes, meshes);
  /// // irises[0] contains 152 iris and eye mesh points (76 per eye) for the first face
  /// ```
  ///
  /// See also:
  /// - [getEyeMeshWithIris] for single-face iris detection
  /// - [getFaceMeshFromDetections] to generate the required mesh input
  /// - [detectFaces] for the complete pipeline in one call (recommended)
  @Deprecated(
    'Use detectFaces() with FaceDetectionMode.full and access face.eyes instead. '
    'Will be removed in v5.0.0.',
  )
  Future<List<List<Point>>> getIrisFromMesh(
    Uint8List imageBytes,
    List<List<Point>> meshesPerFace,
  ) async {
    if (meshesPerFace.isEmpty) return const <List<Point>>[];
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final out = <List<Point>>[];
    for (final meshPts in meshesPerFace) {
      if (meshPts.isEmpty) {
        out.add(const <Point>[]);
        continue;
      }
      final List<AlignedRoi> rois = eyeRoisFromMesh(meshPts);
      final List<Point> iris = await irisFromEyeRois(decoded, rois);
      out.add(iris);
    }
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
  List<List<Point>> splitMeshesIfConcatenated(
    List<Point> meshPts,
  ) {
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

  Future<List<Point>> _getMeshForFace(AlignedFace aligned) async {
    return await meshFromAlignedFace(
      aligned.faceCrop,
      aligned,
    );
  }

  Future<List<Point>> _getIrisForFace(
    img.Image decoded,
    List<Point> meshPx,
  ) async {
    if (meshPx.isEmpty) return <Point>[];

    final List<AlignedRoi> rois = eyeRoisFromMesh(meshPx);
    return await irisFromEyeRois(decoded, rois);
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
    // Fast path: decode and register in worker, avoid RGB round-trip
    if (_worker != null && _worker!.isInitialized) {
      final (frameId, w, h) = await _worker!.decodeAndRegisterFrame(imageBytes);
      final Size imgSize = Size(w.toDouble(), h.toDouble());
      return _detectFacesWithFrameId(frameId, w, h, mode, imgSize);
    }

    // Fallback: no worker available
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final Size imgSize = Size(
      decoded.width.toDouble(),
      decoded.height.toDouble(),
    );

    final bool computeIris = mode == FaceDetectionMode.full;
    final bool computeMesh =
        mode == FaceDetectionMode.standard || mode == FaceDetectionMode.full;
    final List<_DetectionFeatures> features =
        computeIris ? <_DetectionFeatures>[] : const <_DetectionFeatures>[];
    final List<Detection> dets = await _detectDetectionsWithDecoded(
      decoded,
      refineEyesWithIris: computeIris,
      featuresOut: computeIris ? features : null,
    );
    final List<AlignedFace> allAligned = computeMesh && !computeIris
        ? await Future.wait(
            dets.map((det) => estimateAlignedFace(decoded, det)),
          )
        : <AlignedFace>[];

    final List<Face> faces = <Face>[];
    for (int i = 0; i < dets.length; i++) {
      try {
        final _DetectionFeatures? feat =
            computeIris && i < features.length ? features[i] : null;
        final Detection det = feat?.detection ?? dets[i];

        final List<Point> meshPx =
            computeMesh && feat != null && feat.mesh.isNotEmpty
                ? feat.mesh
                : computeMesh && i < allAligned.length
                    ? await _getMeshForFace(allAligned[i])
                    : <Point>[];

        final List<Point> irisPx =
            computeIris && feat != null && feat.iris.isNotEmpty
                ? feat.iris
                : computeIris && i < allAligned.length
                    ? await _getIrisForFace(decoded, meshPx)
                    : computeIris
                        ? await _getIrisForFace(decoded, meshPx)
                        : <Point>[];

        final FaceMesh? faceMesh = meshPx.isNotEmpty ? FaceMesh(meshPx) : null;

        faces.add(
          Face(
            detection: det,
            mesh: faceMesh,
            irises: irisPx,
            originalSize: imgSize,
          ),
        );
      } catch (e) {
        // ignore: silently skip failed face
      }
    }

    return faces;
  }

  /// Optimized face detection using frame registration to avoid redundant transfers.
  ///
  /// The frame is already registered in the worker isolate, so we don't need
  /// to transfer the decoded image.
  Future<List<Face>> _detectFacesWithFrameId(
    int frameId,
    int width,
    int height,
    FaceDetectionMode mode,
    Size imgSize,
  ) async {
    final IsolateWorker worker = _worker!;

    try {
      final bool computeIris = mode == FaceDetectionMode.full;
      final List<_DetectionFeatures> features =
          computeIris ? <_DetectionFeatures>[] : const <_DetectionFeatures>[];

      final bool computeMesh =
          mode == FaceDetectionMode.standard || mode == FaceDetectionMode.full;
      final List<Detection> dets = await _detectDetectionsWithFrameId(
          frameId, width, height, computeIris, computeIris ? features : null);
      final List<AlignedFace> allAligned = computeMesh && !computeIris
          ? await Future.wait(dets.map((det) => estimateAlignedFaceWithFrameId(
                frameId,
                width,
                height,
                det,
              )))
          : <AlignedFace>[];

      final List<Face> faces = <Face>[];
      for (int i = 0; i < dets.length; i++) {
        try {
          final _DetectionFeatures? feat =
              computeIris && i < features.length ? features[i] : null;
          final Detection det = feat?.detection ?? dets[i];

          final List<Point> meshPx =
              computeMesh && feat != null && feat.mesh.isNotEmpty
                  ? feat.mesh
                  : computeMesh && i < allAligned.length
                      ? await _getMeshForFace(allAligned[i])
                      : <Point>[];

          final List<Point> irisPx = computeIris &&
                  feat != null &&
                  feat.iris.isNotEmpty
              ? feat.iris
              : computeIris && i < allAligned.length
                  ? await _getIrisForFaceWithFrameId(
                      frameId, meshPx, width.toDouble(), height.toDouble())
                  : computeIris
                      ? await _getIrisForFaceWithFrameId(
                          frameId, meshPx, width.toDouble(), height.toDouble())
                      : <Point>[];

          final FaceMesh? faceMesh =
              meshPx.isNotEmpty ? FaceMesh(meshPx) : null;

          faces.add(
            Face(
              detection: det,
              mesh: faceMesh,
              irises: irisPx,
              originalSize: imgSize,
            ),
          );
        } catch (e) {
          // ignore: silently skip failed face
        }
      }

      return faces;
    } finally {
      await worker.releaseFrame(frameId);
    }
  }

  /// Detection pipeline using frame ID to avoid transferring full image.
  Future<List<Detection>> _detectDetectionsWithFrameId(
    int frameId,
    int width,
    int height,
    bool refineEyesWithIris,
    List<_DetectionFeatures>? featuresOut,
  ) async {
    final FaceDetection? d = _detector;
    if (d == null) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before detection.',
      );
    }
    if (_irisLeft == null || _irisRight == null) {
      throw StateError('Iris models not initialized.');
    }

    final List<Detection> dets =
        await d.callWithFrameId(frameId, width, height, worker: _worker);
    if (dets.isEmpty) return dets;
    featuresOut?.clear();

    if (!refineEyesWithIris) {
      final double imgW = width.toDouble();
      final double imgH = height.toDouble();
      return dets
          .map((det) => Detection(
                boundingBox: det.boundingBox,
                score: det.score,
                keypointsXY: det.keypointsXY,
                imageSize: Size(imgW, imgH),
              ))
          .toList();
    }

    final results = await Future.wait(
      dets.map((det) async {
        final AlignedFace aligned = await estimateAlignedFaceWithFrameId(
          frameId,
          width,
          height,
          det,
        );
        final List<Point> mesh = await meshFromAlignedFace(
          aligned.faceCrop,
          aligned,
        );
        final List<AlignedRoi> rois = eyeRoisFromMesh(mesh);

        final double imgW = width.toDouble();
        final double imgH = height.toDouble();
        final Offset lf = Offset(
          det.keypointsXY[FaceLandmarkType.leftEye.index * 2] * imgW,
          det.keypointsXY[FaceLandmarkType.leftEye.index * 2 + 1] * imgH,
        );
        final Offset rf = Offset(
          det.keypointsXY[FaceLandmarkType.rightEye.index * 2] * imgW,
          det.keypointsXY[FaceLandmarkType.rightEye.index * 2 + 1] * imgH,
        );

        final List<Point> iris = <Point>[];
        final List<Offset> centers = await _computeIrisCentersWithFrameId(
          frameId,
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

        return _DetectionFeatures(
          detection: detection,
          alignedFace: aligned,
          mesh: mesh,
          iris: iris,
        );
      }),
    );

    for (final r in results) {
      featuresOut?.add(r);
    }
    return results.map((r) => r.detection).toList();
  }

  /// Estimate aligned face using frame ID.
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

  /// Compute iris centers using frame ID.
  Future<List<Offset>> _computeIrisCentersWithFrameId(
    int frameId,
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
      final List<Offset> pts =
          lm.map((p) => Offset(p[0].toDouble(), p[1].toDouble())).toList();
      int bestIdx = 0;
      double bestScore = double.infinity;
      for (int k = 0; k < pts.length; k++) {
        double s = 0;
        for (int j = 0; j < pts.length; j++) {
          if (j == k) continue;
          final double dx = pts[j].dx - pts[k].dx;
          final double dy = pts[j].dy - pts[k].dy;
          s += dx * dx + dy * dy;
        }
        if (s < bestScore) {
          bestScore = s;
          bestIdx = k;
        }
      }
      return pts[bestIdx];
    }

    // Run left and right eye iris detection in parallel using separate models
    final results = await Future.wait([
      // Left eye (index 0)
      if (rois.isNotEmpty && rois[0].size > 0)
        _withIrisLeftLock(() => irisLeft.runOnImageAlignedIrisWithFrameId(
              frameId,
              rois[0],
              isRight: false,
              worker: _worker,
            ))
      else
        Future.value(<List<double>>[]),
      // Right eye (index 1)
      if (rois.length > 1 && rois[1].size > 0)
        _withIrisRightLock(() => irisRight.runOnImageAlignedIrisWithFrameId(
              frameId,
              rois[1],
              isRight: true,
              worker: _worker,
            ))
      else
        Future.value(<List<double>>[]),
    ]);

    final List<List<double>> leftLm = results[0];
    final List<List<double>> rightLm = results[1];

    // Build centers list and allLandmarks
    final List<Offset> centers = <Offset>[];
    bool usedFallback = false;

    // Process left eye
    if (rois.isEmpty || rois[0].size <= 0) {
      centers.add(leftFallback ?? const Offset(0, 0));
      usedFallback = true;
      irisUsedFallbackCount++;
    } else {
      if (allLandmarks != null) {
        allLandmarks.addAll(
          leftLm.map((p) => Point(
                p[0].toDouble(),
                p[1].toDouble(),
                p.length > 2 ? p[2].toDouble() : 0.0,
              )),
        );
      }
      centers.add(pickCenter(leftLm, leftFallback));
    }

    // Process right eye
    if (rois.length <= 1 || rois[1].size <= 0) {
      centers.add(rightFallback ?? const Offset(0, 0));
      usedFallback = true;
      irisUsedFallbackCount++;
    } else {
      if (allLandmarks != null) {
        allLandmarks.addAll(
          rightLm.map((p) => Point(
                p[0].toDouble(),
                p[1].toDouble(),
                p.length > 2 ? p[2].toDouble() : 0.0,
              )),
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

  /// Get iris for face using frame ID.
  Future<List<Point>> _getIrisForFaceWithFrameId(
    int frameId,
    List<Point> meshPx,
    double imgW,
    double imgH,
  ) async {
    if (_irisLeft == null || _irisRight == null) return <Point>[];
    if (meshPx.isEmpty) return <Point>[];

    final List<AlignedRoi> rois = eyeRoisFromMesh(meshPx);
    return await irisFromEyeRoisWithFrameId(frameId, rois);
  }

  /// Get iris landmarks from eye ROIs using frame ID.
  Future<List<Point>> irisFromEyeRoisWithFrameId(
    int frameId,
    List<AlignedRoi> rois,
  ) async {
    if (_irisLeft == null || _irisRight == null) return <Point>[];
    if (rois.length < 2) return <Point>[];

    final IrisLandmark irisLeft = _irisLeft!;
    final IrisLandmark irisRight = _irisRight!;

    // Run left and right eye iris detection in parallel using separate models
    final results = await Future.wait([
      // Left eye (index 0)
      _withIrisLeftLock(() => irisLeft.runOnImageAlignedIrisWithFrameId(
            frameId,
            rois[0],
            isRight: false,
            worker: _worker,
          )),
      // Right eye (index 1)
      _withIrisRightLock(() => irisRight.runOnImageAlignedIrisWithFrameId(
            frameId,
            rois[1],
            isRight: true,
            worker: _worker,
          )),
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
        // No locking needed here - this is in a separate isolate with its own model instance
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
