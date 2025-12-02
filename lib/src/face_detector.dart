part of '../face_detection_tflite.dart';

class _DetectionFeatures {
  const _DetectionFeatures({
    required this.detection,
    required this.alignedFace,
    required this.meshAbs,
    required this.irisAbs,
  });

  final Detection detection;
  final AlignedFace alignedFace;
  final List<Offset> meshAbs;
  final List<Offset> irisAbs;
}

/// A complete face detection and analysis system using TensorFlow Lite models.
///
/// This class orchestrates three TensorFlow Lite models to provide comprehensive
/// facial analysis:
/// - Face detection with 6 keypoints (eyes, nose, mouth corners)
/// - 468-point face mesh for detailed facial geometry
/// - Iris landmark detection with 10 points per eye
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

  ImageProcessingWorker? _worker;
  FaceDetection? _detector;
  FaceLandmark? _faceLm;
  IrisLandmark? _iris;

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
  /// Intended to track cases where iris detection falls back to original
  /// eye keypoint positions. Note: currently not incremented in the codebase.
  int irisUsedFallbackCount = 0;

  /// Duration of the most recent iris landmark detection operation.
  ///
  /// Updated after each iris center computation in [_computeIrisCentersOnMainThread].
  /// Useful for profiling iris detection performance. Initialized to [Duration.zero].
  Duration lastIrisTime = Duration.zero;

  /// Returns true if all models are loaded and ready for inference.
  ///
  /// You must call [initialize] before this returns true.
  bool get isReady => _detector != null && _faceLm != null && _iris != null;

  /// Loads the face detection, face mesh, and iris landmark models and prepares the interpreters for inference.
  ///
  /// This must be called before running any detections.
  /// The [model] argument specifies which detection model variant to load
  /// (for example, `FaceDetectionModel.backCamera`).
  ///
  /// Optionally, you can pass [options] to configure interpreter settings
  /// such as the number of threads or delegate type.
  Future<void> initialize({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    InterpreterOptions? options,
  }) async {
    await _ensureTFLiteLoaded();
    try {
      // Create the image processing worker for optimized performance
      _worker = ImageProcessingWorker();
      await _worker!.initialize();

      _detector = await FaceDetection.create(
        model,
        options: options,
        useIsolate: true,
      );
      _faceLm = await FaceLandmark.create(options: options, useIsolate: true);
      _iris = await IrisLandmark.create(options: options, useIsolate: true);
    } catch (e) {
      _worker?.dispose();
      _detector?.dispose();
      _faceLm?.dispose();
      _iris?.dispose();
      _worker = null;
      _detector = null;
      _faceLm = null;
      _iris = null;
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

  /// Returns face detections with eye keypoints refined by iris center detection.
  ///
  /// This method performs face detection, generates the face mesh for the first
  /// detected face, and uses iris landmark detection to compute precise iris centers.
  /// These iris centers replace the original eye keypoints for improved accuracy.
  ///
  /// The iris center refinement process:
  /// 1. Detects faces and extracts the first detection
  /// 2. Generates 468-point face mesh to locate eye regions
  /// 3. Runs iris detection on both eyes
  /// 4. Replaces eye keypoints with computed iris centers
  /// 5. Falls back to original eye positions if iris detection fails
  ///
  /// The [imageBytes] parameter should contain encoded image data (JPEG, PNG, etc.).
  ///
  /// Returns a list of detections with iris-refined keypoints for the first face;
  /// other detected faces are included but without iris refinement. Returns an
  /// empty list if no faces are detected.
  ///
  /// **Performance:** This method is computationally intensive as it runs the full
  /// pipeline (detection + mesh + iris) for the first face only.
  ///
  /// Throws [StateError] if the iris model has not been initialized.
  ///
  /// See also:
  /// - [detectFaces] with [FaceDetectionMode.full] for multi-face iris tracking
  /// - [getDetections] for basic detection without iris refinement
  Future<List<Detection>> getDetectionsWithIrisCenters(
    Uint8List imageBytes,
  ) async {
    if (_iris == null) {
      throw StateError(
        'Iris model not initialized. Call initialize() before getDetectionsWithIrisCenters().',
      );
    }

    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final List<Detection> dets = await _detectDetectionsWithDecoded(decoded);
    if (dets.isEmpty) return dets;

    final Detection det = dets.first;
    final AlignedFace aligned = await estimateAlignedFace(decoded, det);
    final List<Offset> meshPts = await meshFromAlignedFace(
      aligned.faceCrop,
      aligned,
    );
    final List<AlignedRoi> rois = eyeRoisFromMesh(meshPts);

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

    final List<Offset> centers = await _computeIrisCentersWithDecoded(
      decoded,
      rois,
      leftFallback: lf,
      rightFallback: rf,
    );

    final List<double> kp = List<double>.from(det.keypointsXY);
    kp[FaceLandmarkType.leftEye.index * 2] = centers[0].dx / imgW;
    kp[FaceLandmarkType.leftEye.index * 2 + 1] = centers[0].dy / imgH;
    kp[FaceLandmarkType.rightEye.index * 2] = centers[1].dx / imgW;
    kp[FaceLandmarkType.rightEye.index * 2 + 1] = centers[1].dy / imgH;

    final Detection updatedFirst = Detection(
      boundingBox: det.boundingBox,
      score: det.score,
      keypointsXY: kp,
      imageSize: Size(imgW, imgH),
    );

    return [updatedFirst, ...dets.skip(1)];
  }

  Future<List<Offset>> _computeIrisCentersWithDecoded(
    img.Image decoded,
    List<AlignedRoi> rois, {
    Offset? leftFallback,
    Offset? rightFallback,
    List<Offset>? allLandmarks,
  }) async {
    final Stopwatch sw = Stopwatch()..start();
    final IrisLandmark iris = _iris!;
    final List<Offset> centers = <Offset>[];
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

    for (int i = 0; i < rois.length; i++) {
      final bool isRight = i == 1;
      final List<List<double>> lm = await iris.runOnImageAlignedIris(
        decoded,
        rois[i],
        isRight: isRight,
      );
      if (allLandmarks != null) {
        allLandmarks.addAll(
          lm.map((p) => Offset(p[0].toDouble(), p[1].toDouble())),
        );
      }
      final Offset? fb = i == 0 ? leftFallback : rightFallback;
      centers.add(pickCenter(lm, fb));
    }

    sw.stop();
    lastIrisTime = sw.elapsed;

    if (centers.isNotEmpty) {
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
    // Decode once and delegate to optimized path
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
    if (_iris == null) {
      throw StateError(
        'Iris model not initialized. initialize() must succeed before detectDetections().',
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

    final List<Detection> updated = <Detection>[];
    for (final det in dets) {
      final AlignedFace aligned = await estimateAlignedFace(decoded, det);
      final List<Offset> meshPts = await meshFromAlignedFace(
        aligned.faceCrop,
        aligned,
      );
      final List<AlignedRoi> rois = eyeRoisFromMesh(meshPts);

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

      final List<Offset> irisLandmarks = <Offset>[];
      final List<Offset> centers = await _computeIrisCentersWithDecoded(
        decoded,
        rois,
        leftFallback: lf,
        rightFallback: rf,
        allLandmarks: irisLandmarks,
      );

      final List<double> kp = List<double>.from(det.keypointsXY);
      kp[FaceLandmarkType.leftEye.index * 2] = centers[0].dx / imgW;
      kp[FaceLandmarkType.leftEye.index * 2 + 1] = centers[0].dy / imgH;
      kp[FaceLandmarkType.rightEye.index * 2] = centers[1].dx / imgW;
      kp[FaceLandmarkType.rightEye.index * 2 + 1] = centers[1].dy / imgH;

      updated.add(
        Detection(
          boundingBox: det.boundingBox,
          score: det.score,
          keypointsXY: kp,
          imageSize: Size(imgW, imgH),
        ),
      );
      if (featuresOut != null) {
        featuresOut.add(
          _DetectionFeatures(
            detection: updated.last,
            alignedFace: aligned,
            meshAbs: meshPts,
            irisAbs: irisLandmarks,
          ),
        );
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
  /// Returns a list of 468 [Offset] points in absolute pixel coordinates
  /// relative to the original decoded image. Returns an empty list if the
  /// face landmark model is not initialized.
  ///
  /// Each point represents a specific facial feature as defined by MediaPipe's
  /// canonical face mesh topology (eyes, eyebrows, nose, mouth, face contours).
  Future<List<Offset>> meshFromAlignedFace(
    img.Image faceCrop,
    AlignedFace aligned,
  ) async {
    final fl = _faceLm;
    if (fl == null) return const <Offset>[];
    final lmNorm = await fl.call(faceCrop);
    final ct = math.cos(aligned.theta);
    final st = math.sin(aligned.theta);
    final s = aligned.size;
    final cx = aligned.cx;
    final cy = aligned.cy;
    final out = <Offset>[];
    for (final p in lmNorm) {
      final lx2 = (p[0] - 0.5) * s;
      final ly2 = (p[1] - 0.5) * s;
      final x = cx + lx2 * ct - ly2 * st;
      final y = cy + lx2 * st + ly2 * ct;
      out.add(Offset(x.toDouble(), y.toDouble()));
    }
    return out;
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
  List<AlignedRoi> eyeRoisFromMesh(List<Offset> meshAbs) {
    AlignedRoi fromCorners(int a, int b) {
      final Offset p0 = meshAbs[a];
      final Offset p1 = meshAbs[b];
      final double cx = (p0.dx + p1.dx) * 0.5;
      final double cy = (p0.dy + p1.dy) * 0.5;
      final double dx = p1.dx - p0.dx;
      final double dy = p1.dy - p0.dy;
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
  /// and returns all iris keypoints in absolute pixel coordinates. Each eye
  /// produces 5 iris keypoints (center + 4 contour points).
  ///
  /// The detection process:
  /// 1. For each eye ROI in [rois], extracts an aligned eye crop from [decoded]
  /// 2. Runs iris landmark detection (right eyes are horizontally flipped first)
  /// 3. Transforms the normalized iris coordinates back to absolute pixels
  /// 4. Concatenates results from both eyes
  ///
  /// The [decoded] parameter is the source image containing the face.
  ///
  /// The [rois] parameter is a list of aligned eye regions, typically from
  /// [eyeRoisFromMesh]. The first ROI should be the left eye, the second
  /// should be the right eye.
  ///
  /// Returns a list of [Offset] points in absolute pixel coordinates containing
  /// iris landmarks for all eyes (typically 10 points total: 5 per eye).
  /// Returns an empty list if the iris model is not initialized.
  ///
  /// The returned points from each eye include:
  /// - Iris center (typically the most stable point)
  /// - 4 iris contour points
  Future<List<Offset>> irisFromEyeRois(
    img.Image decoded,
    List<AlignedRoi> rois,
  ) async {
    final IrisLandmark? ir = _iris;
    if (ir == null) return const <Offset>[];

    final List<Offset> pts = <Offset>[];
    for (int i = 0; i < rois.length; i++) {
      final bool isRight = (i == 1);
      final List<List<double>> irisLm = await ir.runOnImageAlignedIris(
        decoded,
        rois[i],
        isRight: isRight,
      );
      for (final List<double> p in irisLm) {
        pts.add(Offset(p[0].toDouble(), p[1].toDouble()));
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
  Future<List<math.Point<double>>> getFaceMesh(Uint8List imageBytes) async {
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final List<Detection> dets = await _detectDetectionsWithDecoded(decoded);
    if (dets.isEmpty) return const <math.Point<double>>[];

    final AlignedFace aligned = await estimateAlignedFace(decoded, dets.first);
    final List<Offset> meshAbs = await meshFromAlignedFace(
      aligned.faceCrop,
      aligned,
    );
    return meshAbs
        .map((p) => math.Point<double>(p.dx, p.dy))
        .toList(growable: false);
  }

  /// Detects iris landmarks for the first detected face in a full image.
  ///
  /// The [imageBytes] should contain encoded image data (e.g., JPEG/PNG).
  /// Returns up to 10 points (5 per eye: center + 4 contour points) in absolute pixels.
  /// If no face or iris is found, returns an empty list.
  Future<List<math.Point<double>>> getIris(Uint8List imageBytes) async {
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);

    final List<Detection> dets = await _detectDetectionsWithDecoded(decoded);
    if (dets.isEmpty) return const <math.Point<double>>[];
    final AlignedFace aligned = await estimateAlignedFace(decoded, dets.first);
    final List<Offset> meshAbs = await meshFromAlignedFace(
      aligned.faceCrop,
      aligned,
    );
    final List<AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
    final List<Offset> irisAbs = await irisFromEyeRois(decoded, rois);
    return irisAbs
        .map((p) => math.Point<double>(p.dx, p.dy))
        .toList(growable: false);
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
  Future<List<List<math.Point<double>>>> getFaceMeshFromDetections(
    Uint8List imageBytes,
    List<Detection> dets,
  ) async {
    if (dets.isEmpty) return const <List<math.Point<double>>>[];
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final List<List<math.Point<double>>> out = <List<math.Point<double>>>[];
    for (final Detection det in dets) {
      final AlignedFace aligned = await estimateAlignedFace(decoded, det);
      final List<Offset> meshAbs = await meshFromAlignedFace(
        aligned.faceCrop,
        aligned,
      );
      out.add(
        meshAbs
            .map((p) => math.Point<double>(p.dx, p.dy))
            .toList(growable: false),
      );
    }
    return out;
  }

  /// Generates iris landmarks from face mesh data.
  ///
  /// This method takes pre-computed 468-point face meshes and extracts iris
  /// landmarks for each face. The face mesh is used to locate eye regions,
  /// which are then processed by the iris landmark model to detect 5 keypoints
  /// per iris (10 points total per face).
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
  /// // irises[0] contains 10 iris points (5 per eye) for the first face
  /// ```
  ///
  /// See also:
  /// - [getIris] for single-face iris detection
  /// - [getFaceMeshFromDetections] to generate the required mesh input
  /// - [detectFaces] for the complete pipeline in one call
  Future<List<List<math.Point<double>>>> getIrisFromMesh(
    Uint8List imageBytes,
    List<List<math.Point<double>>> meshesPerFace,
  ) async {
    if (meshesPerFace.isEmpty) return const <List<math.Point<double>>>[];
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final out = <List<math.Point<double>>>[];
    for (final meshPts in meshesPerFace) {
      if (meshPts.isEmpty) {
        out.add(const <math.Point<double>>[]);
        continue;
      }
      final List<Offset> meshAbs =
          meshPts.map((p) => Offset(p.x, p.y)).toList(growable: false);
      final List<AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
      final List<Offset> irisAbs = await irisFromEyeRois(decoded, rois);
      out.add(
        irisAbs
            .map((p) => math.Point<double>(p.dx, p.dy))
            .toList(growable: false),
      );
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
  List<List<math.Point<double>>> splitMeshesIfConcatenated(
    List<math.Point<double>> meshPts,
  ) {
    if (meshPts.isEmpty) return const <List<math.Point<double>>>[];
    if (meshPts.length % kMeshPoints != 0) return [meshPts];
    final int faces = meshPts.length ~/ kMeshPoints;
    final List<List<math.Point<double>>> out = <List<math.Point<double>>>[];
    for (int i = 0; i < faces; i++) {
      final int start = i * kMeshPoints;
      out.add(meshPts.sublist(start, start + kMeshPoints));
    }
    return out;
  }

  Future<List<math.Point<double>>> _getMeshForFace(AlignedFace aligned) async {
    final meshAbs = await meshFromAlignedFace(aligned.faceCrop, aligned);
    return meshAbs
        .map((p) => math.Point<double>(p.dx, p.dy))
        .toList(growable: false);
  }

  Future<List<math.Point<double>>> _getIrisForFace(
    img.Image decoded,
    List<math.Point<double>> meshPx,
  ) async {
    if (meshPx.isEmpty) return <math.Point<double>>[];

    final List<Offset> meshAbs =
        meshPx.map((p) => Offset(p.x, p.y)).toList(growable: false);
    final List<AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
    final List<Offset> irisAbs = await irisFromEyeRois(decoded, rois);
    return irisAbs
        .map((p) => math.Point<double>(p.dx, p.dy))
        .toList(growable: false);
  }

  /// Detects faces in the provided image and returns detailed results.
  ///
  /// The [imageBytes] parameter should contain encoded image data (JPEG, PNG, etc.).
  /// The [mode] parameter controls which features are computed:
  /// - [FaceDetectionMode.fast]: Only detection and landmarks
  /// - [FaceDetectionMode.standard]: Adds 468-point face mesh
  /// - [FaceDetectionMode.full]: Adds iris tracking (10 points)
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
    // Decode once using worker
    final DecodedRgb d = await decodeImageWithWorker(imageBytes, _worker);
    final img.Image decoded = _imageFromDecodedRgb(d);
    final Size imgSize = Size(
      decoded.width.toDouble(),
      decoded.height.toDouble(),
    );

    final bool computeIris = mode == FaceDetectionMode.full;
    final List<_DetectionFeatures> features =
        computeIris ? <_DetectionFeatures>[] : const <_DetectionFeatures>[];
    // Use decoded image to avoid redundant decoding
    final List<Detection> dets = await _detectDetectionsWithDecoded(
      decoded,
      refineEyesWithIris: computeIris,
      featuresOut: computeIris ? features : null,
    );

    final bool computeMesh =
        mode == FaceDetectionMode.standard || mode == FaceDetectionMode.full;
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

        final List<math.Point<double>> meshPx =
            computeMesh && feat != null && feat.meshAbs.isNotEmpty
                ? feat.meshAbs
                    .map((p) => math.Point<double>(p.dx, p.dy))
                    .toList(growable: false)
                : computeMesh && i < allAligned.length
                    ? await _getMeshForFace(allAligned[i])
                    : <math.Point<double>>[];

        final List<math.Point<double>> irisPx =
            computeIris && feat != null && feat.irisAbs.isNotEmpty
                ? feat.irisAbs
                    .map((p) => math.Point<double>(p.dx, p.dy))
                    .toList(growable: false)
                : computeIris && i < allAligned.length
                    ? await _getIrisForFace(decoded, meshPx)
                    : computeIris
                        ? await _getIrisForFace(decoded, meshPx)
                        : <math.Point<double>>[];

        faces.add(
          Face(
            detection: det,
            mesh: meshPx,
            irises: irisPx,
            originalSize: imgSize,
          ),
        );
      } catch (e) {
        // Skip this face if processing fails, continue with remaining faces
      }
    }

    return faces;
  }

  /// Releases all resources held by the detector.
  ///
  /// Call this when you're done using the detector to free up memory.
  /// After calling dispose, you must call [initialize] again before
  /// running any detections.
  void dispose() {
    _worker?.dispose();
    _detector?.dispose();
    _faceLm?.dispose();
    _iris?.dispose();
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
      final IrisLandmark iris = await IrisLandmark.create(useIsolate: false);
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
