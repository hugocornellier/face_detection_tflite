part of face_detection_tflite;

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

  FaceDetection? _detector;
  FaceLandmark? _faceLm;
  IrisLandmark? _iris;
  int irisOkCount = 0;
  int irisFailCount = 0;
  int irisUsedFallbackCount = 0;
  Duration lastIrisTime = Duration.zero;

  /// Returns true if all models are loaded and ready for inference.
  ///
  /// You must call [initialize] before this returns true.
  bool get isReady => _detector != null && _faceLm != null && _iris != null;

  /// Loads the face detection model and prepares the interpreter for inference.
  ///
  /// This must be called before running any detections.
  /// The [model] argument specifies which model variant to load
  /// (for example, `FaceDetectionModel.backCamera`).
  ///
  /// Optionally, you can pass [options] to configure interpreter settings
  /// such as the number of threads or delegate type.
  Future<void> initialize({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    InterpreterOptions? options}
  ) async {
    await _ensureTFLiteLoaded();
    try {
      _detector = await FaceDetection.create(model, options: options, useIsolate: true);
      _faceLm = await FaceLandmark.create(options: options, useIsolate: true);
      _iris = await IrisLandmark.create(options: options, useIsolate: true);
    } catch (e) {
      _detector?.dispose();
      _faceLm?.dispose();
      _iris?.dispose();
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
      hint =
      'Make sure your Windows plugin CMakeLists.txt sets:\n'
          '  set(PLUGIN_NAME_bundled_libraries ".../libtensorflowlite_c-win.dll" PARENT_SCOPE)\n'
          'so Flutter copies it next to the app EXE.';
    } else if (Platform.isLinux) {
      candidates = [
        p.join(exeDir.path, 'lib', 'libtensorflowlite_c-linux.so'),
        'libtensorflowlite_c-linux.so',
      ];
      hint =
      'Ensure linux/CMakeLists.txt sets:\n'
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

  Future<List<_Detection>> getDetectionsWithIrisCenters(Uint8List imageBytes) async {
    if (_iris == null) {
      throw StateError('Iris model not initialized. Call initialize() before getDetectionsWithIrisCenters().');
    }

    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final List<_Detection> dets = await _detectDetections(imageBytes);
    if (dets.isEmpty) return dets;

    final _Detection det = dets.first;
    final _AlignedFace aligned = await estimateAlignedFace(decoded, det);
    final List<Offset> meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
    final List<_AlignedRoi> rois = eyeRoisFromMesh(meshPts);

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

    final List<Offset> centers = await _computeIrisCentersOnMainThread(
      imageBytes, rois,
      leftFallback: lf,
      rightFallback: rf,
    );

    final List<double> kp = List<double>.from(det.keypointsXY);
    kp[FaceLandmarkType.leftEye.index * 2]     = centers[0].dx / imgW;
    kp[FaceLandmarkType.leftEye.index * 2 + 1] = centers[0].dy / imgH;
    kp[FaceLandmarkType.rightEye.index * 2]    = centers[1].dx / imgW;
    kp[FaceLandmarkType.rightEye.index * 2 + 1]= centers[1].dy / imgH;

    final _Detection updatedFirst = _Detection(
      bbox: det.bbox,
      score: det.score,
      keypointsXY: kp,
      imageSize: Size(imgW, imgH),
    );

    return [updatedFirst, ...dets.skip(1)];
  }

  Future<List<Offset>> _computeIrisCentersOnMainThread(
    Uint8List imageBytes,
    List<_AlignedRoi> rois, {
    Offset? leftFallback,
    Offset? rightFallback,
  }) async {
    final Stopwatch sw = Stopwatch()..start();
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final IrisLandmark iris = _iris!;
    final List<Offset> centers = <Offset>[];

    Offset _pickCenter(List<List<double>> lm, Offset? fallback) {
      if (lm.isEmpty) return fallback ?? const Offset(0, 0);
      final List<Offset> pts = lm.map((p) => Offset(p[0].toDouble(), p[1].toDouble())).toList();
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
        isRight: isRight
      );
      final Offset? fb = i == 0 ? leftFallback : rightFallback;
      centers.add(_pickCenter(lm, fb));
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

  Future<List<_Detection>> _detectDetections(
    Uint8List imageBytes, {
    _RectF? roi,
    bool refineEyesWithIris = true
  }) async {
    final FaceDetection? d = _detector;
    if (d == null) {
      throw StateError('FaceDetector not initialized. Call initialize() before detectDetections().');
    }
    if (_iris == null) {
      throw StateError('Iris model not initialized. initialize() must succeed before detectDetections().');
    }

    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final List<_Detection> dets = await d.call(imageBytes, roi: roi);
    if (dets.isEmpty) return dets;

    if (!refineEyesWithIris) {
      final double imgW = decoded.width.toDouble();
      final double imgH = decoded.height.toDouble();
      return dets.map((det) => _Detection(
        bbox: det.bbox,
        score: det.score,
        keypointsXY: det.keypointsXY,
        imageSize: Size(imgW, imgH),
      )).toList();
    }

    final List<_Detection> updated = <_Detection>[];
    for (final det in dets) {
      final _AlignedFace aligned = await estimateAlignedFace(decoded, det);
      final List<Offset> meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
      final List<_AlignedRoi> rois = eyeRoisFromMesh(meshPts);

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

      final List<Offset> centers = await _computeIrisCentersOnMainThread(
        imageBytes, rois,
        leftFallback: lf,
        rightFallback: rf,
      );

      final List<double> kp = List<double>.from(det.keypointsXY);
      kp[FaceLandmarkType.leftEye.index * 2]     = centers[0].dx / imgW;
      kp[FaceLandmarkType.leftEye.index * 2 + 1] = centers[0].dy / imgH;
      kp[FaceLandmarkType.rightEye.index * 2]    = centers[1].dx / imgW;
      kp[FaceLandmarkType.rightEye.index * 2 + 1]= centers[1].dy / imgH;

      updated.add(_Detection(
        bbox: det.bbox,
        score: det.score,
        keypointsXY: kp,
        imageSize: Size(imgW, imgH),
      ));
    }
    return updated;
  }

  Future<_AlignedFace> estimateAlignedFace(img.Image decoded, _Detection det) async {
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

    final faceCrop = await extractAlignedSquare(decoded, cx, cy, size, -theta);

    return _AlignedFace(cx: cx, cy: cy, size: size, theta: theta, faceCrop: faceCrop);
  }

  Future<List<Offset>> meshFromAlignedFace(img.Image faceCrop, _AlignedFace aligned) async {
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

  List<_AlignedRoi> eyeRoisFromMesh(List<Offset> meshAbs) {
    _AlignedRoi fromCorners(int a, int b) {
      final Offset p0 = meshAbs[a];
      final Offset p1 = meshAbs[b];
      final double cx = (p0.dx + p1.dx) * 0.5;
      final double cy = (p0.dy + p1.dy) * 0.5;
      final double dx = p1.dx - p0.dx;
      final double dy = p1.dy - p0.dy;
      final double eyeDist = math.sqrt(dx * dx + dy * dy);
      final double size = eyeDist * 2.3;
      return _AlignedRoi(cx, cy, size, math.atan2(dy, dx));
    }
    final _AlignedRoi left = fromCorners(33, 133);
    final _AlignedRoi right = fromCorners(362, 263);
    return [left, right];
  }

  Future<List<Offset>> irisFromEyeRois(
    img.Image decoded,
    List<_AlignedRoi> rois
  ) async {
    final IrisLandmark? ir = _iris;
    if (ir == null) return const <Offset>[];

    final List<Offset> pts = <Offset>[];
    for (int i = 0; i < rois.length; i++) {
      final bool isRight = (i == 1);
      final List<List<double>> irisLm = await ir.runOnImageAlignedIris(
        decoded,
        rois[i],
        isRight: isRight
      );
      for (final List<double> p in irisLm) {
        pts.add(Offset(p[0].toDouble(), p[1].toDouble()));
      }
    }
    return pts;
  }

  Future<List<_Detection>> getDetections(Uint8List imageBytes) async {
    return await _detectDetections(imageBytes);
  }

  /// Predicts 468 facial landmarks for the given image region.
  ///
  /// The input [imageBytes] is an encoded image (e.g., JPEG/PNG).
  /// Returns an empty list when no face is found.
  Future<List<math.Point<double>>> getFaceMesh(Uint8List imageBytes) async {
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final List<_Detection> dets = await _detectDetections(imageBytes);
    if (dets.isEmpty) return const <math.Point<double>>[];

    final _AlignedFace aligned = await estimateAlignedFace(decoded, dets.first);
    final List<Offset> meshAbs = await meshFromAlignedFace(aligned.faceCrop, aligned);
    return meshAbs.map((p) => math.Point<double>(p.dx, p.dy)).toList(growable: false);
  }

  /// Detects iris centers within a cropped eye region.
  ///
  /// The [image] should correspond to the eye crop from a previous face mesh.
  /// Returns a list of two points representing the left and right iris centers.
  /// If detection fails, it falls back to estimated centers from mesh landmarks.
  Future<List<math.Point<double>>> getIris(Uint8List imageBytes) async {
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

    final List<_Detection> dets = await _detectDetections(imageBytes);
    if (dets.isEmpty) return const <math.Point<double>>[];
    final _AlignedFace aligned = await estimateAlignedFace(decoded, dets.first);
    final List<Offset> meshAbs = await meshFromAlignedFace(aligned.faceCrop, aligned);
    final List<_AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
    final List<Offset> irisAbs = await irisFromEyeRois(decoded, rois);
    return irisAbs.map((p) => math.Point<double>(p.dx, p.dy)).toList(growable: false);
  }

  Future<Size> getOriginalSize(Uint8List imageBytes) async {
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

    return Size(decoded.width.toDouble(), decoded.height.toDouble());
  }

  Future<List<List<math.Point<double>>>> getFaceMeshFromDetections(
    Uint8List imageBytes,
    List<_Detection> dets
  ) async {
    if (dets.isEmpty) return const <List<math.Point<double>>>[];
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final List<List<math.Point<double>>> out = <List<math.Point<double>>>[];
    for (final _Detection det in dets) {
      final _AlignedFace aligned = await estimateAlignedFace(decoded, det);
      final List<Offset> meshAbs = await meshFromAlignedFace(
        aligned.faceCrop,
        aligned
      );
      out.add(meshAbs.map((p) => math.Point<double>(p.dx, p.dy)).toList(growable: false));
    }
    return out;
  }

  Future<List<List<math.Point<double>>>> getIrisFromMesh(
    Uint8List imageBytes,
    List<List<math.Point<double>>> meshesPerFace
  ) async {
    if (meshesPerFace.isEmpty) return const <List<math.Point<double>>>[];
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final out = <List<math.Point<double>>>[];
    for (final meshPts in meshesPerFace) {
      if (meshPts.isEmpty) {
        out.add(const <math.Point<double>>[]);
        continue;
      }
      final List<Offset> meshAbs = meshPts.map((p) => Offset(p.x, p.y)).toList(growable: false);
      final List<_AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
      final List<Offset> irisAbs = await irisFromEyeRois(decoded, rois);
      out.add(irisAbs.map((p) => math.Point<double>(p.dx, p.dy)).toList(growable: false));
    }
    return out;
  }

  List<List<math.Point<double>>> splitMeshesIfConcatenated(
    List<math.Point<double>> meshPts
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

  Future<List<math.Point<double>>> _getMeshForFace(_AlignedFace aligned) async {
    final meshAbs = await meshFromAlignedFace(aligned.faceCrop, aligned);
    return meshAbs.map((p) => math.Point<double>(p.dx, p.dy)).toList(growable: false);
  }

  Future<List<math.Point<double>>> _getIrisForFace(
    img.Image decoded,
    List<math.Point<double>> meshPx,
  ) async {
    if (meshPx.isEmpty) return <math.Point<double>>[];

    final List<Offset> meshAbs = meshPx.map((p) => Offset(p.x, p.y)).toList(growable: false);
    final List<_AlignedRoi> rois = eyeRoisFromMesh(meshAbs);
    final List<Offset> irisAbs = await irisFromEyeRois(decoded, rois);
    return irisAbs.map((p) => math.Point<double>(p.dx, p.dy)).toList(growable: false);
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
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final Size imgSize = Size(
      decoded.width.toDouble(),
      decoded.height.toDouble()
    );

    final bool computeIris = mode == FaceDetectionMode.full;
    final List<_Detection> dets = await _detectDetections(
        imageBytes,
        refineEyesWithIris: computeIris
    );

    final bool computeMesh = mode == FaceDetectionMode.standard || mode == FaceDetectionMode.full;
    final List<_AlignedFace> allAligned = computeMesh
        ? await Future.wait(dets.map((det) => estimateAlignedFace(decoded, det)))
        : <_AlignedFace>[];

    final List<Face> faces = <Face>[];
    for (int i = 0; i < dets.length; i++) {
      try {
        final _Detection det = dets[i];

        final List<math.Point<double>> meshPx = computeMesh && i < allAligned.length
            ? await _getMeshForFace(allAligned[i])
            : <math.Point<double>>[];

        final List<math.Point<double>> irisPx = computeIris && i < allAligned.length
            ? await _getIrisForFace(decoded, meshPx)
            : <math.Point<double>>[];

        faces.add(Face(
          detection: det,
          mesh: meshPx,
          irises: irisPx,
          originalSize: imgSize,
        ));
      } catch (e) {
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
        final _AlignedRoi roi = _AlignedRoi(
          (m['cx'] as num).toDouble(),
          (m['cy'] as num).toDouble(),
          (m['size'] as num).toDouble(),
          (m['theta'] as num).toDouble(),
        );
        final List<List<double>> lm = await iris.runOnImageAlignedIris(
          decoded,
          roi,
          isRight: i == 1
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