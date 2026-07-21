part of '../native/face_native_lib.dart';

/// Data passed to the detection isolate during startup.
class _DetectionIsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData faceDetectionBytes;
  final TransferableTypedData faceLandmarkBytes;
  final TransferableTypedData irisLandmarkBytes;
  final TransferableTypedData faceBlendshapesBytes;
  final TransferableTypedData embeddingBytes;
  final String modelName;
  final String performanceModeName;
  final int? numThreads;
  final int meshPoolSize;
  final bool useCompiledModel;
  final List<int> acceleratorIndices;
  final int precisionIndex;
  final double minScore;
  final double minFaceSize;
  final double minFacePresenceConfidence;

  _DetectionIsolateStartupData({
    required this.sendPort,
    required this.faceDetectionBytes,
    required this.faceLandmarkBytes,
    required this.irisLandmarkBytes,
    required this.faceBlendshapesBytes,
    required this.embeddingBytes,
    required this.modelName,
    required this.performanceModeName,
    required this.numThreads,
    required this.meshPoolSize,
    required this.useCompiledModel,
    required this.acceleratorIndices,
    required this.precisionIndex,
    required this.minScore,
    required this.minFaceSize,
    required this.minFacePresenceConfidence,
  });
}

/// Data passed to the segmentation isolate during startup.
class _SegmentationIsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData modelBytes;
  final String performanceModeName;
  final int? numThreads;
  final int maxOutputSize;
  final bool validateModel;
  final String modelName;
  final int modelIndex;
  final bool useCompiledModel;
  final List<int> acceleratorIndices;
  final int precisionIndex;

  _SegmentationIsolateStartupData({
    required this.sendPort,
    required this.modelBytes,
    required this.performanceModeName,
    required this.numThreads,
    required this.maxOutputSize,
    required this.validateModel,
    required this.modelName,
    required this.modelIndex,
    required this.useCompiledModel,
    required this.acceleratorIndices,
    required this.precisionIndex,
  });
}

/// Direct-mode TFLite inference core used inside the detection background isolate.
///
/// This class holds all TFLite interpreters and runs face detection entirely
/// on the calling thread (no further isolate spawning). It is created inside
/// [FaceDetector]'s background isolate by [FaceDetector._detectionIsolateEntry].
class _FaceDetectorCore {
  FaceDetection? _detector;
  RoundRobinPool<FaceLandmark>? _meshPool;
  List<FaceLandmark> _meshItems = [];
  IrisLandmark? _irisLeft;
  IrisLandmark? _irisRight;
  FaceBlendshapesModel? _blendshapes;
  FaceEmbedding? _embedding;

  /// Detection gates, applied right after the detector stage so gated-out
  /// detections skip the per-face mesh/iris/blendshape stages. Configured
  /// once at initialization; 0.0 means no filtering.
  double _minScore = 0.0;
  double _minFaceSize = 0.0;

  /// Face-presence gate (MediaPipe `min_face_presence_confidence`), applied
  /// right after the mesh stage so faces the landmark model does not confirm
  /// skip the iris/blendshape stages and are not emitted. Configured once at
  /// initialization; 0.0 means no filtering.
  double _minFacePresenceConfidence = 0.0;

  /// Whether [meshScore] clears the configured [_minFacePresenceConfidence]
  /// gate. A null score (fast mode, or a mesh model with no presence output)
  /// always passes: absence of a presence score is "cannot evaluate", never a
  /// rejection.
  bool _passesPresence(double? meshScore) =>
      _minFacePresenceConfidence <= 0.0 ||
      (meshScore ?? double.infinity) >= _minFacePresenceConfidence;

  final _detectorLock = AsyncLock();
  final _irisLeftLock = AsyncLock();
  final _irisRightLock = AsyncLock();
  final _blendshapesLock = AsyncLock();
  final _embeddingLock = AsyncLock();

  /// Returns true when the core has been initialized with model data.
  bool get isReady => _detector != null;

  /// Returns true when the embedding model is loaded and ready.
  bool get isEmbeddingReady => _embedding != null;

  /// Initializes all TFLite models from pre-loaded bytes.
  Future<void> initializeFromBuffers({
    required Uint8List faceDetectionBytes,
    required Uint8List faceLandmarkBytes,
    required Uint8List irisLandmarkBytes,
    Uint8List? faceBlendshapesBytes,
    Uint8List? embeddingBytes,
    required FaceDetectionModel model,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
    bool useCompiledModel = false,
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
    double minScore = 0.0,
    double minFaceSize = 0.0,
    double minFacePresenceConfidence = 0.0,
  }) async {
    _minScore = minScore;
    _minFaceSize = minFaceSize;
    _minFacePresenceConfidence = minFacePresenceConfidence;
    try {
      _detector = useCompiledModel
          ? await FaceDetection.createCompiledFromBuffer(
              faceDetectionBytes,
              model,
              accelerators: accelerators,
              precision: precision,
            )
          : await FaceDetection.createFromBuffer(
              faceDetectionBytes,
              model,
              performanceConfig: performanceConfig,
            );

      _meshItems = [];
      for (int i = 0; i < meshPoolSize; i++) {
        _meshItems.add(
          useCompiledModel
              ? await FaceLandmark.createCompiledFromBuffer(
                  faceLandmarkBytes,
                  accelerators: accelerators,
                  precision: precision,
                )
              : await FaceLandmark.createFromBuffer(
                  faceLandmarkBytes,
                  performanceConfig: performanceConfig,
                ),
        );
      }
      _meshPool = RoundRobinPool(_meshItems);

      // Iris landmark is intentionally pinned to CPU regardless of user
      // preference: the 64x64 model is below the compute size where GPU
      // dispatch pays off (CPU runs ~0.50 ms vs ~0.68 ms GPU|CPU on macOS
      // arm64). The accelerators/precision params are NOT forwarded here.
      _irisLeft = useCompiledModel
          ? await IrisLandmark.createCompiledFromBuffer(irisLandmarkBytes)
          : await IrisLandmark.createFromBuffer(
              irisLandmarkBytes,
              performanceConfig: performanceConfig,
            );
      _irisRight = useCompiledModel
          ? await IrisLandmark.createCompiledFromBuffer(irisLandmarkBytes)
          : await IrisLandmark.createFromBuffer(
              irisLandmarkBytes,
              performanceConfig: performanceConfig,
            );

      // Blendshape classification (smile / eye-open). Like iris, this is a
      // small model pinned to the CPU regardless of user preference: a
      // 292-in / 52-out MLP is well below the GPU-dispatch payoff threshold.
      if (faceBlendshapesBytes != null) {
        _blendshapes = useCompiledModel
            ? await FaceBlendshapesModel.createCompiledFromBuffer(
                faceBlendshapesBytes,
              )
            : await FaceBlendshapesModel.createFromBuffer(faceBlendshapesBytes);
      }

      if (embeddingBytes != null) {
        _embedding = useCompiledModel
            ? await FaceEmbedding.createCompiledFromBuffer(
                embeddingBytes,
                accelerators: accelerators,
                precision: precision,
              )
            : await FaceEmbedding.createFromBuffer(
                embeddingBytes,
                performanceConfig: performanceConfig,
              );
      }
    } catch (e) {
      _cleanupOnInitError();
      rethrow;
    }
  }

  /// Runs face detection directly on the calling thread.
  Future<List<Face>> detectFacesDirect(
    cv.Mat image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    if (_detector == null) {
      throw StateError(
        'FaceDetectorCore not initialized. Call initializeFromBuffers() first.',
      );
    }

    final int width = image.cols;
    final int height = image.rows;
    final Size imgSize = Size(width.toDouble(), height.toDouble());

    final bool computeIris = mode == FaceDetectionMode.full;
    final bool computeMesh =
        mode == FaceDetectionMode.standard || mode == FaceDetectionMode.full;

    List<Detection> dets = await _detectDetections(image);
    if (dets.isEmpty) return <Face>[];

    // Early gate: drop detections failing minScore/minFaceSize before any
    // per-face work (alignment warp, mesh, iris, blendshapes). Runs after
    // NMS and letterbox removal, so results are identical to the late gate
    // on the main isolate; only the wasted per-face stages are skipped.
    dets = applyDetectionGates(
      dets,
      minScore: _minScore,
      minFaceSize: _minFaceSize,
      imageWidth: width.toDouble(),
    );
    if (dets.isEmpty) return <Face>[];

    final List<(Detection, AlignedFace?)?> alignedFaces =
        <(Detection, AlignedFace?)?>[];
    for (final Detection det in dets) {
      try {
        if (computeMesh) {
          final aligned = await _estimateAlignedFace(image, det);
          alignedFaces.add((det, aligned));
        } else {
          // Fast mode never uses the crop: compute only the alignment
          // geometry, keeping the same degenerate-size drop condition as
          // _estimateAlignedFace without paying for the warp.
          final (theta: _, cx: _, cy: _, :size) = _computeFaceAlignment(
            det,
            width.toDouble(),
            height.toDouble(),
          );
          alignedFaces.add(size.round() > 0 ? (det, null) : null);
        }
      } catch (_) {
        alignedFaces.add(null);
      }
    }

    final List<({List<Point> points, double? score})?> meshResults;
    if (computeMesh) {
      meshResults = await Future.wait(
        alignedFaces.map((data) async {
          final AlignedFace? aligned = data?.$2;
          if (aligned == null) return null;
          try {
            return await _meshFromAlignedFace(
              aligned.faceCrop,
              aligned.cx,
              aligned.cy,
              aligned.size,
              aligned.theta,
            );
          } catch (_) {
            return null;
          }
        }),
      );
    } else {
      meshResults = List<({List<Point> points, double? score})?>.filled(
        alignedFaces.length,
        null,
      );
    }

    for (final data in alignedFaces) {
      data?.$2?.faceCrop.dispose();
    }

    final List<List<Point>?> irisResults = List<List<Point>?>.filled(
      dets.length,
      null,
    );
    if (computeIris) {
      for (int i = 0; i < meshResults.length; i++) {
        final meshPx = meshResults[i]?.points;
        if (meshPx == null || meshPx.isEmpty) continue;
        // Face-presence gate: a crop the mesh model does not confirm as a face
        // is dropped here, before the iris (and blendshape) stages, so rejected
        // detections cost nothing beyond the mesh they were rejected by.
        if (!_passesPresence(meshResults[i]?.score)) continue;
        try {
          irisResults[i] = await _irisFromMesh(image, meshPx);
        } catch (_) {}
      }
    }

    // Blendshape classification: repack the mesh + iris points into the model's
    // 146-landmark input and run one sub-millisecond inference per face. Only
    // in full mode (iris present); failures leave that face's scores null.
    final List<Float32List?> blendshapeResults = List<Float32List?>.filled(
      dets.length,
      null,
    );
    if (computeIris && _blendshapes != null) {
      for (int i = 0; i < meshResults.length; i++) {
        final meshPx = meshResults[i]?.points;
        final irisPx = irisResults[i];
        if (meshPx == null || irisPx == null) continue;
        if (!_passesPresence(meshResults[i]?.score)) continue;
        final Float32List? packed = packBlendshapeInput(meshPx, irisPx);
        if (packed == null) continue;
        try {
          blendshapeResults[i] = await _blendshapesLock.run(
            () => _blendshapes!.call(packed),
          );
        } catch (_) {}
      }
    }

    final List<Face> faces = <Face>[];
    for (int i = 0; i < dets.length; i++) {
      final aligned = alignedFaces[i];
      if (aligned == null) continue;

      final Detection det = aligned.$1;
      final List<Point> meshPx = meshResults[i]?.points ?? <Point>[];
      final double? meshScore = meshResults[i]?.score;
      // Face-presence gate: drop faces the mesh model did not confirm so they
      // are never emitted (mirrors the iris/blendshape skips above). Fast mode
      // and presence-less mesh models report a null score and always pass.
      if (!_passesPresence(meshScore)) continue;
      final List<Point> irisPx = irisResults[i] ?? <Point>[];

      List<double> kp = det.keypointsXY;
      if (computeIris && irisPx.isNotEmpty) {
        kp = List<double>.from(det.keypointsXY);
        if (irisPx.length >= _kLeftIrisEnd) {
          final leftCenter = irisCenterFromPoints(
            irisPx.sublist(_kLeftIrisStart, _kLeftIrisEnd),
          );
          kp[FaceLandmarkType.leftEye.index * 2] = leftCenter.x / width;
          kp[FaceLandmarkType.leftEye.index * 2 + 1] = leftCenter.y / height;
        }
        if (irisPx.length >= _kRightIrisEnd) {
          final rightCenter = irisCenterFromPoints(
            irisPx.sublist(_kRightIrisStart, _kRightIrisEnd),
          );
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

      faces.add(
        Face(
          detection: refinedDet,
          mesh: meshPx.isNotEmpty ? FaceMesh(meshPx, score: meshScore) : null,
          irises: irisPx,
          blendshapeScores: blendshapeResults[i],
          originalSize: imgSize,
        ),
      );
    }

    return faces;
  }

  /// Generates a face embedding directly on the calling thread.
  Future<Float32List> getFaceEmbeddingDirect(Face face, cv.Mat image) async {
    if (_embedding == null) {
      throw StateError('Embedding model not initialized.');
    }

    final landmarks = face.landmarks;
    final leftEye = landmarks.leftEye;
    final rightEye = landmarks.rightEye;

    if (leftEye == null || rightEye == null) {
      throw StateError('Face must have left and right eye landmarks');
    }

    return getFaceEmbeddingFromEyesDirect(leftEye, rightEye, image);
  }

  /// Generates a face embedding from just the two eye landmark points (in
  /// absolute pixels, iris-refined when the source face had iris data).
  ///
  /// The embedding alignment consumes nothing else from a [Face], so the RPC
  /// protocol ships these four doubles instead of serializing the whole face
  /// (mesh and iris included) across the isolate boundary.
  Future<Float32List> getFaceEmbeddingFromEyesDirect(
    Point leftEye,
    Point rightEye,
    cv.Mat image,
  ) async {
    if (_embedding == null) {
      throw StateError('Embedding model not initialized.');
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
      outSize: _embedding!.inputWidth,
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

  /// Disposes all model resources.
  void dispose() => _disposeFields();

  /// Detection tensor scratch, reused across calls. Written only inside
  /// [_detectorLock] - conversion must not run outside it, or a concurrent
  /// request could overwrite the tensor while inference is still reading it.
  Float32List? _detectionScratch;

  Future<List<Detection>> _detectDetections(cv.Mat image) async {
    final FaceDetection? d = _detector;
    if (d == null) throw StateError('FaceDetectorCore not initialized.');

    return await _detectorLock.run(() {
      final ImageTensor tensor = convertImageToTensor(
        image,
        outW: d.inputWidth,
        outH: d.inputHeight,
        buffer: _detectionScratch ??= Float32List(
          d.inputWidth * d.inputHeight * 3,
        ),
      );
      return d.callWithTensor(tensor);
    });
  }

  Future<AlignedFace> _estimateAlignedFace(cv.Mat image, Detection det) async {
    final (:theta, :cx, :cy, :size) = _computeFaceAlignment(
      det,
      image.cols.toDouble(),
      image.rows.toDouble(),
    );

    // Warp straight to the mesh model's input resolution: one resample
    // instead of a full-size crop followed by a resize inside
    // convertImageToTensor.
    final cv.Mat? faceCrop = extractAlignedSquare(
      image,
      cx,
      cy,
      size,
      -theta,
      outSize: _meshItems.isNotEmpty ? _meshItems.first.inputWidth : null,
    );
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

  Future<({List<Point> points, double? score})> _meshFromAlignedFace(
    cv.Mat faceCrop,
    double cx,
    double cy,
    double size,
    double theta,
  ) async {
    if (_meshPool == null || _meshPool!.isEmpty) {
      return (points: <Point>[], score: null);
    }
    final res = await _meshPool!.withItem((fl) => fl.callWithScore(faceCrop));
    return (
      points: _transformMeshToAbsolute(res.landmarks, cx, cy, size, theta),
      score: res.score,
    );
  }

  /// Eye crop extraction (warpAffine) is done serially to avoid opencv_dart
  /// freeze issues, but TFLite inference runs in parallel for performance.
  ///
  /// Eye ROIs are computed using the same geometry as
  /// [FaceDetector.eyeRoisFromMesh] to keep iris alignment consistent between
  /// the public API and the isolate path.
  Future<List<Point>> _irisFromMesh(cv.Mat image, List<Point> meshAbs) async {
    if (_irisLeft == null || _irisRight == null) return <Point>[];
    if (meshAbs.length < 468) return <Point>[];

    List<AlignedRoi> roisFromMesh(List<Point> mesh) {
      AlignedRoi fromCorners(int a, int b) {
        final p0 = mesh[a];
        final p1 = mesh[b];
        final cx = (p0.x + p1.x) * 0.5;
        final cy = (p0.y + p1.y) * 0.5;
        final dx = p1.x - p0.x;
        final dy = p1.y - p0.y;
        final eyeDist = math.sqrt(dx * dx + dy * dy);
        return AlignedRoi(cx, cy, eyeDist * 2.3, math.atan2(dy, dx));
      }

      return [fromCorners(33, 133), fromCorners(362, 263)];
    }

    final List<AlignedRoi> rois = roisFromMesh(meshAbs);
    if (rois.length < 2) return <Point>[];

    // Eye crops are warped straight to the iris model's input resolution
    // (single resample); the right-eye flip below then runs on the small
    // crop. Flip-then-resize and resize-then-flip sample identically, so
    // this matches the previous full-size-crop behavior.
    final int irisInputSize = _irisLeft!.inputWidth;
    final cv.Mat? leftCrop = extractAlignedSquare(
      image,
      rois[0].cx,
      rois[0].cy,
      rois[0].size,
      rois[0].theta,
      outSize: irisInputSize,
    );
    final cv.Mat? rightCropRaw = extractAlignedSquare(
      image,
      rois[1].cx,
      rois[1].cy,
      rois[1].size,
      rois[1].theta,
      outSize: irisInputSize,
    );

    if (leftCrop == null || rightCropRaw == null) {
      leftCrop?.dispose();
      rightCropRaw?.dispose();
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

    return <Point>[
      for (final p in leftAbs) Point(p[0], p[1], p[2]),
      for (final p in rightAbs) Point(p[0], p[1], p[2]),
    ];
  }

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
    d(() => _blendshapes?.dispose());
    d(() => _embedding?.dispose());
    _detector = null;
    _irisLeft = null;
    _irisRight = null;
    _blendshapes = null;
    _embedding = null;
  }

  void _cleanupOnInitError() => _disposeFields(safe: true);
}
