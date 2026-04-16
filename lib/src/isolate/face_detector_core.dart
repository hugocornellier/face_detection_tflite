part of '../../face_detection_tflite.dart';

/// Data passed to the detection isolate during startup.
class _DetectionIsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData faceDetectionBytes;
  final TransferableTypedData faceLandmarkBytes;
  final TransferableTypedData irisLandmarkBytes;
  final TransferableTypedData embeddingBytes;
  final String modelName;
  final String performanceModeName;
  final int? numThreads;
  final int meshPoolSize;

  _DetectionIsolateStartupData({
    required this.sendPort,
    required this.faceDetectionBytes,
    required this.faceLandmarkBytes,
    required this.irisLandmarkBytes,
    required this.embeddingBytes,
    required this.modelName,
    required this.performanceModeName,
    required this.numThreads,
    required this.meshPoolSize,
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

  _SegmentationIsolateStartupData({
    required this.sendPort,
    required this.modelBytes,
    required this.performanceModeName,
    required this.numThreads,
    required this.maxOutputSize,
    required this.validateModel,
    required this.modelName,
    required this.modelIndex,
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
  FaceEmbedding? _embedding;

  final _irisLeftLock = _InferenceLock();
  final _irisRightLock = _InferenceLock();
  final _embeddingLock = _InferenceLock();

  /// Returns true when the core has been initialized with model data.
  bool get isReady => _detector != null;

  /// Returns true when the embedding model is loaded and ready.
  bool get isEmbeddingReady => _embedding != null;

  /// Initializes all TFLite models from pre-loaded bytes.
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

    final List<Detection> dets = await _detectDetections(image);
    if (dets.isEmpty) return <Face>[];

    final List<(Detection, AlignedFace)?> alignedFaces =
        <(Detection, AlignedFace)?>[];
    for (final Detection det in dets) {
      try {
        final aligned = await _estimateAlignedFace(image, det);
        alignedFaces.add((det, aligned));
      } catch (_) {
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
          } catch (_) {
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
          final leftCenter = _irisCenterFromPoints(
              irisPx.sublist(_kLeftIrisStart, _kLeftIrisEnd));
          kp[FaceLandmarkType.leftEye.index * 2] = leftCenter.x / width;
          kp[FaceLandmarkType.leftEye.index * 2 + 1] = leftCenter.y / height;
        }
        if (irisPx.length >= _kRightIrisEnd) {
          final rightCenter = _irisCenterFromPoints(
              irisPx.sublist(_kRightIrisStart, _kRightIrisEnd));
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
          mesh: meshPx.isNotEmpty ? FaceMesh(meshPx) : null,
          irises: irisPx,
          originalSize: imgSize,
        ),
      );
    }

    return faces;
  }

  /// Generates a face embedding directly on the calling thread.
  Future<Float32List> getFaceEmbeddingDirect(
    Face face,
    cv.Mat image,
  ) async {
    if (_embedding == null) {
      throw StateError('Embedding model not initialized.');
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

  /// Disposes all model resources.
  void dispose() => _disposeFields();

  Future<List<Detection>> _detectDetections(cv.Mat image) async {
    final FaceDetection? d = _detector;
    if (d == null) throw StateError('FaceDetectorCore not initialized.');

    final ImageTensor tensor = convertImageToTensor(
      image,
      outW: d.inputWidth,
      outH: d.inputHeight,
    );

    return await d.callWithTensor(tensor);
  }

  Future<AlignedFace> _estimateAlignedFace(
    cv.Mat image,
    Detection det,
  ) async {
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
        cx: cx, cy: cy, size: size, theta: theta, faceCrop: faceCrop);
  }

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

  /// Eye crop extraction (warpAffine) is done serially to avoid opencv_dart
  /// freeze issues, but TFLite inference runs in parallel for performance.
  Future<List<Point>> _irisFromMesh(
    cv.Mat image,
    List<Point> meshAbs,
  ) async {
    if (_irisLeft == null || _irisRight == null) return <Point>[];
    if (meshAbs.length < 468) return <Point>[];

    // Compute eye ROIs using the same geometry as FaceDetector.eyeRoisFromMesh.
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
    d(() => _embedding?.dispose());
    _detector = null;
    _irisLeft = null;
    _irisRight = null;
    _embedding = null;
  }

  void _cleanupOnInitError() => _disposeFields(safe: true);
}
