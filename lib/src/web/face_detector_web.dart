// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:async';
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:flutter_litert/flutter_litert.dart'
    show Point, PerformanceConfig;
import 'package:web/web.dart' as web;

import 'models/face_detection_model_web.dart';
import 'models/face_landmark_model_web.dart';
import 'models/iris_landmark_model_web.dart';
import 'models/selfie_segmentation_web.dart';
import 'types.dart';

/// Per-stage timing accumulator for the web pipeline (microseconds). Populated
/// by [FaceDetector.detectFaces] when [FaceDetector.debugTimings] is true.
class WebDetectTimings {
  int decodeUs = 0;
  int detPreUs = 0;
  int detInferUs = 0;
  int meshPreUs = 0;
  int meshInferUs = 0;
  int irisPreUs = 0;
  int irisInferUs = 0;
  int totalUs = 0;
  int detections = 0;

  Map<String, int> toJsonUs() => {
        'decode_us': decodeUs,
        'det_pre_us': detPreUs,
        'det_infer_us': detInferUs,
        'mesh_pre_us': meshPreUs,
        'mesh_infer_us': meshInferUs,
        'iris_pre_us': irisPreUs,
        'iris_infer_us': irisInferUs,
        'total_us': totalUs,
        'detections': detections,
      };
}

/// Web implementation of FaceDetector.
///
/// Mirrors the public API surface of the native FaceDetector for the
/// detect-from-bytes use case. Native-only methods (filepath, mat, camera
/// frames) throw [UnsupportedError] on web.
class FaceDetector {
  static const String modelVersion = '1.0.0';

  FaceDetector();

  static Future<FaceDetector> create({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
    bool withSegmentation = false,
    SegmentationConfig? segmentationConfig,
    bool useLiteRt = true,
    String liteRtAccelerator = 'auto',
  }) async {
    final detector = FaceDetector();
    await detector.initialize(
      model: model,
      performanceConfig: performanceConfig,
      meshPoolSize: meshPoolSize,
      withSegmentation: withSegmentation,
      segmentationConfig: segmentationConfig,
      useLiteRt: useLiteRt,
      liteRtAccelerator: liteRtAccelerator,
    );
    return detector;
  }

  final FaceDetectionModelWeb _detector = FaceDetectionModelWeb();
  final FaceLandmarkModelWeb _mesh = FaceLandmarkModelWeb();
  final IrisLandmarkModelWeb _iris = IrisLandmarkModelWeb();
  SelfieSegmentationWeb? _segmenter;

  bool _detectorReady = false;
  bool _meshReady = false;
  bool _irisReady = false;
  bool _segmentationReady = false;

  String _liteRtAccelerator = 'auto';

  /// Last-call per-stage timings (set when [debugTimings] is true).
  WebDetectTimings? lastTimings;

  /// When true, [detectFaces] populates [lastTimings].
  bool debugTimings = false;

  bool get isReady => _detectorReady && _meshReady && _irisReady;
  bool get isEmbeddingReady => false;
  bool get isSegmentationReady => _segmentationReady;

  Future<void> initialize({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
    bool withSegmentation = false,
    SegmentationConfig? segmentationConfig,
    bool useLiteRt = true,
    String liteRtAccelerator = 'auto',
  }) async {
    if (isReady) {
      throw StateError('FaceDetector already initialized');
    }
    _liteRtAccelerator = liteRtAccelerator;
    await _detector.initialize(
      model,
      liteRtAccelerator: liteRtAccelerator,
    );
    _detectorReady = true;
    await _mesh.initialize(liteRtAccelerator: liteRtAccelerator);
    _meshReady = true;
    await _iris.initialize(liteRtAccelerator: liteRtAccelerator);
    _irisReady = true;

    if (withSegmentation) {
      final cfg = segmentationConfig ?? SegmentationConfig.safe;
      _segmenter = SelfieSegmentationWeb();
      await _segmenter!.initialize(
        model: cfg.model,
        liteRtAccelerator: liteRtAccelerator,
      );
      _segmentationReady = true;
    }
  }

  Future<void> initializeSegmentation({SegmentationConfig? config}) async {
    if (!isReady) {
      throw StateError('FaceDetector must be initialized first.');
    }
    if (_segmentationReady) return;
    final cfg = config ?? SegmentationConfig.safe;
    _segmenter = SelfieSegmentationWeb();
    await _segmenter!.initialize(
      model: cfg.model,
      liteRtAccelerator: _liteRtAccelerator,
    );
    _segmentationReady = true;
  }

  Future<void> dispose() async {
    await _detector.dispose();
    await _mesh.dispose();
    await _iris.dispose();
    await _segmenter?.dispose();
    _segmenter = null;
    _detectorReady = false;
    _meshReady = false;
    _irisReady = false;
    _segmentationReady = false;
  }

  /// Detects faces in encoded image bytes (JPEG/PNG/...).
  Future<List<Face>> detectFaces(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    if (!isReady) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before using.',
      );
    }
    final t = debugTimings ? WebDetectTimings() : null;
    final totalSw = t != null ? (Stopwatch()..start()) : Stopwatch();
    final sw = Stopwatch();

    if (t != null) sw.start();
    final web.ImageBitmap? bitmap = await _decodeBitmap(imageBytes);
    if (t != null) {
      sw.stop();
      t.decodeUs = sw.elapsedMicroseconds;
      sw.reset();
    }
    if (bitmap == null) {
      if (t != null) {
        totalSw.stop();
        t.totalUs = totalSw.elapsedMicroseconds;
        lastTimings = t;
      }
      return const <Face>[];
    }
    final int imageWidth = bitmap.width;
    final int imageHeight = bitmap.height;

    if (t != null) sw.start();
    final List<Detection> dets = await _detector.detect(
      bitmap,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );
    if (t != null) {
      sw.stop();
      t.detInferUs = sw.elapsedMicroseconds;
      sw.reset();
    }

    final imgSize = Size(imageWidth.toDouble(), imageHeight.toDouble());
    if (mode == FaceDetectionMode.fast || dets.isEmpty) {
      bitmap.close();
      final result = <Face>[
        for (final d in dets)
          Face(
            detection: d.imageSize == null
                ? Detection(
                    boundingBox: d.boundingBox,
                    score: d.score,
                    keypointsXY: d.keypointsXY,
                    imageSize: imgSize,
                  )
                : d,
            mesh: null,
            irises: const <Point>[],
            originalSize: imgSize,
          ),
      ];
      if (t != null) {
        totalSw.stop();
        t.totalUs = totalSw.elapsedMicroseconds;
        t.detections = dets.length;
        lastTimings = t;
      }
      return result;
    }

    final List<Face> faces = <Face>[];
    for (final d in dets) {
      final align = FaceDetectionModelWeb.faceAlignment(
        Detection(
          boundingBox: d.boundingBox,
          score: d.score,
          keypointsXY: d.keypointsXY,
          imageSize: imgSize,
        ),
        imageWidth.toDouble(),
        imageHeight.toDouble(),
      );

      if (t != null) sw.start();
      final mesh = await _mesh.runOnCrop(
        bitmap,
        cx: align.cx,
        cy: align.cy,
        size: align.size,
        theta: align.theta,
      );
      if (t != null) {
        sw.stop();
        t.meshInferUs += sw.elapsedMicroseconds;
        sw.reset();
      }

      final List<Point> meshPoints = _transformMeshToAbsolute(
        mesh.landmarks,
        align.cx,
        align.cy,
        align.size,
        align.theta,
        _mesh.inputWidth,
        _mesh.inputHeight,
      );
      if (meshPoints.length > kMeshPoints) {
        meshPoints.removeRange(kMeshPoints, meshPoints.length);
      }

      List<Point> irisPoints = const <Point>[];
      if (mode == FaceDetectionMode.full) {
        // Compute eye ROIs from mesh landmark indices used by MediaPipe:
        // left:  33  (inner)  / 133 (outer)
        // right: 362 (inner)  / 263 (outer)
        AlignedRoi roiFromCorners(int a, int b) {
          final p0 = meshPoints[a];
          final p1 = meshPoints[b];
          final cx = (p0.x + p1.x) * 0.5;
          final cy = (p0.y + p1.y) * 0.5;
          final dx = p1.x - p0.x;
          final dy = p1.y - p0.y;
          final eyeDist = math.sqrt(dx * dx + dy * dy);
          return AlignedRoi(cx, cy, eyeDist * 2.3, math.atan2(dy, dx));
        }

        final leftRoi = roiFromCorners(33, 133);
        final rightRoi = roiFromCorners(362, 263);

        if (t != null) sw.start();
        final leftFlat = await _iris.runOnEyeCrop(
          bitmap,
          cx: leftRoi.cx,
          cy: leftRoi.cy,
          size: leftRoi.size,
          theta: leftRoi.theta,
          isRight: false,
        );
        final rightFlat = await _iris.runOnEyeCrop(
          bitmap,
          cx: rightRoi.cx,
          cy: rightRoi.cy,
          size: rightRoi.size,
          theta: rightRoi.theta,
          isRight: true,
        );
        if (t != null) {
          sw.stop();
          t.irisInferUs += sw.elapsedMicroseconds;
          sw.reset();
        }

        final leftPts = _transformIrisToAbsolute(
          leftFlat,
          leftRoi,
          false,
          _iris.inputWidth,
          _iris.inputHeight,
        );
        final rightPts = _transformIrisToAbsolute(
          rightFlat,
          rightRoi,
          true,
          _iris.inputWidth,
          _iris.inputHeight,
        );
        irisPoints = <Point>[...leftPts, ...rightPts];
      }

      faces.add(
        Face(
          detection: Detection(
            boundingBox: d.boundingBox,
            score: d.score,
            keypointsXY: d.keypointsXY,
            imageSize: imgSize,
          ),
          mesh: meshPoints.length == kMeshPoints ? FaceMesh(meshPoints) : null,
          irises: irisPoints,
          originalSize: imgSize,
        ),
      );
    }

    bitmap.close();
    if (t != null) {
      totalSw.stop();
      t.totalUs = totalSw.elapsedMicroseconds;
      t.detections = dets.length;
      lastTimings = t;
    }
    return faces;
  }

  /// Detects faces and returns a [SegmentationMask] alongside the faces.
  /// Web runs them sequentially (no isolates).
  Future<DetectionWithSegmentationResult> detectFacesWithSegmentation(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) async {
    if (!isReady) {
      throw StateError('FaceDetector not initialized.');
    }
    if (!_segmentationReady) {
      throw StateError(
        'Segmentation not initialized. Call initializeSegmentation() or '
        'initialize(withSegmentation: true).',
      );
    }
    final detSw = Stopwatch()..start();
    final faces = await detectFaces(imageBytes, mode: mode);
    detSw.stop();
    final segSw = Stopwatch()..start();
    final mask = await getSegmentationMask(imageBytes);
    segSw.stop();
    return DetectionWithSegmentationResult(
      faces: faces,
      segmentationMask: mask,
      detectionTimeMs: detSw.elapsedMilliseconds,
      segmentationTimeMs: segSw.elapsedMilliseconds,
    );
  }

  Future<SegmentationMask> getSegmentationMask(
    Uint8List imageBytes, {
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) async {
    if (!_segmentationReady || _segmenter == null) {
      throw StateError(
        'Segmentation not initialized. Call initializeSegmentation() or '
        'initialize(withSegmentation: true).',
      );
    }
    final web.ImageBitmap? bitmap = await _decodeBitmap(imageBytes);
    if (bitmap == null) {
      throw const SegmentationException(
        SegmentationError.imageDecodeFailed,
        'Failed to decode image bytes via createImageBitmap.',
      );
    }
    try {
      return await _segmenter!.segment(
        bitmap,
        imageWidth: bitmap.width,
        imageHeight: bitmap.height,
      );
    } finally {
      bitmap.close();
    }
  }

  // ---- API parity stubs that throw on web -----------------------------------

  Future<List<Face>> detectFacesFromFilepath(
    String path, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    throw UnsupportedError(
      'detectFacesFromFilepath is not supported on web. '
      'Use detectFaces(bytes) instead.',
    );
  }

  Future<List<Face>> detectFacesFromMat(
    Object image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    throw UnsupportedError(
      'detectFacesFromMat is not supported on web. '
      'Use detectFaces(bytes) instead.',
    );
  }

  Future<List<Face>> detectFacesFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    throw UnsupportedError(
      'detectFacesFromMatBytes is not supported on web.',
    );
  }

  Future<List<Face>> detectFacesFromCameraFrame(
    Object frame, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    int? maxDim,
  }) {
    throw UnsupportedError(
      'detectFacesFromCameraFrame is not supported on web.',
    );
  }

  Future<List<Face>> detectFacesFromCameraImage(
    Object cameraImage, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    Object? rotation,
    bool isBgra = true,
    int? maxDim,
  }) {
    throw UnsupportedError(
      'detectFacesFromCameraImage is not supported on web.',
    );
  }

  Future<Float32List> getFaceEmbedding(Face face, Uint8List imageBytes) {
    throw UnsupportedError(
      'getFaceEmbedding is not supported on web in this version.',
    );
  }

  Future<Float32List> getFaceEmbeddingFromFilepath(Face face, String path) {
    throw UnsupportedError(
      'getFaceEmbeddingFromFilepath is not supported on web.',
    );
  }

  Future<Float32List> getFaceEmbeddingFromMatBytes(
    Face face,
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) {
    throw UnsupportedError(
      'getFaceEmbeddingFromMatBytes is not supported on web.',
    );
  }

  Future<Float32List> getFaceEmbeddingFromMat(Face face, Object image) {
    throw UnsupportedError('getFaceEmbeddingFromMat is not supported on web.');
  }

  Future<List<Float32List?>> getFaceEmbeddings(
    List<Face> faces,
    Uint8List imageBytes,
  ) {
    throw UnsupportedError('getFaceEmbeddings is not supported on web.');
  }

  static double compareFaces(Float32List a, Float32List b) {
    throw UnsupportedError('compareFaces is not supported on web.');
  }

  static double faceDistance(Float32List a, Float32List b) {
    throw UnsupportedError('faceDistance is not supported on web.');
  }

  Future<SegmentationMask> getSegmentationMaskFromMat(
    Object image, {
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) {
    throw UnsupportedError(
      'getSegmentationMaskFromMat is not supported on web.',
    );
  }

  Future<DetectionWithSegmentationResult> detectFacesWithSegmentationFromMat(
    Object image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
  }) {
    throw UnsupportedError(
      'detectFacesWithSegmentationFromMat is not supported on web.',
    );
  }

  Future<SegmentationMask> getSegmentationMaskFromCameraFrame(
    Object frame, {
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
    int? maxDim,
  }) {
    throw UnsupportedError(
      'getSegmentationMaskFromCameraFrame is not supported on web.',
    );
  }

  Future<DetectionWithSegmentationResult>
      detectFacesWithSegmentationFromCameraFrame(
    Object frame, {
    FaceDetectionMode mode = FaceDetectionMode.full,
    IsolateOutputFormat outputFormat = IsolateOutputFormat.float32,
    double binaryThreshold = 0.5,
    int? maxDim,
  }) {
    throw UnsupportedError(
      'detectFacesWithSegmentationFromCameraFrame is not supported on web.',
    );
  }

  /// Eye ROI extraction kept as a public helper for parity with native.
  List<AlignedRoi> eyeRoisFromMesh(List<Point> meshAbs) {
    AlignedRoi fromCorners(int a, int b) {
      final p0 = meshAbs[a];
      final p1 = meshAbs[b];
      final cx = (p0.x + p1.x) * 0.5;
      final cy = (p0.y + p1.y) * 0.5;
      final dx = p1.x - p0.x;
      final dy = p1.y - p0.y;
      final eyeDist = math.sqrt(dx * dx + dy * dy);
      return AlignedRoi(cx, cy, eyeDist * 2.3, math.atan2(dy, dx));
    }

    return [fromCorners(33, 133), fromCorners(362, 263)];
  }

  List<List<Point>> splitMeshesIfConcatenated(List<Point> meshPts) {
    if (meshPts.isEmpty) return const <List<Point>>[];
    if (meshPts.length % kMeshPoints != 0) return [meshPts];
    final int faces = meshPts.length ~/ kMeshPoints;
    return [
      for (int i = 0; i < faces; i++)
        meshPts.sublist(i * kMeshPoints, (i + 1) * kMeshPoints),
    ];
  }

  // ---------------------------------------------------------------------------

  static Future<web.ImageBitmap?> _decodeBitmap(Uint8List bytes) async {
    final web.Blob blob = web.Blob([bytes.toJS].toJS);
    try {
      final JSPromise<web.ImageBitmap> promise = web.window.createImageBitmap(
        blob,
      );
      return await promise.toDart;
    } catch (_) {
      return null;
    }
  }

  /// Inverse of the rotation+scale crop in [FaceLandmarkModelWeb.runOnCrop].
  /// Mesh landmarks come back in the model's input pixel space; this maps
  /// them back to original-image absolute coordinates.
  List<Point> _transformMeshToAbsolute(
    Float32List flat,
    double cx,
    double cy,
    double size,
    double theta,
    int inW,
    int inH,
  ) {
    final int n = flat.length ~/ 3;
    final List<Point> out = List<Point>.filled(n, const Point(0, 0, 0));
    final double ct = math.cos(theta);
    final double st = math.sin(theta);
    final double scale = size / inW;
    for (int i = 0; i < n; i++) {
      final double mx = flat[i * 3] - inW / 2.0;
      final double my = flat[i * 3 + 1] - inH / 2.0;
      final double mz = flat[i * 3 + 2];
      final double rx = ct * mx - st * my;
      final double ry = st * mx + ct * my;
      out[i] = Point(cx + rx * scale, cy + ry * scale, mz * size);
    }
    return out;
  }

  /// Inverse of the rotation+scale eye crop in [IrisLandmarkModelWeb.runOnEyeCrop].
  List<Point> _transformIrisToAbsolute(
    Float32List flat,
    AlignedRoi roi,
    bool isRight,
    int inW,
    int inH,
  ) {
    final int n = flat.length ~/ 3;
    final List<Point> out = List<Point>.filled(n, const Point(0, 0, 0));
    final double ct = math.cos(roi.theta);
    final double st = math.sin(roi.theta);
    final double scale = roi.size / inW;
    for (int i = 0; i < n; i++) {
      double mx = flat[i * 3] - inW / 2.0;
      final double my = flat[i * 3 + 1] - inH / 2.0;
      final double mz = flat[i * 3 + 2];
      if (isRight) mx = -mx;
      final double rx = ct * mx - st * my;
      final double ry = st * mx + ct * my;
      out[i] = Point(roi.cx + rx * scale, roi.cy + ry * scale, mz * roi.size);
    }
    return out;
  }
}
