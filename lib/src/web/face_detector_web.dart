// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:async';
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter_litert/flutter_litert.dart'
    show aggregateActiveAccelerator, Point, PerformanceConfig, Precision;
import 'package:flutter_litert/src/web/web_detector_utils.dart'
    show decodeBitmap, WebGpuFallback;
import 'package:web/web.dart' as web;

import '../shared/blendshape_input.dart' show packBlendshapeInput;
import '../shared/face_model_config.dart'
    show kDefaultMinFacePresenceConfidence;
import '../shared/face_gates.dart'
    show applyDetectionGates, applyFaceGates, validateFaceGates;
import '../shared/face_geometry.dart' show transformMeshFlatToAbsolute;
import 'models/face_blendshapes_model_web.dart';
import 'models/face_detection_model_web.dart';
import 'models/face_landmark_model_web.dart';
import 'models/iris_landmark_model_web.dart';
import 'models/selfie_segmentation_web.dart';
import 'models/web_model_runner.dart' show WebEngine, WebRunnerConfig;
import 'types.dart';

/// Per-stage timing accumulator for the web pipeline (microseconds). Populated
/// by [FaceDetector.detectFacesFromBytes] when [FaceDetector.debugTimings] is true.
class WebDetectTimings {
  int decodeUs = 0;
  int detPreUs = 0;
  int detInferUs = 0;
  int meshPreUs = 0;
  int meshInferUs = 0;
  int irisPreUs = 0;
  int irisInferUs = 0;
  int blendshapeInferUs = 0;
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
    'blendshape_infer_us': blendshapeInferUs,
    'total_us': totalUs,
    'detections': detections,
  };
}

/// Web implementation of FaceDetector.
///
/// Mirrors the public API surface of the native FaceDetector for the
/// detect-from-bytes use case. Native-only methods (filepath, mat, camera
/// frames) throw [UnsupportedError] on web.
class FaceDetector with WebGpuFallback {
  static const String modelVersion = '1.1.1';

  FaceDetector();

  static Future<FaceDetector> create({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
    bool withSegmentation = false,
    SegmentationConfig? segmentationConfig,
    bool useCompiledModel = false,
    bool useLiteRt = true,
    String liteRtAccelerator = 'auto',
    bool strictWebGpu = false,
    Precision precision = Precision.fp16,
    double minScore = 0.0,
    double minFaceSize = 0.0,
    double minFacePresenceConfidence = kDefaultMinFacePresenceConfidence,
  }) async {
    final detector = FaceDetector();
    await detector.initialize(
      model: model,
      performanceConfig: performanceConfig,
      meshPoolSize: meshPoolSize,
      withSegmentation: withSegmentation,
      segmentationConfig: segmentationConfig,
      useCompiledModel: useCompiledModel,
      useLiteRt: useLiteRt,
      liteRtAccelerator: liteRtAccelerator,
      strictWebGpu: strictWebGpu,
      precision: precision,
      minScore: minScore,
      minFaceSize: minFaceSize,
      minFacePresenceConfidence: minFacePresenceConfidence,
    );
    return detector;
  }

  final FaceDetectionModelWeb _detector = FaceDetectionModelWeb();
  final FaceLandmarkModelWeb _mesh = FaceLandmarkModelWeb();
  final IrisLandmarkModelWeb _iris = IrisLandmarkModelWeb();
  final FaceBlendshapesModelWeb _blendshapes = FaceBlendshapesModelWeb();
  SelfieSegmentationWeb? _segmenter;

  bool _detectorReady = false;
  bool _meshReady = false;
  bool _irisReady = false;
  bool _blendshapesReady = false;
  bool _segmentationReady = false;

  String _liteRtAccelerator = 'auto';
  bool _useCompiledModel = false;
  bool _strictWebGpu = false;
  Precision _precision = Precision.fp16;
  double _minScore = 0.0;
  double _minFaceSize = 0.0;
  double _minFacePresenceConfidence = kDefaultMinFacePresenceConfidence;

  /// Builds the runner config for the current engine selection at the given
  /// [accelerator] (`'auto'` / `'webgpu'` / `'wasm'`). Every per-model
  /// `initialize` goes through this so the LiteRT.js-interpreter and
  /// CompiledModel paths stay in lockstep.
  WebRunnerConfig _runnerConfig(String accelerator) => WebRunnerConfig(
    engine: _useCompiledModel ? WebEngine.compiledModel : WebEngine.liteRt,
    accelerator: accelerator,
    strictWebGpu: _strictWebGpu,
    precision: _precision,
  );

  /// Minimum detection confidence (0.0 to 1.0) a face must have to be returned.
  /// Defaults to 0.0 (no additional filtering above the internal 0.5 floor).
  double get minScore => _minScore;

  /// Minimum face size (face width / image width, 0.0 to 1.0) a detection must
  /// have to be returned. Defaults to 0.0 (no filtering).
  double get minFaceSize => _minFaceSize;

  /// Minimum face-presence confidence (0.0 to 1.0), gating the mesh model's
  /// "face flag" output ([Face.meshScore]). This is MediaPipe's
  /// `min_face_presence_confidence`; defaults to
  /// [kDefaultMinFacePresenceConfidence] (0.5). Only applies in
  /// `standard`/`full` modes (no mesh in `fast` mode). Pass 0.0 to disable.
  double get minFacePresenceConfidence => _minFacePresenceConfidence;

  List<Face> _applyGates(List<Face> faces) => applyFaceGates(
    faces,
    minScore: _minScore,
    minFaceSize: _minFaceSize,
    minFacePresenceConfidence: _minFacePresenceConfidence,
  );

  /// Last-call per-stage timings (set when [debugTimings] is true).
  WebDetectTimings? lastTimings;

  /// When true, [detectFacesFromBytes] populates [lastTimings].
  bool debugTimings = false;

  bool get isReady =>
      _detectorReady && _meshReady && _irisReady && _blendshapesReady;
  bool get isEmbeddingReady => false;
  bool get isSegmentationReady => _segmentationReady;

  /// The accelerator currently in use across all model runners (`'webgpu'`
  /// / `'wasm'`), or null pre-init. May change at runtime if a GPU error
  /// fires on WebGPU and the detector swaps to WASM.
  ///
  /// Reports `'webgpu'` when any runner is still on WebGPU so runtime fallback
  /// and the slow-WebGPU warmup stay enabled under mixed compile outcomes; see
  /// [aggregateActiveAccelerator].
  @override
  String? get activeAccelerator => aggregateActiveAccelerator(<String?>[
    _detector.activeAccelerator,
    _mesh.activeAccelerator,
    _iris.activeAccelerator,
    _blendshapes.activeAccelerator,
    _segmenter?.activeAccelerator,
  ]);

  /// Per-runner backend after initialization (`'webgpu'` / `'wasm'`, or null
  /// for a runner that is absent or not yet initialized).
  ///
  /// Each model compiles independently and can fall back from WebGPU to WASM on
  /// its own, so these are not guaranteed to agree. Exposed for diagnostics and
  /// tests that need to observe the real per-model backend rather than the
  /// aggregate reported by [activeAccelerator].
  Map<String, String?> get acceleratorReport => <String, String?>{
    'detector': _detector.activeAccelerator,
    'mesh': _mesh.activeAccelerator,
    'iris': _iris.activeAccelerator,
    'blendshapes': _blendshapes.activeAccelerator,
    'segmenter': _segmenter?.activeAccelerator,
  };

  // Cached init args so [swapToWasm] can re-init the runners with the same
  // model selection but a forced 'wasm' accelerator.
  FaceDetectionModel _model = FaceDetectionModel.backCamera;
  bool _withSegmentation = false;
  SegmentationConfig? _segmentationConfig;

  @override
  Future<void> swapToWasm() async {
    _liteRtAccelerator = 'wasm';
    try {
      await _detector.dispose();
      await _mesh.dispose();
      await _iris.dispose();
      await _blendshapes.dispose();
      await _segmenter?.dispose();
    } catch (_) {
      // Best-effort: an interpreter that already errored may not dispose
      // cleanly. Continue to re-init regardless.
    }
    final WebRunnerConfig config = _runnerConfig('wasm');
    await _detector.initialize(_model, config: config);
    await _mesh.initialize(config: config);
    await _iris.initialize(config: config);
    await _blendshapes.initialize(config: config);
    if (_withSegmentation) {
      _segmenter = SelfieSegmentationWeb();
      await _segmenter!.initialize(
        model: (_segmentationConfig ?? SegmentationConfig.safe).model,
        config: config,
      );
    }
  }

  // Median per-inference budget for the BlazeFace stage on WebGPU before
  // 'auto' abandons it. Chrome's Dawn lands at 2-5ms on typical hardware
  // while WASM SIMD is ~8-10ms, so a WebGPU path past this budget loses to
  // WASM by a wide margin (Firefox 152 measures ~200ms here).
  static const double _kAutoWebGpuBudgetMs = 50.0;
  static const int _kAutoWarmupRuns = 2;
  static const int _kAutoTimedRuns = 3;

  /// Catches the slow-but-functional WebGPU case (e.g. Firefox) that the
  /// error-driven fallback can never see: times a few detector inferences on
  /// a synthetic frame and swaps every runner to WASM when the median
  /// exceeds [_kAutoWebGpuBudgetMs]. Runs only when the caller asked for
  /// `'auto'` and the resolver landed on WebGPU.
  Future<void> _swapToWasmIfWebGpuSlow() async {
    final web.HTMLCanvasElement probe = web.HTMLCanvasElement()
      ..width = 64
      ..height = 64;
    final ctx = probe.getContext('2d') as web.CanvasRenderingContext2D;
    ctx.fillStyle = 'rgb(127,127,127)'.toJS;
    ctx.fillRect(0, 0, 64, 64);
    try {
      for (int i = 0; i < _kAutoWarmupRuns; i++) {
        await _detector.detect(probe, imageWidth: 64, imageHeight: 64);
      }
      final List<double> timesMs = <double>[];
      for (int i = 0; i < _kAutoTimedRuns; i++) {
        final Stopwatch sw = Stopwatch()..start();
        await _detector.detect(probe, imageWidth: 64, imageHeight: 64);
        sw.stop();
        timesMs.add(sw.elapsedMicroseconds / 1000.0);
      }
      timesMs.sort();
      final double medianMs = timesMs[timesMs.length ~/ 2];
      if (medianMs > _kAutoWebGpuBudgetMs) {
        debugPrint(
          'face_detection_tflite: WebGPU warmup median '
          '${medianMs.toStringAsFixed(1)}ms exceeds the '
          '${_kAutoWebGpuBudgetMs.toStringAsFixed(0)}ms auto budget; '
          'switching to WASM.',
        );
        await swapToWasm();
      } else {
        debugPrint(
          'face_detection_tflite: WebGPU warmup median '
          '${medianMs.toStringAsFixed(1)}ms; keeping WebGPU.',
        );
      }
    } catch (_) {
      // A GPU failure this early is the same signal, just louder.
      await swapToWasm();
    }
  }

  Future<void> initialize({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    int meshPoolSize = 3,
    bool withSegmentation = false,
    SegmentationConfig? segmentationConfig,
    bool useCompiledModel = false,
    bool useLiteRt = true,
    String liteRtAccelerator = 'auto',
    bool strictWebGpu = false,
    Precision precision = Precision.fp16,
    double minScore = 0.0,
    double minFaceSize = 0.0,
    double minFacePresenceConfidence = kDefaultMinFacePresenceConfidence,
  }) async {
    if (isReady) {
      throw StateError('FaceDetector already initialized');
    }
    validateFaceGates(
      minScore: minScore,
      minFaceSize: minFaceSize,
      minFacePresenceConfidence: minFacePresenceConfidence,
    );
    _minScore = minScore;
    _minFaceSize = minFaceSize;
    _minFacePresenceConfidence = minFacePresenceConfidence;
    _liteRtAccelerator = liteRtAccelerator;
    _useCompiledModel = useCompiledModel;
    _strictWebGpu = strictWebGpu;
    _precision = precision;
    _model = model;
    _withSegmentation = withSegmentation;
    _segmentationConfig = segmentationConfig;
    final WebRunnerConfig config = _runnerConfig(liteRtAccelerator);
    await _detector.initialize(model, config: config);
    _detectorReady = true;
    await _mesh.initialize(config: config);
    _meshReady = true;
    await _iris.initialize(config: config);
    _irisReady = true;
    await _blendshapes.initialize(config: config);
    _blendshapesReady = true;

    if (withSegmentation) {
      final cfg = segmentationConfig ?? SegmentationConfig.safe;
      _segmenter = SelfieSegmentationWeb();
      await _segmenter!.initialize(model: cfg.model, config: config);
      _segmentationReady = true;
    }

    if (liteRtAccelerator == 'auto' && activeAccelerator == 'webgpu') {
      await _swapToWasmIfWebGpuSlow();
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
      config: _runnerConfig(_liteRtAccelerator),
    );
    _segmentationReady = true;
  }

  Future<void> dispose() async {
    await _detector.dispose();
    await _mesh.dispose();
    await _iris.dispose();
    await _blendshapes.dispose();
    await _segmenter?.dispose();
    _segmenter = null;
    _detectorReady = false;
    _meshReady = false;
    _irisReady = false;
    _blendshapesReady = false;
    _segmentationReady = false;
  }

  /// Detects faces in encoded image bytes (JPEG/PNG/...).
  ///
  /// Internally decodes via `createImageBitmap` (off-thread) and routes
  /// through the same pipeline used by [detectFacesFromVideo].
  Future<List<Face>> detectFacesFromBytes(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    if (!isReady) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before using.',
      );
    }
    return withFallback(() => _detectFacesInner(imageBytes, mode: mode));
  }

  /// Deprecated alias for [detectFacesFromBytes].
  ///
  /// Renamed for clarity: the input is encoded image bytes (JPEG/PNG/...).
  @Deprecated(
    'Use detectFacesFromBytes instead. Will be removed in a future release.',
  )
  Future<List<Face>> detectFaces(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) => detectFacesFromBytes(imageBytes, mode: mode);

  Future<List<Face>> _detectFacesInner(
    Uint8List imageBytes, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    final t = debugTimings ? WebDetectTimings() : null;
    final totalSw = t != null ? (Stopwatch()..start()) : Stopwatch();

    final Stopwatch decodeSw = Stopwatch()..start();
    final web.ImageBitmap? bitmap = await decodeBitmap(imageBytes);
    decodeSw.stop();
    if (t != null) t.decodeUs = decodeSw.elapsedMicroseconds;
    if (bitmap == null) {
      if (t != null) {
        totalSw.stop();
        t.totalUs = totalSw.elapsedMicroseconds;
        lastTimings = t;
      }
      return const <Face>[];
    }
    try {
      final faces = await _runPipelineOnSource(
        bitmap,
        imageWidth: bitmap.width,
        imageHeight: bitmap.height,
        mode: mode,
        timings: t,
      );
      if (t != null) {
        totalSw.stop();
        t.totalUs = totalSw.elapsedMicroseconds;
        lastTimings = t;
      }
      return faces;
    } finally {
      bitmap.close();
    }
  }

  /// Detects faces from a live `<video>` element (webcam feed).
  ///
  /// The video must be playing and have non-zero `videoWidth`/`videoHeight`
  /// (i.e. past the `loadedmetadata` event). Returns an empty list if those
  /// dimensions are still zero - useful while the camera is warming up.
  ///
  /// This skips the JPEG/PNG decode stage entirely: `ctx.drawImage` accepts
  /// `HTMLVideoElement` directly, so we save ~1ms per frame at 30fps.
  Future<List<Face>> detectFacesFromVideo(
    web.HTMLVideoElement video, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    if (!isReady) {
      throw StateError(
        'FaceDetector not initialized. Call initialize() before using.',
      );
    }
    final int width = video.videoWidth;
    final int height = video.videoHeight;
    if (width == 0 || height == 0) return const <Face>[];

    return withFallback(
      () => _detectFacesFromVideoInner(
        video,
        width: width,
        height: height,
        mode: mode,
      ),
    );
  }

  Future<List<Face>> _detectFacesFromVideoInner(
    web.HTMLVideoElement video, {
    required int width,
    required int height,
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) async {
    final t = debugTimings ? WebDetectTimings() : null;
    final totalSw = t != null ? (Stopwatch()..start()) : Stopwatch();
    final faces = await _runPipelineOnSource(
      video,
      imageWidth: width,
      imageHeight: height,
      mode: mode,
      timings: t,
    );
    if (t != null) {
      totalSw.stop();
      t.totalUs = totalSw.elapsedMicroseconds;
      lastTimings = t;
    }
    return faces;
  }

  /// Core pipeline: BlazeFace → mesh → iris on whatever drawable the caller
  /// hands us (`ImageBitmap`, `HTMLVideoElement`, `HTMLCanvasElement`, etc.).
  Future<List<Face>> _runPipelineOnSource(
    JSObject source, {
    required int imageWidth,
    required int imageHeight,
    required FaceDetectionMode mode,
    WebDetectTimings? timings,
  }) async {
    final sw = Stopwatch();
    if (timings != null) sw.start();
    List<Detection> dets = await _detector.detect(
      source,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );
    if (timings != null) {
      sw.stop();
      timings.detInferUs = sw.elapsedMicroseconds;
      timings.detections = dets.length;
      sw.reset();
    }

    // Early gate: same arithmetic as the late _applyGates pass (shared
    // helpers), applied before the per-face mesh/iris/blendshape stages so
    // gated-out detections skip that work entirely. detections above stays
    // the raw post-NMS count for diagnostics.
    dets = applyDetectionGates(
      dets,
      minScore: _minScore,
      minFaceSize: _minFaceSize,
      imageWidth: imageWidth.toDouble(),
    );

    final imgSize = Size(imageWidth.toDouble(), imageHeight.toDouble());
    if (mode == FaceDetectionMode.fast || dets.isEmpty) {
      return _applyGates(<Face>[
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
      ]);
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

      if (timings != null) sw.start();
      final mesh = await _mesh.runOnCrop(
        source,
        cx: align.cx,
        cy: align.cy,
        size: align.size,
        theta: align.theta,
      );
      if (timings != null) {
        sw.stop();
        timings.meshInferUs += sw.elapsedMicroseconds;
        sw.reset();
      }

      final List<Point> meshPoints = transformMeshFlatToAbsolute(
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

      // Face-presence gate (MediaPipe min_face_presence_confidence): drop crops
      // the mesh model does not confirm as a face here, before the iris and
      // blendshape stages, so rejected detections cost nothing further. The
      // score is read the same way Face.meshScore exposes it (null when no
      // full mesh, which always passes); the final _applyGates pass re-checks.
      final double? presenceScore = meshPoints.length == kMeshPoints
          ? mesh.score
          : null;
      if (_minFacePresenceConfidence > 0.0 &&
          (presenceScore ?? double.infinity) < _minFacePresenceConfidence) {
        continue;
      }

      List<Point> irisPoints = const <Point>[];
      if (mode == FaceDetectionMode.full) {
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

        if (timings != null) sw.start();
        final leftFlat = await _iris.runOnEyeCrop(
          source,
          cx: leftRoi.cx,
          cy: leftRoi.cy,
          size: leftRoi.size,
          theta: leftRoi.theta,
          isRight: false,
        );
        final rightFlat = await _iris.runOnEyeCrop(
          source,
          cx: rightRoi.cx,
          cy: rightRoi.cy,
          size: rightRoi.size,
          theta: rightRoi.theta,
          isRight: true,
        );
        if (timings != null) {
          sw.stop();
          timings.irisInferUs += sw.elapsedMicroseconds;
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

      // Blendshape classification: repack the mesh + iris points and run one
      // sub-millisecond WASM inference. Full mode only; null when the packer
      // rejects the input or the model produces a NaN.
      List<double>? blendshapeScores;
      if (mode == FaceDetectionMode.full) {
        final Float32List? packed = packBlendshapeInput(meshPoints, irisPoints);
        if (packed != null) {
          if (timings != null) sw.start();
          blendshapeScores = await _blendshapes.run(packed);
          if (timings != null) {
            sw.stop();
            timings.blendshapeInferUs += sw.elapsedMicroseconds;
            sw.reset();
          }
        }
      }

      faces.add(
        Face(
          detection: Detection(
            boundingBox: d.boundingBox,
            score: d.score,
            keypointsXY: d.keypointsXY,
            imageSize: imgSize,
          ),
          mesh: meshPoints.length == kMeshPoints
              ? FaceMesh(meshPoints, score: mesh.score)
              : null,
          irises: irisPoints,
          blendshapeScores: blendshapeScores,
          originalSize: imgSize,
        ),
      );
    }
    return _applyGates(faces);
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
    // Decode the image once and feed the same bitmap to both stages; the
    // previous implementation decoded the identical bytes twice (once per
    // stage), adding a full createImageBitmap to every combined call.
    return withFallback(() async {
      final detSw = Stopwatch()..start();
      final web.ImageBitmap? bitmap = await decodeBitmap(imageBytes);
      if (bitmap == null) {
        // Match the previous behavior: detection yields no faces on decode
        // failure and the segmentation stage throws.
        throw const SegmentationException(
          SegmentationError.imageDecodeFailed,
          'Failed to decode image bytes via createImageBitmap.',
        );
      }
      try {
        final faces = await _runPipelineOnSource(
          bitmap,
          imageWidth: bitmap.width,
          imageHeight: bitmap.height,
          mode: mode,
        );
        detSw.stop();
        final segSw = Stopwatch()..start();
        final mask = await _segmenter!.segment(
          bitmap,
          imageWidth: bitmap.width,
          imageHeight: bitmap.height,
        );
        segSw.stop();
        return DetectionWithSegmentationResult(
          faces: faces,
          segmentationMask: mask,
          detectionTimeMs: detSw.elapsedMilliseconds,
          segmentationTimeMs: segSw.elapsedMilliseconds,
        );
      } finally {
        bitmap.close();
      }
    });
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
    return withFallback(() => _getSegmentationMaskInner(imageBytes));
  }

  Future<SegmentationMask> _getSegmentationMaskInner(
    Uint8List imageBytes,
  ) async {
    final web.ImageBitmap? bitmap = await decodeBitmap(imageBytes);
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

  /// Runs segmentation on a live `<video>` frame (webcam feed).
  Future<SegmentationMask> getSegmentationMaskFromVideo(
    web.HTMLVideoElement video,
  ) async {
    if (!_segmentationReady || _segmenter == null) {
      throw StateError(
        'Segmentation not initialized. Call initializeSegmentation() or '
        'initialize(withSegmentation: true).',
      );
    }
    final int width = video.videoWidth;
    final int height = video.videoHeight;
    if (width == 0 || height == 0) {
      throw const SegmentationException(
        SegmentationError.imageTooSmall,
        'Video has no dimensions yet (loadedmetadata not fired).',
      );
    }
    return withFallback(
      () => _getSegmentationMaskFromVideoInner(
        video,
        width: width,
        height: height,
      ),
    );
  }

  Future<SegmentationMask> _getSegmentationMaskFromVideoInner(
    web.HTMLVideoElement video, {
    required int width,
    required int height,
  }) => _segmenter!.segment(video, imageWidth: width, imageHeight: height);

  // ---- API parity stubs that throw on web -----------------------------------

  Future<List<Face>> detectFacesFromFilepath(
    String path, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    throw UnsupportedError(
      'detectFacesFromFilepath is not supported on web. '
      'Use detectFacesFromBytes(bytes) instead.',
    );
  }

  Future<List<Face>> detectFacesFromMat(
    Object image, {
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    throw UnsupportedError(
      'detectFacesFromMat is not supported on web. '
      'Use detectFacesFromBytes(bytes) instead.',
    );
  }

  Future<List<Face>> detectFacesFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
    FaceDetectionMode mode = FaceDetectionMode.full,
  }) {
    throw UnsupportedError('detectFacesFromMatBytes is not supported on web.');
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
