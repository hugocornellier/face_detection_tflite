// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart'
    hide Detection, resolveWebAccelerator, logCompileFallback;
import 'package:web/web.dart' as web;

import '../../shared/face_geometry.dart' show computeFaceAlignment;
import '../../shared/face_model_config.dart';
import '../../shared/face_types.dart';
import '../detection_decode.dart';
import 'web_model_runner.dart';

/// Web BlazeFace runner. Runs on the LiteRT.js interpreter or a LiteRT Next
/// `CompiledModel`; auto-prefers WebGPU and otherwise falls back to WASM SIMD.
class FaceDetectionModelWeb {
  WebModelRunner? _runner;

  String? _activeAccelerator;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`),
  /// or null before [initialize] completes.
  String? get activeAccelerator => _runner != null ? _activeAccelerator : null;

  Float32List? _boxesOut;
  Float32List? _scoresOut;
  late int _inW;
  late int _inH;
  late List<List<double>> _anchors;
  // BlazeFace output geometry, derived from output element counts at init:
  // boxes tensor is [1, _n, _k], scores is [1, _n, 1].
  int _n = 0;
  int _k = 0;
  int _boxesIdx = 0;
  int _scoresIdx = 1;

  Float32List? _inputBuffer;
  web.HTMLCanvasElement? _canvas;
  web.CanvasRenderingContext2D? _ctx;
  bool _initialized = false;

  FaceDetectionModel _model = FaceDetectionModel.backCamera;

  bool get isInitialized => _initialized;
  int get inputWidth => _inW;
  int get inputHeight => _inH;

  Future<void> initialize(
    FaceDetectionModel model, {
    WebRunnerConfig config = const WebRunnerConfig(),
  }) async {
    if (_initialized) await dispose();

    _model = model;
    final SSDAnchorOptions opts = ssdOptionsFor(model);
    _inW = opts.inputSizeWidth;
    _inH = opts.inputSizeHeight;
    _anchors = generateAnchors(opts);

    final String assetPath =
        'packages/face_detection_tflite/assets/models/${faceDetectionModelFile(model)}';
    final ByteData raw = await rootBundle.load(assetPath);
    final bytes = raw.buffer.asUint8List();

    final runner = await WebModelRunner.create(
      bytes,
      config: config,
      modelLabel: 'BlazeFace',
    );
    _runner = runner;
    _activeAccelerator = runner.activeAccelerator;

    // BlazeFace has two outputs: boxes [1, N, K>=16] and scores [1, N, 1].
    // CompiledModel exposes byte sizes only, so identify them by element
    // count: boxes has the larger count (N*K), scores the smaller (N).
    final List<int> counts = runner.outputElementCounts;
    if (counts.length < 2) {
      throw StateError(
        'BlazeFace must have at least 2 outputs; got ${counts.length}.',
      );
    }
    int boxesIdx = 0;
    for (int i = 1; i < counts.length; i++) {
      if (counts[i] > counts[boxesIdx]) boxesIdx = i;
    }
    int scoresIdx = boxesIdx == 0 ? 1 : 0;
    for (int i = 0; i < counts.length; i++) {
      if (i != boxesIdx && counts[i] < counts[scoresIdx]) scoresIdx = i;
    }
    final int boxesElems = counts[boxesIdx];
    final int scoresElems = counts[scoresIdx];
    if (scoresElems <= 0 ||
        boxesElems % scoresElems != 0 ||
        boxesElems ~/ scoresElems < 16) {
      throw StateError(
        'BlazeFace outputs do not match expected geometry. Counts: $counts',
      );
    }
    _boxesIdx = boxesIdx;
    _scoresIdx = scoresIdx;
    _n = scoresElems;
    _k = boxesElems ~/ scoresElems;
    _boxesOut = Float32List(boxesElems);
    _scoresOut = Float32List(scoresElems);

    _inputBuffer = Float32List(_inW * _inH * 3);
    _canvas = web.HTMLCanvasElement()
      ..width = _inW
      ..height = _inH;
    _ctx = _canvas!.getContext('2d') as web.CanvasRenderingContext2D;

    _initialized = true;
  }

  Future<void> dispose() async {
    _runner?.close();
    _runner = null;
    _activeAccelerator = null;
    _boxesOut = null;
    _scoresOut = null;
    _inputBuffer = null;
    _canvas = null;
    _ctx = null;
    _initialized = false;
  }

  /// Detects faces in a decoded image bitmap.
  ///
  /// Letterboxes the source image into the model's input space, runs
  /// inference, and decodes the SSD output via [generateAnchors] and
  /// [weightedNms] (from flutter_litert) before mapping back to original
  /// image coordinates.
  ///
  /// Returns detections in NORMALIZED image coordinates.
  Future<List<Detection>> detect(
    JSObject canvasSource, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (!_initialized) {
      throw StateError('FaceDetectionModelWeb not initialized.');
    }

    final lb = computeLetterboxParams(
      srcWidth: imageWidth,
      srcHeight: imageHeight,
      targetWidth: _inW,
      targetHeight: _inH,
    );

    final ctx = _ctx!;
    ctx.fillStyle = 'rgb(0,0,0)'.toJS;
    ctx.fillRect(0, 0, _inW, _inH);
    ctx.drawImage(
      canvasSource,
      0,
      0,
      imageWidth,
      imageHeight,
      lb.padLeft,
      lb.padTop,
      lb.newWidth,
      lb.newHeight,
    );

    final web.ImageData imageData = ctx.getImageData(0, 0, _inW, _inH);
    final rgba = imageData.data.toDart;
    final Float32List input = _inputBuffer!;
    rgbaToSignedRgbFloat32(Uint8List.view(rgba.buffer), input);

    await _runner!.run(
      <Float32List>[input],
      <int, Float32List>{_boxesIdx: _boxesOut!, _scoresIdx: _scoresOut!},
    );

    // Decode candidates (scores paired with their boxes; degenerate boxes
    // skipped safely). Pure Dart, unit-tested on the host VM.
    final List<DecodedCandidate> decoded = decodeBlazeFaceCandidates(
      scoresRaw: _scoresOut!,
      boxesRaw: _boxesOut!,
      anchors: _anchors,
      anchorCount: _n,
      valuesPerBox: _k,
      scale: _inH.toDouble(),
    );
    if (decoded.isEmpty) return const <Detection>[];

    // Sort by score and run weighted NMS in flutter_litert.
    final List<int> order = List<int>.generate(decoded.length, (i) => i)
      ..sort((a, b) => decoded[b].score.compareTo(decoded[a].score));

    final List<List<double>> sortedBoxes = <List<double>>[
      for (final i in order)
        [
          decoded[i].box.xmin,
          decoded[i].box.ymin,
          decoded[i].box.xmax,
          decoded[i].box.ymax,
        ],
    ];
    final List<double> sortedScores = <double>[
      for (final i in order) decoded[i].score,
    ];
    final List<List<double>> sortedKps = <List<double>>[
      for (final i in order) decoded[i].keypointsXY,
    ];

    final results = weightedNms(
      sortedBoxes,
      sortedScores,
      iouThres: kMinSuppressionThreshold,
      maxDet: 100,
    );

    // Letterbox-removal: undo padding/scaling to original-image normalized
    // coordinates.
    final double padTopNorm = lb.padTop / _inH;
    final double padBottomNorm = lb.padBottom / _inH;
    final double padLeftNorm = lb.padLeft / _inW;
    final double padRightNorm = lb.padRight / _inW;
    final double sx = 1.0 - (padLeftNorm + padRightNorm);
    final double sy = 1.0 - (padTopNorm + padBottomNorm);
    if (sx <= 0 || sy <= 0) return const <Detection>[];

    final List<Detection> dets = <Detection>[];
    for (final r in results) {
      final List<double> b = r.box;
      final double xmin = ((b[0] - padLeftNorm) / sx).clamp(0.0, 1.0);
      final double ymin = ((b[1] - padTopNorm) / sy).clamp(0.0, 1.0);
      final double xmax = ((b[2] - padLeftNorm) / sx).clamp(0.0, 1.0);
      final double ymax = ((b[3] - padTopNorm) / sy).clamp(0.0, 1.0);
      final List<double> srcKp = sortedKps[r.index];
      final List<double> kp = List<double>.filled(srcKp.length, 0);
      for (int j = 0; j < srcKp.length; j += 2) {
        kp[j] = ((srcKp[j] - padLeftNorm) / sx).clamp(0.0, 1.0);
        kp[j + 1] = ((srcKp[j + 1] - padTopNorm) / sy).clamp(0.0, 1.0);
      }
      dets.add(
        Detection(
          boundingBox: RectF(xmin, ymin, xmax, ymax),
          score: r.score,
          keypointsXY: kp,
          imageSize: Size(imageWidth.toDouble(), imageHeight.toDouble()),
        ),
      );
    }
    return dets;
  }

  /// The currently-active model variant.
  FaceDetectionModel get model => _model;

  /// Re-exposes [computeFaceAlignment] from `shared/face_geometry.dart` so the
  /// web detector can locate it as `FaceDetectionModelWeb.faceAlignment(...)`
  /// without a separate import.
  static ({double theta, double cx, double cy, double size}) faceAlignment(
    Detection det,
    double imgW,
    double imgH,
  ) => computeFaceAlignment(det, imgW, imgH);
}
