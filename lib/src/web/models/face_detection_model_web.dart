// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart' hide Detection;
import 'package:web/web.dart' as web;

import '../../shared/face_geometry.dart' show computeFaceAlignment;
import '../../shared/face_model_config.dart';
import '../../shared/face_types.dart';

/// Web BlazeFace runner. Auto-prefers WebGPU on the LiteRT.js path and
/// otherwise falls back to WASM SIMD.
class FaceDetectionModelWeb {
  LiteRtInterpreter? _liteRtItp;

  String? _activeAccelerator;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`),
  /// or null before [initialize] completes.
  String? get activeAccelerator =>
      _liteRtItp != null ? _activeAccelerator : null;

  Float32List? _boxesOut;
  Float32List? _scoresOut;
  late int _inW;
  late int _inH;
  late List<List<double>> _anchors;
  late List<int> _boxesShape;
  late List<int> _scoresShape;
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
    String liteRtAccelerator = 'auto',
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

    final String resolved = liteRtAccelerator == 'auto'
        ? 'webgpu'
        : liteRtAccelerator;
    _liteRtItp = await LiteRtInterpreter.fromBytes(
      bytes,
      accelerator: resolved,
    );
    _activeAccelerator = resolved;

    final outs = _liteRtItp!.getOutputTensors();
    int boxesIdx = -1;
    int scoresIdx = -1;
    int boxesElems = 0;
    int scoresElems = 0;
    List<int>? boxesShape;
    List<int>? scoresShape;
    for (int i = 0; i < outs.length; i++) {
      final shape = List<int>.from(outs[i].shape);
      int n = 1;
      for (final d in shape) {
        n *= d;
      }
      // BlazeFace: boxes tensor has shape [1, N, K] where K >= 16
      // (xc,yc,w,h + 6 keypoints * 2). Scores shape is [1, N] or [1, N, 1].
      final last = shape.last;
      if (last >= 16 && last % 2 == 0) {
        boxesIdx = i;
        boxesElems = n;
        boxesShape = shape;
      } else if (last == 1 || shape.length == 2) {
        scoresIdx = i;
        scoresElems = n;
        scoresShape = shape;
      }
    }
    if (boxesIdx < 0 ||
        scoresIdx < 0 ||
        boxesShape == null ||
        scoresShape == null) {
      throw StateError(
        'BlazeFace outputs do not match expected shapes. Got '
        '${[for (final t in outs) t.shape]}',
      );
    }
    _boxesIdx = boxesIdx;
    _scoresIdx = scoresIdx;
    _boxesShape = boxesShape;
    _scoresShape = scoresShape;
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
    _liteRtItp?.close();
    _liteRtItp = null;
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

    await _liteRtItp!.runForMultipleInputs(
      <Object>[input],
      <int, Object>{_boxesIdx: _boxesOut!, _scoresIdx: _scoresOut!},
    );

    // Decode candidate scores.
    final raw = _scoresOut!;
    final int n = _scoresShape[1];
    final List<int> candIndices = <int>[];
    final List<double> candScores = <double>[];
    for (int i = 0; i < n; i++) {
      final double s = sigmoidClipped(raw[i], limit: kRawScoreLimit);
      if (s >= kMinScore) {
        candIndices.add(i);
        candScores.add(s);
      }
    }
    if (candIndices.isEmpty) return const <Detection>[];

    // Decode bounding boxes for those indices.
    final Float32List boxesRaw = _boxesOut!;
    final int k = _boxesShape[2];
    final double scale = _inH.toDouble();
    final List<({RectF box, List<double> kp})> decoded =
        <({RectF box, List<double> kp})>[];
    final Float32List tmp = Float32List(k);
    for (final int i in candIndices) {
      final int base = i * k;
      for (int j = 0; j < k; j++) {
        tmp[j] = boxesRaw[base + j] / scale;
      }
      final double ax = _anchors[i][0];
      final double ay = _anchors[i][1];
      tmp[0] += ax;
      tmp[1] += ay;
      for (int j = 4; j < k; j += 2) {
        tmp[j] += ax;
        tmp[j + 1] += ay;
      }
      final double xc = tmp[0], yc = tmp[1], w = tmp[2], h = tmp[3];
      if (w <= 0 || h <= 0) continue;
      final double xmin = xc - w * 0.5;
      final double ymin = yc - h * 0.5;
      final double xmax = xc + w * 0.5;
      final double ymax = yc + h * 0.5;
      final List<double> kp = <double>[];
      for (int j = 4; j < k; j += 2) {
        kp.add(tmp[j]);
        kp.add(tmp[j + 1]);
      }
      decoded.add((box: RectF(xmin, ymin, xmax, ymax), kp: kp));
    }
    if (decoded.isEmpty) return const <Detection>[];

    // Sort by score and run weighted NMS in flutter_litert.
    final List<int> order = List<int>.generate(decoded.length, (i) => i)
      ..sort((a, b) => candScores[b].compareTo(candScores[a]));

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
      for (final i in order) candScores[i],
    ];
    final List<List<double>> sortedKps = <List<double>>[
      for (final i in order) decoded[i].kp,
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
