// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import 'package:flutter_litert/flutter_litert.dart' show rgbaToSignedRgbFloat32;
import 'web_model_runner.dart';

/// 468-point face mesh runner for web. Runs on the LiteRT.js interpreter or a
/// LiteRT Next `CompiledModel`, with auto WebGPU/WASM.
class FaceLandmarkModelWeb {
  // face_landmark.tflite input edge; used when the engine does not expose an
  // input shape (CompiledModel reports byte sizes only).
  static const int _kInputSize = 192;

  WebModelRunner? _runner;

  String? _activeAccelerator;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`),
  /// or null pre-init.
  String? get activeAccelerator => _runner != null ? _activeAccelerator : null;

  Float32List? _landmarksOut;
  Float32List? _scoreOut;
  int _landmarksIdx = 0;
  int _scoreIdx = 1;
  int _landmarksLen = 0;
  late int _inW;
  late int _inH;
  Float32List? _inputBuffer;
  web.HTMLCanvasElement? _canvas;
  web.CanvasRenderingContext2D? _ctx;
  bool _initialized = false;

  bool get isInitialized => _initialized;
  int get inputWidth => _inW;
  int get inputHeight => _inH;

  Future<void> initialize({
    WebRunnerConfig config = const WebRunnerConfig(),
  }) async {
    if (_initialized) await dispose();
    const String assetPath =
        'packages/face_detection_tflite/assets/models/face_landmark.tflite';
    final ByteData raw = await rootBundle.load(assetPath);
    final bytes = raw.buffer.asUint8List();

    final runner = await WebModelRunner.create(
      bytes,
      config: config,
      modelLabel: 'FaceMesh',
    );
    _runner = runner;
    _activeAccelerator = runner.activeAccelerator;

    final List<int>? inShape = runner.inputShape0;
    _inH = inShape != null ? inShape[1] : _kInputSize;
    _inW = inShape != null ? inShape[2] : _kInputSize;

    // FaceMesh has multiple outputs: 468*3 mesh, 1 score, plus auxiliary
    // tensors. Locate by element count: mesh is the largest, score is 1.
    final List<int> counts = runner.outputElementCounts;
    int landmarksIdx = -1;
    int scoreIdx = -1;
    int landmarksLen = 0;
    for (int i = 0; i < counts.length; i++) {
      final int n = counts[i];
      if (n == 1 && scoreIdx < 0) scoreIdx = i;
      if (n >= 468 * 3 && n > landmarksLen) {
        landmarksIdx = i;
        landmarksLen = n;
      }
    }
    // Only the landmark output is required. The face-presence score output is
    // optional (mirrors native, where `_scoreIdx == -1` yields a null score),
    // so a mesh-only model still works instead of failing to initialize.
    if (landmarksIdx < 0) {
      throw StateError(
        'Face landmark model outputs do not match expected shapes. '
        'Output element counts: $counts',
      );
    }
    _landmarksIdx = landmarksIdx;
    _scoreIdx = scoreIdx;
    _landmarksLen = landmarksLen;
    _landmarksOut = Float32List(landmarksLen);
    _scoreOut = scoreIdx < 0 ? null : Float32List(1);
    _inputBuffer = Float32List(_inH * _inW * 3);

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
    _landmarksOut = null;
    _scoreOut = null;
    _inputBuffer = null;
    _canvas = null;
    _ctx = null;
    _initialized = false;
  }

  /// Runs the mesh model on a square aligned face crop. Returns a flat list of
  /// 468 (x, y, z) triples in pixel space of the input crop (NOT normalized).
  ///
  /// [bitmap] should be the source image. [cx], [cy], [size], [theta] specify
  /// the rotation-aware crop. Output is the model's tensor as Float32List.
  Future<({Float32List landmarks, double? score})> runOnCrop(
    JSObject canvasSource, {
    required double cx,
    required double cy,
    required double size,
    required double theta,
  }) async {
    if (!_initialized) {
      throw StateError('FaceLandmarkModelWeb not initialized.');
    }
    final ctx = _ctx!;

    // Aligned crop via canvas transform: translate to model center, rotate by
    // -theta, scale, then draw the bitmap with origin offset at (cx, cy).
    ctx.save();
    ctx.fillStyle = 'rgb(0,0,0)'.toJS;
    ctx.fillRect(0, 0, _inW, _inH);
    final double scale = _inW / size;
    ctx.translate(_inW / 2.0, _inH / 2.0);
    ctx.rotate(-theta);
    ctx.scale(scale, scale);
    ctx.translate(-cx, -cy);
    ctx.drawImage(canvasSource, 0, 0);
    ctx.restore();

    final web.ImageData imageData = ctx.getImageData(0, 0, _inW, _inH);
    final rgba = imageData.data.toDart;
    final input = _inputBuffer!;
    rgbaToSignedRgbFloat32(Uint8List.view(rgba.buffer), input);

    final scoreOut = _scoreOut;
    await _runner!.run(
      <Float32List>[input],
      <int, Float32List>{_landmarksIdx: _landmarksOut!, _scoreIdx: ?scoreOut},
    );

    return (
      landmarks: Float32List.fromList(_landmarksOut!),
      score: scoreOut == null ? null : _sigmoidClipped(scoreOut[0]),
    );
  }

  // Comparison-based clip to +/-80 then logistic, matching the native
  // `sigmoidClipped` exactly (including NaN: `clip` propagates NaN, unlike
  // `num.clamp` which would coerce NaN to the upper bound).
  static double _sigmoidClipped(double x) {
    final double v = x < -80.0 ? -80.0 : (x > 80.0 ? 80.0 : x);
    return 1.0 / (1.0 + math.exp(-v));
  }

  int get landmarksLen => _landmarksLen;
}
