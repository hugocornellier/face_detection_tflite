// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import 'package:flutter_litert/flutter_litert.dart' show rgbaToSignedRgbFloat32;
import 'web_model_runner.dart';

/// Iris landmark runner for web. The model emits 76 points per eye (71 eye
/// mesh + 5 iris keypoints). Right-eye crops are mirrored before inference;
/// the detector flips the results back. Runs on the LiteRT.js interpreter or a
/// LiteRT Next `CompiledModel`.
class IrisLandmarkModelWeb {
  // iris_landmark.tflite input edge; used when the engine does not expose an
  // input shape (CompiledModel reports byte sizes only).
  static const int _kInputSize = 64;

  WebModelRunner? _runner;

  String? _activeAccelerator;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`),
  /// or null pre-init.
  String? get activeAccelerator => _runner != null ? _activeAccelerator : null;

  late int _inW;
  late int _inH;
  Float32List? _inputBuffer;
  web.HTMLCanvasElement? _canvas;
  web.CanvasRenderingContext2D? _ctx;
  // Output tensor indices found at init time by element count.
  final List<int> _outIndices = <int>[];
  final List<Float32List> _outBuffers = <Float32List>[];
  bool _initialized = false;

  bool get isInitialized => _initialized;
  int get inputWidth => _inW;
  int get inputHeight => _inH;

  Future<void> initialize({
    WebRunnerConfig config = const WebRunnerConfig(),
  }) async {
    if (_initialized) await dispose();
    const String assetPath =
        'packages/face_detection_tflite/assets/models/iris_landmark.tflite';
    final ByteData raw = await rootBundle.load(assetPath);
    final bytes = raw.buffer.asUint8List();
    final runner = await WebModelRunner.create(
      bytes,
      config: config,
      modelLabel: 'IrisLandmark',
    );
    _runner = runner;
    _activeAccelerator = runner.activeAccelerator;

    final List<int>? inShape = runner.inputShape0;
    _inH = inShape != null ? inShape[1] : _kInputSize;
    _inW = inShape != null ? inShape[2] : _kInputSize;

    _outIndices.clear();
    _outBuffers.clear();
    final List<int> counts = runner.outputElementCounts;
    for (int i = 0; i < counts.length; i++) {
      _outIndices.add(i);
      _outBuffers.add(Float32List(counts[i]));
    }

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
    _inputBuffer = null;
    _canvas = null;
    _ctx = null;
    _outIndices.clear();
    _outBuffers.clear();
    _initialized = false;
  }

  /// Runs iris detection on a single eye crop. Returns the concatenated
  /// landmark output in normalized [0, 1] coordinates (76 points * 3 = 228
  /// floats for the iris model).
  Future<Float32List> runOnEyeCrop(
    JSObject canvasSource, {
    required double cx,
    required double cy,
    required double size,
    required double theta,
    required bool isRight,
  }) async {
    if (!_initialized) {
      throw StateError('IrisLandmarkModelWeb not initialized.');
    }
    final ctx = _ctx!;
    ctx.save();
    ctx.fillStyle = 'rgb(0,0,0)'.toJS;
    ctx.fillRect(0, 0, _inW, _inH);
    final double scale = _inW / size;
    ctx.translate(_inW / 2.0, _inH / 2.0);
    if (isRight) {
      ctx.scale(-1.0, 1.0);
    }
    ctx.rotate(-theta);
    ctx.scale(scale, scale);
    ctx.translate(-cx, -cy);
    ctx.drawImage(canvasSource, 0, 0);
    ctx.restore();

    final web.ImageData imageData = ctx.getImageData(0, 0, _inW, _inH);
    final rgba = imageData.data.toDart;
    final input = _inputBuffer!;
    rgbaToSignedRgbFloat32(Uint8List.view(rgba.buffer), input);

    final Map<int, Float32List> outputs = <int, Float32List>{
      for (int i = 0; i < _outIndices.length; i++)
        _outIndices[i]: _outBuffers[i],
    };
    await _runner!.run(<Float32List>[input], outputs);

    // Concatenate all output tensors into a single flat buffer (mirrors the
    // native code that calls `_unpackLandmarks` on each output).
    int total = 0;
    for (final b in _outBuffers) {
      total += b.length;
    }
    final result = Float32List(total);
    int off = 0;
    for (final b in _outBuffers) {
      result.setRange(off, off + b.length, b);
      off += b.length;
    }
    return result;
  }
}
