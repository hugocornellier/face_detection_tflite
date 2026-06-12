// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;
import 'package:web/web.dart' as web;

import '../../shared/model_bytes_loader.dart';
import '../../util/web_image_utils.dart';

/// Iris landmark runner for web. The model emits 76 points per eye (71 eye
/// mesh + 5 iris keypoints). Right-eye crops are mirrored before inference;
/// the detector flips the results back.
class IrisLandmarkModelWeb {
  LiteRtInterpreter? _liteRtItp;

  String? _activeAccelerator;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`),
  /// or null pre-init.
  String? get activeAccelerator =>
      _liteRtItp != null ? _activeAccelerator : null;

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
    String liteRtAccelerator = 'auto',
    ModelBytesLoader? loadModelBytes,
  }) async {
    if (_initialized) await dispose();
    const String fileName = 'iris_landmark.tflite';
    final Uint8List bytes;
    if (loadModelBytes != null) {
      bytes = await loadModelBytes(fileName);
    } else {
      final ByteData raw = await rootBundle.load(
        'packages/face_detection_tflite/assets/models/$fileName',
      );
      bytes = raw.buffer.asUint8List();
    }
    final String resolved = liteRtAccelerator == 'auto'
        ? 'webgpu'
        : liteRtAccelerator;
    _liteRtItp = await LiteRtInterpreter.fromBytes(
      bytes,
      accelerator: resolved,
    );
    _activeAccelerator = resolved;

    final inT = _liteRtItp!.getInputTensor(0);
    _inH = inT.shape[1];
    _inW = inT.shape[2];

    _outIndices.clear();
    _outBuffers.clear();
    final outs = _liteRtItp!.getOutputTensors();
    for (int i = 0; i < outs.length; i++) {
      int n = 1;
      for (final d in outs[i].shape) {
        n *= d;
      }
      _outIndices.add(i);
      _outBuffers.add(Float32List(n));
    }

    _inputBuffer = Float32List(_inH * _inW * 3);
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

    final Map<int, Object> outputs = <int, Object>{
      for (int i = 0; i < _outIndices.length; i++)
        _outIndices[i]: _outBuffers[i],
    };
    await _liteRtItp!.runForMultipleInputs(<Object>[input], outputs);

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
