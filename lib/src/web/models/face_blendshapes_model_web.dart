// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;

import '../../shared/blendshape_input.dart'
    show kBlendshapeInputFloats, kBlendshapeCount;

/// Face blendshapes runner for web. Consumes a packed `[1, 146, 2]` landmark
/// tensor (292 floats) and returns 52 expression coefficients in `[0, 1]`.
///
/// Pinned to the WASM backend regardless of the requested accelerator: a
/// 292-in / 52-out MLP is far below the WebGPU compile/dispatch payoff, and
/// WASM sidesteps GPU tensor-format questions for a non-image input (CPU-pinning
/// parity with the native path).
class FaceBlendshapesModelWeb {
  LiteRtInterpreter? _liteRtItp;
  String? _activeAccelerator;
  Float32List? _outBuffer;
  bool _initialized = false;

  bool get isInitialized => _initialized;

  /// The accelerator that compiled this model (always `'wasm'`), or null
  /// pre-init.
  String? get activeAccelerator =>
      _liteRtItp != null ? _activeAccelerator : null;

  Future<void> initialize({String liteRtAccelerator = 'auto'}) async {
    if (_initialized) await dispose();
    const String assetPath =
        'packages/face_detection_tflite/assets/models/face_blendshapes.tflite';
    final ByteData raw = await rootBundle.load(assetPath);
    final bytes = raw.buffer.asUint8List();
    // Always WASM (see class doc); the requested accelerator is ignored.
    _liteRtItp = await LiteRtInterpreter.fromBytes(
      bytes,
      accelerator: 'wasm',
    );
    _activeAccelerator = _liteRtItp!.activeAccelerator;

    final int inCount = _elementCount(_liteRtItp!.getInputTensor(0).shape);
    final outs = _liteRtItp!.getOutputTensors();
    final int outCount = outs.isEmpty ? 0 : _elementCount(outs[0].shape);
    if (inCount != kBlendshapeInputFloats || outCount != kBlendshapeCount) {
      await dispose();
      throw UnsupportedError(
        'face_blendshapes.tflite expects $kBlendshapeInputFloats-in / '
        '$kBlendshapeCount-out; got $inCount-in / $outCount-out.',
      );
    }

    _outBuffer = Float32List(kBlendshapeCount);
    _initialized = true;
  }

  static int _elementCount(List<int> shape) {
    int n = 1;
    for (final int d in shape) {
      n *= d;
    }
    return n;
  }

  Future<void> dispose() async {
    _liteRtItp?.close();
    _liteRtItp = null;
    _activeAccelerator = null;
    _outBuffer = null;
    _initialized = false;
  }

  /// Runs the model on a packed 292-float landmark tensor. Returns the 52
  /// coefficients clamped to `[0, 1]`, or null when the input is the wrong size
  /// or the model produced a NaN (matching the native runner's behavior).
  Future<Float32List?> run(Float32List packed) async {
    if (!_initialized) {
      throw StateError('FaceBlendshapesModelWeb not initialized.');
    }
    if (packed.length != kBlendshapeInputFloats) return null;

    final Float32List out = _outBuffer!;
    await _liteRtItp!.runForMultipleInputs(
      <Object>[packed],
      <int, Object>{0: out},
    );

    final Float32List result = Float32List(kBlendshapeCount);
    for (int i = 0; i < kBlendshapeCount; i++) {
      final double v = out[i];
      if (v.isNaN) return null;
      result[i] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
    return result;
  }
}
