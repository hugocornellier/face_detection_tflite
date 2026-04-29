// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;
import 'package:web/web.dart' as web;

import '../../util/web_image_utils.dart';

/// 468-point face mesh runner for web. Uses LiteRT.js with auto WebGPU/WASM.
class FaceLandmarkModelWeb {
  LiteRtInterpreter? _liteRtItp;
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

  Future<void> initialize({String liteRtAccelerator = 'auto'}) async {
    if (_initialized) await dispose();
    const String assetPath =
        'packages/face_detection_tflite/assets/models/face_landmark.tflite';
    final ByteData raw = await rootBundle.load(assetPath);
    final bytes = raw.buffer.asUint8List();

    final String resolved =
        liteRtAccelerator == 'auto' ? 'webgpu' : liteRtAccelerator;
    _liteRtItp = await LiteRtInterpreter.fromBytes(
      bytes,
      accelerator: resolved,
    );

    final inT = _liteRtItp!.getInputTensor(0);
    _inH = inT.shape[1];
    _inW = inT.shape[2];

    // FaceMesh has multiple outputs: 468*3 mesh, 1 score, plus auxiliary
    // tensors. Locate by element count: mesh is the largest, score is 1.
    final outs = _liteRtItp!.getOutputTensors();
    int landmarksIdx = -1;
    int scoreIdx = -1;
    int landmarksLen = 0;
    for (int i = 0; i < outs.length; i++) {
      int n = 1;
      for (final d in outs[i].shape) {
        n *= d;
      }
      if (n == 1 && scoreIdx < 0) scoreIdx = i;
      if (n >= 468 * 3 && n > landmarksLen) {
        landmarksIdx = i;
        landmarksLen = n;
      }
    }
    if (landmarksIdx < 0 || scoreIdx < 0) {
      throw StateError(
        'Face landmark model outputs do not match expected shapes. Got '
        '${[for (final t in outs) t.shape]}',
      );
    }
    _landmarksIdx = landmarksIdx;
    _scoreIdx = scoreIdx;
    _landmarksLen = landmarksLen;
    _landmarksOut = Float32List(landmarksLen);
    _scoreOut = Float32List(1);
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
  Future<({Float32List landmarks, double score})> runOnCrop(
    web.ImageBitmap bitmap, {
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
    ctx.drawImage(bitmap, 0, 0);
    ctx.restore();

    final web.ImageData imageData = ctx.getImageData(0, 0, _inW, _inH);
    final rgba = imageData.data.toDart;
    final input = _inputBuffer!;
    rgbaToSignedRgbFloat32(Uint8List.view(rgba.buffer), input);

    await _liteRtItp!.runForMultipleInputs(<Object>[
      input
    ], <int, Object>{
      _landmarksIdx: _landmarksOut!,
      _scoreIdx: _scoreOut!,
    });

    return (
      landmarks: Float32List.fromList(_landmarksOut!),
      score: _scoreOut![0],
    );
  }

  int get landmarksLen => _landmarksLen;
}
