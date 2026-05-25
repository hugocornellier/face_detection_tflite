// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart' show computeLetterboxParams;
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;
import 'package:web/web.dart' as web;

import '../../shared/face_model_config.dart' show segmentationModelFile;
import '../../shared/face_types.dart';
import '../../util/web_image_utils.dart';

/// Selfie segmentation runner for web. Loads `selfie_segmenter.tflite`,
/// `selfie_segmenter_landscape.tflite`, or `selfie_multiclass.tflite` via
/// LiteRT.js. Output is a per-pixel float mask, or per-class probabilities
/// for the multiclass model (post-softmax).
class SelfieSegmentationWeb {
  LiteRtInterpreter? _liteRtItp;

  String? _activeAccelerator;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`),
  /// or null pre-init.
  String? get activeAccelerator =>
      _liteRtItp != null ? _activeAccelerator : null;

  late int _inW;
  late int _inH;
  Float32List? _inputBuffer;
  Float32List? _outputBuffer;
  bool _isMulticlass = false;
  web.HTMLCanvasElement? _canvas;
  web.CanvasRenderingContext2D? _ctx;
  bool _initialized = false;
  SegmentationModel _model = SegmentationModel.general;

  bool get isInitialized => _initialized;
  int get inputWidth => _inW;
  int get inputHeight => _inH;
  SegmentationModel get model => _model;

  Future<void> initialize({
    SegmentationModel model = SegmentationModel.general,
    String liteRtAccelerator = 'auto',
  }) async {
    if (_initialized) await dispose();

    _model = model;
    _isMulticlass = model == SegmentationModel.multiclass;

    final String fileName = segmentationModelFile(model);
    final String assetPath =
        'packages/face_detection_tflite/assets/models/$fileName';
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

    final inT = _liteRtItp!.getInputTensor(0);
    _inH = inT.shape[1];
    _inW = inT.shape[2];

    final outs = _liteRtItp!.getOutputTensors();
    if (outs.isEmpty) {
      throw StateError('Segmentation model has no outputs.');
    }
    int n = 1;
    for (final d in outs[0].shape) {
      n *= d;
    }
    _outputBuffer = Float32List(n);
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
    _outputBuffer = null;
    _canvas = null;
    _ctx = null;
    _initialized = false;
  }

  /// Runs segmentation on a decoded image bitmap. Returns a [SegmentationMask]
  /// at model resolution with letterbox padding info so the caller can call
  /// [SegmentationMask.upsample] for a full-image mask.
  Future<SegmentationMask> segment(
    JSObject canvasSource, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (!_initialized) {
      throw const SegmentationException(
        SegmentationError.inferenceFailed,
        'Segmentation runner not initialized',
      );
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
    final input = _inputBuffer!;
    rgbaToSignedRgbFloat32(Uint8List.view(rgba.buffer), input);

    await _liteRtItp!.runForMultipleInputs(
      <Object>[input],
      <int, Object>{0: _outputBuffer!},
    );

    final padding = <double>[
      lb.padTop / _inH,
      lb.padBottom / _inH,
      lb.padLeft / _inW,
      lb.padRight / _inW,
    ];

    if (_isMulticlass) {
      final raw = _outputBuffer!;
      final int hw = _inW * _inH;
      final Float32List person = Float32List(hw);
      final Float32List classData = Float32List(hw * 6);
      for (int i = 0; i < hw; i++) {
        double maxV = raw[i * 6];
        for (int c = 1; c < 6; c++) {
          final v = raw[i * 6 + c];
          if (v > maxV) maxV = v;
        }
        double sum = 0;
        for (int c = 0; c < 6; c++) {
          final e = math.exp(raw[i * 6 + c] - maxV);
          classData[i * 6 + c] = e;
          sum += e;
        }
        if (sum > 0) {
          for (int c = 0; c < 6; c++) {
            classData[i * 6 + c] /= sum;
          }
        }
        person[i] = 1.0 - classData[i * 6];
      }
      return MulticlassSegmentationMask(
        data: person,
        width: _inW,
        height: _inH,
        originalWidth: imageWidth,
        originalHeight: imageHeight,
        padding: padding,
        classData: classData,
      );
    }

    final raw = _outputBuffer!;
    final Float32List data = Float32List.fromList(raw);
    return SegmentationMask(
      data: data,
      width: _inW,
      height: _inH,
      originalWidth: imageWidth,
      originalHeight: imageHeight,
      padding: padding,
    );
  }
}
