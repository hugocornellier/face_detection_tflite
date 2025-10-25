part of face_detection_tflite;

class IrisLandmark {
  final Interpreter _itp;
  final int _inW, _inH;

  IsolateInterpreter? _iso;

  late final Tensor _inputTensor;
  late final Float32List _inputBuf;

  late final Map<int, List<int>> _outShapes;
  late final Map<int, Float32List> _outBuffers;

  IrisLandmark._(this._itp, this._inW, this._inH);

  static Future<IrisLandmark> create(
    {InterpreterOptions? options, bool useIsolate = true}
  ) async {
    final itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$_irisLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final ishape = itp.getInputTensor(0).shape;
    final inH = ishape[1];
    final inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final obj = IrisLandmark._(itp, inW, inH);

    obj._inputTensor = itp.getInputTensor(0);
    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();

    final shapes = <int, List<int>>{};
    final buffers = <int, Float32List>{};
    for (var i = 0;; i++) {
      try {
        final t = itp.getOutputTensor(i);
        shapes[i] = t.shape;
        buffers[i] = t.data.buffer.asFloat32List();
      } catch (_) {
        break;
      }
    }
    obj._outShapes = shapes;
    obj._outBuffers = buffers;

    if (useIsolate) {
      obj._iso = await IsolateInterpreter.create(address: itp.address);
    }

    return obj;
  }

  static Future<IrisLandmark> createFromFile(String modelPath,
      {InterpreterOptions? options, bool useIsolate = true}) async {
    final itp = await Interpreter.fromFile(
      File(modelPath),
      options: options ?? InterpreterOptions(),
    );
    final ishape = itp.getInputTensor(0).shape;
    final inH = ishape[1];
    final inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final obj = IrisLandmark._(itp, inW, inH);

    obj._inputTensor = itp.getInputTensor(0);
    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();

    final shapes = <int, List<int>>{};
    final buffers = <int, Float32List>{};
    for (var i = 0;; i++) {
      try {
        final t = itp.getOutputTensor(i);
        shapes[i] = t.shape;
        buffers[i] = t.data.buffer.asFloat32List();
      } catch (_) {
        break;
      }
    }
    obj._outShapes = shapes;
    obj._outBuffers = buffers;

    if (useIsolate) {
      obj._iso = await IsolateInterpreter.create(address: itp.address);
    }

    return obj;
  }

  List<List<List<List<double>>>> _asNHWC4D(Float32List flat, int h, int w) {
    final out = List<List<List<List<double>>>>.filled(
      1,
      List.generate(
          h,
              (_) => List.generate(
              w, (_) => List<double>.filled(3, 0.0, growable: false),
              growable: false),
          growable: false),
      growable: false,
    );
    var k = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final px = out[0][y][x];
        px[0] = flat[k++];
        px[1] = flat[k++];
        px[2] = flat[k++];
      }
    }
    return out;
  }

  Object _allocForShape(List<int> shape) {
    if (shape.isEmpty) return <double>[];
    Object build(List<int> s, int d) {
      if (d == s.length - 1) {
        return List<double>.filled(s[d], 0.0, growable: false);
      }
      return List.generate(s[d], (_) => build(s, d + 1), growable: false);
    }

    return build(shape, 0);
  }

  Float32List _flattenDynamicToFloat(dynamic x) {
    final out = <double>[];
    void walk(dynamic v) {
      if (v is num) {
        out.add(v.toDouble());
      } else if (v is List) {
        for (final e in v) {
          walk(e);
        }
      } else {
        throw StateError('Unexpected type');
      }
    }

    walk(x);
    return Float32List.fromList(out);
  }

  Future<List<List<double>>> call(img.Image eyeCrop) async {
    final pack = await _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();

      final lm = <List<double>>[];
      for (final flat in _outBuffers.values) {
        lm.addAll(
            _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false));
      }
      return lm;
    } else {
      final input4d = _asNHWC4D(pack.tensorNHWC, _inH, _inW);
      final inputs = [input4d];
      final outputs = <int, Object>{};
      _outShapes.forEach((i, shape) {
        outputs[i] = _allocForShape(shape);
      });

      await _iso!.runForMultipleInputs(inputs, outputs);

      final lm = <List<double>>[];
      _outShapes.forEach((i, _) {
        final flat = _flattenDynamicToFloat(outputs[i]);
        lm.addAll(
            _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false));
      });
      return lm;
    }
  }

  Future<List<List<double>>> runOnImage(img.Image src, RectF eyeRoi) async {
    final eyeCrop = await cropFromRoi(src, eyeRoi);
    final lmNorm = await call(eyeCrop);
    final imgW = src.width.toDouble();
    final imgH = src.height.toDouble();
    final dx = eyeRoi.xmin * imgW;
    final dy = eyeRoi.ymin * imgH;
    final sx = eyeRoi.w * imgW;
    final sy = eyeRoi.h * imgH;
    final mapped = <List<double>>[];
    for (final p in lmNorm) {
      final x = dx + p[0] * sx;
      final y = dy + p[1] * sy;
      mapped.add([x, y, p[2]]);
    }
    return mapped;
  }

  static Future<List<List<double>>> callWithIsolate(
      Uint8List eyeCropBytes, String modelPath,
      {bool irisOnly = false}) async {
    final rp = ReceivePort();
    final params = {
      'sendPort': rp.sendPort,
      'modelPath': modelPath,
      'eyeCropBytes': eyeCropBytes,
      'mode': irisOnly ? 'irisOnly' : 'full',
    };
    final iso = await Isolate.spawn(IrisLandmark._isolateEntry, params);
    final msg = await rp.first as Map;
    rp.close();
    iso.kill(priority: Isolate.immediate);
    if (msg['ok'] == true) {
      final List pts = msg['points'] as List;
      return pts
          .map<List<double>>(
              (e) => (e as List).map((n) => (n as num).toDouble()).toList())
          .toList();
    } else {
      throw StateError(msg['err'] as String);
    }
  }

  @pragma('vm:entry-point')
  static Future<void> _isolateEntry(Map<String, dynamic> params) async {
    final SendPort sendPort = params['sendPort'] as SendPort;
    final String modelPath = params['modelPath'] as String;
    final Uint8List eyeCropBytes = params['eyeCropBytes'] as Uint8List;
    final String mode = params['mode'] as String;

    try {
      final iris =
      await IrisLandmark.createFromFile(modelPath, useIsolate: false);
      final img.Image? eye = img.decodeImage(eyeCropBytes);
      if (eye == null) {
        sendPort.send({'ok': false, 'err': 'decode_failed'});
        return;
      }
      final List<List<double>> res = mode == 'irisOnly'
          ? await iris.callIrisOnly(eye)
          : await iris.call(eye);
      iris.dispose();
      sendPort.send({'ok': true, 'points': res});
    } catch (e) {
      sendPort.send({'ok': false, 'err': e.toString()});
    }
  }

  Future<List<List<double>>> callIrisOnly(img.Image eyeCrop) async {
    final pack = await _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();

      Float32List? irisFlat;
      _outBuffers.forEach((_, buf) {
        if (buf.length == 15) {
          irisFlat = buf;
        }
      });
      if (irisFlat == null) {
        return const <List<double>>[];
      }

      final pt = pack.padding[0],
          pb = pack.padding[1],
          pl = pack.padding[2],
          pr = pack.padding[3];
      final sx = 1.0 - (pl + pr);
      final sy = 1.0 - (pt + pb);

      final flat = irisFlat!;
      final lm = <List<double>>[];
      for (var i = 0; i < 5; i++) {
        var x = flat[i * 3 + 0] / _inW;
        var y = flat[i * 3 + 1] / _inH;
        final z = flat[i * 3 + 2];
        x = (x - pl) / sx;
        y = (y - pt) / sy;
        lm.add([x, y, z]);
      }
      return lm;
    } else {
      final input4d = _asNHWC4D(pack.tensorNHWC, _inH, _inW);
      final inputs = [input4d];
      final outputs = <int, Object>{};
      _outShapes.forEach((i, shape) {
        outputs[i] = _allocForShape(shape);
      });

      await _iso!.runForMultipleInputs(inputs, outputs);

      final pt = pack.padding[0],
          pb = pack.padding[1],
          pl = pack.padding[2],
          pr = pack.padding[3];
      final sx = 1.0 - (pl + pr);
      final sy = 1.0 - (pt + pb);

      Float32List? irisFlat;
      _outShapes.forEach((i, shape) {
        final flat = _flattenDynamicToFloat(outputs[i]);
        if (flat.length == 15) {
          irisFlat = flat;
        }
      });
      if (irisFlat == null) {
        return const <List<double>>[];
      }

      final flat = irisFlat!;
      final lm = <List<double>>[];
      for (var i = 0; i < 5; i++) {
        var x = flat[i * 3 + 0] / _inW;
        var y = flat[i * 3 + 1] / _inH;
        final z = flat[i * 3 + 2];
        x = (x - pl) / sx;
        y = (y - pt) / sy;
        lm.add([x, y, z]);
      }
      return lm;
    }
  }

  Future<List<List<double>>> runOnImageAlignedIris(
    img.Image src, AlignedRoi roi,
    { bool isRight = false }
  ) async {
    final crop = await extractAlignedSquare(src, roi.cx, roi.cy, roi.size, roi.theta);
    final eye = isRight ? await _flipHorizontal(crop) : crop;
    final lmNorm = await callIrisOnly(eye);
    final ct = math.cos(roi.theta);
    final st = math.sin(roi.theta);
    final s = roi.size;
    final out = <List<double>>[];
    for (final p in lmNorm) {
      final px = isRight ? (1.0 - p[0]) : p[0];
      final py = p[1];
      final lx2 = (px - 0.5) * s;
      final ly2 = (py - 0.5) * s;
      final x = roi.cx + lx2 * ct - ly2 * st;
      final y = roi.cy + lx2 * st + ly2 * ct;
      out.add([x, y, p[2]]);
    }
    return out;
  }

  void dispose() {
    _iso?.close();
    _itp.close();
  }
}

extension on IrisLandmark {
  Future<img.Image> _flipHorizontal(img.Image src) async {
    return img.flipHorizontal(src);
  }
}

