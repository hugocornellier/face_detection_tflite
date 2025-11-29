part of face_detection_tflite;

/// Estimates dense iris keypoints within cropped eye regions and lets callers
/// derive a robust iris center (with fallback if inference fails).
class IrisLandmark {
  IsolateInterpreter? _iso;
  final Interpreter _itp;
  final int _inW, _inH;
  late final Tensor _inputTensor;
  late final Float32List _inputBuf;
  late final Map<int, List<int>> _outShapes;
  late final Map<int, Float32List> _outBuffers;

  IrisLandmark._(this._itp, this._inW, this._inH);

  static Future<IrisLandmark> create({
    InterpreterOptions? options,
    bool useIsolate = true
  }) async {
    final Interpreter itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$_irisLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final IrisLandmark obj = IrisLandmark._(itp, inW, inH);

    obj._inputTensor = itp.getInputTensor(0);
    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();

    final Map<int, List<int>> shapes = <int, List<int>>{};
    final Map<int, Float32List> buffers = <int, Float32List>{};
    for (int i = 0;; i++) {
      try {
        final Tensor t = itp.getOutputTensor(i);
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

  static Future<IrisLandmark> createFromFile(
    String modelPath, {
    InterpreterOptions? options,
    bool useIsolate = true
  }) async {
    final Interpreter itp = await Interpreter.fromFile(
      File(modelPath),
      options: options ?? InterpreterOptions(),
    );
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final IrisLandmark obj = IrisLandmark._(itp, inW, inH);

    obj._inputTensor = itp.getInputTensor(0);
    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();

    final Map<int, List<int>> shapes = <int, List<int>>{};
    final Map<int, Float32List> buffers = <int, Float32List>{};
    for (int i = 0;; i++) {
      try {
        final Tensor t = itp.getOutputTensor(i);
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
          growable: false
        ),
        growable: false
      ),
      growable: false,
    );
    int k = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final List<double> px = out[0][y][x];
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
    final List<double> out = <double>[];
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
    final _ImageTensor pack = await _imageToTensor(
      eyeCrop,
      outW: _inW,
      outH: _inH
    );

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();

      final List<List<double>> lm = <List<double>>[];
      for (final Float32List flat in _outBuffers.values) {
        lm.addAll(
            _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false)
        );
      }
      return lm;
    } else {
      final List<List<List<List<double>>>> input4d = _asNHWC4D(pack.tensorNHWC, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [input4d];
      final Map<int, Object> outputs = <int, Object>{};
      _outShapes.forEach((i, shape) {
        outputs[i] = _allocForShape(shape);
      });

      await _iso!.runForMultipleInputs(inputs, outputs);

      final List<List<double>> lm = <List<double>>[];
      _outShapes.forEach((i, _) {
        final Float32List flat = _flattenDynamicToFloat(outputs[i]);
        lm.addAll(
          _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false)
        );
      });
      return lm;
    }
  }

  Future<List<List<double>>> runOnImage(img.Image src, _RectF eyeRoi) async {
    final img.Image eyeCrop = await cropFromRoi(src, eyeRoi);
    final List<List<double>> lmNorm = await call(eyeCrop);
    final double imgW = src.width.toDouble();
    final double imgH = src.height.toDouble();
    final double dx = eyeRoi.xmin * imgW;
    final double dy = eyeRoi.ymin * imgH;
    final double sx = eyeRoi.w * imgW;
    final double sy = eyeRoi.h * imgH;

    final List<List<double>> mapped = <List<double>>[];
    for (final List<double> p in lmNorm) {
      final double x = dx + p[0] * sx;
      final double y = dy + p[1] * sy;
      mapped.add([x, y, p[2]]);
    }
    return mapped;
  }

  static Future<List<List<double>>> callWithIsolate(
    Uint8List eyeCropBytes, String modelPath, {
    bool irisOnly = false
  }) async {
    final ReceivePort rp = ReceivePort();
    final Isolate iso = await Isolate.spawn(IrisLandmark._isolateEntry, {
      'sendPort': rp.sendPort,
      'modelPath': modelPath,
      'eyeCropBytes': eyeCropBytes,
      'mode': irisOnly ? 'irisOnly' : 'full',
    });
    final Map<dynamic, dynamic> msg = await rp.first as Map;
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
      final IrisLandmark iris = await IrisLandmark.createFromFile(
        modelPath,
        useIsolate: false
      );
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
    final _ImageTensor pack = await _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

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
      final double sx = 1.0 - (pl + pr);
      final double sy = 1.0 - (pt + pb);

      final Float32List flat = irisFlat!;
      final List<List<double>> lm = <List<double>>[];
      for (int i = 0; i < 5; i++) {
        double x = flat[i * 3 + 0] / _inW;
        double y = flat[i * 3 + 1] / _inH;
        final double z = flat[i * 3 + 2];
        x = (x - pl) / sx;
        y = (y - pt) / sy;
        lm.add([x, y, z]);
      }
      return lm;
    } else {
      final List<List<List<List<double>>>> input4d = _asNHWC4D(pack.tensorNHWC, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [input4d];
      final Map<int, Object> outputs = <int, Object>{};
      _outShapes.forEach((i, shape) {
        outputs[i] = _allocForShape(shape);
      });

      await _iso!.runForMultipleInputs(inputs, outputs);

      final double pt = pack.padding[0],
          pb = pack.padding[1],
          pl = pack.padding[2],
          pr = pack.padding[3];
      final double sx = 1.0 - (pl + pr);
      final double sy = 1.0 - (pt + pb);

      Float32List? irisFlat;
      _outShapes.forEach((i, shape) {
        final Float32List flat = _flattenDynamicToFloat(outputs[i]);
        if (flat.length == 15) {
          irisFlat = flat;
        }
      });
      if (irisFlat == null) {
        return const <List<double>>[];
      }

      final Float32List flat = irisFlat!;
      final List<List<double>> lm = <List<double>>[];
      for (int i = 0; i < 5; i++) {
        double x = flat[i * 3 + 0] / _inW;
        double y = flat[i * 3 + 1] / _inH;
        final double z = flat[i * 3 + 2];
        x = (x - pl) / sx;
        y = (y - pt) / sy;
        lm.add([x, y, z]);
      }
      return lm;
    }
  }

  Future<List<List<double>>> runOnImageAlignedIris(
    img.Image src, _AlignedRoi roi, {
    bool isRight = false
  }) async {
    final img.Image crop = await extractAlignedSquare(
      src,
      roi.cx,
      roi.cy,
      roi.size,
      roi.theta
    );
    final img.Image eye = isRight ? await _flipHorizontal(crop) : crop;
    final double ct = math.cos(roi.theta);
    final double st = math.sin(roi.theta);
    final double s = roi.size;

    final List<List<double>> out = <List<double>>[];
    final List<List<double>> lmNorm = await callIrisOnly(eye);
    for (final List<double> p in lmNorm) {
      final double px = isRight ? (1.0 - p[0]) : p[0];
      final double py = p[1];
      final double lx2 = (px - 0.5) * s;
      final double ly2 = (py - 0.5) * s;
      final double x = roi.cx + lx2 * ct - ly2 * st;
      final double y = roi.cy + lx2 * st + ly2 * ct;
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
