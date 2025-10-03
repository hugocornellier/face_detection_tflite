part of face_detection_tflite;

class FaceLandmark {
  final Interpreter _itp;
  final int _inW, _inH;

  IsolateInterpreter? _iso;

  late final int _bestIdx;
  late final Tensor _inputTensor;
  late final Tensor _bestTensor;

  late final Float32List _inputBuf;
  late final Float32List _bestOutBuf;

  late final List<List<int>> _outShapes;

  FaceLandmark._(this._itp, this._inW, this._inH);

  static Future<FaceLandmark> create({InterpreterOptions? options, bool useIsolate = true}) async {
    final itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$_faceLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final ishape = itp.getInputTensor(0).shape;
    final inH = ishape[1];
    final inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final obj = FaceLandmark._(itp, inW, inH);

    obj._inputTensor = itp.getInputTensor(0);

    int numElements(List<int> s) => s.fold(1, (a, b) => a * b);

    final shapes = <int, List<int>>{};
    for (var i = 0;; i++) {
      try {
        final s = itp.getOutputTensor(i).shape;
        shapes[i] = s;
      } catch (_) {
        break;
      }
    }

    int bestIdx = -1;
    int bestLen = -1;
    for (final e in shapes.entries) {
      final len = numElements(e.value);
      if (len > bestLen && len % 3 == 0) {
        bestLen = len;
        bestIdx = e.key;
      }
    }
    obj._bestIdx = bestIdx;

    obj._bestTensor = itp.getOutputTensor(obj._bestIdx);

    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();
    obj._bestOutBuf = obj._bestTensor.data.buffer.asFloat32List();

    final maxIndex = shapes.keys.isEmpty ? -1 : shapes.keys.reduce((a, b) => a > b ? a : b);
    obj._outShapes = List<List<int>>.generate(maxIndex + 1, (i) => shapes[i] ?? const <int>[]);

    if (useIsolate) {
      obj._iso = await IsolateInterpreter.create(address: itp.address);
    }

    return obj;
  }

  List<List<List<List<double>>>> _asNHWC4D(Float32List flat, int h, int w) {
    final out = List<List<List<List<double>>>>.filled(
      1,
      List.generate(h, (_) => List.generate(w, (_) => List<double>.filled(3, 0.0, growable: false), growable: false), growable: false),
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
    Object build(List<int> s, int depth) {
      if (depth == s.length - 1) {
        return List<double>.filled(s[depth], 0.0, growable: false);
      }
      return List.generate(s[depth], (_) => build(s, depth + 1), growable: false);
    }
    return build(shape, 0);
  }

  Future<List<List<double>>> call(img.Image faceCrop) async {
    final pack = await _imageToTensor(faceCrop, outW: _inW, outH: _inH);

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();
      return _unpackLandmarks(_bestOutBuf, _inW, _inH, pack.padding, clamp: true);
    } else {
      final input4d = _asNHWC4D(pack.tensorNHWC, _inH, _inW);
      final inputs = [input4d];
      final outputs = <int, Object>{};
      for (var i = 0; i < _outShapes.length; i++) {
        final s = _outShapes[i];
        if (s.isNotEmpty) {
          outputs[i] = _allocForShape(s);
        }
      }
      await _iso!.runForMultipleInputs(inputs, outputs);

      final dynamic best = outputs[_bestIdx];

      final flat = <double>[];
      void walk(dynamic x) {
        if (x is num) {
          flat.add(x.toDouble());
        } else if (x is List) {
          for (final e in x) {
            walk(e);
          }
        } else {
          throw StateError('Unexpected output element type: ${x.runtimeType}');
        }
      }
      walk(best);

      final bestFlat = Float32List.fromList(flat);
      return _unpackLandmarks(bestFlat, _inW, _inH, pack.padding, clamp: true);
    }
  }

  void dispose() {
    final iso = _iso;
    if (iso != null) {
      iso.close();
    }
    _itp.close();
  }
}
