part of face_detection_tflite;

/// Predicts the full 468-point face mesh (x, y, z per point) for an aligned face crop.
/// Coordinates are normalized before later mapping back to image space.
class FaceLandmark {
  IsolateInterpreter? _iso;
  final Interpreter _itp;
  final int _inW, _inH;
  late final int _bestIdx;
  late final Tensor _inputTensor;
  late final Tensor _bestTensor;
  late final Float32List _inputBuf;
  late final Float32List _bestOutBuf;
  late final List<List<int>> _outShapes;

  FaceLandmark._(this._itp, this._inW, this._inH);

  static Future<FaceLandmark> create({InterpreterOptions? options, bool useIsolate = true}) async {
    final Interpreter itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$_faceLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final List<int> ishape = itp.getInputTensor(0).shape;
    final inH = ishape[1];
    final inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final FaceLandmark obj = FaceLandmark._(itp, inW, inH);
    obj._inputTensor = itp.getInputTensor(0);
    int numElements(List<int> s) => s.fold(1, (a, b) => a * b);

    final Map<int, List<int>> shapes = <int, List<int>>{};
    for (int i = 0;; i++) {
      try {
        final List<int> s = itp.getOutputTensor(i).shape;
        shapes[i] = s;
      } catch (_) {
        break;
      }
    }

    int bestIdx = -1;
    int bestLen = -1;
    for (final MapEntry<int, List<int>> e in shapes.entries) {
      final int len = numElements(e.value);
      if (len > bestLen && len % 3 == 0) {
        bestLen = len;
        bestIdx = e.key;
      }
    }
    obj._bestIdx = bestIdx;
    obj._bestTensor = itp.getOutputTensor(obj._bestIdx);
    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();
    obj._bestOutBuf = obj._bestTensor.data.buffer.asFloat32List();

    final int maxIndex = shapes.keys.isEmpty
        ? -1
        : shapes.keys.reduce((a, b) => a > b ? a : b);
    obj._outShapes = List<List<int>>.generate(maxIndex + 1, (i) => shapes[i] ?? const <int>[]);

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
          w,
          (_) => List<double>.filled(3, 0.0, growable: false),
          growable: false
        ),
        growable: false),
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
    Object build(List<int> s, int depth) {
      if (depth == s.length - 1) {
        return List<double>.filled(s[depth], 0.0, growable: false);
      }
      return List.generate(s[depth], (_) => build(s, depth + 1), growable: false);
    }
    return build(shape, 0);
  }

  Future<List<List<double>>> call(img.Image faceCrop) async {
    final _ImageTensor pack = await _imageToTensor(faceCrop, outW: _inW, outH: _inH);

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();
      return _unpackLandmarks(_bestOutBuf, _inW, _inH, pack.padding, clamp: true);
    } else {
      final input4d = _asNHWC4D(pack.tensorNHWC, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [input4d];
      final Map<int, Object> outputs = <int, Object>{};
      for (int i = 0; i < _outShapes.length; i++) {
        final List<int> s = _outShapes[i];
        if (s.isNotEmpty) {
          outputs[i] = _allocForShape(s);
        }
      }
      await _iso!.runForMultipleInputs(inputs, outputs);

      final dynamic best = outputs[_bestIdx];

      final List<double> flat = <double>[];
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

      final Float32List bestFlat = Float32List.fromList(flat);
      return _unpackLandmarks(bestFlat, _inW, _inH, pack.padding, clamp: true);
    }
  }

  void dispose() {
    final IsolateInterpreter? iso = _iso;
    if (iso != null) {
      iso.close();
    }
    _itp.close();
  }
}
