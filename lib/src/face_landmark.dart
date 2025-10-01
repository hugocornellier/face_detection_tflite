part of face_detection_tflite;

class FaceLandmark {
  final Interpreter _itp;
  final int _inW, _inH;

  late final int _bestIdx;
  late final Tensor _inputTensor;
  late final Tensor _bestTensor;

  late final Float32List _inputBuf;
  late final Float32List _bestOutBuf;

  FaceLandmark._(this._itp, this._inW, this._inH);

  static Future<FaceLandmark> create({InterpreterOptions? options}) async {
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

    return obj;
  }

  Future<List<List<double>>> call(img.Image faceCrop) async {
    final pack = _imageToTensor(faceCrop, outW: _inW, outH: _inH);

    _inputBuf.setAll(0, pack.tensorNHWC);
    _itp.invoke();

    return _unpackLandmarks(_bestOutBuf, _inW, _inH, pack.padding, clamp: true);
  }

  void dispose() {
    _itp.close();
  }
}
