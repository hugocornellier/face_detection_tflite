part of face_detection_tflite;

class IrisLandmark {
  final Interpreter _itp;
  final int _inW, _inH;

  late final Tensor _inputTensor;
  late final Map<int, Float32List> _outBuffers;
  late final Float32List _inputBuf;

  IrisLandmark._(this._itp, this._inW, this._inH);

  static Future<IrisLandmark> create({InterpreterOptions? options}) async {
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

    final shapes = <int, List<int>>{};
    final tensors = <int, Tensor>{};
    final buffers = <int, Float32List>{};

    for (var i = 0;; i++) {
      try {
        final t = itp.getOutputTensor(i);
        final s = t.shape;
        shapes[i] = s;
        tensors[i] = t;
        buffers[i] = t.data.buffer.asFloat32List();
      } catch (_) {
        break;
      }
    }

    obj._outBuffers = buffers;
    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();

    return obj;
  }

  Future<List<List<double>>> call(img.Image eyeCrop) async {
    final pack = _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

    _inputBuf.setAll(0, pack.tensorNHWC);
    _itp.invoke();

    final lm = <List<double>>[];
    for (final flat in _outBuffers.values) {
      lm.addAll(_unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false));
    }
    return lm;
  }

  Future<List<List<double>>> runOnImage(img.Image src, RectF eyeRoi) async {
    final eyeCrop = cropFromRoi(src, eyeRoi);
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

  void dispose() {
    _itp.close();
  }
}

extension on IrisLandmark {
  img.Image _flipHorizontal(img.Image src) {
    final out = img.Image(width: src.width, height: src.height);
    for (int y = 0; y < src.height; y++) {
      for (int x = 0; x < src.width; x++) {
        out.setPixel(src.width - 1 - x, y, src.getPixel(x, y));
      }
    }
    return out;
  }

  Future<List<List<double>>> callIrisOnly(img.Image eyeCrop) async {
    final pack = _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

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

    final pt = pack.padding[0], pb = pack.padding[1], pl = pack.padding[2], pr = pack.padding[3];
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
  }

  Future<List<List<double>>> runOnImageAlignedIris(img.Image src, AlignedRoi roi, {bool isRight = false}) async {
    final crop = extractAlignedSquare(src, roi.cx, roi.cy, roi.size, roi.theta);
    final eye = isRight ? _flipHorizontal(crop) : crop;
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
}
