part of 'face_core.dart';

class IrisLandmark {
  final Interpreter _itp;
  final int _inW, _inH;

  IrisLandmark._(this._itp, this._inW, this._inH);

  static Future<IrisLandmark> create({InterpreterOptions? options}) async {
    final itp = await Interpreter.fromAsset(
      'assets/models/$_irisLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final ishape = itp.getInputTensor(0).shape;
    final inH = ishape[1];
    final inW = ishape[2];
    return IrisLandmark._(itp, inW, inH);
  }

  Future<List<List<double>>> call(img.Image eyeCrop) async {
    final pack = _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

    _itp.resizeInputTensor(0, [1, _inH, _inW, 3]);
    _itp.allocateTensors();

    int _numElements(List<int> s) => s.fold(1, (a, b) => a * b);

    final outShape0 = _itp.getOutputTensor(0).shape;
    final outShape1 = _itp.getOutputTensor(1).shape;

    final input4d = List.generate(1, (_) => List.generate(_inH, (y) => List.generate(_inW, (x) {
      final base = (y * _inW + x) * 3;
      return [pack.tensorNHWC[base], pack.tensorNHWC[base + 1], pack.tensorNHWC[base + 2]];
    })));

    dynamic out0;
    out0 = List.generate(outShape0[0], (_) => List.filled(outShape0[1], 0.0));
    dynamic out1;
    out1 = List.generate(outShape1[0], (_) => List.filled(outShape1[1], 0.0));

    _itp.runForMultipleInputs([input4d], {0: out0, 1: out1});

    final flat0 = Float32List(_numElements(outShape0));
    var k = 0;
    for (var i = 0; i < outShape0[0]; i++) {
      for (var j = 0; j < outShape0[1]; j++) {
        flat0[k++] = (out0[i][j] as num).toDouble();
      }
    }

    final flat1 = Float32List(_numElements(outShape1));
    k = 0;
    for (var i = 0; i < outShape1[0]; i++) {
      for (var j = 0; j < outShape1[1]; j++) {
        flat1[k++] = (out1[i][j] as num).toDouble();
      }
    }

    final pt = pack.padding[0], pb = pack.padding[1], pl = pack.padding[2], pr = pack.padding[3];
    final sx = 1.0 - (pl + pr);
    final sy = 1.0 - (pt + pb);

    final lm = <List<double>>[];

    final n0 = (flat0.length / 3).floor();
    for (var i = 0; i < n0; i++) {
      var x = flat0[i * 3 + 0];
      var y = flat0[i * 3 + 1];
      final z = flat0[i * 3 + 2];
      x = (x - pl) / sx;
      y = (y - pt) / sy;
      lm.add([x, y, z]);
    }

    final n1 = (flat1.length / 3).floor();
    for (var i = 0; i < n1; i++) {
      var x = flat1[i * 3 + 0];
      var y = flat1[i * 3 + 1];
      final z = flat1[i * 3 + 2];
      x = (x - pl) / sx;
      y = (y - pt) / sy;
      lm.add([x, y, z]);
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
}