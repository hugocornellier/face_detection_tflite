part of 'face_core.dart';

class FaceLandmark {
  final Interpreter _itp;
  final int _inW, _inH;

  FaceLandmark._(this._itp, this._inW, this._inH);

  static Future<FaceLandmark> create({InterpreterOptions? options}) async {
    final itp = await Interpreter.fromAsset(
      'assets/models/$_faceLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final ishape = itp.getInputTensor(0).shape;
    final inH = ishape[1];
    final inW = ishape[2];
    return FaceLandmark._(itp, inW, inH);
  }

  Future<List<List<double>>> call(img.Image faceCrop) async {
    final pack = _imageToTensor(faceCrop, outW: _inW, outH: _inH);

    _itp.resizeInputTensor(0, [1, _inH, _inW, 3]);
    _itp.allocateTensors();

    int _numElements(List<int> s) => s.fold(1, (a, b) => a * b);
    final outShape = _itp.getOutputTensor(0).shape;

    final input4d = List.generate(1, (_) => List.generate(_inH, (y) => List.generate(_inW, (x) {
      final base = (y * _inW + x) * 3;
      return [pack.tensorNHWC[base], pack.tensorNHWC[base + 1], pack.tensorNHWC[base + 2]];
    })));

    dynamic outY;
    if (outShape.length == 3) {
      outY = List.generate(outShape[0], (_) =>
          List.generate(outShape[1], (_) => List.filled(outShape[2], 0.0)));
    } else if (outShape.length == 2) {
      outY = List.generate(outShape[0], (_) => List.filled(outShape[1], 0.0));
    } else {
      outY = List.filled(outShape[0], 0.0);
    }

    _itp.runForMultipleInputs([input4d], {0: outY});

    final flat = Float32List(_numElements(outShape));
    var k = 0;
    if (outShape.length == 3) {
      for (var i = 0; i < outShape[0]; i++) {
        for (var j = 0; j < outShape[1]; j++) {
          for (var l = 0; l < outShape[2]; l++) {
            flat[k++] = (outY[i][j][l] as num).toDouble();
          }
        }
      }
    } else if (outShape.length == 2) {
      for (var i = 0; i < outShape[0]; i++) {
        for (var j = 0; j < outShape[1]; j++) {
          flat[k++] = (outY[i][j] as num).toDouble();
        }
      }
    } else {
      for (var i = 0; i < outShape[0]; i++) {
        flat[k++] = (outY[i] as num).toDouble();
      }
    }

    final n = (flat.length / 3).floor();
    final lm = <List<double>>[];
    for (var i = 0; i < n; i++) {
      final x = flat[i * 3 + 0];
      final y = flat[i * 3 + 1];
      final z = flat[i * 3 + 2];
      lm.add([x, y, z]);
    }
    return lm;
  }
}