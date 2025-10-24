part of face_detection_tflite;

class FaceDetection {
  final Interpreter _itp;
  final int _inW, _inH;
  final int _bboxIndex = 0, _scoreIndex = 1;
  final Float32List _anchors;
  final bool _assumeMirrored;

  IsolateInterpreter? _iso;

  late final int _inputIdx;
  late final List<int> _boxesShape;
  late final List<int> _scoresShape;

  late final Tensor _inputTensor;
  late final Tensor _boxesTensor;
  late final Tensor _scoresTensor;

  late final int _boxesLen;
  late final int _scoresLen;

  late final Float32List _inputBuf;
  late final Float32List _boxesBuf;
  late final Float32List _scoresBuf;

  FaceDetection._(this._itp, this._inW, this._inH, this._anchors, this._assumeMirrored);

  static Future<FaceDetection> create(FaceDetectionModel model, {InterpreterOptions? options, bool useIsolate = true}) async {
    final opts = _optsFor(model);
    final inW = opts['input_size_width'] as int;
    final inH = opts['input_size_height'] as int;
    final anchors = _ssdGenerateAnchors(opts);
    final itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/${_nameFor(model)}',
      options: options ?? InterpreterOptions(),
    );
    final assumeMirrored = switch (model) {
      FaceDetectionModel.backCamera => false,
      _ => true,
    };
    final obj = FaceDetection._(itp, inW, inH, anchors, assumeMirrored);

    int foundIdx = -1;
    for (var i = 0; i < 10; i++) {
      try {
        final s = itp.getInputTensor(i).shape;
        if (s.length == 4 && s.last == 3) {
          foundIdx = i;
          break;
        }
      } catch (_) {
        break;
      }
    }
    if (foundIdx == -1) {
      itp.close();
      throw StateError('No valid input tensor found with shape [batch, height, width, 3]');
    }
    obj._inputIdx = foundIdx;

    itp.resizeInputTensor(obj._inputIdx, [1, inH, inW, 3]);
    itp.allocateTensors();

    obj._boxesShape = itp.getOutputTensor(obj._bboxIndex).shape;
    obj._scoresShape = itp.getOutputTensor(obj._scoreIndex).shape;

    obj._inputTensor = itp.getInputTensor(obj._inputIdx);
    obj._boxesTensor = itp.getOutputTensor(obj._bboxIndex);
    obj._scoresTensor = itp.getOutputTensor(obj._scoreIndex);

    obj._boxesLen = obj._boxesShape.fold(1, (a, b) => a * b);
    obj._scoresLen = obj._scoresShape.fold(1, (a, b) => a * b);

    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();
    obj._boxesBuf = obj._boxesTensor.data.buffer.asFloat32List();
    obj._scoresBuf = obj._scoresTensor.data.buffer.asFloat32List();

    if (useIsolate) {
      obj._iso = await IsolateInterpreter.create(address: itp.address);
    }

    return obj;
  }

  List<List<List<List<double>>>> _asNHWC4D(Float32List flat, int h, int w) {
    final out = List<List<List<List<double>>>>.filled(
        1,
        List.generate(h, (_) => List.generate(w, (_) => List<double>.filled(3, 0.0, growable: false), growable: false),
            growable: false),
        growable: false);

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

  void _flatten3D(List<List<List<num>>> src, Float32List dst) {
    var k = 0;
    for (final a in src) {
      for (final b in a) {
        for (final c in b) {
          dst[k++] = c.toDouble();
        }
      }
    }
  }

  void _flatten2D(List<List<num>> src, Float32List dst) {
    var k = 0;
    for (final a in src) {
      for (final b in a) {
        dst[k++] = b.toDouble();
      }
    }
  }

  Future<List<Detection>> call(Uint8List imageBytes, {RectF? roi}) async {
    if (imageBytes.isEmpty) {
      throw ArgumentError('Image bytes cannot be empty');
    }
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

    final img.Image srcRoi = (roi == null) ? decoded : await cropFromRoi(decoded, roi);
    final pack = await _imageToTensor(srcRoi, outW: _inW, outH: _inH);

    Float32List boxesBuf;
    Float32List scoresBuf;

    if (_iso != null) {
      final input4d = _asNHWC4D(pack.tensorNHWC, _inH, _inW);

      final inputCount = _itp.getInputTensors().length;
      final inputs = List<Object?>.filled(inputCount, null, growable: false);
      inputs[_inputIdx] = input4d;

      final b0 = _boxesShape[0], b1 = _boxesShape[1], b2 = _boxesShape[2];
      final boxesOut3d = List.generate(
          b0, (_) => List.generate(
          b1, (_) => List<double>.filled(b2, 0.0, growable: false),
          growable: false),
          growable: false);

      Object scoresOut;
      if (_scoresShape.length == 3) {
        final s0 = _scoresShape[0], s1 = _scoresShape[1], s2 = _scoresShape[2];
        scoresOut = List.generate(
            s0, (_) => List.generate(
            s1, (_) => List<double>.filled(s2, 0.0, growable: false),
            growable: false),
            growable: false);
      } else {
        final s0 = _scoresShape[0], s1 = _scoresShape[1];
        scoresOut = List.generate(
            s0, (_) => List<double>.filled(s1, 0.0, growable: false),
            growable: false);
      }

      final outputs = <int, Object>{
        _bboxIndex: boxesOut3d,
        _scoreIndex: scoresOut,
      };

      await _iso!.runForMultipleInputs(inputs.cast<Object>(), outputs);

      final outBoxes = Float32List(_boxesLen);
      _flatten3D(boxesOut3d as List<List<List<num>>>, outBoxes);

      final outScores = Float32List(_scoresLen);
      if (_scoresShape.length == 3) {
        _flatten3D(scoresOut as List<List<List<num>>>, outScores);
      } else {
        _flatten2D(scoresOut as List<List<num>>, outScores);
      }

      boxesBuf = outBoxes;
      scoresBuf = outScores;
    } else {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();
      boxesBuf = _boxesBuf;
      scoresBuf = _scoresBuf;
    }

    final boxes = _decodeBoxes(boxesBuf, _boxesShape);
    final scores = _decodeScores(scoresBuf, _scoresShape);

    final dets = _toDetections(boxes, scores);
    final pruned = _nms(dets, _minSuppressionThreshold, _minScore, weighted: true);
    final fixed = _detectionLetterboxRemoval(pruned, pack.padding);

    List<Detection> mapped = roi != null ? fixed.map((d) => _mapDetectionToRoi(d, roi)).toList() : fixed;

    if (_assumeMirrored) {
      mapped = mapped.map((d) {
        final xmin = 1.0 - d.bbox.xmax;
        final xmax = 1.0 - d.bbox.xmin;
        final ymin = d.bbox.ymin;
        final ymax = d.bbox.ymax;
        final kp = List<double>.from(d.keypointsXY);
        for (int i = 0; i < kp.length; i += 2) {
          kp[i] = 1.0 - kp[i];
        }
        return Detection(bbox: RectF(xmin, ymin, xmax, ymax), score: d.score, keypointsXY: kp);
      }).toList();
    }

    return mapped;
  }

  List<_DecodedBox> _decodeBoxes(Float32List raw, List<int> shape) {
    final n = shape[1], k = shape[2];
    final scale = _inH.toDouble();
    final out = <_DecodedBox>[];
    final tmp = Float32List(k);
    for (var i = 0; i < n; i++) {
      final base = i * k;
      for (var j = 0; j < k; j++) {
        tmp[j] = raw[base + j] / scale;
      }
      final ax = _anchors[i * 2 + 0];
      final ay = _anchors[i * 2 + 1];
      tmp[0] += ax;
      tmp[1] += ay;
      for (var j = 4; j < k; j += 2) {
        tmp[j + 0] += ax;
        tmp[j + 1] += ay;
      }
      final xc = tmp[0], yc = tmp[1], w = tmp[2], h = tmp[3];
      final xmin = xc - w * 0.5, ymin = yc - h * 0.5, xmax = xc + w * 0.5, ymax = yc + h * 0.5;
      final kp = <double>[];
      for (var j = 4; j < k; j += 2) {
        kp.add(tmp[j + 0]);
        kp.add(tmp[j + 1]);
      }
      out.add(_DecodedBox(RectF(xmin, ymin, xmax, ymax), kp));
    }
    return out;
  }

  Float32List _decodeScores(Float32List raw, List<int> shape) {
    final n = shape[1];
    final scores = Float32List(n);
    for (var i =  0; i < n; i++) {
      scores[i] = _sigmoidClipped(raw[i]);
    }
    return scores;
  }

  List<Detection> _toDetections(List<_DecodedBox> boxes, Float32List scores) {
    final res = <Detection>[];
    final n = math.min(boxes.length, scores.length);
    for (var i = 0; i < n; i++) {
      final b = boxes[i].bbox;
      if (b.xmax <= b.xmin || b.ymax <= b.ymin) continue;
      res.add(Detection(bbox: b, score: scores[i], keypointsXY: boxes[i].keypointsXY));
    }
    return res;
  }

  void dispose() {
    _iso?.close();
    _itp.close();
  }
}
