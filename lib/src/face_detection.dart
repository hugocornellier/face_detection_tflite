part of 'face_core.dart';

class FaceDetection {
  final Interpreter _itp;
  final int _inW, _inH;
  final int _bboxIndex = 0, _scoreIndex = 1;

  final Float32List _anchors;
  final bool _assumeMirrored;

  FaceDetection._(this._itp, this._inW, this._inH, this._anchors, this._assumeMirrored);

  static Future<FaceDetection> create(FaceDetectionModel model, {InterpreterOptions? options}) async {
    final opts = _optsFor(model);
    final inW = opts['input_size_width'] as int;
    final inH = opts['input_size_height'] as int;
    final anchors = _ssdGenerateAnchors(opts);
    final itp = await Interpreter.fromAsset(
      'assets/models/${_nameFor(model)}',
      options: options ?? InterpreterOptions(),
    );
    final assumeMirrored = switch (model) {
      FaceDetectionModel.backCamera => false,
      _ => true,
    };
    return FaceDetection._(itp, inW, inH, anchors, assumeMirrored);
  }

  Future<List<Detection>> call(Uint8List imageBytes, {RectF? roi}) async {
    final src = img.decodeImage(imageBytes)!;
    final img.Image srcRoi = (roi == null) ? src : cropFromRoi(src, roi);
    final pack = _imageToTensor(srcRoi, outW: _inW, outH: _inH);

    int inputIdx = 0;
    for (final i in [0, 1, 2, 3]) {
      try {
        final s = _itp.getInputTensor(i).shape;
        if (s.length == 4) {
          inputIdx = i;
          break;
        }
      } catch (_) {}
    }
    _itp.resizeInputTensor(inputIdx, [1, _inH, _inW, 3]);
    _itp.allocateTensors();

    int _numElements(List<int> s) => s.fold(1, (a, b) => a * b);
    final boxesShape = _itp.getOutputTensor(_bboxIndex).shape;
    final scoresShape = _itp.getOutputTensor(_scoreIndex).shape;

    final input4d = List.generate(1, (_) => List.generate(_inH, (y) => List.generate(_inW, (x) {
      final base = (y * _inW + x) * 3;
      return [pack.tensorNHWC[base], pack.tensorNHWC[base + 1], pack.tensorNHWC[base + 2]];
    })));

    dynamic outBoxes;
    if (boxesShape.length == 3) {
      outBoxes = List.generate(boxesShape[0], (_) =>
          List.generate(boxesShape[1], (_) => List.filled(boxesShape[2], 0.0)));
    } else if (boxesShape.length == 2) {
      outBoxes = List.generate(boxesShape[0], (_) => List.filled(boxesShape[1], 0.0));
    } else {
      outBoxes = List.filled(boxesShape[0], 0.0);
    }

    dynamic outScores;
    if (scoresShape.length == 3) {
      outScores = List.generate(scoresShape[0], (_) =>
          List.generate(scoresShape[1], (_) => List.filled(scoresShape[2], 0.0)));
    } else if (scoresShape.length == 2) {
      outScores = List.generate(scoresShape[0], (_) => List.filled(scoresShape[1], 0.0));
    } else {
      outScores = List.filled(scoresShape[0], 0.0);
    }

    _itp.runForMultipleInputs([input4d], {
      _bboxIndex: outBoxes,
      _scoreIndex: outScores,
    });

    final rawBoxes = Float32List(_numElements(boxesShape));
    var k = 0;
    if (boxesShape.length == 3) {
      for (var i = 0; i < boxesShape[0]; i++) {
        for (var j = 0; j < boxesShape[1]; j++) {
          for (var l = 0; l < boxesShape[2]; l++) {
            rawBoxes[k++] = (outBoxes[i][j][l] as num).toDouble();
          }
        }
      }
    } else if (boxesShape.length == 2) {
      for (var i = 0; i < boxesShape[0]; i++) {
        for (var j = 0; j < boxesShape[1]; j++) {
          rawBoxes[k++] = (outBoxes[i][j] as num).toDouble();
        }
      }
    } else {
      for (var i = 0; i < boxesShape[0]; i++) {
        rawBoxes[k++] = (outBoxes[i] as num).toDouble();
      }
    }

    final rawScores = Float32List(_numElements(scoresShape));
    k = 0;
    if (scoresShape.length == 3) {
      for (var i = 0; i < scoresShape[0]; i++) {
        for (var j = 0; j < scoresShape[1]; j++) {
          for (var l = 0; l < scoresShape[2]; l++) {
            rawScores[k++] = (outScores[i][j][l] as num).toDouble();
          }
        }
      }
    } else if (scoresShape.length == 2) {
      for (var i = 0; i < scoresShape[0]; i++) {
        for (var j = 0; j < scoresShape[1]; j++) {
          rawScores[k++] = (outScores[i][j] as num).toDouble();
        }
      }
    } else {
      for (var i = 0; i < scoresShape[0]; i++) {
        rawScores[k++] = (outScores[i] as num).toDouble();
      }
    }

    final boxes = _decodeBoxes(rawBoxes, boxesShape);
    final scores = _decodeScores(rawScores, scoresShape);

    final dets = _toDetections(boxes, scores);
    final pruned = _nms(dets, _minSuppressionThreshold, _minScore, weighted: true);
    final fixed = _detectionLetterboxRemoval(pruned, pack.padding);

    List<Detection> mapped;
    if (roi != null) {
      final dx = roi.xmin;
      final dy = roi.ymin;
      final sx = roi.w;
      final sy = roi.h;
      mapped = fixed.map((d) {
        RectF mapRect(RectF r) =>
            RectF(dx + r.xmin * sx, dy + r.ymin * sy, dx + r.xmax * sx, dy + r.ymax * sy);
        List<double> mapKp(List<double> k) {
          final o = List<double>.from(k);
          for (int i = 0; i < o.length; i += 2) {
            o[i] = dx + o[i] * sx;
            o[i + 1] = dy + o[i + 1] * sy;
          }
          return o;
        }
        return Detection(bbox: mapRect(d.bbox), score: d.score, keypointsXY: mapKp(d.keypointsXY));
      }).toList();
    } else {
      mapped = fixed;
    }

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

    for (final det in mapped) {
      final bbox = det.bbox;
      final xminPx = (bbox.xmin * src.width).toInt();
      final yminPx = (bbox.ymin * src.height).toInt();
      final xmaxPx = (bbox.xmax * src.width).toInt();
      final ymaxPx = (bbox.ymax * src.height).toInt();
      print("BBox -> xmin: $xminPx, ymin: $yminPx, xmax: $xmaxPx, ymax: $ymaxPx, score: ${det.score}");
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
    for (var i = 0; i < n; i++) {
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
}