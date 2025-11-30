part of '../face_detection_tflite.dart';

/// Runs face box detection and predicts a small set of facial keypoints
/// (eyes, nose, mouth, tragions) on the detected face(s).
class FaceDetection {
  IsolateInterpreter? _iso;
  final Interpreter _itp;
  final int _inW, _inH;
  final int _bboxIndex = 0, _scoreIndex = 1;
  final Float32List _anchors;
  final bool _assumeMirrored;
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

  FaceDetection._(
    this._itp,
    this._inW,
    this._inH,
    this._anchors,
    this._assumeMirrored,
  );

  /// Creates and initializes a face detection model instance.
  ///
  /// This factory method loads the specified TensorFlow Lite [model] from package
  /// assets and prepares it for inference. Different model variants are optimized
  /// for different use cases (see [FaceDetectionModel] for details).
  ///
  /// The [options] parameter allows you to customize the TFLite interpreter
  /// configuration (e.g., number of threads, use of GPU delegate).
  ///
  /// When [useIsolate] is true (default), inference runs in a separate isolate
  /// to avoid blocking the UI thread. Set to false if you plan to manage isolates
  /// yourself or need synchronous execution.
  ///
  /// Returns a fully initialized [FaceDetection] instance ready to detect faces.
  ///
  /// **Note:** Most users should use the high-level [FaceDetector] class instead
  /// of working with this low-level API directly.
  ///
  /// Example:
  /// ```dart
  /// final detector = await FaceDetection.create(
  ///   FaceDetectionModel.frontCamera,
  ///   useIsolate: true,
  /// );
  /// ```
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<FaceDetection> create(
    FaceDetectionModel model, {
    InterpreterOptions? options,
    bool useIsolate = true,
  }) async {
    final Map<String, Object> opts = _optsFor(model);
    final int inW = opts['input_size_width'] as int;
    final int inH = opts['input_size_height'] as int;
    final bool assumeMirrored = switch (model) {
      FaceDetectionModel.backCamera => false,
      _ => true,
    };

    final Interpreter itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/${_nameFor(model)}',
      options: options ?? InterpreterOptions(),
    );

    final Float32List anchors = _ssdGenerateAnchors(opts);
    final FaceDetection obj = FaceDetection._(
      itp,
      inW,
      inH,
      anchors,
      assumeMirrored,
    );

    int foundIdx = -1;
    for (int i = 0; i < 10; i++) {
      try {
        final List<int> s = itp.getInputTensor(i).shape;
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
      throw StateError(
        'No valid input tensor found with shape [batch, height, width, 3]',
      );
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
      List.generate(
        h,
        (_) => List.generate(
          w,
          (_) => List<double>.filled(3, 0.0, growable: false),
          growable: false,
        ),
        growable: false,
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

  void _flatten3D(List<List<List<num>>> src, Float32List dst) {
    int k = 0;
    for (final List<List<num>> a in src) {
      for (final List<num> b in a) {
        for (final num c in b) {
          dst[k++] = c.toDouble();
        }
      }
    }
  }

  void _flatten2D(List<List<num>> src, Float32List dst) {
    int k = 0;
    for (final List<num> a in src) {
      for (final num b in a) {
        dst[k++] = b.toDouble();
      }
    }
  }

  /// Runs face detection inference on the provided image.
  ///
  /// The [imageBytes] parameter should contain encoded image data (JPEG, PNG, etc.).
  /// The image is decoded, preprocessed, and passed through the face detection model.
  ///
  /// Optionally, you can specify a [roi] (region of interest) to detect faces only
  /// within a specific area of the image. This can improve performance and accuracy
  /// when you know the approximate face location.
  ///
  /// Returns a list of detected faces as [Detection] objects, each containing:
  /// - A bounding box with normalized coordinates (0.0 to 1.0)
  /// - A confidence score
  /// - Coarse facial keypoints (eyes, nose, mouth, tragions)
  ///
  /// The detections are filtered using non-maximum suppression (NMS) to remove
  /// duplicate/overlapping detections.
  ///
  /// **Coordinate system:** All returned coordinates are normalized (0.0 to 1.0)
  /// relative to the image dimensions, where (0, 0) is top-left and (1, 1) is bottom-right.
  ///
  /// **Note:** This is a low-level method. Most users should use [FaceDetector.detectFaces()]
  /// which provides a higher-level API with automatic coordinate mapping.
  ///
  /// Throws [ArgumentError] if [imageBytes] is empty.
  Future<List<Detection>> call(Uint8List imageBytes, {RectF? roi}) async {
    if (imageBytes.isEmpty) {
      throw ArgumentError('Image bytes cannot be empty');
    }
    final DecodedRgb d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(d);
    return callWithDecoded(decoded, roi: roi);
  }

  /// Runs face detection inference on a pre-decoded image.
  ///
  /// This is an optimized variant of [call] that accepts a pre-decoded image
  /// to avoid redundant decoding. When a [worker] is provided, it uses the
  /// long-lived worker isolate for image operations instead of spawning fresh
  /// isolates for each operation.
  ///
  /// The [decoded] parameter should be a fully decoded image (not encoded bytes).
  ///
  /// Optionally, you can specify a [roi] (region of interest) to detect faces only
  /// within a specific area of the image.
  ///
  /// The [worker] parameter allows providing an ImageProcessingWorker for
  /// optimized image operations. When null, falls back to spawning fresh isolates.
  ///
  /// Returns a list of detected faces as [Detection] objects, each containing:
  /// - A bounding box with normalized coordinates (0.0 to 1.0)
  /// - A confidence score
  /// - Coarse facial keypoints (eyes, nose, mouth, tragions)
  ///
  /// **Note:** This method is primarily for internal optimization. Most users
  /// should use [call] or [FaceDetector.detectFaces].
  Future<List<Detection>> callWithDecoded(
    img.Image decoded, {
    RectF? roi,
    ImageProcessingWorker? worker,
  }) async {
    final img.Image srcRoi = (roi == null)
        ? decoded
        : await cropFromRoiWithWorker(decoded, roi, worker);
    final ImageTensor pack = await imageToTensorWithWorker(
      srcRoi,
      outW: _inW,
      outH: _inH,
      worker: worker,
    );

    Float32List boxesBuf;
    Float32List scoresBuf;

    if (_iso != null) {
      final List<List<List<List<double>>>> input4d = _asNHWC4D(
        pack.tensorNHWC,
        _inH,
        _inW,
      );
      final int inputCount = _itp.getInputTensors().length;
      final List<Object?> inputs = List<Object?>.filled(
        inputCount,
        null,
        growable: false,
      );
      inputs[_inputIdx] = input4d;

      final int b0 = _boxesShape[0], b1 = _boxesShape[1], b2 = _boxesShape[2];
      final List<List<List<double>>> boxesOut3d = List.generate(
        b0,
        (_) => List.generate(
          b1,
          (_) => List<double>.filled(b2, 0.0, growable: false),
          growable: false,
        ),
        growable: false,
      );

      Object scoresOut;
      if (_scoresShape.length == 3) {
        final int s0 = _scoresShape[0],
            s1 = _scoresShape[1],
            s2 = _scoresShape[2];
        scoresOut = List.generate(
          s0,
          (_) => List.generate(
            s1,
            (_) => List<double>.filled(s2, 0.0, growable: false),
            growable: false,
          ),
          growable: false,
        );
      } else {
        final int s0 = _scoresShape[0], s1 = _scoresShape[1];
        scoresOut = List.generate(
          s0,
          (_) => List<double>.filled(s1, 0.0, growable: false),
          growable: false,
        );
      }

      final Map<int, Object> outputs = <int, Object>{
        _bboxIndex: boxesOut3d,
        _scoreIndex: scoresOut,
      };

      await _iso!.runForMultipleInputs(inputs.cast<Object>(), outputs);

      final Float32List outBoxes = Float32List(_boxesLen);
      _flatten3D(boxesOut3d as List<List<List<num>>>, outBoxes);

      final Float32List outScores = Float32List(_scoresLen);
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

    final Float32List scores = _decodeScores(scoresBuf, _scoresShape);
    final List<DecodedBox> boxes = _decodeBoxes(boxesBuf, _boxesShape);
    final List<Detection> dets = _toDetections(boxes, scores);
    final List<Detection> pruned = _nms(
      dets,
      _minSuppressionThreshold,
      _minScore,
      weighted: true,
    );
    final List<Detection> fixed = _detectionLetterboxRemoval(
      pruned,
      pack.padding,
    );

    List<Detection> mapped = roi != null
        ? fixed.map((d) => _mapDetectionToRoi(d, roi)).toList()
        : fixed;

    if (_assumeMirrored) {
      mapped = mapped.map((d) {
        final double xmin = 1.0 - d.bbox.xmax;
        final double xmax = 1.0 - d.bbox.xmin;
        final double ymin = d.bbox.ymin;
        final double ymax = d.bbox.ymax;
        final List<double> kp = List<double>.from(d.keypointsXY);
        for (int i = 0; i < kp.length; i += 2) {
          kp[i] = 1.0 - kp[i];
        }
        return Detection(
          bbox: RectF(xmin, ymin, xmax, ymax),
          score: d.score,
          keypointsXY: kp,
        );
      }).toList();
    }

    return mapped;
  }

  List<DecodedBox> _decodeBoxes(Float32List raw, List<int> shape) {
    final int n = shape[1], k = shape[2];
    final double scale = _inH.toDouble();
    final List<DecodedBox> out = <DecodedBox>[];
    final Float32List tmp = Float32List(k);

    for (int i = 0; i < n; i++) {
      final int base = i * k;
      for (int j = 0; j < k; j++) {
        tmp[j] = raw[base + j] / scale;
      }
      final double ax = _anchors[i * 2 + 0];
      final double ay = _anchors[i * 2 + 1];
      tmp[0] += ax;
      tmp[1] += ay;
      for (int j = 4; j < k; j += 2) {
        tmp[j + 0] += ax;
        tmp[j + 1] += ay;
      }
      final double xc = tmp[0], yc = tmp[1], w = tmp[2], h = tmp[3];
      final double xmin = xc - w * 0.5,
          ymin = yc - h * 0.5,
          xmax = xc + w * 0.5,
          ymax = yc + h * 0.5;
      final List<double> kp = <double>[];
      for (int j = 4; j < k; j += 2) {
        kp.add(tmp[j + 0]);
        kp.add(tmp[j + 1]);
      }
      out.add(DecodedBox(RectF(xmin, ymin, xmax, ymax), kp));
    }
    return out;
  }

  Float32List _decodeScores(Float32List raw, List<int> shape) {
    final int n = shape[1];
    final Float32List scores = Float32List(n);
    for (int i = 0; i < n; i++) {
      scores[i] = _sigmoidClipped(raw[i]);
    }
    return scores;
  }

  List<Detection> _toDetections(List<DecodedBox> boxes, Float32List scores) {
    final List<Detection> res = <Detection>[];
    final int n = math.min(boxes.length, scores.length);
    for (int i = 0; i < n; i++) {
      final RectF b = boxes[i].bbox;
      if (b.xmax <= b.xmin || b.ymax <= b.ymin) continue;
      res.add(
        Detection(bbox: b, score: scores[i], keypointsXY: boxes[i].keypointsXY),
      );
    }
    return res;
  }

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the face detection model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// **Note:** Most users should call [FaceDetector.dispose] instead, which
  /// automatically disposes all internal models (detection, mesh, and iris).
  void dispose() {
    _iso?.close();
    _itp.close();
  }
}
