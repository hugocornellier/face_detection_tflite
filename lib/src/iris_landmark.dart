part of '../face_detection_tflite.dart';

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

  /// Creates and initializes an iris landmark model instance.
  ///
  /// This factory method loads the iris landmark TensorFlow Lite model from
  /// package assets and prepares it for inference. The model predicts 5 keypoints
  /// per iris plus eye contour points.
  ///
  /// The [options] parameter allows you to customize the TFLite interpreter
  /// configuration (e.g., number of threads, use of GPU delegate).
  ///
  /// When [useIsolate] is true (default), inference runs in a separate isolate
  /// to avoid blocking the UI thread.
  ///
  /// Returns a fully initialized [IrisLandmark] instance ready to detect irises.
  ///
  /// **Note:** This model expects a cropped eye region as input. For full pipeline
  /// processing, use the high-level [FaceDetector] class with [FaceDetectionMode.full].
  ///
  /// Example:
  /// ```dart
  /// final irisModel = await IrisLandmark.create(useIsolate: true);
  /// final irisPoints = await irisModel(eyeCropImage);
  /// ```
  ///
  /// See also:
  /// - [createFromFile] for loading a model from a custom file path
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
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

  /// Creates and initializes an iris landmark model from a custom file path.
  ///
  /// This factory method loads a TensorFlow Lite model from the specified
  /// [modelPath] on the filesystem instead of from package assets. This is
  /// useful for advanced users who want to use custom-trained or alternative
  /// iris tracking models.
  ///
  /// The [options] parameter allows you to customize the TFLite interpreter
  /// configuration (e.g., number of threads, use of GPU delegate).
  ///
  /// When [useIsolate] is true (default), inference runs in a separate isolate
  /// to avoid blocking the UI thread.
  ///
  /// Returns a fully initialized [IrisLandmark] instance ready to detect irises.
  ///
  /// Example:
  /// ```dart
  /// final customModel = await IrisLandmark.createFromFile(
  ///   '/path/to/custom_iris_model.tflite',
  ///   useIsolate: true,
  /// );
  /// final irisPoints = await customModel(eyeCropImage);
  /// customModel.dispose(); // Clean up when done
  /// ```
  ///
  /// See also:
  /// - [create] for loading the default bundled model from assets
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<IrisLandmark> createFromFile(
    String modelPath, {
    InterpreterOptions? options,
    bool useIsolate = true
  }) async {
    final Interpreter itp = Interpreter.fromFile(
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

  /// Predicts iris and eye contour landmarks for a cropped eye region.
  ///
  /// The [eyeCrop] parameter should contain a tight crop around a single eye,
  /// including the iris, pupil, and surrounding eye contours.
  ///
  /// Returns a list of 3D landmark points in normalized coordinates, where each
  /// point is represented as `[x, y, z]`:
  /// - `x` and `y` are normalized coordinates (0.0 to 1.0) relative to the eye crop
  /// - `z` represents relative depth
  ///
  /// The returned points typically include:
  /// - 5 iris keypoints (center and 4 contour points)
  /// - Additional eye contour landmarks
  ///
  /// **Input requirements:**
  /// - Eye should be roughly centered in the crop
  /// - Crop should be tight around the eye region
  /// - Image will be resized to model input size automatically
  ///
  /// **Note:** For iris tracking in the full face detection pipeline, use
  /// [FaceDetector.detectFaces] with [FaceDetectionMode.full] instead.
  ///
  /// Example:
  /// ```dart
  /// final irisPoints = await irisLandmark(leftEyeCrop);
  /// final irisCenter = irisPoints[0]; // First point is typically iris center
  /// ```
  ///
  /// See also:
  /// - [callIrisOnly] to extract only the 5 iris keypoints
  /// - [runOnImage] to run on a full image with an eye ROI
  Future<List<List<double>>> call(img.Image eyeCrop) async {
    final ImageTensor pack = await _imageToTensor(
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

  /// Runs iris detection on a full image using a specified eye region of interest.
  ///
  /// This method crops the eye region from the full image using the provided ROI,
  /// runs iris landmark detection on the crop, and maps the normalized landmark
  /// coordinates back to absolute pixel coordinates in the original image.
  ///
  /// The [src] parameter is the full decoded image containing the face.
  ///
  /// The [eyeRoi] parameter defines the eye region as a normalized rectangle
  /// with coordinates relative to image dimensions.
  ///
  /// Returns a list of 3D landmark points in absolute image coordinates, where
  /// each point is `[x, y, z]`:
  /// - `x` and `y` are pixel coordinates in the original image
  /// - `z` represents relative depth
  ///
  /// The returned points typically include:
  /// - 5 iris keypoints (center and 4 contour points)
  /// - Additional eye contour landmarks
  ///
  /// Example:
  /// ```dart
  /// final eyeRoi = RectF(xmin: 0.3, ymin: 0.4, w: 0.2, h: 0.15);
  /// final irisPoints = await irisModel.runOnImage(fullImage, eyeRoi);
  /// // irisPoints are in full image pixel coordinates
  /// ```
  ///
  /// See also:
  /// - [call] to run on a pre-cropped eye image
  /// - [runOnImageAlignedIris] for aligned eye ROI processing
  Future<List<List<double>>> runOnImage(img.Image src, RectF eyeRoi) async {
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

  /// Runs iris detection in a separate isolate for non-blocking inference.
  ///
  /// This static method spawns a dedicated isolate to perform iris landmark
  /// detection on encoded eye crop image bytes. This is useful for running
  /// iris detection without blocking the main UI thread, especially for
  /// one-off detections or background processing.
  ///
  /// The [eyeCropBytes] parameter should contain encoded image data (JPEG, PNG)
  /// of a cropped eye region.
  ///
  /// The [modelPath] parameter specifies the filesystem path to the iris model
  /// (.tflite file).
  ///
  /// The [irisOnly] parameter determines the output:
  /// - `true`: Returns only the 5 iris keypoints (center + 4 contour)
  /// - `false`: Returns all landmarks including eye contour points (default)
  ///
  /// Returns a list of 3D landmark points in normalized coordinates (0.0 to 1.0)
  /// relative to the eye crop, where each point is `[x, y, z]`.
  ///
  /// **Performance:** Creates a new isolate for each call. For repeated detections,
  /// prefer creating an [IrisLandmark] instance with `useIsolate: true`.
  ///
  /// Example:
  /// ```dart
  /// final irisPoints = await IrisLandmark.callWithIsolate(
  ///   eyeCropBytes,
  ///   '/path/to/iris_landmark.tflite',
  ///   irisOnly: true,
  /// );
  /// // Returns 5 iris keypoints in normalized coordinates
  /// ```
  ///
  /// Throws [StateError] if the model cannot be loaded or inference fails.
  ///
  /// See also:
  /// - [create] with `useIsolate: true` for persistent isolate inference
  /// - [callIrisOnly] for the instance method alternative
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

  /// Predicts only the 5 iris keypoints for a cropped eye region.
  ///
  /// This is an optimized alternative to [call] that extracts only the core
  /// iris landmarks (1 center + 4 contour points) instead of returning all
  /// eye contour points.
  ///
  /// The [eyeCrop] parameter should contain a tight crop around a single eye,
  /// including the iris, pupil, and surrounding eye contours.
  ///
  /// Returns exactly 5 points in normalized coordinates (0.0 to 1.0) relative
  /// to the eye crop, where each point is `[x, y, z]`:
  /// - Point 0: Iris center (typically most stable)
  /// - Points 1-4: Iris contour points
  ///
  /// **Performance:** Faster than [call] since it skips extracting additional
  /// eye contour landmarks.
  ///
  /// Example:
  /// ```dart
  /// final irisPoints = await irisLandmark.callIrisOnly(leftEyeCrop);
  /// final irisCenter = irisPoints[0]; // [x, y, z] normalized coordinates
  /// ```
  ///
  /// See also:
  /// - [call] to get all iris and eye contour landmarks
  /// - [runOnImageAlignedIris] for aligned eye ROI processing
  Future<List<List<double>>> callIrisOnly(img.Image eyeCrop) async {
    final ImageTensor pack = await _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

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

  /// Runs iris detection on an aligned eye region of interest.
  ///
  /// This method extracts an aligned square crop centered on the eye using the
  /// provided ROI parameters, runs iris landmark detection, and maps the results
  /// back to absolute pixel coordinates in the original image.
  ///
  /// The [src] parameter is the full decoded image containing the face.
  ///
  /// The [roi] parameter defines the eye region with center position, size,
  /// and rotation angle.
  ///
  /// When [isRight] is true, the extracted eye crop is horizontally flipped
  /// before processing to normalize right eyes to left eye orientation.
  ///
  /// Returns a list of 5 iris keypoints in absolute pixel coordinates, where
  /// each point is `[x, y, z]`:
  /// - Point 0: Iris center
  /// - Points 1-4: Iris contour points
  ///
  /// **Note:** This is an internal method used by [FaceDetector]. Most users
  /// should use [FaceDetector.detectFaces] with [FaceDetectionMode.full] instead.
  Future<List<List<double>>> runOnImageAlignedIris(
    img.Image src, AlignedRoi roi, {
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

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the iris landmark model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// **Note:** Most users should call [FaceDetector.dispose] instead, which
  /// automatically disposes all internal models (detection, mesh, and iris).
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
