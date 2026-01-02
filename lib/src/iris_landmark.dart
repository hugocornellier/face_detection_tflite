part of '../face_detection_tflite.dart';

/// Estimates dense iris keypoints within cropped eye regions and lets callers
/// derive a robust iris center (with fallback if inference fails).
class IrisLandmark {
  IsolateInterpreter? _iso;
  final Interpreter _itp;
  final int _inW, _inH;
  Delegate? _delegate;
  late final Tensor _inputTensor;
  late final Float32List _inputBuf;
  late final Map<int, List<int>> _outShapes;
  late final Map<int, Float32List> _outBuffers;
  late final List<List<List<List<double>>>> _input4dCache;
  late final Map<int, Object> _outputsCache;

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
  /// The [performanceConfig] parameter enables hardware acceleration delegates.
  /// Use [PerformanceConfig.xnnpack()] for 2-5x speedup on CPU. If both [options]
  /// and [performanceConfig] are provided, [options] takes precedence.
  ///
  /// Returns a fully initialized [IrisLandmark] instance ready to detect irises.
  ///
  /// **Note:** This model expects a cropped eye region as input. For full pipeline
  /// processing, use the high-level [FaceDetector] class with [FaceDetectionMode.full].
  ///
  /// Example:
  /// ```dart
  /// // Default (no acceleration)
  /// final irisModel = await IrisLandmark.create();
  /// final irisPoints = await irisModel(eyeCropImage);
  ///
  /// // With XNNPACK acceleration
  /// final irisModel = await IrisLandmark.create(
  ///   performanceConfig: PerformanceConfig.xnnpack(),
  /// );
  /// ```
  ///
  /// See also:
  /// - [createFromFile] for loading a model from a custom file path
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<IrisLandmark> create({
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
  }) async {
    Delegate? delegate;
    final InterpreterOptions interpreterOptions;
    if (options != null) {
      interpreterOptions = options;
    } else {
      final result = _createInterpreterOptions(performanceConfig);
      interpreterOptions = result.$1;
      delegate = result.$2;
    }

    final Interpreter itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$_irisLandmarkModel',
      options: interpreterOptions,
    );
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final IrisLandmark obj = IrisLandmark._(itp, inW, inH);
    obj._delegate = delegate;
    await obj._initializeTensors();
    return obj;
  }

  /// Creates an iris landmark model from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetectorIsolate] to initialize models
  /// in a background isolate where asset loading is not available.
  ///
  /// The [modelBytes] parameter should contain the raw TFLite model file contents.
  static Future<IrisLandmark> createFromBuffer(
    Uint8List modelBytes, {
    PerformanceConfig? performanceConfig,
  }) async {
    final result = _createInterpreterOptions(performanceConfig);
    final interpreterOptions = result.$1;
    final delegate = result.$2;

    final Interpreter itp = Interpreter.fromBuffer(
      modelBytes,
      options: interpreterOptions,
    );
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final IrisLandmark obj = IrisLandmark._(itp, inW, inH);
    obj._delegate = delegate;
    await obj._initializeTensors();
    return obj;
  }

  /// Shared tensor initialization logic.
  Future<void> _initializeTensors() async {
    _inputTensor = _itp.getInputTensor(0);
    _inputBuf = _inputTensor.data.buffer.asFloat32List();

    final Map<int, OutputTensorInfo> outputInfo = collectOutputTensorInfo(_itp);
    _outShapes =
        outputInfo.map((int k, OutputTensorInfo v) => MapEntry(k, v.shape));
    _outBuffers =
        outputInfo.map((int k, OutputTensorInfo v) => MapEntry(k, v.buffer));
    _input4dCache = createNHWCTensor4D(_inH, _inW);

    _outputsCache = <int, Object>{};
    _outShapes.forEach((i, shape) {
      _outputsCache[i] = allocTensorShape(shape);
    });

    _iso = await IsolateInterpreter.create(address: _itp.address);
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
  /// The [performanceConfig] parameter enables hardware acceleration delegates.
  /// Use [PerformanceConfig.xnnpack()] for 2-5x speedup on CPU. If both [options]
  /// and [performanceConfig] are provided, [options] takes precedence.
  ///
  /// Returns a fully initialized [IrisLandmark] instance ready to detect irises.
  ///
  /// Example:
  /// ```dart
  /// // Default (no acceleration)
  /// final customModel = await IrisLandmark.createFromFile(
  ///   '/path/to/custom_iris_model.tflite',
  /// );
  /// final irisPoints = await customModel(eyeCropImage);
  /// customModel.dispose(); // Clean up when done
  ///
  /// // With XNNPACK acceleration
  /// final customModel = await IrisLandmark.createFromFile(
  ///   '/path/to/custom_iris_model.tflite',
  ///   performanceConfig: PerformanceConfig.xnnpack(),
  /// );
  /// ```
  ///
  /// See also:
  /// - [create] for loading the default bundled model from assets
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<IrisLandmark> createFromFile(
    String modelPath, {
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
  }) async {
    Delegate? delegate;
    final InterpreterOptions interpreterOptions;
    if (options != null) {
      interpreterOptions = options;
    } else {
      final result = _createInterpreterOptions(performanceConfig);
      interpreterOptions = result.$1;
      delegate = result.$2;
    }

    final Interpreter itp = Interpreter.fromFile(
      File(modelPath),
      options: interpreterOptions,
    );
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final IrisLandmark obj = IrisLandmark._(itp, inW, inH);
    obj._delegate = delegate;

    obj._inputTensor = itp.getInputTensor(0);
    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();

    final Map<int, OutputTensorInfo> outputInfo = collectOutputTensorInfo(itp);
    obj._outShapes =
        outputInfo.map((int k, OutputTensorInfo v) => MapEntry(k, v.shape));
    obj._outBuffers =
        outputInfo.map((int k, OutputTensorInfo v) => MapEntry(k, v.buffer));
    obj._input4dCache = createNHWCTensor4D(inH, inW);

    obj._outputsCache = <int, Object>{};
    obj._outShapes.forEach((i, shape) {
      obj._outputsCache[i] = allocTensorShape(shape);
    });

    obj._iso = await IsolateInterpreter.create(address: itp.address);

    return obj;
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
  /// The returned points include 76 landmarks in this order:
  /// - First 71 points: Eye mesh landmarks (detailed eye region geometry)
  /// - Last 5 points: Iris keypoints (center + 4 contour points)
  ///
  /// The iris center is not guaranteed to be at a fixed index; derive it from
  /// the 5 iris keypoints if needed.
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
  /// // The last 5 points are iris keypoints; derive a center if needed.
  /// ```
  ///
  /// See also:
  /// - [callIrisOnly] to extract only the 5 iris keypoints
  /// - [runOnImage] to run on a full image with an eye ROI
  Future<List<List<double>>> call(
    img.Image eyeCrop, {
    IsolateWorker? worker,
  }) async {
    final ImageTensor pack = await imageToTensorWithWorker(
      eyeCrop,
      outW: _inW,
      outH: _inH,
      worker: worker,
    );

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();

      final List<List<double>> lm = <List<double>>[];
      for (final Float32List flat in _outBuffers.values) {
        lm.addAll(
          _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false),
        );
      }
      return lm;
    } else {
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [_input4dCache];
      await _iso!.runForMultipleInputs(inputs, _outputsCache);

      final List<List<double>> lm = <List<double>>[];
      _outShapes.forEach((i, _) {
        final Float32List flat = flattenDynamicTensor(_outputsCache[i]);
        lm.addAll(
          _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false),
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
  /// The returned points include 76 landmarks in this order:
  /// - First 71 points: Eye mesh landmarks (detailed eye region geometry)
  /// - Last 5 points: Iris keypoints (center + 4 contour points)
  ///
  /// Example:
  /// ```dart
  /// final eyeRoi = RectF(0.3, 0.4, 0.5, 0.55);
  /// final irisPoints = await irisModel.runOnImage(fullImage, eyeRoi);
  /// // irisPoints are in full image pixel coordinates
  /// ```
  ///
  /// See also:
  /// - [call] to run on a pre-cropped eye image
  /// - [runOnImageAlignedIris] for aligned eye ROI processing
  Future<List<List<double>>> runOnImage(
    img.Image src,
    RectF eyeRoi, {
    IsolateWorker? worker,
  }) async {
    final img.Image eyeCrop = await cropFromRoiWithWorker(src, eyeRoi, worker);
    final List<List<double>> lmNorm = await call(eyeCrop, worker: worker);
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
  /// prefer creating a long-lived [IrisLandmark] instance.
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
  /// - [create] for persistent isolate inference
  /// - [callIrisOnly] for the instance method alternative
  static Future<List<List<double>>> callWithIsolate(
    Uint8List eyeCropBytes,
    String modelPath, {
    bool irisOnly = false,
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
            (e) => (e as List).map((n) => (n as num).toDouble()).toList(),
          )
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
      final IrisLandmark iris = await IrisLandmark.createFromFile(modelPath);
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
  /// to the eye crop, where each point is `[x, y, z]`.
  /// The points are in the model's output order; derive a center if needed.
  ///
  /// **Performance:** Faster than [call] since it skips extracting additional
  /// eye contour landmarks.
  ///
  /// Example:
  /// ```dart
  /// final irisPoints = await irisLandmark.callIrisOnly(leftEyeCrop);
  /// // Derive a center from the 5 points if needed.
  /// ```
  ///
  /// See also:
  /// - [call] to get all iris and eye contour landmarks
  /// - [runOnImageAlignedIris] for aligned eye ROI processing
  Future<List<List<double>>> callIrisOnly(
    img.Image eyeCrop, {
    IsolateWorker? worker,
  }) async {
    final ImageTensor pack = await imageToTensorWithWorker(
      eyeCrop,
      outW: _inW,
      outH: _inH,
      worker: worker,
    );

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
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [_input4dCache];
      await _iso!.runForMultipleInputs(inputs, _outputsCache);

      final double pt = pack.padding[0],
          pb = pack.padding[1],
          pl = pack.padding[2],
          pr = pack.padding[3];
      final double sx = 1.0 - (pl + pr);
      final double sy = 1.0 - (pt + pb);

      Float32List? irisFlat;
      _outShapes.forEach((i, shape) {
        final Float32List flat = flattenDynamicTensor(_outputsCache[i]);
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
  /// Returns iris and eye contour landmarks in absolute pixel coordinates, where
  /// each point is `[x, y, z]`:
  /// - First 71 points: Eye mesh landmarks (detailed eye region geometry)
  /// - Last 5 points: Iris landmarks (center + 4 contour points)
  ///
  /// **Note:** This is an internal method used by [FaceDetector]. Most users
  /// should use [FaceDetector.detectFaces] with [FaceDetectionMode.full] instead.
  Future<List<List<double>>> runOnImageAlignedIris(
    img.Image src,
    AlignedRoi roi, {
    bool isRight = false,
    IsolateWorker? worker,
  }) async {
    final img.Image crop = await extractAlignedSquareWithWorker(
      src,
      roi.cx,
      roi.cy,
      roi.size,
      roi.theta,
      worker,
    );
    final img.Image eye = isRight ? await _flipHorizontal(crop) : crop;
    final double ct = math.cos(roi.theta);
    final double st = math.sin(roi.theta);
    final double s = roi.size;

    final List<List<double>> out = <List<double>>[];
    final List<List<double>> lmNorm = await call(eye, worker: worker);
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

  /// Runs iris detection using a registered frame ID.
  ///
  /// This is an optimized variant that uses a pre-registered frame to avoid
  /// transferring the full image data again.
  ///
  /// Returns iris and eye contour landmarks in absolute pixel coordinates.
  Future<List<List<double>>> runOnImageAlignedIrisWithFrameId(
    int frameId,
    AlignedRoi roi, {
    bool isRight = false,
    IsolateWorker? worker,
  }) async {
    if (worker == null || !worker.isInitialized) {
      throw StateError('Worker must be initialized to use frame IDs');
    }

    final img.Image crop = await worker.extractAlignedSquareWithFrameId(
      frameId,
      roi.cx,
      roi.cy,
      roi.size,
      roi.theta,
    );
    final img.Image eye = isRight ? await _flipHorizontal(crop) : crop;
    final double ct = math.cos(roi.theta);
    final double st = math.sin(roi.theta);
    final double s = roi.size;

    final List<List<double>> out = <List<double>>[];
    final List<List<double>> lmNorm = await call(eye, worker: worker);
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

  /// Predicts iris and eye contour landmarks from a cv.Mat eye crop.
  ///
  /// This is the OpenCV-based variant of [call] that accepts a cv.Mat directly,
  /// providing better performance by avoiding image format conversions.
  ///
  /// The [eyeCrop] parameter should contain a tight crop around a single eye as cv.Mat.
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
  ///
  /// The optional [buffer] parameter allows reusing a pre-allocated Float32List
  /// for the tensor conversion to reduce GC pressure.
  ///
  /// Returns a list of 3D landmark points in normalized coordinates.
  ///
  /// Example:
  /// ```dart
  /// final eyeCropMat = cv.imdecode(bytes, cv.IMREAD_COLOR);
  /// final irisPoints = await irisLandmark.callFromMat(eyeCropMat);
  /// eyeCropMat.dispose();
  /// ```
  Future<List<List<double>>> callFromMat(
    cv.Mat eyeCrop, {
    Float32List? buffer,
  }) async {
    final ImageTensor pack = convertImageToTensorFromMat(
      eyeCrop,
      outW: _inW,
      outH: _inH,
      buffer: buffer,
    );

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();

      final List<List<double>> lm = <List<double>>[];
      for (final Float32List flat in _outBuffers.values) {
        lm.addAll(
          _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false),
        );
      }
      return lm;
    } else {
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [_input4dCache];
      await _iso!.runForMultipleInputs(inputs, _outputsCache);

      final List<List<double>> lm = <List<double>>[];
      _outShapes.forEach((i, _) {
        final Float32List flat = flattenDynamicTensor(_outputsCache[i]);
        lm.addAll(
          _unpackLandmarks(flat, _inW, _inH, pack.padding, clamp: false),
        );
      });
      return lm;
    }
  }

  /// Runs iris detection on a cv.Mat using an aligned eye ROI.
  ///
  /// This is the OpenCV-based variant of [runOnImageAlignedIris] that uses
  /// SIMD-accelerated warpAffine for the rotation crop, providing 10-50x
  /// better performance than the Dart-based bilinear interpolation.
  ///
  /// The [src] parameter is the full image as cv.Mat.
  /// The [roi] parameter defines the eye region with center, size, and rotation.
  /// When [isRight] is true, the eye crop is flipped before processing.
  ///
  /// Returns iris landmarks in absolute pixel coordinates.
  ///
  /// Note: The input cv.Mat is NOT disposed by this method.
  Future<List<List<double>>> runOnImageAlignedIrisFromMat(
    cv.Mat src,
    AlignedRoi roi, {
    bool isRight = false,
    Float32List? buffer,
  }) async {
    final cv.Mat? crop = extractAlignedSquareFromMat(
      src,
      roi.cx,
      roi.cy,
      roi.size,
      roi.theta,
    );
    if (crop == null) {
      return const <List<double>>[];
    }

    cv.Mat eye;
    if (isRight) {
      eye = cv.flip(crop, 1); // Horizontal flip
      crop.dispose();
    } else {
      eye = crop;
    }

    final double ct = math.cos(roi.theta);
    final double st = math.sin(roi.theta);
    final double s = roi.size;

    final List<List<double>> out = <List<double>>[];
    final List<List<double>> lmNorm = await callFromMat(eye, buffer: buffer);
    eye.dispose();

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
    _delegate?.delete();
    _delegate = null;
    _iso?.close();
    _itp.close();
  }

  /// Creates interpreter options with delegates based on performance configuration.
  ///
  /// Delegates to [FaceDetection._createInterpreterOptions] for consistent
  /// platform-aware delegate selection across all model types.
  static (InterpreterOptions, Delegate?) _createInterpreterOptions(
      PerformanceConfig? config) {
    return FaceDetection._createInterpreterOptions(config);
  }
}

extension on IrisLandmark {
  Future<img.Image> _flipHorizontal(img.Image src) async {
    return img.flipHorizontal(src);
  }
}
