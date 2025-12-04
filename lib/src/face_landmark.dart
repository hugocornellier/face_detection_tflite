part of '../face_detection_tflite.dart';

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
  late final List<List<List<List<double>>>> _input4dCache;

  FaceLandmark._(this._itp, this._inW, this._inH);

  /// Creates and initializes a face landmark (mesh) model instance.
  ///
  /// This factory method loads the 468-point face mesh TensorFlow Lite model
  /// from package assets and prepares it for inference. The face mesh provides
  /// detailed 3D geometry of facial features.
  ///
  /// The [options] parameter allows you to customize the TFLite interpreter
  /// configuration (e.g., number of threads, use of GPU delegate).
  ///
  /// Returns a fully initialized [FaceLandmark] instance ready to predict face meshes.
  ///
  /// **Note:** This model expects an aligned face crop as input. For full pipeline
  /// processing, use the high-level [FaceDetector] class instead.
  ///
  /// Example:
  /// ```dart
  /// final landmarkModel = await FaceLandmark.create();
  /// final meshPoints = await landmarkModel(alignedFaceCrop);
  /// ```
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<FaceLandmark> create({
    InterpreterOptions? options,
  }) async {
    final Interpreter itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$_faceLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final FaceLandmark obj = FaceLandmark._(itp, inW, inH);
    obj._inputTensor = itp.getInputTensor(0);
    int numElements(List<int> s) => s.fold(1, (a, b) => a * b);

    final Map<int, OutputTensorInfo> outputInfo = collectOutputTensorInfo(itp);
    final Map<int, List<int>> shapes =
        outputInfo.map((int k, OutputTensorInfo v) => MapEntry(k, v.shape));

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

    final int maxIndex =
        shapes.keys.isEmpty ? -1 : shapes.keys.reduce((a, b) => a > b ? a : b);
    obj._outShapes = List<List<int>>.generate(
      maxIndex + 1,
      (i) => shapes[i] ?? const <int>[],
    );

    obj._input4dCache = createNHWCTensor4D(inH, inW);

    obj._iso = await IsolateInterpreter.create(address: itp.address);

    return obj;
  }

  /// Predicts the 468-point face mesh for an aligned face crop.
  ///
  /// The [faceCrop] parameter should contain an aligned, cropped face image.
  /// For best results, the face should be centered, upright, and roughly fill
  /// the image bounds.
  ///
  /// Returns a list of 468 3D landmark points, where each point is represented
  /// as `[x, y, z]`:
  /// - `x` and `y` are normalized coordinates (0.0 to 1.0) relative to the crop
  /// - `z` represents relative depth (units are consistent but not metric)
  ///
  /// The 468 points follow MediaPipe's canonical face mesh topology, providing
  /// detailed geometry for facial features including eyes, eyebrows, nose, mouth,
  /// and face contours.
  ///
  /// **Input requirements:**
  /// - Face should be aligned (rotated upright)
  /// - Face should occupy most of the image
  /// - Image will be resized to model input size automatically
  ///
  /// Example:
  /// ```dart
  /// final meshPoints = await faceLandmark(alignedFaceCrop);
  /// print('Predicted ${meshPoints.length} mesh points'); // 468
  /// ```
  Future<List<List<double>>> call(
    img.Image faceCrop, {
    IsolateWorker? worker,
  }) async {
    final ImageTensor pack = await imageToTensorWithWorker(
      faceCrop,
      outW: _inW,
      outH: _inH,
      worker: worker,
    );

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();
      return _unpackLandmarks(
        _bestOutBuf,
        _inW,
        _inH,
        pack.padding,
        clamp: true,
      );
    } else {
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [_input4dCache];
      final Map<int, Object> outputs = <int, Object>{};
      for (int i = 0; i < _outShapes.length; i++) {
        final List<int> s = _outShapes[i];
        if (s.isNotEmpty) {
          outputs[i] = allocTensorShape(s);
        }
      }
      await _iso!.runForMultipleInputs(inputs, outputs);

      final dynamic best = outputs[_bestIdx];

      final Float32List bestFlat = flattenDynamicTensor(best);
      return _unpackLandmarks(bestFlat, _inW, _inH, pack.padding, clamp: true);
    }
  }

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the face landmark model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// **Note:** Most users should call [FaceDetector.dispose] instead, which
  /// automatically disposes all internal models (detection, mesh, and iris).
  void dispose() {
    final IsolateInterpreter? iso = _iso;
    if (iso != null) {
      iso.close();
    }
    _itp.close();
  }
}
