part of '../face_detection_tflite.dart';

/// Model name for the MobileFaceNet embedding model.
const _embeddingModel = 'mobilefacenet.tflite';

/// Default input size for MobileFaceNet (112x112).
const int kEmbeddingInputSize = 112;

/// Default embedding vector dimension (192-dim for MobileFaceNet).
const int kEmbeddingDimension = 192;

/// Generates face embeddings (identity vectors) from aligned face crops.
///
/// Uses the MobileFaceNet model to produce 192-dimensional embedding vectors
/// that represent a face's identity. These embeddings can be compared using
/// cosine similarity to determine if two faces belong to the same person.
///
/// The model expects 112×112 RGB face images as input. For best results,
/// faces should be aligned (eyes horizontal, centered, face fills the frame).
///
/// ## Usage
///
/// ```dart
/// final embedding = await FaceEmbedding.create();
///
/// // Get embedding from an aligned face crop
/// final vector = await embedding(alignedFaceCrop);
/// print('Embedding dimension: ${vector.length}'); // 192
///
/// // Compare two embeddings
/// final similarity = FaceEmbedding.cosineSimilarity(vector1, vector2);
/// if (similarity > 0.6) {
///   print('Same person!');
/// }
///
/// // Clean up
/// embedding.dispose();
/// ```
///
/// ## Integration with FaceDetector
///
/// For the full face recognition pipeline, use [FaceDetector.getFaceEmbedding]
/// which handles face detection, alignment, and embedding extraction:
///
/// ```dart
/// final detector = FaceDetector();
/// await detector.initialize();
///
/// final faces = await detector.detectFaces(imageBytes);
/// final embedding = await detector.getFaceEmbedding(faces.first, imageBytes);
/// ```
class FaceEmbedding {
  IsolateInterpreter? _iso;
  final Interpreter _itp;
  final int _inW, _inH;
  Delegate? _delegate;
  late final Tensor _inputTensor;
  late final Tensor _outputTensor;
  late final Float32List _inputBuf;
  late final Float32List _outputBuf;
  late final List<List<List<List<double>>>> _input4dCache;
  late final int _embeddingDim;

  FaceEmbedding._(this._itp, this._inW, this._inH);

  /// Creates and initializes a face embedding model instance.
  ///
  /// This factory method loads the MobileFaceNet TensorFlow Lite model
  /// from package assets and prepares it for inference.
  ///
  /// The [performanceConfig] parameter enables hardware acceleration delegates.
  /// Use [PerformanceConfig.xnnpack()] for 2-5x speedup on CPU.
  ///
  /// Returns a fully initialized [FaceEmbedding] instance ready to generate
  /// face embeddings.
  ///
  /// Example:
  /// ```dart
  /// // Default (XNNPACK enabled)
  /// final embedding = await FaceEmbedding.create();
  ///
  /// // With custom configuration
  /// final embedding = await FaceEmbedding.create(
  ///   performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
  /// );
  /// ```
  ///
  /// Throws [StateError] if the model cannot be loaded or initialized.
  static Future<FaceEmbedding> create({
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
      'packages/face_detection_tflite/assets/models/$_embeddingModel',
      options: interpreterOptions,
    );
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final FaceEmbedding obj = FaceEmbedding._(itp, inW, inH);
    obj._delegate = delegate;
    await obj._initializeTensors();
    return obj;
  }

  /// Creates a face embedding model from pre-loaded model bytes.
  ///
  /// This is primarily used by [FaceDetectorIsolate] to initialize models
  /// in a background isolate where asset loading is not available.
  ///
  /// The [modelBytes] parameter should contain the raw TFLite model file contents.
  static Future<FaceEmbedding> createFromBuffer(
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

    final FaceEmbedding obj = FaceEmbedding._(itp, inW, inH);
    obj._delegate = delegate;
    await obj._initializeTensors();
    return obj;
  }

  /// Shared tensor initialization logic.
  Future<void> _initializeTensors() async {
    _inputTensor = _itp.getInputTensor(0);
    _outputTensor = _itp.getOutputTensor(0);
    _inputBuf = _inputTensor.data.buffer.asFloat32List();
    _outputBuf = _outputTensor.data.buffer.asFloat32List();
    _embeddingDim = _outputTensor.shape.last;
    _input4dCache = createNHWCTensor4D(_inH, _inW);
    _iso = await IsolateInterpreter.create(address: _itp.address);
  }

  /// The expected input width for the embedding model (112 pixels).
  int get inputWidth => _inW;

  /// The expected input height for the embedding model (112 pixels).
  int get inputHeight => _inH;

  /// The dimension of the output embedding vector.
  int get embeddingDimension => _embeddingDim;

  /// Generates a face embedding from an aligned face crop.
  ///
  /// The [faceCrop] parameter should contain an aligned face image.
  /// For best results, the face should be:
  /// - Centered in the image
  /// - Eyes roughly horizontal
  /// - Face filling most of the frame
  ///
  /// The image will be resized to the model's input size (112×112) automatically.
  ///
  /// Returns a [Float32List] containing the embedding vector. The vector
  /// is L2-normalized (unit length) for use with cosine similarity.
  ///
  /// Example:
  /// ```dart
  /// final embedding = await faceEmbedding(alignedFaceCrop);
  /// print('Got ${embedding.length}-dimensional embedding');
  /// ```
  Future<Float32List> call(img.Image faceCrop) async {
    final ImageTensor pack = convertImageToTensor(
      faceCrop,
      outW: _inW,
      outH: _inH,
    );

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();
      return _normalizeEmbedding(Float32List.fromList(_outputBuf));
    } else {
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [_input4dCache];

      final List<int> outShape = _outputTensor.shape;
      final Map<int, Object> outputs = <int, Object>{
        0: allocTensorShape(outShape),
      };

      await _iso!.runForMultipleInputs(inputs, outputs);

      final Float32List embedding = flattenDynamicTensor(outputs[0]);
      return _normalizeEmbedding(embedding);
    }
  }

  /// Generates a face embedding from an aligned face crop using cv.Mat.
  ///
  /// This is the OpenCV-based variant of [call] that accepts a cv.Mat directly,
  /// providing better performance by avoiding image format conversions.
  ///
  /// The [faceCrop] parameter should contain an aligned face as cv.Mat.
  /// The Mat is NOT disposed by this method - caller is responsible for disposal.
  ///
  /// The optional [buffer] parameter allows reusing a pre-allocated Float32List
  /// for the tensor conversion to reduce GC pressure.
  ///
  /// Returns a [Float32List] containing the L2-normalized embedding vector.
  ///
  /// Example:
  /// ```dart
  /// final faceCropMat = cv.imdecode(bytes, cv.IMREAD_COLOR);
  /// final embedding = await faceEmbedding.callFromMat(faceCropMat);
  /// faceCropMat.dispose();
  /// ```
  Future<Float32List> callFromMat(cv.Mat faceCrop,
      {Float32List? buffer}) async {
    final ImageTensor pack = convertImageToTensorFromMat(
      faceCrop,
      outW: _inW,
      outH: _inH,
      buffer: buffer,
    );

    if (_iso == null) {
      _inputBuf.setAll(0, pack.tensorNHWC);
      _itp.invoke();
      return _normalizeEmbedding(Float32List.fromList(_outputBuf));
    } else {
      fillNHWC4D(pack.tensorNHWC, _input4dCache, _inH, _inW);
      final List<List<List<List<List<double>>>>> inputs = [_input4dCache];

      final List<int> outShape = _outputTensor.shape;
      final Map<int, Object> outputs = <int, Object>{
        0: allocTensorShape(outShape),
      };

      await _iso!.runForMultipleInputs(inputs, outputs);

      final Float32List embedding = flattenDynamicTensor(outputs[0]);
      return _normalizeEmbedding(embedding);
    }
  }

  /// L2-normalizes an embedding vector to unit length.
  ///
  /// Normalized embeddings allow direct use of dot product as cosine similarity,
  /// since cos(θ) = a·b / (|a||b|) = a·b when |a| = |b| = 1.
  Float32List _normalizeEmbedding(Float32List embedding) {
    double norm = 0.0;
    for (int i = 0; i < embedding.length; i++) {
      norm += embedding[i] * embedding[i];
    }
    norm = math.sqrt(norm);

    if (norm > 0) {
      final Float32List normalized = Float32List(embedding.length);
      for (int i = 0; i < embedding.length; i++) {
        normalized[i] = embedding[i] / norm;
      }
      return normalized;
    }
    return embedding;
  }

  /// Computes the cosine similarity between two embedding vectors.
  ///
  /// Cosine similarity measures the angle between two vectors, ranging from
  /// -1 (opposite) to 1 (identical). For face embeddings:
  /// - Values > 0.6 strongly suggest the same person
  /// - Values > 0.5 suggest the same person
  /// - Values < 0.3 suggest different people
  ///
  /// Both [a] and [b] should be L2-normalized embeddings (as returned by [call]).
  /// If not normalized, this method will still work but may give different thresholds.
  ///
  /// Example:
  /// ```dart
  /// final similarity = FaceEmbedding.cosineSimilarity(embedding1, embedding2);
  /// if (similarity > 0.6) {
  ///   print('Very likely the same person');
  /// }
  /// ```
  static double cosineSimilarity(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError(
          'Embedding dimensions must match: ${a.length} vs ${b.length}');
    }

    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    final double denom = math.sqrt(normA) * math.sqrt(normB);
    return denom > 0 ? dot / denom : 0.0;
  }

  /// Computes the Euclidean (L2) distance between two embedding vectors.
  ///
  /// Euclidean distance measures the straight-line distance between two points.
  /// For face embeddings:
  /// - Values < 0.6 strongly suggest the same person
  /// - Values < 0.8 suggest the same person
  /// - Values > 1.0 suggest different people
  ///
  /// Note: Threshold values assume normalized embeddings.
  ///
  /// Example:
  /// ```dart
  /// final distance = FaceEmbedding.euclideanDistance(embedding1, embedding2);
  /// if (distance < 0.6) {
  ///   print('Very likely the same person');
  /// }
  /// ```
  static double euclideanDistance(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError(
          'Embedding dimensions must match: ${a.length} vs ${b.length}');
    }

    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      final double diff = a[i] - b[i];
      sum += diff * diff;
    }
    return math.sqrt(sum);
  }

  /// Releases all TensorFlow Lite resources held by this model.
  ///
  /// Call this when you're done using the face embedding model to free up memory.
  /// After calling dispose, this instance cannot be used for inference.
  ///
  /// **Note:** Most users should call [FaceDetector.dispose] instead, which
  /// automatically disposes all internal models (detection, mesh, iris, and embedding).
  void dispose() {
    _delegate?.delete();
    _delegate = null;
    final IsolateInterpreter? iso = _iso;
    if (iso != null) {
      iso.close();
    }
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

/// Holds the alignment parameters for extracting a face crop for embedding.
///
/// Similar to [AlignedFace] but optimized for the 112×112 input size
/// required by face embedding models.
class AlignedFaceForEmbedding {
  /// X coordinate of the face center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of the face center in absolute pixel coordinates.
  final double cy;

  /// Length of the square crop edge in absolute pixels.
  final double size;

  /// Rotation applied to align the face, in radians.
  final double theta;

  /// Creates alignment parameters for face embedding extraction.
  const AlignedFaceForEmbedding({
    required this.cx,
    required this.cy,
    required this.size,
    required this.theta,
  });
}

/// Computes alignment parameters for extracting a face crop suitable for embedding.
///
/// This function calculates the optimal crop region for face recognition based on
/// eye positions. The alignment ensures:
/// - Eyes are horizontally aligned
/// - Face is centered in the crop
/// - Adequate padding around the face
///
/// The [leftEye] and [rightEye] parameters are the eye center positions in
/// absolute pixel coordinates.
///
/// Returns [AlignedFaceForEmbedding] with center, size, and rotation parameters.
AlignedFaceForEmbedding computeEmbeddingAlignment({
  required Point leftEye,
  required Point rightEye,
}) {
  final double dx = rightEye.x - leftEye.x;
  final double dy = rightEye.y - leftEye.y;
  final double theta = math.atan2(dy, dx);

  final double eyeDist = math.sqrt(dx * dx + dy * dy);

  final double size = eyeDist * 2.5;

  final double eyeCx = (leftEye.x + rightEye.x) * 0.5;
  final double eyeCy = (leftEye.y + rightEye.y) * 0.5;

  final double ct = math.cos(theta);
  final double st = math.sin(theta);
  final double offsetY = size * 0.15;
  final double cx = eyeCx - offsetY * st;
  final double cy = eyeCy + offsetY * ct;

  return AlignedFaceForEmbedding(
    cx: cx,
    cy: cy,
    size: size,
    theta: theta,
  );
}
