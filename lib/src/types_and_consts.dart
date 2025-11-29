part of face_detection_tflite;

/// Identifies specific facial landmarks returned by face detection.
///
/// Each enum value corresponds to a key facial feature point.
enum FaceLandmarkType { leftEye, rightEye, noseTip, mouth, leftEyeTragion, rightEyeTragion }

/// Specifies which face detection model variant to use.
///
/// Different models are optimized for different use cases:
/// - [frontCamera]: Optimized for selfie/front-facing camera (128x128 input)
/// - [backCamera]: Optimized for rear camera with higher resolution (256x256 input)
/// - [shortRange]: Optimized for close-up faces (128x128 input)
/// - [full]: Full-range detection (192x192 input)
/// - [fullSparse]: Full-range with sparse anchors (192x192 input)
enum FaceDetectionModel { frontCamera, backCamera, shortRange, full, fullSparse }

/// Controls which detection features to compute.
///
/// - [fast]: Only bounding boxes and landmarks (fastest)
/// - [standard]: Bounding boxes, landmarks, and 468-point face mesh
/// - [full]: All features including bounding boxes, landmarks, mesh, and iris tracking
enum FaceDetectionMode { fast, standard, full }

/// Outputs for a single detected face.
///
/// [bboxCorners] are the 4 corner points of the face box in pixel coordinates.
/// [landmarks] are coarse detection keypoints (e.g. eyes, nose, mouth corners).
/// [mesh] contains 468 facial landmarks as pixel coordinates.
/// [irises] contains 10 points (5 per eye) used to estimate iris position/size.
class Face {
  final _Detection _detection;

  /// The 468-point face mesh in pixel coordinates.
  ///
  /// Each point is a 3D coordinate `(x, y, z)` where:
  /// - `x` and `y` are absolute pixel positions in the original image
  /// - `z` represents relative depth (units are consistent but not metric)
  ///
  /// The 468 points follow MediaPipe's canonical face mesh topology, providing
  /// detailed geometry for facial features including eyes, eyebrows, nose, mouth,
  /// and face contours.
  ///
  /// This list is empty when [FaceDetector.detectFaces] is called with
  /// [FaceDetectionMode.fast]. Use [FaceDetectionMode.standard] or
  /// [FaceDetectionMode.full] to populate mesh data.
  ///
  /// See also:
  /// - [kMeshPoints] for the expected mesh point count (468)
  /// - [irises] for iris-specific landmarks
  final List<math.Point<double>> mesh;

  /// Iris landmark points in pixel coordinates.
  ///
  /// Contains 10 points total: 5 keypoints per iris (left and right eyes).
  /// Each iris is represented by a center point and 4 contour points.
  ///
  /// Each point is a 3D coordinate `(x, y, z)` where:
  /// - `x` and `y` are absolute pixel positions in the original image
  /// - `z` represents relative depth
  ///
  /// This list is empty when [FaceDetector.detectFaces] is called with
  /// [FaceDetectionMode.fast] or [FaceDetectionMode.standard]. Use
  /// [FaceDetectionMode.full] to enable iris tracking.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes, mode: FaceDetectionMode.full);
  /// if (faces.isNotEmpty && faces.first.irises.isNotEmpty) {
  ///   final leftIrisPoints = faces.first.irises.sublist(0, 5);
  ///   final rightIrisPoints = faces.first.irises.sublist(5, 10);
  /// }
  /// ```
  final List<math.Point<double>> irises;

  /// The dimensions of the original source image.
  ///
  /// This size is used internally to convert normalized coordinates to pixel
  /// coordinates for [bboxCorners], [landmarks], [mesh], and [irises].
  ///
  /// All coordinate data in [Face] is already scaled to these dimensions,
  /// so users typically don't need to use this field directly unless performing
  /// custom coordinate transformations.
  final Size originalSize;

  /// Creates a face detection result with bounding box, landmarks, and optional mesh/iris data.
  ///
  /// This constructor is typically called internally by [FaceDetector.detectFaces].
  /// Most users should not need to construct [Face] instances directly.
  ///
  /// The [detection] contains the bounding box and coarse facial keypoints.
  /// The [mesh] contains 468 facial landmark points (empty if not computed).
  /// The [irises] contains iris keypoints (empty if not computed).
  /// The [originalSize] specifies the dimensions of the source image for coordinate mapping.
  Face({
    required _Detection detection,
    required this.mesh,
    required this.irises,
    required this.originalSize,
  }) : _detection = detection;

  /// The four corner points of the face bounding box in pixel coordinates.
  ///
  /// Returns points in order: top-left, top-right, bottom-right, bottom-left.
  List<math.Point<double>> get bboxCorners {
    final _RectF r = _detection.bbox;
    final double w = originalSize.width.toDouble();
    final double h = originalSize.height.toDouble();
    return [
      math.Point<double>(r.xmin * w, r.ymin * h),
      math.Point<double>(r.xmax * w, r.ymin * h),
      math.Point<double>(r.xmax * w, r.ymax * h),
      math.Point<double>(r.xmin * w, r.ymax * h),
    ];
  }

  /// Facial landmark positions in pixel coordinates.
  ///
  /// Returns a map where keys are [FaceLandmarkType] values identifying specific
  /// facial features (eyes, nose, mouth, etc.) and values are their pixel positions.
  Map<FaceLandmarkType, math.Point<double>> get landmarks => _detection.landmarks;
}

/// The expected number of 3D landmark points in a complete face mesh.
///
/// MediaPipe's face mesh model produces exactly 468 points covering facial
/// features including eyes, eyebrows, nose, mouth, and face contours.
///
/// Use this constant to validate mesh output or split concatenated mesh data:
/// ```dart
/// assert(meshPoints.length == kMeshPoints); // Validate single face
/// final faces = meshPoints.length ~/ kMeshPoints; // Count faces in batch
/// ```
const int kMeshPoints = 468;

const _modelNameBack = 'face_detection_back.tflite';
const _modelNameFront = 'face_detection_front.tflite';
const _modelNameShort = 'face_detection_short_range.tflite';
const _modelNameFull = 'face_detection_full_range.tflite';
const _modelNameFullSparse = 'face_detection_full_range_sparse.tflite';
const _faceLandmarkModel = 'face_landmark.tflite';
const _irisLandmarkModel = 'iris_landmark.tflite';
const double _rawScoreLimit = 80.0;
const double _minScore = 0.5;
const double _minSuppressionThreshold = 0.3;

const _ssdFront = {
  'num_layers': 4,
  'input_size_height': 128,
  'input_size_width': 128,
  'anchor_offset_x': 0.5,
  'anchor_offset_y': 0.5,
  'strides': [8, 16, 16, 16],
  'interpolated_scale_aspect_ratio': 1.0,
};
const _ssdBack = {
  'num_layers': 4,
  'input_size_height': 256,
  'input_size_width': 256,
  'anchor_offset_x': 0.5,
  'anchor_offset_y': 0.5,
  'strides': [16, 32, 32, 32],
  'interpolated_scale_aspect_ratio': 1.0,
};
const _ssdShort = {
  'num_layers': 4,
  'input_size_height': 128,
  'input_size_width': 128,
  'anchor_offset_x': 0.5,
  'anchor_offset_y': 0.5,
  'strides': [8, 16, 16, 16],
  'interpolated_scale_aspect_ratio': 1.0,
};
const _ssdFull = {
  'num_layers': 1,
  'input_size_height': 192,
  'input_size_width': 192,
  'anchor_offset_x': 0.5,
  'anchor_offset_y': 0.5,
  'strides': [4],
  'interpolated_scale_aspect_ratio': 0.0,
};

class _AlignedFace {
  final double cx;
  final double cy;
  final double size;
  final double theta;
  final img.Image faceCrop;
  _AlignedFace({
    required this.cx,
    required this.cy,
    required this.size,
    required this.theta,
    required this.faceCrop
  });
}

class _RectF {
  final double xmin, ymin, xmax, ymax;
  const _RectF(this.xmin, this.ymin, this.xmax, this.ymax);
  double get w => xmax - xmin;
  double get h => ymax - ymin;
  _RectF scale(double sx, double sy) => _RectF(
      xmin * sx,
      ymin * sy,
      xmax * sx,
      ymax * sy
  );
  _RectF expand(double frac) {
    final double cx = (xmin + xmax) * 0.5;
    final double cy = (ymin + ymax) * 0.5;
    final double hw = (w * (1.0 + frac)) * 0.5;
    final double hh = (h * (1.0 + frac)) * 0.5;
    return _RectF(cx - hw, cy - hh, cx + hw, cy + hh);
  }
}

class _Detection {
  final _RectF bbox;
  final double score;
  final List<double> keypointsXY;
  final Size? imageSize;

  _Detection({
    required this.bbox,
    required this.score,
    required this.keypointsXY,
    this.imageSize,
  });

  double operator [](int i) => keypointsXY[i];

  Map<FaceLandmarkType, math.Point<double>> get landmarks {
    final Size? sz = imageSize;
    if (sz == null) {
      throw StateError(
        '_Detection.imageSize is null; cannot produce pixel landmarks.'
      );
    }
    final double w = sz.width.toDouble(), h = sz.height.toDouble();
    final Map<FaceLandmarkType, math.Point<double>> map = <
      FaceLandmarkType,
      math.Point<double>
    >{};
    for (final FaceLandmarkType idx in FaceLandmarkType.values) {
      final double xn = keypointsXY[idx.index * 2];
      final double yn = keypointsXY[idx.index * 2 + 1];
      map[idx] = math.Point<double>(xn * w, yn * h);
    }
    return map;
  }
}

class _ImageTensor {
  final Float32List tensorNHWC;
  final List<double> padding;
  final int width, height;
  _ImageTensor(this.tensorNHWC, this.padding, this.width, this.height);
}

class _AlignedRoi {
  final double cx;
  final double cy;
  final double size;
  final double theta;
  const _AlignedRoi(this.cx, this.cy, this.size, this.theta);
}

class _DecodedBox {
  final _RectF bbox;
  final List<double> keypointsXY;
  _DecodedBox(this.bbox, this.keypointsXY);
}
