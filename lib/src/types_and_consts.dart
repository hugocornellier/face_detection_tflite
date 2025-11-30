part of '../face_detection_tflite.dart';

/// Identifies specific facial landmarks returned by face detection.
///
/// Each enum value corresponds to a key facial feature point.
enum FaceLandmarkType {
  leftEye,
  rightEye,
  noseTip,
  mouth,
  leftEyeTragion,
  rightEyeTragion
}

/// Specifies which face detection model variant to use.
///
/// Different models are optimized for different use cases:
/// - [frontCamera]: Optimized for selfie/front-facing camera (128x128 input)
/// - [backCamera]: Optimized for rear camera with higher resolution (256x256 input)
/// - [shortRange]: Optimized for close-up faces (128x128 input)
/// - [full]: Full-range detection (192x192 input)
/// - [fullSparse]: Full-range with sparse anchors (192x192 input)
enum FaceDetectionModel {
  frontCamera,
  backCamera,
  shortRange,
  full,
  fullSparse
}

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
  final Detection _detection;

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
    required Detection detection,
    required this.mesh,
    required this.irises,
    required this.originalSize,
  }) : _detection = detection;

  /// The four corner points of the face bounding box in pixel coordinates.
  ///
  /// Returns points in order: top-left, top-right, bottom-right, bottom-left.
  List<math.Point<double>> get bboxCorners {
    final RectF r = _detection.bbox;
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
  Map<FaceLandmarkType, math.Point<double>> get landmarks =>
      _detection.landmarks;
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

/// Holds an aligned face crop and metadata used for downstream landmark models.
///
/// An [AlignedFace] represents a face that has been rotated, scaled, and
/// translated so that the eyes are horizontal and the face roughly fills the
/// crop. Downstream models such as [FaceLandmark] and [IrisLandmark] expect
/// this normalized orientation.
class AlignedFace {
  /// X coordinate of the face center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of the face center in absolute pixel coordinates.
  final double cy;

  /// Length of the square crop edge in absolute pixels.
  final double size;

  /// Rotation applied to align the face, in radians.
  final double theta;

  /// The aligned face crop image provided to landmark models.
  final img.Image faceCrop;

  /// Creates an aligned face crop with pixel-based center, size, rotation,
  /// and the cropped [faceCrop] image ready for landmark inference.
  AlignedFace(
      {required this.cx,
      required this.cy,
      required this.size,
      required this.theta,
      required this.faceCrop});
}

/// Axis-aligned rectangle with normalized coordinates.
///
/// Values are expressed as fractions of the original image dimensions
/// (0.0 - 1.0). Utilities are provided to scale and expand the rectangle.
class RectF {
  /// Minimum X and Y plus maximum X and Y extents.
  final double xmin, ymin, xmax, ymax;
  const RectF(this.xmin, this.ymin, this.xmax, this.ymax);

  /// Rectangle width.
  double get w => xmax - xmin;

  /// Rectangle height.
  double get h => ymax - ymin;

  /// Returns a rectangle scaled independently in X and Y.
  RectF scale(double sx, double sy) =>
      RectF(xmin * sx, ymin * sy, xmax * sx, ymax * sy);

  /// Expands the rectangle by [frac] in all directions, keeping the same center.
  RectF expand(double frac) {
    final double cx = (xmin + xmax) * 0.5;
    final double cy = (ymin + ymax) * 0.5;
    final double hw = (w * (1.0 + frac)) * 0.5;
    final double hh = (h * (1.0 + frac)) * 0.5;
    return RectF(cx - hw, cy - hh, cx + hw, cy + hh);
  }
}

/// Raw detection output from the face detector containing bbox and keypoints.
class Detection {
  /// Normalized bounding box for the face.
  final RectF bbox;

  /// Confidence score for the detection.
  final double score;

  /// Flattened landmark coordinates `[x0, y0, x1, y1, ...]` normalized 0-1.
  final List<double> keypointsXY;

  /// Original image dimensions used to denormalize landmarks.
  final Size? imageSize;

  Detection({
    required this.bbox,
    required this.score,
    required this.keypointsXY,
    this.imageSize,
  });

  /// Convenience accessor for `[keypointsXY]` by index.
  double operator [](int i) => keypointsXY[i];

  /// Returns facial landmarks in pixel coordinates keyed by landmark type.
  Map<FaceLandmarkType, math.Point<double>> get landmarks {
    final Size? sz = imageSize;
    if (sz == null) {
      throw StateError(
          'Detection.imageSize is null; cannot produce pixel landmarks.');
    }
    final double w = sz.width.toDouble(), h = sz.height.toDouble();
    final Map<FaceLandmarkType, math.Point<double>> map =
        <FaceLandmarkType, math.Point<double>>{};
    for (final FaceLandmarkType idx in FaceLandmarkType.values) {
      final double xn = keypointsXY[idx.index * 2];
      final double yn = keypointsXY[idx.index * 2 + 1];
      map[idx] = math.Point<double>(xn * w, yn * h);
    }
    return map;
  }
}

/// Image tensor plus padding metadata used to undo letterboxing.
class ImageTensor {
  /// NHWC float tensor normalized to [-1, 1] expected by MediaPipe models.
  final Float32List tensorNHWC;

  /// Padding fractions `[top, bottom, left, right]` applied during resize.
  final List<double> padding;

  /// Target width and height passed to the model.
  final int width, height;
  ImageTensor(this.tensorNHWC, this.padding, this.width, this.height);
}

/// Rotation-aware region of interest for cropped eye landmarks.
class AlignedRoi {
  /// X coordinate of ROI center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of ROI center in absolute pixel coordinates.
  final double cy;

  /// Square ROI size in absolute pixels.
  final double size;

  /// Rotation applied to align the ROI, in radians.
  final double theta;

  /// Creates a rotation-aware region of interest in absolute pixel coordinates
  /// used to crop around the eyes for iris landmark detection.
  const AlignedRoi(this.cx, this.cy, this.size, this.theta);
}

/// Decoded detection box and keypoints straight from the TFLite model.
class DecodedBox {
  /// Normalized bounding box for a detected face.
  final RectF bbox;

  /// Flattened list of normalized keypoints `[x0, y0, ...]`.
  final List<double> keypointsXY;

  /// Constructs a decoded detection with its normalized bounding box and
  /// flattened landmark coordinates output by the face detector.
  DecodedBox(this.bbox, this.keypointsXY);
}
