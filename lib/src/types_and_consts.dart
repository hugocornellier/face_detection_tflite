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
  rightEyeTragion,
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
  fullSparse,
}

/// Controls which detection features to compute.
///
/// - [fast]: Only bounding boxes and landmarks (fastest)
/// - [standard]: Bounding boxes, landmarks, and 468-point face mesh
/// - [full]: All features including bounding boxes, landmarks, mesh, and iris tracking
enum FaceDetectionMode { fast, standard, full }

/// Performance modes for TensorFlow Lite delegate selection.
///
/// Determines which hardware acceleration delegates are used for inference.
enum PerformanceMode {
  /// No acceleration delegates (CPU-only, backward compatible).
  ///
  /// - Slowest performance
  /// - No additional memory overhead
  /// - Most compatible (works on all platforms)
  disabled,

  /// XNNPACK delegate for CPU optimization.
  ///
  /// - Works on all platforms (iOS, Android, macOS, Linux, Windows)
  /// - 2-5x faster than disabled mode
  /// - Minimal memory overhead (+2-3MB per interpreter)
  /// - Recommended default for most use cases
  ///
  /// Uses SIMD vectorization (NEON on ARM, AVX on x86) and multi-threading.
  xnnpack,

  /// Automatically choose best delegate for current platform.
  ///
  /// Current behavior:
  /// - All platforms: Uses XNNPACK with platform-optimal thread count
  ///
  /// Future: May use GPU/Metal delegates when available.
  auto,
}

/// Configuration for TensorFlow Lite interpreter performance.
///
/// Controls delegate usage and threading for CPU/GPU acceleration.
///
/// Example:
/// ```dart
/// // Default (XNNPACK enabled with auto thread detection)
/// final detector = FaceDetector();
/// await detector.initialize();
///
/// // XNNPACK with custom threads
/// final detector = FaceDetector();
/// await detector.initialize(
///   performanceConfig: PerformanceConfig.xnnpack(numThreads: 2),
/// );
///
/// // Disable XNNPACK (not recommended)
/// final detector = FaceDetector();
/// await detector.initialize(
///   performanceConfig: PerformanceConfig.disabled,
/// );
/// ```
class PerformanceConfig {
  /// Performance mode controlling delegate selection.
  final PerformanceMode mode;

  /// Number of threads for XNNPACK delegate.
  ///
  /// - null: Auto-detect optimal count (min(4, Platform.numberOfProcessors))
  /// - 0: No thread pool (single-threaded, good for tiny models)
  /// - 1-8: Explicit thread count
  ///
  /// Diminishing returns after 4 threads for typical models.
  /// Only applies when mode is [PerformanceMode.xnnpack] or [PerformanceMode.auto].
  final int? numThreads;

  /// Creates a performance configuration.
  ///
  /// Parameters:
  /// - [mode]: Performance mode. Default: [PerformanceMode.xnnpack]
  /// - [numThreads]: Number of threads (null for auto-detection)
  const PerformanceConfig({
    this.mode = PerformanceMode.xnnpack,
    this.numThreads,
  });

  /// Creates config with XNNPACK enabled and auto thread detection.
  const PerformanceConfig.xnnpack({this.numThreads})
      : mode = PerformanceMode.xnnpack;

  /// Creates config with auto mode (currently uses XNNPACK).
  const PerformanceConfig.auto({this.numThreads}) : mode = PerformanceMode.auto;

  /// Default configuration (no delegates, backward compatible).
  static const PerformanceConfig disabled = PerformanceConfig(
    mode: PerformanceMode.disabled,
  );
}

/// Connections between eye contour landmarks for rendering the visible eyeball outline.
///
/// These define which of the 71 eye contour points should be connected with lines
/// to form the visible eye shape (eyelids). The connections form the outline of the
/// visible eyeball by connecting the first 15 eye contour landmarks.
///
/// Based on MediaPipe's iris rendering configuration.
///
/// Example usage:
/// ```dart
/// for (final connection in eyeLandmarkConnections) {
///   final p1 = iris.eyeContour[connection[0]];
///   final p2 = iris.eyeContour[connection[1]];
///   // Draw line from p1 to p2
/// }
/// ```
const List<List<int>> eyeLandmarkConnections = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [9, 10],
  [10, 11],
  [11, 12],
  [12, 13],
  [13, 14],
  [0, 9],
  [8, 14]
];

/// Number of eye contour points that form the visible eyeball outline.
///
/// The first 15 points of the 71-point eye contour represent the visible
/// eyelid outline. The remaining 56 points are used for eyebrows and
/// additional tracking halos around the eye region.
const int kMaxEyeLandmark = 15;

/// A point with x, y, and optional z coordinates.
///
/// Used to represent landmarks with optional depth information.
/// The x and y coordinates are in absolute pixel positions relative to the original image.
/// The z coordinate represents relative depth (scale-dependent) when 3D computation is enabled.
///
/// When [z] is null, this represents a 2D point. When [z] is non-null, it represents
/// a 3D point with depth information.
class Point {
  /// The x-coordinate in absolute pixels.
  final double x;

  /// The y-coordinate in absolute pixels.
  final double y;

  /// The z-coordinate representing relative depth, or null for 2D points.
  ///
  /// This is a scale-dependent depth value. The magnitude depends on the face size
  /// and alignment used during detection. Negative values indicate points closer to
  /// the camera, positive values indicate points further away.
  ///
  /// Will be null for 2D-only landmarks (such as face detection keypoints).
  /// Face mesh and iris landmarks always include z-coordinates.
  final double? z;

  /// Creates a point with the given x, y, and optional z coordinates.
  const Point(this.x, this.y, [this.z]);

  /// Whether this point has depth information (z-coordinate).
  ///
  /// Returns true if z-coordinate is non-null, false otherwise.
  bool get is3D => z != null;

  @override
  String toString() => z != null ? 'Point($x, $y, $z)' : 'Point($x, $y)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Point &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y &&
          z == other.z;

  @override
  int get hashCode => Object.hash(x, y, z);

  /// Converts this point to a map for isolate serialization.
  Map<String, dynamic> toMap() => {'x': x, 'y': y, if (z != null) 'z': z};

  /// Creates a point from a map (isolate deserialization).
  factory Point.fromMap(Map<String, dynamic> map) =>
      Point(map['x'] as double, map['y'] as double, map['z'] as double?);
}

/// A 468-point face mesh with optional depth information.
///
/// Encapsulates the MediaPipe face mesh data. Each point has x and y coordinates
/// in absolute pixels, and an optional z coordinate representing depth.
/// 3D coordinates are always computed for face mesh landmarks.
///
/// The mesh contains 468 points following MediaPipe's canonical face mesh topology,
/// providing detailed geometry for facial features including eyes, eyebrows, nose,
/// mouth, and face contours.
///
/// Example:
/// ```dart
/// final FaceMesh? mesh = face.mesh;
/// if (mesh != null) {
///   // Access points
///   final points = mesh.points;
///   print('Nose tip: (${points[1].x}, ${points[1].y}, ${points[1].z})');
///
///   // Direct indexed access
///   final noseTip = mesh[1];
///   if (noseTip.z != null) {
///     print('Depth available: ${noseTip.z}');
///   }
/// }
/// ```
class FaceMesh {
  final List<Point> _points;

  /// Creates a face mesh from 468 points.
  FaceMesh(this._points) : assert(_points.length == kMeshPoints);

  /// The 468 mesh points with depth information.
  ///
  /// Each point has x, y, and z coordinates. The z coordinate represents
  /// relative depth and is always computed for face mesh landmarks.
  List<Point> get points => _points;

  /// Returns the point at the given index.
  ///
  /// Example: `mesh[1]` returns the nose tip.
  Point operator [](int index) => _points[index];

  /// The number of points in the mesh (always 468).
  int get length => _points.length;

  @override
  String toString() => 'FaceMesh(${_points.length} points)';

  /// Converts this mesh to a map for isolate serialization.
  Map<String, dynamic> toMap() =>
      {'points': _points.map((p) => p.toMap()).toList()};

  /// Creates a face mesh from a map (isolate deserialization).
  factory FaceMesh.fromMap(Map<String, dynamic> map) => FaceMesh(
        (map['points'] as List).map((p) => Point.fromMap(p)).toList(),
      );
}

/// Comprehensive eye tracking data including iris center, iris contour, and eye mesh.
///
/// Each eye contains:
/// - An iris center point for gaze tracking
/// - Four iris contour points outlining the iris boundary
/// - Eye mesh landmarks covering the entire eye region (71 points including eyelid contour)
///
/// All coordinates are in absolute pixel positions relative to the original image.
///
/// **Naming clarity:**
/// - [irisCenter]: Iris center point for gaze tracking
/// - [irisContour]: 4 points outlining the iris boundary (the colored part of the eye)
/// - [contour]: 15 points outlining the eyelid (visible eyeball outline)
/// - [mesh]: 71 points covering the entire eye region (eyelids, eyebrows, tracking halos)
///
/// The eye mesh landmarks provide detailed geometry for the entire eye region including
/// eyelids, eye corners, and surrounding area. These can be useful for blink detection,
/// eye openness estimation, and advanced eye tracking applications.
///
/// See also:
/// - [EyePair] for accessing both eyes' data
class Eye {
  /// Center point of the iris in absolute pixel coordinates for gaze tracking.
  final Point irisCenter;

  /// Four points outlining the iris boundary in absolute pixel coordinates.
  ///
  /// These points form the contour of the iris and can be used to estimate
  /// the iris size and shape.
  final List<Point> irisContour;

  /// Complete eye mesh with 71 landmark points in absolute pixel coordinates.
  ///
  /// These 71 points form a detailed mesh of the entire eye region including
  /// eyelids, eye corners, eyebrows, and surrounding area. They provide comprehensive
  /// geometry information about the eye beyond just the iris.
  ///
  /// **Structure:**
  /// - First 15 points: Visible eyelid outline (use [contour] to access these)
  /// - Remaining 56 points: Eyebrow landmarks and tracking halos
  ///
  /// Useful for:
  /// - Eyelid position tracking
  /// - Blink detection
  /// - Eye openness estimation
  /// - Eyebrow tracking
  /// - Detailed eye region analysis
  ///
  /// See also:
  /// - [contour] for just the 15-point eyelid outline
  /// - [eyeLandmarkConnections] for connecting the eyelid points
  final List<Point> mesh;

  /// Creates an eye with iris center point, iris contour, and eye mesh landmarks.
  const Eye({
    required this.irisCenter,
    required this.irisContour,
    this.mesh = const <Point>[],
  });

  /// The visible eyelid contour (first 15 points of the mesh).
  ///
  /// These 15 points form the outline of the visible eyeball (upper and lower eyelids).
  /// Use [eyeLandmarkConnections] to determine which points to connect with lines
  /// when rendering the eyelid outline.
  ///
  /// The remaining points in [mesh] (indices 15-70) represent eyebrows and
  /// additional tracking halos around the eye region.
  ///
  /// Example:
  /// ```dart
  /// final eyelidPoints = eye.contour;
  /// for (final connection in eyeLandmarkConnections) {
  ///   final p1 = eyelidPoints[connection[0]];
  ///   final p2 = eyelidPoints[connection[1]];
  ///   canvas.drawLine(p1, p2, paint);
  /// }
  /// ```
  List<Point> get contour =>
      mesh.length >= kMaxEyeLandmark ? mesh.sublist(0, kMaxEyeLandmark) : mesh;

  /// Converts this eye to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'irisCenter': irisCenter.toMap(),
        'irisContour': irisContour.map((p) => p.toMap()).toList(),
        'mesh': mesh.map((p) => p.toMap()).toList(),
      };

  /// Creates an eye from a map (isolate deserialization).
  factory Eye.fromMap(Map<String, dynamic> map) => Eye(
        irisCenter: Point.fromMap(map['irisCenter']),
        irisContour:
            (map['irisContour'] as List).map((p) => Point.fromMap(p)).toList(),
        mesh: (map['mesh'] as List).map((p) => Point.fromMap(p)).toList(),
      );
}

/// Eye tracking data for both eyes including iris and eye mesh landmarks.
///
/// Contains comprehensive eye tracking data for the left and right eyes. Each eye includes:
/// - Iris center point for gaze tracking
/// - Four iris contour points (iris boundary)
/// - Eye mesh landmarks (71 points covering the entire eye region)
///
/// Individual eye data may be null if not detected or if called with a detection
/// mode that doesn't include iris tracking.
///
/// Only available when using [FaceDetectionMode.full]. Returns null for
/// [FaceDetectionMode.fast] and [FaceDetectionMode.standard].
///
/// Example:
/// ```dart
/// final faces = await detector.detectFaces(imageBytes, mode: FaceDetectionMode.full);
/// final eyes = faces.first.eyes;
/// if (eyes != null) {
///   final leftIrisCenter = eyes.leftEye?.irisCenter;
///   final rightIrisContour = eyes.rightEye?.irisContour;
///   final leftEyeMesh = eyes.leftEye?.mesh;
/// }
/// ```
///
/// See also:
/// - [Eye] for the structure of individual eye data
class EyePair {
  /// The left eye data, or null if not detected.
  final Eye? leftEye;

  /// The right eye data, or null if not detected.
  final Eye? rightEye;

  /// Creates an eye pair with optional left and right eye data.
  const EyePair({this.leftEye, this.rightEye});

  /// Converts this eye pair to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        if (leftEye != null) 'leftEye': leftEye!.toMap(),
        if (rightEye != null) 'rightEye': rightEye!.toMap(),
      };

  /// Creates an eye pair from a map (isolate deserialization).
  factory EyePair.fromMap(Map<String, dynamic> map) => EyePair(
        leftEye: map['leftEye'] != null ? Eye.fromMap(map['leftEye']) : null,
        rightEye: map['rightEye'] != null ? Eye.fromMap(map['rightEye']) : null,
      );
}

/// Facial landmark points with convenient named access.
///
/// Provides 6 key facial feature points in pixel coordinates with both
/// named property access and map-like access for backwards compatibility.
///
/// All coordinates are in absolute pixel positions relative to the original image.
///
/// Example:
/// ```dart
/// final landmarks = face.landmarks;
/// print('Left eye: (${landmarks.leftEye?.x}, ${landmarks.leftEye?.y})');
/// print('Nose: (${landmarks.noseTip?.x}, ${landmarks.noseTip?.y})');
///
/// // Map-like access also works for backwards compatibility
/// final leftEye = landmarks[FaceLandmarkType.leftEye];
/// ```
class FaceLandmarks {
  final Map<FaceLandmarkType, Point> _landmarks;

  /// Creates facial landmarks from a map of landmark types to points.
  const FaceLandmarks(this._landmarks);

  /// Left eye center point in pixel coordinates.
  Point? get leftEye => _landmarks[FaceLandmarkType.leftEye];

  /// Right eye center point in pixel coordinates.
  Point? get rightEye => _landmarks[FaceLandmarkType.rightEye];

  /// Nose tip point in pixel coordinates.
  Point? get noseTip => _landmarks[FaceLandmarkType.noseTip];

  /// Mouth center point in pixel coordinates.
  Point? get mouth => _landmarks[FaceLandmarkType.mouth];

  /// Left eye tragion point in pixel coordinates.
  ///
  /// The tragion is the notch just above the ear canal opening.
  Point? get leftEyeTragion => _landmarks[FaceLandmarkType.leftEyeTragion];

  /// Right eye tragion point in pixel coordinates.
  ///
  /// The tragion is the notch just above the ear canal opening.
  Point? get rightEyeTragion => _landmarks[FaceLandmarkType.rightEyeTragion];

  /// Access landmark by type (backwards compatible with map access).
  ///
  /// Example:
  /// ```dart
  /// final leftEye = landmarks[FaceLandmarkType.leftEye];
  /// ```
  Point? operator [](FaceLandmarkType type) => _landmarks[type];

  /// All landmark points as an iterable (backwards compatible with map.values).
  ///
  /// Example:
  /// ```dart
  /// for (final point in landmarks.values) {
  ///   print('(${point.x}, ${point.y})');
  /// }
  /// ```
  Iterable<Point> get values => _landmarks.values;

  /// All available landmark types in this detection.
  Iterable<FaceLandmarkType> get keys => _landmarks.keys;

  /// Returns all landmarks as an unmodifiable map.
  ///
  /// Use this when you need explicit Map type for compatibility.
  Map<FaceLandmarkType, Point> toMap() => Map.unmodifiable(_landmarks);

  /// Converts landmarks to a serializable map for isolate transfer.
  Map<String, dynamic> toSerializableMap() => {
        for (final entry in _landmarks.entries)
          entry.key.name: entry.value.toMap(),
      };

  /// Creates landmarks from a serializable map (isolate deserialization).
  factory FaceLandmarks.fromSerializableMap(Map<String, dynamic> map) {
    final landmarks = <FaceLandmarkType, Point>{};
    for (final entry in map.entries) {
      final type = FaceLandmarkType.values.firstWhere((t) => t.name == entry.key);
      landmarks[type] = Point.fromMap(entry.value);
    }
    return FaceLandmarks(landmarks);
  }
}

/// Face bounding box with corner points in pixel coordinates.
///
/// Represents a rectangular bounding box around a detected face with convenient
/// access to corner points, dimensions, and center coordinates.
///
/// All coordinates are in absolute pixel positions relative to the original image.
///
/// Example:
/// ```dart
/// final boundingBox = face.boundingBox;
/// print('Width: ${boundingBox.width}, Height: ${boundingBox.height}');
/// print('Top-left corner: (${boundingBox.topLeft.x}, ${boundingBox.topLeft.y})');
/// print('Center: (${boundingBox.center.x}, ${boundingBox.center.y})');
/// ```
class BoundingBox {
  /// Top-left corner point in absolute pixel coordinates.
  final Point topLeft;

  /// Top-right corner point in absolute pixel coordinates.
  final Point topRight;

  /// Bottom-right corner point in absolute pixel coordinates.
  final Point bottomRight;

  /// Bottom-left corner point in absolute pixel coordinates.
  final Point bottomLeft;

  /// Creates a bounding box with four corner points.
  ///
  /// Points should be in order: top-left, top-right, bottom-right, bottom-left.
  const BoundingBox({
    required this.topLeft,
    required this.topRight,
    required this.bottomRight,
    required this.bottomLeft,
  });

  /// The four corner points as a list in order: top-left, top-right,
  /// bottom-right, bottom-left.
  ///
  /// Useful for iteration or when you need all corners at once.
  List<Point> get corners => [topLeft, topRight, bottomRight, bottomLeft];

  /// Width of the bounding box in pixels.
  double get width => topRight.x - topLeft.x;

  /// Height of the bounding box in pixels.
  double get height => bottomLeft.y - topLeft.y;

  /// Center point of the bounding box in absolute pixel coordinates.
  Point get center => Point(
        (topLeft.x + topRight.x + bottomRight.x + bottomLeft.x) / 4,
        (topLeft.y + topRight.y + bottomRight.y + bottomLeft.y) / 4,
      );

  /// Converts this bounding box to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'topLeft': topLeft.toMap(),
        'topRight': topRight.toMap(),
        'bottomRight': bottomRight.toMap(),
        'bottomLeft': bottomLeft.toMap(),
      };

  /// Creates a bounding box from a map (isolate deserialization).
  factory BoundingBox.fromMap(Map<String, dynamic> map) => BoundingBox(
        topLeft: Point.fromMap(map['topLeft']),
        topRight: Point.fromMap(map['topRight']),
        bottomRight: Point.fromMap(map['bottomRight']),
        bottomLeft: Point.fromMap(map['bottomLeft']),
      );
}

/// Outputs for a single detected face.
///
/// [boundingBox] is the face bounding box in pixel coordinates.
/// [landmarks] provides convenient access to 6 key facial landmarks (eyes, nose, mouth).
/// [mesh] contains 468 facial landmarks as pixel coordinates.
/// [eyes] contains iris center, iris contour, and eye mesh landmarks for both eyes.
class Face {
  final Detection _detection;

  /// The 468-point face mesh with optional depth information.
  ///
  /// The mesh provides convenient access to points with x, y coordinates and
  /// an optional z coordinate representing depth when 3D computation is enabled.
  ///
  /// The 468 points follow MediaPipe's canonical face mesh topology, providing
  /// detailed geometry for facial features including eyes, eyebrows, nose, mouth,
  /// and face contours.
  ///
  /// This is null when [FaceDetector.detectFaces] is called with [FaceDetectionMode.fast].
  /// Use [FaceDetectionMode.standard] or [FaceDetectionMode.full] to populate mesh data.
  ///
  /// Example:
  /// ```dart
  /// final FaceMesh? mesh = face.mesh;
  /// if (mesh != null) {
  ///   final points = mesh.points;
  ///   for (final point in points) {
  ///     if (point.is3D) {
  ///       print('Point with depth: (${point.x}, ${point.y}, ${point.z})');
  ///     } else {
  ///       print('Point: (${point.x}, ${point.y})');
  ///     }
  ///   }
  /// }
  /// ```
  ///
  /// See also:
  /// - [FaceMesh] for the mesh class documentation
  /// - [kMeshPoints] for the expected mesh point count (468)
  /// - [eyes] for iris and eye mesh landmarks
  final FaceMesh? mesh;

  /// Raw iris and eye mesh landmark points with depth information.
  ///
  /// Contains landmarks for both eyes (left eye data followed by right eye data).
  /// The iris model outputs 76 points per eye in this order:
  /// - 71 eye mesh landmarks (detailed eye region geometry)
  /// - 5 iris keypoints (1 center + 4 contour points)
  ///
  /// Total: 152 points (76 per eye × 2 eyes).
  ///
  /// Each point contains `x`, `y`, and `z` coordinates where:
  /// - `x` and `y` are absolute pixel positions in the original image
  /// - `z` represents relative depth (scale-dependent)
  ///
  /// This list is empty when [FaceDetector.detectFaces] is called with
  /// [FaceDetectionMode.fast] or [FaceDetectionMode.standard]. Use
  /// [FaceDetectionMode.full] to enable iris tracking.
  ///
  /// For a more convenient structured API, use the [eyes] getter instead,
  /// which returns an [EyePair] with separate left/right eye data including
  /// parsed iris center, iris contour, and eye mesh landmarks.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes, mode: FaceDetectionMode.full);
  /// if (faces.isNotEmpty && faces.first.irisPoints.isNotEmpty) {
  ///   // Use structured API for easier access
  ///   final eyes = faces.first.eyes;
  ///   final leftIrisCenter = eyes?.leftEye?.irisCenter;
  ///   final leftEyeMesh = eyes?.leftEye?.mesh;
  ///
  ///   // Access 3D depth information
  ///   if (leftIrisCenter.is3D) {
  ///     print('Iris depth: ${leftIrisCenter.z}');
  ///   }
  /// }
  /// ```
  final List<Point> irisPoints;

  /// The dimensions of the original source image.
  ///
  /// This size is used internally to convert normalized coordinates to pixel
  /// coordinates for [boundingBox], [landmarks], [mesh], and [eyes].
  ///
  /// All coordinate data in [Face] is already scaled to these dimensions,
  /// so users typically don't need to use this field directly unless performing
  /// custom coordinate transformations.
  final Size originalSize;

  /// Creates a face detection result with bounding box, landmarks, and optional mesh/eye data.
  ///
  /// This constructor is typically called internally by [FaceDetector.detectFaces].
  /// Most users should not need to construct [Face] instances directly.
  ///
  /// The [detection] contains the bounding box and coarse facial keypoints.
  /// The [mesh] contains the 468-point face mesh with 3D coordinates (null if not computed).
  /// The [irisPoints] contains iris and eye mesh keypoints with 3D coordinates (empty if not computed).
  /// The [originalSize] specifies the dimensions of the source image for coordinate mapping.
  Face({
    required Detection detection,
    required this.mesh,
    required List<Point> irises,
    required this.originalSize,
  })  : _detection = detection,
        irisPoints = irises;

  /// Parses raw iris and eye contour points into a structured Eye object.
  ///
  /// The iris landmark model outputs 76 points per eye in this order:
  /// - First 71 points: eye mesh landmarks (detailed eye region geometry)
  /// - Last 5 points: iris landmarks (center + 4 contour points)
  ///
  /// For backward compatibility, also handles legacy format (5 points only).
  ///
  /// Identifies the iris center as the point with minimum sum of squared
  /// distances to all other iris points, and treats the remaining 4 as contour.
  ///
  /// Returns null if the input contains fewer than 5 points.
  static Eye? _parseIris(List<Point> points) {
    if (points.length < 5) return null;

    List<Point> eyeMesh;
    List<Point> irisPoints;

    if (points.length == 76) {
      eyeMesh = points.sublist(0, 71);
      irisPoints = points.sublist(71, 76);
    } else if (points.length > 5) {
      final irisStart = points.length - 5;
      eyeMesh = points.sublist(0, irisStart);
      irisPoints = points.sublist(irisStart);
    } else {
      eyeMesh = const <Point>[];
      irisPoints = points;
    }
    int centerIdx = 0;
    double minDistSum = double.infinity;

    for (int i = 0; i < 5; i++) {
      double distSum = 0;
      for (int j = 0; j < 5; j++) {
        if (i == j) continue;
        final dx = irisPoints[j].x - irisPoints[i].x;
        final dy = irisPoints[j].y - irisPoints[i].y;
        distSum += dx * dx + dy * dy;
      }
      if (distSum < minDistSum) {
        minDistSum = distSum;
        centerIdx = i;
      }
    }

    final center = irisPoints[centerIdx];
    final contour = <Point>[];
    for (int i = 0; i < 5; i++) {
      if (i != centerIdx) contour.add(irisPoints[i]);
    }

    return Eye(irisCenter: center, irisContour: contour, mesh: eyeMesh);
  }

  /// Comprehensive eye tracking data for both eyes.
  ///
  /// Returns an [EyePair] containing left and right eye data with:
  /// - Iris center points
  /// - Iris contour boundaries (4 points per iris)
  /// - Eye mesh landmarks (71 points per eye covering the entire eye region)
  ///
  /// Individual eyes may be null if not detected.
  ///
  /// Returns null when [FaceDetector.detectFaces] is called with
  /// [FaceDetectionMode.fast] or [FaceDetectionMode.standard]. Use
  /// [FaceDetectionMode.full] to enable eye tracking.
  ///
  /// For raw iris point data, use [irisPoints] instead.
  ///
  /// Example:
  /// ```dart
  /// final eyes = face.eyes;
  /// final leftIrisCenter = eyes?.leftEye?.irisCenter;
  /// final leftIrisContour = eyes?.leftEye?.irisContour;
  /// final leftContour = eyes?.leftEye?.contour;
  /// final rightIrisCenter = eyes?.rightEye?.irisCenter;
  /// ```
  EyePair? get eyes {
    if (irisPoints.isEmpty) return null;

    Eye? leftEye;
    Eye? rightEye;

    if (irisPoints.length == 152) {
      leftEye = _parseIris(irisPoints.sublist(0, 76));
      rightEye = _parseIris(irisPoints.sublist(76, 152));
    } else if (irisPoints.length == 76) {
      leftEye = _parseIris(irisPoints);
    } else if (irisPoints.length == 10) {
      leftEye = _parseIris(irisPoints.sublist(0, 5));
      rightEye = _parseIris(irisPoints.sublist(5, 10));
    } else if (irisPoints.length > 10 && irisPoints.length.isEven) {
      final int pointsPerEye = irisPoints.length ~/ 2;
      leftEye = _parseIris(irisPoints.sublist(0, pointsPerEye));
      rightEye = _parseIris(irisPoints.sublist(pointsPerEye));
    } else if (irisPoints.length >= 5) {
      leftEye = _parseIris(irisPoints);
    }

    if (leftEye == null && rightEye == null) return null;

    return EyePair(leftEye: leftEye, rightEye: rightEye);
  }

  /// The face bounding box in pixel coordinates.
  ///
  /// Provides convenient access to corner points, dimensions, and center
  /// of the bounding box. Use [BoundingBox.topLeft], [BoundingBox.topRight],
  /// [BoundingBox.bottomRight], [BoundingBox.bottomLeft] to access individual
  /// corners, or [BoundingBox.width], [BoundingBox.height], and
  /// [BoundingBox.center] for dimensions and center point.
  ///
  /// Example:
  /// ```dart
  /// final boundingBox = face.boundingBox;
  /// print('Face at (${boundingBox.center.x}, ${boundingBox.center.y})');
  /// print('Size: ${boundingBox.width} x ${boundingBox.height}');
  /// ```
  BoundingBox get boundingBox {
    final RectF r = _detection.boundingBox;
    final double w = originalSize.width.toDouble();
    final double h = originalSize.height.toDouble();
    return BoundingBox(
      topLeft: Point(r.xmin * w, r.ymin * h),
      topRight: Point(r.xmax * w, r.ymin * h),
      bottomRight: Point(r.xmax * w, r.ymax * h),
      bottomLeft: Point(r.xmin * w, r.ymax * h),
    );
  }

  /// Facial landmark positions in pixel coordinates.
  ///
  /// Returns a [FaceLandmarks] object with convenient named access to key
  /// facial features. Use named properties like [FaceLandmarks.leftEye],
  /// [FaceLandmarks.rightEye], [FaceLandmarks.noseTip], etc. for cleaner code.
  ///
  /// When iris tracking is enabled ([FaceDetectionMode.full]), the left and
  /// right eye landmarks are automatically replaced with the precise iris centers
  /// for improved accuracy.
  ///
  /// Example:
  /// ```dart
  /// final landmarks = face.landmarks;
  /// final leftEye = landmarks.leftEye;
  /// final noseTip = landmarks.noseTip;
  /// print('Left eye: (${leftEye?.x}, ${leftEye?.y})');
  /// ```
  ///
  /// For backwards compatibility, you can still use map-like access:
  /// ```dart
  /// final leftEye = landmarks[FaceLandmarkType.leftEye];
  /// for (final point in landmarks.values) { ... }
  /// ```
  FaceLandmarks get landmarks {
    final Map<FaceLandmarkType, Point> landmarkMap = _detection.landmarks;

    final EyePair? eyeData = eyes;
    if (eyeData != null) {
      if (eyeData.leftEye?.irisCenter != null) {
        landmarkMap[FaceLandmarkType.leftEye] = eyeData.leftEye!.irisCenter;
      }
      if (eyeData.rightEye?.irisCenter != null) {
        landmarkMap[FaceLandmarkType.rightEye] = eyeData.rightEye!.irisCenter;
      }
    }

    return FaceLandmarks(landmarkMap);
  }

  /// Converts this face to a map for isolate serialization.
  ///
  /// Serializes all face data including detection, mesh, iris points, and
  /// original image size. Use [Face.fromMap] to reconstruct.
  Map<String, dynamic> toMap() => {
        'detection': _detection.toMap(),
        if (mesh != null) 'mesh': mesh!.toMap(),
        'irisPoints': irisPoints.map((p) => p.toMap()).toList(),
        'originalSize': {
          'width': originalSize.width,
          'height': originalSize.height
        },
      };

  /// Creates a face from a map (isolate deserialization).
  ///
  /// Reconstructs a Face object from data serialized via [toMap].
  factory Face.fromMap(Map<String, dynamic> map) => Face(
        detection: Detection.fromMap(map['detection']),
        mesh: map['mesh'] != null ? FaceMesh.fromMap(map['mesh']) : null,
        irises: (map['irisPoints'] as List).map((p) => Point.fromMap(p)).toList(),
        originalSize:
            Size(map['originalSize']['width'], map['originalSize']['height']),
      );
}

/// The expected number of landmark points in a complete face mesh.
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
  AlignedFace({
    required this.cx,
    required this.cy,
    required this.size,
    required this.theta,
    required this.faceCrop,
  });
}

/// Aligned face crop data holder for OpenCV-based processing.
///
/// Similar to [AlignedFace] but holds a cv.Mat instead of img.Image.
/// Used internally by the OpenCV-accelerated detection pipeline.
class AlignedFaceFromMat {
  /// X coordinate of the face center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of the face center in absolute pixel coordinates.
  final double cy;

  /// Length of the square crop edge in absolute pixels.
  final double size;

  /// Rotation applied to align the face, in radians.
  final double theta;

  /// The aligned face crop as cv.Mat. Caller must dispose when done.
  final cv.Mat faceCrop;

  /// Creates an aligned face crop with cv.Mat instead of img.Image.
  AlignedFaceFromMat({
    required this.cx,
    required this.cy,
    required this.size,
    required this.theta,
    required this.faceCrop,
  });
}

/// Axis-aligned rectangle with normalized coordinates.
///
/// Values are expressed as fractions of the original image dimensions
/// (0.0 - 1.0). Utilities are provided to scale and expand the rectangle.
class RectF {
  /// Minimum X and Y plus maximum X and Y extents.
  final double xmin, ymin, xmax, ymax;

  /// Creates a normalized rectangle given its minimum and maximum extents.
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

  /// Converts this rect to a map for isolate serialization.
  Map<String, dynamic> toMap() =>
      {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax};

  /// Creates a rect from a map (isolate deserialization).
  factory RectF.fromMap(Map<String, dynamic> map) => RectF(
        map['xmin'] as double,
        map['ymin'] as double,
        map['xmax'] as double,
        map['ymax'] as double,
      );
}

/// Raw detection output from the face detector containing the bounding box and keypoints.
class Detection {
  /// Normalized bounding box for the face.
  final RectF boundingBox;

  /// Confidence score for the detection.
  final double score;

  /// Flattened landmark coordinates `[x0, y0, x1, y1, ...]` normalized 0-1.
  final List<double> keypointsXY;

  /// Original image dimensions used to denormalize landmarks.
  final Size? imageSize;

  /// Creates a detection with normalized geometry and optional source size.
  Detection({
    required this.boundingBox,
    required this.score,
    required this.keypointsXY,
    this.imageSize,
  });

  /// Convenience accessor for `[keypointsXY]` by index.
  double operator [](int i) => keypointsXY[i];

  /// Returns facial landmarks in pixel coordinates keyed by landmark type.
  Map<FaceLandmarkType, Point> get landmarks {
    final Size? sz = imageSize;
    if (sz == null) {
      throw StateError(
        'Detection.imageSize is null; cannot produce pixel landmarks.',
      );
    }
    final double w = sz.width.toDouble(), h = sz.height.toDouble();
    final Map<FaceLandmarkType, Point> map = <FaceLandmarkType, Point>{};
    for (final FaceLandmarkType idx in FaceLandmarkType.values) {
      final double xn = keypointsXY[idx.index * 2];
      final double yn = keypointsXY[idx.index * 2 + 1];
      map[idx] = Point(xn * w, yn * h);
    }
    return map;
  }

  /// Converts this detection to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'boundingBox': boundingBox.toMap(),
        'score': score,
        'keypointsXY': keypointsXY,
        if (imageSize != null)
          'imageSize': {'width': imageSize!.width, 'height': imageSize!.height},
      };

  /// Creates a detection from a map (isolate deserialization).
  factory Detection.fromMap(Map<String, dynamic> map) => Detection(
        boundingBox: RectF.fromMap(map['boundingBox']),
        score: map['score'] as double,
        keypointsXY: (map['keypointsXY'] as List).cast<double>(),
        imageSize: map['imageSize'] != null
            ? Size(map['imageSize']['width'], map['imageSize']['height'])
            : null,
      );
}

/// Image tensor plus padding metadata used to undo letterboxing.
class ImageTensor {
  /// NHWC float tensor normalized to [-1, 1] expected by MediaPipe models.
  final Float32List tensorNHWC;

  /// Padding fractions `[top, bottom, left, right]` applied during resize.
  final List<double> padding;

  /// Target width and height passed to the model.
  final int width, height;

  /// Creates an image tensor paired with the padding used during resize.
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
  final RectF boundingBox;

  /// Flattened list of normalized keypoints `[x0, y0, ...]`.
  final List<double> keypointsXY;

  /// Constructs a decoded detection with its normalized bounding box and
  /// flattened landmark coordinates output by the face detector.
  DecodedBox(this.boundingBox, this.keypointsXY);
}
