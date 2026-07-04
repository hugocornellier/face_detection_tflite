/// Shared pure-Dart public types for face_detection_tflite. Imported and
/// re-exported by both the native and web entry points so user code sees a
/// single canonical definition for [Face], [Detection], [SegmentationMask],
/// etc. across platforms.
library;

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:flutter_litert/flutter_litert.dart'
    show BoundingBox, PerformanceConfig, Point;

import 'face_geometry.dart' show headEulerAnglesFromMesh, rollFromEyes;
import 'blendshape_input.dart' show Blendshape, kBlendshapeCount;

/// Identifies specific facial landmarks returned by face detection.
enum FaceLandmarkType {
  /// Left eye center.
  leftEye,

  /// Right eye center.
  rightEye,

  /// Tip of the nose.
  noseTip,

  /// Center of the mouth.
  mouth,

  /// Left ear tragion (the small notch in front of the ear).
  leftEyeTragion,

  /// Right ear tragion (the small notch in front of the ear).
  rightEyeTragion,
}

/// Named facial contours, mirroring Google ML Kit's `FaceContourType`.
///
/// Each value maps to an ordered group of points on the 468-point face mesh
/// (see [faceContourMeshIndices]); read them with [Face.getContour] or
/// [Face.contours]. Contours are only available when a mesh was computed
/// ([FaceDetectionMode.standard] or [FaceDetectionMode.full]); in
/// [FaceDetectionMode.fast] the getters return null.
///
/// As in ML Kit, "left"/"right" are subject-relative: [leftEye] is the
/// SUBJECT'S left eye, which appears on the RIGHT of an unmirrored image. This
/// matches the convention used by [Face.leftEyeOpenProbability].
enum FaceContourType {
  /// The 36-point oval outlining the whole face (silhouette).
  face,

  /// Upper edge of the subject's left eyebrow (5 points).
  leftEyebrowTop,

  /// Lower edge of the subject's left eyebrow (5 points).
  leftEyebrowBottom,

  /// Upper edge of the subject's right eyebrow (5 points).
  rightEyebrowTop,

  /// Lower edge of the subject's right eyebrow (5 points).
  rightEyebrowBottom,

  /// Outline of the subject's left eye (16 points).
  leftEye,

  /// Outline of the subject's right eye (16 points).
  rightEye,

  /// Top edge of the upper lip (11 points).
  upperLipTop,

  /// Bottom edge of the upper lip, i.e. the top of the mouth opening
  /// (11 points).
  upperLipBottom,

  /// Top edge of the lower lip, i.e. the bottom of the mouth opening
  /// (11 points).
  lowerLipTop,

  /// Bottom edge of the lower lip (11 points).
  lowerLipBottom,

  /// Bridge of the nose, from between the eyes down to the tip (6 points).
  noseBridge,

  /// Base of the nose across the nostrils (5 points).
  noseBottom,

  /// A single point at the center of the subject's left cheek.
  leftCheek,

  /// A single point at the center of the subject's right cheek.
  rightCheek,
}

/// Face detection model variants.
enum FaceDetectionModel {
  /// Front-facing camera model, tuned for close-range selfie framing.
  frontCamera,

  /// Rear-facing camera model, tuned for longer-range scenes.
  backCamera,

  /// Short-range model for faces near the camera.
  shortRange,

  /// Full-range model covering near and far faces.
  full,

  /// Full-range model with a sparse (faster, lighter) architecture.
  fullSparse,
}

/// Detection mode controls which features are computed.
enum FaceDetectionMode {
  /// Detection only: bounding box and the 6 keypoints, no mesh (fastest).
  fast,

  /// Adds the 468-point face mesh on top of [fast].
  standard,

  /// Adds iris and eye tracking on top of [standard] (most detailed, default).
  full,
}

/// Pixel format for RGBA output from segmentation masks.
enum PixelFormat {
  /// Red, green, blue, alpha byte order.
  rgba,

  /// Blue, green, red, alpha byte order.
  bgra,

  /// Alpha, red, green, blue byte order.
  argb,
}

/// Output format options for isolate-based segmentation transfer.
enum IsolateOutputFormat {
  /// Per-pixel float confidences in the range 0.0 to 1.0.
  float32,

  /// Per-pixel 8-bit grayscale values in the range 0 to 255.
  uint8,

  /// Per-pixel binary mask (foreground or background).
  binary,
}

/// Error codes for segmentation operations.
enum SegmentationError {
  /// The model file could not be located.
  modelNotFound,

  /// The TFLite interpreter failed to initialize.
  interpreterCreationFailed,

  /// A hardware delegate was unavailable and inference fell back to CPU.
  delegateFallback,

  /// The input image could not be decoded.
  imageDecodeFailed,

  /// The input image was smaller than the model requires.
  imageTooSmall,

  /// A model tensor had an unexpected shape.
  unexpectedTensorShape,

  /// Inference failed at runtime.
  inferenceFailed,

  /// The operation ran out of memory.
  outOfMemory,
}

/// Exception thrown by segmentation operations.
class SegmentationException implements Exception {
  /// Error code identifying the type of failure.
  final SegmentationError code;

  /// Human-readable description of the error.
  final String message;

  /// Underlying cause of the error, if available.
  final Object? cause;

  /// Creates a segmentation exception.
  const SegmentationException(this.code, this.message, [this.cause]);

  @override
  String toString() => 'SegmentationException($code): $message';
}

/// Selects which segmentation model variant to use.
enum SegmentationModel {
  /// General-purpose 256x256 person/background model.
  general,

  /// Landscape 144x256 model tuned for wide/video framing.
  landscape,

  /// 256x256 model that segments 6 body-part classes.
  multiclass,
}

/// Segmentation class indices for the multiclass model output.
class SegmentationClass {
  /// Background pixels.
  static const int background = 0;

  /// Hair pixels.
  static const int hair = 1;

  /// Body skin pixels.
  static const int bodySkin = 2;

  /// Face skin pixels.
  static const int faceSkin = 3;

  /// Clothes pixels.
  static const int clothes = 4;

  /// Other pixels (accessories, etc).
  static const int other = 5;
}

/// Configuration for segmentation operations.
///
/// On web, [performanceConfig] and [useIsolate] are accepted for API parity
/// but ignored at runtime: there are no platform delegates or background
/// isolates in the browser.
class SegmentationConfig {
  /// Which segmentation model to use.
  final SegmentationModel model;

  /// Performance configuration for the TFLite delegate (native only).
  final PerformanceConfig performanceConfig;

  /// Maximum output dimension for upsampled masks.
  final int maxOutputSize;

  /// Whether to validate model metadata on load.
  final bool validateModel;

  /// Whether to use IsolateInterpreter for inference (native only).
  final bool useIsolate;

  /// Creates a segmentation configuration.
  const SegmentationConfig({
    this.model = SegmentationModel.general,
    this.performanceConfig = const PerformanceConfig.auto(),
    this.maxOutputSize = 2048,
    this.validateModel = true,
    this.useIsolate = true,
  });

  /// CPU-only, limited output size, binary model.
  static const SegmentationConfig safe = SegmentationConfig(
    performanceConfig: PerformanceConfig.disabled,
    maxOutputSize: 1024,
  );

  /// Auto delegate, larger output, binary model.
  static const SegmentationConfig performance = SegmentationConfig(
    performanceConfig: PerformanceConfig.auto(),
    maxOutputSize: 2048,
  );

  /// Auto delegate, no isolate overhead, binary model.
  static const SegmentationConfig fast = SegmentationConfig(
    performanceConfig: PerformanceConfig.auto(),
    maxOutputSize: 2048,
    useIsolate: false,
  );
}

/// A segmentation probability mask indicating foreground vs background.
class SegmentationMask {
  /// Raw float32 mask data; access via [data] (defensive copy) or [at].
  final Float32List internalData;

  /// Mask width in pixels.
  final int width;

  /// Mask height in pixels.
  final int height;

  /// Original source image width.
  final int originalWidth;

  /// Original source image height.
  final int originalHeight;

  /// Padding `[top, bottom, left, right]` applied during letterboxing.
  final List<double> padding;

  /// Internal const constructor that accepts already-validated buffers.
  const SegmentationMask.internal({
    required this.internalData,
    required this.width,
    required this.height,
    required this.originalWidth,
    required this.originalHeight,
    required this.padding,
  });

  /// Creates a segmentation mask with validation.
  factory SegmentationMask({
    required Float32List data,
    required int width,
    required int height,
    required int originalWidth,
    required int originalHeight,
    List<double> padding = const [0.0, 0.0, 0.0, 0.0],
  }) {
    if (data.length != width * height) {
      throw ArgumentError(
        'Data length ${data.length} != width*height ${width * height}',
      );
    }
    return SegmentationMask.internal(
      internalData: Float32List.fromList(data),
      width: width,
      height: height,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      padding: List.unmodifiable(padding),
    );
  }

  /// Returns a defensive copy of the raw probability data (row-major).
  Float32List get data => Float32List.fromList(internalData);

  /// Returns probability at (x, y) in mask coordinates. Out-of-bounds → 0.0.
  double at(int x, int y) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0.0;
    return internalData[y * width + x];
  }

  /// Upsamples mask to target dimensions using bilinear interpolation.
  SegmentationMask upsample({
    int? targetWidth,
    int? targetHeight,
    int maxSize = 2048,
  }) {
    final tw = targetWidth ?? originalWidth;
    final th = targetHeight ?? originalHeight;
    final maxDim = math.max(tw, th);
    final scale = maxSize > 0 && maxDim > maxSize ? maxSize / maxDim : 1.0;
    final finalW = (tw * scale).round();
    final finalH = (th * scale).round();

    Float32List sourceData = internalData;
    int sourceW = width;
    int sourceH = height;

    final pt = padding[0], pb = padding[1], pl = padding[2], pr = padding[3];
    if (pt > 0 || pb > 0 || pl > 0 || pr > 0) {
      final validX0 = (pl * width).round();
      final validY0 = (pt * height).round();
      final validX1 = ((1.0 - pr) * width).round();
      final validY1 = ((1.0 - pb) * height).round();
      final validW = validX1 - validX0;
      final validH = validY1 - validY0;
      if (validW > 0 && validH > 0) {
        sourceData = Float32List(validW * validH);
        for (int y = 0; y < validH; y++) {
          for (int x = 0; x < validW; x++) {
            sourceData[y * validW + x] =
                internalData[(y + validY0) * width + (x + validX0)];
          }
        }
        sourceW = validW;
        sourceH = validH;
      }
    }

    final result = Float32List(finalW * finalH);
    final scaleX = sourceW / finalW;
    final scaleY = sourceH / finalH;
    for (int y = 0; y < finalH; y++) {
      final srcY = y * scaleY;
      final y0 = srcY.floor().clamp(0, sourceH - 1);
      final y1 = (y0 + 1).clamp(0, sourceH - 1);
      final yFrac = srcY - y0;
      for (int x = 0; x < finalW; x++) {
        final srcX = x * scaleX;
        final x0 = srcX.floor().clamp(0, sourceW - 1);
        final x1 = (x0 + 1).clamp(0, sourceW - 1);
        final xFrac = srcX - x0;
        final v00 = sourceData[y0 * sourceW + x0];
        final v10 = sourceData[y0 * sourceW + x1];
        final v01 = sourceData[y1 * sourceW + x0];
        final v11 = sourceData[y1 * sourceW + x1];
        final v0 = v00 * (1 - xFrac) + v10 * xFrac;
        final v1 = v01 * (1 - xFrac) + v11 * xFrac;
        result[y * finalW + x] = v0 * (1 - yFrac) + v1 * yFrac;
      }
    }

    return SegmentationMask.internal(
      internalData: result,
      width: finalW,
      height: finalH,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      padding: const [0.0, 0.0, 0.0, 0.0],
    );
  }

  /// Converts to 8-bit grayscale mask (0-255).
  Uint8List toUint8() {
    final result = Uint8List(width * height);
    for (int i = 0; i < internalData.length; i++) {
      result[i] = (internalData[i].clamp(0.0, 1.0) * 255).round();
    }
    return result;
  }

  /// Converts to binary mask (0 or 255) using threshold.
  Uint8List toBinary({double threshold = 0.5}) {
    final result = Uint8List(width * height);
    for (int i = 0; i < internalData.length; i++) {
      result[i] = internalData[i] >= threshold ? 255 : 0;
    }
    return result;
  }

  /// Converts to RGBA image with configurable colors.
  Uint8List toRgba({
    int foreground = 0xFFFFFFFF,
    int background = 0x00000000,
    PixelFormat format = PixelFormat.rgba,
    double threshold = 0.5,
  }) {
    final result = Uint8List(width * height * 4);
    final fg = _unpackColor(foreground, format);
    final bg = _unpackColor(background, format);
    for (int i = 0; i < internalData.length; i++) {
      final prob = internalData[i].clamp(0.0, 1.0);
      final offset = i * 4;
      if (threshold < 0) {
        for (int c = 0; c < 4; c++) {
          result[offset + c] = (fg[c] * prob + bg[c] * (1 - prob)).round();
        }
      } else {
        final color = prob >= threshold ? fg : bg;
        for (int c = 0; c < 4; c++) {
          result[offset + c] = color[c];
        }
      }
    }
    return result;
  }

  List<int> _unpackColor(int color, PixelFormat format) {
    switch (format) {
      case PixelFormat.rgba:
        return [
          (color >> 24) & 0xFF,
          (color >> 16) & 0xFF,
          (color >> 8) & 0xFF,
          color & 0xFF,
        ];
      case PixelFormat.bgra:
        return [
          (color >> 8) & 0xFF,
          (color >> 16) & 0xFF,
          (color >> 24) & 0xFF,
          color & 0xFF,
        ];
      case PixelFormat.argb:
        return [
          (color >> 16) & 0xFF,
          (color >> 8) & 0xFF,
          color & 0xFF,
          (color >> 24) & 0xFF,
        ];
    }
  }

  /// Serialization for isolate transfer.
  Map<String, dynamic> toMap() => {
    'data': internalData.toList(),
    'width': width,
    'height': height,
    'originalWidth': originalWidth,
    'originalHeight': originalHeight,
    'padding': padding,
  };

  /// Creates a SegmentationMask from a serialized map.
  factory SegmentationMask.fromMap(Map<String, dynamic> map) {
    final width = map['width'] as int;
    final height = map['height'] as int;
    final dataFormat = map['dataFormat'] as String? ?? 'float32';
    final rawData = map['data'] as List;

    Float32List data;
    switch (dataFormat) {
      case 'float32':
        data = Float32List.fromList(rawData.cast<double>());
        break;
      case 'uint8':
        final uint8List = rawData.cast<int>();
        data = Float32List(uint8List.length);
        for (int i = 0; i < uint8List.length; i++) {
          data[i] = uint8List[i] / 255.0;
        }
        break;
      case 'binary':
        final binaryList = rawData.cast<int>();
        data = Float32List(binaryList.length);
        for (int i = 0; i < binaryList.length; i++) {
          data[i] = binaryList[i] == 255 ? 1.0 : 0.0;
        }
        break;
      default:
        throw ArgumentError('Unknown data format: $dataFormat');
    }

    return SegmentationMask(
      data: data,
      width: width,
      height: height,
      originalWidth: map['originalWidth'] as int,
      originalHeight: map['originalHeight'] as int,
      padding:
          (map['padding'] as List?)?.cast<double>() ?? [0.0, 0.0, 0.0, 0.0],
    );
  }

  @override
  String toString() =>
      'SegmentationMask(${width}x$height, original: '
      '${originalWidth}x$originalHeight)';
}

/// Multiclass segmentation mask with per-class probabilities.
class MulticlassSegmentationMask extends SegmentationMask {
  /// Per-class probabilities after softmax, length = width*height*6.
  final Float32List internalClassData;

  /// Internal const constructor.
  const MulticlassSegmentationMask.internal({
    required super.internalData,
    required super.width,
    required super.height,
    required super.originalWidth,
    required super.originalHeight,
    required super.padding,
    required this.internalClassData,
  }) : super.internal();

  /// Creates a multiclass segmentation mask with validation.
  factory MulticlassSegmentationMask({
    required Float32List data,
    required int width,
    required int height,
    required int originalWidth,
    required int originalHeight,
    List<double> padding = const [0.0, 0.0, 0.0, 0.0],
    required Float32List classData,
  }) {
    if (data.length != width * height) {
      throw ArgumentError(
        'Data length ${data.length} != width*height ${width * height}',
      );
    }
    if (classData.length != width * height * 6) {
      throw ArgumentError(
        'ClassData length ${classData.length} != width*height*6 '
        '${width * height * 6}',
      );
    }
    return MulticlassSegmentationMask.internal(
      internalData: Float32List.fromList(data),
      width: width,
      height: height,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      padding: List.unmodifiable(padding),
      internalClassData: Float32List.fromList(classData),
    );
  }

  /// Returns a single-channel probability mask for the given class index.
  Float32List classMask(int classIndex) {
    if (classIndex < 0 || classIndex > 5) {
      throw RangeError.range(classIndex, 0, 5, 'classIndex');
    }
    final int numPixels = width * height;
    final result = Float32List(numPixels);
    for (int i = 0; i < numPixels; i++) {
      result[i] = internalClassData[i * 6 + classIndex];
    }
    return result;
  }

  /// Hair probability mask.
  Float32List get hairMask => classMask(SegmentationClass.hair);

  /// Body skin probability mask.
  Float32List get bodySkinMask => classMask(SegmentationClass.bodySkin);

  /// Face skin probability mask.
  Float32List get faceSkinMask => classMask(SegmentationClass.faceSkin);

  /// Clothes probability mask.
  Float32List get clothesMask => classMask(SegmentationClass.clothes);

  /// Other (accessories, etc.) probability mask.
  Float32List get otherMask => classMask(SegmentationClass.other);

  /// Background probability mask.
  Float32List get backgroundMask => classMask(SegmentationClass.background);

  @override
  String toString() =>
      'MulticlassSegmentationMask(${width}x$height, original: '
      '${originalWidth}x$originalHeight, 6 classes)';
}

/// The expected number of landmark points in a complete face mesh.
const int kMeshPoints = 468;

/// Number of eye contour points that form the visible eyeball outline.
const int kMaxEyeLandmark = 15;

/// Connections between eye contour landmarks for rendering the eyeball outline.
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
  [8, 14],
];

/// Ordered 468-point face-mesh indices for each [FaceContourType].
///
/// These are Google MediaPipe's canonical `FACEMESH_*` connection sets, chained
/// into a single ordered polyline per contour so that connecting consecutive
/// points renders the outline. Point counts and ordering therefore follow
/// MediaPipe, not ML Kit's proprietary contour model, but the semantic groups
/// (and the subject-relative left/right convention) match ML Kit's
/// `FaceContourType`. Every index is in `[0, 468)`.
const Map<FaceContourType, List<int>> faceContourMeshIndices = {
  // FACEMESH_FACE_OVAL, walked as one closed loop (last connects back to first).
  FaceContourType.face: [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, //
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, //
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  ],
  // FACEMESH_LEFT_EYEBROW, split into its upper and lower arcs (outer -> inner).
  FaceContourType.leftEyebrowTop: [300, 293, 334, 296, 336],
  FaceContourType.leftEyebrowBottom: [276, 283, 282, 295, 285],
  // FACEMESH_RIGHT_EYEBROW, split into its upper and lower arcs (outer -> inner).
  FaceContourType.rightEyebrowTop: [70, 63, 105, 66, 107],
  FaceContourType.rightEyebrowBottom: [46, 53, 52, 65, 55],
  // FACEMESH_LEFT_EYE / FACEMESH_RIGHT_EYE, full rings (outer corner, over the
  // top lid to the inner corner, back along the bottom lid).
  FaceContourType.leftEye: [
    263, 466, 388, 387, 386, 385, 384, 398, //
    362, 382, 381, 380, 374, 373, 390, 249,
  ],
  FaceContourType.rightEye: [
    33, 246, 161, 160, 159, 158, 157, 173, //
    133, 155, 154, 153, 145, 144, 163, 7,
  ],
  // FACEMESH_LIPS, four arcs: outer-upper/inner-upper/inner-lower/outer-lower.
  FaceContourType.upperLipTop: [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
  ],
  FaceContourType.upperLipBottom: [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
  ],
  FaceContourType.lowerLipTop: [
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
  ],
  FaceContourType.lowerLipBottom: [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
  ],
  // FACEMESH_NOSE: the dorsal midline and the nostril base.
  FaceContourType.noseBridge: [168, 6, 197, 195, 5, 4],
  FaceContourType.noseBottom: [98, 97, 2, 326, 327],
  // Single canonical cheek-center landmarks (no MediaPipe connection set).
  FaceContourType.leftCheek: [280],
  FaceContourType.rightCheek: [50],
};

List<Point> _clampedContour(List<Point> mesh) =>
    mesh.length >= kMaxEyeLandmark ? mesh.sublist(0, kMaxEyeLandmark) : mesh;

/// 468-point face mesh with optional depth information.
class FaceMesh {
  final List<Point> _points;

  /// Face-presence confidence from the mesh model, 0.0 to 1.0 (higher is more
  /// confident the crop is a face). Null when the model does not report it.
  final double? score;

  /// Creates a face mesh from 468 points.
  FaceMesh(this._points, {this.score}) : assert(_points.length == kMeshPoints);

  /// The 468 mesh points with depth.
  List<Point> get points => _points;

  /// Returns the point at the given index.
  Point operator [](int index) => _points[index];

  /// Number of points in the mesh (always 468).
  int get length => _points.length;

  @override
  String toString() => 'FaceMesh(${_points.length} points)';

  /// Converts this mesh to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
    'points': _points.map((p) => p.toMap()).toList(),
    if (score != null) 'score': score,
  };

  /// Creates a face mesh from a map.
  factory FaceMesh.fromMap(Map<String, dynamic> map) => FaceMesh(
    (map['points'] as List).map((p) => Point.fromMap(p)).toList(),
    score: (map['score'] as num?)?.toDouble(),
  );
}

/// Eye tracking data: iris center + iris contour + eye mesh.
class Eye {
  /// Center point of the iris in absolute pixel coordinates.
  final Point irisCenter;

  /// Four iris boundary points in absolute pixel coordinates.
  final List<Point> irisContour;

  /// Complete eye mesh (71 points) covering the eye region.
  final List<Point> mesh;

  final List<Point> _contourCache;

  /// Creates an eye with iris center, iris contour, and (optional) eye mesh.
  const Eye({
    required this.irisCenter,
    required this.irisContour,
    this.mesh = const <Point>[],
  }) : _contourCache = const <Point>[];

  Eye._withContour({
    required this.irisCenter,
    required this.irisContour,
    required this.mesh,
    required List<Point> contour,
  }) : _contourCache = contour;

  /// Creates an eye with pre-computed contour for faster repeated access.
  factory Eye.optimized({
    required Point irisCenter,
    required List<Point> irisContour,
    List<Point> mesh = const <Point>[],
  }) {
    final contour = _clampedContour(mesh);
    return Eye._withContour(
      irisCenter: irisCenter,
      irisContour: irisContour,
      mesh: mesh,
      contour: contour,
    );
  }

  /// The visible eyelid contour (first 15 points of the mesh).
  List<Point> get contour =>
      _contourCache.isNotEmpty ? _contourCache : _clampedContour(mesh);

  /// Converts this eye to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
    'irisCenter': irisCenter.toMap(),
    'irisContour': irisContour.map((p) => p.toMap()).toList(),
    'mesh': mesh.map((p) => p.toMap()).toList(),
  };

  /// Creates an eye from a map.
  factory Eye.fromMap(Map<String, dynamic> map) => Eye.optimized(
    irisCenter: Point.fromMap(map['irisCenter']),
    irisContour: (map['irisContour'] as List)
        .map((p) => Point.fromMap(p))
        .toList(),
    mesh: (map['mesh'] as List).map((p) => Point.fromMap(p)).toList(),
  );
}

/// Eye tracking data for both eyes.
class EyePair {
  /// Left eye data, or null if not detected.
  final Eye? leftEye;

  /// Right eye data, or null if not detected.
  final Eye? rightEye;

  /// Creates an eye pair.
  const EyePair({this.leftEye, this.rightEye});

  /// Converts this eye pair to a map.
  Map<String, dynamic> toMap() => {
    if (leftEye != null) 'leftEye': leftEye!.toMap(),
    if (rightEye != null) 'rightEye': rightEye!.toMap(),
  };

  /// Creates an eye pair from a map.
  factory EyePair.fromMap(Map<String, dynamic> map) => EyePair(
    leftEye: map['leftEye'] != null ? Eye.fromMap(map['leftEye']) : null,
    rightEye: map['rightEye'] != null ? Eye.fromMap(map['rightEye']) : null,
  );
}

/// Facial landmark points with named access.
class FaceLandmarks {
  final Map<FaceLandmarkType, Point> _landmarks;

  /// Creates facial landmarks from a map of types to points.
  const FaceLandmarks(this._landmarks);

  /// Left eye center.
  Point? get leftEye => _landmarks[FaceLandmarkType.leftEye];

  /// Right eye center.
  Point? get rightEye => _landmarks[FaceLandmarkType.rightEye];

  /// Nose tip.
  Point? get noseTip => _landmarks[FaceLandmarkType.noseTip];

  /// Mouth center.
  Point? get mouth => _landmarks[FaceLandmarkType.mouth];

  /// Left tragion.
  Point? get leftEyeTragion => _landmarks[FaceLandmarkType.leftEyeTragion];

  /// Right tragion.
  Point? get rightEyeTragion => _landmarks[FaceLandmarkType.rightEyeTragion];

  /// Map-style access by landmark type.
  Point? operator [](FaceLandmarkType type) => _landmarks[type];

  /// All landmark points.
  Iterable<Point> get values => _landmarks.values;

  /// All available landmark types.
  Iterable<FaceLandmarkType> get keys => _landmarks.keys;

  /// All landmarks as an unmodifiable map.
  Map<FaceLandmarkType, Point> toMap() => Map.unmodifiable(_landmarks);

  /// Serializable map for isolate transfer.
  Map<String, dynamic> toSerializableMap() => {
    for (final entry in _landmarks.entries) entry.key.name: entry.value.toMap(),
  };

  /// Creates landmarks from a serializable map.
  factory FaceLandmarks.fromSerializableMap(Map<String, dynamic> map) {
    final landmarks = <FaceLandmarkType, Point>{};
    for (final entry in map.entries) {
      final type = FaceLandmarkType.values.firstWhere(
        (t) => t.name == entry.key,
      );
      landmarks[type] = Point.fromMap(entry.value);
    }
    return FaceLandmarks(landmarks);
  }
}

/// Returns the iris point closest to the centroid.
Point irisCenterFromPoints(List<Point> pts) {
  if (pts.isEmpty) return const Point(0, 0, 0);
  if (pts.length == 1) return pts[0];
  double cx = 0, cy = 0;
  for (final p in pts) {
    cx += p.x;
    cy += p.y;
  }
  cx /= pts.length;
  cy /= pts.length;
  int bestIdx = 0;
  double bestDist = double.infinity;
  for (int i = 0; i < pts.length; i++) {
    final dx = pts[i].x - cx;
    final dy = pts[i].y - cy;
    final d = dx * dx + dy * dy;
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }
  return pts[bestIdx];
}

/// Head orientation as Euler angles in degrees, matching the sign conventions
/// documented by Google ML Kit's `Face`:
///
/// - [x] (pitch): rotation about the horizontal axis. Positive when the face
///   tilts up.
/// - [y] (yaw): rotation about the vertical axis. Positive when the face turns
///   toward the right side of the image.
/// - [z] (roll): rotation about the axis pointing out of the image. Positive
///   for a counter-clockwise tilt within the image plane.
class HeadEulerAngles {
  /// Pitch in degrees (up/down).
  final double x;

  /// Yaw in degrees (left/right).
  final double y;

  /// Roll in degrees (in-plane tilt).
  final double z;

  /// Creates head Euler angles from pitch [x], yaw [y], and roll [z] degrees.
  const HeadEulerAngles(this.x, this.y, this.z);

  /// Serializes to a map for isolate transfer.
  Map<String, dynamic> toMap() => {'x': x, 'y': y, 'z': z};

  /// Creates head Euler angles from a serialized map.
  factory HeadEulerAngles.fromMap(Map<String, dynamic> map) => HeadEulerAngles(
    (map['x'] as num).toDouble(),
    (map['y'] as num).toDouble(),
    (map['z'] as num).toDouble(),
  );

  @override
  String toString() =>
      'HeadEulerAngles(x: ${x.toStringAsFixed(1)}, '
      'y: ${y.toStringAsFixed(1)}, z: ${z.toStringAsFixed(1)})';
}

/// All 52 MediaPipe Blendshape V2 coefficients for a detected face.
///
/// Values are in the range 0.0 to 1.0. Names follow the official MediaPipe /
/// Apple ARKit convention, in which "left" and "right" are relative to the
/// **subject** (see [Blendshape]). Populated only in [FaceDetectionMode.full];
/// access via [Face.blendshapes].
///
/// This is a superset of what Google ML Kit exposes: ML Kit surfaces only smile
/// and eye-open likelihoods, which [Face.smilingProbability] and the eye-open
/// probability getters derive from these coefficients.
class FaceBlendshapes {
  /// Creates a blendshape result from 52 coefficients in tensor order.
  FaceBlendshapes(List<double> scores)
    : assert(scores.length == kBlendshapeCount),
      scores = List<double>.unmodifiable(scores);

  /// The 52 coefficients in `[0, 1]`, indexed in [Blendshape] tensor order.
  final List<double> scores;

  /// The coefficient for blendshape [b].
  double operator [](Blendshape b) => scores[b.index];

  /// Serializes to a map for isolate transfer.
  Map<String, dynamic> toMap() => {'scores': scores};

  /// Creates a blendshape result from a serialized map.
  factory FaceBlendshapes.fromMap(Map<String, dynamic> map) => FaceBlendshapes(
    (map['scores'] as List).map((e) => (e as num).toDouble()).toList(),
  );
}

/// Outputs for a single detected face.
class Face {
  /// Underlying detection (bbox + 6 keypoints).
  final Detection detectionData;

  /// 468-point face mesh (null in fast mode).
  final FaceMesh? mesh;

  /// Raw iris/eye points (152 points = 76 per eye in full mode, else empty).
  final List<Point> irisPoints;

  /// Raw 52 blendshape coefficients (full mode only), or null when not
  /// computed. Exposed through [blendshapes] and the probability getters.
  final List<double>? _blendshapeScores;

  /// Original source image dimensions.
  final Size originalSize;

  /// Bounding box in absolute pixel coordinates.
  final BoundingBox boundingBox;

  late final EyePair? _cachedEyes = _computeEyes();

  late final HeadEulerAngles? _cachedHeadAngles = _computeHeadAngles();

  late final FaceBlendshapes? _cachedBlendshapes = _computeBlendshapes();

  /// Creates a face detection result.
  ///
  /// [blendshapeScores] carries the 52 blendshape coefficients (full mode
  /// only); pass null when they were not computed.
  Face({
    required Detection detection,
    required this.mesh,
    required List<Point> irises,
    required this.originalSize,
    List<double>? blendshapeScores,
  }) : detectionData = detection,
       irisPoints = irises,
       _blendshapeScores = blendshapeScores,
       boundingBox = _computeBoundingBox(detection.boundingBox, originalSize);

  static BoundingBox _computeBoundingBox(RectF r, Size originalSize) {
    final double w = originalSize.width.toDouble();
    final double h = originalSize.height.toDouble();
    return BoundingBox(
      topLeft: Point(r.xmin * w, r.ymin * h),
      topRight: Point(r.xmax * w, r.ymin * h),
      bottomRight: Point(r.xmax * w, r.ymax * h),
      bottomLeft: Point(r.xmin * w, r.ymax * h),
    );
  }

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
    final center = irisCenterFromPoints(irisPoints);
    final contour = [
      for (final p in irisPoints)
        if (!identical(p, center)) p,
    ];
    return Eye.optimized(
      irisCenter: center,
      irisContour: contour,
      mesh: eyeMesh,
    );
  }

  /// Detection confidence in the range 0.0 to 1.0 (higher is more confident).
  ///
  /// In practice only faces scoring at or above the detector threshold are
  /// returned, so this is effectively bounded below by that threshold. It
  /// reflects confidence that a face is present, not landmark or mesh quality.
  double get score => detectionData.score;

  /// Mesh model's face-presence confidence, 0.0 to 1.0 (higher is more
  /// confident). Null in fast mode (no mesh) or when the model omits it.
  /// Convenience proxy for `mesh?.score`.
  double? get meshScore => mesh?.score;

  /// This face's visible width as a fraction of the source image width, always
  /// in the range 0.0 to 1.0.
  ///
  /// Inspired by Google ML Kit's `minFaceSize` (the ratio of the face/head width
  /// to the image width), this is the value to compare against a
  /// minimum-face-size threshold. The width is clipped to the image bounds
  /// before the ratio is taken, so a detection box that extends past the image
  /// edge (which the native pipeline allows, unlike the web pipeline which
  /// clamps boxes) still yields a fraction in `[0, 1]` and, importantly, the
  /// same value on every platform. Returns 0.0 when the source image width is
  /// unknown or the face lies entirely outside the image.
  double get widthFraction {
    final double w = originalSize.width;
    if (w <= 0) return 0.0;
    final double left = boundingBox.topLeft.x;
    final double right = boundingBox.topRight.x;
    final double visible = math.min(right, w) - math.max(left, 0.0);
    return visible > 0 ? visible / w : 0.0;
  }

  /// Comprehensive eye tracking data for both eyes (null when no iris data).
  EyePair? get eyes => _cachedEyes;

  /// Head orientation as Euler angles (pitch/yaw/roll in degrees), or null when
  /// it cannot be estimated. Derived from the 468-point mesh when available; in
  /// fast mode (no mesh) only roll ([HeadEulerAngles.z]) is estimated from the
  /// detection's eye keypoints, with pitch and yaw reported as 0.
  ///
  /// Sign conventions match Google ML Kit; see [HeadEulerAngles].
  HeadEulerAngles? get headEulerAngles => _cachedHeadAngles;

  /// Head pitch in degrees (up/down); positive tilts the face up. Null when
  /// unavailable. See [headEulerAngles].
  double? get headEulerAngleX => _cachedHeadAngles?.x;

  /// Head yaw in degrees (left/right); positive turns the face toward the right
  /// side of the image. Null when unavailable. See [headEulerAngles].
  double? get headEulerAngleY => _cachedHeadAngles?.y;

  /// Head roll in degrees (in-plane tilt); positive is counter-clockwise. Null
  /// when unavailable. See [headEulerAngles].
  double? get headEulerAngleZ => _cachedHeadAngles?.z;

  /// All 52 MediaPipe Blendshape V2 coefficients, or null when not computed
  /// (fast/standard modes, or the blendshape stage could not run). See
  /// [FaceBlendshapes]; only populated in [FaceDetectionMode.full].
  FaceBlendshapes? get blendshapes => _cachedBlendshapes;

  /// Probability that the face is smiling, 0.0 to 1.0, or null when not
  /// computed. The average of the `mouthSmileLeft` and `mouthSmileRight`
  /// blendshapes; semantics match Google ML Kit's `smilingProbability`.
  double? get smilingProbability {
    final FaceBlendshapes? b = _cachedBlendshapes;
    if (b == null) return null;
    final double v =
        (b[Blendshape.mouthSmileLeft] + b[Blendshape.mouthSmileRight]) / 2.0;
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
  }

  /// Probability that the SUBJECT'S left eye is open, 0.0 to 1.0, or null when
  /// not computed. Computed as `1 - eyeBlinkLeft`; semantics match Google ML
  /// Kit's `leftEyeOpenProbability`.
  ///
  /// NOTE: "left" is subject-relative (the ML Kit / ARKit convention), i.e. the
  /// eye that appears on the RIGHT of an unmirrored image. That is the OPPOSITE
  /// eye from [eyes]`?.leftEye`, which is image-relative. If the app
  /// horizontally flips frames before detection (common for a selfie preview),
  /// subject left/right swap, exactly as they do in ML Kit.
  double? get leftEyeOpenProbability {
    final FaceBlendshapes? b = _cachedBlendshapes;
    if (b == null) return null;
    final double v = 1.0 - b[Blendshape.eyeBlinkLeft];
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
  }

  /// Probability that the SUBJECT'S right eye is open, 0.0 to 1.0, or null when
  /// not computed. Computed as `1 - eyeBlinkRight`; semantics match Google ML
  /// Kit's `rightEyeOpenProbability`. See [leftEyeOpenProbability] for the
  /// subject-relative caveat.
  double? get rightEyeOpenProbability {
    final FaceBlendshapes? b = _cachedBlendshapes;
    if (b == null) return null;
    final double v = 1.0 - b[Blendshape.eyeBlinkRight];
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
  }

  FaceBlendshapes? _computeBlendshapes() {
    final List<double>? s = _blendshapeScores;
    if (s == null || s.length != kBlendshapeCount) return null;
    return FaceBlendshapes(s);
  }

  HeadEulerAngles? _computeHeadAngles() {
    final FaceMesh? m = mesh;
    if (m != null) {
      final HeadEulerAngles? angles = headEulerAnglesFromMesh(m.points);
      if (angles != null) return angles;
    }
    // Fast-mode fallback: roll only, from the detection's eye keypoints scaled
    // to pixel coordinates via the original image size (which is always known).
    final List<double> kp = detectionData.keypointsXY;
    final int li = FaceLandmarkType.leftEye.index * 2;
    final int ri = FaceLandmarkType.rightEye.index * 2;
    if (kp.length <= ri + 1) return null;
    final double w = originalSize.width, h = originalSize.height;
    final Point left = Point(kp[li] * w, kp[li + 1] * h);
    final Point right = Point(kp[ri] * w, kp[ri + 1] * h);
    return HeadEulerAngles(0.0, 0.0, rollFromEyes(left, right));
  }

  EyePair? _computeEyes() {
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

  /// Facial landmark positions in pixel coordinates. Iris-aware in full mode.
  FaceLandmarks get landmarks {
    final Map<FaceLandmarkType, Point> landmarkMap = detectionData.landmarks;
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

  /// The mesh points for a named [FaceContourType], in absolute pixel
  /// coordinates and ordered so that connecting consecutive points draws the
  /// outline (see [faceContourMeshIndices]).
  ///
  /// Mirrors Google ML Kit's `face.getContour(type).points`. Returns null when
  /// no mesh is available ([FaceDetectionMode.fast]); otherwise a non-empty
  /// list (a single point for [FaceContourType.leftCheek] /
  /// [FaceContourType.rightCheek]).
  List<Point>? getContour(FaceContourType type) {
    final FaceMesh? m = mesh;
    if (m == null) return null;
    final List<int> indices = faceContourMeshIndices[type]!;
    return [for (final i in indices) m[i]];
  }

  /// All [FaceContourType] contours keyed by type, or null when no mesh is
  /// available ([FaceDetectionMode.fast]). See [getContour].
  Map<FaceContourType, List<Point>>? get contours {
    if (mesh == null) return null;
    return {for (final type in FaceContourType.values) type: getContour(type)!};
  }

  /// Serializes this face to a map for isolate transfer.
  Map<String, dynamic> toMap() => {
    'detection': detectionData.toMap(),
    if (mesh != null) 'mesh': mesh!.toMap(),
    'irisPoints': irisPoints.map((p) => p.toMap()).toList(),
    if (_blendshapeScores != null) 'blendshapeScores': _blendshapeScores,
    'originalSize': {
      'width': originalSize.width,
      'height': originalSize.height,
    },
  };

  /// Creates a face from a map.
  factory Face.fromMap(Map<String, dynamic> map) => Face(
    detection: Detection.fromMap(map['detection']),
    mesh: map['mesh'] != null ? FaceMesh.fromMap(map['mesh']) : null,
    irises: (map['irisPoints'] as List).map((p) => Point.fromMap(p)).toList(),
    blendshapeScores: map['blendshapeScores'] != null
        ? (map['blendshapeScores'] as List)
              .map((e) => (e as num).toDouble())
              .toList()
        : null,
    originalSize: Size(
      map['originalSize']['width'],
      map['originalSize']['height'],
    ),
  );
}

/// Result combining face detection and segmentation from parallel processing.
class DetectionWithSegmentationResult {
  /// Detected faces.
  final List<Face> faces;

  /// Segmentation mask (null if segmentation was disabled or failed).
  final SegmentationMask? segmentationMask;

  /// Detection time in milliseconds.
  final int detectionTimeMs;

  /// Segmentation time in milliseconds.
  final int segmentationTimeMs;

  /// Creates a result combining detection and segmentation.
  const DetectionWithSegmentationResult({
    required this.faces,
    this.segmentationMask,
    required this.detectionTimeMs,
    required this.segmentationTimeMs,
  });

  /// Wall-clock time (the larger of the two when run in parallel).
  int get totalTimeMs => detectionTimeMs > segmentationTimeMs
      ? detectionTimeMs
      : segmentationTimeMs;

  /// Serializes this result to a map.
  Map<String, dynamic> toMap() => {
    'faces': faces.map((f) => f.toMap()).toList(),
    if (segmentationMask != null) 'segmentationMask': segmentationMask!.toMap(),
    'detectionTimeMs': detectionTimeMs,
    'segmentationTimeMs': segmentationTimeMs,
  };

  /// Creates a result from a serialized map.
  factory DetectionWithSegmentationResult.fromMap(Map<String, dynamic> map) {
    return DetectionWithSegmentationResult(
      faces: (map['faces'] as List)
          .map((m) => Face.fromMap(Map<String, dynamic>.from(m as Map)))
          .toList(),
      segmentationMask: map['segmentationMask'] != null
          ? SegmentationMask.fromMap(
              Map<String, dynamic>.from(map['segmentationMask'] as Map),
            )
          : null,
      detectionTimeMs: map['detectionTimeMs'] as int,
      segmentationTimeMs: map['segmentationTimeMs'] as int,
    );
  }

  @override
  String toString() =>
      'DetectionWithSegmentationResult(faces: ${faces.length}, '
      'mask: ${segmentationMask != null ? "${segmentationMask!.width}x${segmentationMask!.height}" : "null"}, '
      'time: ${totalTimeMs}ms)';
}

/// Axis-aligned rectangle with normalized coordinates (0..1).
class RectF {
  /// Min/max X and Y extents.
  final double xmin, ymin, xmax, ymax;

  /// Creates a normalized rectangle.
  const RectF(this.xmin, this.ymin, this.xmax, this.ymax);

  /// Width.
  double get w => xmax - xmin;

  /// Height.
  double get h => ymax - ymin;

  /// Returns a rectangle scaled in X and Y.
  RectF scale(double sx, double sy) =>
      RectF(xmin * sx, ymin * sy, xmax * sx, ymax * sy);

  /// Expands the rectangle by [frac] keeping the same center.
  RectF expand(double frac) {
    final double cx = (xmin + xmax) * 0.5;
    final double cy = (ymin + ymax) * 0.5;
    final double hw = (w * (1.0 + frac)) * 0.5;
    final double hh = (h * (1.0 + frac)) * 0.5;
    return RectF(cx - hw, cy - hh, cx + hw, cy + hh);
  }

  /// Serializes to a map.
  Map<String, dynamic> toMap() => {
    'xmin': xmin,
    'ymin': ymin,
    'xmax': xmax,
    'ymax': ymax,
  };

  /// Creates from a map.
  factory RectF.fromMap(Map<String, dynamic> map) => RectF(
    map['xmin'] as double,
    map['ymin'] as double,
    map['xmax'] as double,
    map['ymax'] as double,
  );
}

/// Raw detection output: bounding box + 6 keypoints.
class Detection {
  /// Normalized bounding box.
  final RectF boundingBox;

  /// Confidence score.
  final double score;

  /// Flattened landmark coords `[x0, y0, x1, y1, ...]` normalized 0-1.
  final List<double> keypointsXY;

  /// Original image dimensions used to denormalize landmarks.
  final Size? imageSize;

  /// Creates a detection.
  Detection({
    required this.boundingBox,
    required this.score,
    required this.keypointsXY,
    this.imageSize,
  });

  /// Index access to [keypointsXY].
  double operator [](int i) => keypointsXY[i];

  /// Returns facial landmarks in pixel coordinates.
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

  /// Serializes to a map.
  Map<String, dynamic> toMap() => {
    'boundingBox': boundingBox.toMap(),
    'score': score,
    'keypointsXY': keypointsXY,
    if (imageSize != null)
      'imageSize': {'width': imageSize!.width, 'height': imageSize!.height},
  };

  /// Creates from a map.
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
  /// NHWC float tensor normalized to `[-1, 1]`.
  final Float32List tensorNHWC;

  /// Padding fractions `[top, bottom, left, right]`.
  final List<double> padding;

  /// Target width and height passed to the model.
  final int width, height;

  /// Creates an image tensor.
  ImageTensor(this.tensorNHWC, this.padding, this.width, this.height);
}

/// Rotation-aware ROI used for iris alignment.
class AlignedRoi {
  /// X coordinate of ROI center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of ROI center in absolute pixel coordinates.
  final double cy;

  /// Square ROI size in absolute pixels.
  final double size;

  /// Rotation applied to align the ROI, in radians.
  final double theta;

  /// Creates an aligned ROI.
  const AlignedRoi(this.cx, this.cy, this.size, this.theta);
}

/// Decoded detection box and keypoints straight from the TFLite model.
class DecodedBox {
  /// Normalized bounding box.
  final RectF boundingBox;

  /// Flattened keypoints `[x0, y0, ...]`.
  final List<double> keypointsXY;

  /// Creates a decoded box.
  DecodedBox(this.boundingBox, this.keypointsXY);
}
