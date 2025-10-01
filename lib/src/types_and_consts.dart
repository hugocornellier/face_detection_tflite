part of face_detection_tflite;

enum FaceIndex { leftEye, rightEye, noseTip, mouth, leftEyeTragion, rightEyeTragion }
enum FaceDetectionModel { frontCamera, backCamera, shortRange, full, fullSparse }

const _modelNameBack = 'face_detection_back.tflite';
const _modelNameFront = 'face_detection_front.tflite';
const _modelNameShort = 'face_detection_short_range.tflite';
const _modelNameFull = 'face_detection_full_range.tflite';
const _modelNameFullSparse = 'face_detection_full_range_sparse.tflite';
const _faceLandmarkModel = 'face_landmark.tflite';
const _irisLandmarkModel = 'iris_landmark.tflite';

const _rawScoreLimit = 80.0;
const _minScore = 0.5;
const _minSuppressionThreshold = 0.3;

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

class AlignedFace {
  final double cx;
  final double cy;
  final double size;
  final double theta;
  final img.Image faceCrop;
  AlignedFace({required this.cx, required this.cy, required this.size, required this.theta, required this.faceCrop});
}

class PipelineResult {
  final List<Detection> detections;
  final List<Offset> mesh;
  final List<Offset> irises;
  final Size originalSize;
  PipelineResult({required this.detections, required this.mesh, required this.irises, required this.originalSize});
}

class RectF {
  final double xmin, ymin, xmax, ymax;
  const RectF(this.xmin, this.ymin, this.xmax, this.ymax);
  double get w => xmax - xmin;
  double get h => ymax - ymin;
  RectF scale(double sx, double sy) => RectF(xmin * sx, ymin * sy, xmax * sx, ymax * sy);
  RectF expand(double frac) {
    final cx = (xmin + xmax) * 0.5;
    final cy = (ymin + ymax) * 0.5;
    final hw = (w * (1.0 + frac)) * 0.5;
    final hh = (h * (1.0 + frac)) * 0.5;
    return RectF(cx - hw, cy - hh, cx + hw, cy + hh);
  }
}

class Detection {
  final RectF bbox;
  final double score;
  final List<double> keypointsXY;
  final Size? imageSize;

  Detection({
    required this.bbox,
    required this.score,
    required this.keypointsXY,
    this.imageSize,
  });

  double operator [](int i) => keypointsXY[i];

  Map<FaceIndex, Offset> get landmarks {
    final map = <FaceIndex, Offset>{};
    if (imageSize == null) {
      for (final idx in FaceIndex.values) {
        final x = keypointsXY[idx.index * 2];
        final y = keypointsXY[idx.index * 2 + 1];
        map[idx] = Offset(x, y);
      }
      return map;
    }
    final w = imageSize!.width;
    final h = imageSize!.height;
    for (final idx in FaceIndex.values) {
      final x = keypointsXY[idx.index * 2] * w;
      final y = keypointsXY[idx.index * 2 + 1] * h;
      map[idx] = Offset(x, y);
    }
    return map;
  }
}

class ImageTensor {
  final Float32List tensorNHWC;
  final List<double> padding;
  final int width, height;
  ImageTensor(this.tensorNHWC, this.padding, this.width, this.height);
}

class AlignedRoi {
  final double cx;
  final double cy;
  final double size;
  final double theta;
  const AlignedRoi(this.cx, this.cy, this.size, this.theta);
}

class _DecodedBox {
  final RectF bbox;
  final List<double> keypointsXY;
  _DecodedBox(this.bbox, this.keypointsXY);
}
