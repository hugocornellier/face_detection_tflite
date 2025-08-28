part of 'face_core.dart';

class FaceResult {
  final RectF bbox;
  final double score;
  final List<double> keypointsXY;
  final List<List<double>>? faceLandmarks;
  final EyeLandmarks? iris;

  FaceResult({
    required this.bbox,
    required this.score,
    required this.keypointsXY,
    this.faceLandmarks,
    this.iris,
  });
}

class EyeLandmarks {
  final List<List<double>> leftEye;
  final List<List<double>> rightEye;

  EyeLandmarks({required this.leftEye, required this.rightEye});
}

class FaceDetector {
  final FaceDetection _detector;
  final FaceLandmark _faceLm;
  final IrisLandmark _irisLm;

  FaceDetector._(this._detector, this._faceLm, this._irisLm);

  static Future<FaceDetector> create({
    FaceDetectionModel model = FaceDetectionModel.backCamera,
  }) async {
    final det = await FaceDetection.create(model);
    final faceLm = await FaceLandmark.create();
    final irisLm = await IrisLandmark.create();
    return FaceDetector._(det, faceLm, irisLm);
  }

  Future<List<FaceResult>> detect(
      Uint8List imageBytes, {
        bool withFaceLandmarks = false,
        bool withIris = false,
      }) async {
    final img.Image src = img.decodeImage(imageBytes)!;
    final detections = await _detector(imageBytes);

    final results = <FaceResult>[];
    for (final d in detections) {
      List<List<double>>? faceLm;
      EyeLandmarks? iris;

      if (withFaceLandmarks || withIris) {
        final roi = faceDetectionToRoi(d.bbox, expandFraction: 0.5);
        final faceCrop = cropFromRoi(src, roi);

        if (withFaceLandmarks) {
          faceLm = await _faceLm(faceCrop);
          final imgW = src.width.toDouble();
          final imgH = src.height.toDouble();
          final dx = roi.xmin * imgW;
          final dy = roi.ymin * imgH;
          final sx = roi.w * imgW;
          final sy = roi.h * imgH;
          for (final p in faceLm) {
            p[0] = dx + p[0] * sx;
            p[1] = dy + p[1] * sy;
          }
        }

        if (withIris) {
          final lx = d.keypointsXY[FaceIndex.leftEye.index * 2];
          final ly = d.keypointsXY[FaceIndex.leftEye.index * 2 + 1];
          final rx = d.keypointsXY[FaceIndex.rightEye.index * 2];
          final ry = d.keypointsXY[FaceIndex.rightEye.index * 2 + 1];

          RectF eyeRoi(double cx, double cy, {double box = 0.08}) {
            final xmin = (cx - box).clamp(0.0, 1.0);
            final ymin = (cy - box).clamp(0.0, 1.0);
            final xmax = (cx + box).clamp(0.0, 1.0);
            final ymax = (cy + box).clamp(0.0, 1.0);
            return RectF(xmin, ymin, xmax, ymax);
          }

          final leftRoi = eyeRoi(lx, ly);
          final rightRoi = eyeRoi(rx, ry);

          final leftPts = await _irisLm.runOnImage(src, leftRoi);
          final rightPts = await _irisLm.runOnImage(src, rightRoi);

          iris = EyeLandmarks(leftEye: leftPts, rightEye: rightPts);
        }
      }

      results.add(FaceResult(
        bbox: d.bbox,
        score: d.score,
        keypointsXY: d.keypointsXY,
        faceLandmarks: faceLm,
        iris: iris,
      ));
    }

    return results;
  }
}