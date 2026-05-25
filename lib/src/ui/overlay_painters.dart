part of '../native/face_native_lib.dart';

/// Semantic labels (indexed 0-5) for classes emitted by the multiclass
/// segmentation model: background, hair, body skin, face skin, clothes, other.
const List<String> kSegmentationClassLabels = [
  'BG',
  'Hair',
  'Body',
  'Face',
  'Clothes',
  'Other',
];

/// Default per-class colors for the multiclass segmentation overlay, aligned
/// by index with [kSegmentationClassLabels] (0=BG, 1=Hair, 2=Body, 3=Face,
/// 4=Clothes, 5=Other). Alpha is preserved so overlays composite onto the
/// underlying camera/image.
const List<Color> kSegmentationClassColors = [
  Color(0x99A0A0A0),
  Color(0x99CD853F),
  Color(0x88FFA500),
  Color(0x88FF69B4),
  Color(0x9900BFFF),
  Color(0x9940E0D0),
];

/// Classify detection-time in milliseconds into a display-friendly bucket
/// (`label`, `color`, `icon`) for overlay status indicators.
({String label, Color color, IconData icon}) performanceLevel(int ms) {
  if (ms < 200) {
    return (label: 'Excellent', color: Colors.green, icon: Icons.speed);
  } else if (ms < 500) {
    return (label: 'Good', color: Colors.lightGreen, icon: Icons.thumb_up);
  } else if (ms < 1000) {
    return (label: 'Fair', color: Colors.orange, icon: Icons.warning_amber);
  } else {
    return (label: 'Slow', color: Colors.red, icon: Icons.hourglass_bottom);
  }
}

/// Compute the valid (non-padding) region of a segmentation mask.
({int x0, int y0, int x1, int y1}) maskValidRegion(SegmentationMask mask) {
  final pt = mask.padding[0];
  final pb = mask.padding[1];
  final pl = mask.padding[2];
  final pr = mask.padding[3];
  return (
    x0: (pl * mask.width).round(),
    y0: (pt * mask.height).round(),
    x1: ((1.0 - pr) * mask.width).round(),
    y1: ((1.0 - pb) * mask.height).round(),
  );
}

/// Draw multiclass segmentation labels at class centroids (one label per
/// class index whose pixel count exceeds an internal threshold).
void drawSegmentationClassLabels(
  Canvas canvas,
  List<int> counts,
  List<double> sumX,
  List<double> sumY,
) {
  for (int c = 0; c < 6; c++) {
    if (counts[c] > 100) {
      final centroidX = sumX[c] / counts[c];
      final centroidY = sumY[c] / counts[c];
      final textPainter = TextPainter(
        text: TextSpan(
          text: kSegmentationClassLabels[c],
          style: const TextStyle(
            color: Colors.white,
            fontSize: 10,
            fontWeight: FontWeight.bold,
            shadows: [
              Shadow(color: Colors.black, blurRadius: 2),
              Shadow(color: Colors.black, blurRadius: 4),
            ],
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(
          centroidX - textPainter.width / 2,
          centroidY - textPainter.height / 2,
        ),
      );
    }
  }
}

/// Compute the axis-aligned bounding rect of a set of offsets.
Rect boundsOf(Iterable<Offset> pts) {
  final it = pts.iterator..moveNext();
  double minX = it.current.dx, maxX = it.current.dx;
  double minY = it.current.dy, maxY = it.current.dy;
  while (it.moveNext()) {
    final p = it.current;
    if (p.dx < minX) minX = p.dx;
    if (p.dx > maxX) maxX = p.dx;
    if (p.dy < minY) minY = p.dy;
    if (p.dy > maxY) maxY = p.dy;
  }
  return Rect.fromLTRB(minX, minY, maxX, maxY);
}

class DetectionsPainter extends CustomPainter {
  final List<Face> faces;
  final Rect imageRectOnCanvas;
  final Size originalImageSize;
  final bool showBoundingBoxes;
  final bool showMesh;
  final bool showLandmarks;
  final bool showLandmarkLabels;
  final bool showIrises;
  final bool showEyeContours;
  final bool showEyeMesh;
  final Color boundingBoxColor;
  final Color landmarkColor;
  final Color meshColor;
  final Color irisColor;
  final Color eyeContourColor;
  final Color eyeMeshColor;
  final double boundingBoxThickness;
  final double landmarkSize;
  final double meshSize;
  final double eyeMeshSize;

  DetectionsPainter({
    required this.faces,
    required this.imageRectOnCanvas,
    required this.originalImageSize,
    required this.showBoundingBoxes,
    required this.showMesh,
    required this.showLandmarks,
    required this.showLandmarkLabels,
    required this.showIrises,
    required this.showEyeContours,
    required this.showEyeMesh,
    required this.boundingBoxColor,
    required this.landmarkColor,
    required this.meshColor,
    required this.irisColor,
    required this.eyeContourColor,
    required this.eyeMeshColor,
    required this.boundingBoxThickness,
    required this.landmarkSize,
    required this.meshSize,
    required this.eyeMeshSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (faces.isEmpty) return;

    final ui.Paint boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = boundingBoxThickness
      ..color = boundingBoxColor;

    final ui.Paint detKpPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = landmarkColor;

    final ui.Paint meshPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = meshColor;

    final ui.Paint irisFill = Paint()
      ..style = PaintingStyle.fill
      ..color = irisColor.withAlpha(153)
      ..blendMode = BlendMode.srcOver;

    final ui.Paint irisStroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = irisColor.withAlpha(230);

    final double ox = imageRectOnCanvas.left;
    final double oy = imageRectOnCanvas.top;
    final double scaleX = imageRectOnCanvas.width / originalImageSize.width;
    final double scaleY = imageRectOnCanvas.height / originalImageSize.height;

    for (final Face face in faces) {
      if (showBoundingBoxes) {
        final BoundingBox boundingBox = face.boundingBox;
        final ui.Rect rect = Rect.fromLTRB(
          ox + boundingBox.topLeft.x * scaleX,
          oy + boundingBox.topLeft.y * scaleY,
          ox + boundingBox.bottomRight.x * scaleX,
          oy + boundingBox.bottomRight.y * scaleY,
        );
        canvas.drawRect(rect, boxPaint);
      }

      if (showLandmarks) {
        const labelNames = {
          FaceLandmarkType.leftEye: 'Left Eye',
          FaceLandmarkType.rightEye: 'Right Eye',
          FaceLandmarkType.noseTip: 'Nose Tip',
          FaceLandmarkType.mouth: 'Mouth',
          FaceLandmarkType.leftEyeTragion: 'L Tragion',
          FaceLandmarkType.rightEyeTragion: 'R Tragion',
        };

        for (final entry in face.landmarks.toMap().entries) {
          final p = entry.value;
          final center = Offset(ox + p.x * scaleX, oy + p.y * scaleY);
          canvas.drawCircle(center, landmarkSize, detKpPaint);

          if (showLandmarkLabels) {
            final label = labelNames[entry.key] ?? entry.key.name;
            final textPainter = TextPainter(
              text: TextSpan(
                text: label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                ),
              ),
              textDirection: TextDirection.ltr,
            )..layout();

            const double paddingH = 6;
            const double paddingV = 3;
            const double arrowHeight = 5;
            const double arrowHalfWidth = 4;
            const double gap = 2;

            final double boxWidth = textPainter.width + paddingH * 2;
            final double boxHeight = textPainter.height + paddingV * 2;
            final double tipY = center.dy - landmarkSize - gap;
            final double boxBottom = tipY - arrowHeight;
            final double boxTop = boxBottom - boxHeight;
            final double boxLeft = center.dx - boxWidth / 2;
            final double boxRight = center.dx + boxWidth / 2;

            final bgPaint = Paint()..color = landmarkColor;
            final rrect = RRect.fromLTRBR(
              boxLeft,
              boxTop,
              boxRight,
              boxBottom,
              const Radius.circular(4),
            );
            canvas.drawRRect(rrect, bgPaint);

            final arrowPath = Path()
              ..moveTo(center.dx - arrowHalfWidth, boxBottom)
              ..lineTo(center.dx, tipY)
              ..lineTo(center.dx + arrowHalfWidth, boxBottom)
              ..close();
            canvas.drawPath(arrowPath, bgPaint);

            textPainter.paint(
              canvas,
              Offset(boxLeft + paddingH, boxTop + paddingV),
            );
          }
        }
      }

      if (showMesh) {
        final FaceMesh? faceMesh = face.mesh;
        if (faceMesh != null) {
          final mesh = faceMesh.points;
          final double imgArea =
              imageRectOnCanvas.width * imageRectOnCanvas.height;
          final double radius = meshSize + math.sqrt(imgArea) / 1000.0;

          for (final p in mesh) {
            canvas.drawCircle(
              Offset(ox + p.x * scaleX, oy + p.y * scaleY),
              radius,
              meshPaint,
            );
          }
        }
      }

      if (showIrises || showEyeContours || showEyeMesh) {
        final eyePair = face.eyes;
        if (eyePair != null) {
          for (final iris in [eyePair.leftEye, eyePair.rightEye]) {
            if (iris == null) continue;

            if (showIrises) {
              final bounds = boundsOf(
                [
                  iris.irisCenter,
                  ...iris.irisContour,
                ].map((p) => Offset(ox + p.x * scaleX, oy + p.y * scaleY)),
              );
              canvas.drawOval(bounds, irisFill);
              canvas.drawOval(bounds, irisStroke);
            }

            if (showEyeContours && iris.mesh.isNotEmpty) {
              final Paint eyeOutlinePaint = Paint()
                ..color = eyeContourColor
                ..style = PaintingStyle.stroke
                ..strokeWidth = 1.5;

              final eyelidContour = iris.contour;
              for (final connection in eyeLandmarkConnections) {
                if (connection[0] < eyelidContour.length &&
                    connection[1] < eyelidContour.length) {
                  final p1 = eyelidContour[connection[0]];
                  final p2 = eyelidContour[connection[1]];

                  canvas.drawLine(
                    Offset(ox + p1.x * scaleX, oy + p1.y * scaleY),
                    Offset(ox + p2.x * scaleX, oy + p2.y * scaleY),
                    eyeOutlinePaint,
                  );
                }
              }
            }

            if (showEyeMesh && iris.mesh.isNotEmpty) {
              final Paint eyeMeshPointPaint = Paint()
                ..color = eyeMeshColor
                ..style = PaintingStyle.fill;

              for (final p in iris.mesh) {
                final canvasX = ox + p.x * scaleX;
                final canvasY = oy + p.y * scaleY;
                canvas.drawCircle(
                  Offset(canvasX, canvasY),
                  eyeMeshSize,
                  eyeMeshPointPaint,
                );
              }
            }
          }
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant DetectionsPainter old) {
    return old.faces != faces ||
        old.imageRectOnCanvas != imageRectOnCanvas ||
        old.originalImageSize != originalImageSize ||
        old.showBoundingBoxes != showBoundingBoxes ||
        old.showMesh != showMesh ||
        old.showLandmarks != showLandmarks ||
        old.showLandmarkLabels != showLandmarkLabels ||
        old.showIrises != showIrises ||
        old.showEyeContours != showEyeContours ||
        old.showEyeMesh != showEyeMesh ||
        old.boundingBoxColor != boundingBoxColor ||
        old.landmarkColor != landmarkColor ||
        old.meshColor != meshColor ||
        old.irisColor != irisColor ||
        old.eyeContourColor != eyeContourColor ||
        old.eyeMeshColor != eyeMeshColor ||
        old.boundingBoxThickness != boundingBoxThickness ||
        old.landmarkSize != landmarkSize ||
        old.meshSize != meshSize ||
        old.eyeMeshSize != eyeMeshSize;
  }
}

class CameraDetectionPainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final double cameraAspectRatio;
  final double displayAspectRatio;
  final FaceDetectionMode detectionMode;
  final int sensorOrientation;
  final Orientation deviceOrientation;
  final bool isFrontCamera;
  final bool mirrorHorizontally;

  CameraDetectionPainter({
    required this.faces,
    required this.imageSize,
    required this.cameraAspectRatio,
    required this.displayAspectRatio,
    required this.detectionMode,
    required this.sensorOrientation,
    required this.deviceOrientation,
    required this.isFrontCamera,
    required this.mirrorHorizontally,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (faces.isEmpty) return;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = const Color(0xFF00FFCC);

    final landmarkPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFF89CFF0);

    final meshPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFFF4C2C2);

    final irisFill = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFF22AAFF).withAlpha(153)
      ..blendMode = BlendMode.srcOver;

    final irisStroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = const Color(0xFF22AAFF).withAlpha(230);

    final double displayWidth = size.width;
    final double displayHeight = size.height;

    final double sourceWidth = imageSize.width;
    final double sourceHeight = imageSize.height;

    final double sourceAspectRatio = sourceWidth / sourceHeight;
    final double viewportAspectRatio = displayWidth / displayHeight;

    final double scale;
    double offsetX = 0;
    double offsetY = 0;

    if (sourceAspectRatio > viewportAspectRatio) {
      scale = displayHeight / sourceHeight;
      offsetX = (displayWidth - sourceWidth * scale) / 2;
    } else {
      scale = displayWidth / sourceWidth;
      offsetY = (displayHeight - sourceHeight * scale) / 2;
    }

    Offset transformPoint(double x, double y) {
      if (mirrorHorizontally) {
        x = sourceWidth - x;
      }
      return Offset(x * scale + offsetX, y * scale + offsetY);
    }

    for (final face in faces) {
      final boundingBox = face.boundingBox;
      final p1 = transformPoint(boundingBox.topLeft.x, boundingBox.topLeft.y);
      final p2 = transformPoint(
        boundingBox.bottomRight.x,
        boundingBox.bottomRight.y,
      );

      final rect = Rect.fromLTRB(
        math.min(p1.dx, p2.dx),
        math.min(p1.dy, p2.dy),
        math.max(p1.dx, p2.dx),
        math.max(p1.dy, p2.dy),
      );
      canvas.drawRect(rect, boxPaint);

      for (final landmark in face.landmarks.values) {
        final transformed = transformPoint(landmark.x, landmark.y);
        canvas.drawCircle(transformed, 4.0, landmarkPaint);
      }

      if (detectionMode == FaceDetectionMode.standard ||
          detectionMode == FaceDetectionMode.full) {
        final FaceMesh? faceMesh = face.mesh;
        if (faceMesh != null) {
          final mesh = faceMesh.points;
          final double imgArea = displayWidth * displayHeight;
          final double radius = 1.25 + math.sqrt(imgArea) / 1000.0;

          for (final p in mesh) {
            final transformed = transformPoint(p.x, p.y);
            canvas.drawCircle(transformed, radius, meshPaint);
          }
        }
      }

      if (detectionMode == FaceDetectionMode.full) {
        final eyePair = face.eyes;
        if (eyePair != null) {
          for (final iris in [eyePair.leftEye, eyePair.rightEye]) {
            if (iris == null) continue;

            final oval = boundsOf(
              [
                iris.irisCenter,
                ...iris.irisContour,
              ].map((p) => transformPoint(p.x, p.y)),
            );
            canvas.drawOval(oval, irisFill);
            canvas.drawOval(oval, irisStroke);

            if (iris.mesh.isNotEmpty) {
              final Paint eyeOutlinePaint = Paint()
                ..color = const Color(0xFF22AAFF)
                ..style = PaintingStyle.stroke
                ..strokeWidth = 1.5;

              final eyelidContour = iris.contour;
              for (final connection in eyeLandmarkConnections) {
                if (connection[0] < eyelidContour.length &&
                    connection[1] < eyelidContour.length) {
                  final p1 = eyelidContour[connection[0]];
                  final p2 = eyelidContour[connection[1]];
                  final t1 = transformPoint(p1.x, p1.y);
                  final t2 = transformPoint(p2.x, p2.y);
                  canvas.drawLine(t1, t2, eyeOutlinePaint);
                }
              }

              final Paint eyeMeshPointPaint = Paint()
                ..color = const Color(0xFFFFAA22)
                ..style = PaintingStyle.fill;

              for (final p in iris.mesh) {
                final transformed = transformPoint(p.x, p.y);
                canvas.drawCircle(transformed, 0.8, eyeMeshPointPaint);
              }
            }
          }
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant CameraDetectionPainter old) {
    return old.faces != faces ||
        old.imageSize != imageSize ||
        old.cameraAspectRatio != cameraAspectRatio ||
        old.displayAspectRatio != displayAspectRatio ||
        old.detectionMode != detectionMode ||
        old.sensorOrientation != sensorOrientation ||
        old.deviceOrientation != deviceOrientation ||
        old.isFrontCamera != isFrontCamera ||
        old.mirrorHorizontally != mirrorHorizontally;
  }
}

/// Painter for rendering segmentation mask overlay on live camera feed.
///
/// When [showAllClasses] is true, [classColors] (indexed by class) must be
/// supplied to colourize per-class pixels.
class LiveSegmentationPainter extends CustomPainter {
  /// Mask to render.
  final SegmentationMask mask;

  /// Color used for single-class rendering.
  final Color maskColor;

  /// When true, render all classes (each in its own [classColors] entry).
  final bool showAllClasses;

  /// Whether to flip x-coordinates (match mirrored front-camera preview).
  final bool mirrorHorizontally;

  /// Per-class colors used only when [showAllClasses] is true.
  final List<Color> classColors;

  /// Creates a painter for the given mask + styling.
  LiveSegmentationPainter({
    required this.mask,
    required this.maskColor,
    this.showAllClasses = false,
    this.mirrorHorizontally = false,
    this.classColors = const <Color>[],
  });

  @override
  void paint(Canvas canvas, Size size) {
    final v = maskValidRegion(mask);
    final validW = v.x1 - v.x0;
    final validH = v.y1 - v.y0;

    final fit = coverFitScaleOffset(validW, validH, size.width, size.height);
    final scale = fit.scale;
    final offsetX = fit.offsetX;
    final offsetY = fit.offsetY;

    final double pixelW = scale + 0.5;
    final double pixelH = scale + 0.5;

    final paint = Paint();
    const double threshold = 0.5;

    if (showAllClasses && mask is MulticlassSegmentationMask) {
      final multiMask = mask as MulticlassSegmentationMask;
      final classMasks = List.generate(6, (i) => multiMask.classMask(i));

      final labelCounts = List<int>.filled(6, 0);
      final labelSumX = List<double>.filled(6, 0);
      final labelSumY = List<double>.filled(6, 0);

      for (int y = v.y0; y < v.y1; y++) {
        for (int x = v.x0; x < v.x1; x++) {
          final idx = y * mask.width + x;
          final rawX = (x - v.x0) * scale + offsetX;
          final renderX = mirrorHorizontally
              ? size.width - rawX - pixelW
              : rawX;
          final renderY = (y - v.y0) * scale + offsetY;

          int winningClass = 0;
          double maxProb = classMasks[0][idx];
          for (int c = 1; c < 6; c++) {
            if (classMasks[c][idx] > maxProb) {
              maxProb = classMasks[c][idx];
              winningClass = c;
            }
          }

          if (maxProb >= threshold) {
            final color = classColors[winningClass];
            final baseAlpha = (color.a * 255).round();
            paint.color = color.withAlpha((maxProb * baseAlpha).round());
            canvas.drawRect(
              Rect.fromLTWH(renderX, renderY, pixelW, pixelH),
              paint,
            );

            labelCounts[winningClass]++;
            labelSumX[winningClass] += renderX;
            labelSumY[winningClass] += renderY;
          }
        }
      }

      drawSegmentationClassLabels(canvas, labelCounts, labelSumX, labelSumY);
      return;
    }

    for (int y = v.y0; y < v.y1; y++) {
      for (int x = v.x0; x < v.x1; x++) {
        final prob = mask.at(x, y);
        final alpha = prob >= threshold ? maskColor.a : 0.0;

        if (alpha > 0.01) {
          paint.color = maskColor.withAlpha((alpha * 255).round());
          final rawX = (x - v.x0) * scale + offsetX;
          final renderX = mirrorHorizontally
              ? size.width - rawX - pixelW
              : rawX;
          final renderY = (y - v.y0) * scale + offsetY;
          canvas.drawRect(
            Rect.fromLTWH(renderX, renderY, pixelW, pixelH),
            paint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant LiveSegmentationPainter old) {
    return old.mask != mask ||
        old.maskColor != maskColor ||
        old.showAllClasses != showAllClasses ||
        old.mirrorHorizontally != mirrorHorizontally;
  }
}

/// Painter that draws a background image scaled to fill the canvas.
class BackgroundImagePainter extends CustomPainter {
  final ui.Image image;

  BackgroundImagePainter({required this.image});

  @override
  void paint(Canvas canvas, Size size) {
    final src = Rect.fromLTWH(
      0,
      0,
      image.width.toDouble(),
      image.height.toDouble(),
    );
    final dst = Rect.fromLTWH(0, 0, size.width, size.height);
    canvas.drawImageRect(image, src, dst, Paint());
  }

  @override
  bool shouldRepaint(covariant BackgroundImagePainter old) {
    return old.image != image;
  }
}

/// Painter that draws background image only on non-person (background) areas.
/// This creates the "virtual background" effect by covering the camera's
/// background with the beach image while leaving the person visible.
/// Uses soft alpha blending at edges for smooth transitions.
class VirtualBackgroundOverlayPainter extends CustomPainter {
  final ui.Image background;
  final SegmentationMask mask;
  final bool mirrorHorizontally;

  VirtualBackgroundOverlayPainter({
    required this.background,
    required this.mask,
    this.mirrorHorizontally = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final v = maskValidRegion(mask);
    final validW = v.x1 - v.x0;
    final validH = v.y1 - v.y0;

    if (validW <= 0 || validH <= 0) return;

    final fit = coverFitScaleOffset(validW, validH, size.width, size.height);
    final scale = fit.scale;
    final offsetX = fit.offsetX;
    final offsetY = fit.offsetY;

    final double pixelW = scale + 0.5;
    final double pixelH = scale + 0.5;

    final bgScaleX = background.width / size.width;
    final bgScaleY = background.height / size.height;

    final paint = Paint();

    for (int y = v.y0; y < v.y1; y++) {
      for (int x = v.x0; x < v.x1; x++) {
        final prob = mask.at(x, y).clamp(0.0, 1.0);

        final bgAlpha = (1.0 - prob);

        if (bgAlpha < 0.01) continue;

        final rawX = (x - v.x0) * scale + offsetX;
        final renderX = mirrorHorizontally ? size.width - rawX - pixelW : rawX;
        final renderY = (y - v.y0) * scale + offsetY;

        final bgX = (renderX * bgScaleX)
            .clamp(0, background.width - 1)
            .toDouble();
        final bgY = (renderY * bgScaleY)
            .clamp(0, background.height - 1)
            .toDouble();

        paint.color = Color.fromRGBO(255, 255, 255, bgAlpha);
        final src = Rect.fromLTWH(
          bgX,
          bgY,
          bgScaleX * pixelW,
          bgScaleY * pixelH,
        );
        final dst = Rect.fromLTWH(renderX, renderY, pixelW, pixelH);
        canvas.drawImageRect(background, src, dst, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant VirtualBackgroundOverlayPainter old) {
    return old.background != background ||
        old.mask != mask ||
        old.mirrorHorizontally != mirrorHorizontally;
  }
}

/// Painter for rendering a segmentation mask over a still image.
///
/// When [showAllClasses] is true, [classColors] (indexed by class) must be
/// supplied to colourize per-class pixels.
class SegmentationMaskPainter extends CustomPainter {
  /// Mask to render.
  final SegmentationMask mask;

  /// Original image size the mask was produced from.
  final Size originalSize;

  /// Probability threshold for a pixel to be drawn.
  final double threshold;

  /// When true, render only at the threshold (hard edges); else soft alpha.
  final bool binary;

  /// Color used for single-class rendering.
  final Color maskColor;

  /// Optional single-class index to isolate (rest is hidden).
  final int? classIndex;

  /// When true, render all classes (each in its own [classColors] entry).
  final bool showAllClasses;

  /// Per-class colors used only when [showAllClasses] is true.
  final List<Color> classColors;

  /// Creates a painter for the given mask + styling.
  SegmentationMaskPainter({
    required this.mask,
    required this.originalSize,
    required this.threshold,
    required this.binary,
    required this.maskColor,
    this.classIndex,
    this.showAllClasses = false,
    this.classColors = const <Color>[],
  });

  @override
  void paint(Canvas canvas, Size size) {
    final v = maskValidRegion(mask);
    final validW = v.x1 - v.x0;
    final validH = v.y1 - v.y0;

    final scaleX = validW > 0 ? size.width / validW : 1.0;
    final scaleY = validH > 0 ? size.height / validH : 1.0;

    if (showAllClasses && mask is MulticlassSegmentationMask) {
      final multiMask = mask as MulticlassSegmentationMask;
      final classMasks = List.generate(6, (i) => multiMask.classMask(i));
      final paint = Paint();

      final labelCounts = List<int>.filled(6, 0);
      final labelSumX = List<double>.filled(6, 0);
      final labelSumY = List<double>.filled(6, 0);

      for (int y = v.y0; y < v.y1; y++) {
        for (int x = v.x0; x < v.x1; x++) {
          final idx = y * mask.width + x;
          final renderX = (x - v.x0) * scaleX;
          final renderY = (y - v.y0) * scaleY;

          int winningClass = 0;
          double maxProb = classMasks[0][idx];
          for (int c = 1; c < 6; c++) {
            if (classMasks[c][idx] > maxProb) {
              maxProb = classMasks[c][idx];
              winningClass = c;
            }
          }

          if (maxProb >= threshold) {
            final color = classColors[winningClass];
            final baseAlpha = (color.a * 255).round();
            paint.color = binary
                ? color
                : color.withAlpha((maxProb * baseAlpha).round());
            canvas.drawRect(
              Rect.fromLTWH(renderX, renderY, scaleX + 0.5, scaleY + 0.5),
              paint,
            );

            labelCounts[winningClass]++;
            labelSumX[winningClass] += renderX;
            labelSumY[winningClass] += renderY;
          }
        }
      }

      drawSegmentationClassLabels(canvas, labelCounts, labelSumX, labelSumY);
      return;
    }

    Float32List? classMaskData;
    if (classIndex != null && mask is MulticlassSegmentationMask) {
      classMaskData = (mask as MulticlassSegmentationMask).classMask(
        classIndex!,
      );
    }

    final paint = Paint();

    for (int y = v.y0; y < v.y1; y++) {
      for (int x = v.x0; x < v.x1; x++) {
        final double prob;
        if (classMaskData != null) {
          final idx = y * mask.width + x;
          prob = classMaskData[idx];
        } else {
          prob = mask.at(x, y);
        }

        final double alpha;
        if (binary) {
          alpha = prob >= threshold ? maskColor.a : 0.0;
        } else {
          alpha = prob * maskColor.a;
        }

        if (alpha > 0.01) {
          paint.color = maskColor.withAlpha((alpha * 255).round());
          final renderX = (x - v.x0) * scaleX;
          final renderY = (y - v.y0) * scaleY;
          canvas.drawRect(
            Rect.fromLTWH(renderX, renderY, scaleX + 0.5, scaleY + 0.5),
            paint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant SegmentationMaskPainter old) {
    return old.mask != mask ||
        old.threshold != threshold ||
        old.binary != binary ||
        old.maskColor != maskColor ||
        old.classIndex != classIndex ||
        old.showAllClasses != showAllClasses;
  }
}

/// A composite widget that overlays face-detection and (optional) segmentation
/// results on top of a live camera preview.
///
/// Assembles up to four layers inside an [AspectRatio] sized [Stack]:
/// 1. A full-screen virtual background ([BackgroundImagePainter]) when
///    [virtualBackground] is provided and [showVirtualBackground] is true.
/// 2. The caller-provided [cameraPreview].
/// 3. A virtual-background composite ([VirtualBackgroundOverlayPainter]) that
///    cuts the subject out of the camera and places them on
///    [virtualBackground].
/// 4. Or, when [showSegmentation] is enabled without a virtual background, a
///    [LiveSegmentationPainter] tint.
/// 5. A [CameraDetectionPainter] drawing bounding boxes, mesh, landmarks etc.
///    for the provided [faces] (only when [imageSize] is non-null).
class FaceDetectionCameraOverlay extends StatelessWidget {
  /// The camera preview widget (typically a [CameraPreview]).
  final Widget cameraPreview;

  /// Aspect ratio of the raw camera frames (width / height). Used by the
  /// detection painter to map face coordinates.
  final double cameraAspectRatio;

  /// Aspect ratio used for the display [AspectRatio] box. Often the inverse
  /// of [cameraAspectRatio] when the device is in portrait.
  final double displayAspectRatio;

  /// Whether the detection and segmentation overlays should be mirrored
  /// horizontally (usually true for front cameras on Android).
  final bool mirrorHorizontally;

  /// Sensor mount orientation in degrees (0/90/180/270).
  final int sensorOrientation;

  /// Current device orientation (portrait/landscape).
  final Orientation deviceOrientation;

  /// Whether the active camera is the front-facing camera.
  final bool isFrontCamera;

  /// Detection mode that produced the [faces] (controls which overlays the
  /// painter considers valid).
  final FaceDetectionMode detectionMode;

  /// Detected faces to draw via [CameraDetectionPainter].
  final List<Face> faces;

  /// Size of the image used for detection (post-rotation). The detection
  /// painter is skipped when null.
  final Size? imageSize;

  /// Segmentation mask output from the detector, or null when segmentation
  /// is not active.
  final SegmentationMask? segmentationMask;

  /// Optional virtual-background image painted behind the subject.
  final ui.Image? virtualBackground;

  /// Whether to render the standalone segmentation tint overlay. Ignored when
  /// [showVirtualBackground] is true.
  final bool showSegmentation;

  /// Whether to replace the background with [virtualBackground], cutting the
  /// subject out using [segmentationMask].
  final bool showVirtualBackground;

  /// Tint color used by the standalone segmentation overlay. Ignored for
  /// multiclass rendering (see [segmentationShowAllClasses]).
  final Color segmentationColor;

  /// When true, render every class in [segmentationClassColors] instead of a
  /// single-color tint. Use for multiclass segmentation models.
  final bool segmentationShowAllClasses;

  /// Per-class colors for multiclass segmentation. Defaults to
  /// [kSegmentationClassColors].
  final List<Color> segmentationClassColors;

  const FaceDetectionCameraOverlay({
    super.key,
    required this.cameraPreview,
    required this.cameraAspectRatio,
    required this.displayAspectRatio,
    required this.mirrorHorizontally,
    required this.sensorOrientation,
    required this.deviceOrientation,
    required this.isFrontCamera,
    required this.detectionMode,
    required this.faces,
    this.imageSize,
    this.segmentationMask,
    this.virtualBackground,
    this.showSegmentation = false,
    this.showVirtualBackground = false,
    this.segmentationColor = const Color(0x8800FF00),
    this.segmentationShowAllClasses = false,
    this.segmentationClassColors = kSegmentationClassColors,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: AspectRatio(
        aspectRatio: displayAspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (showVirtualBackground && virtualBackground != null)
              Positioned.fill(
                child: CustomPaint(
                  painter: BackgroundImagePainter(image: virtualBackground!),
                ),
              ),
            cameraPreview,
            if (showVirtualBackground &&
                virtualBackground != null &&
                segmentationMask != null)
              CustomPaint(
                painter: VirtualBackgroundOverlayPainter(
                  background: virtualBackground!,
                  mask: segmentationMask!,
                  mirrorHorizontally: mirrorHorizontally,
                ),
              ),
            if (showSegmentation &&
                !showVirtualBackground &&
                segmentationMask != null)
              CustomPaint(
                painter: LiveSegmentationPainter(
                  mask: segmentationMask!,
                  maskColor: segmentationColor,
                  showAllClasses: segmentationShowAllClasses,
                  mirrorHorizontally: mirrorHorizontally,
                  classColors: segmentationClassColors,
                ),
              ),
            if (imageSize != null)
              CustomPaint(
                painter: CameraDetectionPainter(
                  faces: faces,
                  imageSize: imageSize!,
                  cameraAspectRatio: cameraAspectRatio,
                  displayAspectRatio: displayAspectRatio,
                  detectionMode: detectionMode,
                  sensorOrientation: sensorOrientation,
                  deviceOrientation: deviceOrientation,
                  isFrontCamera: isFrontCamera,
                  mirrorHorizontally: mirrorHorizontally,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
