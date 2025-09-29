import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

class TestPage extends StatefulWidget {
  const TestPage({super.key});
  @override
  State<TestPage> createState() => _TestPageState();
}

class _TestPageState extends State<TestPage> {
  Uint8List? _imageBytes;

  List<Detection> _detections = [];
  List<Offset> _faceMeshPoints = [];
  List<Offset> _irisPoints = [];
  Size? _originalSize;

  final FaceDetector _faceDetector = FaceDetector();

  @override
  void initState() {
    super.initState();
    _initFaceDetector();
  }

  Future<void> _initFaceDetector() async {
    try {
      await _faceDetector.initialize(model: FaceDetectionModel.backCamera);
    } catch (_) {}
    setState(() {});
  }

  Future<void> _pickAndRun() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
    if (picked == null) return;
    final bytes = await picked.readAsBytes();

    if (!_faceDetector.isReady) return;

    final size = await _faceDetector.getOriginalSize(bytes);
    if (!mounted) return;
    setState(() {
      _imageBytes = bytes;
      _originalSize = size;
      _detections = const [];
      _faceMeshPoints = const [];
      _irisPoints = const [];
    });

    final detections = await _faceDetector.getDetections(bytes);
    List<Offset> faceMeshPoints = const [];
    List<Offset> irisPoints = const [];

    if (detections.isNotEmpty) {
      faceMeshPoints = await _faceDetector.getFaceMeshFromDetections(bytes, detections);
      irisPoints = await _faceDetector.getIrisFromMesh(bytes, faceMeshPoints);
    }

    if (!mounted) return;
    setState(() {
      _detections = detections;
      _faceMeshPoints = faceMeshPoints;
      _irisPoints = irisPoints;
    });
  }

  Rect _destRectForContain(Size img, Size box) {
    final srcAR = img.width / img.height;
    final dstAR = box.width / box.height;
    if (srcAR > dstAR) {
      final w = box.width;
      final h = w / srcAR;
      final dy = (box.height - h) / 2;
      return Rect.fromLTWH(0, dy, w, h);
    } else {
      final h = box.height;
      final w = h * srcAR;
      final dx = (box.width - w) / 2;
      return Rect.fromLTWH(dx, 0, w, h);
    }
  }

  @override
  Widget build(BuildContext context) {
    final hasImage = _imageBytes != null && _originalSize != null;
    return Scaffold(
      appBar: AppBar(title: const Text('FDLite Mesh Demo')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _pickAndRun,
        label: const Text('Pick Image'),
        icon: const Icon(Icons.image),
      ),
      body: Center(
        child: hasImage
            ? LayoutBuilder(
          builder: (context, constraints) {
            return FittedBox(
              fit: BoxFit.contain,
              child: SizedBox(
                width: _originalSize!.width,
                height: _originalSize!.height,
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: Image.memory(_imageBytes!, fit: BoxFit.fill),
                    ),
                    Positioned.fill(
                      child: CustomPaint(
                        size: Size(_originalSize!.width, _originalSize!.height),
                        painter: _DetectionsPainter(
                          detections: _detections,
                          faceMeshPoints: _faceMeshPoints,
                          irisPoints: _irisPoints,
                          originalSize: _originalSize!,
                          imageRectOnCanvas: Rect.fromLTWH(
                            0, 0, _originalSize!.width, _originalSize!.height,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        )
            : const Text('Pick an image to run detection'),
      ),
    );
  }

  Size _fitSize(Size src, Size bound) {
    final scale = (bound.width / src.width < bound.height / src.height)
        ? bound.width / src.width
        : bound.height / src.height;
    return Size(src.width * scale, src.height * scale);
  }
}

class _DetectionsPainter extends CustomPainter {
  final List<Detection> detections;
  final List<Offset> faceMeshPoints;
  final List<Offset> irisPoints;
  final Size originalSize;
  final Rect imageRectOnCanvas;

  _DetectionsPainter({
    required this.detections,
    required this.faceMeshPoints,
    required this.irisPoints,
    required this.originalSize,
    required this.imageRectOnCanvas,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = const Color(0xFF00FFCC);

    final detKpPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFF89CFF0);

    final meshPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = const Color(0xFFFF00FF);

    final ox = imageRectOnCanvas.left;
    final oy = imageRectOnCanvas.top;
    final sx = imageRectOnCanvas.width / originalSize.width;
    final sy = imageRectOnCanvas.height / originalSize.height;

    for (final d in detections) {
      final rect = Rect.fromLTRB(
        ox + d.bbox.xmin * originalSize.width * sx,
        oy + d.bbox.ymin * originalSize.height * sy,
        ox + d.bbox.xmax * originalSize.width * sx,
        oy + d.bbox.ymax * originalSize.height * sy,
      );
      canvas.drawRect(rect, boxPaint);

      final lm = d.landmarks;
      for (final p in lm.values) {
        final x = ox + p.dx * sx;
        final y = oy + p.dy * sy;
        canvas.drawCircle(Offset(x, y), 3, detKpPaint);
      }
    }

    if (faceMeshPoints.isNotEmpty) {
      final scaled = faceMeshPoints.map((p) => Offset(ox + p.dx * sx, oy + p.dy * sy)).toList();
      canvas.drawPoints(PointMode.points, scaled, meshPaint);
    }

    if (irisPoints.isNotEmpty) {
      for (int i = 0; i + 4 < irisPoints.length; i += 5) {
        final five = irisPoints.sublist(i, i + 5);

        int centerIdx = 0;
        double best = double.infinity;
        for (int k = 0; k < 5; k++) {
          double s = 0;
          for (int j = 0; j < 5; j++) {
            if (j == k) continue;
            final dx = five[j].dx - five[k].dx;
            final dy = five[j].dy - five[k].dy;
            s += dx * dx + dy * dy;
          }
          if (s < best) {
            best = s;
            centerIdx = k;
          }
        }

        final others = <Offset>[];
        for (int j = 0; j < 5; j++) {
          if (j != centerIdx) others.add(five[j]);
        }

        double minX = others.first.dx, maxX = others.first.dx;
        double minY = others.first.dy, maxY = others.first.dy;
        for (final p in others) {
          if (p.dx < minX) minX = p.dx;
          if (p.dx > maxX) maxX = p.dx;
          if (p.dy < minY) minY = p.dy;
          if (p.dy > maxY) maxY = p.dy;
        }

        final cx = (minX + maxX) * 0.5;
        final cy = (minY + maxY) * 0.5;
        final rx = (maxX - minX) * 0.5;
        final ry = (maxY - minY) * 0.5;

        final centerScaled = Offset(ox + cx * sx, oy + cy * sy);
        final rect = Rect.fromCenter(
          center: centerScaled,
          width: (rx * 2) * sx,
          height: (ry * 2) * sy,
        );

        final fill = Paint()
          ..style = PaintingStyle.fill
          ..color = const Color(0xFF22AAFF).withOpacity(0.35)
          ..blendMode = BlendMode.srcOver;
        final stroke = Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5
          ..color = const Color(0xFF22AAFF).withOpacity(0.9);

        canvas.drawOval(rect, fill);
        canvas.drawOval(rect, stroke);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionsPainter old) {
    return old.detections != detections ||
        old.faceMeshPoints != faceMeshPoints ||
        old.irisPoints != irisPoints ||
        old.originalSize != originalSize ||
        old.imageRectOnCanvas != imageRectOnCanvas;
  }
}
