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
    } catch (e) {
      print(e);
    }

    setState(() {});
  }

  Future<void> _pickAndRun() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
    if (picked == null) return;
    final bytes = await picked.readAsBytes();
    if (!_faceDetector.isReady) {
      print("face detector not ready");
      return;
    }

    print("fetching detections");

    final detections = await _faceDetector.getDetections(bytes);
    if (detections.isNotEmpty) {
      // Example of how to fetch specific landmarks from FaceIndex:
      final lm = detections.first.landmarks;
      final leftEye = lm[FaceIndex.leftEye]!;
      final rightEye = lm[FaceIndex.rightEye]!;
      print('Left eye pixel: $leftEye');
      print('Right eye pixel: $rightEye');
    }

    final faceMeshPoints = await _faceDetector.getFaceMeshFromDetections(bytes, detections);
    final irisPoints = await _faceDetector.getIrisFromMesh(bytes, faceMeshPoints);
    final size = await _faceDetector.getOriginalSize(bytes);

    print(irisPoints);

    setState(() {
      _imageBytes = bytes;
      _detections = detections;
      _faceMeshPoints = faceMeshPoints;
      _irisPoints = irisPoints;
      _originalSize = size;
    });
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
            final fitted = _fitSize(_originalSize!, Size(constraints.maxWidth, constraints.maxHeight));
            return SizedBox(
              width: fitted.width,
              height: fitted.height,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.memory(_imageBytes!, fit: BoxFit.contain),
                  CustomPaint(
                    painter: _DetectionsPainter(
                      detections: _detections,
                      faceMeshPoints: _faceMeshPoints,
                      irisPoints: _irisPoints,
                      originalSize: _originalSize!,
                    ),
                  ),
                ],
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

  _DetectionsPainter({
    required this.detections,
    required this.faceMeshPoints,
    required this.irisPoints,
    required this.originalSize,
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

    final irisPtPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFF89CFF0);

    for (final d in detections) {
      final rect = Rect.fromLTRB(
        d.bbox.xmin * size.width,
        d.bbox.ymin * size.height,
        d.bbox.xmax * size.width,
        d.bbox.ymax * size.height,
      );
      canvas.drawRect(rect, boxPaint);

      final scaleX = size.width / originalSize.width;
      final scaleY = size.height / originalSize.height;
      final lm = d.landmarks;
      for (final p in lm.values) {
        final x = p.dx * scaleX;
        final y = p.dy * scaleY;
        canvas.drawCircle(Offset(x, y), 3, detKpPaint);
      }
    }

    if (faceMeshPoints.isNotEmpty) {
      final scaled = faceMeshPoints
          .map((p) => Offset(p.dx * size.width / originalSize.width, p.dy * size.height / originalSize.height))
          .toList();
      canvas.drawPoints(PointMode.points, scaled, meshPaint);
    }

    if (irisPoints.isNotEmpty) {
      final scaleX = size.width / originalSize.width;
      final scaleY = size.height / originalSize.height;

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

        final centerScaled = Offset(cx * scaleX, cy * scaleY);
        final rect = Rect.fromCenter(center: centerScaled, width: rx * 2 * scaleX, height: ry * 2 * scaleY);

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
        old.originalSize != originalSize;
  }
}