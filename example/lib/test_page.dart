import 'dart:math';
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

  List<FaceResult> _faces = [];
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

    final result = await _faceDetector.detectFaces(bytes);
    if (!mounted) return;

    setState(() {
      _imageBytes = bytes;
      _originalSize = result.originalSize;
      _faces = result.faces;
    });
  }

  @override
  Widget build(BuildContext context) {
    final hasImage = _imageBytes != null && _originalSize != null;
    return Scaffold(
      appBar: AppBar(title: const Text('Face Detection Demo')),
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
                          faces: _faces,
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
}

class _DetectionsPainter extends CustomPainter {
  final List<FaceResult> faces;
  final Rect imageRectOnCanvas;

  _DetectionsPainter({
    required this.faces,
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
      ..style = PaintingStyle.fill
      ..color = const Color(0xFFF4C2C2);

    final irisFill = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFF22AAFF).withOpacity(0.6)
      ..blendMode = BlendMode.srcOver;

    final irisStroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = const Color(0xFF22AAFF).withOpacity(0.9);

    final ox = imageRectOnCanvas.left;
    final oy = imageRectOnCanvas.top;

    for (final face in faces) {
      final c = face.bboxCorners;
      final rect = Rect.fromLTRB(
        ox + c[0].x,
        oy + c[0].y,
        ox + c[2].x,
        oy + c[2].y,
      );
      canvas.drawRect(rect, boxPaint);

      for (final p in face.landmarks.values) {
        canvas.drawCircle(Offset(ox + p.x, oy + p.y), 3, detKpPaint);
      }

      final mesh = face.mesh;
      if (mesh.isNotEmpty) {
        final imgArea = imageRectOnCanvas.width * imageRectOnCanvas.height;
        final radius = 1.25 + sqrt(imgArea) / 1000.0;

        for (final p in mesh) {
          canvas.drawCircle(Offset(ox + p.x, oy + p.y), radius, meshPaint);
        }
      }

      final iris = face.irises;
      for (int i = 0; i + 4 < iris.length; i += 5) {
        final five = iris.sublist(i, i + 5);

        int centerIdx = 0;
        double best = double.infinity;
        for (int k = 0; k < 5; k++) {
          double s = 0;
          for (int j = 0; j < 5; j++) {
            if (j == k) continue;
            final dx = (five[j].x - five[k].x);
            final dy = (five[j].y - five[k].y);
            s += dx * dx + dy * dy;
          }
          if (s < best) {
            best = s;
            centerIdx = k;
          }
        }

        final others = <Point<double>>[];
        for (int j = 0; j < 5; j++) {
          if (j != centerIdx) others.add(five[j]);
        }

        double minX = others.first.x, maxX = others.first.x;
        double minY = others.first.y, maxY = others.first.y;
        for (final p in others) {
          if (p.x < minX) minX = p.x;
          if (p.x > maxX) maxX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.y > maxY) maxY = p.y;
        }

        final cx = ox + ((minX + maxX) * 0.5);
        final cy = oy + ((minY + maxY) * 0.5);
        final rx = (maxX - minX) * 0.5;
        final ry = (maxY - minY) * 0.5;

        final oval = Rect.fromCenter(center: Offset(cx, cy), width: rx * 2, height: ry * 2);
        canvas.drawOval(oval, irisFill);
        canvas.drawOval(oval, irisStroke);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionsPainter old) {
    return old.faces != faces || old.imageRectOnCanvas != imageRectOnCanvas;
  }
}
