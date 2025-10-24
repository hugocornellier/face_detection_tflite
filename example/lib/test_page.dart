import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';

class TestPage extends StatefulWidget {
  const TestPage({super.key});
  @override
  State<TestPage> createState() => _TestPageState();
}

class _TestPageState extends State<TestPage> {
  final FaceDetector _faceDetector = FaceDetector();

  Uint8List? _imageBytes;
  List<FaceResult> _faces = [];
  Size? _originalSize;
  bool _isLoading = false;

  bool _showBoundingBoxes = true;
  bool _showMesh = true;
  bool _showLandmarks = true;
  bool _showIrises = true;
  bool _showSettings = true;

  // Color controls
  Color _boundingBoxColor = const Color(0xFF00FFCC);
  Color _landmarkColor = const Color(0xFF89CFF0);
  Color _meshColor = const Color(0xFFF4C2C2);
  Color _irisColor = const Color(0xFF22AAFF);

  // Size controls
  double _boundingBoxThickness = 2.0;
  double _landmarkSize = 3.0;
  double _meshSize = 1.25;

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

    // Immediately clear old image and show loading state
    setState(() {
      _imageBytes = null;
      _faces = [];
      _originalSize = null;
      _isLoading = true;
    });

    final bytes = await picked.readAsBytes();

    if (!_faceDetector.isReady) {
      setState(() => _isLoading = false);
      return;
    }

    final result = await _faceDetector.detectFaces(bytes);
    if (!mounted) return;

    setState(() {
      _imageBytes = bytes;
      _originalSize = result.originalSize;
      _faces = result.faces;
      _isLoading = false;
    });
  }

  void _pickColor(String label, Color currentColor, ValueChanged<Color> onColorChanged) {
    showDialog(
      context: context,
      builder: (context) {
        Color tempColor = currentColor;
        return AlertDialog(
          title: Text('Pick $label Color'),
          content: SingleChildScrollView(
            child: ColorPicker(
              pickerColor: currentColor,
              onColorChanged: (color) {
                tempColor = color;
              },
              pickerAreaHeightPercent: 0.8,
              displayThumbColor: true,
              enableAlpha: true,
              labelTypes: const [ColorLabelType.hex],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                onColorChanged(tempColor);
                Navigator.of(context).pop();
              },
              child: const Text('Select'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasImage = _imageBytes != null && _originalSize != null;
    return Scaffold(
      body: Stack(
        children: [
          Column(
            children: [
              // Checkboxes panel
              if (_showSettings)
                Container(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Wrap(
                        spacing: 16,
                        runSpacing: 8,
                        children: [
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: ElevatedButton.icon(
                              onPressed: _pickAndRun,
                              icon: const Icon(Icons.image),
                              label: const Text('Pick Image'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.blue,
                                foregroundColor: Colors.white,
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: ElevatedButton.icon(
                              onPressed: () => setState(() => _showSettings = false),
                              icon: const Icon(Icons.visibility_off),
                              label: const Text('Hide Settings'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.black,
                                foregroundColor: Colors.white,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      // Visibility toggles
                      Wrap(
                        spacing: 16,
                        runSpacing: 8,
                        children: [
                          _buildCheckbox(
                            'Show Bounding Boxes',
                            _showBoundingBoxes,
                                (value) => setState(() => _showBoundingBoxes = value ?? false),
                          ),
                          _buildCheckbox(
                            'Show Mesh',
                            _showMesh,
                                (value) => setState(() => _showMesh = value ?? false),
                          ),
                          _buildCheckbox(
                            'Show Landmarks',
                            _showLandmarks,
                                (value) => setState(() => _showLandmarks = value ?? false),
                          ),
                          _buildCheckbox(
                            'Show Irises',
                            _showIrises,
                                (value) => setState(() => _showIrises = value ?? false),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      // Color pickers
                      Wrap(
                        spacing: 12,
                        runSpacing: 8,
                        children: [
                          _buildColorButton(
                            'Bounding Box',
                            _boundingBoxColor,
                                (color) => setState(() => _boundingBoxColor = color),
                          ),
                          _buildColorButton(
                            'Landmarks',
                            _landmarkColor,
                                (color) => setState(() => _landmarkColor = color),
                          ),
                          _buildColorButton(
                            'Mesh',
                            _meshColor,
                                (color) => setState(() => _meshColor = color),
                          ),
                          _buildColorButton(
                            'Irises',
                            _irisColor,
                                (color) => setState(() => _irisColor = color),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      // Size controls
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _buildSlider(
                            'Bounding Box Thickness',
                            _boundingBoxThickness,
                            0.5,
                            10.0,
                                (value) => setState(() => _boundingBoxThickness = value),
                          ),
                          _buildSlider(
                            'Landmark Size',
                            _landmarkSize,
                            0.5,
                            15.0,
                                (value) => setState(() => _landmarkSize = value),
                          ),
                          _buildSlider(
                            'Mesh Point Size',
                            _meshSize,
                            0.1,
                            10.0,
                                (value) => setState(() => _meshSize = value),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              if (_showSettings) const Divider(height: 1),
              // Image display area
              Expanded(
                child: Center(
                  child: _isLoading
                      ? const Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 16),
                      Text('Processing image...'),
                    ],
                  )
                      : hasImage
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
                                    showBoundingBoxes: _showBoundingBoxes,
                                    showMesh: _showMesh,
                                    showLandmarks: _showLandmarks,
                                    showIrises: _showIrises,
                                    boundingBoxColor: _boundingBoxColor,
                                    landmarkColor: _landmarkColor,
                                    meshColor: _meshColor,
                                    irisColor: _irisColor,
                                    boundingBoxThickness: _boundingBoxThickness,
                                    landmarkSize: _landmarkSize,
                                    meshSize: _meshSize,
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
              ),
            ],
          ),
          // Floating button to show settings when hidden
          if (!_showSettings)
            Positioned(
              top: 16,
              right: 16,
              child: Material(
                elevation: 4,
                borderRadius: BorderRadius.circular(8),
                child: InkWell(
                  onTap: () => setState(() => _showSettings = true),
                  borderRadius: BorderRadius.circular(8),
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.black,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(
                      Icons.settings,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildCheckbox(String label, bool value, ValueChanged<bool?> onChanged) {
    return InkWell(
      onTap: () => onChanged(!value),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Checkbox(
            value: value,
            onChanged: onChanged,
            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          ),
          Text(label),
        ],
      ),
    );
  }

  Widget _buildColorButton(String label, Color color, ValueChanged<Color> onColorChanged) {
    return InkWell(
      onTap: () => _pickColor(label, color, onColorChanged),
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 24,
              height: 24,
              decoration: BoxDecoration(
                color: color,
                border: Border.all(color: Colors.grey.shade400),
                borderRadius: BorderRadius.circular(4),
              ),
            ),
            const SizedBox(width: 8),
            Text(label),
            const SizedBox(width: 4),
            const Icon(Icons.arrow_drop_down, size: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildSlider(String label, double value, double min, double max, ValueChanged<double> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        children: [
          SizedBox(
            width: 180,
            child: Text(label, style: const TextStyle(fontSize: 14)),
          ),
          Expanded(
            child: Slider(
              value: value,
              min: min,
              max: max,
              divisions: ((max - min) * 10).round(),
              label: value.toStringAsFixed(1),
              onChanged: onChanged,
            ),
          ),
          SizedBox(
            width: 50,
            child: Text(
              value.toStringAsFixed(1),
              style: const TextStyle(fontSize: 14),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}

class _DetectionsPainter extends CustomPainter {
  final List<FaceResult> faces;
  final Rect imageRectOnCanvas;
  final bool showBoundingBoxes;
  final bool showMesh;
  final bool showLandmarks;
  final bool showIrises;
  final Color boundingBoxColor;
  final Color landmarkColor;
  final Color meshColor;
  final Color irisColor;
  final double boundingBoxThickness;
  final double landmarkSize;
  final double meshSize;

  _DetectionsPainter({
    required this.faces,
    required this.imageRectOnCanvas,
    required this.showBoundingBoxes,
    required this.showMesh,
    required this.showLandmarks,
    required this.showIrises,
    required this.boundingBoxColor,
    required this.landmarkColor,
    required this.meshColor,
    required this.irisColor,
    required this.boundingBoxThickness,
    required this.landmarkSize,
    required this.meshSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = boundingBoxThickness
      ..color = boundingBoxColor;

    final detKpPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = landmarkColor;

    final meshPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = meshColor;

    final irisFill = Paint()
      ..style = PaintingStyle.fill
      ..color = irisColor.withOpacity(0.6)
      ..blendMode = BlendMode.srcOver;

    final irisStroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = irisColor.withOpacity(0.9);

    final ox = imageRectOnCanvas.left;
    final oy = imageRectOnCanvas.top;

    for (final face in faces) {
      // Draw bounding box
      if (showBoundingBoxes) {
        final c = face.bboxCorners;
        final rect = Rect.fromLTRB(
          ox + c[0].x,
          oy + c[0].y,
          ox + c[2].x,
          oy + c[2].y,
        );
        canvas.drawRect(rect, boxPaint);
      }

      // Draw landmarks
      if (showLandmarks) {
        for (final p in face.landmarks.values) {
          canvas.drawCircle(Offset(ox + p.x, oy + p.y), landmarkSize, detKpPaint);
        }
      }

      // Draw mesh
      if (showMesh) {
        final mesh = face.mesh;
        if (mesh.isNotEmpty) {
          final imgArea = imageRectOnCanvas.width * imageRectOnCanvas.height;
          final radius = meshSize + sqrt(imgArea) / 1000.0;

          for (final p in mesh) {
            canvas.drawCircle(Offset(ox + p.x, oy + p.y), radius, meshPaint);
          }
        }
      }

      // Draw irises
      if (showIrises) {
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
  }

  @override
  bool shouldRepaint(covariant _DetectionsPainter old) {
    return old.faces != faces ||
        old.imageRectOnCanvas != imageRectOnCanvas ||
        old.showBoundingBoxes != showBoundingBoxes ||
        old.showMesh != showMesh ||
        old.showLandmarks != showLandmarks ||
        old.showIrises != showIrises ||
        old.boundingBoxColor != boundingBoxColor ||
        old.landmarkColor != landmarkColor ||
        old.meshColor != meshColor ||
        old.irisColor != irisColor ||
        old.boundingBoxThickness != boundingBoxThickness ||
        old.landmarkSize != landmarkSize ||
        old.meshSize != meshSize;
  }
}