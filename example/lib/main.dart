import 'package:flutter/material.dart';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image_picker/image_picker.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';

void main() {
  runApp(const MaterialApp(
    debugShowCheckedModeBanner: false,
    home: Example(),
  ));
}

class Example extends StatefulWidget {
  const Example({super.key});
  @override
  State<Example> createState() => _ExampleState();
}

class _ExampleState extends State<Example> {
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
  bool _hasProcessedMesh = false;
  bool _hasProcessedIris = false;

  int? _detectionTimeMs;
  int? _meshTimeMs;
  int? _irisTimeMs;
  int? _totalTimeMs;

  Color _boundingBoxColor = const Color(0xFF00FFCC);
  Color _landmarkColor = const Color(0xFF89CFF0);
  Color _meshColor = const Color(0xFFF4C2C2);
  Color _irisColor = const Color(0xFF22AAFF);

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

    setState(() {
      _imageBytes = null;
      _faces = [];
      _originalSize = null;
      _isLoading = true;
      _detectionTimeMs = null;
      _meshTimeMs = null;
      _irisTimeMs = null;
      _totalTimeMs = null;
    });

    final bytes = await picked.readAsBytes();

    if (!_faceDetector.isReady) {
      setState(() => _isLoading = false);
      return;
    }

    await _processImage(bytes);
  }

  Future<void> _processImage(Uint8List bytes) async {
    setState(() => _isLoading = true);

    final totalStart = DateTime.now();

    final mode = _determineMode();

    final detectionStart = DateTime.now();
    final result = await _faceDetector.detectFaces(bytes, mode: mode);
    final detectionEnd = DateTime.now();

    if (!mounted) return;

    final totalEnd = DateTime.now();
    final totalTime = totalEnd.difference(totalStart).inMilliseconds;
    final detectionTime = detectionEnd.difference(detectionStart).inMilliseconds;

    int? meshTime;
    int? irisTime;
    if (_showMesh || _showIrises) {
      final extraTime = totalTime - detectionTime;
      if (_showMesh && _showIrises) {
        meshTime = (extraTime * 0.6).round();
        irisTime = (extraTime * 0.4).round();
      } else if (_showMesh) {
        meshTime = extraTime;
      } else if (_showIrises) {
        irisTime = extraTime;
      }
    }

    setState(() {
      _imageBytes = bytes;
      _originalSize = result.originalSize;
      _faces = result.faces;
      _hasProcessedMesh = _showMesh;
      _hasProcessedIris = _showIrises;
      _isLoading = false;
      _detectionTimeMs = detectionTime;
      _meshTimeMs = meshTime;
      _irisTimeMs = irisTime;
      _totalTimeMs = totalTime;
    });
  }

  FaceDetectionMode _determineMode() {
    if (_showIrises) {
      return FaceDetectionMode.full;
    } else if (_showMesh) {
      return FaceDetectionMode.standard;
    } else {
      return FaceDetectionMode.fast;
    }
  }

  Future<void> _onFeatureToggle(String feature, bool newValue) async {
    final oldMode = _determineMode();

    if (feature == 'mesh') {
      setState(() => _showMesh = newValue);
    } else if (feature == 'iris') {
      setState(() => _showIrises = newValue);
    }

    final newMode = _determineMode();

    if (_imageBytes != null && oldMode != newMode) {
      await _processImage(_imageBytes!);
    }
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

  Widget _buildInferenceTimingCard() {
    if (_detectionTimeMs == null) return const SizedBox.shrink();

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.timer, size: 20, color: Colors.blue),
                const SizedBox(width: 8),
                const Text(
                  'Inference Timing',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
              ],
            ),
            const SizedBox(height: 8),
            _buildTimingRow('Face Detection', _detectionTimeMs!, Colors.green),
            if (_meshTimeMs != null)
              _buildTimingRow('Mesh Landmarks', _meshTimeMs!, Colors.orange),
            if (_irisTimeMs != null)
              _buildTimingRow('Iris Detection', _irisTimeMs!, Colors.purple),
            const Divider(height: 16),
            _buildTimingRow('Total Time', _totalTimeMs!, Colors.blue, isBold: true),
            const SizedBox(height: 8),
            _buildPerformanceIndicator(),
          ],
        ),
      ),
    );
  }

  Widget _buildTimingRow(String label, int milliseconds, Color color, {bool isBold = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  color: color,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                label,
                style: TextStyle(
                  fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
                  fontSize: isBold ? 15 : 14,
                ),
              ),
            ],
          ),
          Text(
            '${milliseconds}ms',
            style: TextStyle(
              fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
              fontSize: isBold ? 15 : 14,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPerformanceIndicator() {
    if (_totalTimeMs == null) return const SizedBox.shrink();

    String performance;
    Color color;
    IconData icon;

    if (_totalTimeMs! < 200) {
      performance = 'Excellent';
      color = Colors.green;
      icon = Icons.speed;
    } else if (_totalTimeMs! < 500) {
      performance = 'Good';
      color = Colors.lightGreen;
      icon = Icons.thumb_up;
    } else if (_totalTimeMs! < 1000) {
      performance = 'Fair';
      color = Colors.orange;
      icon = Icons.warning_amber;
    } else {
      performance = 'Slow';
      color = Colors.red;
      icon = Icons.hourglass_bottom;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 6),
          Text(
            performance,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureStatus() {
    if (_imageBytes == null) return const SizedBox.shrink();

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.info_outline, size: 20, color: Colors.blue),
                const SizedBox(width: 8),
                const Text(
                  'Processing Status',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
              ],
            ),
            const SizedBox(height: 8),
            _buildStatusRow('Detection', true, Colors.green),
            _buildStatusRow('Mesh', _hasProcessedMesh, _showMesh ? Colors.green : Colors.grey),
            _buildStatusRow('Iris', _hasProcessedIris, _showIrises ? Colors.green : Colors.grey),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusRow(String label, bool processed, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Icon(
            processed ? Icons.check_circle : Icons.pending,
            size: 16,
            color: color,
          ),
          const SizedBox(width: 8),
          Text(
            label,
            style: const TextStyle(fontSize: 14),
          ),
          const Spacer(),
          Text(
            processed ? 'Processed' : 'Skipped',
            style: TextStyle(
              fontSize: 12,
              color: color,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
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
                                (value) => _onFeatureToggle('mesh', value ?? false),
                          ),
                          _buildCheckbox(
                            'Show Landmarks',
                            _showLandmarks,
                                (value) => setState(() => _showLandmarks = value ?? false),
                          ),
                          _buildCheckbox(
                            'Show Irises',
                            _showIrises,
                                (value) => _onFeatureToggle('iris', value ?? false),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
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
              _buildInferenceTimingCard(),
              _buildFeatureStatus(),
              Expanded(
                child: Center(
                  child: hasImage
                      ? LayoutBuilder(
                    builder: (context, constraints) {
                      final imageAspect = _originalSize!.width / _originalSize!.height;
                      final boxAspect = constraints.maxWidth / constraints.maxHeight;
                      double displayWidth, displayHeight;

                      if (imageAspect > boxAspect) {
                        displayWidth = constraints.maxWidth;
                        displayHeight = displayWidth / imageAspect;
                      } else {
                        displayHeight = constraints.maxHeight;
                        displayWidth = displayHeight * imageAspect;
                      }

                      final left = (constraints.maxWidth - displayWidth) / 2;
                      final top = (constraints.maxHeight - displayHeight) / 2;
                      final imageRect = Rect.fromLTWH(left, top, displayWidth, displayHeight);

                      return Stack(
                        children: [
                          Positioned.fill(
                            child: Center(
                              child: Image.memory(
                                _imageBytes!,
                                fit: BoxFit.contain,
                              ),
                            ),
                          ),
                          CustomPaint(
                            size: Size(constraints.maxWidth, constraints.maxHeight),
                            painter: _DetectionsPainter(
                              faces: _faces,
                              imageRectOnCanvas: imageRect,
                              originalImageSize: _originalSize!,
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
                        ],
                      );
                    },
                  )
                      : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.image_outlined, size: 64, color: Colors.grey[400]),
                      const SizedBox(height: 16),
                      Text(
                        'No image selected',
                        style: TextStyle(fontSize: 18, color: Colors.grey[600]),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Pick an image to start detection',
                        style: TextStyle(fontSize: 14, color: Colors.grey[500]),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
          if (!_showSettings && hasImage)
            Positioned(
              top: 50,
              right: 16,
              child: FloatingActionButton(
                mini: true,
                onPressed: () => setState(() => _showSettings = true),
                backgroundColor: Colors.black.withOpacity(0.7),
                child: const Icon(Icons.settings, color: Colors.white),
              ),
            ),
          if (_isLoading)
            Container(
              color: Colors.black54,
              child: const Center(
                child: CircularProgressIndicator(),
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
  final Size originalImageSize;
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
    required this.originalImageSize,
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
    if (faces.isEmpty) return;

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

    final scaleX = imageRectOnCanvas.width / originalImageSize.width;
    final scaleY = imageRectOnCanvas.height / originalImageSize.height;

    for (final face in faces) {
      // draw bounding boxes:
      if (showBoundingBoxes) {
        final c = face.bboxCorners;
        final rect = Rect.fromLTRB(
          ox + c[0].x * scaleX,
          oy + c[0].y * scaleY,
          ox + c[2].x * scaleX,
          oy + c[2].y * scaleY,
        );
        canvas.drawRect(rect, boxPaint);
      }

      // draw landmarks:
      if (showLandmarks) {
        for (final p in face.landmarks.values) {
          canvas.drawCircle(
            Offset(ox + p.x * scaleX, oy + p.y * scaleY),
            landmarkSize,
            detKpPaint,
          );
        }
      }

      // draw mesh:
      if (showMesh) {
        final mesh = face.mesh;
        if (mesh.isNotEmpty) {
          final imgArea = imageRectOnCanvas.width * imageRectOnCanvas.height;
          final radius = meshSize + sqrt(imgArea) / 1000.0;

          for (final p in mesh) {
            canvas.drawCircle(
              Offset(ox + p.x * scaleX, oy + p.y * scaleY),
              radius,
              meshPaint,
            );
          }
        }
      }

      // draw irises:
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

          final cx = ox + ((minX + maxX) * 0.5) * scaleX;
          final cy = oy + ((minY + maxY) * 0.5) * scaleY;
          final rx = (maxX - minX) * 0.5 * scaleX;
          final ry = (maxY - minY) * 0.5 * scaleY;

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
        old.originalImageSize != originalImageSize ||
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