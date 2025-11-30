import 'dart:io';
import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:camera/camera.dart';
import 'package:camera_macos/camera_macos.dart';
import 'package:image/image.dart' as img;
import 'package:face_detection_tflite/face_detection_tflite.dart';

Future<void> main() async {
  // Ensure platform plugins (camera_macos, etc.) are registered before use.
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(
    debugShowCheckedModeBanner: false,
    home: HomeScreen(),
  ));
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Detection Demo'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.face, size: 100, color: Colors.blue[300]),
            const SizedBox(height: 40),
            const Text(
              'Choose Detection Mode',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 40),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const Example()),
                );
              },
              icon: const Icon(Icons.image, size: 32),
              label: const Text('Still Image Detection',
                  style: TextStyle(fontSize: 18)),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                padding:
                    const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                minimumSize: const Size(300, 70),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const LiveCameraScreen()),
                );
              },
              icon: const Icon(Icons.videocam, size: 32),
              label: const Text('Live Camera Detection',
                  style: TextStyle(fontSize: 18)),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
                padding:
                    const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                minimumSize: const Size(300, 70),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class Example extends StatefulWidget {
  const Example({super.key});
  @override
  State<Example> createState() => _ExampleState();
}

class _ExampleState extends State<Example> {
  final FaceDetector _faceDetector = FaceDetector();
  Uint8List? _imageBytes;
  List<Face> _faces = [];
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
    final ImagePicker picker = ImagePicker();
    final XFile? picked =
        await picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
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

    final Uint8List bytes = await picked.readAsBytes();

    if (!_faceDetector.isReady) {
      setState(() => _isLoading = false);
      return;
    }

    await _processImage(bytes);
  }

  Future<void> _processImage(Uint8List bytes) async {
    setState(() => _isLoading = true);

    final DateTime totalStart = DateTime.now();
    final FaceDetectionMode mode = _determineMode();

    final DateTime detectionStart = DateTime.now();
    final List<Face> faces = await _faceDetector.detectFaces(bytes, mode: mode);
    final DateTime detectionEnd = DateTime.now();
    final Size decodedSize = faces.isNotEmpty
        ? faces.first.originalSize
        : await _faceDetector.getOriginalSize(bytes);

    if (!mounted) return;

    final DateTime totalEnd = DateTime.now();
    final int totalTime = totalEnd.difference(totalStart).inMilliseconds;
    final int detectionTime =
        detectionEnd.difference(detectionStart).inMilliseconds;

    int? meshTime;
    int? irisTime;
    if (_showMesh || _showIrises) {
      final int extraTime = totalTime - detectionTime;
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
      _originalSize = decodedSize;
      _faces = faces;
      _hasProcessedMesh =
          mode == FaceDetectionMode.standard || mode == FaceDetectionMode.full;
      _hasProcessedIris = mode == FaceDetectionMode.full;
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
    final FaceDetectionMode oldMode = _determineMode();

    if (feature == 'mesh') {
      setState(() => _showMesh = newValue);
    } else if (feature == 'iris') {
      setState(() => _showIrises = newValue);
    }

    final FaceDetectionMode newMode = _determineMode();

    if (_imageBytes != null && oldMode != newMode) {
      await _processImage(_imageBytes!);
    }
  }

  void _pickColor(
      String label, Color currentColor, ValueChanged<Color> onColorChanged) {
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

  Widget _buildTimingRow(String label, int milliseconds, Color color,
      {bool isBold = false}) {
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
        color: color.withAlpha(26),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withAlpha(77)),
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
            _buildStatusRow('Mesh', _hasProcessedMesh,
                _showMesh ? Colors.green : Colors.grey),
            _buildStatusRow('Iris', _hasProcessedIris,
                _showIrises ? Colors.green : Colors.grey),
            if (_detectionTimeMs != null ||
                _meshTimeMs != null ||
                _irisTimeMs != null ||
                _totalTimeMs != null) ...[
              const Divider(height: 16),
              if (_detectionTimeMs != null)
                _buildTimingRow('Detection', _detectionTimeMs!, Colors.green),
              if (_meshTimeMs != null)
                _buildTimingRow('Mesh Refinement', _meshTimeMs!, Colors.pink),
              if (_irisTimeMs != null)
                _buildTimingRow(
                    'Iris Refinement', _irisTimeMs!, Colors.blueAccent),
              if (_totalTimeMs != null)
                _buildTimingRow('Inference Time', _totalTimeMs!, Colors.blue,
                    isBold: true),
              const SizedBox(height: 8),
              if (_totalTimeMs != null) _buildPerformanceIndicator(),
            ],
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
    final bool hasImage = _imageBytes != null && _originalSize != null;
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
                              onPressed: () =>
                                  setState(() => _showSettings = false),
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
                            (value) => setState(
                                () => _showBoundingBoxes = value ?? false),
                          ),
                          _buildCheckbox(
                            'Show Mesh',
                            _showMesh,
                            (value) => _onFeatureToggle('mesh', value ?? false),
                          ),
                          _buildCheckbox(
                            'Show Landmarks',
                            _showLandmarks,
                            (value) =>
                                setState(() => _showLandmarks = value ?? false),
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
                            (color) =>
                                setState(() => _boundingBoxColor = color),
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
                            (value) =>
                                setState(() => _boundingBoxThickness = value),
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
              _buildFeatureStatus(),
              Expanded(
                child: Center(
                  child: hasImage
                      ? LayoutBuilder(
                          builder: (context, constraints) {
                            final fitted = applyBoxFit(
                              BoxFit.contain,
                              _originalSize!,
                              Size(constraints.maxWidth, constraints.maxHeight),
                            );
                            final Size renderSize = fitted.destination;
                            final Rect imageRect = Alignment.center.inscribe(
                              renderSize,
                              Offset.zero &
                                  Size(constraints.maxWidth,
                                      constraints.maxHeight),
                            );

                            return Stack(
                              children: [
                                Positioned.fromRect(
                                  rect: imageRect,
                                  child: SizedBox.fromSize(
                                    size: renderSize,
                                    child: Image.memory(
                                      _imageBytes!,
                                      fit: BoxFit.fill,
                                    ),
                                  ),
                                ),
                                Positioned(
                                  left: imageRect.left,
                                  top: imageRect.top,
                                  width: imageRect.width,
                                  height: imageRect.height,
                                  child: CustomPaint(
                                    size:
                                        Size(imageRect.width, imageRect.height),
                                    painter: _DetectionsPainter(
                                      faces: _faces,
                                      imageRectOnCanvas: Rect.fromLTWH(0, 0,
                                          imageRect.width, imageRect.height),
                                      originalImageSize: _originalSize!,
                                      showBoundingBoxes: _showBoundingBoxes,
                                      showMesh: _showMesh,
                                      showLandmarks: _showLandmarks,
                                      showIrises: _showIrises,
                                      boundingBoxColor: _boundingBoxColor,
                                      landmarkColor: _landmarkColor,
                                      meshColor: _meshColor,
                                      irisColor: _irisColor,
                                      boundingBoxThickness:
                                          _boundingBoxThickness,
                                      landmarkSize: _landmarkSize,
                                      meshSize: _meshSize,
                                    ),
                                  ),
                                ),
                              ],
                            );
                          },
                        )
                      : Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.image_outlined,
                                size: 64, color: Colors.grey[400]),
                            const SizedBox(height: 16),
                            Text(
                              'No image selected',
                              style: TextStyle(
                                  fontSize: 18, color: Colors.grey[600]),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Pick an image to start detection',
                              style: TextStyle(
                                  fontSize: 14, color: Colors.grey[500]),
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
                backgroundColor: Colors.black.withAlpha(179),
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

  Widget _buildCheckbox(
      String label, bool value, ValueChanged<bool?> onChanged) {
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

  Widget _buildColorButton(
      String label, Color color, ValueChanged<Color> onColorChanged) {
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

  Widget _buildSlider(String label, double value, double min, double max,
      ValueChanged<double> onChanged) {
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
  final List<Face> faces;
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
        final BoundingBox bbox = face.bbox;
        final ui.Rect rect = Rect.fromLTRB(
          ox + bbox.topLeft.x * scaleX,
          oy + bbox.topLeft.y * scaleY,
          ox + bbox.bottomRight.x * scaleX,
          oy + bbox.bottomRight.y * scaleY,
        );
        canvas.drawRect(rect, boxPaint);
      }

      if (showLandmarks) {
        for (final Point<double> p in face.landmarks.values) {
          canvas.drawCircle(
            Offset(ox + p.x * scaleX, oy + p.y * scaleY),
            landmarkSize,
            detKpPaint,
          );
        }
      }

      if (showMesh) {
        final List<Point<double>> mesh = face.mesh;
        if (mesh.isNotEmpty) {
          final double imgArea =
              imageRectOnCanvas.width * imageRectOnCanvas.height;
          final double radius = meshSize + sqrt(imgArea) / 1000.0;

          for (final Point<double> p in mesh) {
            canvas.drawCircle(
              Offset(ox + p.x * scaleX, oy + p.y * scaleY),
              radius,
              meshPaint,
            );
          }
        }
      }

      if (showIrises) {
        final irisPair = face.irises;
        if (irisPair != null) {
          for (final iris in [irisPair.leftIris, irisPair.rightIris]) {
            if (iris == null) continue;

            // Access contour
            final List<Point<double>> contour = iris.contour;
            double minX = contour.first.x, maxX = contour.first.x;
            double minY = contour.first.y, maxY = contour.first.y;
            for (final p in contour) {
              if (p.x < minX) minX = p.x;
              if (p.x > maxX) maxX = p.x;
              if (p.y < minY) minY = p.y;
              if (p.y > maxY) maxY = p.y;
            }

            final cx = ox + ((minX + maxX) * 0.5) * scaleX;
            final cy = oy + ((minY + maxY) * 0.5) * scaleY;
            final rx = (maxX - minX) * 0.5 * scaleX;
            final ry = (maxY - minY) * 0.5 * scaleY;

            final oval = Rect.fromCenter(
                center: Offset(cx, cy), width: rx * 2, height: ry * 2);
            canvas.drawOval(oval, irisFill);
            canvas.drawOval(oval, irisStroke);
          }
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

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  bool get _isMacOS => !kIsWeb && Platform.isMacOS;

  CameraController? _cameraController;
  CameraMacOSController? _macCameraController;
  Size? _macPreviewSize;
  final FaceDetector _faceDetector = FaceDetector();
  List<Face> _faces = [];
  Size? _imageSize;
  bool _isProcessing = false;
  bool _isInitialized = false;
  int _frameCounter = 0;
  int _processEveryNFrames =
      3; // Process every 3rd frame for better performance
  int _detectionTimeMs = 0;
  int _fps = 0;
  DateTime? _lastFpsUpdate;
  int _framesSinceLastUpdate = 0;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      // Initialize face detector
      await _faceDetector.initialize(model: FaceDetectionModel.backCamera);

      if (_isMacOS) {
        if (mounted) {
          setState(() {
            _isInitialized = true;
          });
        }
        return;
      }

      // Get available cameras
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No cameras available')),
          );
        }
        return;
      }

      // Use the first available camera (usually front camera)
      final camera = cameras.first;

      _cameraController = CameraController(
        camera,
        ResolutionPreset
            .medium, // Use medium for balance between quality and speed
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420, // Efficient format
      );

      await _cameraController!.initialize();

      if (!mounted) return;

      setState(() {
        _isInitialized = true;
      });

      // Start image stream
      _cameraController!.startImageStream(_processCameraImage);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error initializing camera: $e')),
        );
      }
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    _frameCounter++;

    // Calculate FPS
    _framesSinceLastUpdate++;
    final now = DateTime.now();
    if (_lastFpsUpdate != null) {
      final diff = now.difference(_lastFpsUpdate!).inMilliseconds;
      if (diff >= 1000) {
        setState(() {
          _fps = (_framesSinceLastUpdate * 1000 / diff).round();
          _framesSinceLastUpdate = 0;
          _lastFpsUpdate = now;
        });
      }
    } else {
      _lastFpsUpdate = now;
    }

    // Skip frames for better performance
    if (_frameCounter % _processEveryNFrames != 0) return;

    // Skip if already processing
    if (_isProcessing) return;

    _isProcessing = true;

    try {
      final startTime = DateTime.now();

      // Convert CameraImage to bytes
      final bytes = await _convertCameraImageToBytes(image);

      if (bytes == null) {
        _isProcessing = false;
        return;
      }

      // Run face detection (fast mode for bounding boxes only)
      final faces = await _faceDetector.detectFaces(
        bytes,
        mode: FaceDetectionMode.fast, // Fast mode for real-time performance
      );

      final endTime = DateTime.now();
      final detectionTime = endTime.difference(startTime).inMilliseconds;

      if (mounted) {
        setState(() {
          _faces = faces;
          _imageSize = Size(image.width.toDouble(), image.height.toDouble());
          _detectionTimeMs = detectionTime;
        });
      }
    } catch (e) {
      // Silently handle errors during processing
    } finally {
      _isProcessing = false;
    }
  }

  Future<Uint8List?> _convertCameraImageToBytes(CameraImage image) async {
    try {
      // Convert YUV420 to RGB
      final int width = image.width;
      final int height = image.height;

      // Create an image using the img package
      final imgLib = img.Image(width: width, height: height);

      // YUV420 format has Y plane and UV planes
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int uvIndex =
              uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
          final int index = y * width + x;

          final yp = image.planes[0].bytes[index];
          final up = image.planes[1].bytes[uvIndex];
          final vp = image.planes[2].bytes[uvIndex];

          // Convert YUV to RGB
          int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
          int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
              .round()
              .clamp(0, 255);
          int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

          imgLib.setPixelRgb(x, y, r, g, b);
        }
      }

      // Encode to JPEG
      return Uint8List.fromList(img.encodeJpg(imgLib));
    } catch (e) {
      return null;
    }
  }

  Uint8List? _convertMacImageToBytes(CameraImageData image) {
    try {
      final imgLib = img.Image(width: image.width, height: image.height);
      final bytes = image.bytes;
      final stride = image.bytesPerRow;

      for (int y = 0; y < image.height; y++) {
        final rowStart = y * stride;
        for (int x = 0; x < image.width; x++) {
          final pixelStart = rowStart + x * 4;
          if (pixelStart + 3 >= bytes.length) break;

          final a = bytes[pixelStart];
          final r = bytes[pixelStart + 1];
          final g = bytes[pixelStart + 2];
          final b = bytes[pixelStart + 3];

          imgLib.setPixelRgba(x, y, r, g, b, a);
        }
      }

      return Uint8List.fromList(img.encodeJpg(imgLib));
    } catch (_) {
      return null;
    }
  }

  @override
  void dispose() {
    if (_isMacOS) {
      _macCameraController?.stopImageStream();
      _macCameraController?.destroy();
    } else {
      _cameraController?.stopImageStream();
      _cameraController?.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isMacOS) {
      return _buildMacOSCamera(context);
    }

    if (!_isInitialized || _cameraController == null) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Live Camera Detection'),
          backgroundColor: Colors.green,
          foregroundColor: Colors.white,
        ),
        body: const Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    final cameraAspectRatio = _cameraController!.value.aspectRatio;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Camera Detection'),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
        actions: [
          Center(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'FPS: $_fps | Detection: ${_detectionTimeMs}ms',
                style: const TextStyle(fontSize: 14),
              ),
            ),
          ),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: cameraAspectRatio,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(_cameraController!),
                  if (_imageSize != null)
                    CustomPaint(
                      painter: _CameraDetectionPainter(
                        faces: _faces,
                        imageSize: _imageSize!,
                        cameraAspectRatio: cameraAspectRatio,
                      ),
                    ),
                ],
              ),
            ),
          ),
          // Info panel
          Positioned(
            bottom: 20,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.black.withAlpha(179),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'Faces Detected: ${_faces.length}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Text(
                          'Frame Skip: ',
                          style: TextStyle(color: Colors.white70, fontSize: 14),
                        ),
                        DropdownButton<int>(
                          value: _processEveryNFrames,
                          dropdownColor: Colors.black87,
                          style: const TextStyle(color: Colors.white),
                          items: [1, 2, 3, 4, 5]
                              .map((n) => DropdownMenuItem(
                                    value: n,
                                    child: Text('1/$n'),
                                  ))
                              .toList(),
                          onChanged: (value) {
                            if (value != null) {
                              setState(() {
                                _processEveryNFrames = value;
                              });
                            }
                          },
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMacOSCamera(BuildContext context) {
    if (!_isInitialized) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Live Camera Detection'),
          backgroundColor: Colors.green,
          foregroundColor: Colors.white,
        ),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    final size = MediaQuery.of(context).size;
    final double cameraAspectRatio = _macPreviewSize != null
        ? _macPreviewSize!.width / _macPreviewSize!.height
        : size.width / size.height;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Camera Detection'),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
        actions: [
          Center(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'FPS: $_fps | Detection: ${_detectionTimeMs}ms',
                style: const TextStyle(fontSize: 14),
              ),
            ),
          ),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: cameraAspectRatio,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraMacOSView(
                    cameraMode: CameraMacOSMode.photo,
                    fit: BoxFit.contain,
                    onCameraInizialized: _onMacCameraInitialized,
                    onCameraLoading: (_) =>
                        const Center(child: CircularProgressIndicator()),
                  ),
                  if (_imageSize != null)
                    CustomPaint(
                      painter: _CameraDetectionPainter(
                        faces: _faces,
                        imageSize: _imageSize!,
                        cameraAspectRatio: cameraAspectRatio,
                      ),
                    ),
                ],
              ),
            ),
          ),
          Positioned(
            bottom: 20,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.black.withAlpha(179),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'Faces Detected: ${_faces.length}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Text(
                          'Frame Skip: ',
                          style: TextStyle(color: Colors.white70, fontSize: 14),
                        ),
                        DropdownButton<int>(
                          value: _processEveryNFrames,
                          dropdownColor: Colors.black87,
                          style: const TextStyle(color: Colors.white),
                          items: [1, 2, 3, 4, 5]
                              .map((n) => DropdownMenuItem(
                                    value: n,
                                    child: Text('1/$n'),
                                  ))
                              .toList(),
                          onChanged: (value) {
                            if (value != null) {
                              setState(() {
                                _processEveryNFrames = value;
                              });
                            }
                          },
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _onMacCameraInitialized(CameraMacOSController controller) {
    _macCameraController = controller;
    _macPreviewSize = controller.args.size;
    setState(() {
      _isInitialized = true;
    });
    _startMacImageStream();
  }

  void _startMacImageStream() {
    if (_macCameraController == null) return;

    _macCameraController!.startImageStream((image) async {
      if (image == null) return;

      _frameCounter++;
      _framesSinceLastUpdate++;
      final now = DateTime.now();
      if (_lastFpsUpdate != null) {
        final diff = now.difference(_lastFpsUpdate!).inMilliseconds;
        if (diff >= 1000) {
          setState(() {
            _fps = (_framesSinceLastUpdate * 1000 / diff).round();
            _framesSinceLastUpdate = 0;
            _lastFpsUpdate = now;
          });
        }
      } else {
        _lastFpsUpdate = now;
      }

      if (_frameCounter % _processEveryNFrames != 0) return;
      if (_isProcessing) return;

      _isProcessing = true;

      try {
        final startTime = DateTime.now();
        final bytes = _convertMacImageToBytes(image);
        if (bytes == null) {
          _isProcessing = false;
          return;
        }

        final faces = await _faceDetector.detectFaces(
          bytes,
          mode: FaceDetectionMode.fast,
        );
        final detectionTime =
            DateTime.now().difference(startTime).inMilliseconds;

        if (mounted) {
          setState(() {
            _faces = faces;
            _imageSize = Size(image.width.toDouble(), image.height.toDouble());
            _macPreviewSize ??=
                Size(image.width.toDouble(), image.height.toDouble());
            _detectionTimeMs = detectionTime;
          });
        }
      } catch (_) {
        // Ignore frame errors to keep streaming
      } finally {
        _isProcessing = false;
      }
    });
  }
}

class _CameraDetectionPainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final double cameraAspectRatio;

  _CameraDetectionPainter({
    required this.faces,
    required this.imageSize,
    required this.cameraAspectRatio,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (faces.isEmpty) return;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = const Color(0xFF00FFCC);

    // Calculate the display area for the camera preview using the actual canvas size
    final screenAspectRatio = size.width / size.height;
    double displayWidth, displayHeight;
    double offsetX = 0, offsetY = 0;

    if (cameraAspectRatio > screenAspectRatio) {
      displayWidth = size.width;
      displayHeight = size.width / cameraAspectRatio;
      offsetY = (size.height - displayHeight) / 2;
    } else {
      displayHeight = size.height;
      displayWidth = size.height * cameraAspectRatio;
      offsetX = (size.width - displayWidth) / 2;
    }

    final scaleX = displayWidth / imageSize.width;
    final scaleY = displayHeight / imageSize.height;

    // Draw bounding boxes
    for (final face in faces) {
      final bbox = face.bbox;
      final rect = Rect.fromLTRB(
        offsetX + bbox.topLeft.x * scaleX,
        offsetY + bbox.topLeft.y * scaleY,
        offsetX + bbox.bottomRight.x * scaleX,
        offsetY + bbox.bottomRight.y * scaleY,
      );
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _CameraDetectionPainter old) {
    return old.faces != faces ||
        old.imageSize != imageSize ||
        old.cameraAspectRatio != cameraAspectRatio;
  }
}
