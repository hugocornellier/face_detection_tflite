import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:camera/camera.dart';
import 'package:camera_macos/camera_macos.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

// Shared segmentation class colors for multiclass visualization.
const List<Color> _kSegmentationClassColors = [
  Color(0x99A0A0A0), // 0: Background - light gray
  Color(0x99CD853F), // 1: Hair - peru/tan brown
  Color(0x88FFA500), // 2: Body Skin - orange
  Color(0x88FF69B4), // 3: Face Skin - pink
  Color(0x9900BFFF), // 4: Clothes - deep sky blue
  Color(0x9940E0D0), // 5: Other - turquoise
];

const List<String> _kSegmentationClassLabels = [
  'BG',
  'Hair',
  'Body',
  'Face',
  'Clothes',
  'Other',
];

// Shared detection model dropdown items.
const List<DropdownMenuItem<FaceDetectionModel>> _kDetectionModelItems = [
  DropdownMenuItem(value: FaceDetectionModel.frontCamera, child: Text('Front')),
  DropdownMenuItem(value: FaceDetectionModel.backCamera, child: Text('Back')),
  DropdownMenuItem(value: FaceDetectionModel.shortRange, child: Text('Short')),
  DropdownMenuItem(value: FaceDetectionModel.full, child: Text('Full Range')),
  DropdownMenuItem(
      value: FaceDetectionModel.fullSparse, child: Text('Full Sparse')),
];

// Shared detection mode dropdown items.
const List<DropdownMenuItem<FaceDetectionMode>> _kDetectionModeItems = [
  DropdownMenuItem(value: FaceDetectionMode.fast, child: Text('Fast')),
  DropdownMenuItem(value: FaceDetectionMode.standard, child: Text('Standard')),
  DropdownMenuItem(value: FaceDetectionMode.full, child: Text('Full')),
];

// Performance classification based on processing time.
({String label, Color color, IconData icon}) _performanceLevel(int ms) {
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

// Compute the valid (non-padding) region of a segmentation mask.
({int x0, int y0, int x1, int y1}) _maskValidRegion(SegmentationMask mask) {
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

// Cover-fit scale and offset for rendering a source region into a viewport.
({double scale, double offsetX, double offsetY}) _coverFitScaleOffset(
    int sourceW, int sourceH, double viewW, double viewH) {
  final sourceAspect = sourceW / sourceH;
  final viewAspect = viewW / viewH;
  if (sourceAspect > viewAspect) {
    final s = viewH / sourceH;
    return (scale: s, offsetX: (viewW - sourceW * s) / 2, offsetY: 0.0);
  } else {
    final s = viewW / sourceW;
    return (scale: s, offsetX: 0.0, offsetY: (viewH - sourceH * s) / 2);
  }
}

// Draw multiclass segmentation labels at class centroids.
void _drawClassLabels(
    Canvas canvas, List<int> counts, List<double> sumX, List<double> sumY) {
  for (int c = 0; c < 6; c++) {
    if (counts[c] > 100) {
      final centroidX = sumX[c] / counts[c];
      final centroidY = sumY[c] / counts[c];
      final textPainter = TextPainter(
        text: TextSpan(
          text: _kSegmentationClassLabels[c],
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
        Offset(centroidX - textPainter.width / 2,
            centroidY - textPainter.height / 2),
      );
    }
  }
}

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
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const SegmentationDemoScreen()),
                );
              },
              icon: const Icon(Icons.person_outline, size: 32),
              label: const Text('Selfie Segmentation',
                  style: TextStyle(fontSize: 18)),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.purple,
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
  FaceDetectorIsolate? _faceDetectorIsolate;
  Uint8List? _imageBytes;
  List<Face> _faces = [];
  Size? _originalSize;

  bool _isLoading = false;
  bool _showBoundingBoxes = true;
  bool _showMesh = true;
  bool _showLandmarks = true;
  bool _showIrises = true;
  bool _showEyeContours = true;
  bool _showEyeMesh = true;
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
  Color _eyeContourColor = const Color(0xFF22AAFF);
  Color _eyeMeshColor = const Color(0xFFFFAA22);

  double _boundingBoxThickness = 2.0;
  double _landmarkSize = 3.0;
  double _meshSize = 1.25;
  double _eyeMeshSize = 0.8;

  FaceDetectionModel _detectionModel =
      !kIsWeb && (Platform.isMacOS || Platform.isWindows || Platform.isLinux)
          ? FaceDetectionModel.backCamera
          : FaceDetectionModel.frontCamera;

  @override
  void initState() {
    super.initState();
    _initFaceDetector();
  }

  Future<void> _initFaceDetector() async {
    try {
      _faceDetectorIsolate?.dispose();
      _faceDetectorIsolate = await FaceDetectorIsolate.spawn(
        model: _detectionModel,
      );
    } catch (_) {}
    setState(() {});
  }

  @override
  void dispose() {
    _faceDetectorIsolate?.dispose();
    super.dispose();
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

    if (_faceDetectorIsolate == null || !_faceDetectorIsolate!.isReady) {
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
    final List<Face> faces =
        await _faceDetectorIsolate!.detectFaces(bytes, mode: mode);
    final DateTime detectionEnd = DateTime.now();

    // Get image size from Face or decode it
    Size decodedSize;
    if (faces.isNotEmpty) {
      decodedSize = faces.first.originalSize;
    } else {
      // Decode image to get dimensions when no faces detected
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      decodedSize =
          Size(frame.image.width.toDouble(), frame.image.height.toDouble());
      frame.image.dispose();
    }

    if (!mounted) return;

    final DateTime totalEnd = DateTime.now();
    final int totalTime = totalEnd.difference(totalStart).inMilliseconds;
    final int detectionTime =
        detectionEnd.difference(detectionStart).inMilliseconds;

    int? meshTime;
    int? irisTime;
    if (_showMesh || _showIrises || _showEyeContours || _showEyeMesh) {
      final int extraTime = totalTime - detectionTime;
      if (_showMesh && (_showIrises || _showEyeContours || _showEyeMesh)) {
        meshTime = (extraTime * 0.6).round();
        irisTime = (extraTime * 0.4).round();
      } else if (_showMesh) {
        meshTime = extraTime;
      } else if (_showIrises || _showEyeContours || _showEyeMesh) {
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
    if (_showIrises || _showEyeContours || _showEyeMesh) {
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
    } else if (feature == 'eyeContour') {
      setState(() => _showEyeContours = newValue);
    } else if (feature == 'eyeMesh') {
      setState(() => _showEyeMesh = newValue);
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
    final perf = _performanceLevel(_totalTimeMs!);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: perf.color.withAlpha(26),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: perf.color.withAlpha(77)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(perf.icon, size: 16, color: perf.color),
          const SizedBox(width: 6),
          Text(
            perf.label,
            style: TextStyle(
              color: perf.color,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
        ],
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

  void _showSettingsSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.3,
        maxChildSize: 0.9,
        builder: (context, scrollController) => Container(
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
          ),
          child: Column(
            children: [
              // Drag handle
              Container(
                margin: const EdgeInsets.symmetric(vertical: 8),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Expanded(
                child: ListView(
                  controller: scrollController,
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  children: [
                    // Display Options
                    ExpansionTile(
                      title: const Text('Display Options',
                          style: TextStyle(fontWeight: FontWeight.bold)),
                      initiallyExpanded: true,
                      children: [
                        Wrap(
                          spacing: 8,
                          runSpacing: 4,
                          children: [
                            _buildCheckbox(
                                'Bounding Boxes',
                                _showBoundingBoxes,
                                (value) => setState(
                                    () => _showBoundingBoxes = value ?? false)),
                            _buildCheckbox(
                                'Mesh',
                                _showMesh,
                                (value) =>
                                    _onFeatureToggle('mesh', value ?? false)),
                            _buildCheckbox(
                                'Landmarks',
                                _showLandmarks,
                                (value) => setState(
                                    () => _showLandmarks = value ?? false)),
                            _buildCheckbox(
                                'Irises',
                                _showIrises,
                                (value) =>
                                    _onFeatureToggle('iris', value ?? false)),
                            _buildCheckbox(
                                'Eye Contour',
                                _showEyeContours,
                                (value) => _onFeatureToggle(
                                    'eyeContour', value ?? false)),
                            _buildCheckbox(
                                'Eye Mesh',
                                _showEyeMesh,
                                (value) => _onFeatureToggle(
                                    'eyeMesh', value ?? false)),
                          ],
                        ),
                        const SizedBox(height: 8),
                      ],
                    ),
                    // Colors
                    ExpansionTile(
                      title: const Text('Colors',
                          style: TextStyle(fontWeight: FontWeight.bold)),
                      children: [
                        Wrap(
                          spacing: 6,
                          runSpacing: 6,
                          children: [
                            _buildColorButton(
                                'BBox',
                                _boundingBoxColor,
                                (color) =>
                                    setState(() => _boundingBoxColor = color)),
                            _buildColorButton(
                                'Landmarks',
                                _landmarkColor,
                                (color) =>
                                    setState(() => _landmarkColor = color)),
                            _buildColorButton('Mesh', _meshColor,
                                (color) => setState(() => _meshColor = color)),
                            _buildColorButton('Irises', _irisColor,
                                (color) => setState(() => _irisColor = color)),
                            _buildColorButton(
                                'Eye Contour',
                                _eyeContourColor,
                                (color) =>
                                    setState(() => _eyeContourColor = color)),
                            _buildColorButton(
                                'Eye Mesh',
                                _eyeMeshColor,
                                (color) =>
                                    setState(() => _eyeMeshColor = color)),
                          ],
                        ),
                        const SizedBox(height: 8),
                      ],
                    ),
                    // Sizes
                    ExpansionTile(
                      title: const Text('Sizes',
                          style: TextStyle(fontWeight: FontWeight.bold)),
                      children: [
                        _buildCompactSlider(
                            'BBox',
                            _boundingBoxThickness,
                            0.5,
                            10.0,
                            (value) =>
                                setState(() => _boundingBoxThickness = value)),
                        _buildCompactSlider(
                            'Landmark',
                            _landmarkSize,
                            0.5,
                            15.0,
                            (value) => setState(() => _landmarkSize = value)),
                        _buildCompactSlider('Mesh', _meshSize, 0.1, 10.0,
                            (value) => setState(() => _meshSize = value)),
                        _buildCompactSlider('Eye Mesh', _eyeMeshSize, 0.1, 10.0,
                            (value) => setState(() => _eyeMeshSize = value)),
                        const SizedBox(height: 8),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCompactSlider(String label, double value, double min, double max,
      ValueChanged<double> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2.0),
      child: Row(
        children: [
          SizedBox(
            width: 70,
            child: Text(label, style: const TextStyle(fontSize: 12)),
          ),
          Expanded(
            child: SliderTheme(
              data: SliderTheme.of(context).copyWith(
                trackHeight: 2.0,
                thumbShape:
                    const RoundSliderThumbShape(enabledThumbRadius: 6.0),
                overlayShape:
                    const RoundSliderOverlayShape(overlayRadius: 12.0),
              ),
              child: Slider(
                value: value,
                min: min,
                max: max,
                divisions: ((max - min) * 10).round(),
                label: value.toStringAsFixed(1),
                onChanged: onChanged,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCompactPerformanceBadge() {
    if (_totalTimeMs == null) return const SizedBox.shrink();
    final perf = _performanceLevel(_totalTimeMs!);
    return Positioned(
      top: 12,
      left: 12,
      child: GestureDetector(
        onTap: _showTimingDetails,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: Colors.black.withAlpha(179),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(perf.icon, size: 14, color: perf.color),
              const SizedBox(width: 6),
              Text(
                '${_totalTimeMs}ms',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
              const SizedBox(width: 4),
              Text(
                perf.label,
                style: TextStyle(color: perf.color, fontSize: 12),
              ),
              const SizedBox(width: 4),
              const Icon(Icons.info_outline, size: 12, color: Colors.white54),
            ],
          ),
        ),
      ),
    );
  }

  void _showTimingDetails() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.timer, color: Colors.blue),
            const SizedBox(width: 8),
            const Text('Processing Details'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildStatusRow('Detection', true, Colors.green),
            _buildStatusRow('Mesh', _hasProcessedMesh,
                _showMesh ? Colors.green : Colors.grey),
            _buildStatusRow('Iris', _hasProcessedIris,
                _showIrises ? Colors.green : Colors.grey),
            const Divider(height: 16),
            if (_detectionTimeMs != null)
              _buildTimingRow('Detection', _detectionTimeMs!, Colors.green),
            if (_meshTimeMs != null)
              _buildTimingRow('Mesh Refinement', _meshTimeMs!, Colors.pink),
            if (_irisTimeMs != null)
              _buildTimingRow(
                  'Iris Refinement', _irisTimeMs!, Colors.blueAccent),
            if (_totalTimeMs != null)
              _buildTimingRow('Total', _totalTimeMs!, Colors.blue,
                  isBold: true),
            const SizedBox(height: 12),
            _buildPerformanceIndicator(),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bool hasImage = _imageBytes != null && _originalSize != null;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Still Image Detection'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        actions: [
          // Pick Image button
          IconButton(
            onPressed: _pickAndRun,
            icon: const Icon(Icons.add_photo_alternate),
            tooltip: 'Pick Image',
          ),
          // Model dropdown
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0),
            child: DropdownButton<FaceDetectionModel>(
              value: _detectionModel,
              dropdownColor: Colors.blue[800],
              style: const TextStyle(color: Colors.white, fontSize: 14),
              underline: const SizedBox(),
              icon: const Icon(Icons.arrow_drop_down, color: Colors.white),
              items: const [
                DropdownMenuItem(
                  value: FaceDetectionModel.frontCamera,
                  child: Text('Front'),
                ),
                DropdownMenuItem(
                  value: FaceDetectionModel.backCamera,
                  child: Text('Back'),
                ),
                DropdownMenuItem(
                  value: FaceDetectionModel.shortRange,
                  child: Text('Short'),
                ),
                DropdownMenuItem(
                  value: FaceDetectionModel.full,
                  child: Text('Full'),
                ),
                DropdownMenuItem(
                  value: FaceDetectionModel.fullSparse,
                  child: Text('Sparse'),
                ),
              ],
              onChanged: (value) async {
                if (value != null && value != _detectionModel) {
                  setState(() => _detectionModel = value);
                  await _initFaceDetector();
                  if (_imageBytes != null) {
                    await _processImage(_imageBytes!);
                  }
                }
              },
            ),
          ),
          // Settings button
          IconButton(
            onPressed: _showSettingsSheet,
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: Stack(
        children: [
          // Main image display area - takes full space
          Center(
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
                            Size(constraints.maxWidth, constraints.maxHeight),
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
                              size: Size(imageRect.width, imageRect.height),
                              painter: _DetectionsPainter(
                                faces: _faces,
                                imageRectOnCanvas: Rect.fromLTWH(
                                    0, 0, imageRect.width, imageRect.height),
                                originalImageSize: _originalSize!,
                                showBoundingBoxes: _showBoundingBoxes,
                                showMesh: _showMesh,
                                showLandmarks: _showLandmarks,
                                showIrises: _showIrises,
                                showEyeContours: _showEyeContours,
                                showEyeMesh: _showEyeMesh,
                                boundingBoxColor: _boundingBoxColor,
                                landmarkColor: _landmarkColor,
                                meshColor: _meshColor,
                                irisColor: _irisColor,
                                eyeContourColor: _eyeContourColor,
                                eyeMeshColor: _eyeMeshColor,
                                boundingBoxThickness: _boundingBoxThickness,
                                landmarkSize: _landmarkSize,
                                meshSize: _meshSize,
                                eyeMeshSize: _eyeMeshSize,
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
                      Icon(Icons.add_photo_alternate,
                          size: 80, color: Colors.grey[300]),
                      const SizedBox(height: 16),
                      Text(
                        'No image selected',
                        style: TextStyle(fontSize: 18, color: Colors.grey[600]),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Tap the + icon to pick an image',
                        style: TextStyle(fontSize: 14, color: Colors.grey[500]),
                      ),
                    ],
                  ),
          ),
          // Floating performance badge
          if (hasImage) _buildCompactPerformanceBadge(),
          // Loading overlay
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
          SizedBox(
            width: 24,
            height: 24,
            child: Checkbox(
              value: value,
              onChanged: onChanged,
              materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
              visualDensity: VisualDensity.compact,
            ),
          ),
          const SizedBox(width: 4),
          Text(label, style: const TextStyle(fontSize: 12)),
        ],
      ),
    );
  }

  Widget _buildColorButton(
      String label, Color color, ValueChanged<Color> onColorChanged) {
    return InkWell(
      onTap: () => _pickColor(label, color, onColorChanged),
      borderRadius: BorderRadius.circular(6),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 18,
              height: 18,
              decoration: BoxDecoration(
                color: color,
                border: Border.all(color: Colors.grey.shade400),
                borderRadius: BorderRadius.circular(3),
              ),
            ),
            const SizedBox(width: 6),
            Text(label, style: const TextStyle(fontSize: 12)),
            const SizedBox(width: 2),
            const Icon(Icons.arrow_drop_down, size: 16),
          ],
        ),
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

  _DetectionsPainter({
    required this.faces,
    required this.imageRectOnCanvas,
    required this.originalImageSize,
    required this.showBoundingBoxes,
    required this.showMesh,
    required this.showLandmarks,
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
        // Iterate over all landmarks using .values
        // You can also access specific landmarks using named properties:
        // face.landmarks.leftEye, face.landmarks.rightEye, etc.
        for (final p in face.landmarks.values) {
          canvas.drawCircle(
            Offset(ox + p.x * scaleX, oy + p.y * scaleY),
            landmarkSize,
            detKpPaint,
          );
        }
      }

      if (showMesh) {
        final FaceMesh? faceMesh = face.mesh;
        if (faceMesh != null) {
          final mesh = faceMesh.points;
          final double imgArea =
              imageRectOnCanvas.width * imageRectOnCanvas.height;
          final double radius = meshSize + sqrt(imgArea) / 1000.0;

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

            // Draw iris (center + contour as oval)
            if (showIrises) {
              // Build bounding box from iris center + iris contour points
              final allIrisPoints = [iris.irisCenter, ...iris.irisContour];
              double minX = allIrisPoints.first.x, maxX = allIrisPoints.first.x;
              double minY = allIrisPoints.first.y, maxY = allIrisPoints.first.y;
              for (final p in allIrisPoints) {
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

            // Draw eye contour landmarks
            if (showEyeContours && iris.mesh.isNotEmpty) {
              // Draw the visible eyeball contour (eyelid outline) as connected lines
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

            // Draw all 71 eye mesh points as small dots
            // (includes eyebrows and tracking halos)
            if (showEyeMesh && iris.mesh.isNotEmpty) {
              final Paint eyeMeshPointPaint = Paint()
                ..color = eyeMeshColor
                ..style = PaintingStyle.fill;

              for (final p in iris.mesh) {
                final canvasX = ox + p.x * scaleX;
                final canvasY = oy + p.y * scaleY;
                canvas.drawCircle(
                    Offset(canvasX, canvasY), eyeMeshSize, eyeMeshPointPaint);
              }
            }
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

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  bool get _isMacOS => !kIsWeb && Platform.isMacOS;
  bool get _isWindows => !kIsWeb && Platform.isWindows;

  CameraController? _cameraController;
  CameraMacOSController? _macCameraController;
  Size? _macPreviewSize;
  FaceDetectorIsolate? _faceDetectorIsolate;
  List<Face> _faces = [];
  Size? _imageSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;
  bool _isProcessing = false;
  bool _isInitialized = false;
  int _frameCounter = 0;
  int _processEveryNFrames = 1; // Process every frame
  int _detectionTimeMs = 0;
  int _fps = 0;
  DateTime? _lastFpsUpdate;
  int _framesSinceLastUpdate = 0;
  bool _isImageStreamStarted = false;
  Timer? _windowsFrameTimer;
  bool _isWindowsFrameCaptureInProgress = false;
  int _windowsLoopIntervalMs = 180;

  // Detection settings
  FaceDetectionMode _detectionMode = FaceDetectionMode.full;
  FaceDetectionModel _detectionModel =
      !kIsWeb && (Platform.isMacOS || Platform.isWindows || Platform.isLinux)
          ? FaceDetectionModel.backCamera
          : FaceDetectionModel.frontCamera;

  // Segmentation settings
  bool _showSegmentation = false;
  SegmentationMask? _segmentationMask;
  final Color _segmentationColor = const Color(0x8800FF00);
  SegmentationModel _liveSegmentationModel = SegmentationModel.general;

  // Virtual background settings
  bool _showVirtualBackground = false;
  ui.Image? _beachBackground;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadBeachBackground();
  }

  Future<void> _loadBeachBackground() async {
    final data = await rootBundle.load('assets/beach_background.jpg');
    final codec = await ui.instantiateImageCodec(data.buffer.asUint8List());
    final frame = await codec.getNextFrame();
    if (mounted) {
      setState(() {
        _beachBackground = frame.image;
      });
    }
  }

  void _updateFps() {
    _framesSinceLastUpdate++;
    final now = DateTime.now();
    if (_lastFpsUpdate != null) {
      final diff = now.difference(_lastFpsUpdate!).inMilliseconds;
      if (diff >= 1000 && mounted) {
        setState(() {
          _fps = (_framesSinceLastUpdate * 1000 / diff).round();
          _framesSinceLastUpdate = 0;
          _lastFpsUpdate = now;
        });
      }
    } else {
      _lastFpsUpdate = now;
    }
  }

  Future<void> _reinitDetectorIsolate() async {
    final old = _faceDetectorIsolate;
    _faceDetectorIsolate = null;
    old?.dispose();
    _faceDetectorIsolate = await FaceDetectorIsolate.spawn(
      model: _detectionModel,
      withSegmentation: true,
      segmentationConfig: SegmentationConfig(model: _liveSegmentationModel),
    );
  }

  Future<({List<Face> faces, SegmentationMask? segMask})> _detectFromMat(
      cv.Mat mat) async {
    if ((_showSegmentation || _showVirtualBackground) &&
        _faceDetectorIsolate!.isSegmentationReady) {
      final result = await _faceDetectorIsolate!
          .detectFacesWithSegmentationFromMat(mat, mode: _detectionMode);
      return (faces: result.faces, segMask: result.segmentationMask);
    } else {
      final faces = await _faceDetectorIsolate!.detectFacesFromMat(
        mat,
        mode: _detectionMode,
      );
      return (faces: faces, segMask: null);
    }
  }

  Future<void> _switchLiveSegmentationModel(SegmentationModel model) async {
    if (model == _liveSegmentationModel) return;
    setState(() {
      _liveSegmentationModel = model;
      _segmentationMask = null;
    });
    await _reinitDetectorIsolate();
  }

  Widget _segModelButton(SegmentationModel model, String label) {
    final isSelected = _liveSegmentationModel == model;
    return GestureDetector(
      onTap: () => _switchLiveSegmentationModel(model),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: isSelected ? Colors.purple : Colors.white24,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.white70,
            fontSize: 12,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }

  AppBar _buildCameraAppBar() {
    return AppBar(
      title: const Text('Live Camera Detection'),
      backgroundColor: Colors.green,
      foregroundColor: Colors.white,
      actions: [
        Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8.0),
            child: DropdownButton<FaceDetectionMode>(
              value: _detectionMode,
              dropdownColor: Colors.green[800],
              style: const TextStyle(color: Colors.white, fontSize: 14),
              underline: const SizedBox(),
              items: _kDetectionModeItems,
              onChanged: (value) {
                if (value != null) {
                  setState(() => _detectionMode = value);
                }
              },
            ),
          ),
        ),
        Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8.0),
            child: DropdownButton<FaceDetectionModel>(
              value: _detectionModel,
              dropdownColor: Colors.green[800],
              style: const TextStyle(color: Colors.white, fontSize: 14),
              underline: const SizedBox(),
              items: _kDetectionModelItems,
              onChanged: (value) async {
                if (value != null && value != _detectionModel) {
                  setState(() => _detectionModel = value);
                  await _reinitDetectorIsolate();
                }
              },
            ),
          ),
        ),
        Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Text(
              'FPS: $_fps | ${_detectionTimeMs}ms',
              style: const TextStyle(fontSize: 14),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCameraOverlays({
    required Widget cameraWidget,
    required double cameraAspectRatio,
    required double displayAspectRatio,
    required bool mirrorHorizontally,
    required int sensorOrientation,
    required Orientation deviceOrientation,
    required bool isFrontCamera,
  }) {
    return Center(
      child: AspectRatio(
        aspectRatio: displayAspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (_showVirtualBackground && _beachBackground != null)
              Positioned.fill(
                child: CustomPaint(
                  painter: _BackgroundImagePainter(image: _beachBackground!),
                ),
              ),
            cameraWidget,
            if (_showVirtualBackground &&
                _beachBackground != null &&
                _segmentationMask != null)
              CustomPaint(
                painter: _VirtualBackgroundOverlayPainter(
                  background: _beachBackground!,
                  mask: _segmentationMask!,
                  mirrorHorizontally: mirrorHorizontally,
                ),
              ),
            if (_showSegmentation &&
                !_showVirtualBackground &&
                _segmentationMask != null)
              CustomPaint(
                painter: _LiveSegmentationPainter(
                  mask: _segmentationMask!,
                  maskColor: _segmentationColor,
                  showAllClasses:
                      _liveSegmentationModel == SegmentationModel.multiclass,
                  mirrorHorizontally: mirrorHorizontally,
                ),
              ),
            if (_imageSize != null)
              CustomPaint(
                painter: _CameraDetectionPainter(
                  faces: _faces,
                  imageSize: _imageSize!,
                  cameraAspectRatio: cameraAspectRatio,
                  displayAspectRatio: displayAspectRatio,
                  detectionMode: _detectionMode,
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

  Widget _buildInfoPanel() {
    return Positioned(
      bottom: 20,
      left: 0,
      right: 0,
      child: Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
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
                  const SizedBox(width: 16),
                  const Text(
                    'Segmentation: ',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                  Switch(
                    value: _showSegmentation,
                    activeTrackColor: Colors.green,
                    onChanged: (value) {
                      setState(() {
                        _showSegmentation = value;
                        if (!value) _segmentationMask = null;
                      });
                    },
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    'Seg Model: ',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                  const SizedBox(width: 8),
                  _segModelButton(SegmentationModel.general, 'Binary'),
                  const SizedBox(width: 4),
                  _segModelButton(SegmentationModel.multiclass, '6-Class'),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    'Virtual Background: ',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                  Switch(
                    value: _showVirtualBackground,
                    activeTrackColor: Colors.blue,
                    onChanged: (value) {
                      setState(() {
                        _showVirtualBackground = value;
                        if (!value) _segmentationMask = null;
                      });
                    },
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _initCamera() async {
    try {
      await _reinitDetectorIsolate();

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

      // Prefer the front camera on mobile; fall back to the first camera
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

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
        _sensorOrientation = _cameraController!.description.sensorOrientation;
        _isFrontCamera = _cameraController!.description.lensDirection ==
            CameraLensDirection.front;
      });

      if (_isWindows) {
        _startWindowsDetectionLoop();
        return;
      }

      // Start image stream on platforms that support it.
      await _cameraController!.startImageStream(_processCameraImage);
      _isImageStreamStarted = true;
    } catch (e, st) {
      debugPrint('Camera init failed: $e');
      debugPrint('$st');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error initializing camera: $e')),
        );
      }
    }
  }

  void _startWindowsDetectionLoop() {
    _windowsFrameTimer?.cancel();
    _lastFpsUpdate = DateTime.now();
    _windowsFrameTimer =
        Timer.periodic(Duration(milliseconds: _windowsLoopIntervalMs), (_) {
      _captureAndProcessWindowsFrame();
    });
  }

  void _retuneWindowsDetectionLoop({
    required bool segmentationEnabled,
    required int processingTimeMs,
  }) {
    // Keep camera cadence adaptive:
    // - Face-only: tighter interval for responsiveness.
    // - Face+segmentation: slower interval to avoid CPU saturation/jank.
    final int minInterval = segmentationEnabled ? 280 : 120;
    final int maxInterval = segmentationEnabled ? 650 : 320;
    final int safetyMargin = segmentationEnabled ? 90 : 40;

    final int nextInterval =
        (processingTimeMs + safetyMargin).clamp(minInterval, maxInterval);

    // Avoid timer churn from tiny fluctuations.
    if ((nextInterval - _windowsLoopIntervalMs).abs() < 40) return;

    _windowsLoopIntervalMs = nextInterval;
    _startWindowsDetectionLoop();
  }

  Future<void> _captureAndProcessWindowsFrame() async {
    if (!_isWindows ||
        !mounted ||
        _cameraController == null ||
        !_cameraController!.value.isInitialized ||
        _isWindowsFrameCaptureInProgress) {
      return;
    }

    _isWindowsFrameCaptureInProgress = true;
    try {
      final XFile frame = await _cameraController!.takePicture();
      final Uint8List bytes = await frame.readAsBytes();
      try {
        final file = File(frame.path);
        if (await file.exists()) {
          await file.delete();
        }
      } catch (_) {}

      await _processWindowsImageBytes(bytes);
    } catch (_) {
      // Ignore intermittent capture errors on Windows.
    } finally {
      _isWindowsFrameCaptureInProgress = false;
    }
  }

  Future<void> _processWindowsImageBytes(Uint8List bytes) async {
    _frameCounter++;
    _updateFps();

    if (_frameCounter % _processEveryNFrames != 0) return;
    if (_isProcessing || _faceDetectorIsolate == null) return;

    _isProcessing = true;
    try {
      final startTime = DateTime.now();
      final bool segmentationEnabled =
          (_showSegmentation || _showVirtualBackground) &&
              _faceDetectorIsolate!.isSegmentationReady;

      List<Face> faces;
      SegmentationMask? segMask;
      int processingTimeMs;

      if (segmentationEnabled) {
        final result = await _faceDetectorIsolate!.detectFacesWithSegmentation(
          bytes,
          mode: _detectionMode,
        );
        faces = result.faces;
        segMask = result.segmentationMask;
        processingTimeMs = result.totalTimeMs;
      } else {
        faces = await _faceDetectorIsolate!.detectFaces(
          bytes,
          mode: _detectionMode,
        );
        segMask = null;
        processingTimeMs = DateTime.now().difference(startTime).inMilliseconds;
      }

      final Size imageSize = faces.isNotEmpty
          ? faces.first.originalSize
          : await _decodeImageSize(bytes);

      if (mounted) {
        setState(() {
          _faces = faces;
          _imageSize = imageSize;
          _detectionTimeMs = processingTimeMs;
          _segmentationMask = segMask;
        });
      }

      _retuneWindowsDetectionLoop(
        segmentationEnabled: segmentationEnabled,
        processingTimeMs: processingTimeMs,
      );
    } catch (_) {
      // Ignore frame-level processing errors.
    } finally {
      _isProcessing = false;
    }
  }

  Future<Size> _decodeImageSize(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final size =
        Size(frame.image.width.toDouble(), frame.image.height.toDouble());
    frame.image.dispose();
    return size;
  }

  DeviceOrientation _effectiveDeviceOrientation(BuildContext context) {
    final controller = _cameraController;
    if (controller != null) {
      return controller.value.deviceOrientation;
    }

    return MediaQuery.of(context).orientation == Orientation.portrait
        ? DeviceOrientation.portraitUp
        : DeviceOrientation.landscapeLeft;
  }

  int? _rotationFlagForFrame({
    required int width,
    required int height,
  }) {
    final DeviceOrientation orientation = _effectiveDeviceOrientation(context);
    final bool isPortrait = orientation == DeviceOrientation.portraitUp ||
        orientation == DeviceOrientation.portraitDown;

    if (!isPortrait) return null;

    // If the incoming buffer is already portrait, don't rotate it.
    if (height >= width) return null;

    final int? sensor = _sensorOrientation;
    if (sensor == 90) {
      return cv.ROTATE_90_COUNTERCLOCKWISE;
    }
    if (sensor == 270) {
      return cv.ROTATE_90_CLOCKWISE;
    }

    return null;
  }

  Future<void> _processCameraImage(CameraImage image) async {
    _frameCounter++;
    _updateFps();

    // Skip frames for better performance
    if (_frameCounter % _processEveryNFrames != 0) return;

    // Skip if already processing
    if (_isProcessing) return;

    _isProcessing = true;

    try {
      final startTime = DateTime.now();

      // Convert CameraImage to cv.Mat for OpenCV-accelerated processing
      final mat = await _convertCameraImageToMat(image);

      if (mat == null || _faceDetectorIsolate == null) {
        _isProcessing = false;
        return;
      }

      final result = await _detectFromMat(mat);
      mat.dispose();
      final faces = result.faces;
      final segMask = result.segMask;

      final endTime = DateTime.now();
      final detectionTime = endTime.difference(startTime).inMilliseconds;

      if (mounted) {
        // Image size is the size after rotation (if any)
        final rotationFlag = _rotationFlagForFrame(
          width: image.width,
          height: image.height,
        );
        final bool isRotated = rotationFlag != null;
        // When rotated 90°, width and height swap
        final Size processedSize = isRotated
            ? Size(image.height.toDouble(), image.width.toDouble())
            : Size(image.width.toDouble(), image.height.toDouble());

        setState(() {
          _faces = faces;
          _imageSize = processedSize;
          _detectionTimeMs = detectionTime;
          _segmentationMask = segMask;
        });
      }
    } catch (e) {
      // Silently handle errors during processing
    } finally {
      _isProcessing = false;
    }
  }

  /// Converts CameraImage (YUV420) to cv.Mat (BGR) for OpenCV processing.
  ///
  /// This avoids the JPEG encode/decode overhead by creating cv.Mat directly.
  Future<cv.Mat?> _convertCameraImageToMat(CameraImage image) async {
    try {
      final int width = image.width;
      final int height = image.height;
      final int yRowStride = image.planes[0].bytesPerRow;
      final int yPixelStride = image.planes[0].bytesPerPixel ?? 1;

      // Allocate BGR buffer for OpenCV (3 bytes per pixel)
      final bgrBytes = Uint8List(width * height * 3);

      void writePixel(int x, int y, int yp, int up, int vp) {
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        final int bgrIdx = (y * width + x) * 3;
        bgrBytes[bgrIdx] = b;
        bgrBytes[bgrIdx + 1] = g;
        bgrBytes[bgrIdx + 2] = r;
      }

      if (image.planes.length == 2) {
        // iOS NV12 format
        final int uvRowStride = image.planes[1].bytesPerRow;
        final int uvPixelStride = image.planes[1].bytesPerPixel ?? 2;

        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final int uvIndex =
                uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
            final int index = y * yRowStride + x * yPixelStride;
            writePixel(
                x,
                y,
                image.planes[0].bytes[index],
                image.planes[1].bytes[uvIndex],
                image.planes[1].bytes[uvIndex + 1]);
          }
        }
      } else if (image.planes.length >= 3) {
        // Android I420 format
        final int uvRowStride = image.planes[1].bytesPerRow;
        final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final int uvIndex =
                uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
            final int index = y * yRowStride + x * yPixelStride;
            writePixel(x, y, image.planes[0].bytes[index],
                image.planes[1].bytes[uvIndex], image.planes[2].bytes[uvIndex]);
          }
        }
      } else {
        return null;
      }

      // Create cv.Mat from BGR bytes
      cv.Mat mat = cv.Mat.fromList(height, width, cv.MatType.CV_8UC3, bgrBytes);

      // Rotate image for portrait mode so face detector sees upright faces.
      final rotationFlag = _rotationFlagForFrame(width: width, height: height);
      if (rotationFlag != null) {
        final rotated = cv.rotate(mat, rotationFlag);
        mat.dispose();
        return rotated;
      }

      return mat;
    } catch (e) {
      return null;
    }
  }

  /// Converts macOS CameraImageData (ARGB) to cv.Mat (BGR) for OpenCV processing.
  cv.Mat? _convertMacImageToMat(CameraImageData image) {
    try {
      final bytes = image.bytes;
      final stride = image.bytesPerRow;
      final width = image.width;
      final height = image.height;

      // Allocate BGR buffer for OpenCV (3 bytes per pixel)
      final bgrBytes = Uint8List(width * height * 3);

      for (int y = 0; y < height; y++) {
        final rowStart = y * stride;
        for (int x = 0; x < width; x++) {
          final pixelStart = rowStart + x * 4;
          if (pixelStart + 3 >= bytes.length) break;

          // macOS uses ARGB format
          final r = bytes[pixelStart + 1];
          final g = bytes[pixelStart + 2];
          final b = bytes[pixelStart + 3];

          // Write BGR (OpenCV format)
          final int bgrIdx = (y * width + x) * 3;
          bgrBytes[bgrIdx] = b;
          bgrBytes[bgrIdx + 1] = g;
          bgrBytes[bgrIdx + 2] = r;
        }
      }

      return cv.Mat.fromList(height, width, cv.MatType.CV_8UC3, bgrBytes);
    } catch (_) {
      return null;
    }
  }

  @override
  void dispose() {
    _windowsFrameTimer?.cancel();
    if (_isMacOS) {
      _macCameraController?.stopImageStream();
      _macCameraController?.destroy();
    } else {
      if (_isImageStreamStarted) {
        _cameraController?.stopImageStream();
      }
      _cameraController?.dispose();
    }
    _faceDetectorIsolate?.dispose();
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
    final deviceOrientation = MediaQuery.of(context).orientation;
    final effectiveOrientation = _effectiveDeviceOrientation(context);
    final bool isPortrait =
        effectiveOrientation == DeviceOrientation.portraitUp ||
            effectiveOrientation == DeviceOrientation.portraitDown;

    final double displayAspectRatio =
        isPortrait ? 1.0 / cameraAspectRatio : cameraAspectRatio;

    return Scaffold(
      appBar: _buildCameraAppBar(),
      body: Stack(
        fit: StackFit.expand,
        children: [
          _buildCameraOverlays(
            cameraWidget: CameraPreview(_cameraController!),
            cameraAspectRatio: cameraAspectRatio,
            displayAspectRatio: displayAspectRatio,
            mirrorHorizontally: _isWindows && _isFrontCamera,
            sensorOrientation: _sensorOrientation ?? 0,
            deviceOrientation: deviceOrientation,
            isFrontCamera: _isFrontCamera,
          ),
          _buildInfoPanel(),
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
      appBar: _buildCameraAppBar(),
      body: Stack(
        fit: StackFit.expand,
        children: [
          _buildCameraOverlays(
            cameraWidget: CameraMacOSView(
              cameraMode: CameraMacOSMode.photo,
              fit: BoxFit.contain,
              onCameraInizialized: _onMacCameraInitialized,
              onCameraLoading: (_) =>
                  const Center(child: CircularProgressIndicator()),
            ),
            cameraAspectRatio: cameraAspectRatio,
            displayAspectRatio: cameraAspectRatio,
            mirrorHorizontally: false,
            sensorOrientation: 0,
            deviceOrientation: Orientation.landscape,
            isFrontCamera: true,
          ),
          _buildInfoPanel(),
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
      _updateFps();

      if (_frameCounter % _processEveryNFrames != 0) return;
      if (_isProcessing) return;

      _isProcessing = true;

      try {
        final startTime = DateTime.now();
        // Use OpenCV-based processing for better performance
        final mat = _convertMacImageToMat(image);
        if (mat == null || _faceDetectorIsolate == null) {
          _isProcessing = false;
          return;
        }

        final result = await _detectFromMat(mat);
        mat.dispose();
        final faces = result.faces;
        final segMask = result.segMask;

        final detectionTime =
            DateTime.now().difference(startTime).inMilliseconds;

        if (mounted) {
          setState(() {
            _faces = faces;
            _imageSize = Size(image.width.toDouble(), image.height.toDouble());
            _macPreviewSize ??=
                Size(image.width.toDouble(), image.height.toDouble());
            _detectionTimeMs = detectionTime;
            _segmentationMask = segMask;
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
  final double displayAspectRatio;
  final FaceDetectionMode detectionMode;
  final int sensorOrientation;
  final Orientation deviceOrientation;
  final bool isFrontCamera;
  final bool mirrorHorizontally;

  _CameraDetectionPainter({
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

    // The canvas fills the AspectRatio widget, so size IS the display area
    final double displayWidth = size.width;
    final double displayHeight = size.height;

    // Detection was done on raw camera frame (imageSize.width x imageSize.height)
    // CameraPreview rotates the display based on sensor orientation
    // We need to transform detection coords to match CameraPreview's display

    final double sourceWidth = imageSize.width;
    final double sourceHeight = imageSize.height;

    // Match CameraPreview's cover behavior to avoid stretched/squashed overlays.
    final double sourceAspectRatio = sourceWidth / sourceHeight;
    final double viewportAspectRatio = displayWidth / displayHeight;

    final double scale;
    double offsetX = 0;
    double offsetY = 0;

    if (sourceAspectRatio > viewportAspectRatio) {
      // Source is wider: fit height and crop left/right.
      scale = displayHeight / sourceHeight;
      offsetX = (displayWidth - sourceWidth * scale) / 2;
    } else {
      // Source is taller: fit width and crop top/bottom.
      scale = displayWidth / sourceWidth;
      offsetY = (displayHeight - sourceHeight * scale) / 2;
    }

    // Transform a detection coordinate to canvas coordinate
    Offset transformPoint(double x, double y) {
      if (mirrorHorizontally) {
        x = sourceWidth - x;
      }
      return Offset(x * scale + offsetX, y * scale + offsetY);
    }

    // Draw bounding boxes and features for each face
    for (final face in faces) {
      // Draw bounding box - need to handle that top-left might not be top-left after rotation
      final boundingBox = face.boundingBox;
      final p1 = transformPoint(boundingBox.topLeft.x, boundingBox.topLeft.y);
      final p2 =
          transformPoint(boundingBox.bottomRight.x, boundingBox.bottomRight.y);

      // After rotation, we need to find the actual min/max
      final rect = Rect.fromLTRB(
        min(p1.dx, p2.dx),
        min(p1.dy, p2.dy),
        max(p1.dx, p2.dx),
        max(p1.dy, p2.dy),
      );
      canvas.drawRect(rect, boxPaint);

      // Draw the 6 simple landmarks (available in all modes)
      for (final landmark in face.landmarks.values) {
        final transformed = transformPoint(landmark.x, landmark.y);
        canvas.drawCircle(transformed, 4.0, landmarkPaint);
      }

      // Draw mesh if in standard or full mode
      if (detectionMode == FaceDetectionMode.standard ||
          detectionMode == FaceDetectionMode.full) {
        final FaceMesh? faceMesh = face.mesh;
        if (faceMesh != null) {
          final mesh = faceMesh.points;
          final double imgArea = displayWidth * displayHeight;
          final double radius = 1.25 + sqrt(imgArea) / 1000.0;

          for (final p in mesh) {
            final transformed = transformPoint(p.x, p.y);
            canvas.drawCircle(transformed, radius, meshPaint);
          }
        }
      }

      // Draw iris and eye contours if in full mode
      if (detectionMode == FaceDetectionMode.full) {
        final eyePair = face.eyes;
        if (eyePair != null) {
          for (final iris in [eyePair.leftEye, eyePair.rightEye]) {
            if (iris == null) continue;

            // Draw iris (center + contour as oval)
            final allIrisPoints = [iris.irisCenter, ...iris.irisContour];
            final transformedIrisPoints =
                allIrisPoints.map((p) => transformPoint(p.x, p.y)).toList();

            double minX = transformedIrisPoints.first.dx,
                maxX = transformedIrisPoints.first.dx;
            double minY = transformedIrisPoints.first.dy,
                maxY = transformedIrisPoints.first.dy;
            for (final p in transformedIrisPoints) {
              if (p.dx < minX) minX = p.dx;
              if (p.dx > maxX) maxX = p.dx;
              if (p.dy < minY) minY = p.dy;
              if (p.dy > maxY) maxY = p.dy;
            }

            final cx = (minX + maxX) * 0.5;
            final cy = (minY + maxY) * 0.5;
            final rx = (maxX - minX) * 0.5;
            final ry = (maxY - minY) * 0.5;

            final oval = Rect.fromCenter(
                center: Offset(cx, cy), width: rx * 2, height: ry * 2);
            canvas.drawOval(oval, irisFill);
            canvas.drawOval(oval, irisStroke);

            // Draw eye contour landmarks
            if (iris.mesh.isNotEmpty) {
              // Draw the visible eyeball contour (eyelid outline) as connected lines
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

              // Draw all 71 eye mesh points as small dots
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
  bool shouldRepaint(covariant _CameraDetectionPainter old) {
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
class _LiveSegmentationPainter extends CustomPainter {
  final SegmentationMask mask;
  final Color maskColor;
  final bool showAllClasses;
  final bool mirrorHorizontally;

  _LiveSegmentationPainter({
    required this.mask,
    required this.maskColor,
    this.showAllClasses = false,
    this.mirrorHorizontally = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final v = _maskValidRegion(mask);
    final validW = v.x1 - v.x0;
    final validH = v.y1 - v.y0;

    final fit = _coverFitScaleOffset(validW, validH, size.width, size.height);
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
          final renderX =
              mirrorHorizontally ? size.width - rawX - pixelW : rawX;
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
            final color = _kSegmentationClassColors[winningClass];
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

      _drawClassLabels(canvas, labelCounts, labelSumX, labelSumY);
      return;
    }

    for (int y = v.y0; y < v.y1; y++) {
      for (int x = v.x0; x < v.x1; x++) {
        final prob = mask.at(x, y);
        final alpha = prob >= threshold ? maskColor.a : 0.0;

        if (alpha > 0.01) {
          paint.color = maskColor.withAlpha((alpha * 255).round());
          final rawX = (x - v.x0) * scale + offsetX;
          final renderX =
              mirrorHorizontally ? size.width - rawX - pixelW : rawX;
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
  bool shouldRepaint(covariant _LiveSegmentationPainter old) {
    return old.mask != mask ||
        old.maskColor != maskColor ||
        old.showAllClasses != showAllClasses ||
        old.mirrorHorizontally != mirrorHorizontally;
  }
}

/// Painter that draws a background image scaled to fill the canvas.
class _BackgroundImagePainter extends CustomPainter {
  final ui.Image image;

  _BackgroundImagePainter({required this.image});

  @override
  void paint(Canvas canvas, Size size) {
    final src =
        Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final dst = Rect.fromLTWH(0, 0, size.width, size.height);
    canvas.drawImageRect(image, src, dst, Paint());
  }

  @override
  bool shouldRepaint(covariant _BackgroundImagePainter old) {
    return old.image != image;
  }
}

/// Painter that draws background image only on non-person (background) areas.
/// This creates the "virtual background" effect by covering the camera's
/// background with the beach image while leaving the person visible.
/// Uses soft alpha blending at edges for smooth transitions.
class _VirtualBackgroundOverlayPainter extends CustomPainter {
  final ui.Image background;
  final SegmentationMask mask;
  final bool mirrorHorizontally;

  _VirtualBackgroundOverlayPainter({
    required this.background,
    required this.mask,
    this.mirrorHorizontally = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final v = _maskValidRegion(mask);
    final validW = v.x1 - v.x0;
    final validH = v.y1 - v.y0;

    if (validW <= 0 || validH <= 0) return;

    final fit = _coverFitScaleOffset(validW, validH, size.width, size.height);
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

        // Calculate background opacity (inverse of person probability)
        // Apply a slight contrast boost for cleaner edges
        final bgAlpha = (1.0 - prob);

        // Skip fully transparent pixels for performance
        if (bgAlpha < 0.01) continue;

        final rawX = (x - v.x0) * scale + offsetX;
        final renderX = mirrorHorizontally ? size.width - rawX - pixelW : rawX;
        final renderY = (y - v.y0) * scale + offsetY;

        // Sample from background image
        final bgX =
            (renderX * bgScaleX).clamp(0, background.width - 1).toDouble();
        final bgY =
            (renderY * bgScaleY).clamp(0, background.height - 1).toDouble();

        // Draw background with alpha based on inverse mask probability
        paint.color = Color.fromRGBO(255, 255, 255, bgAlpha);
        final src =
            Rect.fromLTWH(bgX, bgY, bgScaleX * pixelW, bgScaleY * pixelH);
        final dst = Rect.fromLTWH(renderX, renderY, pixelW, pixelH);
        canvas.drawImageRect(background, src, dst, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _VirtualBackgroundOverlayPainter old) {
    return old.background != background ||
        old.mask != mask ||
        old.mirrorHorizontally != mirrorHorizontally;
  }
}

// ============================================================================
// Selfie Segmentation Demo Screen
// ============================================================================

class SegmentationDemoScreen extends StatefulWidget {
  const SegmentationDemoScreen({super.key});

  @override
  State<SegmentationDemoScreen> createState() => _SegmentationDemoScreenState();
}

class _SegmentationDemoScreenState extends State<SegmentationDemoScreen> {
  SelfieSegmentation? _segmenter;
  Uint8List? _imageBytes;
  SegmentationMask? _mask;
  Size? _originalSize;
  bool _isLoading = false;
  bool _isInitializing = true;
  bool _isSwitchingModel = false;
  int? _inferenceTimeMs;
  String? _error;

  // Model selection
  SegmentationModel _selectedModel = SegmentationModel.general;

  // Display options
  double _threshold = 0.5;
  bool _showMaskOnly = false;
  bool _showBinaryMask = true;
  Color _maskColor = const Color(0x8800FF00);

  // Multiclass display - which class to show (null = combined person mask)
  int? _selectedClassIndex;

  @override
  void initState() {
    super.initState();
    _initSegmenter();
  }

  Future<void> _initSegmenter() async {
    setState(() {
      _isInitializing = true;
      _error = null;
    });

    try {
      await _segmenter?.disposeAsync();
      _segmenter = await SelfieSegmentation.create(
        config: SegmentationConfig(model: _selectedModel),
      );
    } catch (e, st) {
      _error = 'Failed to initialize: $e';
      debugPrint('Segmentation init error: $e\n$st');
    }

    if (mounted) {
      setState(() => _isInitializing = false);
    }
  }

  Future<void> _switchModel(SegmentationModel model) async {
    if (model == _selectedModel || _isSwitchingModel) return;

    setState(() {
      _isSwitchingModel = true;
      _selectedModel = model;
      _selectedClassIndex = null; // Reset class selection
      _mask = null; // Clear current mask
    });

    try {
      await _initSegmenter();

      // Re-segment current image if we have one
      if (_imageBytes != null) {
        await _segmentCurrentImage();
      }
    } finally {
      if (mounted) {
        setState(() => _isSwitchingModel = false);
      }
    }
  }

  Future<void> _segmentCurrentImage() async {
    if (_imageBytes == null || _segmenter == null) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final stopwatch = Stopwatch()..start();
      final mask = await _segmenter!.callFromBytes(_imageBytes!);
      stopwatch.stop();

      final Size originalSize =
          Size(mask.originalWidth.toDouble(), mask.originalHeight.toDouble());

      if (mounted) {
        setState(() {
          _mask = mask;
          _originalSize = originalSize;
          _inferenceTimeMs = stopwatch.elapsedMilliseconds;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _error = 'Segmentation failed: $e';
        });
      }
    }
  }

  @override
  void dispose() {
    _segmenter?.dispose();
    super.dispose();
  }

  Future<void> _pickAndSegment() async {
    final ImagePicker picker = ImagePicker();
    final XFile? picked =
        await picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
    if (picked == null) return;

    final Uint8List bytes = await picked.readAsBytes();

    setState(() {
      _imageBytes = bytes;
      _mask = null;
      _originalSize = null;
      _inferenceTimeMs = null;
      _error = null;
      _selectedClassIndex = null;
    });

    await _segmentCurrentImage();
  }

  void _showSettings() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) => DraggableScrollableSheet(
          initialChildSize: 0.6,
          minChildSize: 0.3,
          maxChildSize: 0.85,
          builder: (context, scrollController) => Container(
            decoration: const BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
            ),
            child: Column(
              children: [
                Container(
                  margin: const EdgeInsets.symmetric(vertical: 8),
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                Expanded(
                  child: ListView(
                    controller: scrollController,
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    children: [
                      // Model Selection
                      const Text('Model',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 16)),
                      const SizedBox(height: 8),
                      _modelOption(
                        SegmentationModel.general,
                        'General',
                        '256×256 • Binary person/background',
                        setModalState,
                      ),
                      _modelOption(
                        SegmentationModel.landscape,
                        'Landscape',
                        '144×256 • Optimized for 16:9 video',
                        setModalState,
                      ),
                      _modelOption(
                        SegmentationModel.multiclass,
                        'Multiclass',
                        '256×256 • 6 body part classes',
                        setModalState,
                      ),
                      const SizedBox(height: 16),

                      // Multiclass class selection (only show when multiclass)
                      if (_selectedModel == SegmentationModel.multiclass) ...[
                        const Text('Body Part Class',
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 16)),
                        const SizedBox(height: 4),
                        const Text(
                            'Default shows all classes with rainbow colors',
                            style: TextStyle(fontSize: 12, color: Colors.grey)),
                        const SizedBox(height: 8),
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: [
                            _classOption(null, 'All Classes', Colors.purple,
                                setModalState),
                            _classOption(
                                0, 'Background', Colors.grey, setModalState),
                            _classOption(
                                1, 'Hair', Colors.brown, setModalState),
                            _classOption(
                                2, 'Body Skin', Colors.orange, setModalState),
                            _classOption(
                                3, 'Face Skin', Colors.pink, setModalState),
                            _classOption(
                                4, 'Clothes', Colors.blue, setModalState),
                            _classOption(
                                5, 'Other', Colors.teal, setModalState),
                          ],
                        ),
                        const SizedBox(height: 16),
                      ],

                      // Display Options
                      const Text('Display Options',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 16)),
                      const SizedBox(height: 8),
                      SwitchListTile(
                        title: const Text('Show mask only'),
                        subtitle: const Text('Hide original image'),
                        value: _showMaskOnly,
                        onChanged: (value) {
                          setState(() => _showMaskOnly = value);
                          setModalState(() {});
                        },
                      ),
                      SwitchListTile(
                        title: const Text('Binary mask'),
                        subtitle: const Text('Sharp edges vs soft blend'),
                        value: _showBinaryMask,
                        onChanged: (value) {
                          setState(() => _showBinaryMask = value);
                          setModalState(() {});
                        },
                      ),
                      const SizedBox(height: 8),
                      Text('Threshold: ${_threshold.toStringAsFixed(2)}'),
                      Slider(
                        value: _threshold,
                        min: 0.0,
                        max: 1.0,
                        divisions: 20,
                        label: _threshold.toStringAsFixed(2),
                        onChanged: (value) {
                          setState(() => _threshold = value);
                          setModalState(() {});
                        },
                      ),
                      const SizedBox(height: 16),
                      const Text('Mask Color',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 16)),
                      const SizedBox(height: 8),
                      Wrap(
                        spacing: 8,
                        children: [
                          _colorOption(const Color(0x8800FF00), 'Green'),
                          _colorOption(const Color(0x88FF0000), 'Red'),
                          _colorOption(const Color(0x880000FF), 'Blue'),
                          _colorOption(const Color(0x88FFFF00), 'Yellow'),
                          _colorOption(const Color(0x88FF00FF), 'Magenta'),
                          _colorOption(const Color(0x8800FFFF), 'Cyan'),
                        ],
                      ),
                      const SizedBox(height: 24),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _modelOption(
    SegmentationModel model,
    String title,
    String subtitle,
    StateSetter setModalState,
  ) {
    final isSelected = _selectedModel == model;
    return ListTile(
      leading: Icon(
        isSelected ? Icons.radio_button_checked : Icons.radio_button_off,
        color: isSelected ? Colors.purple : Colors.grey,
      ),
      title: Text(title,
          style: TextStyle(
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal)),
      subtitle: Text(subtitle, style: const TextStyle(fontSize: 12)),
      onTap: () {
        Navigator.pop(context);
        _switchModel(model);
      },
    );
  }

  Widget _classOption(
    int? classIndex,
    String label,
    Color color,
    StateSetter setModalState,
  ) {
    final isSelected = _selectedClassIndex == classIndex;
    return GestureDetector(
      onTap: () {
        setState(() => _selectedClassIndex = classIndex);
        setModalState(() {});
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? color : color.withAlpha(77),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected ? Colors.black : Colors.transparent,
            width: 2,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.black87,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }

  Widget _colorOption(Color color, String label) {
    final isSelected = _maskColor == color;
    return GestureDetector(
      onTap: () {
        setState(() => _maskColor = color);
        Navigator.pop(context);
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected ? Colors.black : Colors.grey,
            width: isSelected ? 2 : 1,
          ),
        ),
        child: Text(label, style: const TextStyle(color: Colors.white)),
      ),
    );
  }

  String _getModelBadgeText() {
    final modelName = switch (_selectedModel) {
      SegmentationModel.general => 'General (256×256)',
      SegmentationModel.landscape => 'Landscape (144×256)',
      SegmentationModel.multiclass => 'Multiclass (256×256)',
    };

    if (_selectedModel == SegmentationModel.multiclass) {
      if (_selectedClassIndex == null) {
        return '$modelName • All Classes';
      }
      final className = switch (_selectedClassIndex) {
        0 => 'Background',
        1 => 'Hair',
        2 => 'Body Skin',
        3 => 'Face Skin',
        4 => 'Clothes',
        5 => 'Other',
        _ => 'Unknown',
      };
      return '$modelName • $className';
    }

    return modelName;
  }

  @override
  Widget build(BuildContext context) {
    final bool hasImage = _imageBytes != null && _originalSize != null;
    final bool hasMask = _mask != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Selfie Segmentation'),
        backgroundColor: Colors.purple,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            onPressed: _pickAndSegment,
            icon: const Icon(Icons.add_photo_alternate),
            tooltip: 'Pick Image',
          ),
          IconButton(
            onPressed: _showSettings,
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: Stack(
        children: [
          Center(
            child: _isInitializing
                ? const Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(color: Colors.purple),
                      SizedBox(height: 16),
                      Text('Initializing segmentation model...'),
                    ],
                  )
                : hasImage
                    ? LayoutBuilder(
                        builder: (context, constraints) {
                          final fitted = applyBoxFit(
                            BoxFit.contain,
                            _originalSize!,
                            Size(constraints.maxWidth, constraints.maxHeight),
                          );
                          final Size renderSize = fitted.destination;

                          return Stack(
                            alignment: Alignment.center,
                            children: [
                              // Original image (or gray background for mask-only)
                              if (!_showMaskOnly)
                                Image.memory(
                                  _imageBytes!,
                                  width: renderSize.width,
                                  height: renderSize.height,
                                  fit: BoxFit.contain,
                                )
                              else
                                Container(
                                  width: renderSize.width,
                                  height: renderSize.height,
                                  color: Colors.grey[900],
                                ),
                              // Mask overlay
                              if (hasMask)
                                CustomPaint(
                                  size: renderSize,
                                  painter: _SegmentationMaskPainter(
                                    mask: _mask!,
                                    originalSize: _originalSize!,
                                    threshold: _threshold,
                                    binary: _showBinaryMask,
                                    maskColor: _maskColor,
                                    classIndex: _selectedClassIndex,
                                    showAllClasses: _selectedModel ==
                                            SegmentationModel.multiclass &&
                                        _selectedClassIndex == null,
                                  ),
                                ),
                            ],
                          );
                        },
                      )
                    : Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.person_outline,
                              size: 100, color: Colors.purple[200]),
                          const SizedBox(height: 24),
                          const Text(
                            'Pick an image to segment',
                            style: TextStyle(fontSize: 18, color: Colors.grey),
                          ),
                          const SizedBox(height: 16),
                          ElevatedButton.icon(
                            onPressed: _pickAndSegment,
                            icon: const Icon(Icons.add_photo_alternate),
                            label: const Text('Select Image'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.purple,
                              foregroundColor: Colors.white,
                            ),
                          ),
                        ],
                      ),
          ),
          // Loading overlay
          if (_isLoading)
            Container(
              color: Colors.black54,
              child: const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(color: Colors.purple),
                    SizedBox(height: 16),
                    Text('Segmenting...',
                        style: TextStyle(color: Colors.white)),
                  ],
                ),
              ),
            ),
          // Performance badge
          if (_inferenceTimeMs != null && !_isLoading)
            Positioned(
              top: 12,
              left: 12,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black.withAlpha(179),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _inferenceTimeMs! < 100 ? Icons.speed : Icons.timer,
                      size: 16,
                      color: _inferenceTimeMs! < 100
                          ? Colors.green
                          : _inferenceTimeMs! < 300
                              ? Colors.lightGreen
                              : Colors.orange,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${_inferenceTimeMs}ms',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          // Model info badge
          if (hasMask && !_isLoading)
            Positioned(
              top: 12,
              right: 12,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black.withAlpha(179),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text(
                  _getModelBadgeText(),
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ),
            ),
          // Error display
          if (_error != null)
            Positioned(
              bottom: 20,
              left: 20,
              right: 20,
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red[800],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _error!,
                  style: const TextStyle(color: Colors.white),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _SegmentationMaskPainter extends CustomPainter {
  final SegmentationMask mask;
  final Size originalSize;
  final double threshold;
  final bool binary;
  final Color maskColor;
  final int? classIndex;
  final bool showAllClasses;

  _SegmentationMaskPainter({
    required this.mask,
    required this.originalSize,
    required this.threshold,
    required this.binary,
    required this.maskColor,
    this.classIndex,
    this.showAllClasses = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final v = _maskValidRegion(mask);
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
            final color = _kSegmentationClassColors[winningClass];
            final baseAlpha = (color.a * 255).round();
            paint.color =
                binary ? color : color.withAlpha((maxProb * baseAlpha).round());
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

      _drawClassLabels(canvas, labelCounts, labelSumX, labelSumY);
      return;
    }

    Float32List? classMaskData;
    if (classIndex != null && mask is MulticlassSegmentationMask) {
      classMaskData =
          (mask as MulticlassSegmentationMask).classMask(classIndex!);
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
  bool shouldRepaint(covariant _SegmentationMaskPainter old) {
    return old.mask != mask ||
        old.threshold != threshold ||
        old.binary != binary ||
        old.maskColor != maskColor ||
        old.classIndex != classIndex ||
        old.showAllClasses != showAllClasses;
  }
}
