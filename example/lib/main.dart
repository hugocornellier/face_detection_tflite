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
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:sensors_plus/sensors_plus.dart';

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

// Widget shown for the selected dropdown item (inherits parent text style).
class _DropdownSelected extends StatelessWidget {
  final String text;
  const _DropdownSelected(this.text);
  @override
  Widget build(BuildContext context) =>
      Align(alignment: Alignment.centerLeft, child: Text(text));
}

DropdownMenuItem<T> _whiteItem<T>(T value, String label) => DropdownMenuItem<T>(
      value: value,
      child: Text(label, style: const TextStyle(color: Colors.white)),
    );

// Compute the axis-aligned bounding rect of a set of offsets.
Rect _boundsOf(Iterable<Offset> pts) {
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

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    title: 'Face Detection Demo',
    theme: ThemeData(
      colorSchemeSeed: Colors.blue,
      useMaterial3: true,
    ),
    home: const HomeScreen(),
  ));
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Detection Demo'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Choose Detection Mode',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 48),
            _buildModeCard(
              context,
              icon: Icons.image,
              title: 'Still Image',
              description: 'Detect faces in photos from gallery or camera',
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const Example()),
                );
              },
            ),
            const SizedBox(height: 24),
            _buildModeCard(
              context,
              icon: Icons.videocam,
              title: 'Live Camera',
              description: 'Real-time face detection from camera feed',
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const LiveCameraScreen()),
                );
              },
            ),
            const SizedBox(height: 24),
            _buildModeCard(
              context,
              icon: Icons.person_outline,
              title: 'Selfie Segmentation',
              description: 'Segment selfie foreground from background',
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const SegmentationDemoScreen()),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModeCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String description,
    required VoidCallback onTap,
  }) {
    return SizedBox(
      width: 400,
      child: Card(
        elevation: 4,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Row(
              children: [
                Icon(icon, size: 64, color: Colors.blue),
                const SizedBox(width: 24),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        description,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: Colors.grey[600],
                            ),
                      ),
                    ],
                  ),
                ),
                const Icon(Icons.arrow_forward_ios),
              ],
            ),
          ),
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
  FaceDetector? _faceDetector;
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
  bool _showLandmarkLabels = false;
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

  FaceDetectionModel _detectionModel = FaceDetectionModel.backCamera;

  @override
  void initState() {
    super.initState();
    _initFaceDetector();
  }

  Future<void> _initFaceDetector() async {
    try {
      await _faceDetector?.dispose();
      _faceDetector = FaceDetector();
      await _faceDetector!.initialize(model: _detectionModel);
    } catch (_) {}
    setState(() {});
  }

  @override
  void dispose() {
    _faceDetector?.dispose();
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

    if (_faceDetector == null || !_faceDetector!.isReady) {
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
        await _faceDetector!.detectFaces(bytes, mode: mode);
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
        builder: (context, scrollController) => StatefulBuilder(
          builder: (context, setSheetState) {
            void updateState(VoidCallback fn) {
              fn();
              setSheetState(() {});
              setState(() {});
            }

            Future<void> onSheetFeatureToggle(
                void Function(bool) assign, bool newValue) async {
              final FaceDetectionMode oldMode = _determineMode();
              assign(newValue);
              setSheetState(() {});
              setState(() {});

              final FaceDetectionMode newMode = _determineMode();
              if (_imageBytes != null && oldMode != newMode) {
                await _processImage(_imageBytes!);
              }
            }

            Widget cb(String label, bool v, void Function(bool) set) =>
                _buildCheckbox(
                    label, v, (x) => updateState(() => set(x ?? false)));
            Widget fcb(String label, bool v, void Function(bool) set) =>
                _buildCheckbox(
                    label, v, (x) => onSheetFeatureToggle(set, x ?? false));
            Widget col(String label, Color c, void Function(Color) set) =>
                _buildColorButton(label, c, (x) => updateState(() => set(x)));
            Widget sl(String label, double v, double mn, double mx,
                    void Function(double) set) =>
                _buildCompactSlider(
                    label, v, mn, mx, (x) => updateState(() => set(x)));

            return Container(
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
                                cb('Bounding Boxes', _showBoundingBoxes,
                                    (v) => _showBoundingBoxes = v),
                                fcb('Mesh', _showMesh, (v) => _showMesh = v),
                                cb('Landmarks', _showLandmarks,
                                    (v) => _showLandmarks = v),
                                fcb('Irises', _showIrises,
                                    (v) => _showIrises = v),
                                fcb('Eye Contour', _showEyeContours,
                                    (v) => _showEyeContours = v),
                                fcb('Eye Mesh', _showEyeMesh,
                                    (v) => _showEyeMesh = v),
                                cb('Landmark Labels', _showLandmarkLabels,
                                    (v) => _showLandmarkLabels = v),
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
                                col('BBox', _boundingBoxColor,
                                    (c) => _boundingBoxColor = c),
                                col('Landmarks', _landmarkColor,
                                    (c) => _landmarkColor = c),
                                col('Mesh', _meshColor, (c) => _meshColor = c),
                                col('Irises', _irisColor,
                                    (c) => _irisColor = c),
                                col('Eye Contour', _eyeContourColor,
                                    (c) => _eyeContourColor = c),
                                col('Eye Mesh', _eyeMeshColor,
                                    (c) => _eyeMeshColor = c),
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
                            sl('BBox', _boundingBoxThickness, 0.5, 10.0,
                                (v) => _boundingBoxThickness = v),
                            sl('Landmark', _landmarkSize, 0.5, 15.0,
                                (v) => _landmarkSize = v),
                            sl('Mesh', _meshSize, 0.1, 10.0,
                                (v) => _meshSize = v),
                            sl('Eye Mesh', _eyeMeshSize, 0.1, 10.0,
                                (v) => _eyeMeshSize = v),
                            const SizedBox(height: 8),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
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
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('MODEL',
                    style: TextStyle(
                        color: Colors.grey[500],
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1.0)),
                const SizedBox(width: 4),
                DropdownButton<FaceDetectionModel>(
                  value: _detectionModel,
                  dropdownColor: Colors.blue[800],
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onSurface,
                    fontSize: 14,
                  ),
                  underline: const SizedBox(),
                  icon: Icon(Icons.arrow_drop_down,
                      color: Theme.of(context).colorScheme.onSurface),
                  selectedItemBuilder: (context) => const [
                    _DropdownSelected('Front'),
                    _DropdownSelected('Back'),
                    _DropdownSelected('Short'),
                    _DropdownSelected('Full'),
                    _DropdownSelected('Sparse'),
                  ],
                  items: [
                    _whiteItem(FaceDetectionModel.frontCamera, 'Front'),
                    _whiteItem(FaceDetectionModel.backCamera, 'Back'),
                    _whiteItem(FaceDetectionModel.shortRange, 'Short'),
                    _whiteItem(FaceDetectionModel.full, 'Full'),
                    _whiteItem(FaceDetectionModel.fullSparse, 'Sparse'),
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
              ],
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
                                showLandmarkLabels: _showLandmarkLabels,
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

  _DetectionsPainter({
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
              final bounds = _boundsOf([iris.irisCenter, ...iris.irisContour]
                  .map((p) => Offset(ox + p.x * scaleX, oy + p.y * scaleY)));
              canvas.drawOval(bounds, irisFill);
              canvas.drawOval(bounds, irisStroke);
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

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = const [];
  FaceDetector? _faceDetector;
  List<Face> _faces = [];
  Size? _imageSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;
  bool _isSwitchingCamera = false;
  bool _isProcessing = false;
  bool _isInitialized = false;
  String _deviceOrientation = 'Portrait Up';
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;
  int _detectionTimeMs = 0;
  int _fps = 0;
  DateTime? _lastFpsUpdate;
  int _framesSinceLastUpdate = 0;
  bool _isImageStreamStarted = false;

  // Detection settings
  FaceDetectionMode _detectionMode = FaceDetectionMode.full;
  FaceDetectionModel _detectionModel = FaceDetectionModel.backCamera;

  /// One-shot-per-orientation probe for iOS rotation debugging.
  /// See `_rotationFlagForFrame`. Used to settle whether iOS buffers are
  /// sensor-native (same as Android) or already rotated by the plugin.
  DeviceOrientation? _iosProbeOrientation;

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

    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      _accelerometerSub = accelerometerEventStream().listen((event) {
        final next = event.x.abs() > event.y.abs()
            ? (event.x > 0 ? 'Landscape Left' : 'Landscape Right')
            : (event.y > 0 ? 'Portrait Up' : 'Portrait Down');
        if (next == 'Portrait Down' &&
            (_deviceOrientation == 'Landscape Left' ||
                _deviceOrientation == 'Landscape Right')) {
          return;
        }
        if (next != _deviceOrientation && mounted) {
          setState(() => _deviceOrientation = next);
        }
      });
    }
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
    final old = _faceDetector;
    _faceDetector = null;
    await old?.dispose();
    _faceDetector = FaceDetector();
    await _faceDetector!.initialize(model: _detectionModel);
    await _faceDetector!.initializeSegmentation(
      config: SegmentationConfig(model: _liveSegmentationModel),
    );
  }

  Future<({List<Face> faces, SegmentationMask? segMask})>
      _detectFromCameraFrame(
    CameraFrame frame, {
    required int maxDim,
  }) async {
    if ((_showSegmentation || _showVirtualBackground) &&
        _faceDetector!.isSegmentationReady) {
      final result =
          await _faceDetector!.detectFacesWithSegmentationFromCameraFrame(
        frame,
        mode: _detectionMode,
        maxDim: maxDim,
      );
      return (faces: result.faces, segMask: result.segmentationMask);
    } else {
      final faces = await _faceDetector!.detectFacesFromCameraFrame(
        frame,
        mode: _detectionMode,
        maxDim: maxDim,
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

  int get _barQuarterTurns {
    switch (_deviceOrientation) {
      case 'Landscape Left':
        return 1;
      case 'Landscape Right':
        return 3;
      default:
        return 0;
    }
  }

  /// Builds the live-camera top bar as a plain [Material]-backed [Row] rather
  /// than an [AppBar]. AppBar doesn't play well inside a [RotatedBox] (applies
  /// rotated [MediaQuery] safe-area padding, depends on [Scaffold.appBar]-slot
  /// theming) so its `actions` would render invisibly when the phone is in
  /// landscape; a plain Row sidesteps all of that.
  Widget _buildCameraTopBar() {
    final canPop = Navigator.of(context).canPop();
    final isMobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);

    final fpsText = SizedBox(
      width: 70,
      child: Text(
        'FPS: $_fps',
        style: const TextStyle(color: Colors.white, fontSize: 14),
        textAlign: isMobile ? TextAlign.left : TextAlign.right,
      ),
    );
    const separator = Text(
      ' | ',
      style: TextStyle(color: Colors.white, fontSize: 14),
    );
    final msText = SizedBox(
      width: 70,
      child: Text(
        '${_detectionTimeMs}ms',
        style: const TextStyle(color: Colors.white, fontSize: 14),
      ),
    );

    return Material(
      color: Colors.black.withAlpha(179),
      elevation: 4,
      child: SizedBox(
        height: kToolbarHeight,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Row(
            children: [
              if (canPop)
                IconButton(
                  tooltip: 'Back',
                  color: Colors.white,
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => Navigator.of(context).maybePop(),
                ),
              if (isMobile) ...[
                const SizedBox(width: 8),
                fpsText,
                separator,
                msText,
                const Spacer(),
              ] else
                const Expanded(
                  child: Padding(
                    padding: EdgeInsets.symmetric(horizontal: 8),
                    child: Text(
                      'Live Camera Detection',
                      style: TextStyle(color: Colors.white, fontSize: 18),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ),
              if (_canSwitchCamera)
                IconButton(
                  tooltip: _isFrontCamera
                      ? 'Switch to back camera'
                      : 'Switch to front camera',
                  color: Colors.white,
                  icon: Icon(Platform.isIOS
                      ? Icons.flip_camera_ios
                      : Icons.flip_camera_android),
                  onPressed: _isSwitchingCamera ? null : _switchCamera,
                ),
              PopupMenuButton<void>(
                tooltip: 'Settings',
                icon: const Icon(Icons.settings, color: Colors.white),
                color: Colors.blueGrey[900],
                padding: EdgeInsets.zero,
                itemBuilder: (context) => [
                  PopupMenuItem<void>(
                    enabled: false,
                    padding: EdgeInsets.zero,
                    child: StatefulBuilder(
                      builder: (context, setMenuState) {
                        return _buildSettingsMenuContent(setMenuState);
                      },
                    ),
                  ),
                ],
              ),
              if (!isMobile) ...[
                const SizedBox(width: 8),
                fpsText,
                separator,
                msText,
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSettingsMenuContent(StateSetter setMenuState) {
    void update(VoidCallback fn) {
      setState(fn);
      setMenuState(() {});
    }

    Widget chip<T>({
      required T value,
      required T current,
      required String label,
      required VoidCallback onTap,
    }) {
      final isSelected = current == value;
      return GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: isSelected ? Colors.blue : Colors.white12,
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

    const sectionLabelStyle = TextStyle(
      color: Colors.white60,
      fontSize: 10,
      fontWeight: FontWeight.w600,
      letterSpacing: 1.2,
    );

    return SizedBox(
      width: 260,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('SPEED', style: sectionLabelStyle),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final (v, label) in const [
                  (FaceDetectionMode.fast, 'Fast'),
                  (FaceDetectionMode.standard, 'Standard'),
                  (FaceDetectionMode.full, 'Full'),
                ])
                  chip<FaceDetectionMode>(
                    value: v,
                    current: _detectionMode,
                    label: label,
                    onTap: () => update(() => _detectionMode = v),
                  ),
              ],
            ),
            const Divider(color: Colors.white24, height: 24),
            const Text('MODEL', style: sectionLabelStyle),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final (v, label) in const [
                  (FaceDetectionModel.frontCamera, 'Front'),
                  (FaceDetectionModel.backCamera, 'Back'),
                  (FaceDetectionModel.shortRange, 'Short'),
                  (FaceDetectionModel.full, 'Full Range'),
                  (FaceDetectionModel.fullSparse, 'Full Sparse'),
                ])
                  chip<FaceDetectionModel>(
                    value: v,
                    current: _detectionModel,
                    label: label,
                    onTap: () async {
                      if (v == _detectionModel) return;
                      update(() => _detectionModel = v);
                      await _reinitDetectorIsolate();
                      if (mounted) setMenuState(() {});
                    },
                  ),
              ],
            ),
            const Divider(color: Colors.white24, height: 24),
            const Text('SEGMENTATION', style: sectionLabelStyle),
            const SizedBox(height: 4),
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Show',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                ),
                Switch(
                  value: _showSegmentation,
                  activeTrackColor: Colors.blue,
                  onChanged: (value) {
                    update(() {
                      _showSegmentation = value;
                      if (!value) _segmentationMask = null;
                    });
                  },
                ),
              ],
            ),
            if (_showSegmentation) ...[
              const SizedBox(height: 4),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [
                  for (final (v, label) in const [
                    (SegmentationModel.general, 'Binary'),
                    (SegmentationModel.multiclass, '6-Class'),
                  ])
                    chip<SegmentationModel>(
                      value: v,
                      current: _liveSegmentationModel,
                      label: label,
                      onTap: () async {
                        if (v == _liveSegmentationModel) return;
                        await _switchLiveSegmentationModel(v);
                        if (mounted) setMenuState(() {});
                      },
                    ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  const Expanded(
                    child: Text(
                      'Virtual Background',
                      style: TextStyle(color: Colors.white70, fontSize: 14),
                    ),
                  ),
                  Switch(
                    value: _showVirtualBackground,
                    activeTrackColor: Colors.blue,
                    onChanged: (value) {
                      update(() {
                        _showVirtualBackground = value;
                        if (!value) _segmentationMask = null;
                      });
                    },
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
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

  Future<void> _initCamera() async {
    try {
      try {
        await _reinitDetectorIsolate();
      } catch (e) {
        debugPrint('Detector isolate init failed (segmentation may be '
            'unavailable): $e');
        // Retry without segmentation so face detection still works.
        _faceDetector = FaceDetector();
        await _faceDetector!.initialize(model: _detectionModel);
      }

      // Unified camera initialization for ALL platforms
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No cameras available')),
          );
        }
        return;
      }
      _availableCameras = cameras;

      // Prefer the front camera; fall back to the first camera
      // (desktop cameras may report lensDirection=external, so fallback is important)
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      await _startControllerFor(camera, markInitialized: true);
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

  Future<void> _startControllerFor(
    CameraDescription camera, {
    required bool markInitialized,
  }) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await controller.initialize();

    if (!mounted) {
      await controller.dispose();
      return;
    }

    setState(() {
      _cameraController = controller;
      if (markInitialized) _isInitialized = true;
      _sensorOrientation = controller.description.sensorOrientation;
      _isFrontCamera =
          controller.description.lensDirection == CameraLensDirection.front;
    });

    await controller.startImageStream(_processCameraImage);
    _isImageStreamStarted = true;
  }

  bool get _canSwitchCamera {
    if (kIsWeb) return false;
    if (!(Platform.isAndroid || Platform.isIOS)) return false;
    final hasFront = _availableCameras
        .any((c) => c.lensDirection == CameraLensDirection.front);
    final hasBack = _availableCameras
        .any((c) => c.lensDirection == CameraLensDirection.back);
    return hasFront && hasBack;
  }

  Future<void> _switchCamera() async {
    if (_isSwitchingCamera) return;
    if (!_canSwitchCamera) return;

    final target =
        _isFrontCamera ? CameraLensDirection.back : CameraLensDirection.front;
    final next = _availableCameras.firstWhere(
      (c) => c.lensDirection == target,
      orElse: () => _availableCameras.first,
    );

    final prev = _cameraController;
    setState(() {
      _isSwitchingCamera = true;
      _cameraController = null;
      _faces = [];
      _imageSize = null;
      _segmentationMask = null;
    });
    try {
      if (prev != null) {
        if (_isImageStreamStarted) {
          try {
            await prev.stopImageStream();
          } catch (_) {}
          _isImageStreamStarted = false;
        }
        await prev.dispose();
      }

      await _startControllerFor(next, markInitialized: false);
    } catch (e, st) {
      debugPrint('Camera switch failed: $e');
      debugPrint('$st');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error switching camera: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isSwitchingCamera = false);
    }
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

  CameraFrameRotation? _rotationForFrame({
    required int width,
    required int height,
  }) {
    final int? sensor = _sensorOrientation;
    if (sensor == null) return null;

    // iOS: the camera plugin pre-rotates the image stream per
    // AVCaptureConnection.videoOrientation, so the historical portrait-only
    // rotation path still applies.
    if (Platform.isIOS) {
      final DeviceOrientation orientation =
          _effectiveDeviceOrientation(context);
      final bool isPortrait = orientation == DeviceOrientation.portraitUp ||
          orientation == DeviceOrientation.portraitDown;
      if (!isPortrait) return null;
      if (height >= width) return null;
      if (sensor == 90) return CameraFrameRotation.cw90;
      if (sensor == 270) return CameraFrameRotation.cw270;
      return null;
    }

    // Android: combined formula covering all four device orientations.
    if (Platform.isAndroid) {
      final DeviceOrientation d = _effectiveDeviceOrientation(context);
      final int deviceRotation = switch (d) {
        DeviceOrientation.portraitUp => 0,
        DeviceOrientation.landscapeLeft => 90,
        DeviceOrientation.portraitDown => 180,
        DeviceOrientation.landscapeRight => 270,
      };

      final int total = _isFrontCamera
          ? (sensor + deviceRotation) % 360
          : (sensor - deviceRotation + 360) % 360;

      return switch (total) {
        90 => CameraFrameRotation.cw90,
        180 => CameraFrameRotation.cw180,
        270 => CameraFrameRotation.cw270,
        _ => null,
      };
    }

    // Desktop / web: camera_desktop delivers already-upright frames.
    return null;
  }

  /// Computes the final detection-image size (post-rotate, post-downscale)
  /// used by the overlay painter. Deterministic from the same inputs the
  /// detection isolate receives, so no round-trip is needed.
  Size _detectionSize({
    required int width,
    required int height,
    required CameraFrameRotation? rotation,
    required int maxDim,
  }) {
    int w = width;
    int h = height;
    if (rotation == CameraFrameRotation.cw90 ||
        rotation == CameraFrameRotation.cw270) {
      final int t = w;
      w = h;
      h = t;
    }
    if (w > maxDim || h > maxDim) {
      final double scale = maxDim / (w > h ? w : h);
      w = (w * scale).toInt();
      h = (h * scale).toInt();
    }
    return Size(w.toDouble(), h.toDouble());
  }

  Future<void> _processCameraImage(CameraImage image) async {
    _updateFps();

    if (Platform.isIOS) {
      final DeviceOrientation d = _effectiveDeviceOrientation(context);
      if (d != _iosProbeOrientation) {
        _iosProbeOrientation = d;
        debugPrint('[ios-probe] orient=${d.name} '
            'sensor=$_sensorOrientation front=$_isFrontCamera '
            'raw=${image.width}x${image.height} '
            'planes=${image.planes.length}');
      }
    }

    // Skip if already processing
    if (_isProcessing) return;

    _isProcessing = true;

    try {
      final startTime = DateTime.now();

      // All OpenCV work (YUV→BGR, rotate, downscale) happens in the detection
      // isolate. The UI thread just packs plane metadata.
      final rotation =
          _rotationForFrame(width: image.width, height: image.height);
      final frame = prepareCameraFrame(
        width: image.width,
        height: image.height,
        planes: [
          for (final p in image.planes)
            (
              bytes: p.bytes,
              rowStride: p.bytesPerRow,
              pixelStride: p.bytesPerPixel ?? 1,
            ),
        ],
        rotation: rotation,
        isBgra: Platform.isMacOS,
      );

      if (frame == null || _faceDetector == null) {
        _isProcessing = false;
        return;
      }

      // Detection model internally resizes to 128–256px, so full-res frames
      // just waste IPC bandwidth.
      const int maxDim = 640;
      final Size detectionSize = _detectionSize(
        width: image.width,
        height: image.height,
        rotation: rotation,
        maxDim: maxDim,
      );

      final result = await _detectFromCameraFrame(frame, maxDim: maxDim);
      final faces = result.faces;
      final segMask = result.segMask;

      final endTime = DateTime.now();
      final detectionTime = endTime.difference(startTime).inMilliseconds;

      if (mounted) {
        setState(() {
          _faces = faces;
          _imageSize = detectionSize;
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

  @override
  void dispose() {
    _accelerometerSub?.cancel();
    if (_isImageStreamStarted) {
      _cameraController?.stopImageStream();
    }
    _cameraController?.dispose();
    _faceDetector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized || _cameraController == null) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Live Camera Detection'),
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

    final int turns = _barQuarterTurns;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          _buildCameraOverlays(
            cameraWidget: CameraPreview(_cameraController!),
            cameraAspectRatio: cameraAspectRatio,
            displayAspectRatio: displayAspectRatio,
            // On Android the camera plugin mirrors the front preview texture,
            // but the image stream remains unmirrored. Mirror the overlay to
            // match. iOS delivers the preview and image stream in aligned
            // spaces, so no flip is needed there.
            mirrorHorizontally: Platform.isAndroid && _isFrontCamera,
            sensorOrientation: _sensorOrientation ?? 0,
            deviceOrientation: deviceOrientation,
            isFrontCamera: _isFrontCamera,
          ),
          _positionedTopBar(turns),
        ],
      ),
    );
  }

  Widget _positionedTopBar(int turns) {
    final bar = _buildCameraTopBar();
    final padding = MediaQuery.of(context).padding;
    if (turns == 0) {
      return Positioned(
        top: padding.top,
        left: padding.left,
        right: padding.right,
        child: bar,
      );
    }
    return Positioned(
      top: padding.top,
      bottom: padding.bottom,
      left: turns == 3 ? padding.left : null,
      right: turns == 1 ? padding.right : null,
      width: kToolbarHeight,
      child: RotatedBox(quarterTurns: turns, child: bar),
    );
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
            final oval = _boundsOf([iris.irisCenter, ...iris.irisContour]
                .map((p) => transformPoint(p.x, p.y)));
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
                            for (final o in const <(int?, String, Color)>[
                              (null, 'All Classes', Colors.blue),
                              (0, 'Background', Colors.grey),
                              (1, 'Hair', Colors.brown),
                              (2, 'Body Skin', Colors.orange),
                              (3, 'Face Skin', Colors.pink),
                              (4, 'Clothes', Colors.blue),
                              (5, 'Other', Colors.teal),
                            ])
                              _classOption(o.$1, o.$2, o.$3, setModalState),
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
        color: isSelected ? Colors.blue : Colors.grey,
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
                      CircularProgressIndicator(),
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
                              size: 100, color: Colors.blue[200]),
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
                    CircularProgressIndicator(),
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
