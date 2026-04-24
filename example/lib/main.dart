import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:camera/camera.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:sensors_plus/sensors_plus.dart';

/// Per-class colors for multiclass segmentation overlay, aligned with the
/// class indices in `kSegmentationClassLabels` (0=BG, 1=Hair, 2=Body,
/// 3=Face, 4=Clothes, 5=Other).
const List<Color> _kSegmentationClassColors = [
  Color(0x99A0A0A0),
  Color(0x99CD853F),
  Color(0x88FFA500),
  Color(0x88FF69B4),
  Color(0x9900BFFF),
  Color(0x9940E0D0),
];

/// Widget shown for the selected dropdown item (inherits parent text style).
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

    Size decodedSize;
    if (faces.isNotEmpty) {
      decodedSize = faces.first.originalSize;
    } else {
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
    final perf = performanceLevel(_totalTimeMs!);
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
    final perf = performanceLevel(_totalTimeMs!);
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
          IconButton(
            onPressed: _pickAndRun,
            icon: const Icon(Icons.add_photo_alternate),
            tooltip: 'Pick Image',
          ),
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
          IconButton(
            onPressed: _showSettingsSheet,
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: Stack(
        children: [
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
                              painter: DetectionsPainter(
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
          if (hasImage) _buildCompactPerformanceBadge(),
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
  DeviceOrientation _deviceOrientation = DeviceOrientation.portraitUp;
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;
  int _detectionTimeMs = 0;
  final FpsCounter _fpsCounter = FpsCounter();
  int _fps = 0;
  bool _isImageStreamStarted = false;

  FaceDetectionMode _detectionMode = FaceDetectionMode.full;
  FaceDetectionModel _detectionModel = FaceDetectionModel.backCamera;

  /// One-shot-per-orientation probe for iOS rotation debugging.
  /// See `_rotationFlagForFrame`. Used to settle whether iOS buffers are
  /// sensor-native (same as Android) or already rotated by the plugin.
  DeviceOrientation? _iosProbeOrientation;

  bool _showSegmentation = false;
  SegmentationMask? _segmentationMask;
  final Color _segmentationColor = const Color(0x8800FF00);
  SegmentationModel _liveSegmentationModel = SegmentationModel.general;

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
            ? (event.x > 0
                ? DeviceOrientation.landscapeLeft
                : DeviceOrientation.landscapeRight)
            : (event.y > 0
                ? DeviceOrientation.portraitUp
                : DeviceOrientation.portraitDown);
        if (next == DeviceOrientation.portraitDown &&
            (_deviceOrientation == DeviceOrientation.landscapeLeft ||
                _deviceOrientation == DeviceOrientation.landscapeRight)) {
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

  Future<({List<Face> faces, SegmentationMask? segMask})> _detectForLiveCamera(
    CameraImage image, {
    required CameraFrameRotation? rotation,
    required int maxDim,
  }) async {
    if ((_showSegmentation || _showVirtualBackground) &&
        _faceDetector!.isSegmentationReady) {
      final frame = prepareCameraFrameFromImage(
        image,
        rotation: rotation,
        isBgra: Platform.isMacOS,
      );
      if (frame == null) {
        return (faces: <Face>[], segMask: null);
      }
      final result =
          await _faceDetector!.detectFacesWithSegmentationFromCameraFrame(
        frame,
        mode: _detectionMode,
        maxDim: maxDim,
      );
      return (faces: result.faces, segMask: result.segmentationMask);
    }
    final faces = await _faceDetector!.detectFacesFromCameraImage(
      image,
      rotation: rotation,
      isBgra: Platform.isMacOS,
      mode: _detectionMode,
      maxDim: maxDim,
    );
    return (faces: faces, segMask: null);
  }

  Future<void> _switchLiveSegmentationModel(SegmentationModel model) async {
    if (model == _liveSegmentationModel) return;
    setState(() {
      _liveSegmentationModel = model;
      _segmentationMask = null;
    });
    await _reinitDetectorIsolate();
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
                  painter: BackgroundImagePainter(image: _beachBackground!),
                ),
              ),
            cameraWidget,
            if (_showVirtualBackground &&
                _beachBackground != null &&
                _segmentationMask != null)
              CustomPaint(
                painter: VirtualBackgroundOverlayPainter(
                  background: _beachBackground!,
                  mask: _segmentationMask!,
                  mirrorHorizontally: mirrorHorizontally,
                ),
              ),
            if (_showSegmentation &&
                !_showVirtualBackground &&
                _segmentationMask != null)
              CustomPaint(
                painter: LiveSegmentationPainter(
                  mask: _segmentationMask!,
                  maskColor: _segmentationColor,
                  showAllClasses:
                      _liveSegmentationModel == SegmentationModel.multiclass,
                  mirrorHorizontally: mirrorHorizontally,
                  classColors: _kSegmentationClassColors,
                ),
              ),
            if (_imageSize != null)
              CustomPaint(
                painter: CameraDetectionPainter(
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
        _faceDetector = FaceDetector();
        await _faceDetector!.initialize(model: _detectionModel);
      }

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

  Future<void> _processCameraImage(CameraImage image) async {
    if (_fpsCounter.tick() && mounted) {
      setState(() => _fps = _fpsCounter.fps);
    }

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

    if (_isProcessing) return;

    _isProcessing = true;

    try {
      final startTime = DateTime.now();

      if (_faceDetector == null || !mounted) {
        _isProcessing = false;
        return;
      }
      final sensor = _sensorOrientation;
      final CameraFrameRotation? rotation = sensor == null
          ? null
          : rotationForFrame(
              width: image.width,
              height: image.height,
              sensorOrientation: sensor,
              isFrontCamera: _isFrontCamera,
              deviceOrientation: _effectiveDeviceOrientation(context),
            );

      const int maxDim = 640;
      final Size size = detectionSize(
        width: image.width,
        height: image.height,
        rotation: rotation,
        maxDim: maxDim,
      );

      final result = await _detectForLiveCamera(
        image,
        rotation: rotation,
        maxDim: maxDim,
      );
      final faces = result.faces;
      final segMask = result.segMask;

      final endTime = DateTime.now();
      final detectionTime = endTime.difference(startTime).inMilliseconds;

      if (mounted) {
        setState(() {
          _faces = faces;
          _imageSize = size;
          _detectionTimeMs = detectionTime;
          _segmentationMask = segMask;
        });
      }
    } catch (_) {
      /// Silently handle errors during processing to keep the stream alive.
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

    final int turns = barQuarterTurns(_deviceOrientation);

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          _buildCameraOverlays(
            cameraWidget: CameraPreview(_cameraController!),
            cameraAspectRatio: cameraAspectRatio,
            displayAspectRatio: displayAspectRatio,
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

  SegmentationModel _selectedModel = SegmentationModel.general;

  double _threshold = 0.5;
  bool _showMaskOnly = false;
  bool _showBinaryMask = true;
  Color _maskColor = const Color(0x8800FF00);

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
      _selectedClassIndex = null;
      _mask = null;
    });

    try {
      await _initSegmenter();

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
                              if (hasMask)
                                CustomPaint(
                                  size: renderSize,
                                  painter: SegmentationMaskPainter(
                                    mask: _mask!,
                                    originalSize: _originalSize!,
                                    threshold: _threshold,
                                    binary: _showBinaryMask,
                                    maskColor: _maskColor,
                                    classIndex: _selectedClassIndex,
                                    showAllClasses: _selectedModel ==
                                            SegmentationModel.multiclass &&
                                        _selectedClassIndex == null,
                                    classColors: _kSegmentationClassColors,
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
