import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:camera/camera.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';
import 'package:flutter_litert/flutter_litert.dart' show OneEuroFilter;
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:video_player/video_player.dart';

/// Centers [child] in the available space; when the viewport is too small to
/// fit it, the content scrolls vertically instead of overflowing.
class _ScrollableCentered extends StatelessWidget {
  final Widget child;
  const _ScrollableCentered({required this.child});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) => SingleChildScrollView(
        child: ConstrainedBox(
          constraints: BoxConstraints(minHeight: constraints.maxHeight),
          child: Center(child: child),
        ),
      ),
    );
  }
}

/// Compact labeled swatch that opens a color picker dialog on tap.
class _ColorPickerButton extends StatelessWidget {
  final String label;
  final Color color;
  final ValueChanged<Color> onColorChanged;

  const _ColorPickerButton({
    required this.label,
    required this.color,
    required this.onColorChanged,
  });

  void _pick(BuildContext context) {
    Color tempColor = color;
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Pick $label Color'),
        content: SingleChildScrollView(
          child: ColorPicker(
            pickerColor: color,
            onColorChanged: (c) => tempColor = c,
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
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => _pick(context),
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
      body: _ScrollableCentered(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Text(
                    'Choose a Demo',
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                ),
                const SizedBox(height: 28),
                _buildSection(
                  context,
                  'Face Detection / Segmentation',
                  [
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
                    _buildModeCard(
                      context,
                      icon: Icons.image,
                      title: 'Still Image',
                      description:
                          'Detect faces in photos from gallery or camera',
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => const Example()),
                        );
                      },
                    ),
                    _buildModeCard(
                      context,
                      icon: Icons.movie_creation_outlined,
                      title: 'Video File',
                      description:
                          'Process an MP4 frame-by-frame with smoothed '
                          'face detection',
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => const VideoFileScreen()),
                        );
                      },
                    ),
                  ],
                ),
                const SizedBox(height: 28),
                _buildSection(
                  context,
                  'Embeddings / Facial Recognition',
                  [
                    _buildModeCard(
                      context,
                      icon: Icons.face_retouching_natural,
                      title: 'Face Recognition',
                      description:
                          'Compare two photos and measure face similarity',
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => const EmbeddingsScreen()),
                        );
                      },
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSection(
    BuildContext context,
    String title,
    List<Widget> cards,
  ) {
    final List<Widget> row = [];
    for (int i = 0; i < cards.length; i++) {
      if (i > 0) row.add(const SizedBox(width: 12));
      row.add(cards[i]);
    }
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            title,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: Colors.grey[700],
                ),
          ),
        ),
        const SizedBox(height: 12),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: IntrinsicHeight(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: row,
            ),
          ),
        ),
      ],
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
      width: 190,
      child: Card(
        elevation: 4,
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, size: 40, color: Colors.blue),
                const SizedBox(height: 12),
                Text(
                  title,
                  style: Theme.of(context).textTheme.titleMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 6),
                Text(
                  description,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey[600],
                      ),
                  textAlign: TextAlign.center,
                ),
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
      await _faceDetector!.initialize(
        model: _detectionModel,
        useCompiledModel: true,
      );
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
        await _faceDetector!.detectFacesFromBytes(bytes, mode: mode);
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
                CompactCheckbox(
                    label: label,
                    value: v,
                    onChanged: (x) => updateState(() => set(x ?? false)));
            Widget fcb(String label, bool v, void Function(bool) set) =>
                CompactCheckbox(
                    label: label,
                    value: v,
                    onChanged: (x) => onSheetFeatureToggle(set, x ?? false));
            Widget col(String label, Color c, void Function(Color) set) =>
                _ColorPickerButton(
                    label: label,
                    color: c,
                    onColorChanged: (x) => updateState(() => set(x)));
            Widget sl(String label, double v, double mn, double mx,
                    void Function(double) set) =>
                CompactSlider(
                    label: label,
                    value: v,
                    min: mn,
                    max: mx,
                    onChanged: (x) => updateState(() => set(x)));

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
                : _ScrollableCentered(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.add_photo_alternate,
                            size: 80, color: Colors.grey[300]),
                        const SizedBox(height: 16),
                        Text(
                          'No image selected',
                          style:
                              TextStyle(fontSize: 18, color: Colors.grey[600]),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Tap the + icon to pick an image',
                          style:
                              TextStyle(fontSize: 14, color: Colors.grey[500]),
                        ),
                      ],
                    ),
                  ),
          ),
          if (hasImage && _totalTimeMs != null)
            Positioned(
              top: 12,
              left: 12,
              child: TimingBadge(
                totalMs: _totalTimeMs!,
                detectionMs: _detectionTimeMs,
                meshMs: _meshTimeMs,
                irisMs: _irisTimeMs,
                meshEnabled: _showMesh,
                irisEnabled: _showIrises,
                meshProcessed: _hasProcessedMesh,
                irisProcessed: _hasProcessedIris,
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

  // Live backend benchmarking: default to CompiledModel, with a one-tap
  // XNNPACK fallback for immediate A/B checks in the camera view.
  bool _useCompiledModel = true;
  PerformanceConfig get _perfConfig => const PerformanceConfig.xnnpack();
  final List<int> _recentInferenceMs = [];
  int _detThisSec = 0;

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
    try {
      await _faceDetector!.initialize(
        model: _detectionModel,
        performanceConfig: _perfConfig,
        useCompiledModel: _useCompiledModel,
      );
    } catch (e) {
      if (!_useCompiledModel) rethrow;
      debugPrint(
        'Live camera CompiledModel init failed; falling back to XNNPACK: $e',
      );
      if (mounted) {
        setState(() => _useCompiledModel = false);
      } else {
        _useCompiledModel = false;
      }
      await _faceDetector?.dispose();
      _faceDetector = FaceDetector();
      await _faceDetector!.initialize(
        model: _detectionModel,
        performanceConfig: _perfConfig,
        useCompiledModel: false,
      );
    }
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
      mode: _detectionMode,
      maxDim: maxDim,
    );
    return (faces: faces, segMask: null);
  }

  Future<void> _toggleAccelerator() async {
    setState(() {
      _useCompiledModel = !_useCompiledModel;
      _recentInferenceMs.clear();
    });
    // ignore: avoid_print
    print('[live-bench] switching backend -> '
        '${_useCompiledModel ? 'compiledmodel' : 'xnnpack'}');
    await _reinitDetectorIsolate();
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
              TextButton(
                onPressed: _toggleAccelerator,
                style: TextButton.styleFrom(
                  minimumSize: const Size(48, 36),
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                ),
                child: Text(
                  _useCompiledModel ? 'CM' : 'XNN',
                  style: const TextStyle(
                    color: Colors.amberAccent,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
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

  Future<void> _initCamera() async {
    try {
      try {
        await _reinitDetectorIsolate();
      } catch (e) {
        debugPrint('Detector isolate init failed (segmentation may be '
            'unavailable): $e');
        _faceDetector = FaceDetector();
        await _faceDetector!.initialize(
          model: _detectionModel,
          performanceConfig: _perfConfig,
          useCompiledModel: false,
        );
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
      imageFormatGroup: ImageFormatGroup
          .yuv420, // prevents JPEG fallback on Android; ignored on desktop
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
      final n = _recentInferenceMs.length;
      final meanMs =
          n == 0 ? 0 : (_recentInferenceMs.reduce((a, b) => a + b) / n).round();
      final backend = _useCompiledModel ? 'compiledmodel' : 'xnnpack';
      // ignore: avoid_print
      print('[live-bench] backend=$backend '
          'cameraFps=$_fps detPerSec=$_detThisSec meanInferMs=$meanMs '
          'lastMs=$_detectionTimeMs faces=${_faces.length} '
          'mode=${_detectionMode.name}');
      _recentInferenceMs.clear();
      _detThisSec = 0;
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
      _recentInferenceMs.add(detectionTime);
      _detThisSec++;

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
    final bool mirrorOverlayHorizontally =
        (Platform.isAndroid && _isFrontCamera) || Platform.isWindows;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          FaceDetectionCameraOverlay(
            cameraPreview: CameraPreview(_cameraController!),
            cameraAspectRatio: cameraAspectRatio,
            displayAspectRatio: displayAspectRatio,
            mirrorHorizontally: mirrorOverlayHorizontally,
            sensorOrientation: _sensorOrientation ?? 0,
            deviceOrientation: deviceOrientation,
            isFrontCamera: _isFrontCamera,
            detectionMode: _detectionMode,
            faces: _faces,
            imageSize: _imageSize,
            segmentationMask: _segmentationMask,
            virtualBackground: _beachBackground,
            showSegmentation: _showSegmentation,
            showVirtualBackground: _showVirtualBackground,
            segmentationColor: _segmentationColor,
            segmentationShowAllClasses:
                _liveSegmentationModel == SegmentationModel.multiclass,
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
                ? const _ScrollableCentered(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text('Initializing segmentation model...'),
                      ],
                    ),
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
                                    classColors: kSegmentationClassColors,
                                  ),
                                ),
                            ],
                          );
                        },
                      )
                    : _ScrollableCentered(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.person_outline,
                                size: 100, color: Colors.blue[200]),
                            const SizedBox(height: 24),
                            const Text(
                              'Pick an image to segment',
                              style:
                                  TextStyle(fontSize: 18, color: Colors.grey),
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

// ─────────────────────────── Video File Screen ────────────────────────────

class VideoFileScreen extends StatefulWidget {
  const VideoFileScreen({super.key});

  @override
  State<VideoFileScreen> createState() => _VideoFileScreenState();
}

class _VideoFileScreenState extends State<VideoFileScreen> {
  FaceDetector? _detector;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _cancelRequested = false;
  String? _errorMessage;
  String? _statusMessage;

  String? _inputPath;
  String? _outputPath;
  int _totalFrames = 0;
  int _processedFrames = 0;
  double _videoFps = 0;
  int _videoWidth = 0;
  int _videoHeight = 0;
  Duration _elapsed = Duration.zero;
  final Stopwatch _wallClock = Stopwatch();

  VideoPlayerController? _playerController;
  bool _playerReady = false;
  String? _playerError;

  bool _smoothingEnabled = true;
  final FaceSmoother _smoother = FaceSmoother(enabled: true);

  // Paint options, mirroring the Still Image screen. Style options (colors,
  // sizes, toggles) are read per frame; detection mode and segmentation are
  // captured when a run starts.
  bool _showBoundingBoxes = true;
  bool _showMesh = true;
  bool _showLandmarks = true;
  bool _showIrises = true;
  bool _showEyeContours = true;
  bool _showEyeMesh = true;
  bool _showLandmarkLabels = false;

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

  bool _showSegmentation = false;
  SegmentationModel _segModel = SegmentationModel.general;
  double _segThreshold = 0.5;
  double _segOpacity = 0.5;
  Color _segMaskColor = const Color(0xFF00FF00);
  SegmentationModel? _segReadyModel;

  FaceDetectionMode _determineMode() {
    if (_showIrises || _showEyeContours || _showEyeMesh) {
      return FaceDetectionMode.full;
    } else if (_showMesh) {
      return FaceDetectionMode.standard;
    } else {
      return FaceDetectionMode.fast;
    }
  }

  /// Makes sure the detector has segmentation ready for [_segModel],
  /// recreating the detector when the model changed (segmentation models
  /// cannot be swapped in place).
  Future<void> _ensureSegmentation() async {
    if (_segReadyModel == _segModel &&
        (_detector?.isSegmentationReady ?? false)) {
      return;
    }
    if (_segReadyModel != null && _segReadyModel != _segModel) {
      final old = _detector;
      _detector = null;
      await old?.dispose();
      _detector = await FaceDetector.create(
        performanceConfig: const PerformanceConfig.xnnpack(),
      );
    }
    await _detector!.initializeSegmentation(
      config: SegmentationConfig(model: _segModel),
    );
    _segReadyModel = _segModel;
  }

  bool get _supportsInAppPlayer {
    if (kIsWeb) return true;
    return Platform.isAndroid || Platform.isIOS || Platform.isMacOS;
  }

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    try {
      final detector = await FaceDetector.create(
        performanceConfig: const PerformanceConfig.xnnpack(),
      );
      if (!mounted) {
        await detector.dispose();
        return;
      }
      setState(() {
        _detector = detector;
        _isInitialized = true;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _errorMessage = 'Failed to initialize detector: $e');
    }
  }

  @override
  void dispose() {
    _cancelRequested = true;
    _detector?.dispose();
    _playerController?.dispose();
    super.dispose();
  }

  Future<void> _disposePlayer() async {
    final c = _playerController;
    _playerController = null;
    _playerReady = false;
    _playerError = null;
    await c?.dispose();
  }

  Future<void> _initPlayerForOutput(String path) async {
    await _disposePlayer();
    if (!_supportsInAppPlayer) return;
    final controller = VideoPlayerController.file(File(path));
    _playerController = controller;
    try {
      await controller.initialize();
      await controller.setLooping(true);
      if (!mounted) {
        await controller.dispose();
        _playerController = null;
        return;
      }
      setState(() => _playerReady = true);
      await controller.play();
    } catch (e) {
      if (!mounted) return;
      setState(() => _playerError = 'Could not load video: $e');
    }
  }

  Future<void> _pickVideo() async {
    const typeGroup = XTypeGroup(
      label: 'Videos',
      extensions: ['mp4', 'mov', 'm4v'],
    );
    final XFile? file = await openFile(acceptedTypeGroups: [typeGroup]);
    if (file == null) return;
    await _processVideo(file.path);
  }

  Future<void> _processVideo(String path) async {
    final inputFile = File(path);
    if (!await inputFile.exists()) {
      setState(() => _errorMessage = 'File does not exist: $path');
      return;
    }

    final FaceDetectionMode mode = _determineMode();
    final bool segEnabled = _showSegmentation;
    final bool segMulticlass = _segModel == SegmentationModel.multiclass;
    if (segEnabled) {
      try {
        await _ensureSegmentation();
      } catch (e) {
        setState(() => _errorMessage = 'Failed to initialize segmentation: $e');
        return;
      }
    }

    final cap = cv.VideoCapture.fromFile(path);
    if (!cap.isOpened) {
      cap.release();
      String hint = '';
      if (Platform.isLinux) {
        hint = '\n\nLinux requires GStreamer plugins. Try:\n'
            '  sudo apt install gstreamer1.0-libav '
            'gstreamer1.0-plugins-good gstreamer1.0-plugins-bad';
      }
      setState(
        () => _errorMessage =
            'Could not open video.\nFormat may not be supported by the OS '
                'video backend.$hint',
      );
      return;
    }

    final fps = cap.get(cv.CAP_PROP_FPS);
    final width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
    final height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
    final total = cap.get(cv.CAP_PROP_FRAME_COUNT).toInt();

    final docs = await getApplicationDocumentsDirectory();
    final outName = 'face_${DateTime.now().millisecondsSinceEpoch}.mp4';
    final outPath = '${docs.path}/$outName';

    final writer = cv.VideoWriter.fromFile(outPath, 'avc1', fps, (
      width,
      height,
    ));
    if (!writer.isOpened) {
      cap.release();
      setState(
        () => _errorMessage =
            'Could not open writer for $outPath. The "avc1" (H.264) codec '
                'may not be available on this OS backend.',
      );
      return;
    }

    if (!mounted) {
      cap.release();
      writer.release();
      return;
    }
    await _disposePlayer();
    setState(() {
      _inputPath = path;
      _outputPath = outPath;
      _videoFps = fps;
      _videoWidth = width;
      _videoHeight = height;
      _totalFrames = total;
      _processedFrames = 0;
      _isProcessing = true;
      _cancelRequested = false;
      _errorMessage = null;
      _statusMessage = 'Processing...';
      _elapsed = Duration.zero;
    });
    _wallClock
      ..reset()
      ..start();

    cv.Mat? frame;
    _smoother.reset();
    try {
      int idx = 0;
      while (mounted && !_cancelRequested) {
        final result = cap.read(m: frame);
        final ok = result.$1;
        frame = result.$2;
        if (!ok || frame.isEmpty) break;

        List<Face> raw;
        SegmentationMask? segMask;
        if (segEnabled) {
          final result = await _detector!.detectFacesWithSegmentationFromMat(
            frame,
            mode: mode,
          );
          raw = result.faces;
          segMask = result.segmentationMask;
        } else {
          raw = await _detector!.detectFacesFromMat(frame, mode: mode);
        }
        final double tSec = fps > 0 ? idx / fps : idx / 30.0;
        final List<Face> faces = _smoother.apply(raw, tSec);
        if (segMask != null) {
          _blendMaskOnMat(frame, segMask, multiclass: segMulticlass);
        }
        _drawFacesOnMat(frame, faces);
        writer.write(frame);

        idx++;
        if (idx % 4 == 0) {
          if (!mounted) break;
          setState(() {
            _processedFrames = idx;
            _elapsed = _wallClock.elapsed;
          });
          await Future<void>.delayed(Duration.zero);
        }
      }
      if (mounted) {
        setState(() {
          _processedFrames = idx;
          _elapsed = _wallClock.elapsed;
          _statusMessage = _cancelRequested
              ? 'Cancelled after $idx frames.'
              : 'Done. Wrote $idx frames to:\n$outPath';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _errorMessage = 'Error during processing: $e');
      }
    } finally {
      _wallClock.stop();
      cap.release();
      writer.release();
      frame?.dispose();
      if (mounted) setState(() => _isProcessing = false);
      if (mounted && !_cancelRequested && _outputPath != null) {
        await _initPlayerForOutput(_outputPath!);
      }
    }
  }

  /// Converts a Flutter [Color] to an OpenCV BGR scalar (alpha ignored).
  cv.Scalar _bgr(Color c) => cv.Scalar(
        (c.b * 255).roundToDouble(),
        (c.g * 255).roundToDouble(),
        (c.r * 255).roundToDouble(),
      );

  /// Burns a segmentation tint into [frame], mirroring what
  /// SegmentationMaskPainter draws on screen: pixels whose probability (or
  /// winning multiclass probability) clears [_segThreshold] are blended with
  /// the overlay color at [_segOpacity].
  void _blendMaskOnMat(cv.Mat frame, SegmentationMask mask,
      {required bool multiclass}) {
    final v = maskValidRegion(mask);
    final int vw = v.x1 - v.x0;
    final int vh = v.y1 - v.y0;
    if (vw <= 0 || vh <= 0) return;

    final colorData = Uint8List(vw * vh * 3);
    final selData = Uint8List(vw * vh);

    if (multiclass && mask is MulticlassSegmentationMask) {
      final classMasks = List.generate(6, (i) => mask.classMask(i));
      final classBgr = [
        for (final c in kSegmentationClassColors)
          [(c.b * 255).round(), (c.g * 255).round(), (c.r * 255).round()],
      ];
      for (int y = 0; y < vh; y++) {
        for (int x = 0; x < vw; x++) {
          final idx = (y + v.y0) * mask.width + (x + v.x0);
          int winner = 0;
          double maxProb = classMasks[0][idx];
          for (int c = 1; c < 6; c++) {
            if (classMasks[c][idx] > maxProb) {
              maxProb = classMasks[c][idx];
              winner = c;
            }
          }
          if (maxProb >= _segThreshold) {
            final o = y * vw + x;
            selData[o] = 255;
            colorData[o * 3] = classBgr[winner][0];
            colorData[o * 3 + 1] = classBgr[winner][1];
            colorData[o * 3 + 2] = classBgr[winner][2];
          }
        }
      }
    } else {
      final int cb = (_segMaskColor.b * 255).round();
      final int cg = (_segMaskColor.g * 255).round();
      final int cr = (_segMaskColor.r * 255).round();
      for (int y = 0; y < vh; y++) {
        for (int x = 0; x < vw; x++) {
          if (mask.at(x + v.x0, y + v.y0) >= _segThreshold) {
            final o = y * vw + x;
            selData[o] = 255;
            colorData[o * 3] = cb;
            colorData[o * 3 + 1] = cg;
            colorData[o * 3 + 2] = cr;
          }
        }
      }
    }

    // Write the packed bytes straight into each Mat's native buffer. Mat.fromList
    // takes a List<num> and copies element-by-element; create + data.setAll is a
    // single memcpy and byte-identical for tightly packed data.
    final colorSmall =
        cv.Mat.create(rows: vh, cols: vw, type: cv.MatType.CV_8UC3);
    colorSmall.data.setAll(0, colorData);
    final selSmall =
        cv.Mat.create(rows: vh, cols: vw, type: cv.MatType.CV_8UC1);
    selSmall.data.setAll(0, selData);
    final colorBig = cv.resize(colorSmall, (frame.cols, frame.rows),
        interpolation: cv.INTER_NEAREST);
    final selBig = cv.resize(selSmall, (frame.cols, frame.rows),
        interpolation: cv.INTER_NEAREST);
    final blended =
        cv.addWeighted(frame, 1.0 - _segOpacity, colorBig, _segOpacity, 0);
    blended.copyTo(frame, mask: selBig);
    for (final m in [colorSmall, selSmall, colorBig, selBig, blended]) {
      m.dispose();
    }
  }

  /// Draws the enabled overlays onto [mat] with OpenCV, mirroring what
  /// DetectionsPainter draws on screen for the Still Image mode.
  void _drawFacesOnMat(cv.Mat mat, List<Face> faces) {
    if (faces.isEmpty) return;
    final black = cv.Scalar(0, 0, 0);

    final w = mat.cols;
    final h = mat.rows;

    for (final face in faces) {
      if (_showBoundingBoxes) {
        final bboxColor = _bgr(_boundingBoxColor);
        final bb = face.boundingBox;
        final l = bb.left.toInt().clamp(0, w - 1);
        final t = bb.top.toInt().clamp(0, h - 1);
        final r = bb.right.toInt().clamp(0, w - 1);
        final b = bb.bottom.toInt().clamp(0, h - 1);
        cv.rectangle(
          mat,
          cv.Rect(l, t, (r - l).clamp(1, w), (b - t).clamp(1, h)),
          bboxColor,
          thickness: math.max(1, _boundingBoxThickness.round()),
        );

        final label =
            'Face ${(face.detectionData.score * 100).toStringAsFixed(0)}%';
        final (sz, _) = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2);
        final labelTop = (t - sz.height - 8).clamp(0, h - 1);
        final labelW = (sz.width + 8).clamp(1, w - l);
        final labelH = (sz.height + 8).clamp(1, h - labelTop);
        cv.rectangle(
          mat,
          cv.Rect(l, labelTop, labelW, labelH),
          bboxColor,
          thickness: -1,
        );
        cv.putText(
          mat,
          label,
          cv.Point(l + 4, labelTop + sz.height + 2),
          cv.FONT_HERSHEY_SIMPLEX,
          0.6,
          black,
          thickness: 2,
        );
      }

      if (_showLandmarks) {
        final landmarkColor = _bgr(_landmarkColor);
        for (final entry in face.landmarks.toMap().entries) {
          final p = entry.value;
          final center = cv.Point(p.x.toInt(), p.y.toInt());
          cv.circle(
            mat,
            center,
            math.max(1, _landmarkSize.round()),
            landmarkColor,
            thickness: -1,
          );
          if (_showLandmarkLabels) {
            cv.putText(
              mat,
              entry.key.name,
              cv.Point(center.x + 6, center.y - 6),
              cv.FONT_HERSHEY_SIMPLEX,
              0.5,
              landmarkColor,
              thickness: 1,
            );
          }
        }
      }

      if (_showMesh) {
        final mesh = face.mesh;
        if (mesh != null) {
          final meshColor = _bgr(_meshColor);
          final radius =
              math.max(1, (_meshSize + math.sqrt(w * h) / 1000.0).round());
          for (final pt in mesh.points) {
            cv.circle(
              mat,
              cv.Point(pt.x.toInt(), pt.y.toInt()),
              radius,
              meshColor,
              thickness: -1,
            );
          }
        }
      }

      if (_showIrises || _showEyeContours || _showEyeMesh) {
        final eyePair = face.eyes;
        if (eyePair == null) continue;
        for (final iris in [eyePair.leftEye, eyePair.rightEye]) {
          if (iris == null) continue;

          if (_showIrises) {
            double minX = double.infinity, minY = double.infinity;
            double maxX = double.negativeInfinity;
            double maxY = double.negativeInfinity;
            for (final p in [iris.irisCenter, ...iris.irisContour]) {
              if (p.x < minX) minX = p.x;
              if (p.x > maxX) maxX = p.x;
              if (p.y < minY) minY = p.y;
              if (p.y > maxY) maxY = p.y;
            }
            if (maxX > minX && maxY > minY) {
              cv.ellipse(
                mat,
                cv.Point(
                    ((minX + maxX) / 2).round(), ((minY + maxY) / 2).round()),
                cv.Point(((maxX - minX) / 2).round().clamp(1, w),
                    ((maxY - minY) / 2).round().clamp(1, h)),
                0,
                0,
                360,
                _bgr(_irisColor),
                thickness: 2,
              );
            }
          }

          if (_showEyeContours && iris.mesh.isNotEmpty) {
            final contourColor = _bgr(_eyeContourColor);
            final contour = iris.contour;
            for (final connection in eyeLandmarkConnections) {
              if (connection[0] < contour.length &&
                  connection[1] < contour.length) {
                final p1 = contour[connection[0]];
                final p2 = contour[connection[1]];
                cv.line(
                  mat,
                  cv.Point(p1.x.toInt(), p1.y.toInt()),
                  cv.Point(p2.x.toInt(), p2.y.toInt()),
                  contourColor,
                  thickness: 2,
                );
              }
            }
          }

          if (_showEyeMesh && iris.mesh.isNotEmpty) {
            final eyeMeshColor = _bgr(_eyeMeshColor);
            final radius = math.max(1, _eyeMeshSize.round());
            for (final pt in iris.mesh) {
              cv.circle(
                mat,
                cv.Point(pt.x.toInt(), pt.y.toInt()),
                radius,
                eyeMeshColor,
                thickness: -1,
              );
            }
          }
        }
      }
    }
  }

  Future<void> _openOutputFile() async {
    final path = _outputPath;
    if (path == null) return;
    try {
      if (Platform.isMacOS) {
        await Process.run('open', [path]);
      } else if (Platform.isLinux) {
        await Process.run('xdg-open', [path]);
      } else if (Platform.isWindows) {
        await Process.run('cmd', ['/c', 'start', '', path]);
      } else if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Saved to: $path')));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Could not open: $e')));
      }
    }
  }

  void _showVideoSettings() {
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

            Widget cb(String label, bool v, void Function(bool) set) =>
                CompactCheckbox(
                    label: label,
                    value: v,
                    onChanged: (x) => updateState(() => set(x ?? false)));
            Widget col(String label, Color c, void Function(Color) set) =>
                _ColorPickerButton(
                    label: label,
                    color: c,
                    onColorChanged: (x) => updateState(() => set(x)));
            Widget sl(String label, double v, double mn, double mx,
                    void Function(double) set) =>
                CompactSlider(
                    label: label,
                    value: v,
                    min: mn,
                    max: mx,
                    onChanged: (x) => updateState(() => set(x)));

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
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 4),
                          child: Text(
                            'Detail level and segmentation apply when '
                            'processing starts; styles apply to the '
                            'remaining frames.',
                            style: TextStyle(fontSize: 12, color: Colors.grey),
                          ),
                        ),
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
                                cb('Mesh', _showMesh, (v) => _showMesh = v),
                                cb('Landmarks', _showLandmarks,
                                    (v) => _showLandmarks = v),
                                cb('Irises', _showIrises,
                                    (v) => _showIrises = v),
                                cb('Eye Contour', _showEyeContours,
                                    (v) => _showEyeContours = v),
                                cb('Eye Mesh', _showEyeMesh,
                                    (v) => _showEyeMesh = v),
                                cb('Landmark Labels', _showLandmarkLabels,
                                    (v) => _showLandmarkLabels = v),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Segmentation',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          initiallyExpanded: _showSegmentation,
                          children: [
                            SwitchListTile(
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                              title: const Text('Segmentation overlay'),
                              subtitle: const Text(
                                  'Burn a segmentation tint into the output'),
                              value: _showSegmentation,
                              onChanged: (v) =>
                                  updateState(() => _showSegmentation = v),
                            ),
                            if (_showSegmentation) ...[
                              Wrap(
                                spacing: 8,
                                children: [
                                  for (final (m, label) in const [
                                    (SegmentationModel.general, 'General'),
                                    (SegmentationModel.landscape, 'Landscape'),
                                    (
                                      SegmentationModel.multiclass,
                                      'Multiclass (6 classes)'
                                    ),
                                  ])
                                    ChoiceChip(
                                      label: Text(label),
                                      selected: _segModel == m,
                                      onSelected: (sel) {
                                        if (sel) {
                                          updateState(() => _segModel = m);
                                        }
                                      },
                                    ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              sl('Threshold', _segThreshold, 0.0, 1.0,
                                  (v) => _segThreshold = v),
                              sl('Opacity', _segOpacity, 0.1, 1.0,
                                  (v) => _segOpacity = v),
                              if (_segModel != SegmentationModel.multiclass)
                                Align(
                                  alignment: Alignment.centerLeft,
                                  child: col('Mask Color', _segMaskColor,
                                      (c) => _segMaskColor = c),
                                ),
                            ],
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video File - Face Detection'),
        actions: [
          IconButton(
            onPressed: _showVideoSettings,
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized && !_isProcessing
          ? FloatingActionButton.extended(
              onPressed: _pickVideo,
              icon: const Icon(Icons.video_file),
              label: const Text('Pick Video'),
            )
          : (_isProcessing
              ? FloatingActionButton.extended(
                  onPressed: () => setState(() => _cancelRequested = true),
                  icon: const Icon(Icons.cancel),
                  label: const Text('Cancel'),
                  backgroundColor: Colors.red,
                )
              : null),
    );
  }

  Widget _buildBody() {
    if (!_isInitialized && _errorMessage == null) {
      return const _ScrollableCentered(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing detector...'),
          ],
        ),
      );
    }

    final progress = (_totalFrames > 0)
        ? (_processedFrames / _totalFrames).clamp(0.0, 1.0)
        : 0.0;
    final processedFps = (_elapsed.inMilliseconds > 0)
        ? _processedFrames * 1000.0 / _elapsed.inMilliseconds
        : 0.0;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_errorMessage != null)
            Card(
              color: Colors.red[50],
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    const Icon(Icons.error_outline, color: Colors.red),
                    const SizedBox(width: 12),
                    Expanded(child: Text(_errorMessage!)),
                  ],
                ),
              ),
            ),
          if (_inputPath != null) ...[
            const SizedBox(height: 8),
            _infoRow('Input', _inputPath!),
            if (_videoWidth > 0)
              _infoRow(
                'Source',
                '$_videoWidth×$_videoHeight @ '
                    '${_videoFps.toStringAsFixed(2)} fps · '
                    '$_totalFrames frames',
              ),
          ],
          if (!_isProcessing)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              title: const Text('Smoothing (One-Euro filter)'),
              subtitle: Text(
                _smoothingEnabled
                    ? 'On: mesh points filtered across frames'
                    : 'Off: raw per-frame detections',
              ),
              value: _smoothingEnabled,
              onChanged: (v) {
                setState(() {
                  _smoothingEnabled = v;
                  _smoother.enabled = v;
                  _smoother.reset();
                });
              },
            ),
          if (!_isProcessing && _inputPath != null) ...[
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerLeft,
              child: OutlinedButton.icon(
                onPressed: () => _processVideo(_inputPath!),
                icon: const Icon(Icons.refresh),
                label: const Text('Re-run with current settings'),
              ),
            ),
          ],
          const SizedBox(height: 16),
          if (_isProcessing) ...[
            LinearProgressIndicator(value: _totalFrames > 0 ? progress : null),
            const SizedBox(height: 8),
            Text(
              'Frame $_processedFrames / $_totalFrames · '
              '${(progress * 100).toStringAsFixed(1)}% · '
              '${processedFps.toStringAsFixed(1)} fps · '
              'elapsed ${_formatDuration(_elapsed)}',
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ] else if (_outputPath != null && _statusMessage != null)
            VideoResultCard(
              statusMessage: _statusMessage!,
              summary: 'Total time: ${_formatDuration(_elapsed)} '
                  '(${processedFps.toStringAsFixed(1)} fps avg)',
              preview: _buildOutputPreview(),
              onOpenOutput:
                  (Platform.isMacOS || Platform.isLinux || Platform.isWindows)
                      ? _openOutputFile
                      : null,
            )
          else
            Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(height: 32),
                  Icon(
                    Icons.movie_creation_outlined,
                    size: 96,
                    color: Colors.grey[400],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Pick an MP4 to run face detection on every frame.\n'
                    'Output is written to the app documents directory.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey[700]),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 70,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
          Expanded(child: SelectableText(value)),
        ],
      ),
    );
  }

  String _formatDuration(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    if (d.inHours > 0) {
      return '${d.inHours}:$m:$s';
    }
    return '$m:$s';
  }

  Widget _buildOutputPreview() {
    if (!_supportsInAppPlayer) return const SizedBox.shrink();
    if (_playerError != null) {
      return Text(_playerError!, style: const TextStyle(color: Colors.red));
    }
    final controller = _playerController;
    if (controller == null || !_playerReady) {
      return const SizedBox(
        height: 64,
        child: Center(
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              SizedBox(width: 12),
              Flexible(child: Text('Loading preview...')),
            ],
          ),
        ),
      );
    }
    return _OutputVideoPlayer(controller: controller);
  }
}

// ─────────────────────────── Video Result Card ────────────────────────────

/// Result card shown after a video finishes processing.
///
/// Built so it cannot overflow regardless of viewport: it contains no
/// fixed-height content, every row child that can vary in width is flexible,
/// and the preview passed in is expected to size itself to the incoming
/// constraints (see [VideoPlayerChrome]). The card relies on the scrollable
/// body it sits in for vertical space.
class VideoResultCard extends StatelessWidget {
  final String statusMessage;
  final String summary;
  final Widget preview;
  final VoidCallback? onOpenOutput;

  const VideoResultCard({
    super.key,
    required this.statusMessage,
    required this.summary,
    required this.preview,
    this.onOpenOutput,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.green),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    statusMessage,
                    style: const TextStyle(fontWeight: FontWeight.w500),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(summary),
            const SizedBox(height: 12),
            preview,
            if (onOpenOutput != null) ...[
              const SizedBox(height: 12),
              Align(
                alignment: Alignment.centerLeft,
                child: ElevatedButton.icon(
                  onPressed: onOpenOutput,
                  icon: const Icon(Icons.play_circle_outline),
                  label: const Text('Open output video'),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Layout chrome for the output video preview: the video is aspect-fitted
/// within the available width and capped to a fraction of the screen height
/// (so portrait videos cannot blow the layout up), and the controls row
/// drops the time label instead of overflowing when width gets tight.
class VideoPlayerChrome extends StatelessWidget {
  final double aspectRatio;
  final Widget video;
  final Widget progress;
  final bool isPlaying;
  final String positionLabel;
  final VoidCallback onTogglePlay;

  const VideoPlayerChrome({
    super.key,
    required this.aspectRatio,
    required this.video,
    required this.progress,
    required this.isPlaying,
    required this.positionLabel,
    required this.onTogglePlay,
  });

  @override
  Widget build(BuildContext context) {
    final double maxPreviewHeight =
        math.max(120.0, MediaQuery.sizeOf(context).height * 0.45);
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: maxPreviewHeight),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: AspectRatio(
                aspectRatio: aspectRatio,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Container(color: Colors.black),
                    video,
                  ],
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),
        LayoutBuilder(
          builder: (context, constraints) {
            final bool showTime = constraints.maxWidth >= 180;
            return Row(
              children: [
                IconButton(
                  icon: Icon(isPlaying ? Icons.pause : Icons.play_arrow),
                  onPressed: onTogglePlay,
                ),
                Expanded(child: progress),
                if (showTime) ...[
                  const SizedBox(width: 8),
                  Text(
                    positionLabel,
                    style: const TextStyle(
                      fontFeatures: [FontFeature.tabularFigures()],
                    ),
                  ),
                ],
              ],
            );
          },
        ),
      ],
    );
  }
}

// ─────────────────────────── Output Video Player ──────────────────────────

class _OutputVideoPlayer extends StatefulWidget {
  final VideoPlayerController controller;
  const _OutputVideoPlayer({required this.controller});

  @override
  State<_OutputVideoPlayer> createState() => _OutputVideoPlayerState();
}

class _OutputVideoPlayerState extends State<_OutputVideoPlayer> {
  void _onTick() {
    if (mounted) setState(() {});
  }

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onTick);
  }

  @override
  void didUpdateWidget(covariant _OutputVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      oldWidget.controller.removeListener(_onTick);
      widget.controller.addListener(_onTick);
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onTick);
    super.dispose();
  }

  String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.controller;
    final value = c.value;
    return VideoPlayerChrome(
      aspectRatio: value.aspectRatio == 0 ? 16 / 9 : value.aspectRatio,
      video: VideoPlayer(c),
      progress: VideoProgressIndicator(
        c,
        allowScrubbing: true,
        padding: const EdgeInsets.symmetric(vertical: 12),
      ),
      isPlaying: value.isPlaying,
      positionLabel: '${_fmt(value.position)} / ${_fmt(value.duration)}',
      onTogglePlay: () {
        if (value.isPlaying) {
          c.pause();
        } else {
          c.play();
        }
      },
    );
  }
}

// ─────────────────────────── Face Smoother ────────────────────────────────

class _FaceTrack {
  final Map<int, List<OneEuroFilter>> filters = {};
  double lastLeft = 0, lastTop = 0, lastRight = 0, lastBottom = 0;
  bool hasBox = false;
  int missedFrames = 0;
}

class FaceSmoother {
  bool enabled;
  static const int _maxMissed = 5;
  static const double _minIou = 0.2;
  final List<_FaceTrack> _tracks = [];

  FaceSmoother({this.enabled = true});

  void reset() {
    _tracks.clear();
  }

  List<Face> apply(List<Face> faces, double tSec) {
    if (!enabled || faces.isEmpty) {
      if (!enabled) _tracks.clear();
      return faces;
    }

    final unmatched = List<int>.generate(_tracks.length, (i) => i);
    final matchedTrack = List<int?>.filled(faces.length, null);

    for (int p = 0; p < faces.length; p++) {
      double bestIou = _minIou;
      int bestT = -1;
      for (final t in unmatched) {
        if (!_tracks[t].hasBox) continue;
        final iou = _iou(faces[p], _tracks[t]);
        if (iou > bestIou) {
          bestIou = iou;
          bestT = t;
        }
      }
      if (bestT >= 0) {
        matchedTrack[p] = bestT;
        unmatched.remove(bestT);
      }
    }

    final out = <Face>[];
    for (int p = 0; p < faces.length; p++) {
      _FaceTrack track;
      if (matchedTrack[p] != null) {
        track = _tracks[matchedTrack[p]!];
        track.missedFrames = 0;
      } else {
        track = _FaceTrack();
        _tracks.add(track);
      }
      final bb = faces[p].boundingBox;
      track.lastLeft = bb.left;
      track.lastTop = bb.top;
      track.lastRight = bb.right;
      track.lastBottom = bb.bottom;
      track.hasBox = true;
      out.add(_smoothFace(faces[p], track, tSec));
    }

    for (final t in unmatched) {
      _tracks[t].missedFrames++;
    }
    _tracks.removeWhere((t) => t.missedFrames > _maxMissed);

    return out;
  }

  Face _smoothFace(Face face, _FaceTrack track, double tSec) {
    final mesh = face.mesh;
    if (mesh == null) return face;

    final smoothedPoints = <Point>[];
    for (int i = 0; i < mesh.points.length; i++) {
      final pt = mesh.points[i];
      var fs = track.filters[i];
      if (fs == null) {
        fs = [
          OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
          OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
        ];
        track.filters[i] = fs;
      }
      smoothedPoints.add(Point(
        fs[0].filter(pt.x, tSec),
        fs[1].filter(pt.y, tSec),
      ));
    }

    final smoothedIrises = <Point>[];
    for (int i = 0; i < face.irisPoints.length; i++) {
      final pt = face.irisPoints[i];
      final key = mesh.points.length + i;
      var fs = track.filters[key];
      if (fs == null) {
        fs = [
          OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
          OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
        ];
        track.filters[key] = fs;
      }
      smoothedIrises.add(Point(
        fs[0].filter(pt.x, tSec),
        fs[1].filter(pt.y, tSec),
      ));
    }

    return Face(
      detection: face.detectionData,
      mesh: FaceMesh(smoothedPoints),
      irises: smoothedIrises,
      originalSize: face.originalSize,
    );
  }

  double _iou(Face a, _FaceTrack b) {
    final box = a.boundingBox;
    final l = math.max(box.left, b.lastLeft);
    final t = math.max(box.top, b.lastTop);
    final r = math.min(box.right, b.lastRight);
    final bo = math.min(box.bottom, b.lastBottom);
    final iw = math.max(0.0, r - l);
    final ih = math.max(0.0, bo - t);
    final inter = iw * ih;
    final aa = math.max(0.0, box.right - box.left) *
        math.max(0.0, box.bottom - box.top);
    final bb = math.max(0.0, b.lastRight - b.lastLeft) *
        math.max(0.0, b.lastBottom - b.lastTop);
    final union = aa + bb - inter;
    if (union <= 0) return 0;
    return inter / union;
  }
}

/// Holds the picked image and its computed embedding for one comparison slot.
class _FaceSlot {
  Uint8List? bytes;
  Float32List? embedding;
  bool isLoading = false;
}

/// Verdict styling derived from the raw cosine similarity score.
class _Verdict {
  final String label;
  final Color color;
  final IconData icon;
  const _Verdict(this.label, this.color, this.icon);
}

/// Picks two photos, extracts a face embedding from each, and reports how
/// similar the two faces are (MobileFaceNet embeddings + cosine similarity).
class EmbeddingsScreen extends StatefulWidget {
  const EmbeddingsScreen({super.key});
  @override
  State<EmbeddingsScreen> createState() => _EmbeddingsScreenState();
}

class _EmbeddingsScreenState extends State<EmbeddingsScreen> {
  FaceDetector? _detector;
  bool _initializing = true;

  final _FaceSlot _a = _FaceSlot();
  final _FaceSlot _b = _FaceSlot();

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      _detector = FaceDetector();
      await _detector!.initialize(
        model: FaceDetectionModel.backCamera,
        useCompiledModel: true,
      );
    } catch (_) {}
    if (mounted) setState(() => _initializing = false);
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  Future<void> _pickFor(_FaceSlot slot) async {
    if (_detector == null || !_detector!.isReady) return;

    final ImagePicker picker = ImagePicker();
    final XFile? picked =
        await picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
    if (picked == null) return;

    setState(() {
      slot.bytes = null;
      slot.embedding = null;
      slot.isLoading = true;
    });

    try {
      final Uint8List bytes = await picked.readAsBytes();
      // Only the BlazeFace eye keypoints are needed for embedding alignment,
      // so the lightweight `fast` detection mode is enough here.
      final List<Face> faces = await _detector!
          .detectFacesFromBytes(bytes, mode: FaceDetectionMode.fast);

      Float32List? embedding;
      if (faces.isNotEmpty) {
        final Face face = _largestFace(faces);
        embedding = await _detector!.getFaceEmbedding(face, bytes);
      }

      if (!mounted) return;
      setState(() {
        slot.bytes = bytes;
        slot.embedding = embedding;
        slot.isLoading = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        slot.embedding = null;
        slot.isLoading = false;
      });
    }
  }

  Face _largestFace(List<Face> faces) {
    Face best = faces.first;
    double bestArea = _faceArea(best);
    for (final Face f in faces.skip(1)) {
      final double a = _faceArea(f);
      if (a > bestArea) {
        best = f;
        bestArea = a;
      }
    }
    return best;
  }

  double _faceArea(Face f) {
    final box = f.boundingBox;
    return math.max(0.0, box.right - box.left) *
        math.max(0.0, box.bottom - box.top);
  }

  void _reset() {
    setState(() {
      for (final _FaceSlot slot in [_a, _b]) {
        slot.bytes = null;
        slot.embedding = null;
        slot.isLoading = false;
      }
    });
  }

  double? get _cosine {
    final Float32List? a = _a.embedding;
    final Float32List? b = _b.embedding;
    if (a == null || b == null) return null;
    return FaceDetector.compareFaces(a, b);
  }

  /// Maps the raw cosine score (where ~0.6 already means "same person") onto a
  /// more intuitive 0-100% scale via piecewise-linear interpolation.
  double _calibratedPercent(double cosine) {
    const List<List<double>> anchors = [
      [0.0, 0.0],
      [0.3, 45.0],
      [0.5, 72.0],
      [0.65, 86.0],
      [0.8, 95.0],
      [1.0, 100.0],
    ];
    if (cosine <= anchors.first[0]) return 0.0;
    if (cosine >= anchors.last[0]) return 100.0;
    for (int i = 1; i < anchors.length; i++) {
      if (cosine <= anchors[i][0]) {
        final double c0 = anchors[i - 1][0];
        final double p0 = anchors[i - 1][1];
        final double c1 = anchors[i][0];
        final double p1 = anchors[i][1];
        final double t = (cosine - c0) / (c1 - c0);
        return p0 + t * (p1 - p0);
      }
    }
    return 100.0;
  }

  _Verdict _verdictFor(double cosine) {
    if (cosine >= 0.6) {
      return const _Verdict('Same person', Color(0xFF2E7D32), Icons.verified);
    }
    if (cosine >= 0.5) {
      return const _Verdict('Likely the same person', Color(0xFF558B2F),
          Icons.thumb_up_alt_outlined);
    }
    if (cosine >= 0.3) {
      return const _Verdict('Uncertain', Color(0xFFEF6C00), Icons.help_outline);
    }
    return const _Verdict(
        'Different people', Color(0xFFC62828), Icons.cancel_outlined);
  }

  @override
  Widget build(BuildContext context) {
    final bool anyPicked = _a.bytes != null || _b.bytes != null;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Recognition'),
        actions: [
          IconButton(
            tooltip: 'Reset',
            onPressed: anyPicked ? _reset : null,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: SafeArea(
        child: _initializing
            ? const Center(child: CircularProgressIndicator())
            : LayoutBuilder(
                builder: (context, constraints) {
                  final bool wide = constraints.maxWidth >= 640;
                  final Widget slotA = _buildSlot(context, 'Image A', _a);
                  final Widget slotB = _buildSlot(context, 'Image B', _b);
                  final Widget slots = wide
                      ? Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(child: slotA),
                            const SizedBox(width: 16),
                            Expanded(child: slotB),
                          ],
                        )
                      : Column(
                          children: [
                            slotA,
                            const SizedBox(height: 16),
                            slotB,
                          ],
                        );

                  return SingleChildScrollView(
                    padding: const EdgeInsets.all(16),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 820),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            slots,
                            const SizedBox(height: 20),
                            _buildResult(context),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
      ),
    );
  }

  Widget _buildSlot(BuildContext context, String label, _FaceSlot slot) {
    return Card(
      elevation: 2,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: slot.isLoading ? null : () => _pickFor(slot),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            AspectRatio(
              aspectRatio: 4 / 3,
              child: Container(
                width: double.infinity,
                color: Colors.grey.shade100,
                child: _slotPreview(slot),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      label,
                      style: Theme.of(context).textTheme.titleSmall,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 8),
                  _slotStatus(slot),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _slotPreview(_FaceSlot slot) {
    if (slot.isLoading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (slot.bytes == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.add_a_photo_outlined,
                size: 40, color: Colors.grey.shade400),
            const SizedBox(height: 8),
            Text('Tap to pick', style: TextStyle(color: Colors.grey.shade600)),
          ],
        ),
      );
    }
    return Image.memory(
      slot.bytes!,
      fit: BoxFit.contain,
      width: double.infinity,
    );
  }

  Widget _slotStatus(_FaceSlot slot) {
    if (slot.isLoading) {
      return const SizedBox(
        width: 16,
        height: 16,
        child: CircularProgressIndicator(strokeWidth: 2),
      );
    }
    if (slot.bytes == null) return const SizedBox.shrink();
    if (slot.embedding != null) {
      return _statusChip(Icons.check_circle, 'Face found', Colors.green);
    }
    return _statusChip(Icons.error_outline, 'No face', Colors.orange);
  }

  Widget _statusChip(IconData icon, String text, MaterialColor color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.shade50,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: color.shade700),
          const SizedBox(width: 4),
          Text(text, style: TextStyle(fontSize: 12, color: color.shade700)),
        ],
      ),
    );
  }

  Widget _buildResult(BuildContext context) {
    final double? cosine = _cosine;
    final bool processing = _a.isLoading || _b.isLoading;
    final bool bothPicked = _a.bytes != null && _b.bytes != null;

    Widget content;
    if (cosine != null) {
      final double pct = _calibratedPercent(cosine);
      final _Verdict v = _verdictFor(cosine);
      content = Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FittedBox(
            fit: BoxFit.scaleDown,
            alignment: Alignment.centerLeft,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text(
                  '${pct.round()}%',
                  style: Theme.of(context).textTheme.displaySmall?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: v.color,
                      ),
                ),
                const SizedBox(width: 8),
                Text(
                  'match',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: Colors.grey[600],
                      ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: (pct / 100).clamp(0.0, 1.0),
              minHeight: 10,
              backgroundColor: Colors.grey.shade200,
              valueColor: AlwaysStoppedAnimation<Color>(v.color),
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Icon(v.icon, color: v.color, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  v.label,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: v.color,
                        fontWeight: FontWeight.w600,
                      ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            'Cosine similarity: ${cosine.toStringAsFixed(2)}',
            style: Theme.of(context)
                .textTheme
                .bodySmall
                ?.copyWith(color: Colors.grey[600]),
          ),
        ],
      );
    } else if (processing) {
      content = Row(
        children: const [
          SizedBox(
            width: 20,
            height: 20,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          SizedBox(width: 12),
          Expanded(child: Text('Analyzing faces...')),
        ],
      );
    } else {
      final String msg = !bothPicked
          ? 'Pick a photo for both Image A and Image B to compare faces.'
          : 'A clear face is needed in both photos to compare them.';
      content = Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, color: Colors.grey[500], size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              msg,
              style: TextStyle(color: Colors.grey[700]),
            ),
          ),
        ],
      );
    }

    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: content,
      ),
    );
  }
}
