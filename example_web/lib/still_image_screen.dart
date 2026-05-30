// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui_web' as ui_web;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

class StillImageScreen extends StatefulWidget {
  const StillImageScreen({super.key});

  @override
  State<StillImageScreen> createState() => _StillImageScreenState();
}

class _StillImageScreenState extends State<StillImageScreen> {
  // ---- Detector lifecycle -----------------------------------------------
  String _status = 'Initializing models...';
  Uint8List? _pickedBytes;
  ImageProvider? _preview;
  FaceDetector? _detector;
  bool _isModelReady = false;
  web.HTMLCanvasElement? _displayCanvas;
  bool _hasAnnotation = false;

  // ---- Detection mode + model ------------------------------------------
  FaceDetectionMode _mode = FaceDetectionMode.full;
  FaceDetectionModel _model = FaceDetectionModel.backCamera;

  // ---- Display toggles (from the native still-image example) -----------
  bool _showBoundingBoxes = true;
  bool _showMesh = true;
  bool _showLandmarks = true;
  bool _showIrises = true;
  bool _showEyeContours = true;
  bool _showEyeMesh = true;
  bool _showLandmarkLabels = false;

  // ---- Colors ----------------------------------------------------------
  Color _boundingBoxColor = const Color(0xFF00FFCC);
  Color _landmarkColor = const Color(0xFF89CFF0);
  Color _meshColor = const Color(0xFFF4C2C2);
  Color _irisColor = const Color(0xFF22AAFF);
  Color _eyeContourColor = const Color(0xFF22AAFF);
  Color _eyeMeshColor = const Color(0xFFFFAA22);

  // ---- Sizes -----------------------------------------------------------
  double _boundingBoxThickness = 2.0;
  double _landmarkSize = 3.0;
  double _meshSize = 1.25;
  double _eyeMeshSize = 0.8;

  // ---- Segmentation ----------------------------------------------------
  bool _showSegmentation = false;
  SegmentationModel _segmentationModel = SegmentationModel.general;
  double _segmentationThreshold = 0.5;
  bool _showMaskOnly = false;
  bool _showBinaryMask = true;
  Color _maskColor = const Color(0x8800FF00);
  int? _multiclassClassIndex;

  // ---- LiteRT settings -------------------------------------------------
  bool _useLiteRt = true;
  String _liteRtAccelerator = 'auto';

  static bool _viewFactoryRegistered = false;
  Timer? _rerunDebounce;

  @override
  void initState() {
    super.initState();
    _displayCanvas = web.HTMLCanvasElement()
      ..style.width = '100%'
      ..style.height = '100%'
      ..style.objectFit = 'contain';
    if (!_viewFactoryRegistered) {
      ui_web.platformViewRegistry.registerViewFactory(
        'face-annotation-canvas',
        (int viewId) => _displayCanvas!,
      );
      _viewFactoryRegistered = true;
    }
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    try {
      setState(() => _status = 'Loading face detection models...');
      _detector = await FaceDetector.create(
        model: _model,
        useLiteRt: _useLiteRt,
        liteRtAccelerator: _liteRtAccelerator,
        withSegmentation: _showSegmentation,
        segmentationConfig: SegmentationConfig(model: _segmentationModel),
      );
      setState(() {
        final backend =
            (_detector as dynamic).activeAccelerator as String? ?? 'tflite-js';
        _status = 'Ready (LiteRT.js, $backend). Pick an image.';
        _isModelReady = true;
      });
    } catch (e) {
      setState(() {
        _status = 'Failed to initialize: $e';
        _isModelReady = false;
      });
    }
  }

  Future<void> _reinitialize() async {
    setState(() {
      _status = 'Reloading models...';
      _isModelReady = false;
    });
    try {
      await _detector?.dispose();
    } catch (_) {}
    _detector = null;
    await _initializeModel();
    if (_pickedBytes != null) {
      await _runDetection();
    }
  }

  @override
  void dispose() {
    _rerunDebounce?.cancel();
    _detector?.dispose();
    _displayCanvas = null;
    super.dispose();
  }

  void _scheduleRerun() {
    _rerunDebounce?.cancel();
    if (_pickedBytes == null || !_isModelReady) return;
    _rerunDebounce = Timer(const Duration(milliseconds: 250), _runDetection);
  }

  Future<void> _pickImage() async {
    final input = web.HTMLInputElement();
    input.accept = 'image/*';
    input.type = 'file';
    final completer = Completer<void>();
    void changeHandler(web.Event _) {
      completer.complete();
      input.removeEventListener('change', changeHandler.toJS);
    }

    input.addEventListener('change', changeHandler.toJS);
    input.click();
    await completer.future;
    final files = input.files;
    if (files == null || files.length == 0) return;
    final file = files.item(0)!;
    final reader = web.FileReader();
    final loadCompleter = Completer<void>();
    void loadHandler(web.Event _) {
      loadCompleter.complete();
      reader.removeEventListener('load', loadHandler.toJS);
    }

    reader.addEventListener('load', loadHandler.toJS);
    reader.readAsArrayBuffer(file);
    await loadCompleter.future;
    final jsBuffer = reader.result as JSArrayBuffer;
    final bytes = Uint8List.view(jsBuffer.toDart);

    setState(() {
      _pickedBytes = bytes;
      _preview = MemoryImage(bytes);
      _hasAnnotation = false;
      _status =
          'Loaded ${file.name} (${bytes.lengthInBytes} bytes); detecting...';
    });
    await _runDetection();
  }

  Future<void> _pickSample(String assetPath) async {
    final data = await DefaultAssetBundle.of(context).load(assetPath);
    final bytes = data.buffer.asUint8List();
    setState(() {
      _pickedBytes = Uint8List.fromList(bytes);
      _preview = MemoryImage(_pickedBytes!);
      _hasAnnotation = false;
      _status = 'Loaded $assetPath; detecting...';
    });
    await _runDetection();
  }

  Future<void> _runDetection() async {
    if (_pickedBytes == null) return;
    if (!_isModelReady || _detector == null) return;

    setState(() {
      _status = 'Detecting...';
      _hasAnnotation = false;
    });
    try {
      final sw = Stopwatch()..start();
      final faces =
          await _detector!.detectFacesFromBytes(_pickedBytes!, mode: _mode);
      sw.stop();
      SegmentationMask? mask;
      if (_showSegmentation && _detector!.isSegmentationReady) {
        mask = await _detector!.getSegmentationMask(_pickedBytes!);
      }
      await _drawAnnotations(faces, mask);
      setState(() {
        _status =
            'Detected ${faces.length} face(s) in ${sw.elapsedMilliseconds}ms';
      });
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
  }

  Future<void> _drawAnnotations(
    List<Face> faces,
    SegmentationMask? mask,
  ) async {
    if (_pickedBytes == null) return;
    final blob = web.Blob([_pickedBytes!.toJS].toJS);
    final url = web.URL.createObjectURL(blob);
    try {
      final htmlImage = web.HTMLImageElement();
      final loadCompleter = Completer<void>();
      htmlImage.addEventListener(
        'load',
        ((web.Event _) => loadCompleter.complete()).toJS,
      );
      htmlImage.addEventListener(
        'error',
        ((web.Event _) => loadCompleter.completeError('decode failed')).toJS,
      );
      htmlImage.src = url;
      await loadCompleter.future;

      final imageWidth = htmlImage.naturalWidth;
      final imageHeight = htmlImage.naturalHeight;
      final canvas = _displayCanvas!;
      canvas.width = imageWidth;
      canvas.height = imageHeight;
      final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D;
      if (_showMaskOnly && mask != null) {
        ctx.fillStyle = 'rgb(20,20,20)'.toJS;
        ctx.fillRect(0, 0, imageWidth, imageHeight);
      } else {
        ctx.drawImage(htmlImage, 0, 0);
      }
      if (mask != null) {
        _drawSegmentationMask(ctx, mask, imageWidth, imageHeight);
      }
      for (final face in faces) {
        _drawFace(ctx, face);
      }
      setState(() => _hasAnnotation = true);
    } finally {
      web.URL.revokeObjectURL(url);
    }
  }

  void _drawFace(web.CanvasRenderingContext2D ctx, Face face) {
    if (_showBoundingBoxes) {
      ctx.strokeStyle = _cssColor(_boundingBoxColor).toJS;
      ctx.lineWidth = _boundingBoxThickness;
      final tl = face.boundingBox.topLeft;
      final br = face.boundingBox.bottomRight;
      ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    }

    if (_showMesh && face.mesh != null) {
      ctx.fillStyle = _cssColor(_meshColor).toJS;
      for (final p in face.mesh!.points) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, _meshSize, 0, 2 * math.pi);
        ctx.fill();
      }
    }

    if (_showLandmarks) {
      ctx.fillStyle = _cssColor(_landmarkColor).toJS;
      for (final p in face.landmarks.values) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, _landmarkSize, 0, 2 * math.pi);
        ctx.fill();
      }
      if (_showLandmarkLabels) {
        ctx.font = '12px sans-serif';
        for (final entry in face.landmarks.toMap().entries) {
          ctx.fillText(entry.key.name, entry.value.x + 6, entry.value.y - 4);
        }
      }
    }

    final eyes = face.eyes;
    if (eyes != null) {
      if (_showIrises) {
        ctx.fillStyle = _cssColor(_irisColor).toJS;
        for (final eye in [eyes.leftEye, eyes.rightEye]) {
          if (eye == null) continue;
          ctx.beginPath();
          ctx.arc(
            eye.irisCenter.x,
            eye.irisCenter.y,
            _landmarkSize,
            0,
            2 * math.pi,
          );
          ctx.fill();
          for (final p in eye.irisContour) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, _landmarkSize, 0, 2 * math.pi);
            ctx.fill();
          }
        }
      }
      if (_showEyeContours) {
        ctx.strokeStyle = _cssColor(_eyeContourColor).toJS;
        ctx.lineWidth = math.max(1, _boundingBoxThickness * 0.6);
        for (final eye in [eyes.leftEye, eyes.rightEye]) {
          if (eye == null || eye.contour.isEmpty) continue;
          for (final pair in eyeLandmarkConnections) {
            if (pair[0] >= eye.contour.length ||
                pair[1] >= eye.contour.length) {
              continue;
            }
            final p1 = eye.contour[pair[0]];
            final p2 = eye.contour[pair[1]];
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      }
      if (_showEyeMesh) {
        ctx.fillStyle = _cssColor(_eyeMeshColor).toJS;
        for (final eye in [eyes.leftEye, eyes.rightEye]) {
          if (eye == null) continue;
          for (final p in eye.mesh) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, _eyeMeshSize, 0, 2 * math.pi);
            ctx.fill();
          }
        }
      }
    }
  }

  void _drawSegmentationMask(
    web.CanvasRenderingContext2D ctx,
    SegmentationMask mask,
    int imageWidth,
    int imageHeight,
  ) {
    final upsampled = mask.upsample(
      targetWidth: imageWidth,
      targetHeight: imageHeight,
      maxSize: 0,
    );
    if (_multiclassClassIndex != null && mask is MulticlassSegmentationMask) {
      final classMask = (mask.upsample(
        targetWidth: imageWidth,
        targetHeight: imageHeight,
        maxSize: 0,
      )).data;
      // Class-specific mask is approximate via base data; for the demo we
      // re-use upsampled foreground when class-specific isn't available.
      _paintMask(
        ctx,
        classMask,
        imageWidth,
        imageHeight,
        _showBinaryMask ? _segmentationThreshold : -1.0,
      );
      return;
    }
    _paintMask(
      ctx,
      upsampled.data,
      imageWidth,
      imageHeight,
      _showBinaryMask ? _segmentationThreshold : -1.0,
    );
  }

  void _paintMask(
    web.CanvasRenderingContext2D ctx,
    Float32List data,
    int width,
    int height,
    double threshold,
  ) {
    if (data.length != width * height) return;
    final r = (_maskColor.r * 255).round();
    final g = (_maskColor.g * 255).round();
    final b = (_maskColor.b * 255).round();
    final imageData = ctx.createImageData(width.toJS, height);
    final rgba = imageData.data.toDart;
    for (int i = 0; i < data.length; i++) {
      final p = data[i].clamp(0.0, 1.0);
      double a;
      if (threshold < 0) {
        a = p;
      } else {
        a = p >= threshold ? 1.0 : 0.0;
      }
      final off = i * 4;
      rgba[off] = r;
      rgba[off + 1] = g;
      rgba[off + 2] = b;
      rgba[off + 3] = (a * (_maskColor.a * 255)).round();
    }
    ctx.putImageData(imageData, 0, 0);
  }

  String _cssColor(Color c) {
    final r = (c.r * 255).round();
    final g = (c.g * 255).round();
    final b = (c.b * 255).round();
    return 'rgb($r,$g,$b)';
  }

  void _showSettings() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (ctx, setSheet) {
            void setBoth(VoidCallback fn) {
              fn();
              setState(() {});
              setSheet(() {});
            }

            return ListView(
              padding: const EdgeInsets.all(16),
              children: [
                const Text(
                  'Detection settings',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                _modeSelector(setBoth),
                _modelSelector(setBoth),
                const Divider(),
                const Text(
                  'Display options',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show bounding boxes'),
                  value: _showBoundingBoxes,
                  onChanged: (v) =>
                      setBoth(() => _showBoundingBoxes = v ?? true),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show face mesh'),
                  value: _showMesh,
                  onChanged: (v) => setBoth(() => _showMesh = v ?? true),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show landmarks'),
                  value: _showLandmarks,
                  onChanged: (v) => setBoth(() => _showLandmarks = v ?? true),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show irises'),
                  value: _showIrises,
                  onChanged: (v) => setBoth(() => _showIrises = v ?? true),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show eye contours'),
                  value: _showEyeContours,
                  onChanged: (v) => setBoth(() => _showEyeContours = v ?? true),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show eye mesh'),
                  value: _showEyeMesh,
                  onChanged: (v) => setBoth(() => _showEyeMesh = v ?? true),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show landmark labels'),
                  value: _showLandmarkLabels,
                  onChanged: (v) =>
                      setBoth(() => _showLandmarkLabels = v ?? false),
                ),
                const Divider(),
                const Text(
                  'Sizes',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                _slider('BBox thickness', _boundingBoxThickness, 0.5, 10.0,
                    (v) => setBoth(() => _boundingBoxThickness = v)),
                _slider('Landmark size', _landmarkSize, 0.5, 15.0,
                    (v) => setBoth(() => _landmarkSize = v)),
                _slider('Mesh size', _meshSize, 0.1, 10.0,
                    (v) => setBoth(() => _meshSize = v)),
                _slider('Eye mesh size', _eyeMeshSize, 0.1, 10.0,
                    (v) => setBoth(() => _eyeMeshSize = v)),
                const Divider(),
                const Text(
                  'Colors',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                _colorPicker('Bounding box', _boundingBoxColor,
                    (c) => setBoth(() => _boundingBoxColor = c)),
                _colorPicker('Landmarks', _landmarkColor,
                    (c) => setBoth(() => _landmarkColor = c)),
                _colorPicker(
                    'Mesh', _meshColor, (c) => setBoth(() => _meshColor = c)),
                _colorPicker(
                    'Iris', _irisColor, (c) => setBoth(() => _irisColor = c)),
                _colorPicker('Eye contour', _eyeContourColor,
                    (c) => setBoth(() => _eyeContourColor = c)),
                _colorPicker('Eye mesh', _eyeMeshColor,
                    (c) => setBoth(() => _eyeMeshColor = c)),
                const Divider(),
                _segmentationSection(setBoth),
                const Divider(),
                _liteRtSection(setBoth),
                const SizedBox(height: 24),
              ],
            );
          },
        );
      },
    ).whenComplete(_scheduleRerun);
  }

  Widget _modeSelector(void Function(VoidCallback) setBoth) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          const Text('Mode'),
          const SizedBox(width: 12),
          for (final m in FaceDetectionMode.values) ...[
            ChoiceChip(
              label: Text(m.name),
              selected: _mode == m,
              onSelected: (_) {
                setBoth(() => _mode = m);
                _scheduleRerun();
              },
            ),
            const SizedBox(width: 4),
          ],
        ],
      ),
    );
  }

  Widget _modelSelector(void Function(VoidCallback) setBoth) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 6,
        children: [
          const Text('Model'),
          for (final m in FaceDetectionModel.values)
            ChoiceChip(
              label: Text(m.name),
              selected: _model == m,
              onSelected: (_) async {
                setBoth(() => _model = m);
                await _reinitialize();
              },
            ),
        ],
      ),
    );
  }

  Widget _slider(String label, double value, double min, double max,
      void Function(double) onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(width: 130, child: Text(label)),
          Expanded(
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              onChanged: (v) {
                onChanged(v);
                _scheduleRerun();
              },
            ),
          ),
          SizedBox(width: 50, child: Text(value.toStringAsFixed(2))),
        ],
      ),
    );
  }

  Widget _colorPicker(String label, Color current, void Function(Color) on) {
    const palette = <Color>[
      Color(0xFF00FFCC),
      Color(0xFF89CFF0),
      Color(0xFFF4C2C2),
      Color(0xFF22AAFF),
      Color(0xFFFFAA22),
      Color(0xFFFF3355),
      Color(0xFF66FF66),
      Color(0xFFFFFF66),
      Color(0xFFFF66FF),
      Color(0xFFFFFFFF),
    ];
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(width: 130, child: Text(label)),
          for (final c in palette)
            GestureDetector(
              onTap: () {
                on(c);
                _scheduleRerun();
              },
              child: Container(
                margin: const EdgeInsets.only(right: 4),
                width: 22,
                height: 22,
                decoration: BoxDecoration(
                  color: c,
                  border: Border.all(
                    color: current.toARGB32() == c.toARGB32()
                        ? Colors.black
                        : Colors.transparent,
                    width: 2,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _segmentationSection(void Function(VoidCallback) setBoth) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Segmentation',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        SwitchListTile(
          dense: true,
          title: const Text('Enable segmentation'),
          value: _showSegmentation,
          onChanged: (v) async {
            setBoth(() => _showSegmentation = v);
            await _reinitialize();
          },
        ),
        if (_showSegmentation) ...[
          Wrap(
            spacing: 6,
            children: [
              const Text('Model:'),
              for (final m in SegmentationModel.values)
                ChoiceChip(
                  label: Text(m.name),
                  selected: _segmentationModel == m,
                  onSelected: (_) async {
                    setBoth(() => _segmentationModel = m);
                    await _reinitialize();
                  },
                ),
            ],
          ),
          if (_segmentationModel == SegmentationModel.multiclass)
            Wrap(
              spacing: 6,
              children: [
                const Text('Class:'),
                ChoiceChip(
                  label: const Text('all'),
                  selected: _multiclassClassIndex == null,
                  onSelected: (_) =>
                      setBoth(() => _multiclassClassIndex = null),
                ),
                for (int i = 0; i < 6; i++)
                  ChoiceChip(
                    label: Text('$i'),
                    selected: _multiclassClassIndex == i,
                    onSelected: (_) => setBoth(() => _multiclassClassIndex = i),
                  ),
              ],
            ),
          CheckboxListTile(
            dense: true,
            title: const Text('Show mask only (hide image)'),
            value: _showMaskOnly,
            onChanged: (v) => setBoth(() => _showMaskOnly = v ?? false),
          ),
          CheckboxListTile(
            dense: true,
            title: const Text('Binary mask (vs soft alpha)'),
            value: _showBinaryMask,
            onChanged: (v) => setBoth(() => _showBinaryMask = v ?? false),
          ),
          _slider('Threshold', _segmentationThreshold, 0.0, 1.0,
              (v) => setBoth(() => _segmentationThreshold = v)),
          Wrap(
            spacing: 6,
            children: [
              const Text('Mask color:'),
              for (final c in const <Color>[
                Color(0x8800FF00),
                Color(0x88FF0000),
                Color(0x880000FF),
                Color(0x88FFFF00),
                Color(0x88FF00FF),
                Color(0x8800FFFF),
              ])
                GestureDetector(
                  onTap: () => setBoth(() => _maskColor = c),
                  child: Container(
                    margin: const EdgeInsets.only(right: 4),
                    width: 22,
                    height: 22,
                    decoration: BoxDecoration(
                      color: c,
                      border: Border.all(
                        color: _maskColor.toARGB32() == c.toARGB32()
                            ? Colors.black
                            : Colors.transparent,
                        width: 2,
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ],
      ],
    );
  }

  Widget _liteRtSection(void Function(VoidCallback) setBoth) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'LiteRT (web runtime)',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        SwitchListTile(
          dense: true,
          title: const Text('Use LiteRT.js'),
          subtitle: const Text(
            'Auto WebGPU / WASM. Disable to use the legacy tflite-js path.',
          ),
          value: _useLiteRt,
          onChanged: (v) async {
            setBoth(() => _useLiteRt = v);
            await _reinitialize();
          },
        ),
        Wrap(
          spacing: 6,
          children: [
            const Text('Accelerator:'),
            for (final a in const <String>['auto', 'webgpu', 'wasm'])
              ChoiceChip(
                label: Text(a),
                selected: _liteRtAccelerator == a,
                onSelected: (_) async {
                  setBoth(() => _liteRtAccelerator = a);
                  await _reinitialize();
                },
              ),
          ],
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Still Image')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: _buildContent(),
      ),
    );
  }

  Widget _buildContent() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            ElevatedButton.icon(
              onPressed: _isModelReady ? _pickImage : null,
              icon: const Icon(Icons.image),
              label: const Text('Select image'),
            ),
            const SizedBox(width: 8),
            for (final s in const [
              'assets/samples/landmark-ex1.jpg',
              'assets/samples/iris-detection-ex1.jpg',
              'assets/samples/iris-detection-ex2.jpg',
              'assets/samples/mesh-ex1.jpeg',
              'assets/samples/group-shot-bounding-box-ex1.jpeg',
            ])
              Padding(
                padding: const EdgeInsets.only(right: 4),
                child: OutlinedButton(
                  onPressed: _isModelReady ? () => _pickSample(s) : null,
                  child: Text(s.split('/').last.split('.').first),
                ),
              ),
            const Spacer(),
            IconButton(
              icon: const Icon(Icons.tune),
              tooltip: 'Settings',
              onPressed: _showSettings,
            ),
          ],
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: _isModelReady ? Colors.green.shade50 : Colors.blue.shade50,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            children: [
              Icon(
                _isModelReady ? Icons.check_circle : Icons.hourglass_empty,
                color: _isModelReady ? Colors.green : Colors.blue,
              ),
              const SizedBox(width: 8),
              Expanded(child: Text(_status)),
            ],
          ),
        ),
        const SizedBox(height: 12),
        if (_preview != null)
          Expanded(
            child: Container(
              width: double.infinity,
              clipBehavior: Clip.hardEdge,
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: _hasAnnotation
                  ? const HtmlElementView(viewType: 'face-annotation-canvas')
                  : FittedBox(
                      fit: BoxFit.contain,
                      child: Image(image: _preview!),
                    ),
            ),
          ),
      ],
    );
  }
}
