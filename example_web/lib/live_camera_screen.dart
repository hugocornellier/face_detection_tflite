// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:math' as math;
import 'dart:ui_web' as ui_web;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

/// Live camera face detection demo using `getUserMedia` + `<video>`.
///
/// Mirrors the native example's `LiveCameraScreen` settings (mode, model,
/// segmentation, virtual background) but uses the browser's webcam APIs
/// directly instead of `package:camera`.
class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

enum _CameraState {
  idle,
  modelLoading,
  awaitingPermission,
  starting,
  running,
  paused,
  permissionDenied,
  noDevice,
  error,
}

class _LiveCameraScreenState extends State<LiveCameraScreen>
    with WidgetsBindingObserver {
  // ---- Detector --------------------------------------------------------
  FaceDetector? _detector;
  bool _isModelReady = false;

  // ---- Camera stream ---------------------------------------------------
  web.MediaStream? _stream;
  web.HTMLVideoElement? _video;
  web.HTMLCanvasElement? _displayCanvas;
  static bool _viewFactoryRegistered = false;
  static const String _viewType = 'face-live-camera-canvas';

  _CameraState _state = _CameraState.idle;
  String _statusMessage = 'Loading models...';
  String _facingMode = 'user';
  bool _supportsRvfc = false;

  // ---- Detection settings ---------------------------------------------
  FaceDetectionMode _mode = FaceDetectionMode.full;
  FaceDetectionModel _model = FaceDetectionModel.frontCamera;

  // ---- Display toggles -------------------------------------------------
  bool _showBoundingBoxes = true;
  bool _showMesh = true;
  bool _showLandmarks = true;
  bool _showIrises = true;
  bool _showEyeContours = true;
  bool _showEyeMesh = false;

  // ---- Colors / sizes (subset to keep the screen focused) -------------
  final Color _boundingBoxColor = const Color(0xFF00FFCC);
  final Color _landmarkColor = const Color(0xFF89CFF0);
  final Color _meshColor = const Color(0xFFF4C2C2);
  final Color _irisColor = const Color(0xFF22AAFF);
  final Color _eyeContourColor = const Color(0xFF22AAFF);
  final Color _eyeMeshColor = const Color(0xFFFFAA22);
  final double _bboxThickness = 2.0;
  final double _landmarkSize = 3.0;
  final double _meshSize = 1.0;
  final double _eyeMeshSize = 0.6;

  // ---- Segmentation ---------------------------------------------------
  bool _showSegmentation = false;
  SegmentationModel _segmentationModel = SegmentationModel.general;
  bool _showVirtualBackground = false;
  Color _maskColor = const Color(0x6600FF00);
  double _segmentationThreshold = 0.5;

  // ---- LiteRT --------------------------------------------------------
  bool _useLiteRt = true;
  String _liteRtAccelerator = 'auto';

  // ---- Loop / FPS ----------------------------------------------------
  bool _busy = false;
  bool _disposed = false;
  int _rafHandle = 0;
  final FpsCounter _fpsCounter = FpsCounter();
  double _fps = 0;
  int _lastDetectionMs = 0;
  int _detectedFaces = 0;
  JSFunction? _visibilityHandlerJs;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _supportsRvfc = _videoPrototypeHasRvfc();
    _displayCanvas = web.HTMLCanvasElement()
      ..style.width = '100%'
      ..style.height = '100%'
      ..style.objectFit = 'contain'
      ..style.backgroundColor = '#000';
    if (!_viewFactoryRegistered) {
      ui_web.platformViewRegistry.registerViewFactory(
        _viewType,
        (int viewId) => _displayCanvas!,
      );
      _viewFactoryRegistered = true;
    }
    _video = web.HTMLVideoElement()
      ..autoplay = true
      ..muted = true
      ..playsInline = true;
    _attachVisibilityListener();
    _initializeDetector();
  }

  bool _videoPrototypeHasRvfc() {
    try {
      final dynamic ctor = globalContext['HTMLVideoElement'];
      if (ctor == null) return false;
      final proto = (ctor as JSObject)['prototype'];
      if (proto == null) return false;
      return (proto as JSObject).has('requestVideoFrameCallback');
    } catch (_) {
      return false;
    }
  }

  void _attachVisibilityListener() {
    _visibilityHandlerJs = ((web.Event _) {
      if (web.document.visibilityState == 'hidden') {
        _pauseLoop();
      } else if (_state == _CameraState.paused) {
        _resumeLoop();
      }
    }).toJS;
    web.document.addEventListener('visibilitychange', _visibilityHandlerJs!);
  }

  Future<void> _initializeDetector() async {
    setState(() {
      _state = _CameraState.modelLoading;
      _statusMessage = 'Loading face detection models...';
    });
    try {
      _detector = await FaceDetector.create(
        model: _model,
        useLiteRt: _useLiteRt,
        liteRtAccelerator: _liteRtAccelerator,
        withSegmentation: _showSegmentation,
        segmentationConfig: SegmentationConfig(model: _segmentationModel),
      );
      if (!mounted) return;
      final backend =
          (_detector as dynamic).activeAccelerator as String? ?? 'tflite-js';
      setState(() {
        _isModelReady = true;
        _state = _CameraState.idle;
        _statusMessage = 'Ready ($backend). Tap "Start camera" to begin.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _state = _CameraState.error;
        _statusMessage = 'Failed to load models: $e';
      });
    }
  }

  Future<void> _restartDetector() async {
    final wasRunning = _state == _CameraState.running;
    _stopLoop();
    await _detector?.dispose();
    _detector = null;
    setState(() => _isModelReady = false);
    await _initializeDetector();
    if (wasRunning && mounted) {
      await _startCamera();
    }
  }

  Future<void> _startCamera() async {
    if (_video == null || _detector == null) return;
    setState(() {
      _state = _CameraState.awaitingPermission;
      _statusMessage = 'Requesting camera access...';
    });
    try {
      // Stop any previous stream first.
      _stopStream();

      final videoConstraints = <String, Object>{
        'facingMode': _facingMode,
        'width': <String, Object>{'ideal': 640},
        'height': <String, Object>{'ideal': 480},
      };
      final constraints = web.MediaStreamConstraints(
        video: videoConstraints.jsify()!,
        audio: false.toJS,
      );
      final mediaDevices = web.window.navigator.mediaDevices;
      _stream = await mediaDevices.getUserMedia(constraints).toDart;
      _video!.srcObject = _stream;
      await _video!.play().toDart;

      // Wait for metadata so we have non-zero videoWidth/videoHeight.
      await _waitForMetadata(_video!);

      if (_disposed) {
        _stopStream();
        return;
      }

      _displayCanvas!
        ..width = _video!.videoWidth
        ..height = _video!.videoHeight;

      setState(() {
        _state = _CameraState.starting;
        _statusMessage = 'Camera ready. Detecting...';
      });
      _scheduleFrame();
      setState(() => _state = _CameraState.running);
    } catch (e) {
      if (!mounted) return;
      final msg = e.toString();
      if (msg.contains('NotAllowedError') || msg.contains('Permission')) {
        setState(() {
          _state = _CameraState.permissionDenied;
          _statusMessage =
              'Camera permission denied. Click "Start camera" to retry.';
        });
      } else if (msg.contains('NotFoundError') ||
          msg.contains('OverconstrainedError')) {
        setState(() {
          _state = _CameraState.noDevice;
          _statusMessage = 'No camera available matching constraints.';
        });
      } else {
        setState(() {
          _state = _CameraState.error;
          _statusMessage = 'Camera error: $msg';
        });
      }
    }
  }

  Future<void> _waitForMetadata(web.HTMLVideoElement video) async {
    if (video.videoWidth > 0 && video.videoHeight > 0) return;
    final completer = Completer<void>();
    void handler(web.Event _) {
      if (!completer.isCompleted) completer.complete();
    }

    final handlerJs = handler.toJS;
    video.addEventListener('loadedmetadata', handlerJs);
    try {
      await completer.future.timeout(const Duration(seconds: 5));
    } finally {
      video.removeEventListener('loadedmetadata', handlerJs);
    }
  }

  void _scheduleFrame() {
    if (_video == null || _disposed || _state == _CameraState.paused) return;
    if (_supportsRvfc) {
      try {
        // ignore: avoid_dynamic_calls
        (_video! as dynamic).requestVideoFrameCallback(
          ((double now, JSObject metadata) {
            _onFrame();
          }).toJS,
        );
        return;
      } catch (_) {
        // Fall through to rAF.
      }
    }
    _rafHandle = web.window.requestAnimationFrame(
      ((double _) => _onFrame()).toJS,
    );
  }

  void _onFrame() {
    if (_disposed || _state != _CameraState.running) return;
    _runDetection().whenComplete(() {
      if (_state == _CameraState.running) _scheduleFrame();
    });
  }

  Future<void> _runDetection() async {
    if (_busy || _detector == null || _video == null) return;
    if (_video!.videoWidth == 0 || _video!.videoHeight == 0) return;
    _busy = true;
    try {
      final sw = Stopwatch()..start();
      final faces = await _detector!.detectFacesFromVideo(
        _video!,
        mode: _mode,
      );
      SegmentationMask? mask;
      if (_showSegmentation && _detector!.isSegmentationReady) {
        mask = await _detector!.getSegmentationMaskFromVideo(_video!);
      }
      sw.stop();
      _draw(faces, mask);
      _fpsCounter.tick();
      if (mounted) {
        setState(() {
          _fps = _fpsCounter.fps.toDouble();
          _lastDetectionMs = sw.elapsedMilliseconds;
          _detectedFaces = faces.length;
        });
      }
    } catch (_) {
      // Swallow per-frame errors (e.g. video not ready) to avoid stopping the loop.
    } finally {
      _busy = false;
    }
  }

  void _draw(List<Face> faces, SegmentationMask? mask) {
    final canvas = _displayCanvas;
    final video = _video;
    if (canvas == null || video == null) return;
    final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D;
    final int w = video.videoWidth;
    final int h = video.videoHeight;
    if (w == 0 || h == 0) return;

    if (_showVirtualBackground && mask != null) {
      ctx.fillStyle = 'rgb(20,20,80)'.toJS;
      ctx.fillRect(0, 0, w, h);
    }
    ctx.drawImage(video, 0, 0, w, h);

    if (mask != null) {
      _drawMask(ctx, mask, w, h);
    }

    for (final face in faces) {
      _drawFace(ctx, face);
    }
  }

  void _drawMask(
    web.CanvasRenderingContext2D ctx,
    SegmentationMask mask,
    int w,
    int h,
  ) {
    final upsampled = mask.upsample(
      targetWidth: w,
      targetHeight: h,
      maxSize: 0,
    );
    final data = upsampled.data;
    if (data.length != w * h) return;
    final r = (_maskColor.r * 255).round();
    final g = (_maskColor.g * 255).round();
    final b = (_maskColor.b * 255).round();
    final a = (_maskColor.a * 255).round();
    final imageData = ctx.createImageData(w.toJS, h);
    final rgba = imageData.data.toDart;
    for (int i = 0; i < data.length; i++) {
      final p = data[i].clamp(0.0, 1.0);
      final off = i * 4;
      final showFg = p >= _segmentationThreshold;
      if (showFg) {
        rgba[off] = r;
        rgba[off + 1] = g;
        rgba[off + 2] = b;
        rgba[off + 3] = a;
      } else {
        rgba[off + 3] = 0;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }

  void _drawFace(web.CanvasRenderingContext2D ctx, Face face) {
    if (_showBoundingBoxes) {
      ctx.strokeStyle = _cssColor(_boundingBoxColor).toJS;
      ctx.lineWidth = _bboxThickness;
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
        ctx.lineWidth = math.max(1, _bboxThickness * 0.6);
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

  String _cssColor(Color c) {
    final r = (c.r * 255).round();
    final g = (c.g * 255).round();
    final b = (c.b * 255).round();
    return 'rgb($r,$g,$b)';
  }

  void _pauseLoop() {
    if (_state != _CameraState.running) return;
    setState(() {
      _state = _CameraState.paused;
      _statusMessage = 'Paused (tab in background).';
    });
  }

  void _resumeLoop() {
    if (_state != _CameraState.paused) return;
    setState(() {
      _state = _CameraState.running;
      _statusMessage = 'Detecting...';
    });
    _scheduleFrame();
  }

  void _stopLoop() {
    if (_rafHandle != 0) {
      web.window.cancelAnimationFrame(_rafHandle);
      _rafHandle = 0;
    }
  }

  void _stopStream() {
    final stream = _stream;
    if (stream != null) {
      try {
        final tracks = stream.getTracks().toDart;
        for (final t in tracks) {
          t.stop();
        }
      } catch (_) {}
      _stream = null;
    }
    _video?.srcObject = null;
  }

  Future<void> _swapFacingMode() async {
    _facingMode = _facingMode == 'user' ? 'environment' : 'user';
    if (_state == _CameraState.running) {
      await _startCamera();
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive) {
      _pauseLoop();
    } else if (state == AppLifecycleState.resumed) {
      _resumeLoop();
    }
  }

  @override
  void dispose() {
    _disposed = true;
    WidgetsBinding.instance.removeObserver(this);
    if (_visibilityHandlerJs != null) {
      web.document.removeEventListener(
        'visibilitychange',
        _visibilityHandlerJs!,
      );
      _visibilityHandlerJs = null;
    }
    _stopLoop();
    _stopStream();
    _detector?.dispose();
    _video = null;
    _displayCanvas = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Camera'),
        actions: [
          IconButton(
            tooltip: 'Switch camera',
            icon: const Icon(Icons.cameraswitch),
            onPressed: _state == _CameraState.running ? _swapFacingMode : null,
          ),
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.tune),
            onPressed: _isModelReady ? _showSettings : null,
          ),
        ],
      ),
      body: Column(
        children: [
          _buildStatusBar(),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: AspectRatio(
                aspectRatio: 4.0 / 3.0,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: const HtmlElementView(viewType: _viewType),
                ),
              ),
            ),
          ),
          _buildControls(),
        ],
      ),
    );
  }

  Widget _buildStatusBar() {
    final color = switch (_state) {
      _CameraState.running => Colors.green.shade100,
      _CameraState.paused => Colors.amber.shade100,
      _CameraState.permissionDenied ||
      _CameraState.noDevice ||
      _CameraState.error =>
        Colors.red.shade100,
      _ => Colors.blue.shade100,
    };
    final icon = switch (_state) {
      _CameraState.running => Icons.videocam,
      _CameraState.paused => Icons.pause_circle,
      _CameraState.permissionDenied => Icons.lock,
      _CameraState.noDevice => Icons.videocam_off,
      _CameraState.error => Icons.error,
      _ => Icons.hourglass_empty,
    };
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      color: color,
      child: Row(
        children: [
          Icon(icon),
          const SizedBox(width: 8),
          Expanded(child: Text(_statusMessage)),
          if (_state == _CameraState.running) ...[
            const SizedBox(width: 8),
            Text(
              '${(_detector as dynamic).activeAccelerator ?? "?"} · '
              '${_fps.toStringAsFixed(1)} fps · ${_lastDetectionMs}ms · '
              '$_detectedFaces face(s)',
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildControls() {
    final canStart = _isModelReady &&
        (_state == _CameraState.idle ||
            _state == _CameraState.permissionDenied ||
            _state == _CameraState.noDevice ||
            _state == _CameraState.error);
    final canStop =
        _state == _CameraState.running || _state == _CameraState.paused;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          ElevatedButton.icon(
            onPressed: canStart ? _startCamera : null,
            icon: const Icon(Icons.play_arrow),
            label: const Text('Start camera'),
          ),
          const SizedBox(width: 8),
          OutlinedButton.icon(
            onPressed: canStop
                ? () {
                    _stopLoop();
                    _stopStream();
                    setState(() {
                      _state = _CameraState.idle;
                      _statusMessage = 'Camera stopped.';
                    });
                  }
                : null,
            icon: const Icon(Icons.stop),
            label: const Text('Stop'),
          ),
          const SizedBox(width: 16),
          DropdownButton<FaceDetectionMode>(
            value: _mode,
            items: [
              for (final m in FaceDetectionMode.values)
                DropdownMenuItem(value: m, child: Text(m.name)),
            ],
            onChanged: (v) {
              if (v != null) setState(() => _mode = v);
            },
          ),
        ],
      ),
    );
  }

  void _showSettings() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setSheet) {
          void both(VoidCallback fn) {
            fn();
            setState(() {});
            setSheet(() {});
          }

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              const Text(
                'Detection',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              Wrap(
                spacing: 6,
                children: [
                  const Text('Mode:'),
                  for (final m in FaceDetectionMode.values)
                    ChoiceChip(
                      label: Text(m.name),
                      selected: _mode == m,
                      onSelected: (_) => both(() => _mode = m),
                    ),
                ],
              ),
              Wrap(
                spacing: 6,
                children: [
                  const Text('Model:'),
                  for (final m in FaceDetectionModel.values)
                    ChoiceChip(
                      label: Text(m.name),
                      selected: _model == m,
                      onSelected: (_) async {
                        both(() => _model = m);
                        await _restartDetector();
                      },
                    ),
                ],
              ),
              const Divider(),
              const Text(
                'Display',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Bounding boxes'),
                value: _showBoundingBoxes,
                onChanged: (v) => both(() => _showBoundingBoxes = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Face mesh'),
                value: _showMesh,
                onChanged: (v) => both(() => _showMesh = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Landmarks'),
                value: _showLandmarks,
                onChanged: (v) => both(() => _showLandmarks = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Irises'),
                value: _showIrises,
                onChanged: (v) => both(() => _showIrises = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Eye contours'),
                value: _showEyeContours,
                onChanged: (v) => both(() => _showEyeContours = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Eye mesh'),
                value: _showEyeMesh,
                onChanged: (v) => both(() => _showEyeMesh = v ?? false),
              ),
              const Divider(),
              SwitchListTile(
                dense: true,
                title: const Text('Segmentation'),
                value: _showSegmentation,
                onChanged: (v) async {
                  both(() => _showSegmentation = v);
                  await _restartDetector();
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
                          both(() => _segmentationModel = m);
                          await _restartDetector();
                        },
                      ),
                  ],
                ),
                SwitchListTile(
                  dense: true,
                  title: const Text('Virtual background'),
                  value: _showVirtualBackground,
                  onChanged: (v) => both(() => _showVirtualBackground = v),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      const SizedBox(width: 100, child: Text('Threshold')),
                      Expanded(
                        child: Slider(
                          value: _segmentationThreshold,
                          onChanged: (v) =>
                              both(() => _segmentationThreshold = v),
                        ),
                      ),
                      Text(_segmentationThreshold.toStringAsFixed(2)),
                    ],
                  ),
                ),
                Wrap(
                  spacing: 6,
                  children: [
                    const Text('Mask color:'),
                    for (final c in const <Color>[
                      Color(0x6600FF00),
                      Color(0x66FF0000),
                      Color(0x660000FF),
                      Color(0x66FFFF00),
                      Color(0x66FF00FF),
                      Color(0x6600FFFF),
                    ])
                      GestureDetector(
                        onTap: () => both(() => _maskColor = c),
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
              const Divider(),
              const Text(
                'LiteRT.js',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              SwitchListTile(
                dense: true,
                title: const Text('Use LiteRT.js'),
                subtitle: const Text('Auto WebGPU / WASM fallback'),
                value: _useLiteRt,
                onChanged: (v) async {
                  both(() => _useLiteRt = v);
                  await _restartDetector();
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
                        both(() => _liteRtAccelerator = a);
                        await _restartDetector();
                      },
                    ),
                ],
              ),
              const SizedBox(height: 24),
            ],
          );
        },
      ),
    );
  }
}
