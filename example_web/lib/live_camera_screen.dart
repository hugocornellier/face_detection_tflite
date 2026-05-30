// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:js_interop';

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import 'detection_overlay.dart';

/// Live camera face detection demo using `getUserMedia` + `<video>`.
///
/// Mirrors the native example's `LiveCameraScreen` settings (mode, model,
/// segmentation, virtual background) but uses the browser's webcam APIs
/// directly instead of `package:camera`. The detection loop, drawing, and
/// settings sheet are shared with the video-file screen via
/// [DetectionOverlayMixin].
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
    with WidgetsBindingObserver, DetectionOverlayMixin {
  @override
  String get viewType => 'face-live-camera-canvas';

  @override
  bool get loopActive => _state == _CameraState.running;

  // ---- Camera stream ---------------------------------------------------
  web.MediaStream? _stream;

  _CameraState _state = _CameraState.idle;
  String _statusMessage = 'Loading models...';
  String _facingMode = 'user';
  JSFunction? _visibilityHandlerJs;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    supportsRvfc = videoPrototypeHasRvfc();
    registerCanvas();
    video = web.HTMLVideoElement()
      ..autoplay = true
      ..muted = true
      ..playsInline = true;
    _attachVisibilityListener();
    _state = _CameraState.modelLoading;
    _statusMessage = 'Loading face detection models...';
    initializeDetector();
  }

  @override
  void onDetectorReady(String backend) {
    setState(() {
      _state = _CameraState.idle;
      _statusMessage = 'Ready ($backend). Tap "Start camera" to begin.';
    });
  }

  @override
  void onDetectorError(Object error) {
    setState(() {
      _state = _CameraState.error;
      _statusMessage = 'Failed to load models: $error';
    });
  }

  @override
  Future<void> restartDetector() async {
    final wasRunning = _state == _CameraState.running;
    stopLoop();
    await disposeDetector();
    setState(() {});
    await initializeDetector();
    if (wasRunning && mounted) {
      await _startCamera();
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

  Future<void> _startCamera() async {
    if (video == null || detector == null) return;
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
      video!.srcObject = _stream;
      await video!.play().toDart;

      // Wait for metadata so we have non-zero videoWidth/videoHeight.
      await waitForMetadata(video!);

      if (disposed) {
        _stopStream();
        return;
      }

      displayCanvas!
        ..width = video!.videoWidth
        ..height = video!.videoHeight;

      setState(() {
        _state = _CameraState.starting;
        _statusMessage = 'Camera ready. Detecting...';
      });
      setState(() => _state = _CameraState.running);
      scheduleFrame();
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
    scheduleFrame();
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
    video?.srcObject = null;
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
    disposed = true;
    WidgetsBinding.instance.removeObserver(this);
    if (_visibilityHandlerJs != null) {
      web.document.removeEventListener(
        'visibilitychange',
        _visibilityHandlerJs!,
      );
      _visibilityHandlerJs = null;
    }
    stopLoop();
    _stopStream();
    detector?.dispose();
    video = null;
    displayCanvas = null;
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
            onPressed: isModelReady ? showDetectionSettings : null,
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
                  child: HtmlElementView(viewType: viewType),
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
              '$activeAccelerator · '
              '${fps.toStringAsFixed(1)} fps · ${lastDetectionMs}ms · '
              '$detectedFaces face(s)',
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildControls() {
    final canStart = isModelReady &&
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
                    stopLoop();
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
            value: mode,
            items: [
              for (final m in FaceDetectionMode.values)
                DropdownMenuItem(value: m, child: Text(m.name)),
            ],
            onChanged: (v) {
              if (v != null) setState(() => mode = v);
            },
          ),
        ],
      ),
    );
  }
}
