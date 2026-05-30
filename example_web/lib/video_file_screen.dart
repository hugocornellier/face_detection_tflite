// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:js_interop';

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import 'detection_overlay.dart';

/// Face detection on an uploaded video file, played back in a `<video>`
/// element with a real-time overlay drawn on top.
///
/// The browser decodes the file natively, so no native video backend (OpenCV /
/// ffmpeg) is needed: each frame is fed to [FaceDetector.detectFacesFromVideo]
/// exactly like the live camera demo. The detection loop, drawing, and
/// settings sheet are shared via [DetectionOverlayMixin].
class VideoFileScreen extends StatefulWidget {
  const VideoFileScreen({super.key});

  @override
  State<VideoFileScreen> createState() => _VideoFileScreenState();
}

enum _VideoState {
  idle,
  modelLoading,
  ready,
  playing,
  paused,
  ended,
  error,
}

class _VideoFileScreenState extends State<VideoFileScreen>
    with WidgetsBindingObserver, DetectionOverlayMixin {
  @override
  String get viewType => 'face-video-file-canvas';

  @override
  bool get loopActive => _state == _VideoState.playing;

  _VideoState _state = _VideoState.idle;
  String _statusMessage = 'Loading models...';
  String? _fileName;
  String? _objectUrl;
  bool _loop = true;

  double _position = 0;
  double _duration = 0;

  JSFunction? _onPlay;
  JSFunction? _onPause;
  JSFunction? _onEnded;
  JSFunction? _onSeeked;
  JSFunction? _onTimeUpdate;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    supportsRvfc = videoPrototypeHasRvfc();
    registerCanvas();
    video = web.HTMLVideoElement()
      ..muted = false
      ..playsInline = true
      ..loop = _loop;
    _attachVideoListeners();
    _state = _VideoState.modelLoading;
    _statusMessage = 'Loading face detection models...';
    initializeDetector();
  }

  @override
  void onDetectorReady(String backend) {
    setState(() {
      if (_state == _VideoState.modelLoading) _state = _VideoState.idle;
      _statusMessage = _objectUrl == null
          ? 'Ready ($backend). Pick a video to begin.'
          : 'Ready ($backend).';
    });
  }

  @override
  void onDetectorError(Object error) {
    setState(() {
      _state = _VideoState.error;
      _statusMessage = 'Failed to load models: $error';
    });
  }

  @override
  Future<void> restartDetector() async {
    final wasPlaying = _state == _VideoState.playing;
    stopLoop();
    await disposeDetector();
    setState(() {});
    await initializeDetector();
    if (wasPlaying && mounted && video != null && !video!.paused) {
      scheduleFrame();
    }
  }

  void _attachVideoListeners() {
    final v = video!;
    _onPlay = ((web.Event _) {
      if (!mounted) return;
      setState(() {
        _state = _VideoState.playing;
        _statusMessage = 'Playing...';
      });
      scheduleFrame();
    }).toJS;
    _onPause = ((web.Event _) {
      if (!mounted || _state == _VideoState.ended) return;
      setState(() {
        _state = _VideoState.paused;
        _statusMessage = 'Paused.';
      });
    }).toJS;
    _onEnded = ((web.Event _) {
      if (!mounted) return;
      setState(() {
        _state = _VideoState.ended;
        _statusMessage = 'Ended.';
      });
    }).toJS;
    _onSeeked = ((web.Event _) {
      if (!mounted) return;
      setState(() => _position = video?.currentTime.toDouble() ?? 0);
      // Refresh the overlay for the new frame when not actively playing.
      if (_state != _VideoState.playing) runDetection();
    }).toJS;
    _onTimeUpdate = ((web.Event _) {
      if (!mounted) return;
      setState(() => _position = video?.currentTime.toDouble() ?? 0);
    }).toJS;
    v.addEventListener('play', _onPlay!);
    v.addEventListener('pause', _onPause!);
    v.addEventListener('ended', _onEnded!);
    v.addEventListener('seeked', _onSeeked!);
    v.addEventListener('timeupdate', _onTimeUpdate!);
  }

  Future<void> _pickVideo() async {
    final input = web.HTMLInputElement()
      ..accept = 'video/*'
      ..type = 'file';
    final completer = Completer<void>();
    void changeHandler(web.Event _) {
      if (!completer.isCompleted) completer.complete();
      input.removeEventListener('change', changeHandler.toJS);
    }

    input.addEventListener('change', changeHandler.toJS);
    input.click();
    await completer.future;

    final files = input.files;
    if (files == null || files.length == 0) return;
    final file = files.item(0)!;
    await _loadVideoFromBlob(file, file.name);
  }

  Future<void> _loadVideoFromBlob(web.Blob blob, String name) async {
    final v = video;
    if (v == null || detector == null) return;

    stopLoop();
    _revokeUrl();
    final url = web.URL.createObjectURL(blob);
    _objectUrl = url;
    _fileName = name;
    v
      ..src = url
      ..loop = _loop
      ..currentTime = 0;

    setState(() {
      _state = _VideoState.ready;
      _statusMessage = 'Loading "$name"...';
      _position = 0;
      _duration = 0;
    });

    try {
      await waitForMetadata(v);
      if (disposed) return;
      displayCanvas!
        ..width = v.videoWidth
        ..height = v.videoHeight;
      setState(
          () => _duration = v.duration.isFinite ? v.duration.toDouble() : 0);
      await v.play().toDart;
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _state = _VideoState.error;
        _statusMessage = 'Could not play video: $e';
      });
    }
  }

  void _togglePlayPause() {
    final v = video;
    if (v == null) return;
    if (_state == _VideoState.playing) {
      v.pause();
    } else {
      if (_state == _VideoState.ended) v.currentTime = 0;
      v.play();
    }
  }

  void _restart() {
    final v = video;
    if (v == null) return;
    v.currentTime = 0;
    v.play();
  }

  void _toggleLoop() {
    setState(() => _loop = !_loop);
    video?.loop = _loop;
  }

  void _seek(double seconds) {
    final v = video;
    if (v == null) return;
    v.currentTime = seconds;
  }

  void _revokeUrl() {
    final url = _objectUrl;
    if (url != null) {
      web.URL.revokeObjectURL(url);
      _objectUrl = null;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive) {
      if (_state == _VideoState.playing) video?.pause();
    }
  }

  @override
  void dispose() {
    disposed = true;
    WidgetsBinding.instance.removeObserver(this);
    final v = video;
    if (v != null) {
      if (_onPlay != null) v.removeEventListener('play', _onPlay!);
      if (_onPause != null) v.removeEventListener('pause', _onPause!);
      if (_onEnded != null) v.removeEventListener('ended', _onEnded!);
      if (_onSeeked != null) v.removeEventListener('seeked', _onSeeked!);
      if (_onTimeUpdate != null) {
        v.removeEventListener('timeupdate', _onTimeUpdate!);
      }
      v.pause();
      v.removeAttribute('src');
    }
    stopLoop();
    _revokeUrl();
    detector?.dispose();
    video = null;
    displayCanvas = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video File'),
        actions: [
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
                aspectRatio: 16.0 / 9.0,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Container(
                    color: Colors.black,
                    child: _objectUrl == null
                        ? const Center(
                            child: Text(
                              'No video loaded',
                              style: TextStyle(color: Colors.white70),
                            ),
                          )
                        : HtmlElementView(viewType: viewType),
                  ),
                ),
              ),
            ),
          ),
          _buildSeekBar(),
          _buildControls(),
        ],
      ),
    );
  }

  Widget _buildStatusBar() {
    final color = switch (_state) {
      _VideoState.playing => Colors.green.shade100,
      _VideoState.paused || _VideoState.ended => Colors.amber.shade100,
      _VideoState.error => Colors.red.shade100,
      _ => Colors.blue.shade100,
    };
    final icon = switch (_state) {
      _VideoState.playing => Icons.play_circle,
      _VideoState.paused => Icons.pause_circle,
      _VideoState.ended => Icons.replay_circle_filled,
      _VideoState.error => Icons.error,
      _ => Icons.hourglass_empty,
    };
    final showStats = _state == _VideoState.playing ||
        _state == _VideoState.paused ||
        _state == _VideoState.ended;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      color: color,
      child: Row(
        children: [
          Icon(icon),
          const SizedBox(width: 8),
          Expanded(
            child: Text(_fileName != null
                ? '$_fileName · $_statusMessage'
                : _statusMessage),
          ),
          if (showStats) ...[
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

  Widget _buildSeekBar() {
    final hasVideo = _objectUrl != null && _duration > 0;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Row(
        children: [
          Text(_fmt(_position)),
          Expanded(
            child: Slider(
              value: _position.clamp(0, _duration == 0 ? 1 : _duration),
              max: _duration == 0 ? 1 : _duration,
              onChanged: hasVideo
                  ? (v) {
                      setState(() => _position = v);
                      _seek(v);
                    }
                  : null,
            ),
          ),
          Text(_fmt(_duration)),
        ],
      ),
    );
  }

  Widget _buildControls() {
    final hasVideo = _objectUrl != null;
    final isPlaying = _state == _VideoState.playing;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          ElevatedButton.icon(
            onPressed: isModelReady ? _pickVideo : null,
            icon: const Icon(Icons.video_file),
            label: const Text('Pick video'),
          ),
          OutlinedButton.icon(
            onPressed: hasVideo ? _togglePlayPause : null,
            icon: Icon(isPlaying ? Icons.pause : Icons.play_arrow),
            label: Text(isPlaying ? 'Pause' : 'Play'),
          ),
          OutlinedButton.icon(
            onPressed: hasVideo ? _restart : null,
            icon: const Icon(Icons.replay),
            label: const Text('Restart'),
          ),
          FilterChip(
            label: const Text('Loop'),
            selected: _loop,
            onSelected: (_) => _toggleLoop(),
          ),
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

  String _fmt(double seconds) {
    if (!seconds.isFinite || seconds < 0) return '0:00';
    final total = seconds.round();
    final m = total ~/ 60;
    final s = total % 60;
    return '$m:${s.toString().padLeft(2, '0')}';
  }
}
