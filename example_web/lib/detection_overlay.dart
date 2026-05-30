// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:math' as math;
import 'dart:ui_web' as ui_web;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

/// View types registered once per platform-view, mapped to the canvas that
/// should currently back them. Storing the live canvas (rather than capturing
/// it inside the factory closure) keeps revisited screens from rendering into
/// a disposed canvas.
final Map<String, web.HTMLCanvasElement> _activeCanvases = {};
final Set<String> _registeredViewTypes = {};

/// Shared face-detection + canvas-overlay machinery for the web example
/// screens that draw on top of an `<video>` element.
///
/// Both [LiveCameraScreen] and the video-file screen pull frames from an
/// [web.HTMLVideoElement] and run [FaceDetector.detectFacesFromVideo] on each
/// frame, drawing the results onto a 2D canvas. Everything that does not
/// depend on *where* the video frames come from lives here; subclasses supply
/// the frame source (camera stream vs. file) and the playback controls.
mixin DetectionOverlayMixin<T extends StatefulWidget> on State<T> {
  // ---- Detector --------------------------------------------------------
  FaceDetector? detector;
  bool isModelReady = false;

  // ---- Video + canvas --------------------------------------------------
  web.HTMLVideoElement? video;
  web.HTMLCanvasElement? displayCanvas;
  bool supportsRvfc = false;

  // ---- Detection settings ---------------------------------------------
  FaceDetectionMode mode = FaceDetectionMode.full;
  FaceDetectionModel model = FaceDetectionModel.frontCamera;

  // ---- Display toggles -------------------------------------------------
  bool showBoundingBoxes = true;
  bool showMesh = true;
  bool showLandmarks = true;
  bool showIrises = true;
  bool showEyeContours = true;
  bool showEyeMesh = false;

  // ---- Colors / sizes --------------------------------------------------
  final Color boundingBoxColor = const Color(0xFF00FFCC);
  final Color landmarkColor = const Color(0xFF89CFF0);
  final Color meshColor = const Color(0xFFF4C2C2);
  final Color irisColor = const Color(0xFF22AAFF);
  final Color eyeContourColor = const Color(0xFF22AAFF);
  final Color eyeMeshColor = const Color(0xFFFFAA22);
  final double bboxThickness = 2.0;
  final double landmarkSize = 3.0;
  final double meshSize = 1.0;
  final double eyeMeshSize = 0.6;

  // ---- Segmentation ----------------------------------------------------
  bool showSegmentation = false;
  SegmentationModel segmentationModel = SegmentationModel.general;
  bool showVirtualBackground = false;
  Color maskColor = const Color(0x6600FF00);
  double segmentationThreshold = 0.5;

  // ---- LiteRT ----------------------------------------------------------
  bool useLiteRt = true;
  String liteRtAccelerator = 'auto';

  // ---- Loop / FPS ------------------------------------------------------
  bool busy = false;
  bool disposed = false;
  int rafHandle = 0;
  final FpsCounter fpsCounter = FpsCounter();
  double fps = 0;
  int lastDetectionMs = 0;
  int detectedFaces = 0;

  // ---- Hooks the host screen must provide ------------------------------

  /// Unique platform-view type for this screen's canvas.
  String get viewType;

  /// Whether the detection loop should keep scheduling frames. Camera: the
  /// stream is running. Video file: the element is playing.
  bool get loopActive;

  /// Re-create the detector after a setting that requires a fresh instance
  /// (model / segmentation / LiteRT) changes, preserving playback state.
  Future<void> restartDetector();

  /// Called once the detector finishes loading. [backend] is the active
  /// accelerator string (e.g. `webgpu`, `wasm`, `tflite-js`).
  void onDetectorReady(String backend);

  /// Called if detector creation throws.
  void onDetectorError(Object error);

  // ---- Setup -----------------------------------------------------------

  /// Creates the backing canvas and registers it as a platform view. Call
  /// from the host screen's `initState`.
  void registerCanvas() {
    displayCanvas = web.HTMLCanvasElement()
      ..style.width = '100%'
      ..style.height = '100%'
      ..style.objectFit = 'contain'
      ..style.backgroundColor = '#000';
    _activeCanvases[viewType] = displayCanvas!;
    if (_registeredViewTypes.add(viewType)) {
      ui_web.platformViewRegistry.registerViewFactory(
        viewType,
        (int _) => _activeCanvases[viewType]!,
      );
    }
  }

  bool videoPrototypeHasRvfc() {
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

  Future<void> waitForMetadata(web.HTMLVideoElement v) async {
    if (v.videoWidth > 0 && v.videoHeight > 0) return;
    final completer = Completer<void>();
    void handler(web.Event _) {
      if (!completer.isCompleted) completer.complete();
    }

    final handlerJs = handler.toJS;
    v.addEventListener('loadedmetadata', handlerJs);
    try {
      await completer.future.timeout(const Duration(seconds: 5));
    } finally {
      v.removeEventListener('loadedmetadata', handlerJs);
    }
  }

  String get activeAccelerator =>
      (detector as dynamic)?.activeAccelerator as String? ?? '?';

  // ---- Detector lifecycle ---------------------------------------------

  Future<void> initializeDetector() async {
    try {
      detector = await FaceDetector.create(
        model: model,
        useLiteRt: useLiteRt,
        liteRtAccelerator: liteRtAccelerator,
        withSegmentation: showSegmentation,
        segmentationConfig: SegmentationConfig(model: segmentationModel),
      );
      if (!mounted) {
        await detector?.dispose();
        detector = null;
        return;
      }
      isModelReady = true;
      onDetectorReady(
          activeAccelerator == '?' ? 'tflite-js' : activeAccelerator);
    } catch (e) {
      if (!mounted) return;
      onDetectorError(e);
    }
  }

  Future<void> disposeDetector() async {
    final d = detector;
    detector = null;
    isModelReady = false;
    await d?.dispose();
  }

  // ---- Detection loop --------------------------------------------------

  void scheduleFrame() {
    if (video == null || disposed || !loopActive) return;
    if (supportsRvfc) {
      try {
        // ignore: avoid_dynamic_calls
        (video! as dynamic).requestVideoFrameCallback(
          ((double now, JSObject metadata) {
            onFrame();
          }).toJS,
        );
        return;
      } catch (_) {
        // Fall through to rAF.
      }
    }
    rafHandle = web.window.requestAnimationFrame(
      ((double _) => onFrame()).toJS,
    );
  }

  void onFrame() {
    if (disposed || !loopActive) return;
    runDetection().whenComplete(() {
      if (loopActive) scheduleFrame();
    });
  }

  Future<void> runDetection() async {
    if (busy || detector == null || video == null) return;
    if (video!.videoWidth == 0 || video!.videoHeight == 0) return;
    busy = true;
    try {
      final sw = Stopwatch()..start();
      final faces = await detector!.detectFacesFromVideo(video!, mode: mode);
      SegmentationMask? mask;
      if (showSegmentation && detector!.isSegmentationReady) {
        mask = await detector!.getSegmentationMaskFromVideo(video!);
      }
      sw.stop();
      draw(faces, mask);
      fpsCounter.tick();
      if (mounted) {
        setState(() {
          fps = fpsCounter.fps.toDouble();
          lastDetectionMs = sw.elapsedMilliseconds;
          detectedFaces = faces.length;
        });
      }
    } catch (_) {
      // Swallow per-frame errors (e.g. video not ready) to avoid stopping the
      // loop.
    } finally {
      busy = false;
    }
  }

  void stopLoop() {
    if (rafHandle != 0) {
      web.window.cancelAnimationFrame(rafHandle);
      rafHandle = 0;
    }
  }

  // ---- Drawing ---------------------------------------------------------

  void draw(List<Face> faces, SegmentationMask? mask) {
    final canvas = displayCanvas;
    final v = video;
    if (canvas == null || v == null) return;
    final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D;
    final int w = v.videoWidth;
    final int h = v.videoHeight;
    if (w == 0 || h == 0) return;

    if (showVirtualBackground && mask != null) {
      ctx.fillStyle = 'rgb(20,20,80)'.toJS;
      ctx.fillRect(0, 0, w, h);
    }
    ctx.drawImage(v, 0, 0, w, h);

    if (mask != null) {
      drawMask(ctx, mask, w, h);
    }

    for (final face in faces) {
      drawFace(ctx, face);
    }
  }

  void drawMask(
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
    final r = (maskColor.r * 255).round();
    final g = (maskColor.g * 255).round();
    final b = (maskColor.b * 255).round();
    final a = (maskColor.a * 255).round();
    final imageData = ctx.createImageData(w.toJS, h);
    final rgba = imageData.data.toDart;
    for (int i = 0; i < data.length; i++) {
      final p = data[i].clamp(0.0, 1.0);
      final off = i * 4;
      final showFg = p >= segmentationThreshold;
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

  void drawFace(web.CanvasRenderingContext2D ctx, Face face) {
    if (showBoundingBoxes) {
      ctx.strokeStyle = cssColor(boundingBoxColor).toJS;
      ctx.lineWidth = bboxThickness;
      final tl = face.boundingBox.topLeft;
      final br = face.boundingBox.bottomRight;
      ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    }
    if (showMesh && face.mesh != null) {
      ctx.fillStyle = cssColor(meshColor).toJS;
      for (final p in face.mesh!.points) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, meshSize, 0, 2 * math.pi);
        ctx.fill();
      }
    }
    if (showLandmarks) {
      ctx.fillStyle = cssColor(landmarkColor).toJS;
      for (final p in face.landmarks.values) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, landmarkSize, 0, 2 * math.pi);
        ctx.fill();
      }
    }
    final eyes = face.eyes;
    if (eyes != null) {
      if (showIrises) {
        ctx.fillStyle = cssColor(irisColor).toJS;
        for (final eye in [eyes.leftEye, eyes.rightEye]) {
          if (eye == null) continue;
          ctx.beginPath();
          ctx.arc(
            eye.irisCenter.x,
            eye.irisCenter.y,
            landmarkSize,
            0,
            2 * math.pi,
          );
          ctx.fill();
          for (final p in eye.irisContour) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, landmarkSize, 0, 2 * math.pi);
            ctx.fill();
          }
        }
      }
      if (showEyeContours) {
        ctx.strokeStyle = cssColor(eyeContourColor).toJS;
        ctx.lineWidth = math.max(1, bboxThickness * 0.6);
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
      if (showEyeMesh) {
        ctx.fillStyle = cssColor(eyeMeshColor).toJS;
        for (final eye in [eyes.leftEye, eyes.rightEye]) {
          if (eye == null) continue;
          for (final p in eye.mesh) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, eyeMeshSize, 0, 2 * math.pi);
            ctx.fill();
          }
        }
      }
    }
  }

  String cssColor(Color c) {
    final r = (c.r * 255).round();
    final g = (c.g * 255).round();
    final b = (c.b * 255).round();
    return 'rgb($r,$g,$b)';
  }

  // ---- Shared settings sheet ------------------------------------------

  void showDetectionSettings() {
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
                      selected: mode == m,
                      onSelected: (_) => both(() => mode = m),
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
                      selected: model == m,
                      onSelected: (_) async {
                        both(() => model = m);
                        await restartDetector();
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
                value: showBoundingBoxes,
                onChanged: (v) => both(() => showBoundingBoxes = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Face mesh'),
                value: showMesh,
                onChanged: (v) => both(() => showMesh = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Landmarks'),
                value: showLandmarks,
                onChanged: (v) => both(() => showLandmarks = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Irises'),
                value: showIrises,
                onChanged: (v) => both(() => showIrises = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Eye contours'),
                value: showEyeContours,
                onChanged: (v) => both(() => showEyeContours = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Eye mesh'),
                value: showEyeMesh,
                onChanged: (v) => both(() => showEyeMesh = v ?? false),
              ),
              const Divider(),
              SwitchListTile(
                dense: true,
                title: const Text('Segmentation'),
                value: showSegmentation,
                onChanged: (v) async {
                  both(() => showSegmentation = v);
                  await restartDetector();
                },
              ),
              if (showSegmentation) ...[
                Wrap(
                  spacing: 6,
                  children: [
                    const Text('Model:'),
                    for (final m in SegmentationModel.values)
                      ChoiceChip(
                        label: Text(m.name),
                        selected: segmentationModel == m,
                        onSelected: (_) async {
                          both(() => segmentationModel = m);
                          await restartDetector();
                        },
                      ),
                  ],
                ),
                SwitchListTile(
                  dense: true,
                  title: const Text('Virtual background'),
                  value: showVirtualBackground,
                  onChanged: (v) => both(() => showVirtualBackground = v),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      const SizedBox(width: 100, child: Text('Threshold')),
                      Expanded(
                        child: Slider(
                          value: segmentationThreshold,
                          onChanged: (v) =>
                              both(() => segmentationThreshold = v),
                        ),
                      ),
                      Text(segmentationThreshold.toStringAsFixed(2)),
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
                        onTap: () => both(() => maskColor = c),
                        child: Container(
                          margin: const EdgeInsets.only(right: 4),
                          width: 22,
                          height: 22,
                          decoration: BoxDecoration(
                            color: c,
                            border: Border.all(
                              color: maskColor.toARGB32() == c.toARGB32()
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
                value: useLiteRt,
                onChanged: (v) async {
                  both(() => useLiteRt = v);
                  await restartDetector();
                },
              ),
              Wrap(
                spacing: 6,
                children: [
                  const Text('Accelerator:'),
                  for (final a in const <String>['auto', 'webgpu', 'wasm'])
                    ChoiceChip(
                      label: Text(a),
                      selected: liteRtAccelerator == a,
                      onSelected: (_) async {
                        both(() => liteRtAccelerator = a);
                        await restartDetector();
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
