// Headless driver: runs the real FaceDetector in FULL mode (mesh + iris) with
// the 6-class multiclass segmentation model over every .mp4 in a clips dir, and
// dumps, per clip:
//   - data.json : per-frame face geometry (box, score, 6 landmarks, 468 mesh
//                 points, per-eye iris center/contour + eyelid contour, head
//                 Euler angles)
//   - mask/mNNNN.png : per-frame argmax class-index map (CV_8UC1, values 0..5)
// Python then renders the two README demo looks (full-mode overlay + 6-class
// segmentation) from this dump.
//
// Run on macOS desktop from the example directory. By default it processes the
// bundled sample clip under assets/samples (so a fresh clone works with no
// setup) and writes the dump under the system temp dir. Override the input or
// output with FACE_CLIPS_DIR / FACE_OUT_ROOT, and FACE_ONLY filters to clips
// whose file name contains the given string:
//   cd example && flutter test integration_test/gen_face_demo_dump_test.dart -d macos
//   cd example && FACE_CLIPS_DIR=/path/to/clips \
//     flutter test integration_test/gen_face_demo_dump_test.dart -d macos
//
// ignore_for_file: avoid_print
@TestOn('mac-os')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart' show OneEuroFilter;
import 'package:face_detection_tflite/face_detection_tflite_native.dart';

// Input/output defaults are computed in the test body (see clipsDir/outRoot):
// the bundled sample under assets/samples, and the system temp dir. Both are
// overridable via the FACE_CLIPS_DIR / FACE_OUT_ROOT environment variables.

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('dump face detections + segmentation for demo generation',
      (tester) async {
    final String? only = Platform.environment['FACE_ONLY'];
    final String clipsDir =
        Platform.environment['FACE_CLIPS_DIR'] ?? 'assets/samples';
    final String outRoot = Platform.environment['FACE_OUT_ROOT'] ??
        '${Directory.systemTemp.path}/face_dump';
    // Match the live-camera example: the back-camera model plus the One-Euro
    // FaceSmoother (both on by default there). Override via FACE_MODEL (one of
    // frontCamera/backCamera/shortRange/full/fullSparse) and FACE_SMOOTH=0.
    final String modelName = Platform.environment['FACE_MODEL'] ?? 'backCamera';
    final FaceDetectionModel model = FaceDetectionModel.values
        .firstWhere((m) => m.name == modelName, orElse: () {
      throw StateError('unknown FACE_MODEL "$modelName"');
    });
    final bool smooth = Platform.environment['FACE_SMOOTH'] != '0';
    print('gen_face_demo_dump: reading clips from "$clipsDir", '
        'writing dump to "$outRoot" (model=${model.name}, smooth=$smooth)');

    final detector = await FaceDetector.create(
      model: model,
      withSegmentation: true,
      segmentationConfig:
          SegmentationConfig(model: SegmentationModel.multiclass),
    );

    final clips = Directory(clipsDir)
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.mp4'))
        .where((f) => only == null || f.path.contains(only))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    expect(clips.isNotEmpty, true,
        reason: 'no .mp4 clips in "$clipsDir". Run from the example directory, '
            'or set FACE_CLIPS_DIR to a folder containing .mp4 clips.');

    for (final clip in clips) {
      final name = clip.uri.pathSegments.last.replaceAll('.mp4', '');
      final outDir = '$outRoot/$name';
      Directory('$outDir/mask').createSync(recursive: true);

      final smoother = FaceSmoother(enabled: smooth);
      final cap = cv.VideoCapture.fromFile(clip.path);
      expect(cap.isOpened, true, reason: 'cannot open ${clip.path}');
      final double fps = cap.get(cv.CAP_PROP_FPS);
      final double tStep = 1.0 / (fps > 0 ? fps : 30.0);
      final int vw = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
      final int vh = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();

      final frames = <Map<String, dynamic>>[];
      cv.Mat? frame;
      int idx = 0;
      int totalFaces = 0;
      String maskInfo = 'none';
      while (true) {
        final res = cap.read(m: frame);
        if (!res.$1) break;
        frame = res.$2;
        if (frame.isEmpty) break;

        final r = await detector.detectFacesWithSegmentationFromMat(
          frame,
          mode: FaceDetectionMode.full,
        );
        // Same temporal smoothing the live-camera example applies by default:
        // IoU face tracking + a One-Euro filter on every mesh and iris point.
        final faces = smoother.apply(r.faces, idx * tStep);

        final facesJson = <Map<String, dynamic>>[];
        for (final f in faces) {
          final bb = f.boundingBox;
          final mesh = <double>[];
          final m = f.mesh;
          if (m != null) {
            for (final p in m.points) {
              mesh.add(p.x);
              mesh.add(p.y);
            }
          }
          final landmarks = <List<double>>[];
          f.landmarks.toMap().forEach((_, p) {
            landmarks.add([p.x, p.y]);
          });
          Map<String, dynamic>? eyesJson;
          final eyes = f.eyes;
          if (eyes != null) {
            eyesJson = <String, dynamic>{};
            final pairs = {'left': eyes.leftEye, 'right': eyes.rightEye};
            pairs.forEach((side, e) {
              if (e == null) return;
              eyesJson![side] = {
                'irisCenter': [e.irisCenter.x, e.irisCenter.y],
                'irisContour': [
                  for (final p in e.irisContour) [p.x, p.y]
                ],
                'contour': [
                  for (final p in e.contour) [p.x, p.y]
                ],
              };
            });
          }
          final hp = f.headEulerAngles;
          facesJson.add({
            'score': f.score,
            'box': [bb.left, bb.top, bb.right, bb.bottom],
            'landmarks': landmarks,
            'mesh': mesh,
            if (eyesJson != null && eyesJson.isNotEmpty) 'eyes': eyesJson,
            if (hp != null) 'head': [hp.x, hp.y, hp.z],
          });
        }
        totalFaces += faces.length;

        Map<String, dynamic>? maskMeta;
        final mask = r.segmentationMask;
        if (mask is MulticlassSegmentationMask) {
          final int w = mask.width, h = mask.height;
          final cd = mask.internalClassData;
          final argmax = Uint8List(w * h);
          for (int i = 0; i < w * h; i++) {
            final base = i * 6;
            int best = 0;
            double bv = cd[base];
            for (int c = 1; c < 6; c++) {
              final v = cd[base + c];
              if (v > bv) {
                bv = v;
                best = c;
              }
            }
            argmax[i] = best;
          }
          final mm = cv.Mat.create(rows: h, cols: w, type: cv.MatType.CV_8UC1);
          mm.data.setAll(0, argmax);
          cv.imwrite('$outDir/mask/m${idx.toString().padLeft(4, '0')}.png', mm);
          mm.dispose();
          maskMeta = {'w': w, 'h': h, 'padding': mask.padding};
          maskInfo = '${w}x$h pad=${mask.padding}';
        }

        frames.add({
          'i': idx,
          'faces': facesJson,
          if (maskMeta != null) 'mask': maskMeta,
        });
        idx++;
      }
      frame?.dispose();
      cap.release();

      File('$outDir/data.json').writeAsStringSync(
        jsonEncode({'fps': fps, 'w': vw, 'h': vh, 'frames': frames}),
      );
      print('FACE_DUMP $name frames=$idx faces_total=$totalFaces '
          '${vw}x$vh fps=${fps.toStringAsFixed(2)} mask=$maskInfo -> $outDir');
    }

    await detector.dispose();
  }, timeout: const Timeout(Duration(minutes: 45)));
}

// ─────────────────────────── Face Smoother ────────────────────────────────
// Copied verbatim from example/lib/main.dart so the dumped geometry matches
// exactly what the live-camera demo renders: IoU face tracking plus a One-Euro
// filter on every mesh point and iris point. eyes/head/blendshapes are all
// recomputed from the smoothed mesh + irises by the Face constructor.

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
      mesh: FaceMesh(smoothedPoints, score: mesh.score),
      irises: smoothedIrises,
      blendshapeScores: face.blendshapes?.scores,
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
