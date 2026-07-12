// ignore_for_file: avoid_print

// A/B benchmark + parity harness for the perf/validated-optimizations branch.
//
// Prints one JSON document between AB_BENCH_JSON_START / AB_BENCH_JSON_END
// markers. A runner script extracts it and a comparer diffs BEFORE/AFTER.
//
// Scenarios (all default engine: Interpreter/XNNPACK, backCamera model):
//   s1_gated_group_full   gated detector (minFaceSize keeps largest face),
//                         group shot, full mode. Target of early gating.
//   s2_embedding          detect once, then getFaceEmbedding in a loop.
//   s3_single_full        single-face image, full mode loop.
//   s4_two_faces_std      two-face image, standard mode loop.
//   s5_group_full         ungated group shot, full mode loop.
//
// Parity: full-precision output hashes (FNV-1a 64) for ungated and gated
// detections across fast/standard/full, plus raw dump of the first face and
// the embedding vector. Any behavior change flips a hash.

import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite_native.dart';

const int kWarmup = 5;
const int kIters = 100;

const String kGroupShot = 'assets/samples/group-shot-bounding-box-ex1.jpeg';
const String kSingleFace = 'assets/samples/landmark-ex1.jpg';
const String kTwoFaces = 'assets/samples/embedding_test/two_faces.jpg';

Future<Uint8List> _load(String key) async =>
    (await rootBundle.load(key)).buffer.asUint8List();

Map<String, dynamic> _stats(List<double> ms) {
  final sorted = List<double>.from(ms)..sort();
  final n = sorted.length;
  double at(double q) => sorted[(q * (n - 1)).round()];
  final mean = sorted.reduce((a, b) => a + b) / n;
  final variance =
      sorted.map((v) => (v - mean) * (v - mean)).reduce((a, b) => a + b) / n;
  return {
    'n': n,
    'mean_ms': mean,
    'median_ms': at(0.5),
    'p10_ms': at(0.10),
    'p90_ms': at(0.90),
    'min_ms': sorted.first,
    'max_ms': sorted.last,
    'stddev_ms': varianceSqrt(variance),
    'raw_ms': ms,
  };
}

double varianceSqrt(double v) {
  // Newton sqrt to avoid importing dart:math just for this.
  if (v <= 0) return 0;
  double x = v;
  for (int i = 0; i < 40; i++) {
    x = 0.5 * (x + v / x);
  }
  return x;
}

/// FNV-1a 64-bit over a string. Dart VM ints wrap at 64 bits.
int _fnv1a(String s) {
  int h = 0xcbf29ce484222325;
  for (final c in s.codeUnits) {
    h ^= c & 0xff;
    h *= 0x100000001b3;
    h ^= c >> 8;
    h *= 0x100000001b3;
  }
  return h;
}

String _d(double? v) => v == null ? 'n' : v.toString();

String _pointsCanonical(List<Point> pts) =>
    pts.map((p) => '${_d(p.x)},${_d(p.y)},${_d(p.z)}').join(';');

/// Canonical full-precision string of every public output of a face.
String _faceCanonical(Face f) {
  final bb = f.detectionData.boundingBox;
  final lm = f.landmarks;
  final ang = f.headEulerAngles;
  final parts = <String>[
    'score=${_d(f.score)}',
    'bbox=${_d(bb.xmin)},${_d(bb.ymin)},${_d(bb.xmax)},${_d(bb.ymax)}',
    'kp=${f.detectionData.keypointsXY.map(_d).join(',')}',
    'size=${_d(f.originalSize.width)},${_d(f.originalSize.height)}',
    'wf=${_d(f.widthFraction)}',
    'meshScore=${_d(f.meshScore)}',
    'mesh=${f.mesh == null ? 'n' : _pointsCanonical(f.mesh!.points)}',
    'iris=${_pointsCanonical(f.irisPoints)}',
    'bs=${f.blendshapes == null ? 'n' : f.blendshapes!.scores.map(_d).join(',')}',
    'ang=${ang == null ? 'n' : '${_d(ang.x)},${_d(ang.y)},${_d(ang.z)}'}',
    'lmL=${lm.leftEye == null ? 'n' : '${_d(lm.leftEye!.x)},${_d(lm.leftEye!.y)}'}',
    'lmR=${lm.rightEye == null ? 'n' : '${_d(lm.rightEye!.x)},${_d(lm.rightEye!.y)}'}',
    'lmN=${lm.noseTip == null ? 'n' : '${_d(lm.noseTip!.x)},${_d(lm.noseTip!.y)}'}',
    'lmM=${lm.mouth == null ? 'n' : '${_d(lm.mouth!.x)},${_d(lm.mouth!.y)}'}',
  ];
  return parts.join('|');
}

Map<String, dynamic> _firstFaceRaw(List<Face> faces) {
  if (faces.isEmpty) return {'empty': true};
  final f = faces.first;
  final bb = f.detectionData.boundingBox;
  return {
    'score': f.score,
    'bbox': [bb.xmin, bb.ymin, bb.xmax, bb.ymax],
    'widthFraction': f.widthFraction,
    'meshScore': f.meshScore,
    'mesh_first9': f.mesh == null
        ? null
        : [
            for (final p in f.mesh!.points.take(3)) [p.x, p.y, p.z],
          ],
    'iris_first3': [
      for (final p in f.irisPoints.take(3)) [p.x, p.y, p.z],
    ],
    'blendshapes_first5': f.blendshapes?.scores.take(5).toList(),
  };
}

Future<List<double>> _timeLoop(Future<void> Function() op) async {
  for (int i = 0; i < kWarmup; i++) {
    await op();
  }
  final out = <double>[];
  final sw = Stopwatch();
  for (int i = 0; i < kIters; i++) {
    sw
      ..reset()
      ..start();
    await op();
    sw.stop();
    out.add(sw.elapsedMicroseconds / 1000.0);
  }
  return out;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test(
    'ab bench and parity',
    () async {
      final result = <String, dynamic>{
        'timestamp': DateTime.now().toIso8601String(),
        'warmup': kWarmup,
        'iters': kIters,
      };

      final groupBytes = await _load(kGroupShot);
      final singleBytes = await _load(kSingleFace);
      final twoBytes = await _load(kTwoFaces);

      // Probe: ungated detection on the group shot to derive the S1 gate
      // threshold (midpoint between the two largest width fractions, so the
      // gate keeps exactly the largest face).
      final probe = await FaceDetector.create();
      final probeFaces = await probe.detectFacesFromBytes(
        groupBytes,
        mode: FaceDetectionMode.fast,
      );
      final widths = probeFaces.map((f) => f.widthFraction).toList()
        ..sort((a, b) => b.compareTo(a));
      expect(widths.length, greaterThan(1),
          reason: 'group shot must contain multiple faces');
      final double gateThreshold = (widths[0] + widths[1]) / 2.0;
      result['probe'] = {
        'group_faces': widths.length,
        'width_fractions_desc': widths,
        'gate_threshold': gateThreshold,
      };

      // ---------- Parity: ungated, both images, all modes ----------
      final parity = <String, dynamic>{};
      for (final entry in {
        'group': groupBytes,
        'single': singleBytes,
      }.entries) {
        for (final mode in FaceDetectionMode.values) {
          final faces = await probe.detectFacesFromBytes(
            entry.value,
            mode: mode,
          );
          final canonical =
              '${faces.map(_faceCanonical).join('#')}|count=${faces.length}';
          parity['ungated_${entry.key}_${mode.name}'] = {
            'count': faces.length,
            'hash': _fnv1a(canonical).toRadixString(16),
            'first_face': _firstFaceRaw(faces),
          };
        }
      }

      // Embedding parity: full-precision vector for the single-face image.
      final embFaces = await probe.detectFacesFromBytes(
        singleBytes,
        mode: FaceDetectionMode.full,
      );
      final emb = await probe.getFaceEmbedding(embFaces.first, singleBytes);
      parity['embedding_single'] = {
        'len': emb.length,
        'hash':
            _fnv1a(emb.map((v) => v.toString()).join(',')).toRadixString(16),
        'first8': emb.take(8).toList(),
      };
      await probe.dispose();

      // Parity: gated detector on the group shot, all modes.
      final gatedParity = await FaceDetector.create(
        minFaceSize: gateThreshold,
      );
      for (final mode in FaceDetectionMode.values) {
        final faces = await gatedParity.detectFacesFromBytes(
          groupBytes,
          mode: mode,
        );
        final canonical =
            '${faces.map(_faceCanonical).join('#')}|count=${faces.length}';
        parity['gated_group_${mode.name}'] = {
          'count': faces.length,
          'hash': _fnv1a(canonical).toRadixString(16),
          'first_face': _firstFaceRaw(faces),
        };
      }
      await gatedParity.dispose();
      result['parity'] = parity;

      // ---------- S1: gated group shot, full mode ----------
      final bench = <String, dynamic>{};
      {
        final det = await FaceDetector.create(minFaceSize: gateThreshold);
        bench['s1_gated_group_full'] = _stats(await _timeLoop(() async {
          await det.detectFacesFromBytes(groupBytes,
              mode: FaceDetectionMode.full);
        }));
        await det.dispose();
      }

      // ---------- S2: embedding loop ----------
      {
        final det = await FaceDetector.create();
        final faces = await det.detectFacesFromBytes(singleBytes,
            mode: FaceDetectionMode.full);
        final face = faces.first;
        bench['s2_embedding'] = _stats(await _timeLoop(() async {
          await det.getFaceEmbedding(face, singleBytes);
        }));
        await det.dispose();
      }

      // ---------- S3: single face, full mode ----------
      {
        final det = await FaceDetector.create();
        bench['s3_single_full'] = _stats(await _timeLoop(() async {
          await det.detectFacesFromBytes(singleBytes,
              mode: FaceDetectionMode.full);
        }));
        await det.dispose();
      }

      // ---------- S4: two faces, standard mode ----------
      {
        final det = await FaceDetector.create();
        bench['s4_two_faces_std'] = _stats(await _timeLoop(() async {
          await det.detectFacesFromBytes(twoBytes,
              mode: FaceDetectionMode.standard);
        }));
        await det.dispose();
      }

      // ---------- S5: ungated group shot, full mode ----------
      {
        final det = await FaceDetector.create();
        bench['s5_group_full'] = _stats(await _timeLoop(() async {
          await det.detectFacesFromBytes(groupBytes,
              mode: FaceDetectionMode.full);
        }));
        await det.dispose();
      }

      result['bench'] = bench;

      print('AB_BENCH_JSON_START');
      print(const JsonEncoder.withIndent(' ').convert(result));
      print('AB_BENCH_JSON_END');
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );
}
