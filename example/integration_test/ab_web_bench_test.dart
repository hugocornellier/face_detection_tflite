// ignore_for_file: avoid_print

// Web A/B benchmark + parity harness for the perf/validated-optimizations
// branch. Run via flutter drive on Chrome (see runWebBenchmark.sh for the
// chromedriver setup). Prints one JSON document between
// AB_WEB_JSON_START / AB_WEB_JSON_END markers.
//
// Uses the WASM accelerator explicitly: WASM SIMD is deterministic, so
// parity hashes compare exactly across runs; WebGPU timing would be
// device/driver dependent and is not what these changes alter anyway.

import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

const int kWarmup = 8;
const int kIters = 60;

const String kGroupShot = 'assets/samples/group-shot-bounding-box-ex1.jpeg';

Future<Uint8List> _load(String key) async =>
    (await rootBundle.load(key)).buffer.asUint8List();

Map<String, dynamic> _stats(List<double> ms) {
  final sorted = List<double>.from(ms)..sort();
  final n = sorted.length;
  double at(double q) => sorted[(q * (n - 1)).round()];
  final mean = sorted.reduce((a, b) => a + b) / n;
  return {
    'n': n,
    'mean_ms': mean,
    'median_ms': at(0.5),
    'p10_ms': at(0.10),
    'p90_ms': at(0.90),
    'min_ms': sorted.first,
    'max_ms': sorted.last,
    'raw_ms': ms,
  };
}

/// JS-safe rolling hash (dart2js has no 64-bit ints): 47-bit modulus keeps
/// every intermediate below 2^53. Strength only needs to detect changes.
int _fnv1a(String s) {
  const int mod = 140737488355213; // largest prime below 2^47
  int h = 1125899906842597 % mod;
  for (final c in s.codeUnits) {
    h = (h * 31 + c) % mod;
  }
  return h;
}

String _d(double? v) => v == null ? 'n' : v.toString();

String _pointsCanonical(List<Point> pts) =>
    pts.map((p) => '${_d(p.x)},${_d(p.y)},${_d(p.z)}').join(';');

/// Exact raw values for numeric diffing across code versions: detector-level
/// outputs must match exactly; mesh-level values may carry the web runtime's
/// call-order noise and are diffed with a tolerance.
Map<String, dynamic> _faceRaw(Face f) {
  final bb = f.detectionData.boundingBox;
  return {
    'score': f.score,
    'bbox': [bb.xmin, bb.ymin, bb.xmax, bb.ymax],
    'kp': f.detectionData.keypointsXY,
    'wf': f.widthFraction,
    'meshScore': f.meshScore,
    'mesh_samples': f.mesh == null
        ? null
        : [
            for (final i in [0, 100, 300, 467])
              [f.mesh![i].x, f.mesh![i].y, f.mesh![i].z],
          ],
    'iris_first2': [
      for (final p in f.irisPoints.take(2)) [p.x, p.y, p.z],
    ],
    'bs_first3': f.blendshapes?.scores.take(3).toList(),
  };
}

String _faceCanonical(Face f) {
  final bb = f.detectionData.boundingBox;
  final lm = f.landmarks;
  return [
    'score=${_d(f.score)}',
    'bbox=${_d(bb.xmin)},${_d(bb.ymin)},${_d(bb.xmax)},${_d(bb.ymax)}',
    'kp=${f.detectionData.keypointsXY.map(_d).join(',')}',
    'wf=${_d(f.widthFraction)}',
    'meshScore=${_d(f.meshScore)}',
    'mesh=${f.mesh == null ? 'n' : _pointsCanonical(f.mesh!.points)}',
    'iris=${_pointsCanonical(f.irisPoints)}',
    'bs=${f.blendshapes == null ? 'n' : f.blendshapes!.scores.map(_d).join(',')}',
    'lmL=${lm.leftEye == null ? 'n' : '${_d(lm.leftEye!.x)},${_d(lm.leftEye!.y)}'}',
  ].join('|');
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
  final IntegrationTestWidgetsFlutterBinding binding =
      IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test(
    'web ab bench and parity',
    () async {
      final result = <String, dynamic>{
        'timestamp': DateTime.now().toIso8601String(),
        'warmup': kWarmup,
        'iters': kIters,
        'accelerator': 'wasm',
      };

      final groupBytes = await _load(kGroupShot);

      // Warm the detector before deriving the gate. LiteRT WASM's first two
      // calls on this model have slightly different boxes from its steady
      // state; a cold threshold can therefore keep one face initially but two
      // throughout the measured loop.
      final probe = await FaceDetector.create(liteRtAccelerator: 'wasm');
      for (int i = 0; i < kWarmup; i++) {
        await probe.detectFacesFromBytes(
          groupBytes,
          mode: FaceDetectionMode.fast,
        );
      }
      final probeFaces = await probe.detectFacesFromBytes(
        groupBytes,
        mode: FaceDetectionMode.fast,
      );
      final widths = probeFaces.map((f) => f.widthFraction).toList()
        ..sort((a, b) => b.compareTo(a));
      expect(widths.length, greaterThan(1));
      final double gateThreshold = (widths[0] + widths[1]) / 2.0;
      result['probe'] = {
        'group_faces': widths.length,
        'warmup_runs': kWarmup,
        'gate_threshold': gateThreshold,
        'expected_kept_faces': 1,
      };

      // Parity: ungated across modes.
      final parity = <String, dynamic>{};
      for (final mode in FaceDetectionMode.values) {
        final faces = await probe.detectFacesFromBytes(groupBytes, mode: mode);
        parity['ungated_group_${mode.name}'] = {
          'count': faces.length,
          'hash': _fnv1a(
            '${faces.map(_faceCanonical).join('#')}|count=${faces.length}',
          ).toRadixString(16),
        };
      }
      await probe.dispose();

      // Parity: gated across modes.
      final gatedDet = await FaceDetector.create(
        minFaceSize: gateThreshold,
        liteRtAccelerator: 'wasm',
      );
      for (final mode in FaceDetectionMode.values) {
        final faces = await gatedDet.detectFacesFromBytes(
          groupBytes,
          mode: mode,
        );
        expect(
          faces,
          hasLength(1),
          reason: 'steady-state gate must keep one face in ${mode.name}',
        );
        parity['gated_group_${mode.name}'] = {
          'count': faces.length,
          'hash': _fnv1a(
            '${faces.map(_faceCanonical).join('#')}|count=${faces.length}',
          ).toRadixString(16),
          'faces_raw': faces.map(_faceRaw).toList(),
        };
      }

      // Parity: combined faces + segmentation mask.
      final segDet = await FaceDetector.create(
        withSegmentation: true,
        liteRtAccelerator: 'wasm',
      );
      {
        final res = await segDet.detectFacesWithSegmentation(groupBytes);
        final mask = res.segmentationMask;
        final maskSample = <String>[];
        if (mask != null) {
          for (int i = 0; i < mask.internalData.length; i += 997) {
            maskSample.add(mask.internalData[i].toString());
          }
        }
        parity['seg_combo'] = {
          'faces': res.faces.length,
          'faces_hash':
              _fnv1a(res.faces.map(_faceCanonical).join('#')).toRadixString(16),
          'faces_raw': res.faces.map(_faceRaw).toList(),
          'mask_w': mask?.width,
          'mask_h': mask?.height,
          'mask_hash': _fnv1a(maskSample.join(',')).toRadixString(16),
          'mask_samples': [
            if (mask != null)
              for (int i = 0; i < mask.internalData.length; i += 9973)
                mask.internalData[i],
          ],
        };
      }
      result['parity'] = parity;

      // Bench scenarios.
      final bench = <String, dynamic>{};
      bench['w1_gated_group_full'] = _stats(await _timeLoop(() async {
        final faces = await gatedDet.detectFacesFromBytes(
          groupBytes,
          mode: FaceDetectionMode.full,
        );
        if (faces.length != 1) {
          fail(
            'w1_gated_group_full expected one face, got ${faces.length}',
          );
        }
      }));
      await gatedDet.dispose();

      {
        final det = await FaceDetector.create(liteRtAccelerator: 'wasm');
        bench['w5_group_full'] = _stats(await _timeLoop(() async {
          await det.detectFacesFromBytes(
            groupBytes,
            mode: FaceDetectionMode.full,
          );
        }));
        await det.dispose();
      }

      bench['w6_with_segmentation'] = _stats(await _timeLoop(() async {
        await segDet.detectFacesWithSegmentation(groupBytes);
      }));
      await segDet.dispose();

      result['bench'] = bench;

      // Browser prints don't reach the drive log in release mode; the
      // integration driver's responseDataCallback writes this to
      // benchmark_results/ab_web_latest.json on the host instead.
      binding.reportData = <String, dynamic>{'ab_web_latest.json': result};
      // Keep the JSON marker path as a fallback for local flutter run usage.
      print('AB_WEB_JSON_START');
      print(const JsonEncoder.withIndent(' ').convert(result));
      print('AB_WEB_JSON_END');
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );
}
