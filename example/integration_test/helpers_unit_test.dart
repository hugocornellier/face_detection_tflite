// ignore_for_file: avoid_print

/// Integration tests for @visibleForTesting helper functions in helpers.dart.
///
/// Tests cover:
/// - testClip: value clamping
/// - testSigmoidClipped: sigmoid activation with overflow protection
/// - testDetectionLetterboxRemoval: letterbox padding removal from detections
/// - testUnpackLandmarks: landmark tensor unpacking with padding
/// - testNms: Non-Maximum Suppression
library;

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testTimeout = Timeout(Duration(minutes: 2));

  // ===========================================================================
  // testClip
  // ===========================================================================
  group('testClip', () {
    test('value within range returns unchanged', () {
      expect(testClip(0.5, 0.0, 1.0), 0.5);
      expect(testClip(0.0, 0.0, 1.0), 0.0);
      expect(testClip(1.0, 0.0, 1.0), 1.0);
      print('testClip: in-range values passed');
    }, timeout: testTimeout);

    test('value below lo returns lo', () {
      expect(testClip(-1.0, 0.0, 1.0), 0.0);
      expect(testClip(-100.0, -5.0, 5.0), -5.0);
      print('testClip: below-lo values passed');
    }, timeout: testTimeout);

    test('value above hi returns hi', () {
      expect(testClip(2.0, 0.0, 1.0), 1.0);
      expect(testClip(100.0, -5.0, 5.0), 5.0);
      print('testClip: above-hi values passed');
    }, timeout: testTimeout);

    test('lo equals hi clamps to that value', () {
      expect(testClip(0.5, 0.3, 0.3), 0.3);
      expect(testClip(-1.0, 0.3, 0.3), 0.3);
      expect(testClip(1.0, 0.3, 0.3), 0.3);
      print('testClip: lo==hi passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // testSigmoidClipped
  // ===========================================================================
  group('testSigmoidClipped', () {
    test('input 0.0 returns 0.5', () {
      expect(testSigmoidClipped(0.0), closeTo(0.5, 0.001));
      print('testSigmoidClipped: zero input passed');
    }, timeout: testTimeout);

    test('large positive input returns close to 1.0', () {
      final result = testSigmoidClipped(10.0);
      expect(result, greaterThan(0.99));
      expect(result, lessThanOrEqualTo(1.0));
      print('testSigmoidClipped: large positive = $result');
    }, timeout: testTimeout);

    test('large negative input returns close to 0.0', () {
      final result = testSigmoidClipped(-10.0);
      expect(result, lessThan(0.01));
      expect(result, greaterThanOrEqualTo(0.0));
      print('testSigmoidClipped: large negative = $result');
    }, timeout: testTimeout);

    test('very large input is clipped and does not overflow', () {
      // Default limit is 80.0; values beyond are clipped before exp()
      final result1 = testSigmoidClipped(1000.0);
      final result2 = testSigmoidClipped(-1000.0);
      expect(result1, closeTo(1.0, 0.001));
      expect(result2, closeTo(0.0, 0.001));
      expect(result1.isFinite, true);
      expect(result2.isFinite, true);
      print(
        'testSigmoidClipped: overflow protection: '
        'sig(1000)=$result1, sig(-1000)=$result2',
      );
    }, timeout: testTimeout);

    test('custom limit clips at that value', () {
      final result = testSigmoidClipped(5.0, limit: 2.0);
      // 5.0 is clipped to 2.0 before sigmoid
      final expected = testSigmoidClipped(2.0);
      expect(result, closeTo(expected, 0.001));
      print('testSigmoidClipped: custom limit passed');
    }, timeout: testTimeout);

    test('output always in [0, 1] range', () {
      for (final x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]) {
        final result = testSigmoidClipped(x);
        expect(result, greaterThanOrEqualTo(0.0));
        expect(result, lessThanOrEqualTo(1.0));
      }
      print('testSigmoidClipped: range check passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // testDetectionLetterboxRemoval
  // ===========================================================================
  group('testDetectionLetterboxRemoval', () {
    Detection makeDetection(
      double xmin,
      double ymin,
      double xmax,
      double ymax,
      double score,
      List<double> keypoints,
    ) {
      return Detection(
        boundingBox: RectF(xmin, ymin, xmax, ymax),
        score: score,
        keypointsXY: keypoints,
      );
    }

    test('no padding returns unchanged detections', () {
      final det = makeDetection(0.2, 0.3, 0.8, 0.9, 0.95, [0.5, 0.5]);
      final result = testDetectionLetterboxRemoval([det], [0, 0, 0, 0]);

      expect(result.length, 1);
      expect(result[0].boundingBox.xmin, closeTo(0.2, 0.001));
      expect(result[0].boundingBox.ymin, closeTo(0.3, 0.001));
      expect(result[0].boundingBox.xmax, closeTo(0.8, 0.001));
      expect(result[0].boundingBox.ymax, closeTo(0.9, 0.001));
      expect(result[0].keypointsXY[0], closeTo(0.5, 0.001));
      expect(result[0].keypointsXY[1], closeTo(0.5, 0.001));
      print('letterboxRemoval: no padding passed');
    }, timeout: testTimeout);

    test('top/bottom padding rescales Y coordinates', () {
      // Landscape image: padding at top=0.1, bottom=0.1, left=0, right=0
      final padding = [0.1, 0.1, 0.0, 0.0];
      final det = makeDetection(0.2, 0.3, 0.8, 0.7, 0.95, [0.5, 0.5]);
      final result = testDetectionLetterboxRemoval([det], padding);

      expect(result.length, 1);
      // X should be unchanged (no left/right padding)
      expect(result[0].boundingBox.xmin, closeTo(0.2, 0.001));
      expect(result[0].boundingBox.xmax, closeTo(0.8, 0.001));
      // Y should be rescaled: (0.3 - 0.1) / 0.8 = 0.25
      expect(result[0].boundingBox.ymin, closeTo(0.25, 0.001));
      // (0.7 - 0.1) / 0.8 = 0.75
      expect(result[0].boundingBox.ymax, closeTo(0.75, 0.001));
      print('letterboxRemoval: top/bottom padding passed');
    }, timeout: testTimeout);

    test('left/right padding rescales X coordinates', () {
      // Portrait image: padding at top=0, bottom=0, left=0.15, right=0.15
      final padding = [0.0, 0.0, 0.15, 0.15];
      final det = makeDetection(0.3, 0.2, 0.7, 0.8, 0.95, [0.5, 0.5]);
      final result = testDetectionLetterboxRemoval([det], padding);

      expect(result.length, 1);
      // Y should be unchanged
      expect(result[0].boundingBox.ymin, closeTo(0.2, 0.001));
      expect(result[0].boundingBox.ymax, closeTo(0.8, 0.001));
      // X should be rescaled: (0.3 - 0.15) / 0.7 = ~0.2143
      expect(result[0].boundingBox.xmin, closeTo(0.2143, 0.001));
      // (0.7 - 0.15) / 0.7 = ~0.7857
      expect(result[0].boundingBox.xmax, closeTo(0.7857, 0.001));
      print('letterboxRemoval: left/right padding passed');
    }, timeout: testTimeout);

    test('keypoints are also unpadded', () {
      final padding = [0.1, 0.1, 0.0, 0.0];
      final det = makeDetection(
        0.2,
        0.3,
        0.8,
        0.7,
        0.95,
        [0.5, 0.5, 0.2, 0.3],
      );
      final result = testDetectionLetterboxRemoval([det], padding);

      // kp[1] = (0.5 - 0.1) / 0.8 = 0.5
      expect(result[0].keypointsXY[1], closeTo(0.5, 0.001));
      // kp[3] = (0.3 - 0.1) / 0.8 = 0.25
      expect(result[0].keypointsXY[3], closeTo(0.25, 0.001));
      print('letterboxRemoval: keypoints unpadded passed');
    }, timeout: testTimeout);

    test('score is preserved', () {
      final det = makeDetection(0.2, 0.3, 0.8, 0.7, 0.95, []);
      final result = testDetectionLetterboxRemoval([det], [0.1, 0.1, 0, 0]);

      expect(result[0].score, 0.95);
      print('letterboxRemoval: score preserved passed');
    }, timeout: testTimeout);

    test('multiple detections all unpadded', () {
      final padding = [0.1, 0.1, 0.0, 0.0];
      final dets = [
        makeDetection(0.1, 0.2, 0.3, 0.4, 0.9, []),
        makeDetection(0.5, 0.6, 0.7, 0.8, 0.8, []),
      ];
      final result = testDetectionLetterboxRemoval(dets, padding);

      expect(result.length, 2);
      expect(result[0].score, 0.9);
      expect(result[1].score, 0.8);
      print('letterboxRemoval: multiple detections passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // testUnpackLandmarks
  // ===========================================================================
  group('testUnpackLandmarks', () {
    test('no padding produces normalized coords', () {
      // 2 landmarks at pixel coords in a 100x100 model space
      // Landmark 0: (50, 30, 1.0), Landmark 1: (75, 60, 2.0)
      final flat = Float32List.fromList([50, 30, 1.0, 75, 60, 2.0]);
      final result = testUnpackLandmarks(flat, 100, 100, [0, 0, 0, 0]);

      expect(result.length, 2);
      // (50/100 - 0) / 1.0 = 0.5
      expect(result[0][0], closeTo(0.5, 0.001));
      // (30/100 - 0) / 1.0 = 0.3
      expect(result[0][1], closeTo(0.3, 0.001));
      expect(result[0][2], closeTo(1.0, 0.001)); // z preserved
      expect(result[1][0], closeTo(0.75, 0.001));
      expect(result[1][1], closeTo(0.6, 0.001));
      expect(result[1][2], closeTo(2.0, 0.001));
      print('unpackLandmarks: no padding passed');
    }, timeout: testTimeout);

    test('with padding, removes padding from coordinates', () {
      // padding [top=0.1, bottom=0.1, left=0, right=0]
      // sy = 1.0 - 0.2 = 0.8
      // Landmark at pixel (50, 50) in 100x100 model
      // normalized: x=0.5, y=0.5
      // unpadded: x = (0.5 - 0) / 1.0 = 0.5, y = (0.5 - 0.1) / 0.8 = 0.5
      final flat = Float32List.fromList([50, 50, 0.0]);
      final result = testUnpackLandmarks(flat, 100, 100, [0.1, 0.1, 0, 0]);

      expect(result.length, 1);
      expect(result[0][0], closeTo(0.5, 0.001));
      expect(result[0][1], closeTo(0.5, 0.001));
      print('unpackLandmarks: with padding passed');
    }, timeout: testTimeout);

    test('clamp=true clamps out-of-range to [0,1]', () {
      // Landmark at pixel (-10, 110) in 100x100 model, no padding
      // normalized: x=-0.1, y=1.1 -> clamped to x=0.0, y=1.0
      final flat = Float32List.fromList([-10, 110, 0.0]);
      final result = testUnpackLandmarks(
        flat,
        100,
        100,
        [0, 0, 0, 0],
        clamp: true,
      );

      expect(result[0][0], 0.0);
      expect(result[0][1], 1.0);
      print('unpackLandmarks: clamp=true passed');
    }, timeout: testTimeout);

    test('clamp=false preserves out-of-range values', () {
      final flat = Float32List.fromList([-10, 110, 0.0]);
      final result = testUnpackLandmarks(
        flat,
        100,
        100,
        [0, 0, 0, 0],
        clamp: false,
      );

      expect(result[0][0], closeTo(-0.1, 0.001));
      expect(result[0][1], closeTo(1.1, 0.001));
      print('unpackLandmarks: clamp=false passed');
    }, timeout: testTimeout);

    test('returns [x, y, z] triples', () {
      final flat = Float32List.fromList([10, 20, 3.5, 40, 50, 6.7]);
      final result = testUnpackLandmarks(flat, 100, 100, [0, 0, 0, 0]);

      expect(result.length, 2);
      for (final lm in result) {
        expect(lm.length, 3);
      }
      expect(result[0][2], closeTo(3.5, 0.001));
      expect(result[1][2], closeTo(6.7, 0.001));
      print('unpackLandmarks: xyz triples passed');
    }, timeout: testTimeout);
  });

  // ===========================================================================
  // testNms
  // ===========================================================================
  group('testNms', () {
    Detection makeDetection(
      double xmin,
      double ymin,
      double xmax,
      double ymax,
      double score,
    ) {
      return Detection(
        boundingBox: RectF(xmin, ymin, xmax, ymax),
        score: score,
        keypointsXY: const [],
      );
    }

    test('empty input returns empty list', () {
      final result = testNms([], 0.5, 0.5);
      expect(result, isEmpty);
      print('nms: empty input passed');
    }, timeout: testTimeout);

    test('single detection above threshold is kept', () {
      final det = makeDetection(0.1, 0.1, 0.5, 0.5, 0.9);
      final result = testNms([det], 0.5, 0.5);
      expect(result.length, 1);
      expect(result[0].score, 0.9);
      print('nms: single above threshold passed');
    }, timeout: testTimeout);

    test('single detection below threshold is removed', () {
      final det = makeDetection(0.1, 0.1, 0.5, 0.5, 0.3);
      final result = testNms([det], 0.5, 0.5);
      expect(result, isEmpty);
      print('nms: single below threshold passed');
    }, timeout: testTimeout);

    test('overlapping detections: lower-scoring suppressed', () {
      // Two nearly identical boxes
      final det1 = makeDetection(0.1, 0.1, 0.5, 0.5, 0.9);
      final det2 = makeDetection(0.12, 0.12, 0.52, 0.52, 0.7);
      final result = testNms([det1, det2], 0.3, 0.5);

      // High overlap -> only highest-scoring kept (or weighted merge)
      expect(result.length, 1);
      print(
          'nms: overlapping suppression passed, kept score=${result[0].score}');
    }, timeout: testTimeout);

    test('non-overlapping detections: both kept', () {
      final det1 = makeDetection(0.0, 0.0, 0.2, 0.2, 0.9);
      final det2 = makeDetection(0.8, 0.8, 1.0, 1.0, 0.8);
      final result = testNms([det1, det2], 0.3, 0.5);

      expect(result.length, 2);
      print('nms: non-overlapping passed');
    }, timeout: testTimeout);

    test('many detections: only non-overlapping kept', () {
      // Create 3 clusters of overlapping detections
      final dets = <Detection>[
        // Cluster 1: top-left
        makeDetection(0.0, 0.0, 0.3, 0.3, 0.95),
        makeDetection(0.02, 0.02, 0.32, 0.32, 0.85),
        makeDetection(0.01, 0.01, 0.31, 0.31, 0.80),
        // Cluster 2: center
        makeDetection(0.4, 0.4, 0.6, 0.6, 0.90),
        makeDetection(0.42, 0.42, 0.62, 0.62, 0.75),
        // Cluster 3: bottom-right
        makeDetection(0.7, 0.7, 1.0, 1.0, 0.88),
      ];
      final result = testNms(dets, 0.3, 0.5);

      // Should keep ~3 detections (one per cluster)
      expect(result.length, 3);
      print('nms: multi-cluster passed, kept ${result.length} detections');
    }, timeout: testTimeout);

    test('weighted=false uses standard NMS', () {
      final det1 = makeDetection(0.1, 0.1, 0.5, 0.5, 0.9);
      final det2 = makeDetection(0.12, 0.12, 0.52, 0.52, 0.7);
      final result = testNms([det1, det2], 0.3, 0.5, weighted: false);

      expect(result.length, 1);
      // Without weighting, should keep exact highest-scoring box
      expect(result[0].boundingBox.xmin, closeTo(0.1, 0.001));
      print('nms: weighted=false passed');
    }, timeout: testTimeout);

    test('score threshold filters before NMS', () {
      final dets = [
        makeDetection(0.0, 0.0, 0.3, 0.3, 0.9),
        makeDetection(0.5, 0.5, 0.8, 0.8, 0.1), // Below threshold
        makeDetection(0.5, 0.5, 0.8, 0.8, 0.8),
      ];
      final result = testNms(dets, 0.3, 0.5);

      // The 0.1-scored detection should be filtered out
      for (final r in result) {
        expect(r.score, greaterThanOrEqualTo(0.5));
      }
      print('nms: score threshold filtering passed');
    }, timeout: testTimeout);
  });
}
