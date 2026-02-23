import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

void main() {
  globalTestSetup();

  group('testClip', () {
    test('should clamp value below min', () {
      expect(testClip(-5.0, 0.0, 10.0), 0.0);
    });

    test('should clamp value above max', () {
      expect(testClip(15.0, 0.0, 10.0), 10.0);
    });

    test('should return value within range', () {
      expect(testClip(5.0, 0.0, 10.0), 5.0);
    });

    test('should return min when value equals min', () {
      expect(testClip(0.0, 0.0, 10.0), 0.0);
    });

    test('should return max when value equals max', () {
      expect(testClip(10.0, 0.0, 10.0), 10.0);
    });

    test('should handle negative range', () {
      expect(testClip(0.0, -10.0, -5.0), -5.0);
    });
  });

  group('testSigmoidClipped', () {
    test('should return 0.5 for input 0', () {
      expect(testSigmoidClipped(0.0), closeTo(0.5, 0.0001));
    });

    test('should return near 1.0 for large positive input', () {
      expect(testSigmoidClipped(50.0), closeTo(1.0, 0.001));
    });

    test('should return near 0.0 for large negative input', () {
      expect(testSigmoidClipped(-50.0), closeTo(0.0, 0.001));
    });

    test('should be symmetric around 0.5', () {
      final pos = testSigmoidClipped(2.0);
      final neg = testSigmoidClipped(-2.0);
      expect(pos + neg, closeTo(1.0, 0.0001));
    });

    test('should respect custom limit', () {
      final clamped = testSigmoidClipped(200.0, limit: 5.0);
      final atLimit = testSigmoidClipped(5.0, limit: 5.0);
      expect(clamped, closeTo(atLimit, 0.0001));
    });
  });

  group('testDetectionLetterboxRemoval', () {
    Detection makeDet(double xmin, double ymin, double xmax, double ymax,
        {double score = 0.9}) {
      return Detection(
        boundingBox: RectF(xmin, ymin, xmax, ymax),
        score: score,
        keypointsXY: [
          xmin,
          ymin,
          xmax,
          ymax,
          (xmin + xmax) / 2,
          (ymin + ymax) / 2
        ],
      );
    }

    test('should pass through with zero padding', () {
      final dets = [makeDet(0.2, 0.3, 0.8, 0.7)];
      final result = testDetectionLetterboxRemoval(dets, [0.0, 0.0, 0.0, 0.0]);
      expect(result[0].boundingBox.xmin, closeTo(0.2, 0.0001));
      expect(result[0].boundingBox.ymin, closeTo(0.3, 0.0001));
      expect(result[0].boundingBox.xmax, closeTo(0.8, 0.0001));
      expect(result[0].boundingBox.ymax, closeTo(0.7, 0.0001));
    });

    test('should adjust for symmetric padding', () {
      final dets = [makeDet(0.5, 0.5, 0.5, 0.5)];
      final result = testDetectionLetterboxRemoval(dets, [0.1, 0.1, 0.1, 0.1]);
      // center of padded image (0.5) mapped to: (0.5 - 0.1) / 0.8 = 0.5
      expect(result[0].boundingBox.xmin, closeTo(0.5, 0.0001));
      expect(result[0].boundingBox.ymin, closeTo(0.5, 0.0001));
    });

    test('should handle asymmetric padding', () {
      // padding: top=0.2, bottom=0, left=0, right=0
      final dets = [makeDet(0.0, 0.2, 1.0, 1.0)];
      final result = testDetectionLetterboxRemoval(dets, [0.2, 0.0, 0.0, 0.0]);
      // y: (0.2 - 0.2) / 0.8 = 0.0
      expect(result[0].boundingBox.ymin, closeTo(0.0, 0.0001));
      expect(result[0].boundingBox.xmin, closeTo(0.0, 0.0001));
    });

    test('should return empty list for empty input', () {
      final result = testDetectionLetterboxRemoval([], [0.1, 0.1, 0.1, 0.1]);
      expect(result, isEmpty);
    });

    test('should preserve score', () {
      final dets = [makeDet(0.5, 0.5, 0.5, 0.5, score: 0.77)];
      final result = testDetectionLetterboxRemoval(dets, [0.0, 0.0, 0.0, 0.0]);
      expect(result[0].score, 0.77);
    });
  });

  group('testUnpackLandmarks', () {
    test('should unpack flat buffer to normalized coordinates', () {
      // 1 landmark at pixel (64, 64, 0.5) in a 128x128 image
      final flat = Float32List.fromList([64.0, 64.0, 0.5]);
      final result = testUnpackLandmarks(flat, 128, 128, [0.0, 0.0, 0.0, 0.0]);
      expect(result.length, 1);
      expect(result[0][0], closeTo(0.5, 0.0001)); // x
      expect(result[0][1], closeTo(0.5, 0.0001)); // y
      expect(result[0][2], closeTo(0.5, 0.0001)); // z preserved
    });

    test('should apply padding removal', () {
      // landmark at center of padded area
      final flat = Float32List.fromList([64.0, 64.0, 0.0]);
      // padding: left=0.25, right=0.25 → valid x range = [0.25, 0.75], sx=0.5
      final result =
          testUnpackLandmarks(flat, 128, 128, [0.0, 0.0, 0.25, 0.25]);
      // x = (64/128 - 0.25) / 0.5 = (0.5 - 0.25) / 0.5 = 0.5
      expect(result[0][0], closeTo(0.5, 0.0001));
    });

    test('should clamp to [0,1] by default', () {
      // landmark outside valid region
      final flat = Float32List.fromList([0.0, 0.0, 0.0]);
      final result =
          testUnpackLandmarks(flat, 128, 128, [0.25, 0.25, 0.25, 0.25]);
      // x = (0 - 0.25) / 0.5 = -0.5 → clamped to 0.0
      expect(result[0][0], 0.0);
      expect(result[0][1], 0.0);
    });

    test('should not clamp when clamp=false', () {
      final flat = Float32List.fromList([0.0, 0.0, 0.0]);
      final result = testUnpackLandmarks(
          flat, 128, 128, [0.25, 0.25, 0.25, 0.25],
          clamp: false);
      expect(result[0][0], lessThan(0.0));
    });

    test('should handle multiple landmarks', () {
      final flat = Float32List.fromList(
          [0.0, 0.0, 0.0, 64.0, 64.0, 1.0, 128.0, 128.0, 2.0]);
      final result = testUnpackLandmarks(flat, 128, 128, [0.0, 0.0, 0.0, 0.0]);
      expect(result.length, 3);
      expect(result[0][0], closeTo(0.0, 0.0001));
      expect(result[1][0], closeTo(0.5, 0.0001));
      expect(result[2][0], closeTo(1.0, 0.0001));
    });
  });

  group('testNms', () {
    Detection det(
        double xmin, double ymin, double xmax, double ymax, double score) {
      return Detection(
        boundingBox: RectF(xmin, ymin, xmax, ymax),
        score: score,
        keypointsXY: List.filled(12, 0.5),
      );
    }

    test('should return empty for empty input', () {
      final result = testNms([], 0.5, 0.5);
      expect(result, isEmpty);
    });

    test('should filter by score threshold', () {
      final dets = [det(0.0, 0.0, 0.5, 0.5, 0.3)];
      final result = testNms(dets, 0.5, 0.5);
      expect(result, isEmpty);
    });

    test('should keep non-overlapping detections', () {
      final dets = [
        det(0.0, 0.0, 0.2, 0.2, 0.9),
        det(0.8, 0.8, 1.0, 1.0, 0.8),
      ];
      final result = testNms(dets, 0.5, 0.5);
      expect(result.length, 2);
    });

    test('should suppress overlapping detections', () {
      final dets = [
        det(0.0, 0.0, 0.5, 0.5, 0.9),
        det(0.0, 0.0, 0.5, 0.5, 0.8), // identical box
      ];
      final result = testNms(dets, 0.3, 0.5);
      expect(result.length, 1);
    });

    test('should use weighted averaging in weighted mode', () {
      final dets = [
        det(0.0, 0.0, 0.5, 0.5, 0.9),
        det(0.05, 0.05, 0.55, 0.55, 0.8),
      ];
      final result = testNms(dets, 0.3, 0.5, weighted: true);
      expect(result.length, 1);
      // Weighted average should shift box slightly from highest-score box
      expect(result[0].boundingBox.xmin, greaterThan(0.0));
    });

    test('should not average in non-weighted mode', () {
      final dets = [
        det(0.0, 0.0, 0.5, 0.5, 0.9),
        det(0.05, 0.05, 0.55, 0.55, 0.8),
      ];
      final result = testNms(dets, 0.3, 0.5, weighted: false);
      expect(result.length, 1);
      // Should keep highest-score box unchanged
      expect(result[0].boundingBox.xmin, closeTo(0.0, 0.0001));
      expect(result[0].score, 0.9);
    });

    test('should preserve score of highest-confidence detection', () {
      final dets = [
        det(0.0, 0.0, 0.5, 0.5, 0.95),
        det(0.0, 0.0, 0.5, 0.5, 0.6),
      ];
      final result = testNms(dets, 0.3, 0.5);
      expect(result[0].score, 0.95);
    });

    test('should use grid-based NMS for >8 detections', () {
      // Create 10 non-overlapping detections spread across the image
      final dets = List.generate(
        10,
        (i) => det(i * 0.1, 0.0, i * 0.1 + 0.08, 0.08, 0.9 - i * 0.01),
      );
      final result = testNms(dets, 0.5, 0.5);
      expect(result.length, 10); // All should survive (no overlap)
    });
  });

  group('faceDetectionToRoi', () {
    test('should produce a square ROI', () {
      final roi = faceDetectionToRoi(RectF(0.3, 0.4, 0.7, 0.6));
      expect(roi.w, closeTo(roi.h, 0.0001));
    });

    test('should center on the expanded bbox center', () {
      final roi = faceDetectionToRoi(RectF(0.4, 0.4, 0.6, 0.6));
      final cx = (roi.xmin + roi.xmax) / 2;
      final cy = (roi.ymin + roi.ymax) / 2;
      expect(cx, closeTo(0.5, 0.0001));
      expect(cy, closeTo(0.5, 0.0001));
    });

    test('should expand by default factor (0.6)', () {
      final bbox = RectF(0.4, 0.4, 0.6, 0.6);
      final roi = faceDetectionToRoi(bbox);
      // expanded: w = 0.2 * 1.6 = 0.32, h = 0.32, half-side = 0.16
      expect(roi.w, closeTo(0.32, 0.0001));
    });

    test('should respect custom expandFraction', () {
      final bbox = RectF(0.4, 0.4, 0.6, 0.6);
      final roi = faceDetectionToRoi(bbox, expandFraction: 1.0);
      // expanded: w = 0.2 * 2.0 = 0.4
      expect(roi.w, closeTo(0.4, 0.0001));
    });

    test('should handle rectangular bbox (wider than tall)', () {
      final roi = faceDetectionToRoi(RectF(0.2, 0.4, 0.8, 0.6));
      // Always square, based on max(w, h) after expansion
      expect(roi.w, closeTo(roi.h, 0.0001));
      expect(roi.w, greaterThan(0.5)); // expanded wide bbox
    });
  });

  group('testSsdGenerateAnchors', () {
    test('should generate correct number of anchors for front model', () {
      final opts = testOptsFor(FaceDetectionModel.frontCamera);
      final anchors = testSsdGenerateAnchors(opts);
      // Each anchor is 2 values (x, y)
      expect(anchors.length.isEven, isTrue);
      expect(anchors.length, greaterThan(0));
    });

    test('should generate correct number of anchors for back model', () {
      final opts = testOptsFor(FaceDetectionModel.backCamera);
      final anchors = testSsdGenerateAnchors(opts);
      expect(anchors.length.isEven, isTrue);
      expect(anchors.length, greaterThan(0));
    });

    test('should generate correct number of anchors for short model', () {
      final opts = testOptsFor(FaceDetectionModel.shortRange);
      final anchors = testSsdGenerateAnchors(opts);
      expect(anchors.length.isEven, isTrue);
      expect(anchors.length, greaterThan(0));
    });

    test('should generate correct number of anchors for full model', () {
      final opts = testOptsFor(FaceDetectionModel.full);
      final anchors = testSsdGenerateAnchors(opts);
      expect(anchors.length.isEven, isTrue);
      expect(anchors.length, greaterThan(0));
    });

    test('should generate correct number of anchors for fullSparse model', () {
      final opts = testOptsFor(FaceDetectionModel.fullSparse);
      final anchors = testSsdGenerateAnchors(opts);
      expect(anchors.length.isEven, isTrue);
      expect(anchors.length, greaterThan(0));
    });

    test('anchor coordinates should be in (0, 1] range', () {
      final opts = testOptsFor(FaceDetectionModel.frontCamera);
      final anchors = testSsdGenerateAnchors(opts);
      for (int i = 0; i < anchors.length; i++) {
        expect(anchors[i], greaterThan(0.0));
        expect(anchors[i], lessThanOrEqualTo(1.0));
      }
    });

    test('front and short models should produce same anchor count', () {
      // Both use same strides and input size
      final frontAnchors =
          testSsdGenerateAnchors(testOptsFor(FaceDetectionModel.frontCamera));
      final shortAnchors =
          testSsdGenerateAnchors(testOptsFor(FaceDetectionModel.shortRange));
      expect(frontAnchors.length, shortAnchors.length);
    });

    test('full model with interp=0 should produce fewer repeats per cell', () {
      final fullOpts = testOptsFor(FaceDetectionModel.full);
      final fullAnchors = testSsdGenerateAnchors(fullOpts);
      // full model: 1 layer, stride=4, 192/4=48 grid, interp=0 → 1 repeat
      // 48*48*1 = 2304 anchors * 2 = 4608 values
      expect(fullAnchors.length, 48 * 48 * 1 * 2);
    });

    test('front model anchor count matches expected', () {
      final opts = testOptsFor(FaceDetectionModel.frontCamera);
      final anchors = testSsdGenerateAnchors(opts);
      // front model: strides [8,16,16,16], input 128x128, interp=1.0 → 2 repeats
      // layer 0: stride=8, 16x16 grid, 2 repeats = 512 anchors
      // layers 1-3: stride=16, 8x8 grid, 3 layers * 2 repeats = 6 repeats per cell = 384 anchors
      // But layers with same stride are grouped: layers 1,2,3 have stride 16
      //   lastSameStride goes from 1 to 4, repeats = 3*2 = 6
      //   8*8*6 = 384 anchors
      // Total: 512 + 384 = 896 anchors * 2 = 1792 values
      expect(anchors.length, 896 * 2);
    });
  });

  group('testOptsFor', () {
    test('should return config for each model variant', () {
      for (final model in FaceDetectionModel.values) {
        final opts = testOptsFor(model);
        expect(opts.containsKey('num_layers'), isTrue);
        expect(opts.containsKey('strides'), isTrue);
        expect(opts.containsKey('input_size_height'), isTrue);
        expect(opts.containsKey('input_size_width'), isTrue);
      }
    });

    test('full and fullSparse should share same config', () {
      final fullOpts = testOptsFor(FaceDetectionModel.full);
      final sparseOpts = testOptsFor(FaceDetectionModel.fullSparse);
      expect(fullOpts, equals(sparseOpts));
    });
  });

  group('testNameFor', () {
    test('should return model filenames for each variant', () {
      expect(testNameFor(FaceDetectionModel.frontCamera),
          'face_detection_front.tflite');
      expect(testNameFor(FaceDetectionModel.backCamera),
          'face_detection_back.tflite');
      expect(testNameFor(FaceDetectionModel.shortRange),
          'face_detection_short_range.tflite');
      expect(testNameFor(FaceDetectionModel.full),
          'face_detection_full_range.tflite');
      expect(testNameFor(FaceDetectionModel.fullSparse),
          'face_detection_full_range_sparse.tflite');
    });
  });
}
