import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

void main() {
  globalTestSetup();

  group('Eye', () {
    test('const constructor should create eye with empty mesh', () {
      const eye = Eye(
        irisCenter: Point(100, 200),
        irisContour: [
          Point(98, 198),
          Point(102, 198),
          Point(100, 196),
          Point(100, 204)
        ],
      );
      expect(eye.irisCenter.x, 100);
      expect(eye.irisContour.length, 4);
      expect(eye.mesh, isEmpty);
    });

    test('optimized constructor should pre-compute contour', () {
      final mesh = List.generate(20, (i) => Point(i.toDouble(), i.toDouble()));
      final eye = Eye.optimized(
        irisCenter: const Point(10, 10),
        irisContour: const [
          Point(8, 8),
          Point(12, 8),
          Point(10, 6),
          Point(10, 14)
        ],
        mesh: mesh,
      );
      expect(eye.contour.length, kMaxEyeLandmark);
      expect(eye.mesh.length, 20);
    });

    test('contour getter on const constructor computes from mesh', () {
      final mesh = List.generate(20, (i) => Point(i.toDouble(), i.toDouble()));
      final eye = Eye(
        irisCenter: const Point(10, 10),
        irisContour: const [Point(8, 8)],
        mesh: mesh,
      );
      expect(eye.contour.length, kMaxEyeLandmark);
    });

    test('contour returns full mesh when mesh is shorter than kMaxEyeLandmark',
        () {
      final mesh = [const Point(1, 2), const Point(3, 4)];
      final eye = Eye(
        irisCenter: const Point(10, 10),
        irisContour: const [Point(8, 8)],
        mesh: mesh,
      );
      expect(eye.contour.length, 2);
    });

    test('toMap/fromMap round-trip', () {
      final mesh = List.generate(5, (i) => Point(i.toDouble(), i * 2.0));
      final original = Eye.optimized(
        irisCenter: const Point(50, 60),
        irisContour: const [
          Point(48, 58),
          Point(52, 58),
          Point(50, 56),
          Point(50, 64)
        ],
        mesh: mesh,
      );
      final restored = Eye.fromMap(original.toMap());
      expect(restored.irisCenter.x, original.irisCenter.x);
      expect(restored.irisCenter.y, original.irisCenter.y);
      expect(restored.irisContour.length, original.irisContour.length);
      expect(restored.mesh.length, original.mesh.length);
    });
  });

  group('EyePair', () {
    const eye = Eye(
      irisCenter: Point(100, 200),
      irisContour: [Point(98, 198), Point(102, 198)],
    );

    test('should store both eyes', () {
      const pair = EyePair(leftEye: eye, rightEye: eye);
      expect(pair.leftEye, isNotNull);
      expect(pair.rightEye, isNotNull);
    });

    test('should allow single eye', () {
      const pair = EyePair(leftEye: eye);
      expect(pair.leftEye, isNotNull);
      expect(pair.rightEye, isNull);
    });

    test('should allow no eyes', () {
      const pair = EyePair();
      expect(pair.leftEye, isNull);
      expect(pair.rightEye, isNull);
    });

    test('toMap/fromMap round-trip with both eyes', () {
      const pair = EyePair(leftEye: eye, rightEye: eye);
      final restored = EyePair.fromMap(pair.toMap());
      expect(restored.leftEye, isNotNull);
      expect(restored.rightEye, isNotNull);
      expect(restored.leftEye!.irisCenter.x, eye.irisCenter.x);
    });

    test('toMap/fromMap round-trip with no eyes', () {
      const pair = EyePair();
      final map = pair.toMap();
      expect(map.containsKey('leftEye'), isFalse);
      final restored = EyePair.fromMap(map);
      expect(restored.leftEye, isNull);
      expect(restored.rightEye, isNull);
    });
  });

  group('FaceLandmarks', () {
    final landmarks = FaceLandmarks({
      FaceLandmarkType.leftEye: const Point(35, 40),
      FaceLandmarkType.rightEye: const Point(65, 40),
      FaceLandmarkType.noseTip: const Point(50, 55),
      FaceLandmarkType.mouth: const Point(50, 70),
      FaceLandmarkType.leftEyeTragion: const Point(15, 40),
      FaceLandmarkType.rightEyeTragion: const Point(85, 40),
    });

    test('leftEye getter', () {
      expect(landmarks.leftEye!.x, 35);
      expect(landmarks.leftEye!.y, 40);
    });

    test('rightEye getter', () {
      expect(landmarks.rightEye!.x, 65);
    });

    test('noseTip getter', () {
      expect(landmarks.noseTip!.x, 50);
      expect(landmarks.noseTip!.y, 55);
    });

    test('mouth getter', () {
      expect(landmarks.mouth!.y, 70);
    });

    test('leftEyeTragion getter', () {
      expect(landmarks.leftEyeTragion!.x, 15);
    });

    test('rightEyeTragion getter', () {
      expect(landmarks.rightEyeTragion!.x, 85);
    });

    test('operator[] access', () {
      expect(landmarks[FaceLandmarkType.noseTip]!.x, 50);
    });

    test('values returns all points', () {
      expect(landmarks.values.length, 6);
    });

    test('keys returns all landmark types', () {
      expect(landmarks.keys.length, 6);
      expect(landmarks.keys, contains(FaceLandmarkType.leftEye));
    });

    test('returns null for missing landmark', () {
      final partial = FaceLandmarks({
        FaceLandmarkType.leftEye: const Point(35, 40),
      });
      expect(partial.rightEye, isNull);
      expect(partial[FaceLandmarkType.noseTip], isNull);
    });

    test('toSerializableMap/fromSerializableMap round-trip', () {
      final serialized = landmarks.toSerializableMap();
      final restored = FaceLandmarks.fromSerializableMap(serialized);
      expect(restored.leftEye!.x, landmarks.leftEye!.x);
      expect(restored.rightEye!.y, landmarks.rightEye!.y);
      expect(restored.noseTip!.x, landmarks.noseTip!.x);
      expect(restored.mouth!.y, landmarks.mouth!.y);
      expect(restored.keys.length, landmarks.keys.length);
    });

    test('toMap returns unmodifiable map', () {
      final map = landmarks.toMap();
      expect(() => map[FaceLandmarkType.leftEye] = const Point(0, 0),
          throwsA(isA<UnsupportedError>()));
    });
  });

  group('BoundingBox', () {
    const bbox = BoundingBox(
      topLeft: Point(10, 20),
      topRight: Point(110, 20),
      bottomRight: Point(110, 120),
      bottomLeft: Point(10, 120),
    );

    test('corners returns 4 points in order', () {
      final corners = bbox.corners;
      expect(corners.length, 4);
      expect(corners[0].x, 10); // topLeft
      expect(corners[1].x, 110); // topRight
      expect(corners[2].y, 120); // bottomRight
      expect(corners[3].x, 10); // bottomLeft
    });

    test('center computes average of corners', () {
      final center = bbox.center;
      expect(center.x, closeTo(60.0, 0.0001));
      expect(center.y, closeTo(70.0, 0.0001));
    });

    test('width and height', () {
      expect(bbox.width, closeTo(100.0, 0.0001));
      expect(bbox.height, closeTo(100.0, 0.0001));
    });

    test('toMap/fromMap round-trip', () {
      final restored = BoundingBox.fromMap(bbox.toMap());
      expect(restored.topLeft.x, bbox.topLeft.x);
      expect(restored.bottomRight.y, bbox.bottomRight.y);
      expect(restored.center.x, closeTo(bbox.center.x, 0.0001));
    });

    test('center of non-square box', () {
      const rect = BoundingBox(
        topLeft: Point(0, 0),
        topRight: Point(200, 0),
        bottomRight: Point(200, 100),
        bottomLeft: Point(0, 100),
      );
      expect(rect.center.x, closeTo(100.0, 0.0001));
      expect(rect.center.y, closeTo(50.0, 0.0001));
    });
  });

  group('SegmentationMask.toUint8', () {
    SegmentationMask makeMask(List<double> data, int w, int h) {
      return SegmentationMask(
        data: Float32List.fromList(data),
        width: w,
        height: h,
        originalWidth: w,
        originalHeight: h,
      );
    }

    test('should map 0.0 to 0 and 1.0 to 255', () {
      final mask = makeMask([0.0, 1.0, 0.5, 0.0], 2, 2);
      final result = mask.toUint8();
      expect(result[0], 0);
      expect(result[1], 255);
      expect(result[2], 128);
      expect(result[3], 0);
    });

    test('should clamp values outside [0,1]', () {
      final mask = makeMask([-0.5, 1.5], 2, 1);
      final result = mask.toUint8();
      expect(result[0], 0);
      expect(result[1], 255);
    });
  });

  group('SegmentationMask.toBinary', () {
    SegmentationMask makeMask(List<double> data, int w, int h) {
      return SegmentationMask(
        data: Float32List.fromList(data),
        width: w,
        height: h,
        originalWidth: w,
        originalHeight: h,
      );
    }

    test('should threshold at default 0.5', () {
      final mask = makeMask([0.4, 0.5, 0.6, 0.0], 2, 2);
      final result = mask.toBinary();
      expect(result[0], 0);
      expect(result[1], 255); // >= 0.5
      expect(result[2], 255);
      expect(result[3], 0);
    });

    test('should respect custom threshold', () {
      final mask = makeMask([0.7, 0.8, 0.9, 0.3], 2, 2);
      final result = mask.toBinary(threshold: 0.85);
      expect(result[0], 0);
      expect(result[1], 0);
      expect(result[2], 255);
      expect(result[3], 0);
    });

    test('should handle exact threshold value', () {
      final mask = makeMask([0.5], 1, 1);
      final result = mask.toBinary(threshold: 0.5);
      expect(result[0], 255); // >= threshold
    });
  });

  group('SegmentationMask.upsample with padding', () {
    test('should remove letterbox padding', () {
      // 4x4 mask with 25% padding on left and right
      final data = Float32List(16);
      // Set inner region (columns 1-2) to 1.0
      for (int y = 0; y < 4; y++) {
        data[y * 4 + 1] = 1.0;
        data[y * 4 + 2] = 1.0;
      }

      final mask = SegmentationMask(
        data: data,
        width: 4,
        height: 4,
        originalWidth: 8,
        originalHeight: 8,
        padding: [0.0, 0.0, 0.25, 0.25],
      );

      final upsampled = mask.upsample(targetWidth: 8, targetHeight: 8);
      // After removing padding, the valid region should be upsampled
      expect(upsampled.width, 8);
      expect(upsampled.height, 8);
      expect(upsampled.padding, [0.0, 0.0, 0.0, 0.0]);
    });
  });
}
