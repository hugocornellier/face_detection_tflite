import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart' show Size;
import 'package:face_detection_tflite/face_detection_tflite.dart';

import 'test_config.dart';

void main() {
  globalTestSetup();

  // =========================================================================
  // Point
  // =========================================================================
  group('Point', () {
    test('should create 2D point', () {
      const p = Point(1.0, 2.0);
      expect(p.x, 1.0);
      expect(p.y, 2.0);
      expect(p.z, isNull);
      expect(p.is3D, isFalse);
    });

    test('should create 3D point', () {
      const p = Point(1.0, 2.0, 3.0);
      expect(p.x, 1.0);
      expect(p.y, 2.0);
      expect(p.z, 3.0);
      expect(p.is3D, isTrue);
    });

    test('toString for 2D point', () {
      const p = Point(1.5, 2.5);
      expect(p.toString(), 'Point(1.5, 2.5)');
    });

    test('toString for 3D point', () {
      const p = Point(1.5, 2.5, 3.5);
      expect(p.toString(), 'Point(1.5, 2.5, 3.5)');
    });

    test('equality', () {
      const a = Point(1.0, 2.0, 3.0);
      const b = Point(1.0, 2.0, 3.0);
      const c = Point(1.0, 2.0);
      expect(a, equals(b));
      expect(a, isNot(equals(c)));
      expect(a.hashCode, equals(b.hashCode));
    });

    test('toMap and fromMap round trip for 2D', () {
      const original = Point(1.5, 2.5);
      final map = original.toMap();
      final restored = Point.fromMap(map);
      expect(restored.x, original.x);
      expect(restored.y, original.y);
      expect(restored.z, isNull);
    });

    test('toMap and fromMap round trip for 3D', () {
      const original = Point(1.0, 2.0, 3.0);
      final map = original.toMap();
      final restored = Point.fromMap(map);
      expect(restored, equals(original));
    });
  });

  // =========================================================================
  // RectF
  // =========================================================================
  group('RectF', () {
    test('should store coordinates', () {
      const r = RectF(0.1, 0.2, 0.8, 0.9);
      expect(r.xmin, 0.1);
      expect(r.ymin, 0.2);
      expect(r.xmax, 0.8);
      expect(r.ymax, 0.9);
    });

    test('w and h', () {
      const r = RectF(0.1, 0.2, 0.9, 0.7);
      expect(r.w, closeTo(0.8, 0.001));
      expect(r.h, closeTo(0.5, 0.001));
    });

    test('scale', () {
      const r = RectF(0.1, 0.2, 0.5, 0.4);
      final scaled = r.scale(2.0, 3.0);
      expect(scaled.xmin, closeTo(0.2, 0.001));
      expect(scaled.ymin, closeTo(0.6, 0.001));
      expect(scaled.xmax, closeTo(1.0, 0.001));
      expect(scaled.ymax, closeTo(1.2, 0.001));
    });

    test('expand', () {
      const r = RectF(0.2, 0.2, 0.8, 0.8);
      final expanded = r.expand(0.5);
      // center = (0.5, 0.5), width = 0.6 -> half = 0.45, height = 0.6 -> half = 0.45
      expect(expanded.xmin, closeTo(0.05, 0.001));
      expect(expanded.ymin, closeTo(0.05, 0.001));
      expect(expanded.xmax, closeTo(0.95, 0.001));
      expect(expanded.ymax, closeTo(0.95, 0.001));
    });

    test('toMap and fromMap round trip', () {
      const original = RectF(0.1, 0.2, 0.8, 0.9);
      final map = original.toMap();
      final restored = RectF.fromMap(map);
      expect(restored.xmin, original.xmin);
      expect(restored.ymin, original.ymin);
      expect(restored.xmax, original.xmax);
      expect(restored.ymax, original.ymax);
    });
  });

  // =========================================================================
  // Detection
  // =========================================================================
  group('Detection', () {
    test('should store fields', () {
      final det = Detection(
        boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
        score: 0.95,
        keypointsXY: [0.3, 0.4, 0.5, 0.6],
      );
      expect(det.score, 0.95);
      expect(det.keypointsXY.length, 4);
      expect(det.imageSize, isNull);
    });

    test('operator [] accesses keypointsXY', () {
      final det = Detection(
        boundingBox: const RectF(0.0, 0.0, 1.0, 1.0),
        score: 0.9,
        keypointsXY: [0.1, 0.2, 0.3, 0.4],
      );
      expect(det[0], 0.1);
      expect(det[1], 0.2);
    });

    test('landmarks throws when imageSize is null', () {
      final det = Detection(
        boundingBox: const RectF(0.0, 0.0, 1.0, 1.0),
        score: 0.9,
        keypointsXY: TestUtils.generateValidKeypoints(),
      );
      expect(() => det.landmarks, throwsA(isA<StateError>()));
    });

    test('landmarks denormalizes with imageSize', () {
      final det = Detection(
        boundingBox: const RectF(0.0, 0.0, 1.0, 1.0),
        score: 0.9,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: const Size(100, 200),
      );
      final landmarks = det.landmarks;
      expect(landmarks.length, FaceLandmarkType.values.length);
      // Check that coordinates are denormalized
      for (final lm in landmarks.values) {
        expect(lm.x, greaterThanOrEqualTo(0));
        expect(lm.y, greaterThanOrEqualTo(0));
      }
    });

    test('toMap and fromMap round trip', () {
      final original = Detection(
        boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
        score: 0.95,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: const Size(640, 480),
      );
      final map = original.toMap();
      final restored = Detection.fromMap(map);
      expect(restored.score, original.score);
      expect(restored.boundingBox.xmin, original.boundingBox.xmin);
      expect(restored.imageSize!.width, original.imageSize!.width);
      expect(restored.keypointsXY.length, original.keypointsXY.length);
    });

    test('toMap and fromMap without imageSize', () {
      final original = Detection(
        boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
        score: 0.8,
        keypointsXY: [0.3, 0.4],
      );
      final map = original.toMap();
      final restored = Detection.fromMap(map);
      expect(restored.imageSize, isNull);
    });
  });

  // =========================================================================
  // FaceMesh
  // =========================================================================
  group('FaceMesh', () {
    test('should create with 468 points', () {
      final points = List.generate(
        468,
        (i) => Point(i.toDouble(), i.toDouble(), i.toDouble()),
      );
      final mesh = FaceMesh(points);
      expect(mesh.length, 468);
      expect(mesh.points.length, 468);
    });

    test('operator [] accesses correct point', () {
      final points = List.generate(
        468,
        (i) => Point(i.toDouble(), i * 2.0),
      );
      final mesh = FaceMesh(points);
      expect(mesh[0].x, 0.0);
      expect(mesh[100].x, 100.0);
      expect(mesh[100].y, 200.0);
    });

    test('toString', () {
      final points = List.generate(468, (i) => Point(0, 0));
      final mesh = FaceMesh(points);
      expect(mesh.toString(), 'FaceMesh(468 points)');
    });

    test('toMap and fromMap round trip', () {
      final points = List.generate(
        468,
        (i) => Point(i.toDouble(), i * 2.0, i * 3.0),
      );
      final original = FaceMesh(points);
      final map = original.toMap();
      final restored = FaceMesh.fromMap(map);
      expect(restored.length, 468);
      expect(restored[0].x, 0.0);
      expect(restored[100].x, 100.0);
      expect(restored[100].z, 300.0);
    });
  });

  // =========================================================================
  // Face
  // =========================================================================
  group('Face', () {
    test('should create with minimal fields', () {
      final face = Face(
        detection: Detection(
          boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
          score: 0.95,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: const Size(640, 480),
        ),
        mesh: null,
        irises: [],
        originalSize: const Size(640, 480),
      );
      expect(face.mesh, isNull);
      expect(face.irisPoints, isEmpty);
    });

    test('boundingBox returns pixel coordinates', () {
      final face = Face(
        detection: Detection(
          boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
          score: 0.95,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: const Size(100, 200),
        ),
        mesh: null,
        irises: [],
        originalSize: const Size(100, 200),
      );
      final bb = face.boundingBox;
      expect(bb, isNotNull);
    });

    test('landmarks returns landmark map', () {
      final face = Face(
        detection: Detection(
          boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
          score: 0.95,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: const Size(640, 480),
        ),
        mesh: null,
        irises: [],
        originalSize: const Size(640, 480),
      );
      final landmarks = face.landmarks;
      expect(landmarks, isNotNull);
    });

    test('eyes returns null when no iris points', () {
      final face = Face(
        detection: Detection(
          boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
          score: 0.95,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: const Size(640, 480),
        ),
        mesh: null,
        irises: [],
        originalSize: const Size(640, 480),
      );
      expect(face.eyes, isNull);
    });

    test('toMap and fromMap round trip', () {
      final original = Face(
        detection: Detection(
          boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
          score: 0.95,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: const Size(640, 480),
        ),
        mesh: null,
        irises: [],
        originalSize: const Size(640, 480),
      );
      final map = original.toMap();
      final restored = Face.fromMap(map);
      expect(restored.irisPoints, isEmpty);
      expect(restored.mesh, isNull);
    });

    test('toMap and fromMap with mesh', () {
      final meshPoints = List.generate(
        468,
        (i) => Point(i.toDouble(), i.toDouble(), i * 0.1),
      );
      final original = Face(
        detection: Detection(
          boundingBox: const RectF(0.1, 0.2, 0.8, 0.9),
          score: 0.95,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: const Size(640, 480),
        ),
        mesh: FaceMesh(meshPoints),
        irises: [],
        originalSize: const Size(640, 480),
      );
      final map = original.toMap();
      final restored = Face.fromMap(map);
      expect(restored.mesh, isNotNull);
      expect(restored.mesh!.length, 468);
    });
  });

  // =========================================================================
  // SegmentationMask
  // =========================================================================
  group('SegmentationMask', () {
    test('factory validates data length', () {
      expect(
        () => SegmentationMask(
          data: Float32List.fromList([0.5, 0.5]),
          width: 3,
          height: 1,
          originalWidth: 100,
          originalHeight: 100,
        ),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('at returns probability at coordinates', () {
      final data = Float32List.fromList([0.1, 0.2, 0.3, 0.4]);
      final mask = SegmentationMask(
        data: data,
        width: 2,
        height: 2,
        originalWidth: 100,
        originalHeight: 100,
      );
      expect(mask.at(0, 0), closeTo(0.1, 0.001));
      expect(mask.at(1, 0), closeTo(0.2, 0.001));
      expect(mask.at(0, 1), closeTo(0.3, 0.001));
      expect(mask.at(1, 1), closeTo(0.4, 0.001));
    });

    test('at returns 0 for out of bounds', () {
      final mask = SegmentationMask(
        data: Float32List.fromList([0.5]),
        width: 1,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
      );
      expect(mask.at(-1, 0), 0.0);
      expect(mask.at(0, -1), 0.0);
      expect(mask.at(1, 0), 0.0);
      expect(mask.at(0, 1), 0.0);
    });

    test('data returns defensive copy', () {
      final original = Float32List.fromList([0.5, 0.5, 0.5, 0.5]);
      final mask = SegmentationMask(
        data: original,
        width: 2,
        height: 2,
        originalWidth: 10,
        originalHeight: 10,
      );
      final dataCopy = mask.data;
      dataCopy[0] = 99.0;
      // Original should be unmodified
      expect(mask.at(0, 0), closeTo(0.5, 0.001));
    });

    test('upsample to larger dimensions', () {
      final data = Float32List.fromList([0.0, 1.0, 1.0, 0.0]);
      final mask = SegmentationMask(
        data: data,
        width: 2,
        height: 2,
        originalWidth: 4,
        originalHeight: 4,
      );
      final upsampled = mask.upsample();
      expect(upsampled.width, 4);
      expect(upsampled.height, 4);
    });

    test('upsample with maxSize cap', () {
      final data = Float32List.fromList([0.5, 0.5, 0.5, 0.5]);
      final mask = SegmentationMask(
        data: data,
        width: 2,
        height: 2,
        originalWidth: 2000,
        originalHeight: 1000,
      );
      final upsampled = mask.upsample(maxSize: 500);
      expect(upsampled.width, lessThanOrEqualTo(500));
      expect(upsampled.height, lessThanOrEqualTo(500));
    });

    test('upsample with custom target dimensions', () {
      final data = Float32List.fromList([0.5, 0.5, 0.5, 0.5]);
      final mask = SegmentationMask(
        data: data,
        width: 2,
        height: 2,
        originalWidth: 100,
        originalHeight: 100,
      );
      final upsampled = mask.upsample(targetWidth: 10, targetHeight: 10);
      expect(upsampled.width, 10);
      expect(upsampled.height, 10);
    });

    test('upsample with padding', () {
      // 4x4 mask with padding on all sides
      final data = Float32List(16);
      for (int i = 0; i < 16; i++) {
        data[i] = 0.5;
      }
      final mask = SegmentationMask(
        data: data,
        width: 4,
        height: 4,
        originalWidth: 10,
        originalHeight: 10,
        padding: [0.25, 0.25, 0.25, 0.25],
      );
      final upsampled = mask.upsample();
      expect(upsampled.width, 10);
      expect(upsampled.height, 10);
      // Padding should be removed
      expect(upsampled.padding, [0.0, 0.0, 0.0, 0.0]);
    });

    test('toRgba with binary threshold', () {
      final data = Float32List.fromList([0.2, 0.8]);
      final mask = SegmentationMask(
        data: data,
        width: 2,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
      );
      final rgba = mask.toRgba(
        foreground: 0xFFFFFFFF,
        background: 0x00000000,
        threshold: 0.5,
      );
      expect(rgba.length, 8); // 2 pixels * 4 bytes
      // First pixel (0.2 < 0.5): background (transparent)
      expect(rgba[3], 0); // Alpha of first pixel
      // Second pixel (0.8 >= 0.5): foreground (opaque white)
      expect(rgba[7], 255); // Alpha of second pixel
    });

    test('toRgba with soft blend', () {
      final data = Float32List.fromList([0.5]);
      final mask = SegmentationMask(
        data: data,
        width: 1,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
      );
      final rgba = mask.toRgba(
        foreground: 0xFF000000,
        background: 0x00000000,
        threshold: -1, // Soft blend
      );
      expect(rgba.length, 4);
      // At 0.5 blend, R should be ~128
      expect(rgba[0], closeTo(128, 1));
    });

    test('toRgba with BGRA format', () {
      final data = Float32List.fromList([1.0]);
      final mask = SegmentationMask(
        data: data,
        width: 1,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
      );
      final rgba = mask.toRgba(
        foreground: 0xFF0000FF, // In BGRA: B=0x00, G=0x00, R=0xFF, A=0xFF
        format: PixelFormat.bgra,
        threshold: 0.5,
      );
      expect(rgba.length, 4);
    });

    test('toRgba with ARGB format', () {
      final data = Float32List.fromList([1.0]);
      final mask = SegmentationMask(
        data: data,
        width: 1,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
      );
      final rgba = mask.toRgba(
        foreground: 0xFF0000FF,
        format: PixelFormat.argb,
        threshold: 0.5,
      );
      expect(rgba.length, 4);
    });

    test('toMap serialization', () {
      final data = Float32List.fromList([0.1, 0.5, 0.9, 0.2]);
      final mask = SegmentationMask(
        data: data,
        width: 2,
        height: 2,
        originalWidth: 100,
        originalHeight: 200,
        padding: [0.1, 0.1, 0.05, 0.05],
      );
      final map = mask.toMap();
      expect(map['width'], 2);
      expect(map['height'], 2);
      expect(map['originalWidth'], 100);
      expect(map['originalHeight'], 200);
    });

    test('fromMap with uint8 format', () {
      final map = <String, dynamic>{
        'data': [0, 128, 255],
        'dataFormat': 'uint8',
        'width': 3,
        'height': 1,
        'originalWidth': 100,
        'originalHeight': 100,
        'padding': [0.0, 0.0, 0.0, 0.0],
      };
      final mask = SegmentationMask.fromMap(map);
      expect(mask.at(0, 0), closeTo(0.0, 0.01));
      expect(mask.at(1, 0), closeTo(128.0 / 255.0, 0.01));
      expect(mask.at(2, 0), closeTo(1.0, 0.01));
    });

    test('fromMap with binary format', () {
      final map = <String, dynamic>{
        'data': [0, 255, 255],
        'dataFormat': 'binary',
        'width': 3,
        'height': 1,
        'originalWidth': 100,
        'originalHeight': 100,
        'padding': [0.0, 0.0, 0.0, 0.0],
      };
      final mask = SegmentationMask.fromMap(map);
      expect(mask.at(0, 0), 0.0);
      expect(mask.at(1, 0), 1.0);
      expect(mask.at(2, 0), 1.0);
    });

    test('fromMap with unknown format throws', () {
      final map = <String, dynamic>{
        'data': [0.5],
        'dataFormat': 'unknown',
        'width': 1,
        'height': 1,
        'originalWidth': 10,
        'originalHeight': 10,
      };
      expect(
        () => SegmentationMask.fromMap(map),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('fromMap without padding uses defaults', () {
      final map = <String, dynamic>{
        'data': [0.5],
        'dataFormat': 'float32',
        'width': 1,
        'height': 1,
        'originalWidth': 10,
        'originalHeight': 10,
      };
      final mask = SegmentationMask.fromMap(map);
      expect(mask.padding, [0.0, 0.0, 0.0, 0.0]);
    });

    test('toString', () {
      final mask = SegmentationMask(
        data: Float32List.fromList([0.5]),
        width: 1,
        height: 1,
        originalWidth: 100,
        originalHeight: 200,
      );
      expect(mask.toString(), contains('1x1'));
      expect(mask.toString(), contains('100x200'));
    });
  });

  // =========================================================================
  // MulticlassSegmentationMask
  // =========================================================================
  group('MulticlassSegmentationMask', () {
    test('factory validates data length', () {
      expect(
        () => MulticlassSegmentationMask(
          data: Float32List.fromList([0.5]),
          width: 2,
          height: 1,
          originalWidth: 10,
          originalHeight: 10,
          classData: Float32List(12),
        ),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('factory validates classData length', () {
      expect(
        () => MulticlassSegmentationMask(
          data: Float32List.fromList([0.5, 0.5]),
          width: 2,
          height: 1,
          originalWidth: 10,
          originalHeight: 10,
          classData: Float32List(10), // Should be 2*1*6=12
        ),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('classMask throws for out of range index', () {
      final mask = MulticlassSegmentationMask(
        data: Float32List.fromList([0.5]),
        width: 1,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
        classData: Float32List.fromList([0.1, 0.2, 0.1, 0.3, 0.2, 0.1]),
      );
      expect(() => mask.classMask(-1), throwsA(isA<RangeError>()));
      expect(() => mask.classMask(6), throwsA(isA<RangeError>()));
    });

    test('classMask returns correct values', () {
      final classData = Float32List.fromList([
        0.1, 0.3, 0.2, 0.2, 0.1, 0.1, // pixel 0
      ]);
      final mask = MulticlassSegmentationMask(
        data: Float32List.fromList([0.9]),
        width: 1,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
        classData: classData,
      );
      expect(mask.classMask(0)[0], closeTo(0.1, 0.001));
      expect(mask.classMask(1)[0], closeTo(0.3, 0.001));
    });

    test('otherMask returns correct values', () {
      final classData = Float32List.fromList([
        0.1, 0.2, 0.1, 0.1, 0.1, 0.4, // pixel 0: 'other' = 0.4
      ]);
      final mask = MulticlassSegmentationMask(
        data: Float32List.fromList([0.9]),
        width: 1,
        height: 1,
        originalWidth: 10,
        originalHeight: 10,
        classData: classData,
      );
      expect(mask.otherMask[0], closeTo(0.4, 0.001));
    });

    test('toString', () {
      final mask = MulticlassSegmentationMask(
        data: Float32List.fromList([0.9]),
        width: 1,
        height: 1,
        originalWidth: 100,
        originalHeight: 200,
        classData: Float32List.fromList([0.1, 0.2, 0.1, 0.3, 0.2, 0.1]),
      );
      expect(mask.toString(), contains('MulticlassSegmentationMask'));
      expect(mask.toString(), contains('6 classes'));
    });
  });

  // =========================================================================
  // DetectionWithSegmentationResult
  // =========================================================================
  group('DetectionWithSegmentationResult', () {
    test('toMap and fromMap round trip', () {
      final mask = SegmentationMask(
        data: Float32List.fromList([0.5, 0.8]),
        width: 2,
        height: 1,
        originalWidth: 100,
        originalHeight: 100,
      );
      final original = DetectionWithSegmentationResult(
        faces: [],
        segmentationMask: mask,
        detectionTimeMs: 10,
        segmentationTimeMs: 20,
      );
      final map = original.toMap();
      final restored = DetectionWithSegmentationResult.fromMap(map);
      expect(restored.faces, isEmpty);
      expect(restored.segmentationMask, isNotNull);
      expect(restored.detectionTimeMs, 10);
      expect(restored.segmentationTimeMs, 20);
    });

    test('fromMap without segmentationMask', () {
      final original = DetectionWithSegmentationResult(
        faces: [],
        detectionTimeMs: 15,
        segmentationTimeMs: 0,
      );
      final map = original.toMap();
      final restored = DetectionWithSegmentationResult.fromMap(map);
      expect(restored.segmentationMask, isNull);
    });

    test('toString', () {
      final result = DetectionWithSegmentationResult(
        faces: [],
        detectionTimeMs: 10,
        segmentationTimeMs: 20,
      );
      expect(result.toString(), contains('DetectionWithSegmentationResult'));
      expect(result.toString(), contains('20ms'));
    });
  });

  // =========================================================================
  // AlignedRoi
  // =========================================================================
  group('AlignedRoi', () {
    test('should store all coordinates', () {
      const roi = AlignedRoi(10.0, 20.0, 30.0, 0.5);
      expect(roi.cx, 10.0);
      expect(roi.cy, 20.0);
      expect(roi.size, 30.0);
      expect(roi.theta, 0.5);
    });
  });

  // =========================================================================
  // ImageTensor
  // =========================================================================
  group('ImageTensor', () {
    test('should store tensor and padding', () {
      final tensor = ImageTensor(
        Float32List(3),
        [0.1, 0.1, 0.05, 0.05],
        128,
        128,
      );
      expect(tensor.width, 128);
      expect(tensor.height, 128);
      expect(tensor.padding.length, 4);
    });
  });

  // =========================================================================
  // PixelFormat
  // =========================================================================
  group('PixelFormat', () {
    test('should have all expected values', () {
      expect(PixelFormat.values.length, 3);
      expect(PixelFormat.values.map((p) => p.name),
          containsAll(['rgba', 'bgra', 'argb']));
    });
  });

  // =========================================================================
  // SegmentationConfig presets
  // =========================================================================
  group('SegmentationConfig presets', () {
    test('performance preset', () {
      expect(SegmentationConfig.performance.performanceConfig.mode,
          PerformanceMode.auto);
      expect(SegmentationConfig.performance.maxOutputSize, 2048);
    });

    test('fast preset has useIsolate false', () {
      expect(SegmentationConfig.fast.useIsolate, isFalse);
    });

    test('default has useIsolate true', () {
      const config = SegmentationConfig();
      expect(config.useIsolate, isTrue);
    });
  });

  // =========================================================================
  // PerformanceConfig
  // =========================================================================
  group('PerformanceConfig constructors', () {
    test('gpu constructor', () {
      const config = PerformanceConfig.gpu();
      expect(config.mode, PerformanceMode.gpu);
      expect(config.numThreads, isNull);
    });

    test('gpu constructor with threads', () {
      const config = PerformanceConfig.gpu(numThreads: 2);
      expect(config.mode, PerformanceMode.gpu);
      expect(config.numThreads, 2);
    });

    test('auto constructor', () {
      const config = PerformanceConfig.auto();
      expect(config.mode, PerformanceMode.auto);
    });

    test('auto constructor with threads', () {
      const config = PerformanceConfig.auto(numThreads: 3);
      expect(config.mode, PerformanceMode.auto);
      expect(config.numThreads, 3);
    });
  });

  // =========================================================================
  // Constants
  // =========================================================================
  group('Constants', () {
    test('eyeLandmarkConnections should be non-empty', () {
      expect(eyeLandmarkConnections, isNotEmpty);
      // Each connection should have 2 indices
      for (final connection in eyeLandmarkConnections) {
        expect(connection.length, 2);
      }
    });

    test('kMeshPoints should be 468', () {
      expect(kMeshPoints, 468);
    });
  });
}
