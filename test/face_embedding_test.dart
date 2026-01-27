import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'test_config.dart';

/// Unit tests for FaceEmbedding static methods and utility functions.
///
/// Tests cover:
/// - cosineSimilarity() - distance metric for face comparison
/// - euclideanDistance() - alternative distance metric
/// - computeEmbeddingAlignment() - alignment parameter computation
/// - Edge cases and error handling
void main() {
  globalTestSetup();

  group('FaceEmbedding.cosineSimilarity', () {
    test('should return 1.0 for identical vectors', () {
      final embedding = Float32List.fromList([0.5, 0.5, 0.5, 0.5]);
      final similarity = FaceEmbedding.cosineSimilarity(embedding, embedding);
      expect(similarity, closeTo(1.0, 0.0001));
    });

    test('should return 1.0 for parallel vectors with different magnitudes',
        () {
      final a = Float32List.fromList([1.0, 2.0, 3.0]);
      final b = Float32List.fromList([2.0, 4.0, 6.0]);
      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      expect(similarity, closeTo(1.0, 0.0001));
    });

    test('should return -1.0 for opposite vectors', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0]);
      final b = Float32List.fromList([-1.0, 0.0, 0.0]);
      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      expect(similarity, closeTo(-1.0, 0.0001));
    });

    test('should return 0.0 for orthogonal vectors', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0]);
      final b = Float32List.fromList([0.0, 1.0, 0.0]);
      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      expect(similarity, closeTo(0.0, 0.0001));
    });

    test('should handle normalized vectors correctly', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b =
          Float32List.fromList([math.cos(math.pi / 4), math.sin(math.pi / 4)]);

      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      expect(similarity, closeTo(math.cos(math.pi / 4), 0.0001));
    });

    test('should throw ArgumentError for mismatched dimensions', () {
      final a = Float32List.fromList([1.0, 2.0, 3.0]);
      final b = Float32List.fromList([1.0, 2.0]);

      expect(
        () => FaceEmbedding.cosineSimilarity(a, b),
        throwsA(isA<ArgumentError>().having(
          (e) => e.message,
          'message',
          contains('dimensions must match'),
        )),
      );
    });

    test('should return 0.0 for zero vector', () {
      final a = Float32List.fromList([0.0, 0.0, 0.0]);
      final b = Float32List.fromList([1.0, 2.0, 3.0]);

      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      expect(similarity, 0.0);
    });

    test('should handle very small values without precision issues', () {
      final a = Float32List.fromList([1e-10, 1e-10, 1e-10]);
      final b = Float32List.fromList([1e-10, 1e-10, 1e-10]);

      final similarity = FaceEmbedding.cosineSimilarity(a, b);
      expect(similarity, closeTo(1.0, 0.0001));
    });

    test('should handle typical face embedding dimension (192)', () {
      final random = math.Random(42);
      final a = Float32List.fromList(
        List.generate(192, (_) => random.nextDouble()),
      );
      final b = Float32List.fromList(
        List.generate(192, (_) => random.nextDouble()),
      );

      final similarity = FaceEmbedding.cosineSimilarity(a, b);

      expect(similarity, greaterThanOrEqualTo(-1.0));
      expect(similarity, lessThanOrEqualTo(1.0));
    });
  });

  group('FaceEmbedding.euclideanDistance', () {
    test('should return 0.0 for identical vectors', () {
      final embedding = Float32List.fromList([0.5, 0.5, 0.5, 0.5]);
      final distance = FaceEmbedding.euclideanDistance(embedding, embedding);
      expect(distance, closeTo(0.0, 0.0001));
    });

    test('should calculate correct distance for simple vectors', () {
      final a = Float32List.fromList([0.0, 0.0, 0.0]);
      final b = Float32List.fromList([3.0, 4.0, 0.0]);

      final distance = FaceEmbedding.euclideanDistance(a, b);
      expect(distance, closeTo(5.0, 0.0001));
    });

    test('should calculate correct distance for unit vectors', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b = Float32List.fromList([0.0, 1.0]);

      final distance = FaceEmbedding.euclideanDistance(a, b);
      expect(distance, closeTo(math.sqrt(2), 0.0001));
    });

    test('should throw ArgumentError for mismatched dimensions', () {
      final a = Float32List.fromList([1.0, 2.0, 3.0]);
      final b = Float32List.fromList([1.0, 2.0]);

      expect(
        () => FaceEmbedding.euclideanDistance(a, b),
        throwsA(isA<ArgumentError>().having(
          (e) => e.message,
          'message',
          contains('dimensions must match'),
        )),
      );
    });

    test('should handle normalized embeddings correctly', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final angle = math.pi / 3;
      final b = Float32List.fromList([math.cos(angle), math.sin(angle)]);

      final distance = FaceEmbedding.euclideanDistance(a, b);
      final expectedDistance = 2 * math.sin(angle / 2);
      expect(distance, closeTo(expectedDistance, 0.0001));
    });

    test('should be symmetric', () {
      final a = Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]);
      final b = Float32List.fromList([5.0, 4.0, 3.0, 2.0, 1.0]);

      final distAB = FaceEmbedding.euclideanDistance(a, b);
      final distBA = FaceEmbedding.euclideanDistance(b, a);

      expect(distAB, closeTo(distBA, 0.0001));
    });

    test('should handle typical face embedding dimension (192)', () {
      final random = math.Random(42);
      List<double> randomUnit(int dim) {
        final v = List.generate(dim, (_) => random.nextDouble() - 0.5);
        final norm = math.sqrt(v.fold(0.0, (sum, x) => sum + x * x));
        return v.map((x) => x / norm).toList();
      }

      final a = Float32List.fromList(randomUnit(192));
      final b = Float32List.fromList(randomUnit(192));

      final distance = FaceEmbedding.euclideanDistance(a, b);

      expect(distance, greaterThanOrEqualTo(0.0));
      expect(distance, lessThanOrEqualTo(2.0));
    });

    test('should handle zero vector', () {
      final a = Float32List.fromList([0.0, 0.0, 0.0]);
      final b = Float32List.fromList([1.0, 1.0, 1.0]);

      final distance = FaceEmbedding.euclideanDistance(a, b);
      expect(distance, closeTo(math.sqrt(3), 0.0001));
    });

    test('should satisfy triangle inequality', () {
      final a = Float32List.fromList([0.0, 0.0]);
      final b = Float32List.fromList([1.0, 0.0]);
      final c = Float32List.fromList([0.5, 0.5]);

      final ab = FaceEmbedding.euclideanDistance(a, b);
      final bc = FaceEmbedding.euclideanDistance(b, c);
      final ac = FaceEmbedding.euclideanDistance(a, c);

      expect(ac, lessThanOrEqualTo(ab + bc + 0.0001));
      expect(ab, lessThanOrEqualTo(ac + bc + 0.0001));
      expect(bc, lessThanOrEqualTo(ab + ac + 0.0001));
    });
  });

  group('FaceDetector.compareFaces and faceDistance consistency', () {
    test(
        'similarity and distance should be inversely related for normalized vectors',
        () {
      final angles = [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2];

      for (final angle in angles) {
        final a = Float32List.fromList([1.0, 0.0]);
        final b = Float32List.fromList([math.cos(angle), math.sin(angle)]);

        final similarity = FaceDetector.compareFaces(a, b);
        final distance = FaceDetector.faceDistance(a, b);

        expect(similarity, closeTo(math.cos(angle), 0.001));
        expect(distance, closeTo(2 * math.sin(angle / 2), 0.001));
      }
    });

    test('identical embeddings should have similarity 1.0 and distance 0.0',
        () {
      final embedding = Float32List.fromList(
        List.generate(192, (i) => math.sin(i.toDouble())),
      );

      double norm = 0.0;
      for (final v in embedding) {
        norm += v * v;
      }
      norm = math.sqrt(norm);
      final normalized = Float32List.fromList(
        embedding.map((v) => v / norm).toList(),
      );

      final similarity = FaceDetector.compareFaces(normalized, normalized);
      final distance = FaceDetector.faceDistance(normalized, normalized);

      expect(similarity, closeTo(1.0, 0.0001));
      expect(distance, closeTo(0.0, 0.0001));
    });
  });

  group('computeEmbeddingAlignment', () {
    test('should compute correct alignment for horizontal eyes', () {
      final alignment = computeEmbeddingAlignment(
        leftEye: Point(100.0, 100.0),
        rightEye: Point(200.0, 100.0),
      );

      expect(alignment.theta, closeTo(0.0, 0.0001));
      expect(alignment.size, closeTo(250.0, 0.1));
      expect(alignment.cx, closeTo(150.0, 1.0));
      expect(alignment.cy, greaterThan(100.0));
    });

    test('should compute correct alignment for rotated eyes', () {
      final eyeDist = 100.0;
      final dx = eyeDist * math.cos(math.pi / 4);
      final dy = eyeDist * math.sin(math.pi / 4);

      final alignment = computeEmbeddingAlignment(
        leftEye: Point(100.0, 100.0),
        rightEye: Point(100.0 + dx, 100.0 + dy),
      );

      expect(alignment.theta, closeTo(math.pi / 4, 0.01));
      expect(alignment.size, closeTo(eyeDist * 2.5, 0.1));
    });

    test('should handle vertical eyes (90 degree rotation)', () {
      final alignment = computeEmbeddingAlignment(
        leftEye: Point(100.0, 100.0),
        rightEye: Point(100.0, 200.0),
      );

      expect(alignment.theta, closeTo(math.pi / 2, 0.01));
    });

    test('should handle inverted eyes (negative angle)', () {
      final alignment = computeEmbeddingAlignment(
        leftEye: Point(100.0, 100.0),
        rightEye: Point(200.0, 50.0),
      );

      expect(alignment.theta, lessThan(0.0));
    });

    test('should compute consistent size regardless of position', () {
      const eyeDist = 80.0;

      final alignment1 = computeEmbeddingAlignment(
        leftEye: Point(0.0, 0.0),
        rightEye: Point(eyeDist, 0.0),
      );

      final alignment2 = computeEmbeddingAlignment(
        leftEye: Point(500.0, 300.0),
        rightEye: Point(500.0 + eyeDist, 300.0),
      );

      expect(alignment1.size, closeTo(alignment2.size, 0.01));
    });

    test('should handle very small eye distance', () {
      final alignment = computeEmbeddingAlignment(
        leftEye: Point(100.0, 100.0),
        rightEye: Point(101.0, 100.0),
      );

      expect(alignment.size, closeTo(2.5, 0.1));
      expect(alignment.theta, closeTo(0.0, 0.01));
    });

    test('should handle very large eye distance', () {
      final alignment = computeEmbeddingAlignment(
        leftEye: Point(0.0, 500.0),
        rightEye: Point(1000.0, 500.0),
      );

      expect(alignment.size, closeTo(2500.0, 1.0));
      expect(alignment.theta, closeTo(0.0, 0.01));
    });
  });

  group('AlignedFaceForEmbedding', () {
    test('should store alignment parameters correctly', () {
      const aligned = AlignedFaceForEmbedding(
        cx: 100.0,
        cy: 150.0,
        size: 200.0,
        theta: 0.5,
      );

      expect(aligned.cx, 100.0);
      expect(aligned.cy, 150.0);
      expect(aligned.size, 200.0);
      expect(aligned.theta, 0.5);
    });
  });
}
