// ignore_for_file: avoid_print
//
// Test face embedding matching with Day 13 (one face) and Day 14 (two faces) images.

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('Match face from Day 13 to faces in Day 14', () async {
    print('\n${'=' * 60}');
    print('FACE EMBEDDING MATCH TEST');
    print('=' * 60);

    print('\nInitializing FaceDetector...');
    final detector = FaceDetector();
    await detector.initialize();
    print('Ready!\n');

    // Load images
    final ByteData data1 =
        await rootBundle.load('assets/samples/embedding_test/one_face.jpg');
    final ByteData data2 =
        await rootBundle.load('assets/samples/embedding_test/two_faces.jpg');
    final image1Bytes = data1.buffer.asUint8List();
    final image2Bytes = data2.buffer.asUint8List();

    // === Reference Image (Day 13 - single face) ===
    print('=== Reference Image (Day 13 - one_face.jpg) ===');
    final refFaces =
        await detector.detectFaces(image1Bytes, mode: FaceDetectionMode.fast);
    print('Detected ${refFaces.length} face(s)');
    expect(refFaces.length, 1,
        reason: 'Should detect exactly 1 face in Day 13');

    final refEmbedding =
        await detector.getFaceEmbedding(refFaces.first, image1Bytes);
    print('Embedding generated: ${refEmbedding.length} dimensions');
    final refCenter = refFaces.first.boundingBox.center;
    print('Face location: (${refCenter.x.toInt()}, ${refCenter.y.toInt()})\n');

    // === Comparison Image (Day 14 - two faces) ===
    print('=== Comparison Image (Day 14 - two_faces.jpg) ===');
    final faces =
        await detector.detectFaces(image2Bytes, mode: FaceDetectionMode.fast);
    print('Detected ${faces.length} face(s)');
    expect(faces.length, 2, reason: 'Should detect exactly 2 faces in Day 14');

    // === Compare each face ===
    print('\n=== Similarity Scores ===');
    int bestIndex = -1;
    double bestSimilarity = -1.0;

    for (int i = 0; i < faces.length; i++) {
      final embedding = await detector.getFaceEmbedding(faces[i], image2Bytes);
      final similarity = FaceDetector.compareFaces(refEmbedding, embedding);
      final center = faces[i].boundingBox.center;
      final bbox = faces[i].boundingBox;

      print('Face $i:');
      print('  Similarity: ${similarity.toStringAsFixed(4)}');
      print('  Location: (${center.x.toInt()}, ${center.y.toInt()})');
      print('  Size: ${bbox.width.toInt()}x${bbox.height.toInt()}');

      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestIndex = i;
      }
    }

    // === Result ===
    print('\n=== RESULT ===');
    print('Best match: Face $bestIndex');
    print('Similarity: ${bestSimilarity.toStringAsFixed(4)}');

    String confidence;
    if (bestSimilarity > 0.6) {
      confidence = 'HIGH - Very likely the same person';
    } else if (bestSimilarity > 0.5) {
      confidence = 'MEDIUM-HIGH - Probably the same person';
    } else if (bestSimilarity > 0.4) {
      confidence = 'MEDIUM - Likely the same person';
    } else if (bestSimilarity > 0.3) {
      confidence = 'LOW - Possibly the same person';
    } else {
      confidence = 'VERY LOW - Likely different people';
    }
    print('Confidence: $confidence');
    print('=' * 60);

    // The best match should have higher similarity than the other face
    final otherIndex = bestIndex == 0 ? 1 : 0;
    final otherEmbedding =
        await detector.getFaceEmbedding(faces[otherIndex], image2Bytes);
    final otherSimilarity =
        FaceDetector.compareFaces(refEmbedding, otherEmbedding);

    print('\nVerification:');
    print(
        '  Best match (Face $bestIndex): ${bestSimilarity.toStringAsFixed(4)}');
    print(
        '  Other face (Face $otherIndex): ${otherSimilarity.toStringAsFixed(4)}');
    print(
        '  Difference: ${(bestSimilarity - otherSimilarity).toStringAsFixed(4)}');

    expect(bestSimilarity, greaterThan(otherSimilarity),
        reason: 'The young person should match better than the older man');

    detector.dispose();
    print('\nTest complete!');
  });
}
