// ignore_for_file: avoid_print

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('Embedding demo: score same-person and different-person pairs',
      () async {
    final detector = FaceDetector();
    await detector.initialize();

    Future<double> scorePair(String a, String b) async {
      final ByteData da = await rootBundle.load(a);
      final ByteData db = await rootBundle.load(b);
      final ba = da.buffer.asUint8List();
      final bb = db.buffer.asUint8List();

      final facesA = await detector.detectFacesFromBytes(ba);
      final facesB = await detector.detectFacesFromBytes(bb);
      print('DEMO_FACES $a -> ${facesA.length}');
      print('DEMO_FACES $b -> ${facesB.length}');

      // Pick the largest face in each image (foreground subject).
      facesA.sort((x, y) => (y.boundingBox.width * y.boundingBox.height)
          .compareTo(x.boundingBox.width * x.boundingBox.height));
      facesB.sort((x, y) => (y.boundingBox.width * y.boundingBox.height)
          .compareTo(x.boundingBox.width * x.boundingBox.height));

      void printBox(String path, Face f) {
        final bx = f.boundingBox;
        print('DEMO_BBOX $path ${bx.left.toStringAsFixed(1)} '
            '${bx.top.toStringAsFixed(1)} ${bx.width.toStringAsFixed(1)} '
            '${bx.height.toStringAsFixed(1)}');
      }

      printBox(a, facesA.first);
      printBox(b, facesB.first);

      final ea = await detector.getFaceEmbedding(facesA.first, ba);
      final eb = await detector.getFaceEmbedding(facesB.first, bb);
      return FaceDetector.compareFaces(ea, eb);
    }

    final samePair = await scorePair(
      'assets/samples/demo/hugo_day3.jpg',
      'assets/samples/demo/hugo_day4.jpg',
    );
    print('DEMO_RESULT same ${samePair.toStringAsFixed(4)}');

    final diffPair = await scorePair(
      'assets/samples/demo/person_b1.jpg',
      'assets/samples/demo/person_b2.jpg',
    );
    print('DEMO_RESULT different ${diffPair.toStringAsFixed(4)}');

    detector.dispose();
  });
}
