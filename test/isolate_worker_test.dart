import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

import 'test_config.dart';

img.Image _gradientImage(int width, int height) {
  final image = img.Image(width: width, height: height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      image.setPixel(x, y, img.ColorRgb8(x * 10, y * 20, 0));
    }
  }
  return image;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('IsolateWorker', () {
    late IsolateWorker worker;

    setUp(() async {
      worker = IsolateWorker();
      await worker.initialize();
    });

    tearDown(() {
      worker.dispose();
    });

    test('initialization is guarded against double calls', () async {
      expect(worker.isInitialized, true);
      expect(() => worker.initialize(), throwsA(isA<StateError>()));
    });

    test('decodes image bytes', () async {
      final bytes = TestUtils.createDummyImageBytes();

      final decoded = await worker.decodeImage(bytes);

      expect(decoded.width, 1);
      expect(decoded.height, 1);
      expect(decoded.rgb, isNotEmpty);
    });

    test('registers frame and converts to tensor', () async {
      final image = _gradientImage(2, 2);
      final frameId = await worker.registerFrame(image);

      final tensor = await worker.imageToTensorWithFrameId(
        frameId,
        outW: 2,
        outH: 2,
      );

      expect(tensor.width, 2);
      expect(tensor.height, 2);
      expect(tensor.padding, [0.0, 0.0, 0.0, 0.0]);

      await worker.releaseFrame(frameId);
    });

    test('crops ROI from registered frame', () async {
      final image = _gradientImage(4, 4);
      final frameId = await worker.registerFrame(image);

      final roi = RectF(0.25, 0.25, 0.75, 0.75);
      final cropped = await worker.cropFromRoiWithFrameId(frameId, roi);

      expect(cropped.width, 2);
      expect(cropped.height, 2);
      final pixel = cropped.getPixel(0, 0);
      expect(pixel.r, 10);
      expect(pixel.g, 20);

      await worker.releaseFrame(frameId);
    });

    test('extracts aligned square from registered frame', () async {
      final image = _gradientImage(3, 3);
      final frameId = await worker.registerFrame(image);

      final extracted = await worker.extractAlignedSquareWithFrameId(
        frameId,
        1.0,
        1.0,
        2.0,
        0.0,
      );

      expect(extracted.width, 2);
      expect(extracted.height, 2);

      await worker.releaseFrame(frameId);
    });

    test('throws when using released frame', () async {
      final image = _gradientImage(2, 2);
      final frameId = await worker.registerFrame(image);
      await worker.releaseFrame(frameId);

      expect(
        () => worker.cropFromRoiWithFrameId(frameId, RectF(0, 0, 1, 1)),
        throwsA(isA<StateError>()),
      );
    });
  });
}
