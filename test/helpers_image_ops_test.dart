import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

import 'test_config.dart';

img.Image _solidImage(int width, int height, img.ColorRgb8 color) {
  final image = img.Image(width: width, height: height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      image.setPixel(x, y, color);
    }
  }
  return image;
}

void main() {
  globalTestSetup();

  group('convertImageToTensor', () {
    test('converts image without padding', () {
      final image = img.Image(width: 2, height: 2)
        ..setPixel(0, 0, img.ColorRgb8(255, 0, 0))
        ..setPixel(1, 0, img.ColorRgb8(0, 255, 0))
        ..setPixel(0, 1, img.ColorRgb8(0, 0, 255))
        ..setPixel(1, 1, img.ColorRgb8(255, 255, 255));

      final tensor = convertImageToTensor(image, outW: 2, outH: 2);

      expect(tensor.padding, [0.0, 0.0, 0.0, 0.0]);
      expect(tensor.tensorNHWC.length, 12);
      expect(tensor.tensorNHWC[0], closeTo(1.0, 1e-6));
      expect(tensor.tensorNHWC[1], closeTo(-1.0, 1e-6));
      expect(tensor.tensorNHWC[2], closeTo(-1.0, 1e-6));
      expect(tensor.tensorNHWC[9], closeTo(1.0, 1e-6));
      expect(tensor.tensorNHWC[10], closeTo(1.0, 1e-6));
      expect(tensor.tensorNHWC[11], closeTo(1.0, 1e-6));
    });

    test('applies letterbox padding and normalization', () {
      final image = _solidImage(2, 1, img.ColorRgb8(10, 20, 30));

      final tensor = convertImageToTensor(image, outW: 4, outH: 4);

      expect(tensor.padding[0], closeTo(0.25, 1e-6));
      expect(tensor.padding[1], closeTo(0.25, 1e-6));
      expect(tensor.padding[2], closeTo(0.0, 1e-6));
      expect(tensor.padding[3], closeTo(0.0, 1e-6));

      expect(tensor.tensorNHWC[0], closeTo(-1.0, 1e-6));

      final int idx = ((1 * 4) + 0) * 3;
      expect(tensor.tensorNHWC[idx], closeTo((10 / 127.5) - 1.0, 1e-6));
      expect(tensor.tensorNHWC[idx + 1], closeTo((20 / 127.5) - 1.0, 1e-6));
      expect(tensor.tensorNHWC[idx + 2], closeTo((30 / 127.5) - 1.0, 1e-6));
    });
  });

  group('faceDetectionToRoi', () {
    test('expands and centers bounding box to square ROI', () {
      final bbox = RectF(0.2, 0.3, 0.6, 0.7);

      final roi = faceDetectionToRoi(bbox, expandFraction: 0.6);

      expect(roi.xmin, closeTo(0.08, 1e-6));
      expect(roi.ymin, closeTo(0.18, 1e-6));
      expect(roi.xmax, closeTo(0.72, 1e-6));
      expect(roi.ymax, closeTo(0.82, 1e-6));
      expect(roi.w, closeTo(0.64, 1e-6));
      expect(roi.h, closeTo(0.64, 1e-6));
    });
  });

  group('cropFromRoi', () {
    test('crops normalized region via isolate', () async {
      final image = img.Image(width: 4, height: 4);
      for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
          image.setPixel(x, y, img.ColorRgb8(x * 10, y * 20, 0));
        }
      }

      final roi = RectF(0.25, 0.25, 0.75, 0.75);
      final cropped = await cropFromRoi(image, roi);

      expect(cropped.width, 2);
      expect(cropped.height, 2);
      final pixel = cropped.getPixel(0, 0);
      expect(pixel.r, 10);
      expect(pixel.g, 20);
    });

    test('throws on invalid ROI', () async {
      final image = _solidImage(2, 2, img.ColorRgb8(0, 0, 0));
      final roi = RectF(-0.1, 0.0, 0.5, 0.5);

      expect(() => cropFromRoi(image, roi), throwsA(isA<ArgumentError>()));
    });
  });

  group('extractAlignedSquare', () {
    test('extracts square patch with bilinear sampling', () async {
      final image = _solidImage(3, 3, img.ColorRgb8(50, 100, 150));

      final extracted = await extractAlignedSquare(image, 1.0, 1.0, 2.0, 0.0);

      expect(extracted.width, 2);
      expect(extracted.height, 2);
      final px = extracted.getPixel(0, 0);
      expect(px.r, 50);
      expect(px.g, 100);
      expect(px.b, 150);
    });

    test('throws on non-positive size', () async {
      final image = _solidImage(2, 2, img.ColorRgb8(0, 0, 0));

      expect(
        () => extractAlignedSquare(image, 1.0, 1.0, 0.0, 0.0),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('Worker fallbacks', () {
    test('decodeImageWithWorker falls back to isolate decode', () async {
      final pngBytes = TestUtils.createDummyImageBytes();

      final decoded = await decodeImageWithWorker(pngBytes, null);

      expect(decoded.width, 1);
      expect(decoded.height, 1);
      expect(decoded.rgb, isNotEmpty);
    });

    test('imageToTensorWithWorker uses isolate conversion when worker null',
        () async {
      final image = _solidImage(2, 2, img.ColorRgb8(0, 0, 255));

      final tensor = await imageToTensorWithWorker(
        image,
        outW: 2,
        outH: 2,
        worker: null,
      );

      expect(tensor.width, 2);
      expect(tensor.height, 2);
      expect(tensor.padding, [0.0, 0.0, 0.0, 0.0]);
      expect(tensor.tensorNHWC[0], closeTo(-1.0, 1e-6));
      expect(tensor.tensorNHWC[1], closeTo(-1.0, 1e-6));
      expect(tensor.tensorNHWC[2], closeTo(1.0, 1e-6));
    });
  });
}
