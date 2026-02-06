import 'dart:typed_data';

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tflite_flutter_custom/tflite_flutter.dart'
    show Interpreter, Tensor;

void main() {
  group('Private helper wrappers', () {
    test('testClip clamps values', () {
      expect(testClip(5, 0, 1), 1);
      expect(testClip(-2, 0, 1), 0);
      expect(testClip(0.5, 0, 1), 0.5);
    });

    test('testSigmoidClipped respects limit', () {
      final high = testSigmoidClipped(1000, limit: 2);
      final low = testSigmoidClipped(-1000, limit: 2);
      expect(high, closeTo(0.8808, 1e-4));
      expect(low, closeTo(0.1192, 1e-4));
    });

    test('testDetectionLetterboxRemoval unpads boxes and keypoints', () {
      final dets = [
        Detection(
          boundingBox: RectF(0.2, 0.3, 0.6, 0.7),
          score: 0.9,
          keypointsXY: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ),
      ];
      final padding = [0.1, 0.1, 0.05, 0.05];

      final unpadded = testDetectionLetterboxRemoval(dets, padding);

      expect(unpadded.single.boundingBox.xmin, closeTo(0.1667, 1e-4));
      expect(unpadded.single.keypointsXY.first, closeTo(0.1667, 1e-4));
    });

    test('testUnpackLandmarks converts to normalized list', () {
      final flat = Float32List.fromList([10, 20, 1, 30, 40, 2]);
      final padding = [0.1, 0.1, 0.1, 0.1];

      final out = testUnpackLandmarks(flat, 100, 100, padding, clamp: true);

      expect(out.length, 2);
      expect(out[0][0], closeTo(0.0, 1e-4));
      expect(out[0][1], closeTo(0.125, 1e-4));
      expect(out[0][2], 1);
    });

    test('testNms filters by IOU and weighting', () {
      final dets = [
        Detection(
          boundingBox: RectF(0.1, 0.1, 0.4, 0.4),
          score: 0.9,
          keypointsXY: const [],
        ),
        Detection(
          boundingBox: RectF(0.15, 0.15, 0.45, 0.45),
          score: 0.8,
          keypointsXY: const [],
        ),
        Detection(
          boundingBox: RectF(0.7, 0.7, 0.9, 0.9),
          score: 0.7,
          keypointsXY: const [],
        ),
      ];

      final keptWeighted = testNms(dets, 0.3, 0.0, weighted: true);
      expect(keptWeighted.length, 2);

      final keptUnweighted = testNms(dets, 0.3, 0.0, weighted: false);
      expect(keptUnweighted.length, 2);
    });

    test('testCollectOutputTensorInfo stops on exception', () {
      final fakeInterpreter = _FakeInterpreter(outputs: 2);
      final map = testCollectOutputTensorInfo(fakeInterpreter);

      expect(map.length, 2);
      expect(map[0]!.shape, [1, 2, 3]);
      expect(map[1]!.shape, [2, 3, 4]);
    });

    test('testImageFromDecodedRgb rebuilds image', () {
      final decoded = DecodedRgb(
        2,
        1,
        Uint8List.fromList([255, 0, 0, 0, 255, 0]),
      );
      final image = testImageFromDecodedRgb(decoded);

      expect(image.width, 2);
      expect(image.height, 1);
      final pixel = image.getPixel(0, 0);
      expect(pixel.r, 255);
    });

    test('testDecodeImageOffUi validates non-empty bytes', () async {
      expect(
        () => testDecodeImageOffUi(Uint8List(0)),
        throwsA(isA<ArgumentError>()),
      );
    });
  });
}

class _FakeTensor implements Tensor {
  _FakeTensor(this.shape, this.buffer);

  @override
  final List<int> shape;

  final ByteBuffer buffer;

  @override
  Uint8List get data => buffer.asUint8List();

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _FakeInterpreter implements Interpreter {
  _FakeInterpreter({required this.outputs});
  final int outputs;

  @override
  Tensor getOutputTensor(int index) {
    if (index >= outputs) throw StateError('no more tensors');
    final shape = [index + 1, index + 2, index + 3];
    final buffer = Float32List.fromList(
      List.filled(shape.reduce((a, b) => a * b), 0),
    ).buffer;
    return _FakeTensor(shape, buffer);
  }

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
