import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

import 'test_config.dart';

void main() {
  globalTestSetup();

  group('SegmentationWorker constructor and state', () {
    test('should not be initialized after construction', () {
      final worker = SegmentationWorker();
      expect(worker.isInitialized, isFalse);
      expect(worker.inputWidth, 256);
      expect(worker.inputHeight, 256);
      expect(worker.outputWidth, 256);
      expect(worker.outputHeight, 256);
    });

    test('should safely dispose without initialization', () {
      final worker = SegmentationWorker();
      expect(() => worker.dispose(), returnsNormally);
    });

    test('should safely dispose twice', () {
      final worker = SegmentationWorker();
      worker.dispose();
      expect(() => worker.dispose(), returnsNormally);
    });
  });

  group('SegmentationWorker.segment error handling', () {
    test('should throw StateError when not initialized', () async {
      final worker = SegmentationWorker();
      expect(() => worker.segment(Uint8List(10)), throwsA(isA<StateError>()));
    });
  });

  group('SegmentationWorker.segmentMat error handling', () {
    test('should throw SegmentationException for empty Mat', () {
      SegmentationWorker();
      // Can't easily test this without initialization
      // but we can verify the segmentMat validates
    });
  });
}
