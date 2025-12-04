import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('FaceDetectionTfliteDart.registerWith is a no-op', () {
    expect(() => FaceDetectionTfliteDart.registerWith(), returnsNormally);
  });
}
