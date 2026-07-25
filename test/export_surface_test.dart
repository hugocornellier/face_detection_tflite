import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_test/flutter_test.dart';

/// Guards the public export surface for symbols referenced from public
/// dartdoc. `kDefaultMinFacePresenceConfidence` was documented in
/// `FaceDetector.create` / `initialize` from 6.7.0 but never exported from
/// either entry point, so the doc link dangled on pub.dev and callers could
/// not name the default they were being told about.
void main() {
  test('kDefaultMinFacePresenceConfidence is exported with the MediaPipe '
      'default', () {
    expect(kDefaultMinFacePresenceConfidence, 0.5);
  });

  test('kDefaultMaxMissedFrames is exported with its documented default', () {
    expect(kDefaultMaxMissedFrames, 3);
  });
}
