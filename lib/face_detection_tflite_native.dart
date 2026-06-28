/// Native (non-web) entry point for `package:face_detection_tflite`.
///
/// This barrel resolves the full API against the native implementation
/// unconditionally. The portable `face_detection_tflite.dart` entry defaults
/// its conditional export to the web surface so the package is WASM-ready;
/// as a result, native-only symbols (FaceDetector, SegmentationWorker, overlay
/// painters, demo controls, etc.) cannot be reached through it during static
/// analysis on native targets.
///
/// Import this from code that only runs on native platforms (Android, iOS,
/// macOS, Windows, Linux). Do not import it together with
/// `face_detection_tflite.dart` in the same library: both re-export the same
/// types, which would cause ambiguous-import errors.
library;

export 'src/native/face_native_lib.dart';
