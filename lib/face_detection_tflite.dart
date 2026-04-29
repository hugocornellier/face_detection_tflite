/// Face detection and landmark inference utilities backed by MediaPipe-style
/// TFLite models for Flutter apps.
///
/// This is the package's public entry point. It conditionally re-exports the
/// native or web implementation depending on the host platform:
/// - Web (Chrome / Edge / Firefox / Safari): web implementation backed by
///   LiteRT.js (auto WebGPU/WASM) and Canvas preprocessing.
/// - Everything else (mobile, desktop): the native implementation backed by
///   `flutter_litert` + `opencv_dart`.
library;

export 'src/native/face_native_lib.dart'
    if (dart.library.js_interop) 'src/web/face_web_lib.dart';
