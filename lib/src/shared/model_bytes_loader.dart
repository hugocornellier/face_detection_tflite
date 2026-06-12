import 'dart:typed_data';

/// Loads the raw TFLite bytes for a model file.
///
/// [modelFileName] is the bare file name as shipped in the package assets,
/// for example `face_landmark.tflite` or `selfie_segmenter.tflite`.
///
/// Pass an implementation to [FaceDetector.initialize] (or
/// `SelfieSegmentation.create`) to source models from somewhere other than
/// the bundled package assets — a download cache, app documents directory,
/// network, etc. When null, models load from the package assets via
/// `rootBundle` as before.
///
/// Example:
/// ```dart
/// await detector.initialize(
///   loadModelBytes: (fileName) async {
///     final file = File('${cacheDir.path}/$fileName');
///     if (!await file.exists()) {
///       await downloadModel(fileName, to: file);
///     }
///     return file.readAsBytes();
///   },
/// );
/// ```
typedef ModelBytesLoader = Future<Uint8List> Function(String modelFileName);
