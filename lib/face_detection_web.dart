import 'package:flutter_web_plugins/flutter_web_plugins.dart';

/// Web plugin registration for face_detection_tflite.
///
/// This is referenced from `pubspec.yaml`'s `flutter.plugin.platforms.web`
/// block. The package uses conditional imports rather than method channels so
/// this is intentionally a no-op.
class FaceDetectionTfliteWeb {
  /// Registers the web implementation with Flutter's plugin registrar.
  static void registerWith(Registrar registrar) {
    // No-op; conditional imports drive the web implementation.
  }
}
