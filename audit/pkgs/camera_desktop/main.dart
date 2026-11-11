// Minimal probe app for the Flutter 3.50 audit: references the package so it
// is compiled and linked. Nothing runs at build time.
import 'package:flutter/widgets.dart';
import 'package:camera/camera.dart';
import 'package:camera_desktop/camera_desktop.dart';

Future<void> probe() async {
  final cameras = await availableCameras();
  final caps = await CameraDesktopPlugin().getPlatformCapabilities();
  debugPrint('$cameras $caps');
}

void main() {
  runApp(
    Directionality(
      textDirection: TextDirection.ltr,
      child: GestureDetector(
        onTap: probe,
        child: const Center(child: Text('audit')),
      ),
    ),
  );
}
