// Minimal probe app for the Flutter 3.50 audit: references the package so it
// is compiled and linked. Nothing runs at build time.
import 'package:flutter/widgets.dart';
import 'package:animal_detection/animal_detection.dart';

Future<void> probe() async {
  final detector = AnimalDetector();
  await detector.initialize();
  debugPrint('$detector');
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
