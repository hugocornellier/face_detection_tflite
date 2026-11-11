// Minimal probe app for the Flutter 3.50 audit: references the package so it
// is compiled and linked. Nothing runs at build time.
import 'package:flutter/widgets.dart';
import 'package:heic_native/heic_native.dart';

Future<void> probe() async {
  final ok = await HeicNative.convert('in.heic', 'out.png');
  debugPrint('$ok');
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
