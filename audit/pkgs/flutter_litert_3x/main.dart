// Minimal probe app for the Flutter 3.50 audit: references the package so it
// is compiled and linked. Nothing runs at build time.
import 'package:flutter/widgets.dart';
import 'package:flutter_litert/flutter_litert.dart';

Future<void> probe() async {
  final options = InterpreterOptions();
  final interpreter = await Interpreter.fromAsset('assets/none.tflite', options: options);
  debugPrint('$interpreter');
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
