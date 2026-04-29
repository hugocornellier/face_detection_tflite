import 'dart:convert';
import 'dart:io';

import 'package:integration_test/integration_test_driver.dart';

Future<void> main() => integrationDriver(
      responseDataCallback: (Map<String, dynamic>? data) async {
        if (data == null) return;
        final outputDir = Directory('benchmark_results');
        if (!outputDir.existsSync()) outputDir.createSync(recursive: true);
        final encoder = const JsonEncoder.withIndent('  ');
        data.forEach((filename, payload) {
          final f = File('${outputDir.path}/$filename');
          f.writeAsStringSync(encoder.convert(payload));
          // ignore: avoid_print
          print('  Saved: ${f.path}');
        });
      },
    );
