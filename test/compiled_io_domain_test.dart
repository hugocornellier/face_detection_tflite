import 'dart:io';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// Pins the input domain that makes flutter_litert's compiled-IO helpers
/// equivalent to the local ones they replaced.
///
/// The two implementations agree exactly whenever a tensor byte size is a
/// positive multiple of 4; they diverge only at `byteSize <= 0` (the local
/// version returned a non-positive count, litert throws) and on the exception
/// type for a misaligned size. This asserts every real bundled model stays
/// inside the agreeing domain, so the migration cannot change behaviour for
/// any model this package actually ships.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final modelsDir = Directory('${Directory.current.path}/assets/models');

  test('every bundled model reports positive float32-aligned tensor sizes', () {
    expect(modelsDir.existsSync(), isTrue, reason: 'missing assets/models');

    final models =
        modelsDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.tflite'))
            .toList()
          ..sort((a, b) => a.path.compareTo(b.path));
    expect(models, isNotEmpty);

    for (final file in models) {
      final name = file.uri.pathSegments.last;
      final CompiledModel model = CompiledModel.fromBuffer(
        file.readAsBytesSync(),
        accelerators: const {Accelerator.cpu},
      );
      try {
        final sizes = <String, List<int>>{
          'input': model.inputByteSizes,
          'output': model.outputByteSizes,
        };
        sizes.forEach((kind, list) {
          expect(list, isNotEmpty, reason: '$name has no $kind tensors');
          for (int i = 0; i < list.length; i++) {
            final int bytes = list[i];
            expect(
              bytes,
              greaterThan(0),
              reason:
                  '$name $kind[$i] byte size $bytes is not positive; '
                  'litert compiledFloatCount would throw where the previous '
                  'local helper returned a non-positive count',
            );
            expect(
              bytes % 4,
              0,
              reason: '$name $kind[$i] byte size $bytes is not float32-aligned',
            );
            // Inside the agreeing domain the shared helper must reproduce the
            // previous local arithmetic exactly.
            expect(
              compiledFloatCount(bytes, label: '$name $kind[$i]'),
              bytes ~/ 4,
            );
          }
        });
      } finally {
        model.close();
      }
    }
  });
}
