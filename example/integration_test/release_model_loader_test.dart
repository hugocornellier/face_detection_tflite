// ignore_for_file: avoid_print

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:path_provider/path_provider.dart';

/// Exercises [ReleaseModelLoader] end-to-end against the real GitHub Release:
/// download → SHA-256 verification → on-disk cache. Requires network access.
///
/// `useBundledAssets: false` forces the download path even though this example
/// bundles the models, so the test runs on any platform.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // The smallest model files keep the network cost of the test low.
  const smallModel = 'face_detection_front.tflite';

  Future<File> cacheFile(String name) async {
    final dir = await getApplicationSupportDirectory();
    return File('${dir.path}/models/$name');
  }

  test('ReleaseModelLoader is assignable to ModelBytesLoader', () {
    // Compile-time proof that an instance plugs into the loadModelBytes seam.
    final ModelBytesLoader loader = ReleaseModelLoader();
    expect(loader, isNotNull);
  });

  testWidgets('downloads, verifies checksum, and caches on disk',
      (tester) async {
    final file = await cacheFile(smallModel);
    if (await file.exists()) await file.delete();

    final progress = <int>[];
    final loader = ReleaseModelLoader(
      useBundledAssets: false,
      onProgress: (name, received, total) => progress.add(received),
    );

    final bytes = await loader(smallModel);

    expect(bytes, isNotEmpty);
    // Matches the bundled asset / release size; checksum already verified
    // internally (a mismatch would have thrown).
    expect(bytes.length, 229032);
    expect(progress, isNotEmpty, reason: 'progress should be reported');
    expect(await file.exists(), isTrue, reason: 'should cache to disk');
    expect(await file.length(), bytes.length);

    // Second call comes from the disk cache (delete network by deleting the
    // download URL is overkill; a fresh loader must still find the cache).
    final fromCache = await ReleaseModelLoader(useBundledAssets: false)(
      smallModel,
    );
    expect(fromCache.length, bytes.length);
  });

  testWidgets('throws ModelChecksumException on checksum mismatch',
      (tester) async {
    final file = await cacheFile(smallModel);
    if (await file.exists()) await file.delete();

    final loader = ReleaseModelLoader(
      useBundledAssets: false,
      sha256Sums: const {smallModel: 'deadbeef'},
    );

    await expectLater(
      loader(smallModel),
      throwsA(isA<ModelChecksumException>()),
    );
    // A failed verification must not leave a cached file behind.
    expect(await file.exists(), isFalse);
  });
}
