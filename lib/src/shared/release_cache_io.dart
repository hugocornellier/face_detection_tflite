import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

/// Native implementation of the model disk cache.
///
/// Downloaded models are persisted under the application support directory in a
/// `models/` subfolder so they survive across launches. Selected automatically
/// via a conditional import on platforms that expose `dart:io`.
class ModelDiskCache {
  const ModelDiskCache();

  /// Returns the cached bytes for [fileName], or null when not cached yet.
  Future<Uint8List?> read(String fileName) async {
    final file = await _file(fileName);
    if (await file.exists() && await file.length() > 0) {
      return file.readAsBytes();
    }
    return null;
  }

  /// Persists [bytes] for [fileName].
  ///
  /// Writes to a temporary `.part` file then renames it into place, so a crash
  /// mid-write can never leave a truncated model behind to be trusted on the
  /// next launch.
  Future<void> write(String fileName, Uint8List bytes) async {
    final file = await _file(fileName);
    final tmp = File('${file.path}.part');
    await tmp.writeAsBytes(bytes, flush: true);
    await tmp.rename(file.path);
  }

  Future<File> _file(String fileName) async {
    final supportDir = await getApplicationSupportDirectory();
    final modelsDir = Directory('${supportDir.path}/models');
    await modelsDir.create(recursive: true);
    return File('${modelsDir.path}/$fileName');
  }
}
