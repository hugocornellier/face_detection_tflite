import 'dart:typed_data';

/// Web (and any non-`dart:io`) implementation of the model disk cache.
///
/// Browsers have no file system, so there is nothing to persist between
/// sessions. [ReleaseModelLoader] keeps an in-memory cache for the lifetime of
/// the page instead, which is selected automatically via a conditional import.
class ModelDiskCache {
  const ModelDiskCache();

  /// Always a cache miss on platforms without a file system.
  Future<Uint8List?> read(String fileName) async => null;

  /// No-op on platforms without a file system.
  Future<void> write(String fileName, Uint8List bytes) async {}
}
