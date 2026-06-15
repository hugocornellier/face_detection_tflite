import 'dart:async';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:http/http.dart' as http;

import 'model_bytes_loader.dart';
import 'release_cache_stub.dart' if (dart.library.io) 'release_cache_io.dart';

/// Base URL of the GitHub Release that hosts the package's TFLite models.
///
/// Every model file is published as a release asset, so the download URL for a
/// given file is `'$kDefaultModelReleaseBaseUrl/<fileName>'`.
const String kDefaultModelReleaseBaseUrl =
    'https://github.com/hugocornellier/face_detection_tflite/releases/download/models';

/// Expected SHA-256 checksums (lowercase hex) of every model file published to
/// the [kDefaultModelReleaseBaseUrl] release.
///
/// These match the files shipped in the package's `assets/models/` directory
/// byte-for-byte, so a downloaded model is interchangeable with the bundled
/// one. [ReleaseModelLoader] verifies each download against this table.
const Map<String, String> kModelSha256Sums = {
  'face_detection_back.tflite':
      'e376cf6b168d5ece8a3cedb94acc4eb168a136aede125ccc3d903ef38f5beda8',
  'face_detection_front.tflite':
      '3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633',
  'face_detection_full_range.tflite':
      '99bf9494d84f50acc6617d89873f71bf6635a841ea699c17cb3377f9507cfec3',
  'face_detection_full_range_sparse.tflite':
      '671dd2f9ed11a78436fc21cc42357a803dfc6f73e9fb86541be942d5716c2dce',
  'face_detection_short_range.tflite':
      '3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633',
  'face_landmark.tflite':
      '2efcb4f4de43c7614b80a3cc3e8a37354b3b3b40f75cce20f6f38f0f25d65493',
  'iris_landmark.tflite':
      'd1744d2a09c25f501d39eba4faff47e53ecca8852c5ce19bce8eeac39357521f',
  'mobilefacenet.tflite':
      'be4bc7cfc53f7bc336d0f28b1ab92535f618c913a422b683210750f6b5354854',
  'selfie_multiclass.tflite':
      'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0',
  'selfie_segmenter.tflite':
      '191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b',
  'selfie_segmenter_landscape.tflite':
      '490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e',
};

/// Reports download progress for a single model file.
///
/// [totalBytes] is null when the server does not send a `Content-Length`
/// header.
typedef ModelDownloadProgress = void Function(
  String modelFileName,
  int receivedBytes,
  int? totalBytes,
);

/// Thrown when a downloaded model's SHA-256 does not match the expected value.
class ModelChecksumException implements Exception {
  ModelChecksumException(this.fileName, this.expected, this.actual);

  final String fileName;
  final String expected;
  final String actual;

  @override
  String toString() =>
      'ModelChecksumException: $fileName checksum mismatch '
      '(expected $expected, got $actual)';
}

/// Built-in, opt-in model loader that fetches the package's TFLite models from
/// a GitHub Release on first use, caches them, and verifies every download
/// against a known SHA-256 checksum.
///
/// This is provided so apps that strip the ~28 MB of bundled models from their
/// release build (to shrink it) don't each have to reimplement downloading and
/// caching. It plugs into the same seam as a custom [ModelBytesLoader]: an
/// instance is callable, so pass it directly as the `loadModelBytes:` argument
/// of [FaceDetector.initialize], [FaceDetector.initializeSegmentation] or
/// `SelfieSegmentation.create`:
///
/// ```dart
/// final loader = ReleaseModelLoader();
/// await detector.initialize(
///   model: FaceDetectionModel.shortRange,
///   loadModelBytes: loader,
/// );
/// ```
///
/// Resolution order for each file:
/// 1. Bundled package asset (when [useBundledAssets]) — present on every
///    platform unless the app excludes them from its build.
/// 2. On-disk download cache (application support directory; in-memory on web).
/// 3. Download from [baseUrl], verify the checksum, then cache.
///
/// Reuse a single instance across detectors so concurrent and repeat requests
/// for the same file share one download.
class ReleaseModelLoader {
  ReleaseModelLoader({
    this.baseUrl = kDefaultModelReleaseBaseUrl,
    Map<String, String>? sha256Sums,
    this.verifyChecksum = true,
    this.useBundledAssets = true,
    this.onProgress,
    http.Client? httpClient,
  })  : sha256Sums = sha256Sums ?? kModelSha256Sums,
        _injectedClient = httpClient;

  /// Base URL the model files are downloaded from. See
  /// [kDefaultModelReleaseBaseUrl].
  final String baseUrl;

  /// Expected SHA-256 checksums keyed by file name. Defaults to
  /// [kModelSha256Sums]. Files absent from this map are not verified.
  final Map<String, String> sha256Sums;

  /// Whether to verify each download against [sha256Sums]. Defaults to true.
  final bool verifyChecksum;

  /// Whether to try the bundled package asset before the cache/download path.
  /// Defaults to true; set false to force the download path (e.g. in tests).
  final bool useBundledAssets;

  /// Optional download-progress callback.
  final ModelDownloadProgress? onProgress;

  final http.Client? _injectedClient;
  final ModelDiskCache _cache = const ModelDiskCache();
  final Map<String, Uint8List> _memoryCache = {};
  final Map<String, Future<Uint8List>> _inFlight = {};

  /// Resolves the raw bytes for [modelFileName].
  ///
  /// Matches the [ModelBytesLoader] signature, which is what lets an instance
  /// be passed wherever a `loadModelBytes:` callback is expected.
  Future<Uint8List> call(String modelFileName) {
    final pending = _inFlight[modelFileName];
    if (pending != null) return pending;
    final future = _resolve(modelFileName).whenComplete(() {
      _inFlight.remove(modelFileName);
    });
    _inFlight[modelFileName] = future;
    return future;
  }

  Future<Uint8List> _resolve(String fileName) async {
    if (useBundledAssets) {
      try {
        final data = await rootBundle.load(
          'packages/face_detection_tflite/assets/models/$fileName',
        );
        return data.buffer.asUint8List();
      } catch (_) {
        // Asset excluded from this build; fall through to cache/download.
      }
    }

    final memory = _memoryCache[fileName];
    if (memory != null) return memory;

    final cached = await _cache.read(fileName);
    if (cached != null) {
      if (kIsWeb) _memoryCache[fileName] = cached;
      return cached;
    }

    final bytes = await _download(fileName);
    _verify(fileName, bytes);
    await _cache.write(fileName, bytes);
    // The disk cache is the source of truth on native; only web, which has no
    // file system, needs the in-memory copy to avoid re-downloading.
    if (kIsWeb) _memoryCache[fileName] = bytes;
    return bytes;
  }

  void _verify(String fileName, Uint8List bytes) {
    if (!verifyChecksum) return;
    final expected = sha256Sums[fileName];
    if (expected == null) return;
    final actual = sha256.convert(bytes).toString();
    if (actual.toLowerCase() != expected.toLowerCase()) {
      throw ModelChecksumException(fileName, expected, actual);
    }
  }

  Future<Uint8List> _download(String fileName) async {
    final url = '$baseUrl/$fileName';
    final client = _injectedClient ?? http.Client();
    try {
      final request = http.Request('GET', Uri.parse(url));
      final response = await client.send(request);
      if (response.statusCode != 200) {
        throw http.ClientException(
          'Failed to download $fileName: HTTP ${response.statusCode}',
          request.url,
        );
      }
      final total = response.contentLength;
      final builder = BytesBuilder(copy: false);
      await for (final chunk in response.stream) {
        builder.add(chunk);
        onProgress?.call(fileName, builder.length, total);
      }
      final bytes = builder.takeBytes();
      if (bytes.isEmpty) {
        throw http.ClientException('Downloaded $fileName is empty', request.url);
      }
      return bytes;
    } finally {
      if (_injectedClient == null) client.close();
    }
  }
}
