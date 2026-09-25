import 'dart:typed_data';

import 'package:face_detection_tflite/src/native/face_native_lib.dart';
import 'package:flutter/foundation.dart'
    show debugDefaultTargetPlatformOverride, TargetPlatform;
import 'package:flutter_test/flutter_test.dart';

import 'test_config.dart';

/// Runs the desktop half of [FaceDetector.detectFacesFromCameraImage] without
/// a model: pack the frame with [prepareCameraFrameFromImage], passing
/// `isBgra` through as the detector does, then decode it with the detector's
/// own [cameraFrameToBgrMat].
void main() {
  globalTestSetup();
  tearDown(() => debugDefaultTargetPlatformOverride = null);

  const desktops = [
    TargetPlatform.macOS,
    TargetPlatform.linux,
    TargetPlatform.windows,
  ];

  // The same pixel (blue 10, green 20, red 30) in each byte order, and the
  // BGR results for a correct decode and for one with red and blue swapped.
  const bgra = [10, 20, 30, 255];
  const rgba = [30, 20, 10, 255];
  const bgr = [10, 20, 30];
  const swapped = [30, 20, 10];

  /// A 4x2 single-plane desktop frame filled with [pixel] that reports [raw]
  /// as its `format.raw`.
  _FakeCameraImage desktopFrame(Object? raw, List<int> pixel) {
    const width = 4, height = 2;
    final bytes = Uint8List(width * height * 4);
    for (int i = 0; i < bytes.length; i++) {
      bytes[i] = pixel[i % 4];
    }
    return _FakeCameraImage(
      width: width,
      height: height,
      planes: [_FakePlane(bytes, bytesPerRow: width * 4)],
      format: _FakeImageFormat(raw),
    );
  }

  /// Decodes [image] the way the detector does and returns pixel (0, 0) as
  /// [B, G, R].
  List<num> decodeBgr(Object image, {bool? isBgra}) {
    final frame = prepareCameraFrameFromImage(image, isBgra: isBgra)!;
    final mat = cameraFrameToBgrMat(frame);
    try {
      // atPixel returns a view of native memory: copy it before disposing.
      return mat.atPixel(0, 0).toList();
    } finally {
      mat.dispose();
    }
  }

  test("'BGRA' frames keep red and blue on every desktop platform", () {
    // camera_desktop 2.x, including Linux and Windows, where assuming BGRA
    // only on macOS swapped red and blue.
    for (final p in desktops) {
      debugDefaultTargetPlatformOverride = p;
      expect(decodeBgr(desktopFrame('BGRA', bgra)), bgr, reason: '$p');
    }
  });

  test("'RGBA' frames keep red and blue on every desktop platform", () {
    // camera_desktop 1.x on Linux and Windows.
    for (final p in desktops) {
      debugDefaultTargetPlatformOverride = p;
      expect(decodeBgr(desktopFrame('RGBA', rgba)), bgr, reason: '$p');
    }
  });

  test('an explicit isBgra overrides format.raw', () {
    debugDefaultTargetPlatformOverride = TargetPlatform.linux;
    expect(
      decodeBgr(desktopFrame('BGRA', bgra), isBgra: false),
      swapped,
      reason: 'decoded as RGBA because the caller asked for it',
    );
    expect(
      decodeBgr(desktopFrame('RGBA', rgba), isBgra: true),
      swapped,
      reason: 'decoded as BGRA because the caller asked for it',
    );
  });

  test('any other format.raw keeps the platform default', () {
    // iOS reports an integer FourCC (kCVPixelFormatType_32BGRA here).
    final frame = desktopFrame(1111970369, bgra);

    debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
    expect(decodeBgr(frame), bgr, reason: 'BGRA on macOS');

    debugDefaultTargetPlatformOverride = TargetPlatform.linux;
    expect(decodeBgr(frame), swapped, reason: 'RGBA elsewhere');
  });
}

class _FakePlane {
  _FakePlane(this.bytes, {required this.bytesPerRow});

  final Uint8List bytes;
  final int bytesPerRow;
  final int bytesPerPixel = 4;
}

class _FakeImageFormat {
  const _FakeImageFormat(this.raw);

  final Object? raw;
}

class _FakeCameraImage {
  _FakeCameraImage({
    required this.width,
    required this.height,
    required this.planes,
    required this.format,
  });

  final int width;
  final int height;
  final List<_FakePlane> planes;
  final _FakeImageFormat format;
}
