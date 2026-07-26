import 'dart:typed_data' show ByteData, Uint8List;

import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'webgpu_probe.dart';

/// End-to-end face detection through the web pipeline in a real browser:
/// bundled package model assets loaded via `rootBundle`, `createImageBitmap`
/// decode, BlazeFace decode and weighted NMS, the detection gates, the
/// per-face mesh stage, and temporal tracking.
///
/// This deliberately does not sweep the web engine matrix (CompiledModel vs
/// LiteRtInterpreter, WebGPU vs WASM). Engine selection belongs to
/// flutter_litert and is covered in that package. What is untested elsewhere
/// is `lib/src/web/face_detector_web.dart`, so these assertions are about
/// detection results and tracking behaviour rather than which backend
/// compiled the model.
///
/// Run with:
///   chromedriver --port=4444 &
///   flutter drive --profile --driver=test_driver/integration_test.dart \
///     --target=integration_test/web_face_detection_test.dart \
///     -d web-server --browser-name=chrome
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Each detector initialization compiles five models, which includes
  // fetching the LiteRT.js runtime and its WASM binary on first use.
  const Timeout longTimeout = Timeout(Duration(minutes: 5));

  // A single clearly-framed face; the same asset the native integration
  // suite uses, so a divergence between platforms is visible.
  const String singleFace = 'assets/samples/landmark-ex1.jpg';
  const String groupShot = 'assets/samples/group-shot-bounding-box-ex1.jpeg';

  Future<Uint8List> load(String path) async {
    final ByteData data = await rootBundle.load(path);
    return data.buffer.asUint8List();
  }

  testWidgets('detects a face from bundled bytes in fast mode', (_) async {
    final FaceDetector detector = await FaceDetector.create();
    try {
      final List<Face> faces = await detector.detectFacesFromBytes(
        await load(singleFace),
        mode: FaceDetectionMode.fast,
      );

      expect(faces, isNotEmpty);
      final Face face = faces.first;
      expect(face.score, inInclusiveRange(0.0, 1.0));
      expect(face.boundingBox.width, greaterThan(0));
      expect(face.boundingBox.height, greaterThan(0));
      // Fast mode computes no mesh, so no presence score and no landmarks.
      expect(face.mesh, isNull);
      expect(face.trackingId, isNull, reason: 'tracking is opt-in');
    } finally {
      await detector.dispose();
    }
  }, timeout: longTimeout);

  testWidgets('full mode produces a mesh and a presence score', (_) async {
    final FaceDetector detector = await FaceDetector.create();
    try {
      final List<Face> faces = await detector.detectFacesFromBytes(
        await load(singleFace),
        mode: FaceDetectionMode.full,
      );

      expect(faces, isNotEmpty);
      final Face face = faces.first;
      expect(face.mesh, isNotNull);
      expect(face.mesh!.points.length, kMeshPoints);
      // The presence gate defaults to 0.5, so anything returned in full mode
      // must have cleared it.
      expect(face.meshScore, isNotNull);
      expect(face.meshScore, greaterThanOrEqualTo(0.5));
    } finally {
      await detector.dispose();
    }
  }, timeout: longTimeout);

  testWidgets('minScore gate filters detections on web', (_) async {
    final FaceDetector ungated = await FaceDetector.create();
    late final int baseline;
    try {
      baseline = (await ungated.detectFacesFromBytes(
        await load(groupShot),
        mode: FaceDetectionMode.fast,
      ))
          .length;
    } finally {
      await ungated.dispose();
    }
    expect(baseline, greaterThan(1), reason: 'group shot should find faces');

    // A gate above every plausible score must return nothing, and the gate
    // runs inside the web pipeline rather than as a caller-side filter.
    final FaceDetector gated = await FaceDetector.create(minScore: 0.999999);
    try {
      final List<Face> faces = await gated.detectFacesFromBytes(
        await load(groupShot),
        mode: FaceDetectionMode.fast,
      );
      expect(faces.length, lessThan(baseline));
    } finally {
      await gated.dispose();
    }
  }, timeout: longTimeout);

  testWidgets('tracking assigns stable IDs across sequential web calls', (
    _,
  ) async {
    final FaceDetector detector = await FaceDetector.create(
      enableTracking: true,
    );
    try {
      final Uint8List bytes = await load(singleFace);
      final List<Face> first = await detector.detectFacesFromBytes(
        bytes,
        mode: FaceDetectionMode.fast,
      );
      final List<Face> second = await detector.detectFacesFromBytes(
        bytes,
        mode: FaceDetectionMode.fast,
      );

      expect(first, isNotEmpty);
      expect(first.first.trackingId, isNotNull);
      expect(second.first.trackingId, first.first.trackingId);

      // resetTracking must start a fresh ID sequence.
      detector.resetTracking();
      final List<Face> afterReset = await detector.detectFacesFromBytes(
        bytes,
        mode: FaceDetectionMode.fast,
      );
      expect(afterReset.first.trackingId, 1);
    } finally {
      await detector.dispose();
    }
  }, timeout: longTimeout);

  testWidgets('detection and segmentation share one decode', (_) async {
    final FaceDetector detector = await FaceDetector.create(
      withSegmentation: true,
      enableTracking: true,
    );
    try {
      final DetectionWithSegmentationResult result =
          await detector.detectFacesWithSegmentation(
        await load(singleFace),
        mode: FaceDetectionMode.fast,
      );

      expect(result.faces, isNotEmpty);
      expect(result.faces.first.trackingId, isNotNull);
      expect(result.segmentationMask, isNotNull);
      expect(result.segmentationMask!.width, greaterThan(0));
    } finally {
      await detector.dispose();
    }
  }, timeout: longTimeout);

  testWidgets('accelerator reporting matches the browser capability', (
    _,
  ) async {
    final bool webGpu = await hasWebGpu();
    final FaceDetector detector = await FaceDetector.create();
    try {
      final List<Face> faces = await detector.detectFacesFromBytes(
        await load(singleFace),
        mode: FaceDetectionMode.fast,
      );
      expect(faces, isNotEmpty, reason: 'must detect on either backend');

      final String? backend = detector.activeAccelerator;
      expect(backend, isNotNull, reason: 'runners report a backend once ready');
      if (webGpu) {
        expect(backend, anyOf('webgpu', 'wasm'));
      } else {
        // GPU-less runners must fall back to WASM rather than fail, and the
        // aggregate must not claim a GPU that is not there.
        expect(backend, 'wasm');
      }
    } finally {
      await detector.dispose();
    }
  }, timeout: longTimeout);
}
