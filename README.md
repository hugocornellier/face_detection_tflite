# face_detection_tflite

A Dart/Flutter package that runs an on-device face and landmark detection with TensorFlow Lite:

- Face detection (multiple SSD variants)
- 468-point face mesh
- Iris landmarks
- Convenient end-to-end pipeline or step-by-step access

This package is a Flutter/Dart port inspired by and adapted from the original Python project **[patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite)**. Many thanks to the original author.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Step-by-step Usage](#step-by-step-usage)
- [Example](#example)

---

## Features

- Face, landmark and iris detection using TensorFlow Lite models
- Normalized → image-space mapping handled for you
- Works on Android, iOS, macOS, Windows, and Linux
- The `example/` app illustrates how to detect and render normalized results on images: 
bounding boxes, a 468-point face mesh, and iris landmarks.

---

## Quick Start

```dart
import 'dart:typed_data';
import 'package:face_detection_tflite/face_detection_tflite.dart';

final detector = FaceDetector();

// Initialize once; choose the model variant you prefer
Future<void> init() async {
  await detector.initialize(model: FaceDetectionModel.backCamera);
}

// All-in-one pipeline
Future<void> analyze(Uint8List imageBytes) async {
  final result = await detector.runAll(imageBytes);

  final detections = result.detections;   // List<Detection>
  final mesh       = result.mesh;         // List<Offset> in image coordinates
  final irises     = result.irises;       // List<Offset> in image coordinates
  final size       = result.originalSize; // Size(width, height)
}
```

---

## Step-by-step Usage

If you prefer fine-grained control:

```dart
// 1) Face detections (normalized bbox + keypoints)
final detections = await detector.getDetections(imageBytes);
if (detections.isEmpty) return;

// 2) Face mesh (mapped to image coordinates)
final mesh = await detector.getFaceMesh(imageBytes);

// 3) Iris landmarks (mapped to image coordinates)
final irises = await detector.getIris(imageBytes);

// Optionally reuse intermediate outputs:
final meshFromDets = await detector.getFaceMeshFromDetections(imageBytes, detections);
final irisFromMesh = await detector.getIrisFromMesh(imageBytes, meshFromDets);
```

Types you will encounter:
- `Detection` has `bbox` (normalized `RectF`), `score`, and `keypointsXY` (normalized).
- Mesh and iris return `List<Offset>` in your image coordinate system.
- `PipelineResult` packages detections, mesh, irises, and originalSize.

---

## Example

The `example/` directory includes a minimal Flutter app that demonstrates how to paint detections onto an 
image: drawing face bounding boxes, the 468-point face mesh, and iris landmarks. The normalized coordinates 
are rendered correctly so overlays align with the source image’s scale and aspect ratio.