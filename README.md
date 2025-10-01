# face_detection_tflite

A Dart/Flutter package that runs an on-device face analysis pipeline with TensorFlow Lite:

- Face detection (multiple SSD variants)
- 468-point face mesh
- Iris landmarks (both eyes)
- Convenience end-to-end pipeline or step-by-step access
- CPU-only, pure Dart API designed for Flutter apps on mobile and desktop

This package is a Flutter/Dart port inspired by and adapted from the original Python project **[patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite)**. Many thanks to the original author.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Step-by-step Usage](#step-by-step-usage)
- [Painting Notes](#painting-notes)
- [Desktop Support](#desktop-support)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Example](#example)
- [Credits & Attribution](#credits--attribution)
- [License](#license)

---

## Features

- Face, landmark and iris detection using TensorFlow Lite models
- 468-point face mesh with normalized→image-space mapping handled for you
- Iris landmark detection for both eyes via mesh-derived ROIs
- Works on Android, iOS, macOS, Windows, and Linux
- Single-file API centered on `FaceDetector`
- Example app included for testing

---

## Installation

### pub.dev
```yaml
dependencies:
  face_detection_tflite: ^0.1.0
```

Run:
```bash
flutter pub get
```

---

## Quick Start

```dart
import 'dart:typed_data';
import 'package:face_detection_tflite/face_detection_tflite.dart';

final detector = FaceDetector();

Future<void> init() async {
  // Initialize once; choose the model variant you prefer
  await detector.initialize(model: FaceDetectionModel.backCamera);
}

Future<void> analyze(Uint8List imageBytes) async {
  // All-in-one pipeline
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

A minimal example app can be placed in `example/`:

```bash
cd exam