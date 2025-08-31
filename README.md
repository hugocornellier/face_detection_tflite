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

- Face detection using TensorFlow Lite models
- 468-point face mesh with normalized→image-space mapping handled for you
- Iris landmark detection for both eyes via mesh-derived ROIs
- Single-file API centered on `FaceDetector`
- Works on Android, iOS, macOS, Windows, and Linux
- Example app can be included for testing and demonstration

---

## Installation

### Option A: pub.dev (recommended once published)
```yaml
dependencies:
  face_detection_tflite: ^0.1.0
```

### Option B: Git (current)
```yaml
dependencies:
  face_detection_tflite:
    git:
      url: https://github.com/hugocornellier/face_detection_tflite.git
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

## Painting Notes

- Scale a normalized `RectF` to your image size to draw detection boxes.
- Mesh and iris points are already returned in image coordinates; draw them directly.
- For consistent overlays, paint against the same scaled image rectangle you used for display.

---

## Desktop Support

On desktop platforms, ensure TensorFlow Lite native libraries are available. Depending on your Flutter/TFLite setup:
- **macOS**: `libtensorflowlite_c.dylib` inside the app bundle (e.g., `Contents/Frameworks`)
- **Windows**: `tensorflowlite_c.dll` next to your executable
- **Linux**: `libtensorflowlite_c.so` on the library path

Some `tflite_flutter` releases bundle these automatically; if you encounter missing-library errors, add the appropriate binary for your target platform.

---

## Performance Tips

- Use `FaceDetectionModel.shortRange` or `frontCamera` for faster results on low-end devices.
- Reuse a single `FaceDetector` instance; call `initialize` only once.
- If you only need one output (e.g., detections only), call that method instead of `runAll`.
- For real-time use, throttle calls and reuse buffers where possible.

---

## Troubleshooting

- **Model not found**: verify asset paths and `flutter: assets:` entries.
- **Empty results**: confirm lighting/face size, try a different detection model variant.
- **Slow inference**: choose a smaller model, reduce image size upstream, avoid extra copies.

---

## Example

A minimal example app can be placed in `example/`:

```bash
cd exam