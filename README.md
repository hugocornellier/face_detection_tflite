# face_detection_tflite

A Dart/Flutter package that runs an on-device face and landmark detection with TensorFlow Lite:

![Example Screenshot](assets/screenshots/example1.png)

---

## Features

- Face detection (multiple SSD variants)
- 468-point face mesh
- Face landmarks, iris landmarks and bounding boxes
- Convenient end-to-end pipeline

This package is a Flutter/Dart port inspired by and adapted from the original Python project **[patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite)**. Many thanks to the original author.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Types](#types)
- [Example](#example)

---

## Features

- Face (with bounding box), landmark and iris detection using TensorFlow Lite models
- All coordinates are returned directly in **pixel space** (`Point<double>`), no normalization or scaling required
- Works on Android, iOS, macOS, Windows, and Linux
- The `example/` app illustrates how to detect and render results on images: bounding boxes, a 468-point face mesh, and iris landmarks.

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

// Full pipeline
Future<void> analyze(Uint8List imageBytes) async {
  final result = await detector.detectFaces(imageBytes);

  for (final face in result.faces) {
    final bbox    = face.bboxCorners;     // List<Point<double>> (4 pixel corners)
    final mesh    = face.mesh;            // List<Point<double>> (468-point mesh in pixels)
    final irises  = face.irises;          // List<Point<double>> (iris landmarks in pixels)
    final lm      = face.landmarks;       // Map<FaceIndex, Point<double>> (keypoints in pixels)

    // Iterate all landmarks with their FaceIndex keys
    for (final entry in lm.entries) {
      final idx = entry.key;              // FaceIndex.leftEye, rightEye, noseTip, etc.
      final pt  = entry.value;            // Point<double> in pixels
      final px  = pt.x;
      final py  = pt.y;
    }

    // Access landmarks directly by enum
    final leftEye  = face.landmarks[FaceIndex.leftEye];    // Point<double> in pixels
    final rightEye = face.landmarks[FaceIndex.rightEye];   // Point<double> in pixels
  }
}

// When you're done with it
detector.dispose();
```

---

## Models

You can choose from several detection models depending on your use case:

- **FaceDetectionModel.backCamera** – larger model for group shots or images with smaller faces (default).
- **FaceDetectionModel.frontCamera** – optimized for selfies and close-up portraits.
- **FaceDetectionModel.short** – best for short-range images (faces within ~2m).
- **FaceDetectionModel.full** – best for mid-range images (faces within ~5m).
- **FaceDetectionModel.fullSparse** – same detection quality as `full` but runs up to 30% faster on CPU (slightly higher precision, slightly lower recall).

---

## Types

- `FaceResult` contains `bboxCorners`, `mesh`, `irises` and `landmarks`.
- `face.landmarks` is a `Map<FaceIndex, Point<double>>`, where `FaceIndex` is one of:
    - `FaceIndex.leftEye`
    - `FaceIndex.rightEye`
    - `FaceIndex.noseTip`
    - `FaceIndex.mouth`
    - `FaceIndex.leftEyeTragion`
    - `FaceIndex.rightEyeTragion`
- All coordinates are **absolute pixel positions**, ready to use for drawing or measurement.

---

## Example

The `example/` directory includes a minimal Flutter app that demonstrates how to paint detections onto an  
image: drawing face bounding boxes, the 468-point face mesh, and iris landmarks.  
Because results are already in pixel space, overlays align directly with the source image without any extra scaling.