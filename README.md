# face_detection_tflite

A pure Dart/Flutter implementation of Google's MediaPipe face detection and facial landmark models using TensorFlow Lite. 
This package provides on-device face and landmark detection with minimal dependencies, just TensorFlow Lite and image.

#### Bounding Box Example:

![Example Screenshot](assets/screenshots/group-shot-bounding-box-ex1.png)

#### Mesh (468-Point) Example:

![Example Screenshot](assets/screenshots/mesh-ex1.png)

#### Landmark Example:

![Example Screenshot](assets/screenshots/landmark-ex1.png)

#### Iris Example:

![Example Screenshot](assets/screenshots/iris-detection-ex1.png)

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Models](#models)
- [Types](#types)
- [Example](#example)
- [Inspiration](#inspiration)

## Features

- On-device face detection (multiple SSD variants)
- 468-point face mesh, face landmarks, iris landmarks and bounding boxes
- All coordinates are in **absolute pixel coordinates** (`Point<double>`)
  - `x` ranges from `0` to `image.width`
  - `y` ranges from `0` to `image.height`
  - Ready to use co-ordinates without any scaling
- Truly cross-platform: compatible with Android, iOS, macOS, Windows, and Linux
- The `example/` app illustrates how to detect and render results on images: bounding boxes, a 468-point face mesh, and iris landmarks.

## Quick Start

```dart
import 'dart:io';
import 'package:face_detection_tflite/face_detection_tflite.dart';

Future main() async {
  // 1. initialize
  final detector = FaceDetector();
  await detector.initialize(model: FaceDetectionModel.backCamera);

  // 2. detect
  final imageBytes = await File('path/to/image.jpg').readAsBytes();
  final result = await detector.detectFaces(imageBytes);

  // 3. access results
  for (final face in result.faces) {
    final landmarks = face.landmarks;
    final bbox = face.bboxCorners;  
    final mesh = face.mesh;     
    final irises = face.irises;
    
    final leftEye = landmarks[FaceIndex.leftEye];
    final rightEye = landmarks[FaceIndex.rightEye];

    print('Left eye: (${leftEye.x}, ${leftEye.y})');
  }

  // 4. clean-up
  detector.dispose();
}
```

## Models

You can choose from several detection models depending on your use case:

- **FaceDetectionModel.backCamera**: larger model for group shots or images with smaller faces (default).
- **FaceDetectionModel.frontCamera**: optimized for selfies and close-up portraits.
- **FaceDetectionModel.short**: best for short-range images (faces within ~2m).
- **FaceDetectionModel.full**: best for mid-range images (faces within ~5m).
- **FaceDetectionModel.fullSparse**: same detection quality as `full` but runs up to 30% faster on CPU (slightly higher precision, slightly lower recall).

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

## Example

The `example/` directory includes a minimal Flutter app that demonstrates how to paint detections onto an  
image: drawing face bounding boxes, the 468-point face mesh, and iris landmarks.  
Because results are already in pixel space, overlays align directly with the source image without any extra scaling.

## Inspiration

At the time of development, there was no open-source solution for cross-platform, on-device face and landmark detection.
This package took inspiration and was ported from the original Python project **[patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite)**. Many thanks to the original author.