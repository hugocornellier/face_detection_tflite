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

## Features

- On-device face detection, runs fully offline
- 468 point mesh, face landmarks, iris landmarks and bounding boxes
- All coordinates are in absolute pixel coordinates
- Truly cross-platform: compatible with Android, iOS, macOS, Windows, and Linux
- The [example](https://pub.dev/packages/face_detection_tflite/example) app illustrates how to detect and render results on images 
  - Includes demo for bounding boxes, the 468-point mesh, facial landmarks and iris landmarks.

## Quick Start

```dart
import 'dart:io';
import 'dart:math'; // Required to work with Point<double> coordinates
import 'package:face_detection_tflite/face_detection_tflite.dart';

Future main() async {
  // Initialize & set model
  FaceDetector detector = FaceDetector();
  await detector.initialize(model: FaceDetectionModel.backCamera);

  // Detect faces
  final imageBytes = await File('path/to/image.jpg').readAsBytes();
  List<Face> faces = await detector.detectFaces(imageBytes);

  // Access results
  for (Face face in faces) {
    final landmarks = face.landmarks;
    final bbox = face.bbox;
    final mesh = face.mesh;
    final irises = face.irises;

    // FaceLandmarkType can be any of these:
    // leftEye, rightEye, noseTip, mouth, leftEyeTragion, or rightEyeTragion
    final leftEye = landmarks[FaceLandmarkType.leftEye];
    print('Left eye: (${leftEye?.x}, ${leftEye?.y})');
  }

  // Don't forget to clean-up when you're done!
  detector.dispose();
}
```

## Bounding Boxes

The bbox property returns a BoundingBox object representing the face bounding box in
absolute pixel coordinates. The BoundingBox provides convenient access to corner points,
dimensions (width and height), and the center point.

### Accessing Corners

```dart
import 'dart:math'; // Required for Point<double>

final BoundingBox bbox = face.bbox;

// Access individual corners by name (each is a Point<double> with x and y)
final Point<double> topLeft     = bbox.topLeft;       // Top-left corner
final Point<double> topRight    = bbox.topRight;      // Top-right corner
final Point<double> bottomRight = bbox.bottomRight;   // Bottom-right corner
final Point<double> bottomLeft  = bbox.bottomLeft;    // Bottom-left corner

// Access dimensions and center
final double width  = bbox.width;     // Width in pixels
final double height = bbox.height;    // Height in pixels
final Point<double> center = bbox.center;  // Center point

// Access all corners as a list (order: top-left, top-right, bottom-right, bottom-left)
final List<Point<double>> allCorners = bbox.corners;

// Access coordinates
print('Top-left: (${topLeft.x}, ${topLeft.y})');
print('Size: ${width} x ${height}');
print('Center: (${center.x}, ${center.y})');
```

## Landmarks

The landmarks property returns a FaceLandmarks object with 6 key facial feature points
in absolute pixel coordinates. These landmarks provide quick access to common facial
features with convenient named properties.

### Accessing Landmarks

```dart
import 'dart:math'; // Required for Point<double>

final FaceLandmarks landmarks = face.landmarks;

// Access individual landmarks using named properties
final Point<double>? leftEye  = landmarks.leftEye;
final Point<double>? rightEye = landmarks.rightEye;
final Point<double>? noseTip  = landmarks.noseTip;
final Point<double>? mouth    = landmarks.mouth;
final Point<double>? leftEyeTragion  = landmarks.leftEyeTragion;
final Point<double>? rightEyeTragion = landmarks.rightEyeTragion;

// Access coordinates
print('Left eye: (${leftEye?.x}, ${leftEye?.y})');
print('Nose tip: (${noseTip?.x}, ${noseTip?.y})');

// Backwards compatible: map-like access still works
final Point<double>? leftEyeAlt = landmarks[FaceLandmarkType.leftEye];

// Iterate through all landmarks
for (final point in landmarks.values) {
  print('Landmark: (${point.x}, ${point.y})');
}
```

## Face Mesh

The mesh property returns a list of 468 facial landmark points that form a detailed 3D
face mesh in absolute pixel coordinates. These points map to specific facial features
and can be used for precise face tracking and rendering.

### Accessing Points

  ```dart
  import 'dart:math'; // Required for Point<double>

  final List<Point<double>> mesh = face.mesh;

  // Total number of points (always 468)
  print('Mesh points: ${mesh.length}');

  // Iterate through all points
  for (int i = 0; i < mesh.length; i++) {
    final Point<double> point = mesh[i];
    print('Point $i: (${point.x}, ${point.y})');
  }

  // Access individual points (each is a Point<double> with x and y)
  final Point<double> noseTip = mesh[1];     // Nose tip point
  final Point<double> leftEye = mesh[33];    // Left eye point
  final Point<double> rightEye = mesh[263];  // Right eye point
  ```

## Irises

The irises property returns detailed iris tracking data for both eyes in absolute pixel
coordinates. Each iris includes the center point and 5 contour points that outline the
iris boundary. Only available in FaceDetectionMode.full.

### Accessing Iris Data

```dart
import 'dart:math'; // Required for Point<double>

final IrisPair? irises = face.irises;

// Access left and right iris (each is an Iris object)
final Iris? leftIris = irises?.leftIris;
final Iris? rightIris = irises?.rightIris;

// Access iris center
final Point<double>? leftCenter = leftIris?.center;
print('Left iris center: (${leftCenter?.x}, ${leftCenter?.y})');

// Access iris contour points (4 points outlining the iris)
final List<Point<double>>? leftContour = leftIris?.contour;
for (int i = 0; i < (leftContour?.length ?? 0); i++) {
  final Point<double> point = leftContour![i];
  print('Left iris contour $i: (${point.x}, ${point.y})');
}

// Right iris
final Point<double>? rightCenter = rightIris?.center;
final List<Point<double>>? rightContour = rightIris?.contour;
print('Right iris center: (${rightCenter?.x}, ${rightCenter?.y})');
```

## Face Detection Modes

This app supports three detection modes that determine which facial features are detected:

| Mode | Features | Est. Time per Face* |
|------|----------|---------------------|
| **Full** (default) | Bounding boxes, landmarks, 468-point mesh, iris tracking | ~80-120ms           |
| **Standard** | Bounding boxes, landmarks, 468-point mesh | ~60ms               |
| **Fast** | Bounding boxes, landmarks | ~30ms               |

*Est. times per faces are based on 640x480 resolution on modern hardware. Performance scales with image size and number of faces.

### Code Examples

The Face Detection Mode can be set using the `mode` parameter when detectFaces is called. Defaults to FaceDetectionMode.full.

```dart
// Full mode (default): bounding boxes, 6 basic landmarks + mesh + iris
// note: full mode provides superior accuracy for left and right eye landmarks
// compared to fast/standard modes. use full mode when precise eye landmark
// detection is required for your application. trade-off: longer inference
await faceDetector.detectFaces(bytes, mode: FaceDetectionMode.full);

// Standard mode: bounding boxes, 6 basic landmarks + mesh. inference time 
// is faster than full mode, but slower than fast mode.
await faceDetector.detectFaces(bytes, mode: FaceDetectionMode.standard);

// Fast mode: bounding boxes + 6 basic landmarks only. fastest inference
// time of the three modes.
await faceDetector.detectFaces(bytes, mode: FaceDetectionMode.fast);
```

Try the [sample code](https://pub.dev/packages/face_detection_tflite/example) from the pub.dev example tab to easily compare
modes and inferences timing.

## Models

This package supports multiple detection models optimized for different use cases:

| Model | Best For | 
|-------|----------|
| **backCamera** (default) | Group shots, distant faces, rear camera | 
| **frontCamera** | Selfies, close-up portraits, front camera | 
| **shortRange** | Close-range faces (within ~2m) |
| **full** | Mid-range faces (within ~5m) |
| **fullSparse** | Mid-range faces with faster inference (~30% speedup) | 

### Code Examples

The model can be set using the `model` parameter when initialize is called. Defaults to FaceDetectionModel.backCamera.

```dart
FaceDetector faceDetector = FaceDetector();

// backCamera (default): larger model for group shots or images with smaller faces
await faceDetector.initialize(model: FaceDetectionModel.backCamera);

// frontCamera: optimized for selfies and close-up portraits
await faceDetector.initialize(model: FaceDetectionModel.frontCamera);

// shortRange: best for short-range images (faces within ~2m)
await faceDetector.initialize(model: FaceDetectionModel.shortRange);

// full: best for mid-range images (faces within ~5m)
await faceDetector.initialize(model: FaceDetectionModel.full);

// fullSparse: same detection quality as full but runs up to 30% faster on CPU
// (slightly higher precision, slightly lower recall)
await faceDetector.initialize(model: FaceDetectionModel.fullSparse);
```

## Example

The [sample code](https://pub.dev/packages/face_detection_tflite/example) from the pub.dev example tab includes a 
Flutter app that paints detections onto an image: bounding boxes, landmarks, mesh, and iris. The 
example code provides inference time, and demonstrates when to use `FaceDetectionMode.standard` or `FaceDetectionMode.fast`.  

## Inspiration

At the time of development, there was no open-source solution for cross-platform, on-device face and landmark detection.
This package took inspiration and was ported from the original Python project **[patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite)**. Many thanks to the original author.
