# face_detection_tflite

[![pub points](https://img.shields.io/pub/points/face_detection_tflite?color=2E8B57&label=pub%20points)](https://pub.dev/packages/face_detection_tflite/score)
[![pub package](https://img.shields.io/pub/v/face_detection_tflite.svg)](https://pub.dev/packages/face_detection_tflite)

Flutter implementation of Google's MediaPipe face and facial landmark detection models using TensorFlow Lite.
Completely local: no remote API, just pure on-device, offline detection.

### Bounding Boxes

![Example Screenshot](assets/screenshots/group-shot-bounding-box-ex1.png)

### Facial Mesh (468-Point)

![Example Screenshot](assets/screenshots/mesh-ex1.png)

### Facial Landmarks

![Example Screenshot](assets/screenshots/landmark-ex1.png)

### Eye Tracking

#### Iris Detection:

![Example Screenshot](assets/screenshots/iris-detection-ex1.png)

#### Eye Area Mesh (71-Point):

Note: The Facial mesh and eye area mesh are separate. 

![Example Screenshot](assets/screenshots/eyemesh-ex1.png)

#### Eye Contour:

![Example Screenshot](assets/screenshots/eyecontour-ex1.png)

## Features

- On-device face detection, runs fully offline
- **Face recognition**: 192-dim embeddings to identify/compare faces across images
- **Selfie segmentation**: separate person from background, or use multiclass model for 6-class body part segmentation (hair, face, body, clothes, etc.)
- 468 point mesh with **3D depth information** (x, y, z coordinates)
- Face landmarks, comprehensive eye tracking (iris + 71-point eye mesh), and bounding boxes
- All coordinates are in absolute pixel coordinates
- Truly cross-platform: compatible with Android, iOS, macOS, Windows, and Linux
- Native OpenCV preprocessing (resize/letterbox/crops) for 2x+ throughput vs pure Dart
- The [example](https://pub.dev/packages/face_detection_tflite/example) app illustrates how to detect and render results on images
  - Includes demo for bounding boxes, the 468-point mesh, facial landmarks and comprehensive eye tracking.

## Quick Start

```dart
import 'dart:io';
import 'package:face_detection_tflite/face_detection_tflite.dart';

Future main() async {
  FaceDetector detector = FaceDetector();
  await detector.initialize(model: FaceDetectionModel.backCamera);

  final imageBytes = await File('path/to/image.jpg').readAsBytes();
  List<Face> faces = await detector.detectFaces(imageBytes);

  for (final face in faces) {
    final boundingBox = face.boundingBox;
    final landmarks   = face.landmarks;
    final mesh = face.mesh;
    final eyes = face.eyes;
  }

  detector.dispose();
}
```

## Performance

Version 4.1 moved image preprocessing to native OpenCV (via `opencv_dart`) for ~2x faster performance with SIMD acceleration. The standard `detectFaces()` method now uses OpenCV internally, so all existing code automatically gets the performance boost.

### Hardware Acceleration

The package automatically selects the best acceleration strategy for each platform:

| Platform | Default Delegate | Speedup | Notes |
|----------|-----------------|---------|-------|
| **macOS** | XNNPACK | 2-5x | SIMD vectorization (NEON on ARM, AVX on x86) |
| **Linux** | XNNPACK | 2-5x | SIMD vectorization |
| **iOS** | Metal GPU | 2-4x | Hardware GPU acceleration |
| **Android** | CPU | 1x | GPU delegate unreliable (see below) |
| **Windows** | CPU | 1x | XNNPACK crashes on Windows |

No configuration needed - just call `initialize()` and you get the optimal performance for your platform.

### Android Performance Note

The Android GPU delegate has known compatibility issues across different devices and Android versions:
- OpenCL unavailable on many devices (Pixel 6+, Android 12+)
- OpenGL ES 3.1+ required for fallback
- Some devices crash during GPU delegate initialization

For maximum compatibility, Android defaults to CPU-only execution. If you want to experiment with GPU acceleration on Android (at your own risk), see the [Advanced Configuration](#advanced-performance-configuration) section.

### Advanced Performance Configuration

```dart
// Auto mode (default) - optimal for each platform
await detector.initialize();
// Equivalent to:
await detector.initialize(performanceConfig: PerformanceConfig.auto());

// Force XNNPACK (desktop only - macOS/Linux)
await detector.initialize(
  performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
);

// Force GPU delegate (iOS recommended, Android experimental)
await detector.initialize(
  performanceConfig: PerformanceConfig.gpu(),
);

// CPU-only (maximum compatibility)
await detector.initialize(
  performanceConfig: PerformanceConfig.disabled,
);
```

### Advanced: Direct Mat Input

For live camera streams, you can bypass image encoding/decoding entirely by passing a `cv.Mat` directly to `detectFaces()`:

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

Future<void> processFrame(cv.Mat frame) async {
  final detector = FaceDetector();
  await detector.initialize(model: FaceDetectionModel.frontCamera);

  // Direct Mat input - fastest for video streams
  final faces = await detector.detectFacesFromMat(frame, mode: FaceDetectionMode.fast);

  frame.dispose(); // always dispose Mats after use
  detector.dispose();
}
```

**When to use `cv.Mat` input:**
- Live camera streams where frames are already in memory
- When you need to preprocess images with OpenCV before detection
- Maximum throughput scenarios (avoids JPEG encode/decode overhead)

**For all other cases**, pass image bytes (`Uint8List`) to `detectFaces()`.

## Bounding Boxes

The boundingBox property returns a BoundingBox object representing the face bounding box in
absolute pixel coordinates. The BoundingBox provides convenient access to corner points,
dimensions (width and height), and the center point.

### Accessing Corners

```dart

final BoundingBox boundingBox = face.boundingBox;

// Access individual corners by name (each is a Point with x and y)
final Point topLeft     = boundingBox.topLeft;       // Top-left corner
final Point topRight    = boundingBox.topRight;      // Top-right corner
final Point bottomRight = boundingBox.bottomRight;   // Bottom-right corner
final Point bottomLeft  = boundingBox.bottomLeft;    // Bottom-left corner

// Access coordinates
print('Top-left: (${topLeft.x}, ${topLeft.y})');
```

### Additional Bounding Box Parameters

```dart

final BoundingBox boundingBox = face.boundingBox;

// Access dimensions and center
final double width  = boundingBox.width;     // Width in pixels
final double height = boundingBox.height;    // Height in pixels
final Point center = boundingBox.center;  // Center point

// Access coordinates
print('Size: ${width} x ${height}');
print('Center: (${center.x}, ${center.y})');

// Access all corners as a list (order: top-left, top-right, bottom-right, bottom-left)
final List<Point> allCorners = boundingBox.corners;
```

## Landmarks

The landmarks property returns a FaceLandmarks object with 6 key facial feature points
in absolute pixel coordinates. These landmarks provide quick access to common facial
features with convenient named properties.

### Accessing Landmarks

```dart
final FaceLandmarks landmarks = face.landmarks;

// Access individual landmarks using named properties
final leftEye  = landmarks.leftEye;
final rightEye = landmarks.rightEye;
final noseTip  = landmarks.noseTip;
final mouth    = landmarks.mouth;
final leftEyeTragion  = landmarks.leftEyeTragion;
final rightEyeTragion = landmarks.rightEyeTragion;

// Access coordinates
print('Left eye: (${leftEye?.x}, ${leftEye?.y})');
print('Nose tip: (${noseTip?.x}, ${noseTip?.y})');

// Iterate through all landmarks
for (final point in landmarks.values) {
  print('Landmark: (${point.x}, ${point.y})');
}
```

## Face Mesh

The `mesh` property returns a `FaceMesh` object containing 468 facial landmark points with both
2D and 3D coordinate access. These points map to specific facial features and can be used for
precise face tracking and rendering.

### Accessing Mesh Points

  ```dart
  import 'package:face_detection_tflite/face_detection_tflite.dart';

  final FaceMesh? mesh = face.mesh;

  if (mesh != null) {
    // Get mesh points
    final points = mesh.points;

    // Total number of points (always 468)
    print('Mesh points: ${points.length}');

    // Iterate through all points (all mesh points have z-coordinates)
    for (int i = 0; i < points.length; i++) {
      final point = points[i];
      print('Point $i: (${point.x}, ${point.y}, ${point.z})');
    }

    // Access individual points using index operator
    final noseTip = mesh[1];     // Nose tip point
    final leftEye = mesh[33];    // Left eye point
    final rightEye = mesh[263];  // Right eye point
  }
  ```

### Accessing 3D Depth Information

All face mesh points include x, y, and z coordinates. The z coordinate represents
relative depth (scale-dependent). 3D coordinates are always computed for mesh and iris landmarks.

  ```dart
  import 'package:face_detection_tflite/face_detection_tflite.dart';

  final FaceMesh? mesh = face.mesh;

  if (mesh != null) {
    // Get all points
    final points = mesh.points;

    // Iterate through all points (all mesh points have x, y, and z)
    for (final point in points) {
      print('Point: (${point.x}, ${point.y}, ${point.z})');
    }

    // Access individual points directly using index operator
    final noseTip = mesh[1];
    print('Nose tip depth: ${noseTip.z}');
  }
  ```

## Eye Tracking (Iris + Eye Mesh)

The `eyes` property returns comprehensive eye tracking data for both eyes in absolute pixel
coordinates. Each eye includes:
- **Iris center** (`irisCenter`): The iris center point
- **Iris contour** (`irisContour`): 4 points outlining the iris boundary
- **Contour** (`contour`): 15 points outlining the eyelid
- **Mesh** (`mesh`): 71 landmarks covering the entire eye region

Only available in FaceDetectionMode.full.

### Accessing Eye Data

```dart

final EyePair? eyes = face.eyes;

// Access left and right eye data (each is an Eye object containing all eye info)
final Eye? leftEye = eyes?.leftEye;
final Eye? rightEye = eyes?.rightEye;

if (leftEye != null) {
  // Access iris center
  final irisCenter = leftEye.irisCenter;
  print('Left iris center: (${irisCenter.x}, ${irisCenter.y})');

  // Access iris contour points (4 points outlining the iris)
  for (final point in leftEye.irisContour) {
    print('Iris contour: (${point.x}, ${point.y})');
  }

  // Access eye mesh landmarks (71 points covering the entire eye region)
  for (final point in leftEye.mesh) {
    print('Eye mesh point: (${point.x}, ${point.y})');
  }

  // Access just the eyelid contour (first 15 points of the eye mesh)
  for (final point in leftEye.contour) {
    print('Eyelid contour: (${point.x}, ${point.y})');
  }
}

// Right eye works the same way
if (rightEye != null) {
  final irisCenter = rightEye.irisCenter;
  print('Right iris center: (${irisCenter.x}, ${irisCenter.y})');
}
```

### Rendering Eye Contours

For rendering the visible eyelid outline, use the `contour` getter and connect them using `eyeLandmarkConnections`:

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

// Get the visible eyeball contour (first 15 of 71 points)
final List<Point> eyelidOutline = leftEye.contour;

// Draw the eyelid outline by connecting the points
for (final connection in eyeLandmarkConnections) {
  final p1 = eyelidOutline[connection[0]];
  final p2 = eyelidOutline[connection[1]];
  canvas.drawLine(
    Offset(p1.x, p1.y),
    Offset(p2.x, p2.y),
    paint,
  );
}
```

## Face Detection Modes

This package supports three detection modes that determine which facial features are detected:

| Mode | Features | Est. Time per Face* |
|------|----------|---------------------|
| **Full** (default) | Bounding boxes, landmarks, 468-point mesh, eye tracking (iris + 71-point eye mesh) | ~80-120ms           |
| **Standard** | Bounding boxes, landmarks, 468-point mesh | ~60ms               |
| **Fast** | Bounding boxes, landmarks | ~30ms               |

*Est. times per faces are based on 640x480 resolution on modern hardware. Performance scales with image size and number of faces.

### Code Examples

The Face Detection Mode can be set using the `mode` parameter. Defaults to FaceDetectionMode.full.

```dart
// Full mode (default): bounding boxes, 6 basic landmarks + mesh + comprehensive eye tracking
// note: in full mode, landmarks.leftEye and landmarks.rightEye are replaced with
// iris-refined coordinates, providing significantly more accurate eye positions
// compared to the raw detection keypoints used in fast/standard modes.
// use full mode when precise eye tracking (iris center, contour, eyelid shape) is required.
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

## Live Camera Detection

![Example Screenshot](assets/screenshots/livecamera_ex1.gif)

For real-time face detection with a camera feed, pass a `cv.Mat` directly to `detectFacesFromMat()` to avoid repeated JPEG encode/decode overhead. This provides the best performance for video streams.

```dart
import 'package:camera/camera.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

FaceDetector detector = FaceDetector();
await detector.initialize(model: FaceDetectionModel.frontCamera);

final cameras = await availableCameras();
CameraController camera = CameraController(cameras.first, ResolutionPreset.medium);
await camera.initialize();

camera.startImageStream((CameraImage image) async {
  // Convert CameraImage (YUV420) directly to cv.Mat (BGR)
  final cv.Mat mat = convertCameraImageToMat(image); // see example app

  // Detect faces using Mat for maximum performance
  List<Face> faces = await detector.detectFacesFromMat(
    mat,
    mode: FaceDetectionMode.fast,
  );

  // Always dispose Mat after use
  mat.dispose();

  // Process faces...
});
```

**Tips for camera detection:**
- Pass `cv.Mat` directly to `detectFacesFromMat()` to bypass JPEG encoding/decoding
- Convert YUV420 camera frames directly to BGR Mat format
- Always call `mat.dispose()` after detection
- Use `FaceDetectionMode.fast` for real-time performance

See the full [example app](https://pub.dev/packages/face_detection_tflite/example) for complete implementation including YUV-to-Mat conversion and frame throttling.

## Background Isolate Detection

For applications that require guaranteed non-blocking UI, use `FaceDetectorIsolate`. This runs the **entire** detection pipeline in a background isolate, ensuring all processing happens off the main thread.

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

// Spawn isolate (loads models in background)
final detector = await FaceDetectorIsolate.spawn();

// All detection runs in background isolate - UI never blocked
final faces = await detector.detectFaces(imageBytes);

for (final face in faces) {
  print('Face at: ${face.boundingBox.center}');
  print('Mesh points: ${face.mesh?.length ?? 0}');
}

// Cleanup when done
await detector.dispose();
```

### When to Use FaceDetectorIsolate

| Use Case | Recommended |
|----------|-------------|
| Live camera with 60fps UI requirement | `FaceDetectorIsolate` |
| Processing images in a batch queue | `FaceDetectorIsolate` |
| Simple single-image detection | `FaceDetector` |
| Maximum control over pipeline stages | `FaceDetector` |

### Configuration

`FaceDetectorIsolate.spawn()` accepts the same configuration options as `FaceDetector.initialize()`, except for `InterpreterOptions`:

```dart
final detector = await FaceDetectorIsolate.spawn(
  model: FaceDetectionModel.frontCamera,
  performanceConfig: PerformanceConfig.auto(), // Or .gpu() for iOS
  meshPoolSize: 2,
);
```

### OpenCV Mat Support

`FaceDetectorIsolate` fully supports OpenCV `cv.Mat` input, ideal for live camera processing:

```dart
import 'package:opencv_dart/opencv_dart.dart' as cv;

// From cv.Mat (e.g., decoded image or camera frame)
final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
final faces = await detector.detectFacesFromMat(mat);
mat.dispose();

// From raw BGR bytes (e.g., converted camera YUV)
final faces = await detector.detectFacesFromMatBytes(
  bgrBytes,
  width: frameWidth,
  height: frameHeight,
);
```

The Mat is reconstructed in the background isolate using zero-copy transfer, so there's no encoding/decoding overhead.

## Face Recognition (Embeddings) 

Generate 192-dimensional identity vectors to compare faces across images. Useful for identifying the same person in different photos.

```dart
final detector = FaceDetector();
await detector.initialize();

// Get reference embedding from a photo with one face
final refFaces = await detector.detectFaces(photo1Bytes, mode: FaceDetectionMode.fast);
final refEmbedding = await detector.getFaceEmbedding(refFaces.first, photo1Bytes);

// Compare against faces in another photo
final faces = await detector.detectFaces(photo2Bytes, mode: FaceDetectionMode.fast);
for (final face in faces) {
  final embedding = await detector.getFaceEmbedding(face, photo2Bytes);
  final similarity = FaceDetector.compareFaces(refEmbedding, embedding);
  print('Similarity: ${similarity.toStringAsFixed(2)}'); // -1.0 to 1.0
}

detector.dispose();
```

**Similarity thresholds:**
- `> 0.6` — Very likely same person
- `> 0.5` — Probably same person
- `< 0.3` — Different people

Also available: `FaceDetector.faceDistance()` for Euclidean distance, and batch processing with `getFaceEmbeddings()`.

## Selfie Segmentation

Separate people from backgrounds using MediaPipe Selfie Segmentation. Useful for virtual backgrounds, portrait effects, and background blur.

### Standalone Usage 

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

final segmenter = await SelfieSegmentation.create();

final mask = await segmenter.callFromBytes(imageBytes);

// mask.width, mask.height - mask dimensions (model resolution)
// mask.at(x, y) - probability (0.0-1.0) that pixel is a person

// Convert to binary mask (0 or 255)
final binary = mask.toBinary(threshold: 0.5);

// Convert to grayscale (0-255)
final grayscale = mask.toUint8();

// Upsample to original image size
final fullSize = mask.upsample();

segmenter.dispose();
```

### With FaceDetector

```dart
final detector = FaceDetector();
await detector.initialize();

// Defaults to SegmentationConfig.safe (CPU-only, 1024 max output).
// On iOS/desktop, use SegmentationConfig.performance for hardware acceleration.
await detector.initializeSegmentation();

final mask = await detector.getSegmentationMask(imageBytes);
// Use mask for background replacement...

detector.dispose();
```

### With FaceDetectorIsolate

```dart
final detector = await FaceDetectorIsolate.spawn(
  withSegmentation: true,
  segmentationConfig: SegmentationConfig(model: SegmentationModel.general),
);

final mask = await detector.getSegmentationMask(imageBytes);
// Or from cv.Mat for camera streams:
final mask = await detector.getSegmentationMaskFromMat(mat);

await detector.dispose();
```

### Model Variants

| Model | Input Size | Output | Best For |
|-------|------------|--------|----------|
| **general** (default) | 256×256 | Binary | Portraits, square images |
| **landscape** | 144×256 | Binary | Wide images, video streams |
| **multiclass** | 256×256 | 6 classes | Body part segmentation |

```dart
// Use landscape model for video
final segmenter = await SelfieSegmentation.create(
  config: SegmentationConfig(model: SegmentationModel.landscape),
);

// Use multiclass for body part segmentation
final segmenter = await SelfieSegmentation.create(
  config: SegmentationConfig(model: SegmentationModel.multiclass),
);
```

### Multiclass Segmentation

The `multiclass` model segments images into 6 body part classes:

| Class Index | Class Name | Description |
|-------------|------------|-------------|
| 0 | Background | Non-person pixels |
| 1 | Hair | Hair regions |
| 2 | Body Skin | Arms, hands, legs (exposed skin) |
| 3 | Face Skin | Face and neck skin |
| 4 | Clothes | Clothing regions |
| 5 | Other | Accessories, hats, glasses, etc. |

```dart
final segmenter = await SelfieSegmentation.create(
  config: SegmentationConfig(model: SegmentationModel.multiclass),
);

final mask = await segmenter.callFromBytes(imageBytes);

// Check if we got a multiclass mask
if (mask is MulticlassSegmentationMask) {
  // Access individual class probability masks
  final hairMask = mask.hairMask;           // Float32List of probabilities
  final faceSkinMask = mask.faceSkinMask;
  final bodySkinMask = mask.bodySkinMask;
  final clothesMask = mask.clothesMask;
  final backgroundMask = mask.backgroundMask;
  final otherMask = mask.otherMask;

  // Or access by index
  final hairMask2 = mask.classMask(1);  // Same as hairMask

  // The base mask.data still contains combined person probability
  final combinedPerson = mask.at(x, y);
}

segmenter.dispose();
```

### Memory Considerations

The background isolate holds all TFLite models (~26-40MB for full pipeline). Always call `dispose()` when finished to release these resources. Image data is transferred using zero-copy `TransferableTypedData`, minimizing memory overhead.

## Example

The [sample code](https://pub.dev/packages/face_detection_tflite/example) from the pub.dev example tab includes a Flutter app demonstrating all features:

**Face Detection Demo:**
- Bounding boxes, landmarks, 468-point mesh, and comprehensive eye tracking
- Compare `FaceDetectionMode.fast`, `standard`, and `full` modes
- Real-time inference timing display

**Selfie Segmentation Demo:**
- Switch between `general`, `landscape`, and `multiclass` models
- Visualize individual body part masks (hair, face skin, clothes, etc.) with multiclass
- Adjustable threshold, binary/soft mask toggle, and color options
- Virtual background replacement demo in live camera mode

## Running Tests

Integration tests are located in `example/integration_test/`. Due to a Flutter macOS test runner limitation, tests must be run **one file at a time** on macOS (running all together causes app launch failures between files).

### iOS
```bash
cd example
flutter test integration_test/ -d <ios-device-id>
```

### macOS (run each file separately)
```bash
cd example

# Kill any existing instances, then run a single test file
pkill -9 -f "face_detection_tflite_example"; sleep 2
flutter test integration_test/face_detection_integration_test.dart -d macos

# Repeat for each test file:
# - opencv_helpers_test.dart (20 tests)
# - performance_config_test.dart (17 tests)
# - face_detection_integration_test.dart (97 tests)
# - embedding_match_test.dart (1 test)
# - gpu_delegate_test.dart (2 tests)
# - benchmark_test.dart (4 tests)
# - error_recovery_test.dart (27 tests)
# - edge_cases_test.dart (33 tests)
# - all_model_variants_test.dart (18 tests)
# - image_utils_test.dart (31 tests)
# - concurrency_stress_test.dart (18 tests)
# - combined_segmentation_test.dart (18 tests)
# - helpers_unit_test.dart (29 tests)
# - assertion_gaps_test.dart (18 tests)
# - selfie_segmentation_test.dart (78 tests)
# - isolate_mat_debug_test.dart (2 tests)
```

## Migrating to 5.0.0

### What changed (and why)

Version 5.0.0 removes the `package:image` dependency from `face_detection_tflite`.

All image processing (decoding, resizing, cropping, etc.) now uses OpenCV internally, which is significantly faster. This makes the `image` package unnecessary, so it has been removed.

In practice:

- If you already pass image bytes (`Uint8List`): **no changes needed**
- If you already pass a `cv.Mat`: **use the `FromMat` methods** (e.g. `detectFacesFromMat()`)
- If you were passing `img.Image` objects: **those APIs were removed** (see fix below)

### If you pass image bytes (`Uint8List`): nothing changes

This is the most common usage and it works exactly the same as before.

#### From a file

```dart
import 'dart:io';

final bytes = await File('photo.jpg').readAsBytes();
final faces = await detector.detectFaces(bytes);
```

#### From Flutter assets

```dart
import 'package:flutter/services.dart';

final data = await rootBundle.load('assets/images/photo.jpg');
final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
final faces = await detector.detectFaces(bytes);
```

#### From the network

```dart
import 'package:http/http.dart' as http;

final response = await http.get(Uri.parse('https://example.com/photo.jpg'));
final faces = await detector.detectFaces(response.bodyBytes);
```

### If you pass a `cv.Mat`: use the `FromMat` methods

If your app already works with OpenCV matrices (for example, camera frames), use the `FromMat` variant of each method:

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

final mat = imdecode(bytes, IMREAD_COLOR);
final faces = await detector.detectFacesFromMat(mat);
mat.dispose(); // always dispose Mats when you're done
```

### If you were passing `img.Image` objects

The methods that accepted `img.Image` (from `package:image`) have been removed in 5.0.0.

The fix is simple: **pass the raw bytes directly** instead of decoding to `img.Image` first.

#### Before (4.x — no longer works)

```dart
import 'package:image/image.dart' as img;

final bytes = await File('photo.jpg').readAsBytes();
final decoded = img.decodeImage(bytes)!;
final faces = await detector.detectFaces(decoded); // removed in 5.0.0
```

#### After (5.0.0)

```dart
final bytes = await File('photo.jpg').readAsBytes();
final faces = await detector.detectFaces(bytes); // just pass the bytes directly
```

#### Still want to use `package:image` for preprocessing?

If you need to crop, rotate, or otherwise manipulate images with `package:image` before detection, you can still do that. Just encode the result back to bytes before passing it in:

```dart
import 'dart:typed_data';
import 'package:image/image.dart' as img;

final originalBytes = await File('photo.jpg').readAsBytes();
final decoded = img.decodeImage(originalBytes)!;

// Do your preprocessing
final cropped = img.copyCrop(decoded, x: 0, y: 0, width: 300, height: 300);

// Encode back to bytes, then pass to detectFaces
final processedBytes = Uint8List.fromList(img.encodeJpg(cropped));
final faces = await detector.detectFaces(processedBytes);
```

### Separate typed methods

Methods now have typed overloads instead of accepting `Object`:

| Uint8List variant | cv.Mat variant |
|---|---|
| `detectFaces(bytes)` | `detectFacesFromMat(mat)` |
| `getFaceEmbedding(face, bytes)` | `getFaceEmbeddingFromMat(face, mat)` |
| `getSegmentationMask(bytes)` | `getSegmentationMaskFromMat(mat)` |

### OpenCV re-exports (no extra dependency needed)

You do **not** need to add `opencv_dart` to your own `pubspec.yaml` to use OpenCV types with this package. `face_detection_tflite` re-exports `Mat`, `imdecode`, and `IMREAD_COLOR`, so this works out of the box:

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

final mat = imdecode(bytes, IMREAD_COLOR); // no extra import needed
final faces = await detector.detectFacesFromMat(mat);
```

## Inspiration

At the time of development, there was no open-source solution for cross-platform, on-device face and landmark detection.
This package took inspiration and was ported from the original Python project **[patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite)**. Many thanks to the original author.
