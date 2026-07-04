<h1 align="center">face_detection_tflite</h1>

<p align="center">
<a href="https://flutter.dev"><img src="https://img.shields.io/badge/Platform-Flutter-02569B?logo=flutter" alt="Platform"></a>
<a href="https://dart.dev"><img src="https://img.shields.io/badge/language-Dart-blue" alt="Language: Dart"></a>
<br>
<a href="https://pub.dev/packages/face_detection_tflite"><img src="https://img.shields.io/pub/v/face_detection_tflite?label=pub.dev&labelColor=333940&logo=dart" alt="Pub Version"></a>
<a href="https://pub.dev/packages/face_detection_tflite/score"><img src="https://img.shields.io/pub/points/face_detection_tflite?color=2E8B57&label=pub%20points" alt="pub points"></a>
<a href="https://github.com/hugocornellier/face_detection_tflite/actions/workflows/build.yml"><img src="https://github.com/hugocornellier/face_detection_tflite/actions/workflows/build.yml/badge.svg" alt="CI"></a>
<a href="https://github.com/hugocornellier/face_detection_tflite/actions/workflows/integration.yml"><img src="https://github.com/hugocornellier/face_detection_tflite/actions/workflows/integration.yml/badge.svg" alt="Tests"></a>
<a href="https://github.com/hugocornellier/face_detection_tflite/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-007A88.svg?logo=apache" alt="License"></a>
</p>

Flutter implementation of Google's MediaPipe face and facial landmark detection models using LiteRT (formerly TensorFlow Lite).
Runs 100% offline/on-device. Highly performant: full detection runs in ~8ms per face, with a fast mode around 3.5ms. All work on a background isolate so the UI thread is never blocked.

> **~5.5x faster than Google ML Kit** on equivalent face detection tasks ([benchmark source](example/integration_test/mlkit_benchmark_test.dart))<sup>[†](#mlkit-comparison-benchmark)</sup>

<p align="center">
  <img src="assets/demos/mesh-iris-portrait.webp" alt="On-device 468-point face mesh with iris and eye tracking and head pose" width="760"><br>
  <sub><i>468-point face mesh, iris + eye tracking and head pose, fully on-device</i></sub>
</p>
<p align="center">
  <img src="assets/demos/detection-and-segmentation.webp" alt="Full detection (left) and 6-class segmentation (right) of the same clip, frame-locked" width="880"><br>
  <sub><i>Full detection (left) and 6-class segmentation (right), same clip</i></sub>
</p>

## Features

- On-device face detection, runs fully offline
- Face landmarks, bounding boxes & eye tracking (iris + 71-point eye mesh)
- 468 point mesh with 3D depth information (x, y, z coordinates)
- Head pose: pitch, yaw & roll Euler angles (ML Kit compatible conventions)
- Face classification: smile probability + per-eye open probability (ML Kit compatible), plus all 52 MediaPipe blendshape coefficients
- Named face contours (ML Kit `FaceContourType` compatible): face oval, eyebrows, eyes, lips, nose and cheeks, derived from the mesh
- Selfie segmentation: separate person from background, or use multiclass model for 6-class body part segmentation (hair, face, body, clothes, etc.)
- Face recognition (embeddings): identify/compare faces across images
- Truly cross-platform: compatible with Android, iOS, macOS, Windows, Linux and Web
- Full [example](https://pub.dev/packages/face_detection_tflite/example) app (for native platforms) that illustrates how to detect and render results on images
  - Includes demo for bounding boxes, facial mesh, landmarks and eye/iris tracking.
- The <a href="https://github.com/hugocornellier/face_detection_tflite/blob/main/example_web/lib/main.dart" target="_blank">web example</a> provides a complete web demo

## Quick Start

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

Future main() async {
  // Initialize detector, run inference on image
  FaceDetector fd = await FaceDetector.create();
  List<Face> faces = await fd.detectFacesFromFilepath('path/to/image.jpg');

  // Iterate through detected faces
  for (final face in faces) {
    final boundingBox = face.boundingBox;
    final landmarks = face.landmarks;
    final mesh = face.mesh;
    final eyes = face.eyes;
  }

  await fd.dispose();
}
```

Already have bytes (from the network etc.)? Use `detectFacesFromBytes(imageBytes)`. For live camera streams, use `detectFacesFromCameraImage(...)` (keeps all OpenCV work off the UI thread, see below). For a pre-decoded `cv.Mat`, use `detectFacesFromMat(mat)`.

### Detection entry points

Pick the method that matches the input you already have. Each returns `Future<List<Face>>`:

| Input you have | Method | Input type | Example |
|----------------|--------|------------|---------|
| Image file on disk | `detectFacesFromFilepath` | `String` | shown above |
| Encoded image bytes (e.g. from the network) | `detectFacesFromBytes` | `Uint8List` (encoded JPEG/PNG) | [Encoded Image Bytes Example](#encoded-image-bytes-example) |
| Decoded OpenCV matrix | `detectFacesFromMat` | `cv.Mat` | [Direct Mat Example](#direct-mat-example) |
| Raw pixel bytes (+ width/height) | `detectFacesFromMatBytes` | `Uint8List` (raw BGR pixels) | [Raw Pixel Bytes Example](#raw-pixel-bytes-example) |
| Live camera frame | `detectFacesFromCameraImage` | `CameraImage` | [Live Camera Detection](#live-camera-detection) |

> `detectFacesFromBytes` decodes a compressed image file; `detectFacesFromMatBytes` takes already-decoded pixels and so requires `width`/`height`. Same `Uint8List` type, different content.

## Models

All TFLite models are sourced from Google's [MediaPipe](https://mediapipe.dev/) framework. The one exception is `mobilefacenet.tflite`, which is based on [MobileFaceNets](https://arxiv.org/abs/1804.07573). Where available, official model cards are archived in [`doc/model_cards/`](doc/model_cards/):

| Model | File | Model Card |
|-------|------|------------|
| Face Detection (front camera / short range) | `face_detection_front.tflite`, `face_detection_short_range.tflite` | [blazeface_short_range_model_card.pdf](doc/model_cards/blazeface_short_range_model_card.pdf) · [mediapipe.page.link/blazeface-mc](https://mediapipe.page.link/blazeface-mc) |
| Face Detection (back camera / full range) | `face_detection_back.tflite`, `face_detection_full_range.tflite` | [blazeface_full_range_model_card.pdf](doc/model_cards/blazeface_full_range_model_card.pdf) · [mediapipe.page.link/blazeface-back-mc](https://mediapipe.page.link/blazeface-back-mc) |
| Face Detection (full range sparse) | `face_detection_full_range_sparse.tflite` | [blazeface_full_range_sparse_model_card.pdf](doc/model_cards/blazeface_full_range_sparse_model_card.pdf) · [mediapipe.page.link/blazeface-back-sparse-mc](https://mediapipe.page.link/blazeface-back-sparse-mc) |
| Face Mesh (468-point landmark) | `face_landmark.tflite` | [face_landmark_model_card.pdf](doc/model_cards/face_landmark_model_card.pdf) · [mediapipe.page.link/facemesh-mc](https://mediapipe.page.link/facemesh-mc) |
| Iris Landmark (76-point) | `iris_landmark.tflite` | [iris_landmark_model_card.pdf](doc/model_cards/iris_landmark_model_card.pdf) · [mediapipe.page.link/iris-mc](https://mediapipe.page.link/iris-mc) |
| Face Blendshapes (52-coefficient) | `face_blendshapes.tflite` | [blendshape_v2_model_card.pdf](doc/model_cards/blendshape_v2_model_card.pdf) · [official PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf) |
| Selfie Segmentation | `selfie_segmenter.tflite`, `selfie_segmenter_landscape.tflite` | [selfie_segmentation_model_card.pdf](doc/model_cards/selfie_segmentation_model_card.pdf) · [mediapipe.page.link/selfiesegmentation-mc](https://mediapipe.page.link/selfiesegmentation-mc) |
| Multiclass Segmentation | `selfie_multiclass.tflite` | [multiclass_segmentation_model_card.pdf](doc/model_cards/multiclass_segmentation_model_card.pdf) |
| Face Embedding (192-dim) | `mobilefacenet.tflite` | [mobilefacenet_paper.pdf](doc/model_cards/mobilefacenet_paper.pdf) · [arXiv 1804.07573](https://arxiv.org/abs/1804.07573) |

`face_blendshapes.tflite` (955,312 bytes, SHA-256 `4f36dded049db18d76048567439b2a7f58f1daabc00d78bfe8f3ad396a2d2082`) is extracted from Google's official `face_landmarker.task` float16 v1 bundle. All models are Apache 2.0 licensed.

## Bounding Boxes

<img src="assets/screenshots/group-shot-bounding-box-ex1.png" width="600" alt="Bounding Boxes">

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

## Detection Score

Each detected face carries a confidence score: how sure the detector is that the
box actually contains a face.

```dart
final double score = face.score;

print('Confidence: ${(score * 100).toStringAsFixed(1)}%');
```

Key things to know:

- **Range is 0.0 to 1.0**, where higher means more confident.
- **In practice you only ever see scores of 0.5 or higher.** Anything below 0.5
  is filtered out internally and never returned, so 0.5 is effectively the floor.
- It reflects only "is this a face", not the quality of the landmarks, mesh, or
  any recognition/matching result.
- It is not a calibrated probability. Treat it as a relative confidence value,
  useful for ranking faces or applying your own stricter cutoff (for example,
  keeping only faces above 0.9).

```dart
// Example: keep only high-confidence faces
final strongFaces = faces.where((f) => f.score >= 0.9).toList();
```

### Mesh Score

When a face mesh is computed (standard or full mode), the mesh model reports its
own separate confidence that the aligned crop really is a face. Read it via
`face.meshScore` (a convenience proxy for `face.mesh?.score`):

```dart
final double? meshScore = face.meshScore;
if (meshScore != null) {
  print('Mesh confidence: ${(meshScore * 100).toStringAsFixed(1)}%');
}
```

- **Range is 0.0 to 1.0**, higher means more confident.
- **Null in fast mode** (no mesh is computed) or if the model omits the output.
- It is distinct from `face.score`: `face.score` comes from the detector, while
  `meshScore` comes from the mesh model on the aligned crop. A low `meshScore`
  can flag a crop the mesh model was unsure about (for example a bad alignment).
- Like `face.score`, it reflects "is this a face", not lighting, blur, or pose,
  so it is not a face-quality metric.

## Detection Gates (minScore & minFaceSize)

`FaceDetector.create()` accepts two optional gates that filter which faces come
back. Both default to `0.0` (no filtering, so existing behavior is unchanged),
and they apply on every platform and through every detection entry point.

```dart
final detector = await FaceDetector.create(
  minScore: 0.9,      // keep only faces with confidence >= 0.9
  minFaceSize: 0.15,  // keep only faces at least 15% of the image width
);
```

### `minScore`

Drops faces whose `face.score` is below the threshold, in the range `0.0` to
`1.0`. The comparison is inclusive, so a face exactly at the threshold is kept.

Because the detector already discards everything below its internal confidence
floor of `0.5` (this matches MediaPipe's default `min_detection_confidence`), a
`minScore` at or below `0.5` has no effect: only values above `0.5` tighten the
results. It cannot surface faces below `0.5`, and there is no way to loosen the
floor from this API.

### `minFaceSize`

Drops faces whose width, as a fraction of the image width, is below the
threshold (range `0.0` to `1.0`, inclusive). That fraction is exposed on every
face as `face.widthFraction`:

```dart
final double frac = face.widthFraction; // face width / image width, always 0..1
```

`minFaceSize` is inspired by Google ML Kit's `setMinFaceSize` (same head-width /
image-width ratio) with two deliberate differences:

- **Default is `0.0` here, `0.1` in ML Kit.** Defaulting to no filtering means
  adding the option never silently changes existing results.
- **It is a strict post-detection filter, not a detection-time hint.** ML Kit
  documents `minFaceSize` as the "smallest desired face size" that trades speed
  for small-face recall. Here the full detection runs and small faces are then
  removed, so `minFaceSize` changes what you get back, not how fast detection
  runs. `widthFraction` is measured on the detector's bounding box (clipped to
  the image), which approximates but is not identical to ML Kit's head
  measurement.

### Notes

- Invalid values (NaN, or outside `[0.0, 1.0]`) throw `ArgumentError` from
  `create()` / `initialize()`.
- These are filters, not a performance optimization: in `standard` and `full`
  modes the mesh and iris models still run on every detected face before gating,
  so the landmark cost is not saved.
- Native and web filter identically, including for faces near the image border
  (`widthFraction` uses the visible, image-clipped width on both).
- For ad-hoc or per-call filtering you can also use `face.score` /
  `face.widthFraction` directly, for example
  `faces.where((f) => f.widthFraction >= 0.15)`.

## Landmarks

<img src="assets/screenshots/landmark-ex1.png" width="600" alt="Facial Landmarks">

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

## Head Pose (Euler Angles)

Each `Face` exposes the head orientation as pitch, yaw and roll in degrees. The
sign conventions match Google ML Kit, so code migrating from
`google_mlkit_face_detection` can read the same values:

| Getter | Axis | Positive direction |
|--------|------|--------------------|
| `headEulerAngleX` | pitch | face tilts up |
| `headEulerAngleY` | yaw | face turns toward the right side of the image |
| `headEulerAngleZ` | roll | counter-clockwise in-plane tilt |

```dart
final angles = face.headEulerAngles; // HeadEulerAngles? (null if unavailable)
if (angles != null) {
  print('pitch: ${angles.x}, yaw: ${angles.y}, roll: ${angles.z}');
}

// Or read each axis directly (each is a double?):
final double? pitch = face.headEulerAngleX;
final double? yaw   = face.headEulerAngleY;
final double? roll  = face.headEulerAngleZ;
```

Pitch and yaw are derived from the 468-point 3D mesh, so they require a
detection mode that computes the mesh (`standard` or `full`). In `fast` mode
there is no mesh, so only roll is estimated (from the eye keypoints) and
pitch/yaw are reported as `0`.

These angles are estimates from the mesh geometry, not a calibrated pose
solver. Roll is the most reliable; yaw is good; pitch is the noisiest axis
(it depends on the mesh's relative depth, which is approximate). This mirrors
the accuracy characteristics of ML Kit's own head-angle outputs, and is
suitable for gaze/attention cues, filtering non-frontal faces, and AR
alignment rather than precise metrology.

## Face Classification (Smile & Eye-Open)

In `FaceDetectionMode.full`, each `Face` is classified by MediaPipe's
Blendshape V2 model into 52 expression coefficients, from which this package
exposes the same smile and eye-open likelihoods as Google ML Kit, plus the full
coefficient set (which ML Kit does not provide):

| Getter | Range | Meaning |
|--------|-------|---------|
| `smilingProbability` | 0.0 to 1.0 | mean of the `mouthSmileLeft`/`mouthSmileRight` blendshapes |
| `leftEyeOpenProbability` | 0.0 to 1.0 | `1 - eyeBlinkLeft` (the subject's left eye) |
| `rightEyeOpenProbability` | 0.0 to 1.0 | `1 - eyeBlinkRight` (the subject's right eye) |
| `blendshapes` | `FaceBlendshapes?` | all 52 coefficients, indexed by the `Blendshape` enum |

```dart
final faces = await detector.detectFacesFromBytes(bytes,
    mode: FaceDetectionMode.full);
for (final face in faces) {
  // null in fast/standard modes, or if the stage could not run.
  final double? smile = face.smilingProbability;
  final double? leftOpen = face.leftEyeOpenProbability;
  final double? rightOpen = face.rightEyeOpenProbability;
  if (smile != null) {
    print('smile $smile  eyesOpen $leftOpen / $rightOpen');
  }

  // All 52 coefficients (a superset of ML Kit), indexed by name:
  final FaceBlendshapes? bs = face.blendshapes;
  if (bs != null) {
    final double jawOpen = bs[Blendshape.jawOpen];
    final double browUp = bs[Blendshape.browInnerUp];
    print('jawOpen $jawOpen  browInnerUp $browUp');
  }
}
```

**Migrating from ML Kit.** ML Kit's `smilingProbability`,
`leftEyeOpenProbability` and `rightEyeOpenProbability` require
`enableClassification` on the `FaceDetectorOptions`. Here there is no separate
flag: run in `FaceDetectionMode.full` and the values are populated (the model
is a 292-input / 52-output MLP that runs in well under a millisecond, so there
is no added cost worth gating). The getters return `null` in `fast`/`standard`
modes, mirroring ML Kit's `null` when classification is disabled.

**Left/right are subject-relative.** As in ML Kit and ARKit,
`leftEyeOpenProbability` refers to the **subject's** left eye, i.e. the eye that
appears on the **right** side of an unmirrored image. This is the opposite eye
from `face.eyes?.leftEye`, which is image-relative. If your app horizontally
flips frames before detection (common for a selfie preview), subject left/right
swap, exactly as they do in ML Kit.

These are model confidences, not geometric measurements, and are independent: a
wink yields one low and one high eye value, and a smile with closed eyes is
representable. Like ML Kit's classifications, they are most reliable on
near-frontal faces (ML Kit only computes them for a Euler Y between -18 and 18
degrees); this model degrades more gracefully but still favors frontal poses.

## Face Mesh

<img src="assets/screenshots/mesh-ex1.png" width="600" alt="Face Mesh">

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

## Face Contours

Named facial contours, mirroring Google ML Kit's `FaceContourType`. Each
contour is an ordered group of mesh points you can connect to draw the outline
of a facial feature. Read them with `face.getContour(type)` (or `face.contours`
for all of them at once):

```dart
final faces = await detector.detectFacesFromBytes(bytes,
    mode: FaceDetectionMode.standard); // any mode that computes a mesh

for (final face in faces) {
  // null in fast mode (no mesh); otherwise the ordered points in pixels.
  final List<Point>? oval = face.getContour(FaceContourType.face);
  if (oval != null) {
    for (int i = 0; i < oval.length - 1; i++) {
      canvas.drawLine(
        Offset(oval[i].x, oval[i].y),
        Offset(oval[i + 1].x, oval[i + 1].y),
        paint,
      );
    }
  }

  // Or grab everything at once (null in fast mode):
  final Map<FaceContourType, List<Point>>? all = face.contours;
}
```

### Available contours

| `FaceContourType` | Points | Feature |
|-------------------|:------:|---------|
| `face` | 36 | Face oval / silhouette |
| `leftEyebrowTop` / `leftEyebrowBottom` | 5 each | Subject's left eyebrow (upper / lower edge) |
| `rightEyebrowTop` / `rightEyebrowBottom` | 5 each | Subject's right eyebrow (upper / lower edge) |
| `leftEye` / `rightEye` | 16 each | Eye outline (full ring) |
| `upperLipTop` / `upperLipBottom` | 11 each | Upper lip (outer edge / inner mouth line) |
| `lowerLipTop` / `lowerLipBottom` | 11 each | Lower lip (inner mouth line / outer edge) |
| `noseBridge` | 6 | Nose bridge, between the eyes down to the tip |
| `noseBottom` | 5 | Nose base across the nostrils |
| `leftCheek` / `rightCheek` | 1 each | Center of each cheek |

Key things to know:

- **Requires a mesh.** Contours are read off the 468-point mesh, so they are
  populated in `FaceDetectionMode.standard` and `full`. In `fast` mode there is
  no mesh, so `getContour` and `contours` return `null` (mirroring ML Kit's
  `null` when contours are disabled).
- **Left/right are subject-relative.** As in ML Kit, `leftEye` is the
  **subject's** left eye, which appears on the **right** of an unmirrored image.
  This is the same convention as `leftEyeOpenProbability`, and the opposite of
  image-relative `eyes?.leftEye`.
- **Derived from MediaPipe, not ML Kit's contour model.** The points come from
  Google MediaPipe's canonical `FACEMESH_*` connection sets, so the semantic
  groups match ML Kit's `FaceContourType`, but the exact point counts and
  ordering follow MediaPipe. The raw index tables are exposed as
  `faceContourMeshIndices` if you want to map or extend them yourself. Because
  the full 468-point mesh is always available via `face.mesh`, you are never
  limited to these named groups.

## Eye Tracking (Iris + Eye Mesh)

The `eyes` property returns comprehensive eye tracking data for both eyes in absolute pixel
coordinates. Only available in FaceDetectionMode.full.

### Iris Detection

<img src="assets/screenshots/iris-detection-ex2.png" width="600" alt="Iris Detection">

Each eye includes an iris center point and 4 contour points outlining the iris boundary.

```dart
final EyePair? eyes = face.eyes;
final Eye? leftEye = eyes?.leftEye;

if (leftEye != null) {
  final irisCenter = leftEye.irisCenter;
  print('Iris center: (${irisCenter.x}, ${irisCenter.y})');

  for (final point in leftEye.irisContour) {
    print('Iris contour: (${point.x}, ${point.y})');
  }
}
```

### Eye Contour

<img src="assets/screenshots/eyecontour-ex1.png" width="600" alt="Eye Contour">

The eyelid contour consists of 15 points outlining the visible eyelid. Connect them using `eyeLandmarkConnections`:

```dart
final List<Point> eyelidOutline = leftEye.contour;

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

### Eye Area Mesh (71-Point)

<img src="assets/screenshots/eyemesh-ex1.png" width="600" alt="Eye Area Mesh">

71 landmarks covering the entire eye region. Note: The facial mesh and eye area mesh are separate.

```dart
final Eye? leftEye = face.eyes?.leftEye;

if (leftEye != null) {
  for (final point in leftEye.mesh) {
    print('Eye mesh point: (${point.x}, ${point.y})');
  }
}
```

## Face Detection Modes

This package supports three detection modes that determine which facial features are detected:

| Mode | Features | Est. Time per Face* |
|------|----------|---------------------|
| **Full** (default) | Bounding boxes, landmarks, 468-point mesh, named contours, eye tracking (iris + 71-point eye mesh), classification (smile / eye-open / 52 blendshapes) | ~8ms                |
| **Standard** | Bounding boxes, landmarks, 468-point mesh, named contours | ~5ms                |
| **Fast** | Bounding boxes, landmarks | ~3.5ms              |

*Measured on Apple Silicon (M-series) with the default CPU (XNNPACK) engine: a single face at 1280x853, mean of 10 runs after warmup. Performance scales with image size and number of faces, and varies with hardware.

### Code Examples

The Face Detection Mode can be set using the `mode` parameter. Defaults to FaceDetectionMode.full.

```dart
// Full mode (default): bounding boxes, 6 basic landmarks + mesh + comprehensive eye tracking
// note: in full mode, landmarks.leftEye and landmarks.rightEye are replaced with
// iris-refined coordinates, providing significantly more accurate eye positions
// compared to the raw detection keypoints used in fast/standard modes.
// use full mode when precise eye tracking (iris center, contour, eyelid shape) is required.
await fd.detectFacesFromBytes(bytes, mode: FaceDetectionMode.full);

// Standard mode: bounding boxes, 6 basic landmarks + mesh. inference time
// is faster than full mode, but slower than fast mode.
await fd.detectFacesFromBytes(bytes, mode: FaceDetectionMode.standard);

// Fast mode: bounding boxes + 6 basic landmarks only. fastest inference
// time of the three modes.
await fd.detectFacesFromBytes(bytes, mode: FaceDetectionMode.fast);
```

Try the [sample code](https://pub.dev/packages/face_detection_tflite/example) from the pub.dev example tab to easily compare
modes and inferences timing.

## Detection Models

This package supports multiple detection models optimized for different use cases:

| Model | Best For | 
|-------|----------|
| **backCamera** (default) | Group shots, distant faces, rear camera | 
| **frontCamera** | Selfies, close-up portraits, front camera | 
| **shortRange** | Close-range faces (within ~2m) |
| **full** | Mid-range faces (within ~5m) |
| **fullSparse** | Mid-range faces with faster inference (~30% speedup) | 

### Code Examples

The model can be set using the `model` parameter on either `FaceDetector.create()` or `initialize()`. Defaults to `FaceDetectionModel.backCamera`.

```dart
// One-step with create()
final detector = await FaceDetector.create(model: FaceDetectionModel.frontCamera);

// Or two-step with initialize(), same options
final detector = FaceDetector();
await detector.initialize(model: FaceDetectionModel.frontCamera);
```

Available models:

```dart
FaceDetectionModel.backCamera    // (default) larger model, group shots, smaller faces
FaceDetectionModel.frontCamera   // selfies, close-up portraits
FaceDetectionModel.shortRange    // short-range images (faces within ~2m)
FaceDetectionModel.full          // mid-range images (faces within ~5m)
FaceDetectionModel.fullSparse    // same quality as full, ~30% faster on CPU
                                 // (slightly higher precision, slightly lower recall)
```

## Live Camera Detection

<img src="assets/screenshots/livecamera_ex1.gif" width="600" alt="Live Camera Detection">

For real-time face detection from a camera feed, use `detectFacesFromCameraImage`. All processing runs off the UI thread.

> **Desktop (Windows / macOS / Linux):** You must also add [`camera_desktop`](https://pub.dev/packages/camera_desktop) to your `pubspec.yaml`, otherwise `startImageStream` throws `UnimplementedError: onStreamedFrameAvailable() is not implemented`.
> ```yaml
> dependencies:
>   camera: ^0.12.0
>   camera_desktop: ^1.2.0   # required for Windows, macOS, and Linux streaming
> ```

```dart
import 'package:camera/camera.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';

final detector = await FaceDetector.create(model: FaceDetectionModel.frontCamera);

final cameras = await availableCameras();
final camera = CameraController(
  cameras.first,
  ResolutionPreset.medium,
  enableAudio: false,
  imageFormatGroup: ImageFormatGroup.yuv420, // prevents JPEG fallback on Android; ignored on desktop
);
await camera.initialize();

camera.startImageStream((CameraImage image) async {
  final faces = await detector.detectFacesFromCameraImage(
    image,
    // rotation: rotationForFrame(...), // recommended on Android/iOS
    mode: FaceDetectionMode.fast,
    maxDim: 640,
  );
  // Process faces...
});
```

Tips:
- Pass `rotation:` on Android/iOS so the detector sees upright frames. Use `rotationForFrame(...)` to compute the correct value from sensor orientation and device orientation. On desktop frames are always upright so omit it.
- Pass `maxDim: 640` to downscale frames before inference. Recommended: full-res frames waste bandwidth since the model input is much smaller.
- Use `FaceDetectionMode.fast` for real-time performance.
- Mirror the overlay on the front camera to match `CameraPreview`'s auto-mirrored texture.
- For segmentation or advanced use, the two-step API is `prepareCameraFrame(...)` + `detectFacesFromCameraFrame(...)` (or the `...WithSegmentationFromCameraFrame` variant).

See the full [example app](https://pub.dev/packages/face_detection_tflite/example) for a complete implementation.

## Background Processing

All inference runs automatically in a background isolate: the UI thread is never blocked during detection, mesh computation, iris tracking, or embedding generation. No special configuration is needed; `FaceDetector` handles isolate management internally.

## Face Recognition (Embeddings) 

Generate 192-dimensional identity vectors to compare faces across images. Useful for identifying the same person in different photos.

Alignment is handled for you: pass the full image plus the detected `Face`, and the library uses the face's eye landmarks to affine-warp a straightened, scaled crop that is fed to the embedding model. You do not need to crop or align the face yourself. This corrects in-plane rotation and scale (not yaw/pitch), and applies to every embedding method (`getFaceEmbedding`, `getFaceEmbeddingFromMat`, `getFaceEmbeddingFromMatBytes`, `getFaceEmbeddings`). The `Face` must be a real detection result with eye landmarks, or the call throws.

```dart
final detector = await FaceDetector.create();

// Full mode gives the most accurate eye alignment for embeddings.
// Standard mode is a good balance; fast mode is fastest but least accurate.
final refFaces = await detector.detectFacesFromBytes(photo1Bytes, mode: FaceDetectionMode.full);
final refEmbedding = await detector.getFaceEmbedding(refFaces.first, photo1Bytes);

// Compare against faces in another photo
final faces = await detector.detectFacesFromBytes(photo2Bytes, mode: FaceDetectionMode.full);
for (final face in faces) {
  final embedding = await detector.getFaceEmbedding(face, photo2Bytes);
  final similarity = FaceDetector.compareFaces(refEmbedding, embedding);
  print('Similarity: ${similarity.toStringAsFixed(2)}'); // -1.0 to 1.0
}

await detector.dispose();
```

**Similarity thresholds:**
- `> 0.6`, Very likely same person
- `> 0.5`, Probably same person
- `< 0.3`, Different people

Also available: `FaceDetector.faceDistance()` for Euclidean distance, and batch processing with `getFaceEmbeddings()`.

For camera streams or when you already have a decoded `cv.Mat`, use `getFaceEmbeddingFromMat()` to avoid re-encoding overhead. If you have raw pixel bytes (e.g. from an image pipeline), use `getFaceEmbeddingFromMatBytes()` for the fastest path.

## Selfie Segmentation

Separate people from backgrounds using MediaPipe Selfie Segmentation. Useful for virtual backgrounds, portrait effects, and background blur.

| Binary | Multiclass (6 Classes) |
|--------|------------------------|
| ![Segmentation Binary](assets/screenshots/segmentation-binary-ex1.png) | ![Segmentation Multiclass](assets/screenshots/segmentation-multiclass-ex1.png) |

### Standalone Usage 

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

final segmenter = await SelfieSegmentation.create();

final mask = await segmenter.callFromBytes(imageBytes);

// mask.width, mask.height: mask dimensions (model resolution)
// mask.at(x, y): probability (0.0-1.0) that pixel is a person

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
// One-step: initialize detection + segmentation together
final detector = await FaceDetector.create(withSegmentation: true);

// Or initialize segmentation separately after creating the detector:
// final detector = await FaceDetector.create();
// await detector.initializeSegmentation();

// Defaults to SegmentationConfig.safe (CPU-only, 1024 max output).
// On iOS/desktop, pass `segmentationConfig: SegmentationConfig.performance` for
// hardware acceleration.

final mask = await detector.getSegmentationMask(imageBytes);
// Use mask for background replacement...

await detector.dispose();
```

#### Faces + mask in one call

If you need both the faces and the mask from a single frame, use
`detectFacesWithSegmentation()`. It runs detection and segmentation together and
returns a `DetectionWithSegmentationResult`:

```dart
final result = await detector.detectFacesWithSegmentation(imageBytes);

final List<Face> faces        = result.faces;
final SegmentationMask? mask  = result.segmentationMask; // null if unavailable

// Per-stage timings (milliseconds)
print('Detection:    ${result.detectionTimeMs}ms');
print('Segmentation: ${result.segmentationTimeMs}ms');
print('Total:        ${result.totalTimeMs}ms');
```

`totalTimeMs` is the **larger** of the two stage times, not their sum, because
detection and segmentation run concurrently.

### Model Variants

| Model | Input Size | Output | Best For |
|-------|------------|--------|----------|
| **general** (default) | 256×256 | Binary | Portraits, square images |
| **landscape** | 144×256 | Binary | Wide images, video streams |
| **multiclass** | 256×256 | 6 classes | Body part segmentation |

```dart
// Use landscape model for video
final videoSegmenter = await SelfieSegmentation.create(
  config: SegmentationConfig(model: SegmentationModel.landscape),
);

// Use multiclass for body part segmentation
final multiclassSegmenter = await SelfieSegmentation.create(
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

## Web (Flutter Web)

This package supports Flutter Web using the same package import:

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';
```

The following methods work on web: `detectFacesFromBytes(bytes)` and `getSegmentationMask(bytes)`. Methods that require native OpenCV or isolate infrastructure (`detectFacesFromFilepath`, `detectFacesFromMat`, `detectFacesFromMatBytes`, `detectFacesFromCameraImage`, `detectFacesFromCameraFrame`, all face embedding methods (`getFaceEmbedding`, `getFaceEmbeddings`, `compareFaces`, `faceDistance`, etc.), and segmentation Mat/camera variants) throw `UnsupportedError` on web.

On web, load image bytes from a file picker, drag-and-drop, or network response and pass them to `detectFacesFromBytes(imageBytes)`:

```dart
final detector = await FaceDetector.create();

final List<Face> faces = await detector.detectFacesFromBytes(imageBytes);

await detector.dispose();
```

### Web (LiteRT.js + WebGPU, default)

LiteRT.js is the default web runtime, no extra configuration needed. It prefers WebGPU and falls back to SIMD-optimized WASM automatically on unsupported browsers:

```dart
final detector = await FaceDetector.create(
  // liteRtAccelerator defaults to 'auto': prefers WebGPU, falls back to WASM.
);
```

`liteRtAccelerator` accepts:

| Value | Behavior |
|---|---|
| `'auto'` (default) | Try WebGPU; if compile fails (no `navigator.gpu`, or unsupported ops) fall back to WASM. |
| `'webgpu'` | Request WebGPU; falls back to WASM if WebGPU compile fails. |
| `'wasm'` | Use SIMD-optimized WASM. Use this to opt out of GPU even when available. |

To opt into the legacy `tflite-js` runtime, pass `useLiteRt: false` to `FaceDetector.create()` or `initialize()`.

If you need to self-host the runtime (offline, strict CSP, or to pin a specific build), call `flutter_litert`'s `configureLiteRtLoader(moduleUrl: ..., wasmUrl: ...)` before any `FaceDetector.create`, or set `autoLoad: false` and load it from your own `<script>` tag instead.

### Separate `example_web` app

The repository keeps the browser demo in `example_web/` (separate from `example/`) because the web sample uses browser-specific APIs (HTML file picker + canvas overlay) and includes a live webcam mode via `getUserMedia`. Copy from <a href="https://github.com/hugocornellier/face_detection_tflite/blob/main/example_web/lib/main.dart" target="_blank">example_web/lib/main.dart</a> as a starting point.

Run the web demo locally:

```bash
cd example_web
flutter pub get
flutter run -d chrome
```

Build for web:

```bash
cd example_web
flutter build web
```

## Performance

### Hardware Acceleration

`FaceDetector` runs on one of two inference engines, selected at init:

- **Interpreter** (default). Classic TFLite. CPU via XNNPACK on every platform. GPU only via the platform delegates below, which are deprecated and platform-limited.
- **CompiledModel** (opt-in: `useCompiledModel: true`). LiteRT Next. Auto-selects GPU/NPU with automatic CPU fallback on every platform, and it is faster on CPU too (parity-checked: roughly 1.4x to 3.5x vs the plain Interpreter, at or above XNNPACK on most models).

| Platform | Interpreter GPU (default engine) | CompiledModel GPU (`useCompiledModel: true`) |
|----------|:---:|:---:|
| Android | ✅ `GpuDelegateV2`* | ✅ |
| iOS / macOS | ✅ Metal* | ✅ |
| **Windows / Linux** | ❌ CPU only (XNNPACK) | ✅ |
| Web | WebGPU via `liteRtAccelerator` | (n/a) |

> \*Interpreter GPU/Metal delegates are deprecated (removed in flutter_litert 4.0.0). **On Windows and Linux, GPU is available only through CompiledModel**, because the Interpreter has no desktop GPU delegate.

```dart
// Default (Interpreter): CPU everywhere; GPU on Android and Apple only.
final detector = await FaceDetector.create();

// CompiledModel: GPU/NPU where available, automatic CPU fallback.
// This is the only GPU path on Windows and Linux.
final detector = await FaceDetector.create(useCompiledModel: true);
```

### Accelerator selection (CompiledModel)

When `useCompiledModel: true`, two optional parameters control the LiteRT Next backend. They have no effect on the default Interpreter engine.

- `accelerators` (`Set<Accelerator>`, default `{Accelerator.gpu, Accelerator.cpu}`). The accelerators the backend may use. The runtime picks the fastest available and falls back through the set. If none initialize it throws, so include `Accelerator.cpu` to guarantee a fallback. The default requests GPU with CPU fallback.
- `precision` (`Precision`, default `Precision.fp16`). Numeric precision for the compiled graph. `Precision.fp32` trades speed for accuracy.

```dart
// CPU only, using CompiledModel's fast CPU runtime.
await FaceDetector.create(
  useCompiledModel: true,
  accelerators: {Accelerator.cpu},
);

// GPU only. Throws if the GPU backend cannot initialize.
await FaceDetector.create(
  useCompiledModel: true,
  accelerators: {Accelerator.gpu},
);

// NPU first, CPU fallback, at fp32 precision.
await FaceDetector.create(
  useCompiledModel: true,
  accelerators: {Accelerator.npu, Accelerator.cpu},
  precision: Precision.fp32,
);
```

`initializeSegmentation()` accepts the same `accelerators` and `precision` parameters. Under CompiledModel, the iris model always runs on CPU for numerical stability, regardless of the `accelerators` you request. `Accelerator` and `Precision` are exported from the package.

### Advanced Performance Configuration

`performanceConfig` tunes the **Interpreter** engine only. It has no effect when `useCompiledModel: true`.

```dart
// Auto mode (default): optimal for each platform
final detector = await FaceDetector.create();

// Force XNNPACK (all native platforms)
final detector = await FaceDetector.create(
  performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
);

// Force the Interpreter GPU delegate (Android and Apple only; deprecated, prefer CompiledModel)
final detector = await FaceDetector.create(
  performanceConfig: PerformanceConfig.gpu(),
);

// CPU-only (maximum compatibility)
final detector = await FaceDetector.create(
  performanceConfig: PerformanceConfig.disabled,
);
```

### Encoded Image Bytes Example

If you already hold the bytes of an encoded image file (JPEG, PNG, etc.), for example from a network response or a file picker, pass them straight to `detectFacesFromBytes()`. The bytes are decoded inside the detection isolate:

```dart
// The bytes ARE a compressed image file: the contents of a .jpg/.png/...
// e.g. a network download, an asset, or a picked file (no path on disk).
final Uint8List imageBytes = await http.readBytes(Uri.parse(imageUrl));

final faces = await detector.detectFacesFromBytes(imageBytes);
```

This is the right choice whenever your source is a compressed image rather than raw pixels. For raw (already-decoded) pixels, use [detectFacesFromMatBytes](#raw-pixel-bytes-example) instead.

### Direct Mat Example

For live camera streams, you can bypass image encoding/decoding entirely by passing a `Mat` directly to `detectFacesFromMat()`:

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

Future<void> processFrame(Mat frame) async {
  final detector = await FaceDetector.create(model: FaceDetectionModel.frontCamera);

  // Direct Mat input: fastest for video streams
  final faces = await detector.detectFacesFromMat(frame, mode: FaceDetectionMode.fast);

  frame.dispose(); // always dispose Mats after use
  await detector.dispose();
}
```

**When to use `Mat` input:**
- You already have a decoded `cv.Mat` from another OpenCV pipeline
- You need to preprocess images with OpenCV before detection

For live camera streams, prefer `detectFacesFromCameraImage(...)`: it keeps all `cvtColor` / `rotate` / downscale work inside the detection isolate rather than on the UI thread.

**For all other cases**, pass image bytes (`Uint8List`) to `detectFacesFromBytes()`.

### Raw Pixel Bytes Example

If you already have raw pixel data as a `Uint8List` (e.g. from an isolate worker or image processing pipeline), use `detectFacesFromMatBytes()` to skip constructing a `cv.Mat` on the calling thread entirely:

```dart
// The bytes are ALREADY-decoded pixels (no file header), e.g. straight from a
// cv.Mat buffer or a worker isolate. Dimensions can't be inferred, so pass them.
final cv.Mat mat = ...;            // some decoded image
final Uint8List rawPixels = mat.data;
final int width = mat.cols;
final int height = mat.rows;

final faces = await detector.detectFacesFromMatBytes(
  rawPixels,
  width: width,
  height: height,
  // matType: 16 (CV_8UC3/BGR) is the default
);
```

This is the fastest path when you already have raw pixel bytes: the data is transferred to the background isolate via zero-copy `TransferableTypedData`, and the `cv.Mat` is reconstructed there instead of on the calling thread.

### Memory Considerations

`FaceDetector` holds all TFLite models (~26-40MB for full pipeline) in a background isolate. Always call `dispose()` when finished to release these resources. Image data is transferred using zero-copy `TransferableTypedData`, minimizing memory overhead.

## MLKit Comparison Benchmark

<a id="mlkit-comparison-benchmark"></a>

The benchmark test ([`example/integration_test/mlkit_benchmark_test.dart`](example/integration_test/mlkit_benchmark_test.dart)) compares `face_detection_tflite` against `google_mlkit_face_detection`, which the example declares as a dev dependency. Because `google_mlkit_face_detection` does not support Swift Package Manager (as of May 2026 it ships a CocoaPods podspec only), the iOS example is built with CocoaPods. Flutter configures this automatically when you build or test the example, so no manual `pod install` step is needed. The published `face_detection_tflite` plugin is pure Dart and uses Swift Package Manager on every platform.

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

## Inspiration

At the time of development, there was no open-source solution for cross-platform, on-device face and landmark detection.
This package took inspiration and was ported from the original Python project **[patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite)**. Many thanks to the original author.
