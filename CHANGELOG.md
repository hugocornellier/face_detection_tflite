## 5.0.0

**Breaking changes:**

- Remove all deprecated `image` package-based APIs across `FaceDetector`, `FaceDetectorIsolate`, `IsolateWorker`, model runners (`FaceDetectionModel`, `FaceLandmark`, `FaceEmbedding`, `IrisLandmark`, `SelfieSegmentation`), and helper functions
- Remove `image` package dependency

## 4.6.4
- Update flutter_litert to 0.1.12

## 4.6.3
- Swift Package Manager support
- Windows: remove bundled .dll files, as they are no longer needed as of `flutter_litert` 0.1.4

## 4.6.2
- Windows: Custom ops (segmentation) fix
- Fix heap corruption crash when switching between segmentation models

## 4.6.1
- Migrate from `tflite_flutter_custom` to `flutter_litert` 

## 4.6.0
- Fix FaceDetectorIsolate hang on Android during batch face embeddings
- 3-4x performance improvement for FaceDetectorIsolate by eliminating redundant nested isolates
- Models inside worker isolates now invoke TFLite directly instead of routing through nested IsolateInterpreters

## 4.5.3
- Fix Android build: bump tflite_flutter_custom to 1.2.5 (fixes undefined symbol TfLiteIntArrayCreate linker error)

## 4.5.2
- Fix bug causing auto-bundling to fail on MacOS

## 4.5.1
- Update all dependencies to latest version(s)

## 4.5.0
- Selfie segmentation for background removal and virtual backgrounds
- Uses MediaPipe Selfie Segmentation models (general 256×256, landscape 144×256)

## 4.4.1
- Performance optimizations: pre-allocated inference buffers, early score filtering (~17× fewer box decodes), parallel multi-face processing

## 4.4.0
- Fixes #3: bug causing crash on non-XNNPack compatible Android devices

## 4.3.0
- Face recognition via embeddings, enables comparing faces across images
  - `getFaceEmbedding()` / `getFaceEmbeddings()` methods on `FaceDetector` and `FaceDetectorIsolate`
  - `compareFaces()` for cosine similarity, `faceDistance()` for Euclidean distance
  - Uses MobileFaceNet model (~5MB, ~18ms inference)

## 4.2.1
- Fix crash on Windows platforms

## 4.2.0
- Add FaceDetectorIsolate for background thread detection

## 4.1.1
- Re-implement parallel iris inference using native image processing

## 4.1.0
- Native image processing with opencv_dart for ~2x performance improvement via SIMD acceleration
  - `detectFaces()` now uses OpenCV internally
  - New `detectFacesFromMat()` method for camera streams (avoids repeated encode/decode overhead)
- XNNPACK delegate enabled by default for 2-5x CPU speedup (use `PerformanceConfig.disabled` to opt out)
- Benchmark tests

## 4.0.0

**Breaking changes:**

- Replace `math.Point<double>` type references with `Point`
- Change `face.mesh.isEmpty` to `face.mesh == null`
- Access mesh points via `face.mesh?.points[i]` or `face.mesh?[i]`
- Replace `face.irises` → `face.eyes`
- Replace `IrisPair` → `EyePair`
- Replace `iris.center` → `eye.irisCenter`
- Replace `iris.contour` → `eye.irisContour`

**Improvements:**

- Performance and speed improvements 
  - Optimize bilinear sampling with direct buffer access, 20-40% speed improvement
  - Fast-path frame registration
  - Parallel iris refinement 
  - Isolate-based image-to-tensor conversion. 
- Improved test suite, added integration tests

## 3.1.0
- EyePair class and eye mesh landmarks (71 points per eye)
- Add `contour` getter for accessing visible eyelid outline (first 15 of 71 points)
- Add `eyeLandmarkConnections` constant for rendering connected eyelid outline
- Add `kMaxEyeLandmark` constant defining eyeball contour point count

## 3.0.3
- Guard iris ROI size and fall back when eye crop collapses

## 3.0.2
- Add frame registration fast path to reduce transfers 
- Parallel iris refinement for multi-face
- Cache input tensor buffers in FaceDetection, FaceLandmark, and IrisLandmark

## 3.0.1
- Performance improvement: Optimize full mode by reusing mesh and iris landmarks
- Add pub.dev score and version to README

## 3.0.0
**This version contains breaking changes.**
- Remove deprecated bboxCorners and landmarksMap. The new BoundingBox and 
  FaceLandmarks class should be used instead.
- Rename bbox to boundingBox.

## 2.2.1
- FaceLandmarks class instead of Map
- Simplified process of accessing individual landmarks 

## 2.2.0
- BoundingBox class
- Add clarification to README about dart:math requirement for Point

## 2.1.3
- Bundle Point from dart:math with library
- Improved dartdocs 

## 2.1.2
- Fix bug in example related to overlay rect scaling

## 2.1.1
- Added missing dartdoc for entry point lib/face_detection_tflite.dart
  and RectF constructor in lib/src/types_and_consts.dart

## 2.1.0
- New IrisPair class
- Add structured iris types, rename raw iris points
- Update example, README usage examples

## 2.0.3
- Add Dartdocs for AlignedFace, AlignedRoi, DecodedBox, DecodedRgb
- Minor clarifications to existing getDetectionsWithIrisCenters and RectF Dartdocs
- Improved examples in README

## 2.0.2
- Swift Package Manager support

## 2.0.1
- Update tflite_flutter_custom to 1.0.3, equivalent to tflite_flutter 0.12.1.
- Improved dartdocs

## 2.0.0
**This version contains breaking changes.**
- detectFaces now returns List<Face>, PipelineResult/FaceResult removed.
- Public types renamed/privatized (FaceIndex is now FaceLandmarkType, RectF/Detection/
  AlignedFace/AlignedRoi now internal).
- Landmark maps now keyed by FaceLandmarkType
- Full dartdoc coverage

## 1.0.3
-  Update tflite_flutter_custom to 1.0.1, equivalent to tflite_flutter 0.12.0.
-  Unit tests
-  Performance optimization(s) by enabling parallel inferences in images with multiple faces

## 1.0.2
-  Three detection modes: fast, standard & full. Enables faster inferences when the full detection set is not needed.

## 1.0.1
-  Improved error handling 
-  Added samples, improved documentation 
-  Improved example (see example tab on pub.dev)

## 1.0.0
-  Provide end-user with pre-normalized, image-space coords. 
-  Improved readme/documentation & public API as a whole,
-  Removed obsolete methods, change Offset objects to Point.

## 0.1.6
-  Fix bug where IrisLandmark inferences would fail in an Isolate

## 0.1.5
-  Moved heavy operations to Isolates to avoid UI clank/lag

## 0.1.4
-  Refresh iOS/Android example project files to avoid stale tool warnings.

## 0.1.3
- Tweak analysis/lints config to match latest Flutter stable.

## 0.1.2
- Minor bug fixes & improvements
- Clarifications in the README

## 0.1.1
- Add iOS and Android via `dartPluginClass`
- Keep native plugin on desktop (macOS/Windows/Linux) so CMake still bundles TFLite C libs.
- Note: iOS release builds may require Xcode “Strip Style = Non-Global Symbols”; test on device (not simulator).
- Note: Android requires minSdk 26 (handled by the app).

## 0.1.0+1
- Initial public release of `face_detection_tflite`.
- Includes TFLite face detection + landmarks models and platform shims.
- Adds prebuilt `libtensorflowlite_c` for macOS/Windows/Linux.
