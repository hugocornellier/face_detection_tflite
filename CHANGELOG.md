## Unreleased

**Breaking changes:**
- Removed `compute3DMesh` parameter from `FaceDetector.detectFaces()` - 3D coordinates are now always computed when available (the ~3.75KB memory savings was negligible)
- **Removed redundant mesh2D/mesh3D split**: `meshFromAlignedFace()` now returns `List<Point>` instead of record with `mesh2D` and `mesh3D`. All points now include x, y, and z coordinates.
- **Removed redundant iris2D/iris3D split**: `irisFromEyeRois()` now returns `List<Point>` instead of record. All points include depth information.
- **Simplified Face API**: Removed `Face.irisPoints3D` field - `Face.irisPoints` now contains full 3D data (Point with x, y, z).
- **Renamed method**: `getIris()` → `getEyeMeshWithIris()` to accurately reflect that it returns 152 points (76 per eye: 71 eye mesh + 5 iris keypoints), not just 10 iris points.
- **Deprecated method**: `getIrisFromMesh()` is now deprecated - use `detectFaces()` with `FaceDetectionMode.full` and access `face.eyes` instead. Will be removed in v5.0.0.
- **Updated internal types**: `_DetectionFeatures` now uses simplified `mesh` and `iris` fields (both `List<Point>`) instead of separate 2D/3D variants.
- **Signature changes**: `eyeRoisFromMesh()` now accepts `List<Point>` instead of `List<Offset>`.

**Improvements:**
- **Memory optimization**: Eliminated redundant storage of x/y coordinates (previously stored in both 2D and 3D lists).
- **Cleaner API**: Single mesh/iris representations reduce confusion about which to use.
- **Consistent 3D support**: All landmark data now includes depth information by default.

## 4.0.0

**Breaking changes:**
- Renamed `Iris` class to `Eye` (better represents iris + eye mesh data)
- Renamed `Eye.center` to `Eye.irisCenter` for clarity
- Renamed `Eye.eyeMesh` to `Eye.mesh` and `Eye.eyeContour` to `Eye.contour` for brevity
**- Removed `IrisPair` typedef - use `EyePair` instead
- Removed `Face.irises` property - use `Face.eyes` instead
- Removed `EyePair.leftIris` and `EyePair.rightIris` - use `leftEye` and `rightEye` instead
- Removed `Iris.contour` property - use `Eye.irisContour` instead
- Removed `Iris.eyeballContour` property - use `Eye.contour` instead
- Removed `kEyeLandmarkConnections` constant - use `eyeLandmarkConnections` instead
- Changed `Face.mesh` from `List<Point<double>>` to `FaceMesh` class
- Unified point representation: custom `Point` class replaces `dart:math Point<double>` throughout API
- `Point` class now has optional z coordinate (replaces separate 2D/3D types)
- Removed `FaceMesh.points2D` and `FaceMesh.points3D` - use single `FaceMesh.points` getter
- All landmarks, bounding box corners, eye tracking, and mesh points now use custom `Point` class

**New features:**
- Added 3D mesh support with depth information (z-coordinate)
- `FaceMesh` class for unified 2D and 3D mesh access
- `Point` class with optional z coordinate

**Improvements:**
- Cleaner, more semantic naming throughout the API
- Better distinction between iris contour (4 points) and eye contour (15 points)**

## 3.1.0
- EyePair class and eye mesh landmarks (71 points per eye)
- Add `eyeContour` getter for accessing visible eyelid outline (first 15 of 71 points)
- Add `eyeLandmarkConnections` constant for rendering connected eyelid outline
- Add `kMaxEyeLandmark` constant defining eyeball contour point count
- Rename ImageProcessingWorker to IsolateWorker (old name kept as deprecated alias)

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
