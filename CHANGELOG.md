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
