## 1.0.2
-  Three detection modes: fast, standard & full. Enables faster inferences when the full detection set is not needed.
  - 
-  Performance optimization

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
