## 6.8.0

* Feature: opt-in temporal face tracking (`FaceDetector.create(enableTracking: true)`) assigns stable `Face.trackingId` values across sequential native and web detections. Motion-aware geometric association preserves IDs when detector ordering changes and across short detector dropouts; `resetTracking()` clears state when switching streams. `maxMissedFrames` (default `kDefaultMaxMissedFrames`, 3) sets how many processed frames a face may go undetected before its ID is retired, counted in frames the detector actually ran rather than wall-clock time, so a camera loop that drops frames while inference is busy can raise it; negative values throw `ArgumentError` before any model loads. Tracking-enabled calls are sequenced in invocation order, and combined detection + segmentation results are tracked too. Tracking is not face recognition; default behavior remains unchanged with null IDs.
* Export `kDefaultMinFacePresenceConfidence` (0.5) from both entry points. It has been referenced from the public dartdoc on `FaceDetector.create()` and `initialize()` since 6.7.0, but was never exported: the native entry listed only the model-name constants from `face_model_config.dart`, and the web entry did not export that file at all. The doc links therefore dangled on pub.dev and callers could not name the default they were being documented about. Additive only; no behaviour change, and the value is unchanged at 0.5. A test now guards the export so it cannot be dropped again.

## 6.7.0

* Face-presence gate (MediaPipe `min_face_presence_confidence`): `FaceDetector.create()` / `initialize()` now accept `minFacePresenceConfidence`, which drops detections the face-landmark model does not confirm as a face by gating the mesh "face flag" (`face.meshScore`). This is MediaPipe's standard second-stage check and suppresses common first-stage false positives such as a hand or palm, which clear the BlazeFace detector but score near zero on the mesh model. **It defaults to `0.5`, matching MediaPipe** (unlike `minScore`/`minFaceSize`, which default to `0.0`), so upgrading turns the check on: in `standard`/`full` modes, detections whose `meshScore` is below `0.5` are no longer returned. Pass `minFacePresenceConfidence: 0.0` to restore the previous "return every detected box" behavior. The gate has no effect in `fast` mode (no mesh is computed), and a `null` `meshScore` always passes. On both native and web it runs right after the mesh stage, before iris and blendshape, so rejected faces skip that per-face landmark cost. Validated to `[0.0, 1.0]` (out-of-range or NaN throws `ArgumentError`). See the README "Detection Gates" section.
* Match MediaPipe's `score_clipping_thresh` exactly: the BlazeFace raw-logit clip limit (`kRawScoreLimit`) is now `100.0` (was `80.0`), matching the upstream `TensorsToDetectionsCalculator`. This is numerically inert (`sigmoid(80)` and `sigmoid(100)` are both `1.0` in float32), so detector scores and which faces are returned are unchanged; the constant is aligned purely for exactness.
* Fix (web): `activeAccelerator` chained the model runners with `??`, but every
  runner reports a non-null backend once initialized, so the chain always
  short-circuited on the detector model and ignored the other four. Runners
  compile independently and can fall back from WebGPU to WASM on their own, so
  when the detector is the one that falls back the aggregate reported `wasm`
  while other runners were still on the GPU. Both the runtime GPU-error
  fallback and the slow-WebGPU warmup are gated on that value, so neither would
  fire for the runners still on WebGPU. Now uses `aggregateActiveAccelerator`
  from `flutter_litert`, which reports `webgpu` if any runner is on it. A mixed
  state was observed live on Chrome (blendshapes on WASM, the rest on WebGPU).
* Add `FaceDetector.acceleratorReport`, a per-runner map of which backend each
  model actually compiled to, for diagnosing mixed WebGPU/WASM outcomes.
* Adopt the shared `flutter_litert` 3.6.0 helpers in place of local copies:
  `compiledModelFromBufferAuto` for the `{gpu, cpu}` accelerator branch,
  `compiledFloatCount` / `compiledSquareInputSide` for compiled tensor IO at 13
  call sites, and `collectOutputShapes` for output shape collection. The local
  compiled-IO helpers returned a zero or negative element count for a
  degenerate tensor where the shared ones throw; a test asserts every bundled
  model reports positive float32-aligned tensor sizes, the domain where the two
  agree, so no model shipped here changes behaviour.
* Deprecate `OutputTensorInfo`, `collectOutputTensorInfo` and
  `testCollectOutputTensorInfo`. Both call sites only ever read the shapes and
  discarded the buffers; use `collectOutputShapes` from `flutter_litert`.
* Remove the `web_image_utils` re-export shim and import from `flutter_litert`
  directly.
* Update flutter_litert -> 3.6.0.
* Expand the README live camera section with the full production pipeline
  (frame throttling, orientation handling, cover-fit overlay mapping).

## 6.6.3

* Update flutter_litert -> 3.5.1.
* Performance (web): the `minScore`/`minFaceSize` gates now filter detections before the per-face mesh, iris and blendshape stages (as on native), and `detectFacesWithSegmentation` decodes the image once instead of twice. In two interleaved 60-run A/B pairs on Chrome 149 (WASM), using a warmed threshold that retained exactly one face from a 4-face group shot, full mode dropped from 65.8-66.8 ms to 46.1-46.6 ms per call (about 30% faster) and combined detection plus segmentation from 96.8-97.7 ms to 69.2-70.7 ms (about 28% faster); ungated detection was unchanged. Detector-level outputs (scores, boxes, keypoints, which faces are returned) are bit-identical; mesh-stage values for early invocations can shift within the web runtime's pre-existing call-order jitter, which is smaller than the jitter that runtime already shows between identical calls in unchanged code.
* Fix (web): a BlazeFace candidate whose decoded box was degenerate (nonpositive width or height) shifted every later candidate's box onto the wrong confidence score, corrupting weighted NMS and the reported `face.score`/`minScore` gating for those detections. Candidate decode now keeps each box paired with its own score (`decodeBlazeFaceCandidates`, pure Dart and unit-tested), and NaN scores remain rejected; results are unchanged whenever no candidate was skipped, which is the common case. Native was not affected. Because the fix can change web detection output for affected inputs, `FaceDetector.modelVersion` is now `1.1.1`.
* Performance: face meshes returned by the native pipeline now keep their landmark data in the packed float buffer that crossed the isolate boundary and build `Point` objects lazily on first access (`FaceMesh.packed`). Callers that never read `mesh.points` (for example apps that only draw bounding boxes from full-mode results) skip 468 allocations per face per frame; callers that do read them get bit-identical values. Measured ~1.4% faster multi-face full-mode detection and ~3.5% faster adjacent embedding calls (less GC churn), pooled over 200 runs per side; single-face detection within noise; memory usage is equal or lower.
* Performance: embedding requests (`getFaceEmbedding`, `getFaceEmbeddingFromMat`, `getFaceEmbeddingFromMatBytes`, `getFaceEmbeddings`) now send only the two eye landmark points to the detection isolate instead of serializing the whole `Face` (468-point mesh and iris data included), and embedding vectors return as typed data instead of boxed lists. The eye points are exactly what `face.landmarks` reports (iris-refined when available), so embeddings are bit-identical. Measured ~4% faster per `getFaceEmbedding` call (3.41 ms to 3.28 ms median over 100 runs, Apple Silicon, XNNPACK).
* Performance: on native platforms, `minScore`/`minFaceSize` gates now run inside the detection isolate right after the detector stage, so gated-out faces skip the per-face mesh, iris and blendshape work instead of being computed and then discarded. Detection results are byte-identical to the previous late filtering; only the wasted per-face stages are skipped. In a benchmark on a 4-face group shot with a `minFaceSize` keeping one face (full mode, Apple Silicon, XNNPACK, median of 100 runs), latency dropped from ~18.0 ms to ~7.0 ms per call (about 61% faster). Ungated calls are unchanged. Adds the shared `boxVisibleWidthFraction` and `applyDetectionGates` helpers; `Face.widthFraction` now delegates to the former with bit-identical arithmetic.

## 6.6.2

* Update flutter_litert -> 3.5.0

## 6.6.1

* Update flutter_litert -> 3.4.1 (web `CompiledModel` WebGPU compile watchdog: a compile attempt that never settles now falls back to WASM instead of hanging). No API change.

## 6.6.0

* Update flutter_litert -> 3.3.1 (Android Gradle Plugin 9.x build fix; faster `Interpreter.run`/`CompiledModel.run` and fewer per-frame allocations in the camera YUV path). No API change.
* Head pose: `Face.headEulerAngles` (and `headEulerAngleX`/`headEulerAngleY`/`headEulerAngleZ`) report pitch, yaw and roll in degrees, following Google ML Kit's sign conventions. Pitch/yaw come from the 3D mesh (`standard`/`full`); `fast` mode gives roll only. Computed on demand, so no added inference cost.
* Face classification (MediaPipe Blendshape V2, `full` mode): `Face.smilingProbability`, `leftEyeOpenProbability` and `rightEyeOpenProbability` (ML Kit semantics, subject-relative left/right), plus `blendshapes` with all 52 coefficients indexed by the new `Blendshape` enum. Bundles `face_blendshapes.tflite` (955 KB, Apache 2.0); it is a CPU-pinned MLP validated against MediaPipe's golden output, with no cost in `fast`/`standard` (values are `null` there). See the README "Face Classification" section.
* Named face contours (Google ML Kit `FaceContourType` parity): `Face.getContour(type)` and `Face.contours` return ordered mesh points for the face oval, eyebrows, eyes, lips, nose and cheeks, derived from MediaPipe's canonical `FACEMESH_*` connection sets and exposed via the new `FaceContourType` enum and `faceContourMeshIndices` table. Requires a mesh (`standard`/`full`; `null` in `fast`); left/right are subject-relative. See the README "Face Contours" section.
* Detection gates: `FaceDetector.create()` / `initialize()` accept `minScore` and `minFaceSize` (matching Google ML Kit's `setMinFaceSize` convention), both defaulting to `0.0` (no filtering) and validated to `[0.0, 1.0]` (out-of-range or NaN throws `ArgumentError`). Adds `Face.widthFraction` (visible face width / image width), the value `minFaceSize` compares against. `minScore` only tightens results above the detector's internal `0.5` floor. See the README "Detection Gates" section.
* Expose confidence scores: `Face.score` (detector face-presence confidence) and `Face.meshScore` / `FaceMesh.score` (mesh model's confidence, `null` in `fast`), plus `FaceLandmark.callWithScore()`. All from existing outputs, so no added cost. See the README "Detection Score" section.
* Fix: face mesh `z` is now scaled consistently with `x`/`y` (previously left in the model's input-pixel units), making the mesh usable for 3D geometry such as head pose. `x`/`y` rendering and iris landmarks are unaffected.
* Overlay helpers (`DetectionsPainter`, `CameraDetectionPainter`, `FaceDetectionCameraOverlay`) gain an opt-in `showPoseAndScores` flag (default false) drawing a per-face card with confidence and head-pose angles, with toggles in the example app.
* Docs: documented all public enum values and added README sections for the above plus `detectFacesWithSegmentation` / `DetectionWithSegmentationResult`.

## 6.5.0

* Update flutter_litert -> 3.2.0
* Import native-only flutter_litert APIs via `package:flutter_litert/native.dart` so they resolve under static analysis (flutter_litert 3.2.0 moved `InterpreterFactory`, `IsolateRpcClient`, `IsolateWorkerBase`, and `TensorFloat32Views` behind the native conditional export). No runtime or API change.
* Default the public entry's conditional export to the web implementation, gating native behind `dart.library.io`, restoring WASM compatibility (pub.dev WASM-ready). No behavior change on any platform.
* Add `package:face_detection_tflite/face_detection_tflite_native.dart`, a native-only entry point that re-exports the native implementation (isolate workers, model runners, overlay and UI helpers) for code that runs only on native platforms.

## 6.4.1

* Performance: when `getFaceEmbedding` follows `detectFacesFromBytes` on the same encoded image, the detection isolate now reuses the already-decoded image instead of decoding it a second time (one-entry cache keyed by an exact byte match). Saves a full image decode per detect+embed pair (~16 ms at 12 MP; scales with resolution). No API change, and detection and embedding results are byte-identical. The raw-pixel APIs (`detectFacesFromMatBytes`, `getFaceEmbeddingFromMatBytes`) are unaffected; the cache holds at most one decoded frame and is released on dispose.

## 6.4.0

* Update flutter_litert -> 3.1.1
* Add optional LiteRT Next `CompiledModel` inference via `CompiledModel.fromBufferWithGpuFallback` (GPU with automatic CPU fallback); enable with `useCompiledModel: true`. The default engine remains the Interpreter, so existing code is unchanged.
* Decode camera frames through the shared flutter_litert `CameraFrameDecodePlan` helper.

## 6.3.1

* Update flutter_litert -> 2.8.3

## 6.3.0

* Rename `detectFaces` -> `detectFacesFromBytes` for clarity (input is encoded image bytes, vs. raw pixels in `detectFacesFromMatBytes`); `detectFaces` is kept as a deprecated alias and will be removed in a future release
* Update flutter_litert -> 2.8.0
* Complete Swift Package Manager migration: example uses CocoaPods only for the optional MLKit comparison benchmark

## 6.2.9

* Remove unused Darwin podspecs for Dart-only iOS/macOS plugin registration.

## 6.2.8

* Update flutter_litert -> 2.5.8
* Migrate macOS to Swift Package Manager (CocoaPods no longer required)
* Update camera_desktop -> 1.1.6 in example

## 6.2.7

* Update flutter_litert -> 2.5.5

## 6.2.6

* Update flutter_litert to 2.5.3 and camera_desktop to 1.1.4

## 6.2.5

* Add Web mode GPU fallback
* Add video file processing mode to example
* Update flutter_litert -> 2.5.2

## 6.2.4

* Update flutter_litert -> 2.5.0

## 6.2.3

* Update flutter_litert -> 2.4.1

## 6.2.2

* Simplify and DRY example app, extract utility helpers

## 6.2.1

* Update documentation

## 6.2.0

* Update flutter_litert to 2.4.0

## 6.1.1

* Update flutter_litert to 2.2.1

## 6.1.0

* Re-export `packYuv420`, `YuvPlane`, `YuvLayout`, and `PackedYuv` from `flutter_litert` so live-camera consumers can reach the helper through the `face_detection_tflite` barrel without a direct `flutter_litert` import.
* Update `flutter_litert` to `^2.2.0`
* Add `FaceDetector.modelVersion` constant so consumers that persist detection results have a stable cache-invalidation key. Bumped on changes that alter detection output (model swaps, threshold or preprocessing changes); unchanged across pure refactors or API additions.
* Rewrite the README's Live Camera and Direct Mat Input sections around `packYuv420` so every snippet is a real compilable example (no ghost `convertCameraImageToMat`, no duplicate `segmenter` variable, no `cv.Mat` type annotations requiring an unlisted import).

## 6.0.1

* Fix Android live camera overlay in the example app ([#8](https://github.com/hugocornellier/face_detection_tflite/issues/8)):
  * Replace the per-pixel Dart YUV→BGR loop with `flutter_litert`'s shared `packYuv420` helper + native `cv.cvtColor` (~5-10x faster on Android).
  * Correct the rotation direction for `sensorOrientation` 90° / 270° to match the Android CameraX convention.
  * Mirror the detection overlay on Android front camera to match `CameraPreview`'s auto-mirrored preview texture.
* Add option to toggle between front/back camera in example app (mobile only)

## 6.0.0

* Remove `FaceDetectorIsolate` - `FaceDetector` is now the single unified class running all inference in a background isolate
* Remove `irisOkCount` and `irisFailCount` (were deprecated in 5.1.0)
* `FaceDetector()` constructor is now public; `initialize()` replaces the old `spawn()` factory
* `initialize()` gains `withSegmentation` and `segmentationConfig` parameters
* `initializeSegmentation()` no longer requires re-spawning the detection isolate
* Add `getFaceEmbeddingFromMatBytes` to mirror `detectFacesFromMatBytes` for callers with pre-decoded pixel data
* Improve `getFaceEmbeddingFromMat` performance by transferring raw pixel bytes to the background isolate instead of re-encoding

## 5.1.4

* Update flutter_litert to 2.0.13

## 5.1.3

* Update flutter_litert -> 2.0.12

## 5.1.2 

* Update `detectFacesFromMatBytes` documentation 

## 5.1.1

* Add `detectFacesFromMatBytes` to `FaceDetector`: detects faces from raw pixel data without constructing a `cv.Mat` on the calling thread (zero-copy transfer via `TransferableTypedData`)

## 5.1.0

* `FaceDetector` now runs all inference in a background isolate automatically, matching `FaceDetectorIsolate` performance
* `dispose()` is now `Future<void>` (was `void`), existing code compiles but should be awaited
* Deprecate `FaceDetectorIsolate`: use `FaceDetector` instead
* Deprecate `irisOkCount` and `irisFailCount` (not trackable across isolate boundaries)
* Add `detectFacesWithSegmentation` and `detectFacesWithSegmentationFromMat` to `FaceDetector`

## 5.0.13

* Update flutter_litert 2.0.10 -> 2.0.11

## 5.0.12

* Update documentation

## 5.0.11

* Update flutter_litert 2.0.8 -> 2.0.10

## 5.0.10

* Add Windows XNNPack delegate support (2-5x inference speedup)
* Update flutter_litert 2.0.6 -> 2.0.8

## 5.0.9

* Update flutter_litert 2.0.5 -> 2.0.6 

## 5.0.8

* Fix Xcode build warnings by declaring PrivacyInfo.xcprivacy as a resource bundle in iOS and macOS podspecs
  
## 5.0.7

- Update `camera_desktop` 1.0.1 -> 1.0.3
- Use shared `Point` and `BoundingBox` from `flutter_litert` 2.0.0
- Refactor isolate worker to use `IsolateWorkerBase` from flutter_litert
- Consolidate NMS helpers, extract shared `_buildPersonMask` and `_irisCenterFromPoints`
- Deduplicate `FaceDetector` and `FaceDetectorIsolate` internals

## 5.0.6

- Update `flutter_litert` -> 1.2.0
- Refactor to use `flutter_litert` shared utilities (`InterpreterFactory`, `PerformanceConfig`, `generateAnchors`)

## 5.0.5

- Update `opencv_dart` 2.1.0 -> 2.2.1
- Update `flutter_litert` 1.0.2 -> 1.0.3

## 5.0.4

- Update `flutter_litert` 1.0.1 -> 1.0.2

## 5.0.3

- Update `camera` 0.11.3 -> 0.12.0
- Update `flutter_litert` 0.2.2 -> 1.0.1

## 5.0.2

- Update `flutter_litert` to 0.2.2 
- Add original model cards for archival and documentation

## 5.0.1

- Migrate iOS CocoaPods -> Swift Package Manager

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
