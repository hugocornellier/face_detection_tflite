/// Web implementation of face_detection_tflite.
///
/// Loaded by `face_detection_tflite.dart` via a conditional export when the
/// host has `dart.library.js_interop` (i.e. browsers).
library;

export '../dart_registration.dart';
export 'types.dart';
export '../shared/blendshape_input.dart'
    show Blendshape, kBlendshapeNames, kBlendshapeCount;
// Kept in sync with the native entry point so the documented default is
// reachable on every platform, not just native.
export '../shared/face_model_config.dart'
    show kDefaultMinFacePresenceConfidence, kDefaultMaxMissedFrames;
export 'face_detector_web.dart' show FaceDetector, WebDetectTimings;

// Subset of flutter_litert helpers that user code may rely on.
export 'package:flutter_litert/flutter_litert.dart'
    show
        PerformanceMode,
        PerformanceConfig,
        sigmoid,
        sigmoidClipped,
        clamp01,
        clip,
        computeLetterboxParams,
        LetterboxParams,
        Point,
        BoundingBox,
        FpsCounter,
        drawLandmarkMarker,
        drawSkeletonConnections,
        drawBoundingBoxOutline;
