/// Web implementation of face_detection_tflite.
///
/// Loaded by `face_detection_tflite.dart` via a conditional export when the
/// host has `dart.library.js_interop` (i.e. browsers).
library;

export '../dart_registration.dart';
export '../shared/model_bytes_loader.dart';
export '../shared/release_model_loader.dart'
    show
        ReleaseModelLoader,
        ModelDownloadProgress,
        ModelChecksumException,
        kDefaultModelReleaseBaseUrl,
        kModelSha256Sums;
export 'types.dart';
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
