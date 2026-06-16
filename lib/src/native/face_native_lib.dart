/// Native (non-web) implementation of face_detection_tflite.
///
/// This library aggregates all the implementation parts (FaceDetector class,
/// model runners, isolate workers, UI helpers). It imports + re-exports the
/// shared pure-Dart types so user code sees the same `Face`, `Detection`,
/// `SegmentationMask`, etc. on every platform.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart' hide Detection;

import '../shared/face_types.dart';
import '../shared/face_model_config.dart';

export '../dart_registration.dart';

// Single source of truth for all public types and constants.
export '../shared/face_types.dart';
export '../shared/face_model_config.dart'
    show
        kModelNameBack,
        kModelNameFront,
        kModelNameShort,
        kModelNameFull,
        kModelNameFullSparse,
        kFaceLandmarkModel,
        kIrisLandmarkModel,
        kEmbeddingModel,
        kSegmentationGeneralModel,
        kSegmentationLandscapeModel,
        kSegmentationMulticlassModel;
export '../shared/face_geometry.dart' show eyeRoisFromMesh, faceDetectionToRoi;

export 'package:flutter_litert/flutter_litert.dart'
    show
        Accelerator,
        Precision,
        PerformanceMode,
        PerformanceConfig,
        createNHWCTensor4D,
        fillNHWC4D,
        allocTensorShape,
        flattenDynamicTensor,
        sigmoid,
        sigmoidClipped,
        clamp01,
        clip,
        computeLetterboxParams,
        LetterboxParams,
        bgrBytesToRgbFloat32,
        bgrBytesToSignedFloat32,
        Point,
        BoundingBox,
        packYuv420,
        YuvPlane,
        YuvLayout,
        PackedYuv,
        CameraPlane,
        CameraFrame,
        CameraFrameConversion,
        CameraFrameRotation,
        prepareCameraFrame,
        prepareCameraFrameFromImage,
        rotationForFrame,
        detectionSize,
        coverFitScaleOffset,
        barQuarterTurns,
        FpsCounter,
        drawLandmarkMarker,
        drawSkeletonConnections,
        drawBoundingBoxOutline;

export '../exports/opencv_exports.dart';

part '../types_and_consts.dart';
part '../util/helpers.dart';
part '../face_detector.dart';
part '../isolate/face_detector_core.dart';
part '../models/face_detection_model.dart';
part '../models/face_landmark.dart';
part '../models/iris_landmark.dart';
part '../models/face_embedding.dart';
part '../models/selfie_segmentation.dart';
part '../isolate/segmentation_worker.dart';
part '../ui/overlay_painters.dart';
part '../ui/timing_widgets.dart';
part '../ui/demo_controls.dart';
