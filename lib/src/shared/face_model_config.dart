/// Shared model file names, SSD anchor configs, score thresholds, and the
/// model→file mapping helpers used by both the native and web pipelines.
library;

import 'package:flutter_litert/flutter_litert.dart' show SSDAnchorOptions;

import 'face_types.dart' show FaceDetectionModel, SegmentationModel;

/// Asset filename for the back-camera BlazeFace model.
const String kModelNameBack = 'face_detection_back.tflite';

/// Asset filename for the front-camera BlazeFace model.
const String kModelNameFront = 'face_detection_front.tflite';

/// Asset filename for the short-range BlazeFace model.
const String kModelNameShort = 'face_detection_short_range.tflite';

/// Asset filename for the full-range BlazeFace model.
const String kModelNameFull = 'face_detection_full_range.tflite';

/// Asset filename for the full-range sparse BlazeFace model.
const String kModelNameFullSparse = 'face_detection_full_range_sparse.tflite';

/// Asset filename for the 468pt face mesh model.
const String kFaceLandmarkModel = 'face_landmark.tflite';

/// Asset filename for the iris landmark model.
const String kIrisLandmarkModel = 'iris_landmark.tflite';

/// Asset filename for the MediaPipe Blendshape V2 face classification model.
const String kFaceBlendshapesModel = 'face_blendshapes.tflite';

/// Asset filename for the MobileFaceNet embedding model.
const String kEmbeddingModel = 'mobilefacenet.tflite';

/// Asset filename for the general (binary) selfie segmentation model.
const String kSegmentationGeneralModel = 'selfie_segmenter.tflite';

/// Asset filename for the landscape (binary) selfie segmentation model.
const String kSegmentationLandscapeModel = 'selfie_segmenter_landscape.tflite';

/// Asset filename for the multiclass selfie segmentation model.
const String kSegmentationMulticlassModel = 'selfie_multiclass.tflite';

/// Raw score limit applied to BlazeFace logits before sigmoid, matching
/// MediaPipe's `TensorsToDetectionsCalculatorOptions.score_clipping_thresh`
/// (100.0). Numerically inert at this magnitude (`sigmoid(100)` is already 1.0
/// in float32), but kept equal to the upstream graph value for exactness.
const double kRawScoreLimit = 100.0;

/// Minimum sigmoid score for a candidate detection. Matches MediaPipe's
/// `min_detection_confidence` (0.5).
const double kMinScore = 0.5;

/// Default minimum face-presence confidence: the mesh model's "face flag"
/// output (see the MediaPipe Face Mesh model card) a detected face must report
/// to be kept. Matches MediaPipe's `min_face_presence_confidence` (0.5), which
/// is the standard second-stage gate that rejects first-stage detections (e.g.
/// a palm) that the landmark model does not confirm as a face. Gates
/// `Face.meshScore`; only meaningful in `standard`/`full` modes where a mesh
/// (and thus a presence score) is computed. Pass 0.0 to disable the gate.
const double kDefaultMinFacePresenceConfidence = 0.5;

/// Default number of processed frames a tracked face may go undetected before
/// its `Face.trackingId` is retired.
///
/// Counted in frames the detector actually processed, not wall-clock time or
/// camera frames the caller skipped: a frame dropped before it reaches the
/// detector never ages a track. Raise this when frames are processed far
/// apart (heavy modes, or a busy-frame-dropping camera loop), so a face is not
/// given a new ID after a brief occlusion. Only applies when tracking is
/// enabled.
const int kDefaultMaxMissedFrames = 3;

/// IoU threshold used during weighted NMS. Matches MediaPipe's
/// `min_suppression_threshold` (0.3).
const double kMinSuppressionThreshold = 0.3;

/// SSD anchor options for the BlazeFace front-camera model.
const SSDAnchorOptions kSsdFront = SSDAnchorOptions(
  numLayers: 4,
  minScale: 0.1464,
  maxScale: 0.9,
  inputSizeHeight: 128,
  inputSizeWidth: 128,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [8, 16, 16, 16],
  aspectRatios: [1.0],
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 1.0,
  fixedAnchorSize: true,
);

/// SSD anchor options for the BlazeFace back-camera model.
const SSDAnchorOptions kSsdBack = SSDAnchorOptions(
  numLayers: 4,
  minScale: 0.1464,
  maxScale: 0.9,
  inputSizeHeight: 256,
  inputSizeWidth: 256,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [16, 32, 32, 32],
  aspectRatios: [1.0],
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 1.0,
  fixedAnchorSize: true,
);

/// SSD anchor options for the BlazeFace full-range model.
const SSDAnchorOptions kSsdFull = SSDAnchorOptions(
  numLayers: 1,
  minScale: 0.1171875,
  maxScale: 0.75,
  inputSizeHeight: 192,
  inputSizeWidth: 192,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [4],
  aspectRatios: [1.0],
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 0.0,
  fixedAnchorSize: false,
);

/// Returns the SSD anchor options for the given face detection model variant.
SSDAnchorOptions ssdOptionsFor(FaceDetectionModel m) => switch (m) {
  FaceDetectionModel.frontCamera => kSsdFront,
  FaceDetectionModel.backCamera => kSsdBack,
  FaceDetectionModel.shortRange => kSsdFront,
  FaceDetectionModel.full => kSsdFull,
  FaceDetectionModel.fullSparse => kSsdFull,
};

/// Returns the asset filename for the given face detection model variant.
String faceDetectionModelFile(FaceDetectionModel m) => switch (m) {
  FaceDetectionModel.frontCamera => kModelNameFront,
  FaceDetectionModel.backCamera => kModelNameBack,
  FaceDetectionModel.shortRange => kModelNameShort,
  FaceDetectionModel.full => kModelNameFull,
  FaceDetectionModel.fullSparse => kModelNameFullSparse,
};

/// Returns the asset filename for the given segmentation model variant.
String segmentationModelFile(SegmentationModel m) => switch (m) {
  SegmentationModel.general => kSegmentationGeneralModel,
  SegmentationModel.landscape => kSegmentationLandscapeModel,
  SegmentationModel.multiclass => kSegmentationMulticlassModel,
};
