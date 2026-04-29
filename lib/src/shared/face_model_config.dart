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

/// Asset filename for the MobileFaceNet embedding model.
const String kEmbeddingModel = 'mobilefacenet.tflite';

/// Asset filename for the general (binary) selfie segmentation model.
const String kSegmentationGeneralModel = 'selfie_segmenter.tflite';

/// Asset filename for the landscape (binary) selfie segmentation model.
const String kSegmentationLandscapeModel = 'selfie_segmenter_landscape.tflite';

/// Asset filename for the multiclass selfie segmentation model.
const String kSegmentationMulticlassModel = 'selfie_multiclass.tflite';

/// Raw score limit applied to BlazeFace logits before sigmoid.
const double kRawScoreLimit = 80.0;

/// Minimum sigmoid score for a candidate detection.
const double kMinScore = 0.5;

/// IoU threshold used during weighted NMS.
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
