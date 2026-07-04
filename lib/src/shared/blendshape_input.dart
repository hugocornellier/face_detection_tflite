/// Landmark packing for the MediaPipe **Blendshape V2** model, shared by the
/// native and web pipelines so both produce byte-identical model input.
///
/// The blendshape model is not an image model: it consumes a fixed subset of
/// **146 facial landmarks** (x, y in full-image pixel coordinates, no z),
/// selected from MediaPipe's 478-point layout (468 mesh points + 10 iris
/// points), and outputs **52 expression coefficients** in `[0, 1]`.
///
/// This repo already produces every input the model needs, in exactly the
/// right coordinate space:
///
/// * the 468-point face mesh in absolute image pixels (`transformMeshToAbsolute`
///   / `transformMeshFlatToAbsolute`), and
/// * the per-eye iris points in the same absolute space (`irisPoints`).
///
/// So packing is a pure index repack, matching the official
/// `SplitNormalizedLandmarkListCalculator` + `LandmarksToTensorCalculator`
/// chain (which multiplies normalized landmarks by the image size, i.e. yields
/// the same absolute pixels we already hold). No normalization, ROI math, or
/// image work is required.
///
/// All functions here are pure Dart (no `dart:io`, no `cv.Mat`, no platform
/// imports) so they are safe to call from any platform.
library;

import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart' show Point;

/// The 146 landmark indices the Blendshape V2 model consumes, in tensor order.
///
/// Verified verbatim from `face_blendshapes_graph.cc`
/// (`kLandmarksSubsetIdxs`, google-ai-edge/mediapipe master). Indices are into
/// the virtual 478-point layout: `0..467` are mesh points and `468..477` are
/// the 10 iris points (see [packBlendshapeInput] for the iris routing).
///
/// The list is strictly increasing and unique; the final ten entries are the
/// iris slots `468..477`.
const List<int> kBlendshapeLandmarkSubset = <int>[
  0,
  1,
  4,
  5,
  6,
  7,
  8,
  10,
  13,
  14,
  17,
  21,
  33,
  37,
  39,
  40,
  46,
  52,
  53,
  54,
  55, //
  58,
  61,
  63,
  65,
  66,
  67,
  70,
  78,
  80,
  81,
  82,
  84,
  87,
  88,
  91,
  93,
  95,
  103,
  105, //
  107,
  109,
  127,
  132,
  133,
  136,
  144,
  145,
  146,
  148,
  149,
  150,
  152,
  153,
  154,
  155, //
  157,
  158,
  159,
  160,
  161,
  162,
  163,
  168,
  172,
  173,
  176,
  178,
  181,
  185,
  191,
  195, //
  197,
  234,
  246,
  249,
  251,
  263,
  267,
  269,
  270,
  276,
  282,
  283,
  284,
  285,
  288,
  291, //
  293,
  295,
  296,
  297,
  300,
  308,
  310,
  311,
  312,
  314,
  317,
  318,
  321,
  323,
  324,
  332, //
  334,
  336,
  338,
  356,
  361,
  362,
  365,
  373,
  374,
  375,
  377,
  378,
  379,
  380,
  381,
  382, //
  384,
  385,
  386,
  387,
  388,
  389,
  390,
  397,
  398,
  400,
  402,
  405,
  409,
  415,
  454,
  466, //
  468, 469, 470, 471, 472, 473, 474, 475, 476, 477, //
];

/// Number of landmarks fed to the model (`kBlendshapeLandmarkSubset.length`).
const int kBlendshapeLandmarkCount = 146;

/// Length of the packed input tensor: 146 landmarks x 2 (x, y) = 292 floats.
const int kBlendshapeInputFloats = 292;

/// Number of blendshape coefficients the model outputs.
const int kBlendshapeCount = 52;

// --- iris routing within the virtual 478-point layout ---------------------
//
// This repo's `irisPoints` stream carries 76 points per eye (71 eyelid-contour
// points followed by 5 iris keypoints), image-left eye (mesh corners 33/133)
// first, then image-right eye (mesh corners 362/263). The 5 iris keypoints sit
// at offsets 71..75 in the first block and 147..151 in the second. See the
// `_kLeftIrisStart` / `_kRightIrisStart` constants in `face_detector.dart`.
//
// The 478-layout assigns the first eye's iris to slots 468..472 and the second
// eye's iris to slots 473..477 (verified from
// `tensors_to_face_landmarks_with_attention.pbtxt`).

/// First 478-layout slot occupied by iris points; slots below this are mesh.
const int _kIrisSlotStart = 468;

/// Eyelid-ring mesh indices whose positions must come from the **iris model's
/// refined eye contour** instead of the coarse 468-mesh, keyed to the source
/// offset within [packBlendshapeInput]'s `irisPoints` stream.
///
/// The coarse face mesh keeps the eyelids in a canonical open-eye configuration
/// regardless of actual closure (its eye landmarks barely move when the eye
/// shuts), so the Blendshape V2 model — trained on MediaPipe's attention/
/// iris-refined landmarks — reads `eyeBlink*`≈0 and the eye-open getters never
/// register a closed eye. The iris landmark model this repo already runs *does*
/// resolve closure (its eyelid contour collapses vertically when the eye
/// shuts), so we route the 15 eyelid-ring points of each eye from that refined
/// contour, reproducing what MediaPipe's `UpdateFaceLandmarks` step feeds the
/// blendshape model for the eye region.
///
/// The `irisPoints` stream carries 76 points per eye: 71 eye-contour points
/// then 5 iris keypoints. The eyelid ring is the first 15 contour points, in
/// the ring order given by `eyeLandmarkConnections` (offsets `0..14` for the
/// image-left eye, `76..90` for the image-right eye). Mapping offset -> mesh
/// index is fixed by that ring order and is mirror-symmetric across the eyes.
const Map<int, int> kBlendshapeEyeRefineOffsets = <int, int>{
  // image-left eye (mesh corners 33 / 133) <- irisPoints[0..14]
  33: 0, 7: 1, 163: 2, 144: 3, 145: 4, 153: 5, 154: 6, 155: 7, 133: 8, //
  246: 9, 161: 10, 160: 11, 159: 12, 158: 13, 157: 14, //
  // image-right eye (mesh corners 362 / 263) <- irisPoints[76..90]
  263: 76, 249: 77, 390: 78, 373: 79, 374: 80, 380: 81, 381: 82, 382: 83, //
  362: 84, 466: 85, 388: 86, 387: 87, 386: 88, 385: 89, 384: 90, //
};

/// Offset of the image-left eye's 5 iris keypoints within `irisPoints`.
const int _kLeftEyeIrisOffset = 71; // -> 478 slots 468..472

/// Offset of the image-right eye's 5 iris keypoints within `irisPoints`.
const int _kRightEyeIrisOffset = 147; // -> 478 slots 473..477

/// Minimum `meshAbs` length required to pack (mesh must cover indices 0..467).
const int _kMinMeshPoints = 468;

/// Minimum `irisPoints` length required to pack (must cover index 151).
const int _kMinIrisPoints = 152;

/// Resolves a virtual 478-layout slot to a source [Point].
///
/// Slots `0..467` come straight from the mesh; slots `468..472` come from the
/// image-left eye's iris keypoints and `473..477` from the image-right eye's.
Point _resolve478(int slot, List<Point> meshAbs, List<Point> irisPoints) {
  if (slot < _kIrisSlotStart) {
    // Eyelid-ring points come from the iris model's refined contour (which
    // tracks closure); all other mesh points come straight from the mesh.
    final int? refined = kBlendshapeEyeRefineOffsets[slot];
    if (refined != null) return irisPoints[refined];
    return meshAbs[slot];
  }
  final int irisSlot = slot - _kIrisSlotStart; // 0..9
  if (irisSlot < 5) return irisPoints[_kLeftEyeIrisOffset + irisSlot];
  return irisPoints[_kRightEyeIrisOffset + (irisSlot - 5)];
}

/// Packs the 146-landmark input tensor for the Blendshape V2 model.
///
/// [meshAbs] is the 468-point face mesh and [irisPoints] the per-eye iris
/// stream, both in absolute image pixel coordinates. Returns a [Float32List] of
/// length [kBlendshapeInputFloats] holding `x, y` pairs for the 146 subset in
/// tensor order, or `null` when the inputs are too short to pack (fewer than
/// 468 mesh points or 152 iris points), in which case the caller should skip
/// the blendshape stage and leave the face's scores null.
///
/// The iris 5-point internal order is taken directly from `irisPoints`; any
/// per-eye reordering discovered by end-to-end golden validation is a localized
/// change to [_resolve478].
Float32List? packBlendshapeInput(List<Point> meshAbs, List<Point> irisPoints) {
  if (meshAbs.length < _kMinMeshPoints) return null;
  if (irisPoints.length < _kMinIrisPoints) return null;

  final Float32List out = Float32List(kBlendshapeInputFloats);
  int w = 0;
  for (final int slot in kBlendshapeLandmarkSubset) {
    final Point p = _resolve478(slot, meshAbs, irisPoints);
    out[w++] = p.x.toDouble();
    out[w++] = p.y.toDouble();
  }
  return out;
}

/// The 52 MediaPipe Blendshape V2 expression coefficients, in tensor order.
///
/// Names follow the official MediaPipe / Apple ARKit convention, in which
/// "left" and "right" are relative to the **subject**. [label] returns the
/// exact official string for each coefficient (note [neutral] maps to the
/// underscore-prefixed `_neutral`, which has no Dart-identifier equivalent).
///
/// Verified verbatim from `kBlendshapeNames` in the deployed graph, which
/// corrects the model card appendix: `_neutral` is present at index 0 and there
/// is no `tongueOut`.
enum Blendshape {
  neutral('_neutral'),
  browDownLeft('browDownLeft'),
  browDownRight('browDownRight'),
  browInnerUp('browInnerUp'),
  browOuterUpLeft('browOuterUpLeft'),
  browOuterUpRight('browOuterUpRight'),
  cheekPuff('cheekPuff'),
  cheekSquintLeft('cheekSquintLeft'),
  cheekSquintRight('cheekSquintRight'),
  eyeBlinkLeft('eyeBlinkLeft'),
  eyeBlinkRight('eyeBlinkRight'),
  eyeLookDownLeft('eyeLookDownLeft'),
  eyeLookDownRight('eyeLookDownRight'),
  eyeLookInLeft('eyeLookInLeft'),
  eyeLookInRight('eyeLookInRight'),
  eyeLookOutLeft('eyeLookOutLeft'),
  eyeLookOutRight('eyeLookOutRight'),
  eyeLookUpLeft('eyeLookUpLeft'),
  eyeLookUpRight('eyeLookUpRight'),
  eyeSquintLeft('eyeSquintLeft'),
  eyeSquintRight('eyeSquintRight'),
  eyeWideLeft('eyeWideLeft'),
  eyeWideRight('eyeWideRight'),
  jawForward('jawForward'),
  jawLeft('jawLeft'),
  jawOpen('jawOpen'),
  jawRight('jawRight'),
  mouthClose('mouthClose'),
  mouthDimpleLeft('mouthDimpleLeft'),
  mouthDimpleRight('mouthDimpleRight'),
  mouthFrownLeft('mouthFrownLeft'),
  mouthFrownRight('mouthFrownRight'),
  mouthFunnel('mouthFunnel'),
  mouthLeft('mouthLeft'),
  mouthLowerDownLeft('mouthLowerDownLeft'),
  mouthLowerDownRight('mouthLowerDownRight'),
  mouthPressLeft('mouthPressLeft'),
  mouthPressRight('mouthPressRight'),
  mouthPucker('mouthPucker'),
  mouthRight('mouthRight'),
  mouthRollLower('mouthRollLower'),
  mouthRollUpper('mouthRollUpper'),
  mouthShrugLower('mouthShrugLower'),
  mouthShrugUpper('mouthShrugUpper'),
  mouthSmileLeft('mouthSmileLeft'),
  mouthSmileRight('mouthSmileRight'),
  mouthStretchLeft('mouthStretchLeft'),
  mouthStretchRight('mouthStretchRight'),
  mouthUpperUpLeft('mouthUpperUpLeft'),
  mouthUpperUpRight('mouthUpperUpRight'),
  noseSneerLeft('noseSneerLeft'),
  noseSneerRight('noseSneerRight');

  const Blendshape(this.label);

  /// The exact official MediaPipe coefficient name.
  final String label;
}

/// The 52 official blendshape coefficient names, in tensor order.
///
/// Equivalent to `Blendshape.values.map((b) => b.label)`; provided as a flat
/// list for label-map parity with the official
/// `TensorsToClassificationCalculator`.
final List<String> kBlendshapeNames = List<String>.unmodifiable(
  Blendshape.values.map((Blendshape b) => b.label),
);
