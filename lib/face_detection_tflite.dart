/// Face detection and landmark inference utilities backed by MediaPipe-style
/// TFLite models for Flutter apps.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:meta/meta.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';

export 'src/dart_registration.dart';
export 'src/util/image_utils.dart';

export 'package:opencv_dart/opencv_dart.dart' show Mat, imdecode, IMREAD_COLOR;

part 'src/types_and_consts.dart';
part 'src/util/helpers.dart';
part 'src/face_detector.dart';
part 'src/isolate/face_detector_isolate.dart';
part 'src/models/face_detection_model.dart';
part 'src/models/face_landmark.dart';
part 'src/models/iris_landmark.dart';
part 'src/models/face_embedding.dart';
part 'src/models/selfie_segmentation.dart';
part 'src/isolate/segmentation_worker.dart';
