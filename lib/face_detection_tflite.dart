/// Face detection and landmark inference utilities backed by MediaPipe-style
/// TFLite models for Flutter apps.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:ui';
import 'package:flutter/services.dart';
import 'package:meta/meta.dart';
import 'package:path/path.dart' as p;
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter_custom/tflite_flutter.dart';

export 'src/dart_registration.dart';
export 'src/image_utils.dart';

// Re-export opencv_dart types for users who want to use detectFacesFromMat directly
export 'package:opencv_dart/opencv_dart.dart' show Mat, imdecode, IMREAD_COLOR;

part 'src/types_and_consts.dart';
part 'src/helpers.dart';
part 'src/isolate_worker.dart';
part 'src/face_detector.dart';
part 'src/face_detector_isolate.dart';
part 'src/face_detection_model.dart';
part 'src/face_landmark.dart';
part 'src/iris_landmark.dart';
part 'src/face_embedding.dart';
