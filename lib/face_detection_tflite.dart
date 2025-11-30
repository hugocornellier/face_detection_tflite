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
import 'package:path/path.dart' as p;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter_custom/tflite_flutter.dart';

export 'src/dart_registration.dart';

part 'src/types_and_consts.dart';
part 'src/helpers.dart';
part 'src/image_processing_worker.dart';
part 'src/face_detector.dart';
part 'src/face_detection_model.dart';
part 'src/face_landmark.dart';
part 'src/iris_landmark.dart';
