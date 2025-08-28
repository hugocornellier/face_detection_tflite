part of 'face_core.dart';

Float32List _ssdGenerateAnchors(Map<String, Object> opts) {
  final numLayers = opts['num_layers'] as int;
  final strides = (opts['strides'] as List).cast<int>();
  final inputH = opts['input_size_height'] as int;
  final inputW = opts['input_size_width'] as int;
  final ax = (opts['anchor_offset_x'] as num).toDouble();
  final ay = (opts['anchor_offset_y'] as num).toDouble();
  final interp = (opts['interpolated_scale_aspect_ratio'] as num).toDouble();

  final anchors = <double>[];
  var layerId = 0;
  while (layerId < numLayers) {
    var lastSameStride = layerId;
    var repeats = 0;
    while (lastSameStride < numLayers && strides[lastSameStride] == strides[layerId]) {
      lastSameStride++;
      repeats += (interp == 1.0) ? 2 : 1;
    }
    final stride = strides[layerId];
    final fmH = inputH ~/ stride;
    final fmW = inputW ~/ stride;
    for (var y = 0; y < fmH; y++) {
      final yCenter = (y + ay) / fmH;
      for (var x = 0; x < fmW; x++) {
        final xCenter = (x + ax) / fmW;
        for (var r = 0; r < repeats; r++) {
          anchors.add(xCenter);
          anchors.add(yCenter);
        }
      }
    }
    layerId = lastSameStride;
  }
  return Float32List.fromList(anchors);
}

Map<String, Object> _optsFor(FaceDetectionModel m) {
  switch (m) {
    case FaceDetectionModel.frontCamera:
      return _ssdFront;
    case FaceDetectionModel.backCamera:
      return _ssdBack;
    case FaceDetectionModel.shortRange:
      return _ssdShort;
    case FaceDetectionModel.full:
      return _ssdFull;
    case FaceDetectionModel.fullSparse:
      return _ssdFull;
  }
}

String _nameFor(FaceDetectionModel m) {
  switch (m) {
    case FaceDetectionModel.frontCamera:
      return _modelNameFront;
    case FaceDetectionModel.backCamera:
      return _modelNameBack;
    case FaceDetectionModel.shortRange:
      return _modelNameShort;
    case FaceDetectionModel.full:
      return _modelNameFull;
    case FaceDetectionModel.fullSparse:
      return _modelNameFullSparse;
  }
}