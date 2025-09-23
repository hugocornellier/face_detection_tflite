library;

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

enum FaceIndex { leftEye, rightEye, noseTip, mouth, leftEyeTragion, rightEyeTragion }
enum FaceDetectionModel { frontCamera, backCamera, shortRange, full, fullSparse }

const _modelNameBack = 'face_detection_back.tflite';
const _modelNameFront = 'face_detection_front.tflite';
const _modelNameShort = 'face_detection_short_range.tflite';
const _modelNameFull = 'face_detection_full_range.tflite';
const _modelNameFullSparse = 'face_detection_full_range_sparse.tflite';
const _faceLandmarkModel = 'face_landmark.tflite';
const _irisLandmarkModel = 'iris_landmark.tflite';

const _rawScoreLimit = 80.0;
const _minScore = 0.5;
const _minSuppressionThreshold = 0.3;

const _ssdFront = {
  'num_layers': 4,
  'input_size_height': 128,
  'input_size_width': 128,
  'anchor_offset_x': 0.5,
  'anchor_offset_y': 0.5,
  'strides': [8, 16, 16, 16],
  'interpolated_scale_aspect_ratio': 1.0,
};
const _ssdBack = {
  'num_layers': 4,
  'input_size_height': 256,
  'input_size_width': 256,
  'anchor_offset_x': 0.5,
  'anchor_offset_y': 0.5,
  'strides': [16, 32, 32, 32],
  'interpolated_scale_aspect_ratio': 1.0,
};
const _ssdShort = {
  'num_layers': 4,
  'input_size_height': 128,
  'input_size_width': 128,
  'anchor_offset_x': 0.5,
  'anchor_offset_y': 0.5,
  'strides': [8, 16, 16, 16],
  'interpolated_scale_aspect_ratio': 1.0,
};
const _ssdFull = {
  'num_layers': 1,
  'input_size_height': 192,
  'input_size_width': 192,
  'anchor_offset_x': 0.5,
  'anchor_offset_y': 0.5,
  'strides': [4],
  'interpolated_scale_aspect_ratio': 0.0,
};

class AlignedFace {
  final double cx;
  final double cy;
  final double size;
  final double theta;
  final img.Image faceCrop;
  AlignedFace({required this.cx, required this.cy, required this.size, required this.theta, required this.faceCrop});
}

class PipelineResult {
  final List<Detection> detections;
  final List<Offset> mesh;
  final List<Offset> irises;
  final Size originalSize;
  PipelineResult({required this.detections, required this.mesh, required this.irises, required this.originalSize});
}

class FaceDetector {
  FaceDetection? _detector;
  FaceLandmark? _faceLm;
  IrisLandmark? _iris;

  bool get isReady => _detector != null && _faceLm != null && _iris != null;

  Future<List<Detection>> getDetectionsWithIrisCenters(Uint8List imageBytes) async {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return const <Detection>[];
    final dets = await detectFaces(imageBytes);
    if (dets.isEmpty) return dets;
    final det = dets.first;

    final aligned = estimateAlignedFace(decoded, det);
    final meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
    final rois = eyeRoisFromMesh(meshPts, decoded.width, decoded.height);

    final ir = _iris;
    if (ir == null) return dets;

    final centers = <Offset>[];
    for (int i = 0; i < rois.length; i++) {
      final isRight = (i == 1);
      final raw = await ir.runOnImageAlignedIris(decoded, rois[i], isRight: isRight);
      final pts = raw.map((p) => Offset(p[0].toDouble(), p[1].toDouble())).toList();
      if (pts.isEmpty) {
        centers.add(Offset(aligned.cx, aligned.cy));
        continue;
      }
      int centerIdx = 0;
      double best = double.infinity;
      for (int k = 0; k < pts.length; k++) {
        double s = 0;
        for (int j = 0; j < pts.length; j++) {
          if (j == k) continue;
          final dx = pts[j].dx - pts[k].dx;
          final dy = pts[j].dy - pts[k].dy;
          s += dx * dx + dy * dy;
        }
        if (s < best) {
          best = s;
          centerIdx = k;
        }
      }
      centers.add(pts[centerIdx]);
    }

    final imgW = decoded.width.toDouble();
    final imgH = decoded.height.toDouble();
    final kp = List<double>.from(det.keypointsXY);
    kp[FaceIndex.leftEye.index * 2] = centers[0].dx / imgW;
    kp[FaceIndex.leftEye.index * 2 + 1] = centers[0].dy / imgH;
    kp[FaceIndex.rightEye.index * 2] = centers[1].dx / imgW;
    kp[FaceIndex.rightEye.index * 2 + 1] = centers[1].dy / imgH;

    final updatedFirst = Detection(
      bbox: det.bbox,
      score: det.score,
      keypointsXY: kp,
      imageSize: det.imageSize,
    );

    return [updatedFirst, ...dets.skip(1)];
  }

  static ffi.DynamicLibrary? _tfliteLib;
  static Future<void> _ensureTFLiteLoaded() async {
    if (_tfliteLib != null) {
        return;
    }

    if (Platform.isWindows) {
      final exeFile  = File(Platform.resolvedExecutable);
      final exeDir   = exeFile.parent;
      final blobsDir = Directory(p.join(exeDir.path, 'blobs'));
      if (!blobsDir.existsSync()) blobsDir.createSync(recursive: true);

      const dllName = 'libtensorflowlite_c-win.dll';

      final assetsBinDir = p.join(
        exeDir.path,
        'data',
        'flutter_assets',
        'packages',
        'face_detection_tflite',
        'assets',
        'bin',
      );

      final srcDll = File(p.join(assetsBinDir, dllName));
      final dstDll = File(p.join(blobsDir.path, dllName));

      if (srcDll.existsSync()) {
        try {
          if (!dstDll.existsSync()) {
            srcDll.copySync(dstDll.path);
          }
        } catch (_) {}
      } else {
        try {
          final data = await rootBundle.load('packages/face_detection_tflite/assets/bin/$dllName');
          final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
          await dstDll.writeAsBytes(bytes, flush: true);
        } catch (_) {}
      }

      String? toOpen;
      if (dstDll.existsSync()) {
        toOpen = dstDll.path;
      } else if (srcDll.existsSync()) {
        toOpen = srcDll.path;
      }

      if (toOpen == null) {
        throw ArgumentError(
            'libtensorflowlite_c-win.dll not found. Expected at:\n'
                '  ${dstDll.path}\n'
                'or source:\n'
                '  ${srcDll.path}\n'
                'Ensure assets/bin/libtensorflowlite_c-win.dll is included in pubspec.'
        );
      }

      _tfliteLib = ffi.DynamicLibrary.open(toOpen);
      return;
    }

    if (Platform.isLinux) {
      _tfliteLib = ffi.DynamicLibrary.open('libtensorflowlite_c.so');
      return;
    }

    if (Platform.isMacOS) {
      final exe = File(Platform.resolvedExecutable);
      final macOSDir = exe.parent;
      final contents = macOSDir.parent;

      final frameworks = p.join(contents.path, 'Frameworks', 'libtensorflowlite_c-mac.dylib');
      final resources  = p.join(contents.path, 'Resources',  'libtensorflowlite_c-mac.dylib');

      if (File(frameworks).existsSync()) {
        _tfliteLib = ffi.DynamicLibrary.open(frameworks);
        return;
      }
      if (File(resources).existsSync()) {
        _tfliteLib = ffi.DynamicLibrary.open(resources);
        return;
      }

      try {
        _tfliteLib = ffi.DynamicLibrary.open('libtensorflowlite_c-mac.dylib');
        return;
      } catch (_) {
        throw ArgumentError(
          'Failed to locate libtensorflowlite_c-mac.dylib\n'
              'Checked:\n - $frameworks\n - $resources\n'
              'and loader search paths by name.',
        );
      }
    }

    _tfliteLib = ffi.DynamicLibrary.process();
  }

  Future<void> initialize({FaceDetectionModel model = FaceDetectionModel.backCamera, InterpreterOptions? options}) async {
    await _ensureTFLiteLoaded();
    _detector = await FaceDetection.create(model, options: options);
    _faceLm = await FaceLandmark.create(options: options);
    _iris = await IrisLandmark.create(options: options);
  }

  Future<List<Detection>> detectFaces(Uint8List imageBytes, {RectF? roi}) async {
    final d = _detector;
    if (d == null) return const <Detection>[];

    final dets = await d.call(imageBytes, roi: roi);
    if (dets.isEmpty || _iris == null) return dets;

    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return dets;

    final updated = <Detection>[];
    for (final det in dets) {
      final aligned = estimateAlignedFace(decoded, det);
      final meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
      final rois = eyeRoisFromMesh(meshPts, decoded.width, decoded.height);

      final centers = <Offset>[];
      for (int i = 0; i < rois.length; i++) {
        final isRight = (i == 1);
        final raw = await _iris!.runOnImageAlignedIris(decoded, rois[i], isRight: isRight);
        if (raw.isEmpty) {
          final fallback = det.landmarks[i == 0 ? FaceIndex.leftEye : FaceIndex.rightEye]!;
          centers.add(fallback);
          continue;
        }
        final pts = raw.map((p) => Offset(p[0].toDouble(), p[1].toDouble())).toList();

        int centerIdx = 0;
        double best = double.infinity;
        for (int k = 0; k < pts.length; k++) {
          double s = 0;
          for (int j = 0; j < pts.length; j++) {
            if (j == k) continue;
            final dx = pts[j].dx - pts[k].dx;
            final dy = pts[j].dy - pts[k].dy;
            s += dx * dx + dy * dy;
          }
          if (s < best) {
            best = s;
            centerIdx = k;
          }
        }
        centers.add(pts[centerIdx]);
      }

      final imgW = det.imageSize?.width ?? decoded.width.toDouble();
      final imgH = det.imageSize?.height ?? decoded.height.toDouble();
      final kp = List<double>.from(det.keypointsXY);

      kp[FaceIndex.leftEye.index * 2] = centers[0].dx / imgW;
      kp[FaceIndex.leftEye.index * 2 + 1] = centers[0].dy / imgH;
      kp[FaceIndex.rightEye.index * 2] = centers[1].dx / imgW;
      kp[FaceIndex.rightEye.index * 2 + 1] = centers[1].dy / imgH;

      updated.add(Detection(
        bbox: det.bbox,
        score: det.score,
        keypointsXY: kp,
        imageSize: det.imageSize,
      ));
    }
    return updated;
  }

  AlignedFace estimateAlignedFace(img.Image decoded, Detection det) {
    final imgW = decoded.width.toDouble();
    final imgH = decoded.height.toDouble();

    final lx = det.keypointsXY[FaceIndex.leftEye.index * 2] * imgW;
    final ly = det.keypointsXY[FaceIndex.leftEye.index * 2 + 1] * imgH;
    final rx = det.keypointsXY[FaceIndex.rightEye.index * 2] * imgW;
    final ry = det.keypointsXY[FaceIndex.rightEye.index * 2 + 1] * imgH;
    final mx = det.keypointsXY[FaceIndex.mouth.index * 2] * imgW;
    final my = det.keypointsXY[FaceIndex.mouth.index * 2 + 1] * imgH;

    final eyeCx = (lx + rx) * 0.5;
    final eyeCy = (ly + ry) * 0.5;

    final vEx = rx - lx;
    final vEy = ry - ly;
    final vMx = mx - eyeCx;
    final vMy = my - eyeCy;

    final theta = math.atan2(vEy, vEx);
    final eyeDist = math.sqrt(vEx * vEx + vEy * vEy);
    final mouthDist = math.sqrt(vMx * vMx + vMy * vMy);
    final size = math.max(mouthDist * 3.6, eyeDist * 4.0);

    final cx = eyeCx + vMx * 0.1;
    final cy = eyeCy + vMy * 0.1;

    final faceCrop = extractAlignedSquare(decoded, cx, cy, size, -theta);

    return AlignedFace(cx: cx, cy: cy, size: size, theta: theta, faceCrop: faceCrop);
  }

  Future<List<Offset>> meshFromAlignedFace(img.Image faceCrop, AlignedFace aligned) async {
    final fl = _faceLm;
    if (fl == null) return const <Offset>[];
    final lmNorm = await fl.call(faceCrop);
    final ct = math.cos(aligned.theta);
    final st = math.sin(aligned.theta);
    final s = aligned.size;
    final cx = aligned.cx;
    final cy = aligned.cy;
    final out = <Offset>[];
    for (final p in lmNorm) {
      final lx2 = (p[0] - 0.5) * s;
      final ly2 = (p[1] - 0.5) * s;
      final x = cx + lx2 * ct - ly2 * st;
      final y = cy + lx2 * st + ly2 * ct;
      out.add(Offset(x.toDouble(), y.toDouble()));
    }
    return out;
  }

  List<AlignedRoi> eyeRoisFromMesh(List<Offset> meshAbs, int imgW, int imgH) {
    AlignedRoi fromCorners(int a, int b) {
      final p0 = meshAbs[a];
      final p1 = meshAbs[b];
      final cx = (p0.dx + p1.dx) * 0.5;
      final cy = (p0.dy + p1.dy) * 0.5;
      final dx = p1.dx - p0.dx;
      final dy = p1.dy - p0.dy;
      final theta = math.atan2(dy, dx);
      final eyeDist = math.sqrt(dx * dx + dy * dy);
      final size = eyeDist * 2.3;
      return AlignedRoi(cx, cy, size, theta);
    }
    final left = fromCorners(33, 133);
    final right = fromCorners(362, 263);
    return [left, right];
  }

  Future<List<Offset>> irisFromEyeRois(img.Image decoded, List<AlignedRoi> rois) async {
    final ir = _iris;
    if (ir == null) return const <Offset>[];
    final pts = <Offset>[];
    for (int i = 0; i < rois.length; i++) {
      final isRight = (i == 1);
      final irisLm = await ir.runOnImageAlignedIris(decoded, rois[i], isRight: isRight);
      for (final p in irisLm) {
        pts.add(Offset(p[0].toDouble(), p[1].toDouble()));
      }
    }
    return pts;
  }

  Future<List<Detection>> getDetections(Uint8List imageBytes) async {
    return await detectFaces(imageBytes);
  }

  Future<List<Offset>> getFaceMesh(Uint8List imageBytes) async {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return const <Offset>[];
    final dets = await detectFaces(imageBytes);
    if (dets.isEmpty) return const <Offset>[];
    final aligned = estimateAlignedFace(decoded, dets.first);
    return await meshFromAlignedFace(aligned.faceCrop, aligned);
  }

  Future<List<Offset>> getIris(Uint8List imageBytes) async {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return const <Offset>[];
    final dets = await detectFaces(imageBytes);
    if (dets.isEmpty) return const <Offset>[];
    final aligned = estimateAlignedFace(decoded, dets.first);
    final meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
    final rois = eyeRoisFromMesh(meshPts, decoded.width, decoded.height);
    return await irisFromEyeRois(decoded, rois);
  }

  Future<Size> getOriginalSize(Uint8List imageBytes) async {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return const Size(0, 0);
    return Size(decoded.width.toDouble(), decoded.height.toDouble());
  }

  Future<List<Offset>> getFaceMeshFromDetections(Uint8List imageBytes, List<Detection> dets) async {
    if (dets.isEmpty) return const <Offset>[];
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return const <Offset>[];
    final aligned = estimateAlignedFace(decoded, dets.first);
    return await meshFromAlignedFace(aligned.faceCrop, aligned);
  }

  Future<List<Offset>> getIrisFromMesh(Uint8List imageBytes, List<Offset> meshPts) async {
    if (meshPts.isEmpty) return const <Offset>[];
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return const <Offset>[];
    final rois = eyeRoisFromMesh(meshPts, decoded.width, decoded.height);
    return await irisFromEyeRois(decoded, rois);
  }

  Future<PipelineResult> runAll(Uint8List imageBytes) async {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) {
      return PipelineResult(detections: const [], mesh: const [], irises: const [], originalSize: const Size(0, 0));
    }
    final dets = await detectFaces(imageBytes);
    if (dets.isEmpty) {
      return PipelineResult(
        detections: dets,
        mesh: const [],
        irises: const [],
        originalSize: Size(decoded.width.toDouble(), decoded.height.toDouble()),
      );
    }
    final d = dets.first;
    final aligned = estimateAlignedFace(decoded, d);
    final meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
    final rois = eyeRoisFromMesh(meshPts, decoded.width, decoded.height);
    final irisPts = await irisFromEyeRois(decoded, rois);
    return PipelineResult(
      detections: dets,
      mesh: meshPts,
      irises: irisPts,
      originalSize: Size(decoded.width.toDouble(), decoded.height.toDouble()),
    );
  }
}

extension on IrisLandmark {
  img.Image _flipHorizontal(img.Image src) {
    final out = img.Image(width: src.width, height: src.height);
    for (int y = 0; y < src.height; y++) {
      for (int x = 0; x < src.width; x++) {
        out.setPixel(src.width - 1 - x, y, src.getPixel(x, y));
      }
    }
    return out;
  }

  Future<List<List<double>>> callIrisOnly(img.Image eyeCrop) async {
    final pack = _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

    _inputBuf.setAll(0, pack.tensorNHWC);
    _itp.invoke();

    Float32List? irisFlat;
    _outBuffers.forEach((_, buf) {
      if (buf.length == 15) {
        irisFlat = buf;
      }
    });
    if (irisFlat == null) {
      return const <List<double>>[];
    }

    final pt = pack.padding[0], pb = pack.padding[1], pl = pack.padding[2], pr = pack.padding[3];
    final sx = 1.0 - (pl + pr);
    final sy = 1.0 - (pt + pb);

    final flat = irisFlat!;
    final lm = <List<double>>[];
    for (var i = 0; i < 5; i++) {
      var x = flat[i * 3 + 0] / _inW;
      var y = flat[i * 3 + 1] / _inH;
      final z = flat[i * 3 + 2];
      x = (x - pl) / sx;
      y = (y - pt) / sy;
      lm.add([x, y, z]);
    }
    return lm;
  }

  Future<List<List<double>>> runOnImageAlignedIris(img.Image src, AlignedRoi roi, {bool isRight = false}) async {
    final crop = extractAlignedSquare(src, roi.cx, roi.cy, roi.size, roi.theta);
    final eye = isRight ? _flipHorizontal(crop) : crop;
    final lmNorm = await callIrisOnly(eye);
    final ct = math.cos(roi.theta);
    final st = math.sin(roi.theta);
    final s = roi.size;
    final out = <List<double>>[];
    for (final p in lmNorm) {
      final px = isRight ? (1.0 - p[0]) : p[0];
      final py = p[1];
      final lx2 = (px - 0.5) * s;
      final ly2 = (py - 0.5) * s;
      final x = roi.cx + lx2 * ct - ly2 * st;
      final y = roi.cy + lx2 * st + ly2 * ct;
      out.add([x, y, p[2]]);
    }
    return out;
  }
}

class RectF {
  final double xmin, ymin, xmax, ymax;
  const RectF(this.xmin, this.ymin, this.xmax, this.ymax);
  double get w => xmax - xmin;
  double get h => ymax - ymin;
  RectF scale(double sx, double sy) => RectF(xmin * sx, ymin * sy, xmax * sx, ymax * sy);
  RectF expand(double frac) {
    final cx = (xmin + xmax) * 0.5;
    final cy = (ymin + ymax) * 0.5;
    final hw = (w * (1.0 + frac)) * 0.5;
    final hh = (h * (1.0 + frac)) * 0.5;
    return RectF(cx - hw, cy - hh, cx + hw, cy + hh);
  }
}

class Detection {
  final RectF bbox;
  final double score;
  final List<double> keypointsXY;
  final Size? imageSize;

  Detection({
    required this.bbox,
    required this.score,
    required this.keypointsXY,
    this.imageSize,
  });

  double operator [](int i) => keypointsXY[i];

  Map<FaceIndex, Offset> get landmarks {
    final map = <FaceIndex, Offset>{};
    if (imageSize == null) {
      for (final idx in FaceIndex.values) {
        final x = keypointsXY[idx.index * 2];
        final y = keypointsXY[idx.index * 2 + 1];
        map[idx] = Offset(x, y);
      }
      return map;
    }
    final w = imageSize!.width;
    final h = imageSize!.height;
    for (final idx in FaceIndex.values) {
      final x = keypointsXY[idx.index * 2] * w;
      final y = keypointsXY[idx.index * 2 + 1] * h;
      map[idx] = Offset(x, y);
    }
    return map;
  }
}


class ImageTensor {
  final Float32List tensorNHWC;
  final List<double> padding;
  final int width, height;
  ImageTensor(this.tensorNHWC, this.padding, this.width, this.height);
}

double _clip(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);
double _sigmoidClipped(double x, {double limit = _rawScoreLimit}) {
  final v = _clip(x, -limit, limit);
  return 1.0 / (1.0 + math.exp(-v));
}

ImageTensor _imageToTensor(img.Image src, {required int outW, required int outH}) {
  final inW = src.width, inH = src.height;
  final scale = (outW / inW < outH / inH) ? outW / inW : outH / inH;
  final newW = (inW * scale).round();
  final newH = (inH * scale).round();

  final resized = img.copyResize(
    src,
    width: newW,
    height: newH,
    interpolation: img.Interpolation.linear,
  );

  final dx = (outW - newW) ~/ 2;
  final dy = (outH - newH) ~/ 2;

  final canvas = img.Image(width: outW, height: outH);
  img.fill(canvas, color: img.ColorRgb8(0, 0, 0));

  for (var y = 0; y < resized.height; y++) {
    for (var x = 0; x < resized.width; x++) {
      final px = resized.getPixel(x, y);
      canvas.setPixel(x + dx, y + dy, px);
    }
  }

  final t = Float32List(outW * outH * 3);
  var k = 0;
  for (var y = 0; y < outH; y++) {
    for (var x = 0; x < outW; x++) {
      final px = canvas.getPixel(x, y);
      t[k++] = (px.r / 127.5) - 1.0;
      t[k++] = (px.g / 127.5) - 1.0;
      t[k++] = (px.b / 127.5) - 1.0;
    }
  }

  final padTop = dy / outH;
  final padBottom = (outH - dy - newH) / outH;
  final padLeft = dx / outW;
  final padRight = (outW - dx - newW) / outW;

  return ImageTensor(t, [padTop, padBottom, padLeft, padRight], outW, outH);
}

List<Detection> _detectionLetterboxRemoval(List<Detection> dets, List<double> padding) {
  final pt = padding[0], pb = padding[1], pl = padding[2], pr = padding[3];
  final sx = 1.0 - (pl + pr);
  final sy = 1.0 - (pt + pb);
  RectF unpad(RectF r) => RectF((r.xmin - pl) / sx, (r.ymin - pt) / sy, (r.xmax - pl) / sx, (r.ymax - pt) / sy);
  List<double> unpadKp(List<double> kps) {
    final out = List<double>.from(kps);
    for (var i = 0; i < out.length; i += 2) {
      out[i] = (out[i] - pl) / sx;
      out[i + 1] = (out[i + 1] - pt) / sy;
    }
    return out;
  }
  return dets
      .map((d) => Detection(bbox: unpad(d.bbox), score: d.score, keypointsXY: unpadKp(d.keypointsXY)))
      .toList();
}

double _iou(RectF a, RectF b) {
  final x1 = math.max(a.xmin, b.xmin);
  final y1 = math.max(a.ymin, b.ymin);
  final x2 = math.min(a.xmax, b.xmax);
  final y2 = math.min(a.ymax, b.ymax);
  final iw = math.max(0.0, x2 - x1);
  final ih = math.max(0.0, y2 - y1);
  final inter = iw * ih;
  final areaA = math.max(0.0, a.w) * math.max(0.0, a.h);
  final areaB = math.max(0.0, b.w) * math.max(0.0, b.h);
  final uni = areaA + areaB - inter;
  return uni <= 0 ? 0.0 : inter / uni;
}

List<Detection> _nms(List<Detection> dets, double iouThresh, double scoreThresh, {bool weighted = true}) {
  final kept = <Detection>[];
  final cand = dets.where((d) => d.score >= scoreThresh).toList()
    ..sort((a, b) => b.score.compareTo(a.score));
  while (cand.isNotEmpty) {
    final base = cand.removeAt(0);
    final merged = <Detection>[base];
    cand.removeWhere((d) {
      if (_iou(base.bbox, d.bbox) >= iouThresh) {
        merged.add(d);
        return true;
      }
      return false;
    });
    if (!weighted || merged.length == 1) {
      kept.add(base);
    } else {
      double sw = 0, xmin = 0, ymin = 0, xmax = 0, ymax = 0;
      for (final m in merged) {
        sw += m.score;
        xmin += m.bbox.xmin * m.score;
        ymin += m.bbox.ymin * m.score;
        xmax += m.bbox.xmax * m.score;
        ymax += m.bbox.ymax * m.score;
      }
      kept.add(Detection(
        bbox: RectF(xmin / sw, ymin / sw, xmax / sw, ymax / sw),
        score: base.score,
        keypointsXY: base.keypointsXY,
      ));
    }
  }
  return kept;
}

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

class FaceDetection {
  final Interpreter _itp;
  final int _inW, _inH;
  final int _bboxIndex = 0, _scoreIndex = 1;

  final Float32List _anchors;
  final bool _assumeMirrored;

  late final int _inputIdx;
  late final List<int> _boxesShape;
  late final List<int> _scoresShape;

  late final Tensor _inputTensor;
  late final Tensor _boxesTensor;
  late final Tensor _scoresTensor;

  late final Float32List _inputBuf;
  late final Float32List _boxesBuf;
  late final Float32List _scoresBuf;

  FaceDetection._(this._itp, this._inW, this._inH, this._anchors, this._assumeMirrored);

  static Future<FaceDetection> create(FaceDetectionModel model, {InterpreterOptions? options}) async {
    final opts = _optsFor(model);
    final inW = opts['input_size_width'] as int;
    final inH = opts['input_size_height'] as int;
    final anchors = _ssdGenerateAnchors(opts);
    final itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/${_nameFor(model)}',
      options: options ?? InterpreterOptions(),
    );
    final assumeMirrored = switch (model) {
      FaceDetectionModel.backCamera => false,
      _ => true,
    };
    final obj = FaceDetection._(itp, inW, inH, anchors, assumeMirrored);

    int foundIdx = 0;
    for (final i in [0, 1, 2, 3]) {
      try {
        final s = itp.getInputTensor(i).shape;
        if (s.length == 4) {
          foundIdx = i;
          break;
        }
      } catch (_) {}
    }
    obj._inputIdx = foundIdx;
    itp.resizeInputTensor(obj._inputIdx, [1, inH, inW, 3]);
    itp.allocateTensors();

    obj._boxesShape = itp.getOutputTensor(obj._bboxIndex).shape;
    obj._scoresShape = itp.getOutputTensor(obj._scoreIndex).shape;

    obj._inputTensor = itp.getInputTensor(obj._inputIdx);
    obj._boxesTensor = itp.getOutputTensor(obj._bboxIndex);
    obj._scoresTensor = itp.getOutputTensor(obj._scoreIndex);

    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();
    obj._boxesBuf = obj._boxesTensor.data.buffer.asFloat32List();
    obj._scoresBuf = obj._scoresTensor.data.buffer.asFloat32List();

    return obj;
  }

  Future<List<Detection>> call(Uint8List imageBytes, {RectF? roi}) async {
    final src = img.decodeImage(imageBytes)!;
    final img.Image srcRoi = (roi == null) ? src : cropFromRoi(src, roi);
    final pack = _imageToTensor(srcRoi, outW: _inW, outH: _inH);

    _inputBuf.setAll(0, pack.tensorNHWC);
    _itp.invoke();

    final boxes = _decodeBoxes(_boxesBuf, _boxesShape);
    final scores = _decodeScores(_scoresBuf, _scoresShape);

    final dets = _toDetections(boxes, scores);
    final pruned = _nms(dets, _minSuppressionThreshold, _minScore, weighted: true);
    final fixed = _detectionLetterboxRemoval(pruned, pack.padding);

    List<Detection> mapped;
    if (roi != null) {
      final dx = roi.xmin;
      final dy = roi.ymin;
      final sx = roi.w;
      final sy = roi.h;
      mapped = fixed.map((d) {
        RectF mapRect(RectF r) =>
            RectF(dx + r.xmin * sx, dy + r.ymin * sy, dx + r.xmax * sx, dy + r.ymax * sy);
        List<double> mapKp(List<double> k) {
          final o = List<double>.from(k);
          for (int i = 0; i < o.length; i += 2) {
            o[i] = dx + o[i] * sx;
            o[i + 1] = dy + o[i + 1] * sy;
          }
          return o;
        }
        return Detection(bbox: mapRect(d.bbox), score: d.score, keypointsXY: mapKp(d.keypointsXY));
      }).toList();
    } else {
      mapped = fixed;
    }

    if (_assumeMirrored) {
      mapped = mapped.map((d) {
        final xmin = 1.0 - d.bbox.xmax;
        final xmax = 1.0 - d.bbox.xmin;
        final ymin = d.bbox.ymin;
        final ymax = d.bbox.ymax;
        final kp = List<double>.from(d.keypointsXY);
        for (int i = 0; i < kp.length; i += 2) {
          kp[i] = 1.0 - kp[i];
        }
        return Detection(bbox: RectF(xmin, ymin, xmax, ymax), score: d.score, keypointsXY: kp);
      }).toList();
    }
    final imgSize = Size(src.width.toDouble(), src.height.toDouble());
    mapped = mapped
        .map((d) => Detection(
      bbox: d.bbox,
      score: d.score,
      keypointsXY: d.keypointsXY,
      imageSize: imgSize,
    ))
        .toList();
    return mapped;
  }

  List<_DecodedBox> _decodeBoxes(Float32List raw, List<int> shape) {
    final n = shape[1], k = shape[2];
    final scale = _inH.toDouble();
    final out = <_DecodedBox>[];
    final tmp = Float32List(k);
    for (var i = 0; i < n; i++) {
      final base = i * k;
      for (var j = 0; j < k; j++) {
        tmp[j] = raw[base + j] / scale;
      }
      final ax = _anchors[i * 2 + 0];
      final ay = _anchors[i * 2 + 1];
      tmp[0] += ax;
      tmp[1] += ay;
      for (var j = 4; j < k; j += 2) {
        tmp[j + 0] += ax;
        tmp[j + 1] += ay;
      }
      final xc = tmp[0], yc = tmp[1], w = tmp[2], h = tmp[3];
      final xmin = xc - w * 0.5, ymin = yc - h * 0.5, xmax = xc + w * 0.5, ymax = yc + h * 0.5;
      final kp = <double>[];
      for (var j = 4; j < k; j += 2) {
        kp.add(tmp[j + 0]);
        kp.add(tmp[j + 1]);
      }
      out.add(_DecodedBox(RectF(xmin, ymin, xmax, ymax), kp));
    }
    return out;
  }

  Float32List _decodeScores(Float32List raw, List<int> shape) {
    final n = shape[1];
    final scores = Float32List(n);
    for (var i =  0; i < n; i++) {
      scores[i] = _sigmoidClipped(raw[i]);
    }
    return scores;
  }

  List<Detection> _toDetections(List<_DecodedBox> boxes, Float32List scores) {
    final res = <Detection>[];
    final n = math.min(boxes.length, scores.length);
    for (var i = 0; i < n; i++) {
      final b = boxes[i].bbox;
      if (b.xmax <= b.xmin || b.ymax <= b.ymin) continue;
      res.add(Detection(bbox: b, score: scores[i], keypointsXY: boxes[i].keypointsXY));
    }
    return res;
  }
}

class _DecodedBox {
  final RectF bbox;
  final List<double> keypointsXY;
  _DecodedBox(this.bbox, this.keypointsXY);
}

class FaceLandmark {
  final Interpreter _itp;
  final int _inW, _inH;

  late final int _bestIdx;
  late final Tensor _inputTensor;
  late final Tensor _bestTensor;

  late final Float32List _inputBuf;
  late final Float32List _bestOutBuf;

  FaceLandmark._(this._itp, this._inW, this._inH);

  static Future<FaceLandmark> create({InterpreterOptions? options}) async {
    final itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$_faceLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final ishape = itp.getInputTensor(0).shape;
    final inH = ishape[1];
    final inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final obj = FaceLandmark._(itp, inW, inH);

    obj._inputTensor = itp.getInputTensor(0);

    int numElements(List<int> s) => s.fold(1, (a, b) => a * b);

    final shapes = <int, List<int>>{};
    for (var i = 0;; i++) {
      try {
        final s = itp.getOutputTensor(i).shape;
        shapes[i] = s;
      } catch (_) {
        break;
      }
    }

    int bestIdx = -1;
    int bestLen = -1;
    for (final e in shapes.entries) {
      final len = numElements(e.value);
      if (len > bestLen && len % 3 == 0) {
        bestLen = len;
        bestIdx = e.key;
      }
    }
    obj._bestIdx = bestIdx;

    obj._bestTensor = itp.getOutputTensor(obj._bestIdx);

    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();
    obj._bestOutBuf = obj._bestTensor.data.buffer.asFloat32List();

    return obj;
  }

  Future<List<List<double>>> call(img.Image faceCrop) async {
    final pack = _imageToTensor(faceCrop, outW: _inW, outH: _inH);

    _inputBuf.setAll(0, pack.tensorNHWC);
    _itp.invoke();

    final pt = pack.padding[0], pb = pack.padding[1], pl = pack.padding[2], pr = pack.padding[3];
    final sx = 1.0 - (pl + pr);
    final sy = 1.0 - (pt + pb);

    final flat = _bestOutBuf;
    final n = (flat.length / 3).floor();
    final lm = <List<double>>[];
    for (var i = 0; i < n; i++) {
      var x = flat[i * 3 + 0] / _inW;
      var y = flat[i * 3 + 1] / _inH;
      final z = flat[i * 3 + 2];
      x = (x - pl) / sx;
      y = (y - pt) / sy;
      if (x < 0) x = 0; else if (x > 1) x = 1;
      if (y < 0) y = 0; else if (y > 1) y = 1;
      lm.add([x, y, z]);
    }
    return lm;
  }
}

class IrisLandmark {
  final Interpreter _itp;
  final int _inW, _inH;

  late final Tensor _inputTensor;
  late final Map<int, Float32List> _outBuffers;
  late final Float32List _inputBuf;

  IrisLandmark._(this._itp, this._inW, this._inH);

  static Future<IrisLandmark> create({InterpreterOptions? options}) async {
    final itp = await Interpreter.fromAsset(
      'packages/face_detection_tflite/assets/models/$_irisLandmarkModel',
      options: options ?? InterpreterOptions(),
    );
    final ishape = itp.getInputTensor(0).shape;
    final inH = ishape[1];
    final inW = ishape[2];
    itp.resizeInputTensor(0, [1, inH, inW, 3]);
    itp.allocateTensors();

    final obj = IrisLandmark._(itp, inW, inH);

    obj._inputTensor = itp.getInputTensor(0);

    final shapes = <int, List<int>>{};
    final tensors = <int, Tensor>{};
    final buffers = <int, Float32List>{};

    for (var i = 0;; i++) {
      try {
        final t = itp.getOutputTensor(i);
        final s = t.shape;
        shapes[i] = s;
        tensors[i] = t;
        buffers[i] = t.data.buffer.asFloat32List();
      } catch (_) {
        break;
      }
    }

    obj._outBuffers = buffers;
    obj._inputBuf = obj._inputTensor.data.buffer.asFloat32List();

    return obj;
  }

  Future<List<List<double>>> call(img.Image eyeCrop) async {
    final pack = _imageToTensor(eyeCrop, outW: _inW, outH: _inH);

    _inputBuf.setAll(0, pack.tensorNHWC);
    _itp.invoke();

    final lm = <List<double>>[];
    final pt = pack.padding[0], pb = pack.padding[1], pl = pack.padding[2], pr = pack.padding[3];
    final sx = 1.0 - (pl + pr);
    final sy = 1.0 - (pt + pb);

    for (final entry in _outBuffers.entries) {
      final flat = entry.value;
      final n = (flat.length / 3).floor();
      for (var i = 0; i < n; i++) {
        var x = flat[i * 3 + 0] / _inW;
        var y = flat[i * 3 + 1] / _inH;
        final z = flat[i * 3 + 2];
        x = (x - pl) / sx;
        y = (y - pt) / sy;
        lm.add([x, y, z]);
      }
    }

    return lm;
  }

  Future<List<List<double>>> runOnImage(img.Image src, RectF eyeRoi) async {
    final eyeCrop = cropFromRoi(src, eyeRoi);
    final lmNorm = await call(eyeCrop);
    final imgW = src.width.toDouble();
    final imgH = src.height.toDouble();
    final dx = eyeRoi.xmin * imgW;
    final dy = eyeRoi.ymin * imgH;
    final sx = eyeRoi.w * imgW;
    final sy = eyeRoi.h * imgH;
    final mapped = <List<double>>[];
    for (final p in lmNorm) {
      final x = dx + p[0] * sx;
      final y = dy + p[1] * sy;
      mapped.add([x, y, p[2]]);
    }
    return mapped;
  }
}

RectF faceDetectionToRoi(RectF bbox, {double expandFraction = 0.6}) {
  final e = bbox.expand(expandFraction);
  final cx = (e.xmin + e.xmax) * 0.5;
  final cy = (e.ymin + e.ymax) * 0.5;
  final s = math.max(e.w, e.h) * 0.5;
  return RectF(cx - s, cy - s, cx + s, cy + s);
}

img.Image cropFromRoi(img.Image src, RectF roi) {
  final w = src.width.toDouble(), h = src.height.toDouble();
  final x0 = (roi.xmin * w).clamp(0.0, w - 1).toInt();
  final y0 = (roi.ymin * h).clamp(0.0, h - 1).toInt();
  final x1 = (roi.xmax * w).clamp(0.0, w.toDouble()).toInt();
  final y1 = (roi.ymax * h).clamp(0.0, h.toDouble()).toInt();
  final cw = math.max(1, x1 - x0);
  final ch = math.max(1, y1 - y0);
  return img.copyCrop(src, x: x0, y: y0, width: cw, height: ch);
}

class AlignedRoi {
  final double cx;
  final double cy;
  final double size;
  final double theta;
  const AlignedRoi(this.cx, this.cy, this.size, this.theta);
}

img.Image extractAlignedSquare(img.Image src, double cx, double cy, double size, double theta) {
  final side = math.max(1, size.round());
  final ct = math.cos(theta);
  final st = math.sin(theta);
  final out = img.Image(width: side, height: side);
  for (int y = 0; y < side; y++) {
    final vy = ((y + 0.5) / side - 0.5) * size;
    for (int x = 0; x < side; x++) {
      final vx = ((x + 0.5) / side - 0.5) * size;
      final sx = cx + vx * ct - vy * st;
      final sy = cy + vx * st + vy * ct;
      final px = _bilinearSampleRgb8(src, sx, sy);
      out.setPixel(x, y, px);
    }
  }
  return out;
}

img.ColorRgb8 _bilinearSampleRgb8(img.Image src, double fx, double fy) {
  final x0 = fx.floor();
  final y0 = fy.floor();
  final x1 = x0 + 1;
  final y1 = y0 + 1;
  final ax = fx - x0;
  final ay = fy - y0;

  int cx0 = x0.clamp(0, src.width - 1);
  int cx1 = x1.clamp(0, src.width - 1);
  int cy0 = y0.clamp(0, src.height - 1);
  int cy1 = y1.clamp(0, src.height - 1);

  final p00 = src.getPixel(cx0, cy0);
  final p10 = src.getPixel(cx1, cy0);
  final p01 = src.getPixel(cx0, cy1);
  final p11 = src.getPixel(cx1, cy1);

  final r0 = p00.r * (1 - ax) + p10.r * ax;
  final g0 = p00.g * (1 - ax) + p10.g * ax;
  final b0 = p00.b * (1 - ax) + p10.b * ax;

  final r1 = p01.r * (1 - ax) + p11.r * ax;
  final g1 = p01.g * (1 - ax) + p11.g * ax;
  final b1 = p01.b * (1 - ax) + p11.b * ax;

  final r = (r0 * (1 - ay) + r1 * ay).round().clamp(0, 255);
  final g = (g0 * (1 - ay) + g1 * ay).round().clamp(0, 255);
  final b = (b0 * (1 - ay) + b1 * ay).round().clamp(0, 255);

  return img.ColorRgb8(r, g, b);
}