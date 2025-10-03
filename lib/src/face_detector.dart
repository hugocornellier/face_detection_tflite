part of face_detection_tflite;

class FaceDetector {
  FaceDetection? _detector;
  FaceLandmark? _faceLm;
  IrisLandmark? _iris;

  bool get isReady => _detector != null && _faceLm != null && _iris != null;

  Future<List<Detection>> getDetectionsWithIrisCenters(Uint8List imageBytes) async {
    if (_iris == null) {
      throw StateError('Iris model not initialized. Call initialize() before getDetectionsWithIrisCenters().');
    }

    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

    final dets = await detectFaces(imageBytes);
    if (dets.isEmpty) return dets;

    final det = dets.first;
    final aligned = await estimateAlignedFace(decoded, det);
    final meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
    final rois = eyeRoisFromMesh(meshPts);

    final centers = await _computeIrisCentersInIsolate(
      imageBytes,
      rois,
      leftFallback: det.landmarks[FaceIndex.leftEye],
      rightFallback: det.landmarks[FaceIndex.rightEye],
    );

    final imgW = decoded.width.toDouble();
    final imgH = decoded.height.toDouble();
    final kp = List<double>.from(det.keypointsXY);
    kp[FaceIndex.leftEye.index * 2]     = centers[0].dx / imgW;
    kp[FaceIndex.leftEye.index * 2 + 1] = centers[0].dy / imgH;
    kp[FaceIndex.rightEye.index * 2]    = centers[1].dx / imgW;
    kp[FaceIndex.rightEye.index * 2 + 1]= centers[1].dy / imgH;

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
    if (_tfliteLib != null) return;

    final exe = File(Platform.resolvedExecutable);
    final exeDir = exe.parent;

    late final List<String> candidates;
    late final String hint;

    if (Platform.isWindows) {
      candidates = [
        p.join(exeDir.path, 'libtensorflowlite_c-win.dll'),
        'libtensorflowlite_c-win.dll',
      ];
      hint =
      'Make sure your Windows plugin CMakeLists.txt sets:\n'
          '  set(PLUGIN_NAME_bundled_libraries ".../libtensorflowlite_c-win.dll" PARENT_SCOPE)\n'
          'so Flutter copies it next to the app EXE.';
    } else if (Platform.isLinux) {
      candidates = [
        p.join(exeDir.path, 'lib', 'libtensorflowlite_c-linux.so'),
        'libtensorflowlite_c-linux.so',
      ];
      hint =
      'Ensure linux/CMakeLists.txt sets:\n'
          '  set(PLUGIN_NAME_bundled_libraries "../assets/bin/libtensorflowlite_c-linux.so" PARENT_SCOPE)\n'
          'so Flutter copies it into bundle/lib/.';
    } else if (Platform.isMacOS) {
      final contents = exeDir.parent;
      candidates = [
        p.join(contents.path, 'Resources', 'libtensorflowlite_c-mac.dylib'),
        'libtensorflowlite_c-mac.dylib',
      ];
      hint = 'Expected in app bundle Resources, or resolvable by name.';
    } else {
      _tfliteLib = ffi.DynamicLibrary.process();
      return;
    }

    final tried = <String>[];
    for (final c in candidates) {
      try {
        if (c.contains(p.separator)) {
          if (!File(c).existsSync()) {
            tried.add(c);
            continue;
          }
        }
        _tfliteLib = ffi.DynamicLibrary.open(c);
        return;
      } catch (_) {
        tried.add(c);
      }
    }

    throw ArgumentError(
      'Failed to locate TensorFlow Lite C library.\n'
          'Tried:\n - ${tried.join('\n - ')}\n\n$hint',
    );
  }

  Future<void> initialize({FaceDetectionModel model = FaceDetectionModel.backCamera, InterpreterOptions? options}) async {
    await _ensureTFLiteLoaded();
    _detector = await FaceDetection.create(model, options: options, useIsolate: true);
    _faceLm = await FaceLandmark.create(options: options, useIsolate: true);
    _iris = await IrisLandmark.create(options: options, useIsolate: true);
  }

  Future<List<Detection>> detectFaces(Uint8List imageBytes, {RectF? roi}) async {
    final d = _detector;
    if (d == null) {
      throw StateError('FaceDetector not initialized. Call initialize() before detectFaces().');
    }
    if (_iris == null) {
      throw StateError('Iris model not initialized. initialize() must succeed before detectFaces().');
    }

    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

    final dets = await d.call(imageBytes, roi: roi);
    if (dets.isEmpty) return dets;

    final updated = <Detection>[];
    for (final det in dets) {
      final aligned = await estimateAlignedFace(decoded, det);
      final meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
      final rois = eyeRoisFromMesh(meshPts);

      final centers = await _computeIrisCentersInIsolate(
        imageBytes,
        rois,
        leftFallback: det.landmarks[FaceIndex.leftEye],
        rightFallback: det.landmarks[FaceIndex.rightEye],
      );

      final imgW = det.imageSize?.width ?? decoded.width.toDouble();
      final imgH = det.imageSize?.height ?? decoded.height.toDouble();
      final kp = List<double>.from(det.keypointsXY);
      kp[FaceIndex.leftEye.index * 2]     = centers[0].dx / imgW;
      kp[FaceIndex.leftEye.index * 2 + 1] = centers[0].dy / imgH;
      kp[FaceIndex.rightEye.index * 2]    = centers[1].dx / imgW;
      kp[FaceIndex.rightEye.index * 2 + 1]= centers[1].dy / imgH;

      updated.add(Detection(bbox: det.bbox, score: det.score, keypointsXY: kp, imageSize: det.imageSize));
    }
    return updated;
  }

  Future<AlignedFace> estimateAlignedFace(img.Image decoded, Detection det) async {
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

    final faceCrop = await extractAlignedSquare(decoded, cx, cy, size, -theta);

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

  List<AlignedRoi> eyeRoisFromMesh(List<Offset> meshAbs) {
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
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

    final dets = await detectFaces(imageBytes);
    if (dets.isEmpty) return const <Offset>[];
    final aligned = await estimateAlignedFace(decoded, dets.first);
    return await meshFromAlignedFace(aligned.faceCrop, aligned);
  }

  Future<List<Offset>> getIris(Uint8List imageBytes) async {
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

    final dets = await detectFaces(imageBytes);
    if (dets.isEmpty) return const <Offset>[];
    final aligned = await estimateAlignedFace(decoded, dets.first);
    final meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
    final rois = eyeRoisFromMesh(meshPts);
    return await irisFromEyeRois(decoded, rois);
  }

  Future<Size> getOriginalSize(Uint8List imageBytes) async {
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

    return Size(decoded.width.toDouble(), decoded.height.toDouble());
  }

  Future<List<Offset>> getFaceMeshFromDetections(Uint8List imageBytes, List<Detection> dets) async {
    if (dets.isEmpty) return const <Offset>[];
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final aligned = await estimateAlignedFace(decoded, dets.first);
    return await meshFromAlignedFace(aligned.faceCrop, aligned);
  }

  Future<List<Offset>> getIrisFromMesh(Uint8List imageBytes, List<Offset> meshPts) async {
    if (meshPts.isEmpty) return const <Offset>[];
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);
    final rois = eyeRoisFromMesh(meshPts);
    return await irisFromEyeRois(decoded, rois);
  }

  Future<PipelineResult> runAll(Uint8List imageBytes) async {
    final _DecodedRgb _d = await _decodeImageOffUi(imageBytes);
    final img.Image decoded = _imageFromDecodedRgb(_d);

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
    final aligned = await estimateAlignedFace(decoded, d);
    final meshPts = await meshFromAlignedFace(aligned.faceCrop, aligned);
    final rois = eyeRoisFromMesh(meshPts);
    final irisPts = await irisFromEyeRois(decoded, rois);
    return PipelineResult(
      detections: dets,
      mesh: meshPts,
      irises: irisPts,
      originalSize: Size(decoded.width.toDouble(), decoded.height.toDouble()),
    );
  }

  void dispose() {
    _detector?.dispose();
    _faceLm?.dispose();
    _iris?.dispose();
  }

  Future<List<Offset>> _computeIrisCentersInIsolate(
      Uint8List imageBytes,
      List<AlignedRoi> rois, {
        Offset? leftFallback,
        Offset? rightFallback,
      }) async {
    final rp = ReceivePort();
    final params = {
      'sendPort': rp.sendPort,
      'rootToken': RootIsolateToken.instance,
      'imageBytes': imageBytes,
      'rois': rois.map((r) => {'cx': r.cx, 'cy': r.cy, 'size': r.size, 'theta': r.theta}).toList(),
      'leftFx': leftFallback?.dx,
      'leftFy': leftFallback?.dy,
      'rightFx': rightFallback?.dx,
      'rightFy': rightFallback?.dy,
    };
    final iso = await Isolate.spawn(_irisCentersIsolate, params);
    final msg = await rp.first as Map;
    rp.close();
    iso.kill(priority: Isolate.immediate);
    if (msg['ok'] == true) {
      final List centers = msg['centers'] as List;
      return centers
          .map<Offset>((m) => Offset((m['x'] as num).toDouble(), (m['y'] as num).toDouble()))
          .toList();
    } else {
      final lx = (params['leftFx'] as double?) ?? 0.0;
      final ly = (params['leftFy'] as double?) ?? 0.0;
      final rx = (params['rightFx'] as double?) ?? 0.0;
      final ry = (params['rightFy'] as double?) ?? 0.0;
      return [Offset(lx, ly), Offset(rx, ry)];
    }
  }

  @pragma('vm:entry-point')
  static Future<void> _irisCentersIsolate(Map<String, dynamic> params) async {
    final RootIsolateToken token = params['rootToken'] as RootIsolateToken;
    DartPluginRegistrant.ensureInitialized();
    BackgroundIsolateBinaryMessenger.ensureInitialized(token);

    final SendPort sp = params['sendPort'] as SendPort;
    final Uint8List bytes = params['imageBytes'] as Uint8List;
    final List roisData = params['rois'] as List;

    try {
      final iris = await IrisLandmark.create(useIsolate: false);
      final img.Image? decoded = img.decodeImage(bytes);
      if (decoded == null) {
        sp.send({'ok': false});
        return;
      }

      final centers = <Map<String, double>>[];
      for (int i = 0; i < roisData.length; i++) {
        final m = roisData[i] as Map;
        final roi = AlignedRoi(
          (m['cx'] as num).toDouble(),
          (m['cy'] as num).toDouble(),
          (m['size'] as num).toDouble(),
          (m['theta'] as num).toDouble(),
        );
        final lm = await iris.runOnImageAlignedIris(decoded, roi, isRight: i == 1);
        if (lm.isEmpty) {
          final fx = i == 0 ? (params['leftFx'] as double?) : (params['rightFx'] as double?);
          final fy = i == 0 ? (params['leftFy'] as double?) : (params['rightFy'] as double?);
          centers.add({'x': (fx ?? 0.0), 'y': (fy ?? 0.0)});
        } else {
          double sx = 0.0, sy = 0.0;
          for (final p in lm) {
            sx += (p[0] as num).toDouble();
            sy += (p[1] as num).toDouble();
          }
          final cx = sx / lm.length;
          final cy = sy / lm.length;
          centers.add({'x': cx, 'y': cy});
        }
      }

      iris.dispose();
      sp.send({'ok': true, 'centers': centers});
    } catch (_) {
      sp.send({'ok': false});
    }
  }
}