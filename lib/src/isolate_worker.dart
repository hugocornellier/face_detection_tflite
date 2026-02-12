part of '../face_detection_tflite.dart';

/// A long-lived background isolate for image processing operations.
///
/// This worker handles computationally expensive image operations (decoding,
/// tensor conversion, cropping, transformation) in a dedicated isolate to
/// avoid blocking the main UI thread. Unlike spawning fresh isolates for each
/// operation, this worker is initialized once and reused across many requests.
///
/// ## Performance Benefits
///
/// - **Eliminates per-frame isolate spawn overhead** (~1-5ms per spawn)
/// - **Reduces memory allocations** from repeated isolate creation
/// - **Enables live camera processing** at 30fps on mid-range devices
///
/// ## Usage
///
/// This class is designed to be used internally by [FaceDetector]. Most users
/// should not need to interact with it directly.
///
/// Example (internal usage):
/// ```dart
/// final worker = IsolateWorker();
/// await worker.initialize();
///
/// // Process many frames efficiently
/// for (final frame in cameraFrames) {
///   final decoded = await worker.decodeImage(frame);
///   final tensor = await worker.imageToTensor(decoded, outW: 128, outH: 128);
///   // ...
/// }
///
/// worker.dispose();
/// ```
class IsolateWorker {
  /// Creates an uninitialized worker; call [initialize] before sending work.
  IsolateWorker();

  Isolate? _isolate;
  SendPort? _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  final Map<int, Completer<dynamic>> _pending = {};
  int _nextId = 0;
  bool _initialized = false;
  int _nextFrameId = 0;

  /// Returns true if the worker has been initialized and is ready for requests.
  bool get isInitialized => _initialized;

  /// Initializes the worker isolate and establishes communication.
  ///
  /// This must be called before using any processing methods. The worker
  /// spawns a background isolate and sets up a bidirectional communication
  /// channel for sending requests and receiving responses.
  ///
  /// Throws [StateError] if initialization fails or if already initialized.
  Future<void> initialize() async {
    if (_initialized) {
      throw StateError('IsolateWorker already initialized');
    }

    try {
      _isolate = await Isolate.spawn(
        _isolateEntry,
        _receivePort.sendPort,
        debugName: 'IsolateWorker',
      );

      final Completer<SendPort> initCompleter = Completer<SendPort>();
      late final StreamSubscription subscription;

      subscription = _receivePort.listen((message) {
        if (!initCompleter.isCompleted) {
          if (message is SendPort) {
            initCompleter.complete(message);
          } else {
            initCompleter.completeError(
              StateError('Expected SendPort, got ${message.runtimeType}'),
            );
          }
          return;
        }

        _handleResponse(message);
      });

      _sendPort = await initCompleter.future.timeout(
        const Duration(seconds: 5),
        onTimeout: () {
          subscription.cancel();
          throw TimeoutException('Worker initialization timed out');
        },
      );

      _initialized = true;
    } catch (e) {
      _isolate?.kill(priority: Isolate.immediate);
      _receivePort.close();
      _initialized = false;
      rethrow;
    }
  }

  void _handleResponse(dynamic message) {
    if (message is! Map) {
      return; // Ignore malformed messages
    }

    final int? id = message['id'] as int?;
    if (id == null) {
      return; // Ignore messages without ID
    }

    final Completer? completer = _pending.remove(id);
    if (completer == null) {
      return; // Ignore responses for cancelled/unknown requests
    }

    if (message['error'] != null) {
      completer.completeError(message['error']);
    } else {
      completer.complete(message['result']);
    }
  }

  Future<T> _sendRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) async {
    if (!_initialized || _sendPort == null) {
      throw StateError(
        'IsolateWorker not initialized. Call initialize() first.',
      );
    }

    final int id = _nextId++;
    final Completer<T> completer = Completer<T>();
    _pending[id] = completer;

    try {
      _sendPort!.send({'id': id, 'op': operation, ...params});

      return await completer.future;
    } catch (e) {
      _pending.remove(id);
      rethrow;
    }
  }

  /// Decodes an encoded image (JPEG, PNG, etc.) to RGB format.
  ///
  /// The [bytes] parameter should contain encoded image data. The image is
  /// decoded in the worker isolate to avoid blocking the main thread.
  ///
  /// Returns a [DecodedRgb] containing the image dimensions and RGB pixel data.
  ///
  /// Throws [FormatException] if the image format is unsupported or corrupt.
  /// Throws [StateError] if the worker is not initialized.
  @Deprecated('Will be removed in 5.0.0. Use cv.imdecode instead.')
  Future<DecodedRgb> decodeImage(Uint8List bytes) async {
    final Map result = await _sendRequest<Map>('decode', {
      'bytes': TransferableTypedData.fromList([bytes]),
    });

    final ByteBuffer rgbBB =
        (result['rgb'] as TransferableTypedData).materialize();
    final Uint8List rgb = rgbBB.asUint8List();
    final int w = result['w'] as int;
    final int h = result['h'] as int;

    return DecodedRgb(w, h, rgb);
  }

  /// Decodes and registers an image in one operation (optimized fast-path).
  ///
  /// This combines [decodeImage] and [registerFrame] into a single operation
  /// that avoids transferring RGB data back to the main isolate. The image
  /// is decoded and stored in the worker, returning only the frameId and
  /// dimensions.
  ///
  /// This is more efficient than calling [decodeImage] followed by [registerFrame]
  /// when you don't need the decoded image on the main isolate. It eliminates
  /// a wasteful round-trip of RGB data (worker → main → worker).
  ///
  /// Returns a record `(frameId, width, height)`.
  ///
  /// **Important:** Call [releaseFrame] when done with the frame to free memory.
  ///
  /// Example:
  /// ```dart
  /// final (frameId, w, h) = await worker.decodeAndRegisterFrame(imageBytes);
  /// final crop = await worker.cropFromRoiWithFrameId(frameId, roi);
  /// await worker.releaseFrame(frameId); // Clean up
  /// ```
  @Deprecated('Will be removed in 5.0.0. Use cv.imdecode instead.')
  Future<(int, int, int)> decodeAndRegisterFrame(Uint8List bytes) async {
    final int frameId = _nextFrameId++;

    final Map result = await _sendRequest<Map>('decodeAndRegister', {
      'bytes': TransferableTypedData.fromList([bytes]),
      'frameId': frameId,
    });

    final int w = result['w'] as int;
    final int h = result['h'] as int;

    return (frameId, w, h);
  }

  /// Registers a frame in the worker isolate for efficient reuse.
  ///
  /// This method transfers the RGB pixel data to the worker once and returns
  /// a frame ID that can be used for multiple subsequent operations without
  /// re-transferring the full image data.
  ///
  /// The [src] image is converted to RGB format and stored in the worker.
  ///
  /// Returns a unique frame ID that can be passed to operations like
  /// [cropFromRoiWithFrameId], [extractAlignedSquareWithFrameId], etc.
  ///
  /// **Important:** Call [releaseFrame] when done with the frame to free memory.
  ///
  /// Example:
  /// ```dart
  /// final frameId = await worker.registerFrame(image);
  /// final crop1 = await worker.cropFromRoiWithFrameId(frameId, roi1);
  /// final crop2 = await worker.cropFromRoiWithFrameId(frameId, roi2);
  /// await worker.releaseFrame(frameId); // Clean up
  /// ```
  @Deprecated(
    'Will be removed in 5.0.0. The image package dependency will be removed.',
  )
  Future<int> registerFrame(img.Image src) async {
    final Uint8List rgb = src.getBytes(order: img.ChannelOrder.rgb);
    final int frameId = _nextFrameId++;

    await _sendRequest<void>('registerFrame', {
      'frameId': frameId,
      'w': src.width,
      'h': src.height,
      'rgb': TransferableTypedData.fromList([rgb]),
    });

    return frameId;
  }

  /// Releases a registered frame from the worker isolate.
  ///
  /// Frees the memory associated with the frame ID returned by [registerFrame].
  /// After calling this, the frame ID is no longer valid and should not be used.
  ///
  /// It's important to call this when done processing a frame to avoid memory leaks.
  @Deprecated('Will be removed in 5.0.0. The frame pipeline is being removed.')
  Future<void> releaseFrame(int frameId) async {
    await _sendRequest<void>('releaseFrame', {'frameId': frameId});
  }

  /// Converts an image to a normalized tensor for TensorFlow Lite inference.
  ///
  /// The [src] image is resized to [outW]×[outH] using letterboxing (aspect-
  /// preserving resize with black padding). Pixel values are normalized to
  /// the range [-1.0, 1.0] as expected by MediaPipe models.
  ///
  /// Returns an [ImageTensor] containing the normalized tensor data and
  /// padding information needed to reverse the letterbox transformation.
  ///
  /// Throws [StateError] if the worker is not initialized.
  @Deprecated(
    'Will be removed in 5.0.0. Use convertImageToTensorFromMat instead.',
  )
  Future<ImageTensor> imageToTensor(
    img.Image src, {
    required int outW,
    required int outH,
  }) async {
    final Uint8List rgb = src.getBytes(order: img.ChannelOrder.rgb);

    final Map result = await _sendRequest<Map>('tensor', {
      'inW': src.width,
      'inH': src.height,
      'outW': outW,
      'outH': outH,
      'rgb': TransferableTypedData.fromList([rgb]),
    });

    final ByteBuffer tensorBB =
        (result['tensor'] as TransferableTypedData).materialize();
    final Float32List tensor = tensorBB.asUint8List().buffer.asFloat32List();
    final List paddingRaw = result['padding'] as List;
    final List<double> padding =
        paddingRaw.map((e) => (e as num).toDouble()).toList();
    final int ow = result['outW'] as int;
    final int oh = result['outH'] as int;

    return ImageTensor(tensor, padding, ow, oh);
  }

  /// Converts a registered frame to a normalized tensor.
  ///
  /// Similar to [imageToTensor] but uses a previously registered frame ID
  /// instead of transferring the full image data again.
  ///
  /// Returns an [ImageTensor] containing the normalized tensor data and
  /// padding information.
  @Deprecated(
    'Will be removed in 5.0.0. Use convertImageToTensorFromMat instead.',
  )
  Future<ImageTensor> imageToTensorWithFrameId(
    int frameId, {
    required int outW,
    required int outH,
  }) async {
    final Map result = await _sendRequest<Map>('tensorFromFrame', {
      'frameId': frameId,
      'outW': outW,
      'outH': outH,
    });

    final ByteBuffer tensorBB =
        (result['tensor'] as TransferableTypedData).materialize();
    final Float32List tensor = tensorBB.asUint8List().buffer.asFloat32List();
    final List paddingRaw = result['padding'] as List;
    final List<double> padding =
        paddingRaw.map((e) => (e as num).toDouble()).toList();
    final int ow = result['outW'] as int;
    final int oh = result['outH'] as int;

    return ImageTensor(tensor, padding, ow, oh);
  }

  /// Crops a region of interest from an image using normalized coordinates.
  ///
  /// The [src] image is cropped to the region defined by [roi], where all
  /// coordinates are normalized (0.0 to 1.0) relative to image dimensions.
  ///
  /// Returns the cropped [img.Image].
  ///
  /// Throws [ArgumentError] if ROI coordinates are invalid.
  /// Throws [StateError] if the worker is not initialized or crop fails.
  @Deprecated('Will be removed in 5.0.0. Use cropFromRoiMat instead.')
  Future<img.Image> cropFromRoi(img.Image src, RectF roi) async {
    if (roi.xmin < 0 || roi.ymin < 0 || roi.xmax > 1 || roi.ymax > 1) {
      throw ArgumentError(
        'ROI coordinates must be normalized [0,1], got: '
        '(${roi.xmin}, ${roi.ymin}, ${roi.xmax}, ${roi.ymax})',
      );
    }
    if (roi.xmin >= roi.xmax || roi.ymin >= roi.ymax) {
      throw ArgumentError('Invalid ROI: min coordinates must be less than max');
    }

    final Uint8List rgb = src.getBytes(order: img.ChannelOrder.rgb);

    final Map result = await _sendRequest<Map>('crop', {
      'w': src.width,
      'h': src.height,
      'rgb': TransferableTypedData.fromList([rgb]),
      'roi': {
        'xmin': roi.xmin,
        'ymin': roi.ymin,
        'xmax': roi.xmax,
        'ymax': roi.ymax,
      },
    });

    if (result['ok'] != true) {
      final error = result['error'];
      throw StateError('Image crop failed: ${error ?? "unknown error"}');
    }

    final ByteBuffer outBB =
        (result['rgb'] as TransferableTypedData).materialize();
    final Uint8List outRgb = outBB.asUint8List();
    final int ow = result['w'] as int;
    final int oh = result['h'] as int;

    return img.Image.fromBytes(
      width: ow,
      height: oh,
      bytes: outRgb.buffer,
      order: img.ChannelOrder.rgb,
    );
  }

  /// Crops a region of interest from a registered frame.
  ///
  /// Similar to [cropFromRoi] but uses a previously registered frame ID
  /// instead of transferring the full image data again.
  @Deprecated(
    'Will be removed in 5.0.0. The image package dependency will be removed.',
  )
  Future<img.Image> cropFromRoiWithFrameId(int frameId, RectF roi) async {
    if (roi.xmin < 0 || roi.ymin < 0 || roi.xmax > 1 || roi.ymax > 1) {
      throw ArgumentError(
        'ROI coordinates must be normalized [0,1], got: '
        '(${roi.xmin}, ${roi.ymin}, ${roi.xmax}, ${roi.ymax})',
      );
    }
    if (roi.xmin >= roi.xmax || roi.ymin >= roi.ymax) {
      throw ArgumentError('Invalid ROI: min coordinates must be less than max');
    }

    final Map result = await _sendRequest<Map>('cropFromFrame', {
      'frameId': frameId,
      'roi': {
        'xmin': roi.xmin,
        'ymin': roi.ymin,
        'xmax': roi.xmax,
        'ymax': roi.ymax,
      },
    });

    if (result['ok'] != true) {
      final error = result['error'];
      throw StateError('Image crop failed: ${error ?? "unknown error"}');
    }

    final ByteBuffer outBB =
        (result['rgb'] as TransferableTypedData).materialize();
    final Uint8List outRgb = outBB.asUint8List();
    final int ow = result['w'] as int;
    final int oh = result['h'] as int;

    return img.Image.fromBytes(
      width: ow,
      height: oh,
      bytes: outRgb.buffer,
      order: img.ChannelOrder.rgb,
    );
  }

  /// Extracts a rotated square region from an image with bilinear sampling.
  ///
  /// The [src] image is sampled to extract a square patch centered at
  /// ([cx], [cy]) with the specified [size] and rotation angle [theta].
  /// Bilinear interpolation is used for smooth results.
  ///
  /// This is commonly used to align faces to a canonical orientation before
  /// running face mesh detection.
  ///
  /// The [cx] and [cy] parameters are center coordinates in absolute pixels.
  /// The [size] parameter is the output square side length in pixels.
  /// The [theta] parameter is the rotation angle in radians.
  ///
  /// Returns a square [img.Image] of dimensions [size]×[size].
  ///
  /// Throws [ArgumentError] if [size] is not positive.
  /// Throws [StateError] if the worker is not initialized or extraction fails.
  @Deprecated(
    'Will be removed in 5.0.0. Use extractAlignedSquareFromMat instead.',
  )
  Future<img.Image> extractAlignedSquare(
    img.Image src,
    double cx,
    double cy,
    double size,
    double theta,
  ) async {
    if (size <= 0) {
      throw ArgumentError('Size must be positive, got: $size');
    }

    final Uint8List rgb = src.getBytes(order: img.ChannelOrder.rgb);

    final Map result = await _sendRequest<Map>('extract', {
      'w': src.width,
      'h': src.height,
      'rgb': TransferableTypedData.fromList([rgb]),
      'cx': cx,
      'cy': cy,
      'size': size,
      'theta': theta,
    });

    if (result['ok'] != true) {
      final error = result['error'];
      throw StateError('Image extraction failed: ${error ?? "unknown error"}');
    }

    final ByteBuffer outBB =
        (result['rgb'] as TransferableTypedData).materialize();
    final Uint8List outRgb = outBB.asUint8List();
    final int ow = result['w'] as int;
    final int oh = result['h'] as int;

    return img.Image.fromBytes(
      width: ow,
      height: oh,
      bytes: outRgb.buffer,
      order: img.ChannelOrder.rgb,
    );
  }

  /// Extracts a rotated square region from a registered frame.
  ///
  /// Similar to [extractAlignedSquare] but uses a previously registered frame ID
  /// instead of transferring the full image data again.
  @Deprecated(
    'Will be removed in 5.0.0. The image package dependency will be removed.',
  )
  Future<img.Image> extractAlignedSquareWithFrameId(
    int frameId,
    double cx,
    double cy,
    double size,
    double theta,
  ) async {
    if (size <= 0) {
      throw ArgumentError('Size must be positive, got: $size');
    }

    final Map result = await _sendRequest<Map>('extractFromFrame', {
      'frameId': frameId,
      'cx': cx,
      'cy': cy,
      'size': size,
      'theta': theta,
    });

    if (result['ok'] != true) {
      final error = result['error'];
      throw StateError('Image extraction failed: ${error ?? "unknown error"}');
    }

    final ByteBuffer outBB =
        (result['rgb'] as TransferableTypedData).materialize();
    final Uint8List outRgb = outBB.asUint8List();
    final int ow = result['w'] as int;
    final int oh = result['h'] as int;

    return img.Image.fromBytes(
      width: ow,
      height: oh,
      bytes: outRgb.buffer,
      order: img.ChannelOrder.rgb,
    );
  }

  /// Releases all resources held by the worker.
  ///
  /// This kills the background isolate and closes all communication channels.
  /// After calling dispose, this instance cannot be used for processing.
  ///
  /// Any pending requests will be cancelled with an error.
  void dispose() {
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('Worker disposed'));
      }
    }
    _pending.clear();

    _isolate?.kill(priority: Isolate.immediate);
    _receivePort.close();

    _isolate = null;
    _sendPort = null;
    _initialized = false;
  }

  @pragma('vm:entry-point')
  static void _isolateEntry(SendPort mainSendPort) {
    final ReceivePort workerReceivePort = ReceivePort();

    mainSendPort.send(workerReceivePort.sendPort);

    workerReceivePort.listen((message) async {
      if (message is! Map) return;

      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;

      if (id == null || op == null) return;

      try {
        final dynamic result = await _processOperation(op, message);
        mainSendPort.send({'id': id, 'result': result});
      } catch (e, stackTrace) {
        mainSendPort.send({'id': id, 'error': 'Worker error: $e\n$stackTrace'});
      }
    });
  }

  static final Map<int, _RegisteredFrame> _frames = {};

  static Future<dynamic> _processOperation(
    String op,
    Map<dynamic, dynamic> params,
  ) async {
    switch (op) {
      case 'decode':
        return _opDecode(params);
      case 'decodeAndRegister':
        return _opDecodeAndRegister(params);
      case 'tensor':
        return _opTensor(params);
      case 'crop':
        return _opCrop(params);
      case 'extract':
        return _opExtract(params);
      case 'registerFrame':
        return _opRegisterFrame(params);
      case 'releaseFrame':
        return _opReleaseFrame(params);
      case 'tensorFromFrame':
        return _opTensorFromFrame(params);
      case 'cropFromFrame':
        return _opCropFromFrame(params);
      case 'extractFromFrame':
        return _opExtractFromFrame(params);
      default:
        throw ArgumentError('Unknown operation: $op');
    }
  }

  static Map<String, dynamic> _opDecode(Map<dynamic, dynamic> params) {
    final ByteBuffer bb =
        (params['bytes'] as TransferableTypedData).materialize();
    final Uint8List inBytes = bb.asUint8List();

    final img.Image? decoded = img.decodeImage(inBytes);
    if (decoded == null) {
      throw FormatException('Failed to decode image format');
    }

    final Uint8List rgb = decoded.getBytes(order: img.ChannelOrder.rgb);
    return {
      'w': decoded.width,
      'h': decoded.height,
      'rgb': TransferableTypedData.fromList([rgb]),
    };
  }

  static Map<String, dynamic> _opDecodeAndRegister(
    Map<dynamic, dynamic> params,
  ) {
    final int frameId = params['frameId'] as int;
    final ByteBuffer bb =
        (params['bytes'] as TransferableTypedData).materialize();
    final Uint8List inBytes = bb.asUint8List();

    final img.Image? decoded = img.decodeImage(inBytes);
    if (decoded == null) {
      throw FormatException('Failed to decode image format');
    }

    final Uint8List rgb = decoded.getBytes(order: img.ChannelOrder.rgb);
    final img.Image image = img.Image.fromBytes(
      width: decoded.width,
      height: decoded.height,
      bytes: rgb.buffer,
      order: img.ChannelOrder.rgb,
    );

    IsolateWorker._frames[frameId] = _RegisteredFrame(
      decoded.width,
      decoded.height,
      rgb,
      image,
    );

    return {'w': decoded.width, 'h': decoded.height};
  }

  static Map<String, dynamic> _opTensor(Map<dynamic, dynamic> params) {
    final int inW = params['inW'] as int;
    final int inH = params['inH'] as int;
    final int outW = params['outW'] as int;
    final int outH = params['outH'] as int;
    final ByteBuffer rgbBB =
        (params['rgb'] as TransferableTypedData).materialize();
    final Uint8List rgb = rgbBB.asUint8List();

    final img.Image src = img.Image.fromBytes(
      width: inW,
      height: inH,
      bytes: rgb.buffer,
      order: img.ChannelOrder.rgb,
    );

    final ImageTensor result = convertImageToTensor(
      src,
      outW: outW,
      outH: outH,
    );

    return {
      'tensor': TransferableTypedData.fromList([
        result.tensorNHWC.buffer.asUint8List(),
      ]),
      'padding': result.padding,
      'outW': result.width,
      'outH': result.height,
    };
  }

  static Map<String, dynamic> _opCrop(Map<dynamic, dynamic> params) {
    try {
      final int w = params['w'] as int;
      final int h = params['h'] as int;
      final ByteBuffer inBB =
          (params['rgb'] as TransferableTypedData).materialize();
      final Uint8List inRgb = inBB.asUint8List();
      final Map roiMap = params['roi'] as Map;

      final img.Image src = img.Image.fromBytes(
        width: w,
        height: h,
        bytes: inRgb.buffer,
        order: img.ChannelOrder.rgb,
      );

      return _cropAndPackage(src, roiMap);
    } catch (e) {
      return {'ok': false, 'error': e.toString()};
    }
  }

  static Map<String, dynamic> _opExtract(Map<dynamic, dynamic> params) {
    try {
      final int w = params['w'] as int;
      final int h = params['h'] as int;
      final ByteBuffer inBB =
          (params['rgb'] as TransferableTypedData).materialize();
      final Uint8List inRgb = inBB.asUint8List();
      final double cx = (params['cx'] as num).toDouble();
      final double cy = (params['cy'] as num).toDouble();
      final double size = (params['size'] as num).toDouble();
      final double theta = (params['theta'] as num).toDouble();

      final img.Image src = img.Image.fromBytes(
        width: w,
        height: h,
        bytes: inRgb.buffer,
        order: img.ChannelOrder.rgb,
      );

      return _extractAndPackage(src, cx, cy, size, theta);
    } catch (e) {
      return {'ok': false, 'error': e.toString()};
    }
  }

  /// Performs bilinear sampling on RGB8 buffer with direct memory access.
  ///
  /// Optimized version that accesses the raw buffer directly instead of
  /// using getPixel, providing 5-10x faster access for aligned square extraction.
  static void _bilinearSampleRgb8ToBuffer(
    Uint8List srcBuffer,
    int srcWidth,
    int srcHeight,
    double fx,
    double fy,
    Uint8List outBuffer,
    int outOffset,
  ) {
    final int x0 = fx.floor();
    final int y0 = fy.floor();
    final int x1 = x0 + 1;
    final int y1 = y0 + 1;
    final int cx0 = x0.clamp(0, srcWidth - 1);
    final int cx1 = x1.clamp(0, srcWidth - 1);
    final int cy0 = y0.clamp(0, srcHeight - 1);
    final int cy1 = y1.clamp(0, srcHeight - 1);

    final int offset00 = (cy0 * srcWidth + cx0) * 3;
    final int offset10 = (cy0 * srcWidth + cx1) * 3;
    final int offset01 = (cy1 * srcWidth + cx0) * 3;
    final int offset11 = (cy1 * srcWidth + cx1) * 3;

    final double ax = fx - x0;
    final double ay = fy - y0;
    final double ax1 = 1 - ax;
    final double ay1 = 1 - ay;

    final double r0 = srcBuffer[offset00] * ax1 + srcBuffer[offset10] * ax;
    final double g0 =
        srcBuffer[offset00 + 1] * ax1 + srcBuffer[offset10 + 1] * ax;
    final double b0 =
        srcBuffer[offset00 + 2] * ax1 + srcBuffer[offset10 + 2] * ax;
    final double r1 = srcBuffer[offset01] * ax1 + srcBuffer[offset11] * ax;
    final double g1 =
        srcBuffer[offset01 + 1] * ax1 + srcBuffer[offset11 + 1] * ax;
    final double b1 =
        srcBuffer[offset01 + 2] * ax1 + srcBuffer[offset11 + 2] * ax;

    outBuffer[outOffset] = (r0 * ay1 + r1 * ay).round().clamp(0, 255);
    outBuffer[outOffset + 1] = (g0 * ay1 + g1 * ay).round().clamp(0, 255);
    outBuffer[outOffset + 2] = (b0 * ay1 + b1 * ay).round().clamp(0, 255);
  }

  /// Crops an image using normalized ROI coordinates.
  ///
  /// Returns the cropped image with the region defined by normalized
  /// coordinates [xmin], [ymin], [xmax], [ymax] (0.0 to 1.0).
  static img.Image _cropImageWithRoi(
    img.Image src,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
  ) {
    final double W = src.width.toDouble();
    final double H = src.height.toDouble();
    final int x0 = (xmin * W).clamp(0.0, W - 1).toInt();
    final int y0 = (ymin * H).clamp(0.0, H - 1).toInt();
    final int x1 = (xmax * W).clamp(0.0, W).toInt();
    final int y1 = (ymax * H).clamp(0.0, H).toInt();
    final int cw = math.max(1, x1 - x0);
    final int ch = math.max(1, y1 - y0);

    return img.copyCrop(src, x: x0, y: y0, width: cw, height: ch);
  }

  /// Extracts a rotated square region from an image.
  ///
  /// Returns a square image of [size] pixels centered at ([cx], [cy])
  /// rotated by [theta] radians, using bilinear interpolation.
  ///
  /// Optimized to use direct buffer access instead of getPixel/setPixel
  /// for 5-10x faster performance.
  static img.Image _extractAlignedSquareFromImage(
    img.Image src,
    double cx,
    double cy,
    double size,
    double theta,
  ) {
    final int side = math.max(1, size.round());
    final double ct = math.cos(theta);
    final double st = math.sin(theta);

    final img.Image out = img.Image(width: side, height: side);
    final Uint8List srcBuffer = src.buffer.asUint8List();
    final Uint8List outBuffer = out.buffer.asUint8List();
    final int srcWidth = src.width;
    final int srcHeight = src.height;

    int outOffset = 0;
    for (int y = 0; y < side; y++) {
      final double vy = ((y + 0.5) / side - 0.5) * size;
      for (int x = 0; x < side; x++) {
        final double vx = ((x + 0.5) / side - 0.5) * size;
        final double sx = cx + vx * ct - vy * st;
        final double sy = cy + vx * st + vy * ct;

        _bilinearSampleRgb8ToBuffer(
          srcBuffer,
          srcWidth,
          srcHeight,
          sx,
          sy,
          outBuffer,
          outOffset,
        );

        outOffset += 3; // Move to next RGB pixel
      }
    }

    return out;
  }

  /// Helper to crop image and package result.
  static Map<String, dynamic> _cropAndPackage(img.Image src, Map roiMap) {
    final double xmin = (roiMap['xmin'] as num).toDouble();
    final double ymin = (roiMap['ymin'] as num).toDouble();
    final double xmax = (roiMap['xmax'] as num).toDouble();
    final double ymax = (roiMap['ymax'] as num).toDouble();

    final img.Image out = _cropImageWithRoi(src, xmin, ymin, xmax, ymax);
    final Uint8List outRgb = out.getBytes(order: img.ChannelOrder.rgb);

    return {
      'ok': true,
      'w': out.width,
      'h': out.height,
      'rgb': TransferableTypedData.fromList([outRgb]),
    };
  }

  /// Helper to extract aligned square and package result.
  static Map<String, dynamic> _extractAndPackage(
    img.Image src,
    double cx,
    double cy,
    double size,
    double theta,
  ) {
    final img.Image out = _extractAlignedSquareFromImage(
      src,
      cx,
      cy,
      size,
      theta,
    );
    final Uint8List outRgb = out.getBytes(order: img.ChannelOrder.rgb);

    return {
      'ok': true,
      'w': out.width,
      'h': out.height,
      'rgb': TransferableTypedData.fromList([outRgb]),
    };
  }

  static Map<String, dynamic> _opRegisterFrame(Map<dynamic, dynamic> params) {
    final int frameId = params['frameId'] as int;
    final int w = params['w'] as int;
    final int h = params['h'] as int;
    final ByteBuffer bb =
        (params['rgb'] as TransferableTypedData).materialize();
    final Uint8List rgb = bb.asUint8List();

    final img.Image image = img.Image.fromBytes(
      width: w,
      height: h,
      bytes: rgb.buffer,
      order: img.ChannelOrder.rgb,
    );

    IsolateWorker._frames[frameId] = _RegisteredFrame(w, h, rgb, image);
    return {};
  }

  static Map<String, dynamic> _opReleaseFrame(Map<dynamic, dynamic> params) {
    final int frameId = params['frameId'] as int;
    IsolateWorker._frames.remove(frameId);
    return {};
  }

  static Map<String, dynamic> _opTensorFromFrame(Map<dynamic, dynamic> params) {
    final int frameId = params['frameId'] as int;
    final int outW = params['outW'] as int;
    final int outH = params['outH'] as int;

    final _RegisteredFrame? frame = IsolateWorker._frames[frameId];
    if (frame == null) {
      throw StateError('Frame $frameId not found');
    }

    final ImageTensor result = convertImageToTensor(
      frame.image,
      outW: outW,
      outH: outH,
    );

    return {
      'tensor': TransferableTypedData.fromList([
        result.tensorNHWC.buffer.asUint8List(),
      ]),
      'padding': result.padding,
      'outW': result.width,
      'outH': result.height,
    };
  }

  static Map<String, dynamic> _opCropFromFrame(Map<dynamic, dynamic> params) {
    try {
      final int frameId = params['frameId'] as int;
      final Map roiMap = params['roi'] as Map;

      final _RegisteredFrame? frame = IsolateWorker._frames[frameId];
      if (frame == null) {
        return {'ok': false, 'error': 'Frame $frameId not found'};
      }

      return _cropAndPackage(frame.image, roiMap);
    } catch (e) {
      return {'ok': false, 'error': e.toString()};
    }
  }

  static Map<String, dynamic> _opExtractFromFrame(
    Map<dynamic, dynamic> params,
  ) {
    try {
      final int frameId = params['frameId'] as int;
      final double cx = (params['cx'] as num).toDouble();
      final double cy = (params['cy'] as num).toDouble();
      final double size = (params['size'] as num).toDouble();
      final double theta = (params['theta'] as num).toDouble();

      final _RegisteredFrame? frame = IsolateWorker._frames[frameId];
      if (frame == null) {
        return {'ok': false, 'error': 'Frame $frameId not found'};
      }

      return _extractAndPackage(frame.image, cx, cy, size, theta);
    } catch (e) {
      return {'ok': false, 'error': e.toString()};
    }
  }
}

/// Holds a registered frame's data in the worker isolate.
class _RegisteredFrame {
  final int width;
  final int height;
  final Uint8List rgb;
  final img.Image image;

  _RegisteredFrame(this.width, this.height, this.rgb, this.image);
}
