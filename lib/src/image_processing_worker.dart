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
/// final worker = ImageProcessingWorker();
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
class ImageProcessingWorker {
  /// Creates an uninitialized worker; call [initialize] before sending work.
  ImageProcessingWorker();

  Isolate? _isolate;
  SendPort? _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  final Map<int, Completer<dynamic>> _pending = {};
  int _nextId = 0;
  bool _initialized = false;

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
      throw StateError('ImageProcessingWorker already initialized');
    }

    try {
      // Spawn the worker isolate
      _isolate = await Isolate.spawn(
        _isolateEntry,
        _receivePort.sendPort,
        debugName: 'ImageProcessingWorker',
      );

      // Set up response listener
      final Completer<SendPort> initCompleter = Completer<SendPort>();
      late final StreamSubscription subscription;

      subscription = _receivePort.listen((message) {
        // First message is the worker's SendPort
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

        // Subsequent messages are responses to requests
        _handleResponse(message);
      });

      // Wait for worker to send back its SendPort
      _sendPort = await initCompleter.future.timeout(
        const Duration(seconds: 5),
        onTimeout: () {
          subscription.cancel();
          throw TimeoutException('Worker initialization timed out');
        },
      );

      _initialized = true;
    } catch (e) {
      // Clean up on failure
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
        'ImageProcessingWorker not initialized. Call initialize() first.',
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

  /// Crops a region of interest from an image using normalized coordinates.
  ///
  /// The [src] image is cropped to the region defined by [roi], where all
  /// coordinates are normalized (0.0 to 1.0) relative to image dimensions.
  ///
  /// Returns the cropped [img.Image].
  ///
  /// Throws [ArgumentError] if ROI coordinates are invalid.
  /// Throws [StateError] if the worker is not initialized or crop fails.
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

  /// Releases all resources held by the worker.
  ///
  /// This kills the background isolate and closes all communication channels.
  /// After calling dispose, this instance cannot be used for processing.
  ///
  /// Any pending requests will be cancelled with an error.
  void dispose() {
    // Cancel all pending requests
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('Worker disposed'));
      }
    }
    _pending.clear();

    // Kill isolate and close ports
    _isolate?.kill(priority: Isolate.immediate);
    _receivePort.close();

    _isolate = null;
    _sendPort = null;
    _initialized = false;
  }

  // ============================================================================
  // WORKER ISOLATE IMPLEMENTATION
  // ============================================================================

  @pragma('vm:entry-point')
  static void _isolateEntry(SendPort mainSendPort) {
    final ReceivePort workerReceivePort = ReceivePort();

    // Send our SendPort back to main thread
    mainSendPort.send(workerReceivePort.sendPort);

    // Listen for requests
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

  static Future<dynamic> _processOperation(
    String op,
    Map<dynamic, dynamic> params,
  ) async {
    switch (op) {
      case 'decode':
        return _opDecode(params);
      case 'tensor':
        return _opTensor(params);
      case 'crop':
        return _opCrop(params);
      case 'extract':
        return _opExtract(params);
      default:
        throw ArgumentError('Unknown operation: $op');
    }
  }

  // --------------------------------------------------------------------------
  // Operation: decode
  // --------------------------------------------------------------------------

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

  // --------------------------------------------------------------------------
  // Operation: tensor
  // --------------------------------------------------------------------------

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

    // Letterbox resize (aspect-preserving with padding)
    final double s1 = outW / inW;
    final double s2 = outH / inH;
    final double scale = s1 < s2 ? s1 : s2;
    final int newW = (inW * scale).round();
    final int newH = (inH * scale).round();

    final img.Image resized = img.copyResize(
      src,
      width: newW,
      height: newH,
      interpolation: img.Interpolation.linear,
    );

    // Center padding
    final int dx = (outW - newW) ~/ 2;
    final int dy = (outH - newH) ~/ 2;

    final img.Image canvas = img.Image(width: outW, height: outH);
    img.fill(canvas, color: img.ColorRgb8(0, 0, 0));

    for (int y = 0; y < resized.height; y++) {
      for (int x = 0; x < resized.width; x++) {
        final img.Pixel px = resized.getPixel(x, y);
        canvas.setPixel(x + dx, y + dy, px);
      }
    }

    // Convert to normalized tensor [-1.0, 1.0]
    final Float32List tensor = Float32List(outW * outH * 3);
    int k = 0;
    for (int y = 0; y < outH; y++) {
      for (int x = 0; x < outW; x++) {
        final px = canvas.getPixel(x, y);
        tensor[k++] = (px.r / 127.5) - 1.0;
        tensor[k++] = (px.g / 127.5) - 1.0;
        tensor[k++] = (px.b / 127.5) - 1.0;
      }
    }

    // Calculate padding for letterbox removal
    final double padTop = dy / outH;
    final double padBottom = (outH - dy - newH) / outH;
    final double padLeft = dx / outW;
    final double padRight = (outW - dx - newW) / outW;

    return {
      'tensor': TransferableTypedData.fromList([tensor.buffer.asUint8List()]),
      'padding': [padTop, padBottom, padLeft, padRight],
      'outW': outW,
      'outH': outH,
    };
  }

  // --------------------------------------------------------------------------
  // Operation: crop
  // --------------------------------------------------------------------------

  static Map<String, dynamic> _opCrop(Map<dynamic, dynamic> params) {
    try {
      final int w = params['w'] as int;
      final int h = params['h'] as int;
      final ByteBuffer inBB =
          (params['rgb'] as TransferableTypedData).materialize();
      final Uint8List inRgb = inBB.asUint8List();
      final Map roiMap = params['roi'] as Map;

      final double xmin = (roiMap['xmin'] as num).toDouble();
      final double ymin = (roiMap['ymin'] as num).toDouble();
      final double xmax = (roiMap['xmax'] as num).toDouble();
      final double ymax = (roiMap['ymax'] as num).toDouble();

      final img.Image src = img.Image.fromBytes(
        width: w,
        height: h,
        bytes: inRgb.buffer,
        order: img.ChannelOrder.rgb,
      );

      final double W = src.width.toDouble();
      final double H = src.height.toDouble();
      final int x0 = (xmin * W).clamp(0.0, W - 1).toInt();
      final int y0 = (ymin * H).clamp(0.0, H - 1).toInt();
      final int x1 = (xmax * W).clamp(0.0, W).toInt();
      final int y1 = (ymax * H).clamp(0.0, H).toInt();
      final int cw = math.max(1, x1 - x0);
      final int ch = math.max(1, y1 - y0);

      final img.Image out = img.copyCrop(
        src,
        x: x0,
        y: y0,
        width: cw,
        height: ch,
      );
      final Uint8List outRgb = out.getBytes(order: img.ChannelOrder.rgb);

      return {
        'ok': true,
        'w': out.width,
        'h': out.height,
        'rgb': TransferableTypedData.fromList([outRgb]),
      };
    } catch (e) {
      return {'ok': false, 'error': e.toString()};
    }
  }

  // --------------------------------------------------------------------------
  // Operation: extract (aligned square)
  // --------------------------------------------------------------------------

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

      final int side = math.max(1, size.round());
      final double ct = math.cos(theta);
      final double st = math.sin(theta);

      final img.Image out = img.Image(width: side, height: side);

      for (int y = 0; y < side; y++) {
        final double vy = ((y + 0.5) / side - 0.5) * size;
        for (int x = 0; x < side; x++) {
          final double vx = ((x + 0.5) / side - 0.5) * size;
          final double sx = cx + vx * ct - vy * st;
          final double sy = cy + vx * st + vy * ct;
          final img.ColorRgb8 px = _bilinearSampleRgb8(src, sx, sy);
          out.setPixel(x, y, px);
        }
      }

      final Uint8List outRgb = out.getBytes(order: img.ChannelOrder.rgb);

      return {
        'ok': true,
        'w': out.width,
        'h': out.height,
        'rgb': TransferableTypedData.fromList([outRgb]),
      };
    } catch (e) {
      return {'ok': false, 'error': e.toString()};
    }
  }

  static img.ColorRgb8 _bilinearSampleRgb8(
    img.Image src,
    double fx,
    double fy,
  ) {
    final int x0 = fx.floor();
    final int y0 = fy.floor();
    final int x1 = x0 + 1;
    final int y1 = y0 + 1;
    final double ax = fx - x0;
    final double ay = fy - y0;

    final int cx0 = x0.clamp(0, src.width - 1);
    final int cx1 = x1.clamp(0, src.width - 1);
    final int cy0 = y0.clamp(0, src.height - 1);
    final int cy1 = y1.clamp(0, src.height - 1);

    final img.Pixel p00 = src.getPixel(cx0, cy0);
    final img.Pixel p10 = src.getPixel(cx1, cy0);
    final img.Pixel p01 = src.getPixel(cx0, cy1);
    final img.Pixel p11 = src.getPixel(cx1, cy1);

    final double r0 = p00.r * (1 - ax) + p10.r * ax;
    final double g0 = p00.g * (1 - ax) + p10.g * ax;
    final double b0 = p00.b * (1 - ax) + p10.b * ax;

    final double r1 = p01.r * (1 - ax) + p11.r * ax;
    final double g1 = p01.g * (1 - ax) + p11.g * ax;
    final double b1 = p01.b * (1 - ax) + p11.b * ax;

    final int r = (r0 * (1 - ay) + r1 * ay).round().clamp(0, 255);
    final int g = (g0 * (1 - ay) + g1 * ay).round().clamp(0, 255);
    final int b = (b0 * (1 - ay) + b1 * ay).round().clamp(0, 255);

    return img.ColorRgb8(r, g, b);
  }
}
