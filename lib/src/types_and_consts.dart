part of '../face_detection_tflite.dart';

/// Identifies specific facial landmarks returned by face detection.
///
/// Each enum value corresponds to a key facial feature point.
enum FaceLandmarkType {
  leftEye,
  rightEye,
  noseTip,
  mouth,
  leftEyeTragion,
  rightEyeTragion,
}

/// Specifies which face detection model variant to use.
///
/// Different models are optimized for different use cases:
/// - [frontCamera]: Optimized for selfie/front-facing camera (128x128 input)
/// - [backCamera]: Optimized for rear camera with higher resolution (256x256 input)
/// - [shortRange]: Optimized for close-up faces (128x128 input)
/// - [full]: Full-range detection (192x192 input)
/// - [fullSparse]: Full-range with sparse anchors (192x192 input)
enum FaceDetectionModel {
  frontCamera,
  backCamera,
  shortRange,
  full,
  fullSparse,
}

/// Controls which detection features to compute.
///
/// - [fast]: Only bounding boxes and landmarks (fastest)
/// - [standard]: Bounding boxes, landmarks, and 468-point face mesh
/// - [full]: All features including bounding boxes, landmarks, mesh, and iris tracking
enum FaceDetectionMode { fast, standard, full }

/// Performance modes for TensorFlow Lite delegate selection.
///
/// Determines which hardware acceleration delegates are used for inference.
///
/// ## Platform Support
///
/// | Mode | macOS | Linux | Windows | iOS | Android |
/// |------|-------|-------|---------|-----|---------|
/// | [disabled] | ✅ | ✅ | ✅ | ✅ | ✅ |
/// | [xnnpack] | ✅ | ✅ | ❌ | ❌ | ❌ |
/// | [gpu] | ❌ | ❌ | ❌ | ✅ | ⚠️ |
/// | [auto] | ✅ | ✅ | ✅ | ✅ | ✅ |
///
/// ⚠️ = Experimental, may crash on some devices
enum PerformanceMode {
  /// No acceleration delegates (CPU-only, backward compatible).
  ///
  /// - Most compatible (works on all platforms)
  /// - No additional memory overhead
  /// - Baseline performance
  disabled,

  /// XNNPACK delegate for CPU optimization.
  ///
  /// - **Desktop only**: macOS, Linux (crashes on Windows, Android, iOS)
  /// - 2-5x faster than disabled mode
  /// - Uses SIMD vectorization (NEON on ARM, AVX on x86)
  /// - Minimal memory overhead (+2-3MB per interpreter)
  ///
  /// On unsupported platforms, automatically falls back to CPU-only execution.
  xnnpack,

  /// GPU delegate for hardware acceleration.
  ///
  /// - **iOS**: Uses Metal (reliable, recommended)
  /// - **Android**: Uses OpenGL/OpenCL (experimental, may crash on some devices)
  /// - **Desktop**: Not supported (falls back to CPU)
  ///
  /// ## Android GPU Delegate Warning
  ///
  /// The Android GPU delegate has known compatibility issues:
  /// - OpenCL unavailable on many devices (Pixel 6, Android 12+)
  /// - OpenGL ES 3.1+ required for fallback
  /// - Memory issues on some Samsung devices
  /// - Partial op support can cause slower performance than CPU
  ///
  /// Only use on Android if you've tested on your target devices.
  gpu,

  /// Automatically choose best delegate for current platform.
  ///
  /// Current behavior:
  /// - **macOS/Linux**: XNNPACK (2-5x speedup)
  /// - **Windows**: CPU-only (XNNPACK crashes)
  /// - **iOS**: Metal GPU delegate
  /// - **Android**: CPU-only (GPU/XNNPACK unreliable)
  ///
  /// This is the recommended default for cross-platform apps.
  auto,
}

/// Configuration for TensorFlow Lite interpreter performance.
///
/// Controls delegate usage and threading for CPU/GPU acceleration.
///
/// ## Recommended Usage
///
/// For cross-platform apps, use `PerformanceConfig.auto()` (the default):
///
/// ```dart
/// // Auto mode - optimal settings per platform (recommended)
/// final detector = FaceDetector();
/// await detector.initialize(); // Uses PerformanceConfig.auto() by default
/// ```
///
/// ## Platform-Specific Examples
///
/// ```dart
/// // Desktop (macOS/Linux): XNNPACK for 2-5x speedup
/// await detector.initialize(
///   performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
/// );
///
/// // iOS: GPU delegate via Metal (fast and reliable)
/// await detector.initialize(
///   performanceConfig: PerformanceConfig.gpu(),
/// );
///
/// // Android: CPU-only recommended (GPU is experimental)
/// await detector.initialize(
///   performanceConfig: PerformanceConfig.disabled,
/// );
///
/// // Android: GPU delegate (experimental - test on target devices first!)
/// await detector.initialize(
///   performanceConfig: PerformanceConfig.gpu(),
/// );
/// ```
class PerformanceConfig {
  /// Performance mode controlling delegate selection.
  final PerformanceMode mode;

  /// Number of threads for CPU execution.
  ///
  /// - null: Auto-detect optimal count (min(4, Platform.numberOfProcessors))
  /// - 0: No thread pool (single-threaded, good for tiny models)
  /// - 1-8: Explicit thread count
  ///
  /// Diminishing returns after 4 threads for typical models.
  /// Applies to XNNPACK delegate and CPU-only execution.
  final int? numThreads;

  /// Creates a performance configuration.
  ///
  /// Parameters:
  /// - [mode]: Performance mode. Default: [PerformanceMode.auto]
  /// - [numThreads]: Number of threads (null for auto-detection)
  const PerformanceConfig({this.mode = PerformanceMode.auto, this.numThreads});

  /// Creates config with XNNPACK enabled (desktop only).
  ///
  /// XNNPACK provides 2-5x speedup on macOS and Linux.
  /// On unsupported platforms (Windows, Android, iOS), falls back to CPU-only.
  const PerformanceConfig.xnnpack({this.numThreads})
      : mode = PerformanceMode.xnnpack;

  /// Creates config with GPU delegate enabled.
  ///
  /// - **iOS**: Uses Metal (reliable, recommended)
  /// - **Android**: Uses OpenGL/OpenCL (experimental, may crash)
  /// - **Desktop**: Falls back to CPU-only
  const PerformanceConfig.gpu({this.numThreads}) : mode = PerformanceMode.gpu;

  /// Creates config with auto mode (recommended for cross-platform apps).
  ///
  /// Automatically selects the best delegate for each platform:
  /// - macOS/Linux: XNNPACK
  /// - Windows: CPU-only
  /// - iOS: Metal GPU
  /// - Android: CPU-only
  const PerformanceConfig.auto({this.numThreads}) : mode = PerformanceMode.auto;

  /// CPU-only configuration (no delegates, maximum compatibility).
  static const PerformanceConfig disabled = PerformanceConfig(
    mode: PerformanceMode.disabled,
  );
}

/// Pixel format for RGBA output from segmentation masks.
///
/// Use this to match the expected format of your rendering pipeline.
enum PixelFormat {
  /// Red-Green-Blue-Alpha (most common, Flutter default).
  rgba,

  /// Blue-Green-Red-Alpha (OpenCV default on some platforms).
  bgra,

  /// Alpha-Red-Green-Blue (legacy format).
  argb,
}

/// Output format options for isolate-based segmentation to reduce transfer overhead.
///
/// Larger formats provide more precision, smaller formats reduce memory and
/// transfer time when using [FaceDetectorIsolate].
enum IsolateOutputFormat {
  /// Full float32 mask (largest, highest precision).
  float32,

  /// Uint8 grayscale 0-255 (4x smaller than float32).
  uint8,

  /// Binary mask only (smallest, requires threshold).
  binary,
}

/// Error codes for segmentation operations.
///
/// These codes help identify the specific cause of segmentation failures
/// for debugging and error handling.
enum SegmentationError {
  /// Model file not found in assets.
  modelNotFound,

  /// Failed to create TFLite interpreter.
  interpreterCreationFailed,

  /// GPU delegate failed, fell back to CPU.
  delegateFallback,

  /// Image decoding failed.
  imageDecodeFailed,

  /// Image too small (minimum 16x16).
  imageTooSmall,

  /// Unexpected tensor shape from model.
  unexpectedTensorShape,

  /// Inference failed.
  inferenceFailed,

  /// Out of memory during upsampling.
  outOfMemory,
}

/// Exception thrown by segmentation operations.
///
/// Contains a [code] identifying the error type, a human-readable [message],
/// and optionally the underlying [cause] for debugging.
///
/// Example:
/// ```dart
/// try {
///   final mask = await segmenter(imageBytes);
/// } on SegmentationException catch (e) {
///   print('Error ${e.code}: ${e.message}');
///   if (e.cause != null) print('Caused by: ${e.cause}');
/// }
/// ```
class SegmentationException implements Exception {
  /// Error code identifying the type of failure.
  final SegmentationError code;

  /// Human-readable description of the error.
  final String message;

  /// Underlying cause of the error, if available.
  final Object? cause;

  /// Creates a segmentation exception with the given [code], [message], and optional [cause].
  const SegmentationException(this.code, this.message, [this.cause]);

  @override
  String toString() => 'SegmentationException($code): $message';
}

/// Selects which segmentation model variant to use.
///
/// Different models trade off speed, size, and output detail:
/// - [general]: Binary person/background (default, ~244KB, fastest)
/// - [landscape]: Binary person/background optimized for 16:9 video (~244KB)
/// - [multiclass]: 6-class body part segmentation (~16MB, slowest, most detailed)
///
/// ## Platform Support
///
/// All models work on all platforms:
///
/// | Model | macOS | Windows | Linux | iOS | Android |
/// |-------|-------|---------|-------|-----|---------|
/// | general | ✅ | ✅* | ✅* | ✅ | ✅ |
/// | landscape | ✅ | ✅* | ✅* | ✅ | ✅ |
/// | multiclass | ✅ | ✅ | ✅ | ✅ | ✅ |
///
/// *Windows/Linux require custom ops library to be built and bundled.
/// Binary models use the `Convolution2DTransposeBias` custom op which is:
/// - macOS: bundled as dylib
/// - iOS: statically linked via CocoaPods
/// - Android: built via CMake, loaded at runtime
enum SegmentationModel {
  /// Binary segmentation: person vs background.
  ///
  /// Based on MobileNetV3, float16, ~244KB. 256x256 input.
  /// Output: single-channel sigmoid probability (no post-processing needed).
  /// Recommended default for most use cases (background blur, replacement, etc.).
  general,

  /// Binary segmentation optimized for landscape/video frames.
  ///
  /// Based on MobileNetV3, float16, ~244KB. 144x256 input.
  /// Output: single-channel sigmoid probability.
  /// Fastest option for real-time 16:9 video processing.
  ///
  /// Works on all platforms via custom ops integration.
  landscape,

  /// Multiclass segmentation: background, hair, body skin, face skin, clothes, other.
  ///
  /// Based on Vision Transformer, float32, ~16MB. 256x256 input.
  /// Output: 6-channel raw logits (softmax applied in post-processing).
  /// Use when you need per-class masks (hair coloring, virtual try-on, etc.).
  /// Returns [MulticlassSegmentationMask] with per-class accessors.
  ///
  /// Works on all platforms. Largest model but most detailed output.
  multiclass,
}

/// Configuration for segmentation operations.
///
/// Controls model selection, delegate selection, output format, and memory limits.
///
/// ## Presets
///
/// - [SegmentationConfig.safe]: CPU-only, 1024 max output (recommended for Android)
/// - [SegmentationConfig.performance]: Auto delegate, 2048 max output
///
/// ## Example
///
/// ```dart
/// final config = SegmentationConfig(
///   performanceConfig: PerformanceConfig.auto(),
///   maxOutputSize: 1920,
///   maxOutputSize: 1920,
/// );
/// final segmenter = await SelfieSegmentation.create(config: config);
/// ```
class SegmentationConfig {
  /// Which segmentation model to use.
  ///
  /// - [SegmentationModel.general]: Binary person/background (~244KB, default)
  /// - [SegmentationModel.landscape]: Binary for 16:9 video (~244KB)
  /// - [SegmentationModel.multiclass]: 6-class body parts (~16MB)
  final SegmentationModel model;

  /// Performance configuration for TFLite delegate.
  final PerformanceConfig performanceConfig;

  /// Maximum output dimension for upsampled masks.
  ///
  /// If original image exceeds this, upsampled mask is capped to prevent OOM.
  /// Default 2048. Set to 0 for unlimited (use with caution on mobile).
  final int maxOutputSize;

  /// Whether to validate model metadata on load.
  ///
  /// When true, checks that the loaded model has expected input/output shapes.
  /// Disable only if using custom models with known-different shapes.
  final bool validateModel;

  /// Whether to use IsolateInterpreter for inference.
  ///
  /// When true (default), inference runs via IsolateInterpreter which avoids
  /// blocking the main thread but adds ~2-5ms overhead per call due to:
  /// - Nested list structure creation for isolate boundary crossing
  /// - Serialization/deserialization overhead
  /// - Output tensor flattening
  ///
  /// When false, uses direct interpreter invoke which is faster but blocks
  /// the calling thread during inference (~90-100ms). Best for:
  /// - Background isolate processing (already off main thread)
  /// - Maximum throughput when latency is acceptable
  /// - Real-time video processing where every ms counts
  ///
  /// **Benchmark results (macOS, XNNPACK):**
  /// - With isolate: ~99ms mean
  /// - Without isolate: ~95ms mean (4-5% faster)
  final bool useIsolate;

  /// Creates a segmentation configuration.
  ///
  /// Parameters:
  /// - [model]: Which model variant to use. Default: [SegmentationModel.general].
  /// - [performanceConfig]: TFLite delegate settings. Default: auto.
  /// - [maxOutputSize]: Maximum dimension for upsampled output. Default: 2048.
  /// - [validateModel]: Whether to validate model on load. Default: true.
  /// - [useIsolate]: Whether to use IsolateInterpreter. Default: true.
  const SegmentationConfig({
    this.model = SegmentationModel.general,
    this.performanceConfig = const PerformanceConfig.auto(),
    this.maxOutputSize = 2048,
    this.validateModel = true,
    this.useIsolate = true,
  });

  /// Safe defaults: CPU-only, limited output size, binary model.
  ///
  /// Recommended for Android where GPU delegates may be unstable.
  static const SegmentationConfig safe = SegmentationConfig(
    performanceConfig: PerformanceConfig.disabled,
    maxOutputSize: 1024,
  );

  /// Performance defaults: auto delegate, larger output, binary model.
  ///
  /// Good for iOS (Metal) and desktop (XNNPACK).
  static const SegmentationConfig performance = SegmentationConfig(
    performanceConfig: PerformanceConfig.auto(),
    maxOutputSize: 2048,
  );

  /// Maximum speed: auto delegate, no isolate overhead, binary model.
  ///
  /// Best for real-time video processing or background isolate usage.
  /// Blocks calling thread during inference (~90-100ms).
  static const SegmentationConfig fast = SegmentationConfig(
    performanceConfig: PerformanceConfig.auto(),
    maxOutputSize: 2048,
    useIsolate: false,
  );
}

/// A segmentation probability mask indicating foreground vs background.
///
/// Contains per-pixel probabilities (0.0-1.0) where higher values indicate
/// greater likelihood that the pixel belongs to a person/foreground.
///
/// ## Memory Management
///
/// Mask data is stored as Float32List. For large images, prefer:
/// - Use [toBinary] or [toUint8] to reduce memory by 4x
/// - Use [upsample] with maxSize parameter to limit output
/// - Process masks promptly and let GC collect
///
/// ## Thread Safety
///
/// SegmentationMask is immutable. The underlying data buffer is
/// exposed as an unmodifiable view.
///
/// ## Example
///
/// ```dart
/// final mask = await segmenter(imageBytes);
///
/// // Get model-resolution mask values
/// final probability = mask.at(128, 128);
///
/// // Upsample to original image size (capped at 1080)
/// final fullMask = mask.upsample(maxSize: 1080);
///
/// // Convert to binary for compositing
/// final binary = fullMask.toBinary(threshold: 0.5);
/// ```
class SegmentationMask {
  final Float32List _data;

  /// Mask width in pixels.
  final int width;

  /// Mask height in pixels.
  final int height;

  /// Original source image width.
  final int originalWidth;

  /// Original source image height.
  final int originalHeight;

  /// Padding applied during letterboxing [top, bottom, left, right].
  /// All zeros if no letterboxing was used.
  final List<double> padding;

  const SegmentationMask._({
    required Float32List data,
    required this.width,
    required this.height,
    required this.originalWidth,
    required this.originalHeight,
    required this.padding,
  }) : _data = data;

  /// Creates a segmentation mask with validation.
  ///
  /// Throws [ArgumentError] if data length doesn't match width*height.
  factory SegmentationMask({
    required Float32List data,
    required int width,
    required int height,
    required int originalWidth,
    required int originalHeight,
    List<double> padding = const [0.0, 0.0, 0.0, 0.0],
  }) {
    if (data.length != width * height) {
      throw ArgumentError(
        'Data length ${data.length} != width*height ${width * height}',
      );
    }
    return SegmentationMask._(
      data: Float32List.fromList(data),
      width: width,
      height: height,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      padding: List.unmodifiable(padding),
    );
  }

  /// Returns a copy of the raw probability data (row-major order).
  ///
  /// Values are in the range [0.0, 1.0] where higher values indicate
  /// greater foreground probability.
  ///
  /// Note: This returns a defensive copy to maintain immutability.
  /// For performance-critical code that needs direct access without copying,
  /// use [at] for individual pixel access.
  Float32List get data => Float32List.fromList(_data);

  /// Returns probability at (x, y) in mask coordinates.
  ///
  /// Returns 0.0 for out-of-bounds coordinates (safe access).
  double at(int x, int y) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0.0;
    return _data[y * width + x];
  }

  /// Upsamples mask to target dimensions using bilinear interpolation.
  ///
  /// [targetWidth], [targetHeight]: Target dimensions. If null, uses original image size.
  /// [maxSize]: Maximum dimension (width or height). Caps output to prevent OOM.
  ///
  /// Handles unletterboxing automatically if padding was applied during inference.
  ///
  /// Example:
  /// ```dart
  /// // Upsample to original size, capped at 1080px
  /// final fullMask = mask.upsample(maxSize: 1080);
  /// ```
  SegmentationMask upsample({
    int? targetWidth,
    int? targetHeight,
    int maxSize = 2048,
  }) {
    final tw = targetWidth ?? originalWidth;
    final th = targetHeight ?? originalHeight;

    final maxDim = math.max(tw, th);
    final scale = maxSize > 0 && maxDim > maxSize ? maxSize / maxDim : 1.0;
    final finalW = (tw * scale).round();
    final finalH = (th * scale).round();

    Float32List sourceData = _data;
    int sourceW = width;
    int sourceH = height;

    final pt = padding[0], pb = padding[1], pl = padding[2], pr = padding[3];
    if (pt > 0 || pb > 0 || pl > 0 || pr > 0) {
      final validX0 = (pl * width).round();
      final validY0 = (pt * height).round();
      final validX1 = ((1.0 - pr) * width).round();
      final validY1 = ((1.0 - pb) * height).round();
      final validW = validX1 - validX0;
      final validH = validY1 - validY0;

      if (validW > 0 && validH > 0) {
        sourceData = Float32List(validW * validH);
        for (int y = 0; y < validH; y++) {
          for (int x = 0; x < validW; x++) {
            sourceData[y * validW + x] =
                _data[(y + validY0) * width + (x + validX0)];
          }
        }
        sourceW = validW;
        sourceH = validH;
      }
    }

    final result = Float32List(finalW * finalH);
    final scaleX = sourceW / finalW;
    final scaleY = sourceH / finalH;

    for (int y = 0; y < finalH; y++) {
      final srcY = y * scaleY;
      final y0 = srcY.floor().clamp(0, sourceH - 1);
      final y1 = (y0 + 1).clamp(0, sourceH - 1);
      final yFrac = srcY - y0;

      for (int x = 0; x < finalW; x++) {
        final srcX = x * scaleX;
        final x0 = srcX.floor().clamp(0, sourceW - 1);
        final x1 = (x0 + 1).clamp(0, sourceW - 1);
        final xFrac = srcX - x0;

        final v00 = sourceData[y0 * sourceW + x0];
        final v10 = sourceData[y0 * sourceW + x1];
        final v01 = sourceData[y1 * sourceW + x0];
        final v11 = sourceData[y1 * sourceW + x1];

        final v0 = v00 * (1 - xFrac) + v10 * xFrac;
        final v1 = v01 * (1 - xFrac) + v11 * xFrac;
        result[y * finalW + x] = v0 * (1 - yFrac) + v1 * yFrac;
      }
    }

    return SegmentationMask._(
      data: result,
      width: finalW,
      height: finalH,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      padding: const [0.0, 0.0, 0.0, 0.0],
    );
  }

  /// Converts to 8-bit grayscale mask (0-255).
  ///
  /// More memory efficient than float32 for storage/transfer.
  /// Values are scaled linearly: 0.0 → 0, 1.0 → 255.
  Uint8List toUint8() {
    final result = Uint8List(width * height);
    for (int i = 0; i < _data.length; i++) {
      result[i] = (_data[i].clamp(0.0, 1.0) * 255).round();
    }
    return result;
  }

  /// Converts to binary mask (0 or 255) using threshold.
  ///
  /// [threshold]: Probability threshold (default 0.5).
  /// Values >= threshold become 255, others become 0.
  ///
  /// Example:
  /// ```dart
  /// final binary = mask.toBinary(threshold: 0.6);
  /// // binary[i] is 255 if probability >= 0.6, else 0
  /// ```
  Uint8List toBinary({double threshold = 0.5}) {
    final result = Uint8List(width * height);
    for (int i = 0; i < _data.length; i++) {
      result[i] = _data[i] >= threshold ? 255 : 0;
    }
    return result;
  }

  /// Converts to RGBA image with configurable colors.
  ///
  /// [foreground]: Color for person pixels (default white opaque).
  /// [background]: Color for background pixels (default transparent).
  /// [format]: Pixel format (default RGBA).
  /// [threshold]: Binary threshold (default 0.5). Use negative for soft alpha blending.
  ///
  /// Example:
  /// ```dart
  /// // Binary mask with red foreground, transparent background
  /// final rgba = mask.toRgba(
  ///   foreground: 0xFFFF0000,
  ///   background: 0x00000000,
  ///   threshold: 0.5,
  /// );
  ///
  /// // Soft alpha mask (smooth edges)
  /// final softRgba = mask.toRgba(threshold: -1);
  /// ```
  Uint8List toRgba({
    int foreground = 0xFFFFFFFF,
    int background = 0x00000000,
    PixelFormat format = PixelFormat.rgba,
    double threshold = 0.5,
  }) {
    final result = Uint8List(width * height * 4);
    final fg = _unpackColor(foreground, format);
    final bg = _unpackColor(background, format);

    for (int i = 0; i < _data.length; i++) {
      final prob = _data[i].clamp(0.0, 1.0);
      final offset = i * 4;

      if (threshold < 0) {
        for (int c = 0; c < 4; c++) {
          result[offset + c] = (fg[c] * prob + bg[c] * (1 - prob)).round();
        }
      } else {
        final color = prob >= threshold ? fg : bg;
        for (int c = 0; c < 4; c++) {
          result[offset + c] = color[c];
        }
      }
    }
    return result;
  }

  /// Unpacks a color integer to [R, G, B, A] based on format.
  List<int> _unpackColor(int color, PixelFormat format) {
    switch (format) {
      case PixelFormat.rgba:
        return [
          (color >> 24) & 0xFF,
          (color >> 16) & 0xFF,
          (color >> 8) & 0xFF,
          color & 0xFF,
        ];
      case PixelFormat.bgra:
        return [
          (color >> 8) & 0xFF,
          (color >> 16) & 0xFF,
          (color >> 24) & 0xFF,
          color & 0xFF,
        ];
      case PixelFormat.argb:
        return [
          (color >> 16) & 0xFF,
          (color >> 8) & 0xFF,
          color & 0xFF,
          (color >> 24) & 0xFF,
        ];
    }
  }

  /// Serialization for isolate transfer.
  Map<String, dynamic> toMap() => {
        'data': _data.toList(),
        'width': width,
        'height': height,
        'originalWidth': originalWidth,
        'originalHeight': originalHeight,
        'padding': padding,
      };

  /// Creates a mask from a serialized map (isolate deserialization).
  /// Creates a SegmentationMask from a serialized map.
  ///
  /// Handles different data formats from isolate transfer:
  /// - 'float32': Direct float32 data (default)
  /// - 'uint8': 8-bit grayscale (converted to float32)
  /// - 'binary': Binary mask (converted to float32)
  factory SegmentationMask.fromMap(Map<String, dynamic> map) {
    final width = map['width'] as int;
    final height = map['height'] as int;
    final dataFormat = map['dataFormat'] as String? ?? 'float32';
    final rawData = map['data'] as List;

    Float32List data;
    switch (dataFormat) {
      case 'float32':
        data = Float32List.fromList(rawData.cast<double>());
        break;
      case 'uint8':
        final uint8List = rawData.cast<int>();
        data = Float32List(uint8List.length);
        for (int i = 0; i < uint8List.length; i++) {
          data[i] = uint8List[i] / 255.0;
        }
        break;
      case 'binary':
        final binaryList = rawData.cast<int>();
        data = Float32List(binaryList.length);
        for (int i = 0; i < binaryList.length; i++) {
          data[i] = binaryList[i] == 255 ? 1.0 : 0.0;
        }
        break;
      default:
        throw ArgumentError('Unknown data format: $dataFormat');
    }

    return SegmentationMask(
      data: data,
      width: width,
      height: height,
      originalWidth: map['originalWidth'] as int,
      originalHeight: map['originalHeight'] as int,
      padding:
          (map['padding'] as List?)?.cast<double>() ?? [0.0, 0.0, 0.0, 0.0],
    );
  }

  @override
  String toString() =>
      'SegmentationMask(${width}x$height, original: ${originalWidth}x$originalHeight)';
}

/// Extended segmentation mask with per-class probabilities.
///
/// Only returned when using [SegmentationModel.multiclass].
/// Inherits all [SegmentationMask] methods — the base [data] field contains
/// the combined person probability, same as other models.
///
/// Access individual class masks via [classMask] or the convenience getters.
///
/// ## Example
/// ```dart
/// final seg = await SelfieSegmentation.create(
///   config: SegmentationConfig(model: SegmentationModel.multiclass),
/// );
/// final mask = await seg(imageBytes);
/// if (mask is MulticlassSegmentationMask) {
///   final hair = mask.hairMask;
///   final face = mask.faceSkinMask;
/// }
/// ```
class MulticlassSegmentationMask extends SegmentationMask {
  /// Per-class probabilities after softmax, shape [width * height * 6].
  /// Channel order: background(0), hair(1), bodySkin(2), faceSkin(3), clothes(4), other(5).
  final Float32List _classData;

  // ignore: use_super_parameters -- super._() is a named private constructor
  MulticlassSegmentationMask._({
    required Float32List data,
    required int width,
    required int height,
    required int originalWidth,
    required int originalHeight,
    required List<double> padding,
    required Float32List classData,
  })  : _classData = classData,
        super._(
          data: data,
          width: width,
          height: height,
          originalWidth: originalWidth,
          originalHeight: originalHeight,
          padding: padding,
        );

  /// Creates a multiclass segmentation mask with validation and defensive copies.
  factory MulticlassSegmentationMask({
    required Float32List data,
    required int width,
    required int height,
    required int originalWidth,
    required int originalHeight,
    List<double> padding = const [0.0, 0.0, 0.0, 0.0],
    required Float32List classData,
  }) {
    if (data.length != width * height) {
      throw ArgumentError(
        'Data length ${data.length} != width*height ${width * height}',
      );
    }
    if (classData.length != width * height * 6) {
      throw ArgumentError(
        'ClassData length ${classData.length} != width*height*6 ${width * height * 6}',
      );
    }
    return MulticlassSegmentationMask._(
      data: Float32List.fromList(data),
      width: width,
      height: height,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      padding: List.unmodifiable(padding),
      classData: Float32List.fromList(classData),
    );
  }

  /// Returns a single-channel probability mask for the given [SegmentationClass] index.
  ///
  /// Values are in [0.0, 1.0] representing per-pixel probability for that class.
  /// Returns a new [Float32List] each call (defensive copy).
  Float32List classMask(int classIndex) {
    if (classIndex < 0 || classIndex > 5) {
      throw RangeError.range(classIndex, 0, 5, 'classIndex');
    }
    final int numPixels = width * height;
    final result = Float32List(numPixels);
    for (int i = 0; i < numPixels; i++) {
      result[i] = _classData[i * 6 + classIndex];
    }
    return result;
  }

  /// Hair probability mask.
  Float32List get hairMask => classMask(SegmentationClass.hair);

  /// Body skin probability mask.
  Float32List get bodySkinMask => classMask(SegmentationClass.bodySkin);

  /// Face skin probability mask.
  Float32List get faceSkinMask => classMask(SegmentationClass.faceSkin);

  /// Clothes probability mask.
  Float32List get clothesMask => classMask(SegmentationClass.clothes);

  /// Other (accessories, etc.) probability mask.
  Float32List get otherMask => classMask(SegmentationClass.other);

  /// Background probability mask.
  Float32List get backgroundMask => classMask(SegmentationClass.background);

  @override
  String toString() =>
      'MulticlassSegmentationMask(${width}x$height, original: ${originalWidth}x$originalHeight, 6 classes)';
}

/// Result combining face detection and segmentation from parallel processing.
///
/// This class bundles the results of running face detection and selfie
/// segmentation simultaneously in separate isolates. Use with
/// [FaceDetectorIsolate.detectFacesWithSegmentationFromMat] for optimal
/// performance when both features are needed.
///
/// ## Example
///
/// ```dart
/// final detector = await FaceDetectorIsolate.spawn(withSegmentation: true);
/// final result = await detector.detectFacesWithSegmentationFromMat(mat);
///
/// print('Found ${result.faces.length} faces');
/// print('Mask: ${result.segmentationMask?.width}x${result.segmentationMask?.height}');
/// print('Total time: ${result.totalTimeMs}ms (parallel processing)');
/// ```
///
/// ## Performance
///
/// When using parallel processing, [totalTimeMs] represents the maximum of
/// [detectionTimeMs] and [segmentationTimeMs], rather than their sum. This
/// typically results in 40-50% faster processing compared to sequential calls.
class DetectionWithSegmentationResult {
  /// Detected faces with landmarks, mesh, and iris data.
  final List<Face> faces;

  /// Segmentation mask separating foreground (person) from background.
  ///
  /// Null if segmentation was disabled or failed during processing.
  final SegmentationMask? segmentationMask;

  /// Time taken for face detection in milliseconds.
  final int detectionTimeMs;

  /// Time taken for segmentation in milliseconds.
  ///
  /// Zero if segmentation was not run.
  final int segmentationTimeMs;

  /// Creates a result combining detection and segmentation outputs.
  const DetectionWithSegmentationResult({
    required this.faces,
    this.segmentationMask,
    required this.detectionTimeMs,
    required this.segmentationTimeMs,
  });

  /// Total processing time in milliseconds.
  ///
  /// For parallel processing, this is the maximum of [detectionTimeMs] and
  /// [segmentationTimeMs], representing the wall-clock time for the operation.
  int get totalTimeMs => detectionTimeMs > segmentationTimeMs
      ? detectionTimeMs
      : segmentationTimeMs;

  /// Serializes this result to a map for isolate transfer.
  Map<String, dynamic> toMap() => {
        'faces': faces.map((f) => f.toMap()).toList(),
        if (segmentationMask != null)
          'segmentationMask': segmentationMask!.toMap(),
        'detectionTimeMs': detectionTimeMs,
        'segmentationTimeMs': segmentationTimeMs,
      };

  /// Creates a result from a serialized map.
  factory DetectionWithSegmentationResult.fromMap(Map<String, dynamic> map) {
    return DetectionWithSegmentationResult(
      faces: (map['faces'] as List)
          .map((m) => Face.fromMap(Map<String, dynamic>.from(m as Map)))
          .toList(),
      segmentationMask: map['segmentationMask'] != null
          ? SegmentationMask.fromMap(
              Map<String, dynamic>.from(map['segmentationMask'] as Map),
            )
          : null,
      detectionTimeMs: map['detectionTimeMs'] as int,
      segmentationTimeMs: map['segmentationTimeMs'] as int,
    );
  }

  @override
  String toString() =>
      'DetectionWithSegmentationResult(faces: ${faces.length}, '
      'mask: ${segmentationMask != null ? "${segmentationMask!.width}x${segmentationMask!.height}" : "null"}, '
      'time: ${totalTimeMs}ms)';
}

/// Connections between eye contour landmarks for rendering the visible eyeball outline.
///
/// These define which of the 71 eye contour points should be connected with lines
/// to form the visible eye shape (eyelids). The connections form the outline of the
/// visible eyeball by connecting the first 15 eye contour landmarks.
///
/// Based on MediaPipe's iris rendering configuration.
///
/// Example usage:
/// ```dart
/// for (final connection in eyeLandmarkConnections) {
///   final p1 = eye.contour[connection[0]];
///   final p2 = eye.contour[connection[1]];
///   // Draw line from p1 to p2
/// }
/// ```
const List<List<int>> eyeLandmarkConnections = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [9, 10],
  [10, 11],
  [11, 12],
  [12, 13],
  [13, 14],
  [0, 9],
  [8, 14],
];

/// Number of eye contour points that form the visible eyeball outline.
///
/// The first 15 points of the 71-point eye contour represent the visible
/// eyelid outline. The remaining 56 points are used for eyebrows and
/// additional tracking halos around the eye region.
const int kMaxEyeLandmark = 15;

/// A point with x, y, and optional z coordinates.
///
/// Used to represent landmarks with optional depth information.
/// The x and y coordinates are in absolute pixel positions relative to the original image.
/// The z coordinate represents relative depth (scale-dependent) when 3D computation is enabled.
///
/// When [z] is null, this represents a 2D point. When [z] is non-null, it represents
/// a 3D point with depth information.
class Point {
  /// The x-coordinate in absolute pixels.
  final double x;

  /// The y-coordinate in absolute pixels.
  final double y;

  /// The z-coordinate representing relative depth, or null for 2D points.
  ///
  /// This is a scale-dependent depth value. The magnitude depends on the face size
  /// and alignment used during detection. Negative values indicate points closer to
  /// the camera, positive values indicate points further away.
  ///
  /// Will be null for 2D-only landmarks (such as face detection keypoints).
  /// Face mesh and iris landmarks always include z-coordinates.
  final double? z;

  /// Creates a point with the given x, y, and optional z coordinates.
  const Point(this.x, this.y, [this.z]);

  /// Whether this point has depth information (z-coordinate).
  ///
  /// Returns true if z-coordinate is non-null, false otherwise.
  bool get is3D => z != null;

  @override
  String toString() => z != null ? 'Point($x, $y, $z)' : 'Point($x, $y)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Point &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y &&
          z == other.z;

  @override
  int get hashCode => Object.hash(x, y, z);

  /// Converts this point to a map for isolate serialization.
  Map<String, dynamic> toMap() => {'x': x, 'y': y, if (z != null) 'z': z};

  /// Creates a point from a map (isolate deserialization).
  factory Point.fromMap(Map<String, dynamic> map) =>
      Point(map['x'] as double, map['y'] as double, map['z'] as double?);
}

/// A 468-point face mesh with optional depth information.
///
/// Encapsulates the MediaPipe face mesh data. Each point has x and y coordinates
/// in absolute pixels, and an optional z coordinate representing depth.
/// 3D coordinates are always computed for face mesh landmarks.
///
/// The mesh contains 468 points following MediaPipe's canonical face mesh topology,
/// providing detailed geometry for facial features including eyes, eyebrows, nose,
/// mouth, and face contours.
///
/// Example:
/// ```dart
/// final FaceMesh? mesh = face.mesh;
/// if (mesh != null) {
///   // Access points
///   final points = mesh.points;
///   print('Nose tip: (${points[1].x}, ${points[1].y}, ${points[1].z})');
///
///   // Direct indexed access
///   final noseTip = mesh[1];
///   if (noseTip.z != null) {
///     print('Depth available: ${noseTip.z}');
///   }
/// }
/// ```
class FaceMesh {
  final List<Point> _points;

  /// Creates a face mesh from 468 points.
  FaceMesh(this._points) : assert(_points.length == kMeshPoints);

  /// The 468 mesh points with depth information.
  ///
  /// Each point has x, y, and z coordinates. The z coordinate represents
  /// relative depth and is always computed for face mesh landmarks.
  List<Point> get points => _points;

  /// Returns the point at the given index.
  ///
  /// Example: `mesh[1]` returns the nose tip.
  Point operator [](int index) => _points[index];

  /// The number of points in the mesh (always 468).
  int get length => _points.length;

  @override
  String toString() => 'FaceMesh(${_points.length} points)';

  /// Converts this mesh to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'points': _points.map((p) => p.toMap()).toList(),
      };

  /// Creates a face mesh from a map (isolate deserialization).
  factory FaceMesh.fromMap(Map<String, dynamic> map) =>
      FaceMesh((map['points'] as List).map((p) => Point.fromMap(p)).toList());
}

/// Comprehensive eye tracking data including iris center, iris contour, and eye mesh.
///
/// Each eye contains:
/// - An iris center point for gaze tracking
/// - Four iris contour points outlining the iris boundary
/// - Eye mesh landmarks covering the entire eye region (71 points including eyelid contour)
///
/// All coordinates are in absolute pixel positions relative to the original image.
///
/// **Naming clarity:**
/// - [irisCenter]: Iris center point for gaze tracking
/// - [irisContour]: 4 points outlining the iris boundary (the colored part of the eye)
/// - [contour]: 15 points outlining the eyelid (visible eyeball outline)
/// - [mesh]: 71 points covering the entire eye region (eyelids, eyebrows, tracking halos)
///
/// The eye mesh landmarks provide detailed geometry for the entire eye region including
/// eyelids, eye corners, and surrounding area. These can be useful for blink detection,
/// eye openness estimation, and advanced eye tracking applications.
///
/// See also:
/// - [EyePair] for accessing both eyes' data
class Eye {
  /// Center point of the iris in absolute pixel coordinates for gaze tracking.
  final Point irisCenter;

  /// Four points outlining the iris boundary in absolute pixel coordinates.
  ///
  /// These points form the contour of the iris and can be used to estimate
  /// the iris size and shape.
  final List<Point> irisContour;

  /// Complete eye mesh with 71 landmark points in absolute pixel coordinates.
  ///
  /// These 71 points form a detailed mesh of the entire eye region including
  /// eyelids, eye corners, eyebrows, and surrounding area. They provide comprehensive
  /// geometry information about the eye beyond just the iris.
  ///
  /// **Structure:**
  /// - First 15 points: Visible eyelid outline (use [contour] to access these)
  /// - Remaining 56 points: Eyebrow landmarks and tracking halos
  ///
  /// Useful for:
  /// - Eyelid position tracking
  /// - Blink detection
  /// - Eye openness estimation
  /// - Eyebrow tracking
  /// - Detailed eye region analysis
  ///
  /// See also:
  /// - [contour] for just the 15-point eyelid outline
  /// - [eyeLandmarkConnections] for connecting the eyelid points
  final List<Point> mesh;

  /// Pre-computed eyelid contour for O(1) repeated access.
  final List<Point> _contourCache;

  /// Creates an eye with iris center point, iris contour, and eye mesh landmarks.
  ///
  /// This const constructor is preserved for backward compatibility.
  /// For optimized construction with pre-computed contour, use [Eye.optimized].
  const Eye({
    required this.irisCenter,
    required this.irisContour,
    this.mesh = const <Point>[],
  }) : _contourCache = const <Point>[];

  /// Internal constructor with pre-computed contour cache.
  Eye._withContour({
    required this.irisCenter,
    required this.irisContour,
    required this.mesh,
    required List<Point> contour,
  }) : _contourCache = contour;

  /// Creates an eye with pre-computed contour for optimal repeated access.
  ///
  /// This factory pre-computes the [contour] during construction, avoiding
  /// repeated sublist allocation on each access. Use this constructor when
  /// creating Eyes programmatically (e.g., from model output).
  factory Eye.optimized({
    required Point irisCenter,
    required List<Point> irisContour,
    List<Point> mesh = const <Point>[],
  }) {
    final contour = mesh.length >= kMaxEyeLandmark
        ? mesh.sublist(0, kMaxEyeLandmark)
        : mesh;
    return Eye._withContour(
      irisCenter: irisCenter,
      irisContour: irisContour,
      mesh: mesh,
      contour: contour,
    );
  }

  /// The visible eyelid contour (first 15 points of the mesh).
  ///
  /// These 15 points form the outline of the visible eyeball (upper and lower eyelids).
  /// Use [eyeLandmarkConnections] to determine which points to connect with lines
  /// when rendering the eyelid outline.
  ///
  /// The remaining points in [mesh] (indices 15-70) represent eyebrows and
  /// additional tracking halos around the eye region.
  ///
  /// Example:
  /// ```dart
  /// final eyelidPoints = eye.contour;
  /// for (final connection in eyeLandmarkConnections) {
  ///   final p1 = eyelidPoints[connection[0]];
  ///   final p2 = eyelidPoints[connection[1]];
  ///   canvas.drawLine(p1, p2, paint);
  /// }
  /// ```
  List<Point> get contour => _contourCache.isNotEmpty
      ? _contourCache
      : (mesh.length >= kMaxEyeLandmark
          ? mesh.sublist(0, kMaxEyeLandmark)
          : mesh);

  /// Converts this eye to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'irisCenter': irisCenter.toMap(),
        'irisContour': irisContour.map((p) => p.toMap()).toList(),
        'mesh': mesh.map((p) => p.toMap()).toList(),
      };

  /// Creates an eye from a map (isolate deserialization).
  ///
  /// Uses [Eye.optimized] to pre-compute the contour cache.
  factory Eye.fromMap(Map<String, dynamic> map) => Eye.optimized(
        irisCenter: Point.fromMap(map['irisCenter']),
        irisContour:
            (map['irisContour'] as List).map((p) => Point.fromMap(p)).toList(),
        mesh: (map['mesh'] as List).map((p) => Point.fromMap(p)).toList(),
      );
}

/// Eye tracking data for both eyes including iris and eye mesh landmarks.
///
/// Contains comprehensive eye tracking data for the left and right eyes. Each eye includes:
/// - Iris center point for gaze tracking
/// - Four iris contour points (iris boundary)
/// - Eye mesh landmarks (71 points covering the entire eye region)
///
/// Individual eye data may be null if not detected or if called with a detection
/// mode that doesn't include iris tracking.
///
/// Only available when using [FaceDetectionMode.full]. Returns null for
/// [FaceDetectionMode.fast] and [FaceDetectionMode.standard].
///
/// Example:
/// ```dart
/// final faces = await detector.detectFaces(imageBytes, mode: FaceDetectionMode.full);
/// final eyes = faces.first.eyes;
/// if (eyes != null) {
///   final leftIrisCenter = eyes.leftEye?.irisCenter;
///   final rightIrisContour = eyes.rightEye?.irisContour;
///   final leftEyeMesh = eyes.leftEye?.mesh;
/// }
/// ```
///
/// See also:
/// - [Eye] for the structure of individual eye data
class EyePair {
  /// The left eye data, or null if not detected.
  final Eye? leftEye;

  /// The right eye data, or null if not detected.
  final Eye? rightEye;

  /// Creates an eye pair with optional left and right eye data.
  const EyePair({this.leftEye, this.rightEye});

  /// Converts this eye pair to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        if (leftEye != null) 'leftEye': leftEye!.toMap(),
        if (rightEye != null) 'rightEye': rightEye!.toMap(),
      };

  /// Creates an eye pair from a map (isolate deserialization).
  factory EyePair.fromMap(Map<String, dynamic> map) => EyePair(
        leftEye: map['leftEye'] != null ? Eye.fromMap(map['leftEye']) : null,
        rightEye: map['rightEye'] != null ? Eye.fromMap(map['rightEye']) : null,
      );
}

/// Facial landmark points with convenient named access.
///
/// Provides 6 key facial feature points in pixel coordinates with both
/// named property access and map-like access for backwards compatibility.
///
/// All coordinates are in absolute pixel positions relative to the original image.
///
/// Example:
/// ```dart
/// final landmarks = face.landmarks;
/// print('Left eye: (${landmarks.leftEye?.x}, ${landmarks.leftEye?.y})');
/// print('Nose: (${landmarks.noseTip?.x}, ${landmarks.noseTip?.y})');
///
/// // Map-like access also works for backwards compatibility
/// final leftEye = landmarks[FaceLandmarkType.leftEye];
/// ```
class FaceLandmarks {
  final Map<FaceLandmarkType, Point> _landmarks;

  /// Creates facial landmarks from a map of landmark types to points.
  const FaceLandmarks(this._landmarks);

  /// Left eye center point in pixel coordinates.
  Point? get leftEye => _landmarks[FaceLandmarkType.leftEye];

  /// Right eye center point in pixel coordinates.
  Point? get rightEye => _landmarks[FaceLandmarkType.rightEye];

  /// Nose tip point in pixel coordinates.
  Point? get noseTip => _landmarks[FaceLandmarkType.noseTip];

  /// Mouth center point in pixel coordinates.
  Point? get mouth => _landmarks[FaceLandmarkType.mouth];

  /// Left eye tragion point in pixel coordinates.
  ///
  /// The tragion is the notch just above the ear canal opening.
  Point? get leftEyeTragion => _landmarks[FaceLandmarkType.leftEyeTragion];

  /// Right eye tragion point in pixel coordinates.
  ///
  /// The tragion is the notch just above the ear canal opening.
  Point? get rightEyeTragion => _landmarks[FaceLandmarkType.rightEyeTragion];

  /// Access landmark by type (backwards compatible with map access).
  ///
  /// Example:
  /// ```dart
  /// final leftEye = landmarks[FaceLandmarkType.leftEye];
  /// ```
  Point? operator [](FaceLandmarkType type) => _landmarks[type];

  /// All landmark points as an iterable (backwards compatible with map.values).
  ///
  /// Example:
  /// ```dart
  /// for (final point in landmarks.values) {
  ///   print('(${point.x}, ${point.y})');
  /// }
  /// ```
  Iterable<Point> get values => _landmarks.values;

  /// All available landmark types in this detection.
  Iterable<FaceLandmarkType> get keys => _landmarks.keys;

  /// Returns all landmarks as an unmodifiable map.
  ///
  /// Use this when you need explicit Map type for compatibility.
  Map<FaceLandmarkType, Point> toMap() => Map.unmodifiable(_landmarks);

  /// Converts landmarks to a serializable map for isolate transfer.
  Map<String, dynamic> toSerializableMap() => {
        for (final entry in _landmarks.entries)
          entry.key.name: entry.value.toMap(),
      };

  /// Creates landmarks from a serializable map (isolate deserialization).
  factory FaceLandmarks.fromSerializableMap(Map<String, dynamic> map) {
    final landmarks = <FaceLandmarkType, Point>{};
    for (final entry in map.entries) {
      final type = FaceLandmarkType.values.firstWhere(
        (t) => t.name == entry.key,
      );
      landmarks[type] = Point.fromMap(entry.value);
    }
    return FaceLandmarks(landmarks);
  }
}

/// Face bounding box with corner points in pixel coordinates.
///
/// Represents a rectangular bounding box around a detected face with convenient
/// access to corner points, dimensions, and center coordinates.
///
/// All coordinates are in absolute pixel positions relative to the original image.
///
/// Example:
/// ```dart
/// final boundingBox = face.boundingBox;
/// print('Width: ${boundingBox.width}, Height: ${boundingBox.height}');
/// print('Top-left corner: (${boundingBox.topLeft.x}, ${boundingBox.topLeft.y})');
/// print('Center: (${boundingBox.center.x}, ${boundingBox.center.y})');
/// ```
class BoundingBox {
  /// Top-left corner point in absolute pixel coordinates.
  final Point topLeft;

  /// Top-right corner point in absolute pixel coordinates.
  final Point topRight;

  /// Bottom-right corner point in absolute pixel coordinates.
  final Point bottomRight;

  /// Bottom-left corner point in absolute pixel coordinates.
  final Point bottomLeft;

  /// Creates a bounding box with four corner points.
  ///
  /// Points should be in order: top-left, top-right, bottom-right, bottom-left.
  const BoundingBox({
    required this.topLeft,
    required this.topRight,
    required this.bottomRight,
    required this.bottomLeft,
  });

  /// The four corner points as a list in order: top-left, top-right,
  /// bottom-right, bottom-left.
  ///
  /// Useful for iteration or when you need all corners at once.
  List<Point> get corners => [topLeft, topRight, bottomRight, bottomLeft];

  /// Width of the bounding box in pixels.
  double get width => topRight.x - topLeft.x;

  /// Height of the bounding box in pixels.
  double get height => bottomLeft.y - topLeft.y;

  /// Center point of the bounding box in absolute pixel coordinates.
  Point get center => Point(
        (topLeft.x + topRight.x + bottomRight.x + bottomLeft.x) / 4,
        (topLeft.y + topRight.y + bottomRight.y + bottomLeft.y) / 4,
      );

  /// Converts this bounding box to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'topLeft': topLeft.toMap(),
        'topRight': topRight.toMap(),
        'bottomRight': bottomRight.toMap(),
        'bottomLeft': bottomLeft.toMap(),
      };

  /// Creates a bounding box from a map (isolate deserialization).
  factory BoundingBox.fromMap(Map<String, dynamic> map) => BoundingBox(
        topLeft: Point.fromMap(map['topLeft']),
        topRight: Point.fromMap(map['topRight']),
        bottomRight: Point.fromMap(map['bottomRight']),
        bottomLeft: Point.fromMap(map['bottomLeft']),
      );
}

/// Outputs for a single detected face.
///
/// [boundingBox] is the face bounding box in pixel coordinates.
/// [landmarks] provides convenient access to 6 key facial landmarks (eyes, nose, mouth).
/// [mesh] contains 468 facial landmarks as pixel coordinates.
/// [eyes] contains iris center, iris contour, and eye mesh landmarks for both eyes.
class Face {
  final Detection _detection;

  /// The 468-point face mesh with optional depth information.
  ///
  /// The mesh provides convenient access to points with x, y coordinates and
  /// an optional z coordinate representing depth when 3D computation is enabled.
  ///
  /// The 468 points follow MediaPipe's canonical face mesh topology, providing
  /// detailed geometry for facial features including eyes, eyebrows, nose, mouth,
  /// and face contours.
  ///
  /// This is null when [FaceDetector.detectFaces] is called with [FaceDetectionMode.fast].
  /// Use [FaceDetectionMode.standard] or [FaceDetectionMode.full] to populate mesh data.
  ///
  /// Example:
  /// ```dart
  /// final FaceMesh? mesh = face.mesh;
  /// if (mesh != null) {
  ///   final points = mesh.points;
  ///   for (final point in points) {
  ///     if (point.is3D) {
  ///       print('Point with depth: (${point.x}, ${point.y}, ${point.z})');
  ///     } else {
  ///       print('Point: (${point.x}, ${point.y})');
  ///     }
  ///   }
  /// }
  /// ```
  ///
  /// See also:
  /// - [FaceMesh] for the mesh class documentation
  /// - [kMeshPoints] for the expected mesh point count (468)
  /// - [eyes] for iris and eye mesh landmarks
  final FaceMesh? mesh;

  /// Raw iris and eye mesh landmark points with depth information.
  ///
  /// Contains landmarks for both eyes (left eye data followed by right eye data).
  /// The iris model outputs 76 points per eye in this order:
  /// - 71 eye mesh landmarks (detailed eye region geometry)
  /// - 5 iris keypoints (1 center + 4 contour points)
  ///
  /// Total: 152 points (76 per eye × 2 eyes).
  ///
  /// The iris center is not guaranteed to be at a fixed index; use [eyes] to
  /// access a parsed center derived from the 5 iris keypoints.
  ///
  /// Each point contains `x`, `y`, and `z` coordinates where:
  /// - `x` and `y` are absolute pixel positions in the original image
  /// - `z` represents relative depth (scale-dependent)
  ///
  /// This list is empty when [FaceDetector.detectFaces] is called with
  /// [FaceDetectionMode.fast] or [FaceDetectionMode.standard]. Use
  /// [FaceDetectionMode.full] to enable iris tracking.
  ///
  /// For a more convenient structured API, use the [eyes] getter instead,
  /// which returns an [EyePair] with separate left/right eye data including
  /// parsed iris center, iris contour, and eye mesh landmarks.
  ///
  /// Example:
  /// ```dart
  /// final faces = await detector.detectFaces(imageBytes, mode: FaceDetectionMode.full);
  /// if (faces.isNotEmpty && faces.first.irisPoints.isNotEmpty) {
  ///   // Use structured API for easier access
  ///   final eyes = faces.first.eyes;
  ///   final leftIrisCenter = eyes?.leftEye?.irisCenter;
  ///   final leftEyeMesh = eyes?.leftEye?.mesh;
  ///
  ///   // Access 3D depth information
  ///   if (leftIrisCenter.is3D) {
  ///     print('Iris depth: ${leftIrisCenter.z}');
  ///   }
  /// }
  /// ```
  final List<Point> irisPoints;

  /// The dimensions of the original source image.
  ///
  /// This size is used internally to convert normalized coordinates to pixel
  /// coordinates for [boundingBox], [landmarks], [mesh], and [eyes].
  ///
  /// All coordinate data in [Face] is already scaled to these dimensions,
  /// so users typically don't need to use this field directly unless performing
  /// custom coordinate transformations.
  final Size originalSize;

  /// Cached eye pair computation for O(1) repeated access.
  late final EyePair? _cachedEyes = _computeEyes();

  /// Creates a face detection result with bounding box, landmarks, and optional mesh/eye data.
  ///
  /// This constructor is typically called internally by [FaceDetector.detectFaces].
  /// Most users should not need to construct [Face] instances directly.
  ///
  /// The [detection] contains the bounding box and coarse facial keypoints.
  /// The [mesh] contains the 468-point face mesh with 3D coordinates (null if not computed).
  /// The [irisPoints] contains iris and eye mesh keypoints with 3D coordinates (empty if not computed).
  /// The [originalSize] specifies the dimensions of the source image for coordinate mapping.
  Face({
    required Detection detection,
    required this.mesh,
    required List<Point> irises,
    required this.originalSize,
  })  : _detection = detection,
        irisPoints = irises;

  /// Parses raw iris and eye contour points into a structured Eye object.
  ///
  /// The iris landmark model outputs 76 points per eye in this order:
  /// - First 71 points: eye mesh landmarks (detailed eye region geometry)
  /// - Last 5 points: iris landmarks (center + 4 contour points)
  ///
  /// For backward compatibility, also handles legacy format (5 points only).
  ///
  /// Identifies the iris center as the point with minimum sum of squared
  /// distances to all other iris points, and treats the remaining 4 as contour.
  ///
  /// Returns null if the input contains fewer than 5 points.
  static Eye? _parseIris(List<Point> points) {
    if (points.length < 5) return null;

    List<Point> eyeMesh;
    List<Point> irisPoints;

    if (points.length == 76) {
      eyeMesh = points.sublist(0, 71);
      irisPoints = points.sublist(71, 76);
    } else if (points.length > 5) {
      final irisStart = points.length - 5;
      eyeMesh = points.sublist(0, irisStart);
      irisPoints = points.sublist(irisStart);
    } else {
      eyeMesh = const <Point>[];
      irisPoints = points;
    }
    double cx = 0, cy = 0;
    for (int i = 0; i < 5; i++) {
      cx += irisPoints[i].x;
      cy += irisPoints[i].y;
    }
    cx /= 5;
    cy /= 5;

    int centerIdx = 0;
    double minDist = double.infinity;
    for (int i = 0; i < 5; i++) {
      final dx = irisPoints[i].x - cx;
      final dy = irisPoints[i].y - cy;
      final dist = dx * dx + dy * dy;
      if (dist < minDist) {
        minDist = dist;
        centerIdx = i;
      }
    }

    final center = irisPoints[centerIdx];
    final contour = <Point>[];
    for (int i = 0; i < 5; i++) {
      if (i != centerIdx) contour.add(irisPoints[i]);
    }

    return Eye.optimized(
      irisCenter: center,
      irisContour: contour,
      mesh: eyeMesh,
    );
  }

  /// Comprehensive eye tracking data for both eyes.
  ///
  /// Returns an [EyePair] containing left and right eye data with:
  /// - Iris center points
  /// - Iris contour boundaries (4 points per iris)
  /// - Eye mesh landmarks (71 points per eye covering the entire eye region)
  ///
  /// Individual eyes may be null if not detected.
  ///
  /// Returns null when [FaceDetector.detectFaces] is called with
  /// [FaceDetectionMode.fast] or [FaceDetectionMode.standard]. Use
  /// [FaceDetectionMode.full] to enable eye tracking.
  ///
  /// For raw iris point data, use [irisPoints] instead.
  ///
  /// Example:
  /// ```dart
  /// final eyes = face.eyes;
  /// final leftIrisCenter = eyes?.leftEye?.irisCenter;
  /// final leftIrisContour = eyes?.leftEye?.irisContour;
  /// final leftContour = eyes?.leftEye?.contour;
  /// final rightIrisCenter = eyes?.rightEye?.irisCenter;
  /// ```
  EyePair? get eyes => _cachedEyes;

  /// Internal computation for eyes getter, called once via lazy initialization.
  EyePair? _computeEyes() {
    if (irisPoints.isEmpty) return null;

    Eye? leftEye;
    Eye? rightEye;

    if (irisPoints.length == 152) {
      leftEye = _parseIris(irisPoints.sublist(0, 76));
      rightEye = _parseIris(irisPoints.sublist(76, 152));
    } else if (irisPoints.length == 76) {
      leftEye = _parseIris(irisPoints);
    } else if (irisPoints.length == 10) {
      leftEye = _parseIris(irisPoints.sublist(0, 5));
      rightEye = _parseIris(irisPoints.sublist(5, 10));
    } else if (irisPoints.length > 10 && irisPoints.length.isEven) {
      final int pointsPerEye = irisPoints.length ~/ 2;
      leftEye = _parseIris(irisPoints.sublist(0, pointsPerEye));
      rightEye = _parseIris(irisPoints.sublist(pointsPerEye));
    } else if (irisPoints.length >= 5) {
      leftEye = _parseIris(irisPoints);
    }

    if (leftEye == null && rightEye == null) return null;

    return EyePair(leftEye: leftEye, rightEye: rightEye);
  }

  /// The face bounding box in pixel coordinates.
  ///
  /// Provides convenient access to corner points, dimensions, and center
  /// of the bounding box. Use [BoundingBox.topLeft], [BoundingBox.topRight],
  /// [BoundingBox.bottomRight], [BoundingBox.bottomLeft] to access individual
  /// corners, or [BoundingBox.width], [BoundingBox.height], and
  /// [BoundingBox.center] for dimensions and center point.
  ///
  /// Example:
  /// ```dart
  /// final boundingBox = face.boundingBox;
  /// print('Face at (${boundingBox.center.x}, ${boundingBox.center.y})');
  /// print('Size: ${boundingBox.width} x ${boundingBox.height}');
  /// ```
  BoundingBox get boundingBox {
    final RectF r = _detection.boundingBox;
    final double w = originalSize.width.toDouble();
    final double h = originalSize.height.toDouble();
    return BoundingBox(
      topLeft: Point(r.xmin * w, r.ymin * h),
      topRight: Point(r.xmax * w, r.ymin * h),
      bottomRight: Point(r.xmax * w, r.ymax * h),
      bottomLeft: Point(r.xmin * w, r.ymax * h),
    );
  }

  /// Facial landmark positions in pixel coordinates.
  ///
  /// Returns a [FaceLandmarks] object with convenient named access to key
  /// facial features. Use named properties like [FaceLandmarks.leftEye],
  /// [FaceLandmarks.rightEye], [FaceLandmarks.noseTip], etc. for cleaner code.
  ///
  /// When iris tracking is enabled ([FaceDetectionMode.full]), the left and
  /// right eye landmarks are automatically replaced with the precise iris centers
  /// for improved accuracy.
  ///
  /// Example:
  /// ```dart
  /// final landmarks = face.landmarks;
  /// final leftEye = landmarks.leftEye;
  /// final noseTip = landmarks.noseTip;
  /// print('Left eye: (${leftEye?.x}, ${leftEye?.y})');
  /// ```
  ///
  /// For backwards compatibility, you can still use map-like access:
  /// ```dart
  /// final leftEye = landmarks[FaceLandmarkType.leftEye];
  /// for (final point in landmarks.values) { ... }
  /// ```
  FaceLandmarks get landmarks {
    final Map<FaceLandmarkType, Point> landmarkMap = _detection.landmarks;

    final EyePair? eyeData = eyes;
    if (eyeData != null) {
      if (eyeData.leftEye?.irisCenter != null) {
        landmarkMap[FaceLandmarkType.leftEye] = eyeData.leftEye!.irisCenter;
      }
      if (eyeData.rightEye?.irisCenter != null) {
        landmarkMap[FaceLandmarkType.rightEye] = eyeData.rightEye!.irisCenter;
      }
    }

    return FaceLandmarks(landmarkMap);
  }

  /// Converts this face to a map for isolate serialization.
  ///
  /// Serializes all face data including detection, mesh, iris points, and
  /// original image size. Use [Face.fromMap] to reconstruct.
  Map<String, dynamic> toMap() => {
        'detection': _detection.toMap(),
        if (mesh != null) 'mesh': mesh!.toMap(),
        'irisPoints': irisPoints.map((p) => p.toMap()).toList(),
        'originalSize': {
          'width': originalSize.width,
          'height': originalSize.height,
        },
      };

  /// Creates a face from a map (isolate deserialization).
  ///
  /// Reconstructs a Face object from data serialized via [toMap].
  factory Face.fromMap(Map<String, dynamic> map) => Face(
        detection: Detection.fromMap(map['detection']),
        mesh: map['mesh'] != null ? FaceMesh.fromMap(map['mesh']) : null,
        irises:
            (map['irisPoints'] as List).map((p) => Point.fromMap(p)).toList(),
        originalSize: Size(
          map['originalSize']['width'],
          map['originalSize']['height'],
        ),
      );
}

/// The expected number of landmark points in a complete face mesh.
///
/// MediaPipe's face mesh model produces exactly 468 points covering facial
/// features including eyes, eyebrows, nose, mouth, and face contours.
///
/// Use this constant to validate mesh output or split concatenated mesh data:
/// ```dart
/// assert(meshPoints.length == kMeshPoints); // Validate single face
/// final faces = meshPoints.length ~/ kMeshPoints; // Count faces in batch
/// ```
const int kMeshPoints = 468;

const _modelNameBack = 'face_detection_back.tflite';
const _modelNameFront = 'face_detection_front.tflite';
const _modelNameShort = 'face_detection_short_range.tflite';
const _modelNameFull = 'face_detection_full_range.tflite';
const _modelNameFullSparse = 'face_detection_full_range_sparse.tflite';
const _faceLandmarkModel = 'face_landmark.tflite';
const _irisLandmarkModel = 'iris_landmark.tflite';
const double _rawScoreLimit = 80.0;
const double _minScore = 0.5;
const double _minSuppressionThreshold = 0.3;

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

/// Holds an aligned face crop and metadata used for downstream landmark models.
///
/// An [AlignedFace] represents a face that has been rotated, scaled, and
/// translated so that the eyes are horizontal and the face roughly fills the
/// crop. Downstream models such as [FaceLandmark] and [IrisLandmark] expect
/// this normalized orientation.
@Deprecated('Will be removed in 5.0.0. Use AlignedFaceFromMat instead.')
class AlignedFace {
  /// X coordinate of the face center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of the face center in absolute pixel coordinates.
  final double cy;

  /// Length of the square crop edge in absolute pixels.
  final double size;

  /// Rotation applied to align the face, in radians.
  final double theta;

  /// The aligned face crop image provided to landmark models.
  final img.Image faceCrop;

  /// Creates an aligned face crop with pixel-based center, size, rotation,
  /// and the cropped [faceCrop] image ready for landmark inference.
  AlignedFace({
    required this.cx,
    required this.cy,
    required this.size,
    required this.theta,
    required this.faceCrop,
  });
}

/// Aligned face crop data holder for OpenCV-based processing.
///
/// Similar to [AlignedFace] but holds a cv.Mat instead of img.Image.
/// Used internally by the OpenCV-accelerated detection pipeline.
class AlignedFaceFromMat {
  /// X coordinate of the face center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of the face center in absolute pixel coordinates.
  final double cy;

  /// Length of the square crop edge in absolute pixels.
  final double size;

  /// Rotation applied to align the face, in radians.
  final double theta;

  /// The aligned face crop as cv.Mat. Caller must dispose when done.
  final cv.Mat faceCrop;

  /// Creates an aligned face crop with cv.Mat instead of img.Image.
  AlignedFaceFromMat({
    required this.cx,
    required this.cy,
    required this.size,
    required this.theta,
    required this.faceCrop,
  });
}

/// Axis-aligned rectangle with normalized coordinates.
///
/// Values are expressed as fractions of the original image dimensions
/// (0.0 - 1.0). Utilities are provided to scale and expand the rectangle.
class RectF {
  /// Minimum X and Y plus maximum X and Y extents.
  final double xmin, ymin, xmax, ymax;

  /// Creates a normalized rectangle given its minimum and maximum extents.
  const RectF(this.xmin, this.ymin, this.xmax, this.ymax);

  /// Rectangle width.
  double get w => xmax - xmin;

  /// Rectangle height.
  double get h => ymax - ymin;

  /// Returns a rectangle scaled independently in X and Y.
  RectF scale(double sx, double sy) =>
      RectF(xmin * sx, ymin * sy, xmax * sx, ymax * sy);

  /// Expands the rectangle by [frac] in all directions, keeping the same center.
  RectF expand(double frac) {
    final double cx = (xmin + xmax) * 0.5;
    final double cy = (ymin + ymax) * 0.5;
    final double hw = (w * (1.0 + frac)) * 0.5;
    final double hh = (h * (1.0 + frac)) * 0.5;
    return RectF(cx - hw, cy - hh, cx + hw, cy + hh);
  }

  /// Converts this rect to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
      };

  /// Creates a rect from a map (isolate deserialization).
  factory RectF.fromMap(Map<String, dynamic> map) => RectF(
        map['xmin'] as double,
        map['ymin'] as double,
        map['xmax'] as double,
        map['ymax'] as double,
      );
}

/// Raw detection output from the face detector containing the bounding box and keypoints.
class Detection {
  /// Normalized bounding box for the face.
  final RectF boundingBox;

  /// Confidence score for the detection.
  final double score;

  /// Flattened landmark coordinates `[x0, y0, x1, y1, ...]` normalized 0-1.
  final List<double> keypointsXY;

  /// Original image dimensions used to denormalize landmarks.
  final Size? imageSize;

  /// Creates a detection with normalized geometry and optional source size.
  Detection({
    required this.boundingBox,
    required this.score,
    required this.keypointsXY,
    this.imageSize,
  });

  /// Convenience accessor for `[keypointsXY]` by index.
  double operator [](int i) => keypointsXY[i];

  /// Returns facial landmarks in pixel coordinates keyed by landmark type.
  Map<FaceLandmarkType, Point> get landmarks {
    final Size? sz = imageSize;
    if (sz == null) {
      throw StateError(
        'Detection.imageSize is null; cannot produce pixel landmarks.',
      );
    }
    final double w = sz.width.toDouble(), h = sz.height.toDouble();
    final Map<FaceLandmarkType, Point> map = <FaceLandmarkType, Point>{};
    for (final FaceLandmarkType idx in FaceLandmarkType.values) {
      final double xn = keypointsXY[idx.index * 2];
      final double yn = keypointsXY[idx.index * 2 + 1];
      map[idx] = Point(xn * w, yn * h);
    }
    return map;
  }

  /// Converts this detection to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'boundingBox': boundingBox.toMap(),
        'score': score,
        'keypointsXY': keypointsXY,
        if (imageSize != null)
          'imageSize': {'width': imageSize!.width, 'height': imageSize!.height},
      };

  /// Creates a detection from a map (isolate deserialization).
  factory Detection.fromMap(Map<String, dynamic> map) => Detection(
        boundingBox: RectF.fromMap(map['boundingBox']),
        score: map['score'] as double,
        keypointsXY: (map['keypointsXY'] as List).cast<double>(),
        imageSize: map['imageSize'] != null
            ? Size(map['imageSize']['width'], map['imageSize']['height'])
            : null,
      );
}

/// Image tensor plus padding metadata used to undo letterboxing.
class ImageTensor {
  /// NHWC float tensor normalized to [-1, 1] expected by MediaPipe models.
  final Float32List tensorNHWC;

  /// Padding fractions `[top, bottom, left, right]` applied during resize.
  final List<double> padding;

  /// Target width and height passed to the model.
  final int width, height;

  /// Creates an image tensor paired with the padding used during resize.
  ImageTensor(this.tensorNHWC, this.padding, this.width, this.height);
}

/// Rotation-aware region of interest for cropped eye landmarks.
class AlignedRoi {
  /// X coordinate of ROI center in absolute pixel coordinates.
  final double cx;

  /// Y coordinate of ROI center in absolute pixel coordinates.
  final double cy;

  /// Square ROI size in absolute pixels.
  final double size;

  /// Rotation applied to align the ROI, in radians.
  final double theta;

  /// Creates a rotation-aware region of interest in absolute pixel coordinates
  /// used to crop around the eyes for iris landmark detection.
  const AlignedRoi(this.cx, this.cy, this.size, this.theta);
}

/// Decoded detection box and keypoints straight from the TFLite model.
class DecodedBox {
  /// Normalized bounding box for a detected face.
  final RectF boundingBox;

  /// Flattened list of normalized keypoints `[x0, y0, ...]`.
  final List<double> keypointsXY;

  /// Constructs a decoded detection with its normalized bounding box and
  /// flattened landmark coordinates output by the face detector.
  DecodedBox(this.boundingBox, this.keypointsXY);
}
