import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'package:flutter/foundation.dart' show debugPrint;

/// Resolves the requested LiteRT.js accelerator (`'auto'` / `'webgpu'` /
/// `'wasm'`) into a concrete backend for this browser.
///
/// Explicit values pass through untouched. `'auto'` resolves to `'webgpu'`
/// only when the browser is Chromium-based AND `navigator.gpu` yields a
/// hardware adapter; everything else resolves to `'wasm'`.
///
/// The Chromium gate is deliberate: LiteRT.js's WebGPU delegate is developed
/// and tuned against Chrome's Dawn. Firefox 152 (and Safari) expose a WebGPU
/// that compiles and runs these models without ever throwing, but at unusable
/// speed (measured 22x slower than single-threaded WASM SIMD on Firefox 152 /
/// Apple Silicon), so API presence alone must not select it. Callers can
/// still force `'webgpu'` explicitly, and the post-init warmup benchmark in
/// the web `FaceDetector` catches slow-but-functional stacks that slip
/// through.
///
/// The `'auto'` probe runs once per page load and is cached.
Future<String> resolveWebAccelerator(String requested) {
  if (requested != 'auto') return Future<String>.value(requested);
  return _cachedAutoDecision ??= _probeAuto();
}

Future<String>? _cachedAutoDecision;

/// Test hook: clears the cached `'auto'` decision.
void debugResetAcceleratorResolution() {
  _cachedAutoDecision = null;
}

Future<String> _probeAuto() async {
  final String decision = await _probeAutoInner();
  debugPrint(
    "face_detection_tflite: 'auto' accelerator resolved to '$decision'",
  );
  return decision;
}

Future<String> _probeAutoInner() async {
  try {
    final JSObject? nav = globalContext['navigator'] as JSObject?;
    if (nav == null) return 'wasm';

    // Chromium gate. `navigator.userAgentData` only exists on Chromium, and
    // the UA-string check backstops configurations that disable it.
    final String ua = ((nav['userAgent'] as JSString?)?.toDart) ?? '';
    final bool isChromium = nav.has('userAgentData') || ua.contains('Chrome/');
    if (!isChromium) return 'wasm';

    final JSAny? gpu = nav['gpu'];
    if (gpu == null || !gpu.isA<JSObject>()) return 'wasm';

    final JSAny? adapter = await (gpu as JSObject)
        .callMethod<JSPromise<JSAny?>>('requestAdapter'.toJS)
        .toDart
        .timeout(const Duration(seconds: 3), onTimeout: () => null);
    if (adapter == null || !adapter.isA<JSObject>()) return 'wasm';

    // Reject software adapters (headless CI, GPU-less machines): WASM SIMD
    // beats a software-rasterized "GPU" for these models.
    final JSAny? info = (adapter as JSObject)['info'];
    if (info != null && info.isA<JSObject>()) {
      final JSObject i = info as JSObject;
      final String vendor = ((i['vendor'] as JSString?)?.toDart ?? '')
          .toLowerCase();
      final String arch = ((i['architecture'] as JSString?)?.toDart ?? '')
          .toLowerCase();
      if (vendor.contains('swiftshader') ||
          arch.contains('swiftshader') ||
          arch.contains('llvmpipe')) {
        return 'wasm';
      }
    }
    return 'webgpu';
  } catch (_) {
    return 'wasm';
  }
}

/// Logs when LiteRT.js silently compiled a model on a different backend than
/// requested (its compile-time WebGPU-to-WASM fallback), so the swap is
/// visible in the console instead of only in perf traces.
void logCompileFallback({
  required String model,
  required String requested,
  required String actual,
}) {
  if (requested == actual) return;
  debugPrint(
    'face_detection_tflite: $model requested $requested but LiteRT.js '
    'compiled it on $actual (compile-time fallback).',
  );
}
