import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/foundation.dart' show debugPrint;
import 'package:integration_test/integration_test.dart';

Future<bool>? _cached;

/// Whether this browser can initialize the detector on strict WebGPU.
///
/// This is the capability probe the accelerator-flavored tests key off, so a
/// single suite runs everywhere: where WebGPU is present (a workstation
/// browser) they assert real WebGPU compilation, and where it is absent
/// (headless CI runners) they assert the documented WASM-fallback semantics
/// instead. Probing by actually initializing with `strictWebGpu: true`, rather
/// than sniffing `navigator.gpu`, keeps the probe aligned with exactly what
/// the tests then demand of the runtime.
Future<bool> hasWebGpu() {
  return _cached ??= () async {
    bool available;
    try {
      final FaceDetector probe = await FaceDetector.create(
        liteRtAccelerator: 'webgpu',
        strictWebGpu: true,
      );
      await probe.dispose();
      debugPrint('webgpu probe: available');
      available = true;
    } catch (e) {
      debugPrint('webgpu probe: unavailable ($e)');
      available = false;
    }
    // Drive runs do not relay browser-console prints, so also surface the
    // probed branch in the response data the driver writes to
    // build/integration_response_data.json.
    final IntegrationTestWidgetsFlutterBinding binding =
        IntegrationTestWidgetsFlutterBinding.instance;
    binding.reportData = <String, dynamic>{
      ...?binding.reportData,
      'webgpu': available,
    };
    return available;
  }();
}
