// Per-model WebGPU/WASM fallback report for FaceDetector.
//
// FaceDetector compiles five model runners independently, and each can fall
// back from WebGPU to WASM on its own if the browser rejects an op. This test
// initializes the detector with a given backend request and records what each
// runner actually landed on, so the fallback behaviour of a real browser is
// observable rather than inferred.
//
// Run per browser, e.g.:
//   flutter drive \
//     --driver=test_driver/integration_test.dart \
//     --target=integration_test/accelerator_report_test.dart \
//     -d chrome
//
// Results are written to example/benchmark_results/ by the driver's
// responseDataCallback.
//
// It also asserts the invariant the aggregate getter should uphold: whenever
// any runner is on WebGPU, `activeAccelerator` must report 'webgpu', so the
// runtime fallback path stays armed for mixed compile outcomes. On a build
// where the aggregate returns the first non-null runner instead, this
// expectation fails on exactly the browser/model combination that produces the
// mixed state.

import 'dart:convert';

// Default surface: conditionally exports the web implementation off-io, so it
// stays clear of the opencv/ffi native path that will not compile for web.
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // 'auto' and 'webgpu' both attempt GPU; 'wasm' is the control that must
  // report every runner on wasm.
  const List<String> accelerators = <String>['webgpu', 'auto', 'wasm'];

  group('FaceDetector per-model accelerator report', () {
    for (final String requested in accelerators) {
      testWidgets('backend request "$requested"', (tester) async {
        final detector = await FaceDetector.create(
          model: FaceDetectionModel.backCamera,
          useLiteRt: true,
          liteRtAccelerator: requested,
        );

        // Public type does not surface these members; reach them dynamically,
        // matching the existing web benchmark test's pattern.
        final Map<String, String?> report = Map<String, String?>.from(
          (detector as dynamic).acceleratorReport as Map,
        );
        final String? aggregate = (detector as dynamic).activeAccelerator;

        // ignore: avoid_print
        print('\n${'=' * 56}');
        // ignore: avoid_print
        print('ACCELERATOR REPORT  (requested: $requested)');
        // ignore: avoid_print
        print('=' * 56);
        report.forEach((runner, backend) {
          // ignore: avoid_print
          print('  ${runner.padRight(14)} ${backend ?? '(absent)'}');
        });
        // ignore: avoid_print
        print('  ${'aggregate'.padRight(14)} $aggregate');

        final bool anyWebGpu = report.values.contains('webgpu');
        final bool anyWasm = report.values.any((b) => b == 'wasm');
        final bool mixed = anyWebGpu && anyWasm;
        // ignore: avoid_print
        print('  mixed WebGPU/WASM state: $mixed');
        // ignore: avoid_print
        print('=' * 56);

        binding.reportData ??= <String, dynamic>{};
        binding.reportData!['accelerator_report_$requested.json'] =
            jsonDecode(jsonEncode(<String, dynamic>{
          'requested': requested,
          'perRunner': report,
          'aggregate': aggregate,
          'anyWebGpu': anyWebGpu,
          'mixed': mixed,
        }));

        // wasm control: nothing may report webgpu.
        if (requested == 'wasm') {
          expect(anyWebGpu, isFalse,
              reason: 'wasm request should never land any runner on webgpu');
        }

        // The invariant under test: if any runner is on WebGPU, the aggregate
        // must say 'webgpu' so runtime fallback stays enabled.
        if (anyWebGpu) {
          expect(aggregate, 'webgpu',
              reason: 'aggregate activeAccelerator reported "$aggregate" while '
                  'a runner is still on WebGPU: $report. Runtime GPU-error '
                  'fallback is gated on this and would not fire.');
        }

        await detector.dispose();
      });
    }
  });
}
