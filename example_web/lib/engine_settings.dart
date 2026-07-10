import 'package:flutter/material.dart';
import 'package:flutter_litert/flutter_litert.dart' show Precision;

/// Web inference-engine selection for the example, mirroring the flutter_litert
/// example_web dialog: switch between the LiteRT.js Interpreter and LiteRT Next
/// CompiledModel, choose a backend, and (for CompiledModel on WebGPU) precision.
///
/// Maps directly onto `FaceDetector.create(useCompiledModel:, liteRtAccelerator:,
/// strictWebGpu:, precision:)`.
class EngineSettings {
  const EngineSettings({
    this.useCompiledModel = true,
    this.accelerator = 'auto',
    this.strictWebGpu = false,
    this.precision = Precision.fp16,
  });

  /// CompiledModel (true) vs LiteRT.js Interpreter (false).
  final bool useCompiledModel;

  /// Requested backend: `'auto'` | `'webgpu'` | `'wasm'`.
  final String accelerator;

  /// CompiledModel WebGPU-only (no WASM fallback), so a WebGPU compile failure
  /// surfaces instead of silently dropping to WASM.
  final bool strictWebGpu;

  /// CompiledModel WebGPU precision. Cosmetic on the web (LiteRT.js ignores it),
  /// kept for parity with the flutter_litert example.
  final Precision precision;

  EngineSettings copyWith({
    bool? useCompiledModel,
    String? accelerator,
    bool? strictWebGpu,
    Precision? precision,
  }) =>
      EngineSettings(
        useCompiledModel: useCompiledModel ?? this.useCompiledModel,
        accelerator: accelerator ?? this.accelerator,
        strictWebGpu: strictWebGpu ?? this.strictWebGpu,
        precision: precision ?? this.precision,
      );

  /// Short human-readable label, e.g. `CompiledModel · WebGPU+WASM · fp16`.
  String get label {
    if (!useCompiledModel) {
      final String acc = accelerator == 'auto'
          ? 'Auto'
          : (accelerator == 'webgpu' ? 'WebGPU' : 'WASM');
      return 'Interpreter · $acc';
    }
    if (accelerator == 'wasm') return 'CompiledModel · WASM';
    final String acc = strictWebGpu ? 'WebGPU' : 'WebGPU+WASM';
    final String p = precision == Precision.fp16 ? 'fp16' : 'fp32';
    return 'CompiledModel · $acc · $p';
  }
}

/// Reusable engine-settings controls for the example screens' settings dialogs.
/// Emits a new [EngineSettings] on any change; the parent re-initializes the
/// detector.
class EngineSettingsControls extends StatelessWidget {
  const EngineSettingsControls({
    super.key,
    required this.settings,
    required this.onChanged,
  });

  final EngineSettings settings;
  final ValueChanged<EngineSettings> onChanged;

  @override
  Widget build(BuildContext context) {
    final EngineSettings s = settings;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Inference engine',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 6),
        SegmentedButton<bool>(
          segments: const [
            ButtonSegment<bool>(value: false, label: Text('Interpreter')),
            ButtonSegment<bool>(value: true, label: Text('CompiledModel')),
          ],
          selected: {s.useCompiledModel},
          onSelectionChanged: (sel) =>
              onChanged(s.copyWith(useCompiledModel: sel.first)),
        ),
        const SizedBox(height: 8),
        if (!s.useCompiledModel) ..._interpreter(s) else ..._compiledModel(s),
      ],
    );
  }

  List<Widget> _interpreter(EngineSettings s) => [
        Wrap(
          spacing: 6,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            const Text('Accelerator:'),
            for (final (label, value) in const <(String, String)>[
              ('Auto', 'auto'),
              ('WebGPU', 'webgpu'),
              ('WASM', 'wasm'),
            ])
              ChoiceChip(
                label: Text(label),
                selected: s.accelerator == value,
                onSelected: (_) => onChanged(
                    s.copyWith(accelerator: value, strictWebGpu: false)),
              ),
          ],
        ),
        const Padding(
          padding: EdgeInsets.only(top: 4),
          child: Text(
            'LiteRT.js interpreter. WebGPU auto-falls back to WASM.',
            style: TextStyle(fontSize: 11),
          ),
        ),
      ];

  List<Widget> _compiledModel(EngineSettings s) {
    final String mode = s.accelerator == 'wasm'
        ? 'wasm'
        : (s.strictWebGpu ? 'gpu-strict' : 'gpu-fallback');
    final bool gpuOn = s.accelerator != 'wasm';
    return [
      Wrap(
        spacing: 6,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          const Text('Accelerator:'),
          _modeChip('WASM (CPU)', 'wasm', mode, s),
          _modeChip('WebGPU + WASM', 'gpu-fallback', mode, s),
          _modeChip('WebGPU only', 'gpu-strict', mode, s),
        ],
      ),
      const SizedBox(height: 6),
      Wrap(
        spacing: 6,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          const Text('Precision:'),
          for (final p in Precision.values)
            ChoiceChip(
              label: Text(p == Precision.fp16 ? 'fp16' : 'fp32'),
              selected: s.precision == p,
              onSelected:
                  gpuOn ? (_) => onChanged(s.copyWith(precision: p)) : null,
            ),
        ],
      ),
      const Padding(
        padding: EdgeInsets.only(top: 4),
        child: Text(
          'Async dispatch (runAsync). Web inference is always async. '
          'Precision applies to WebGPU only.',
          style: TextStyle(fontSize: 11),
        ),
      ),
    ];
  }

  Widget _modeChip(
    String label,
    String value,
    String current,
    EngineSettings s,
  ) =>
      ChoiceChip(
        label: Text(label),
        selected: current == value,
        onSelected: (_) {
          switch (value) {
            case 'wasm':
              onChanged(s.copyWith(accelerator: 'wasm', strictWebGpu: false));
            case 'gpu-fallback':
              onChanged(s.copyWith(accelerator: 'webgpu', strictWebGpu: false));
            case 'gpu-strict':
              onChanged(s.copyWith(accelerator: 'webgpu', strictWebGpu: true));
          }
        },
      );
}
