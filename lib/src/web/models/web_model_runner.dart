// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:typed_data';

// Only the CompiledModel API is pulled from the barrel; resolveWebAccelerator /
// logCompileFallback live in this package's accelerator_resolver.dart and would
// collide with the identically-named helpers the flutter_litert barrel now
// re-exports (3.4.0+), so the barrel import is restricted with `show`.
import 'package:flutter_litert/flutter_litert.dart'
    show Accelerator, CompiledModel, Precision;
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;

import '../accelerator_resolver.dart';

/// Inference engine backing a web model runner.
///
/// [liteRt] uses LiteRT.js's `LiteRtInterpreter` (the historical web path);
/// [compiledModel] uses LiteRT Next `CompiledModel` (`fromBufferAsync` /
/// `runAsync`). Both run on WASM or WebGPU.
enum WebEngine { liteRt, compiledModel }

/// Engine selection plus its knobs, passed to each web model's `initialize`.
class WebRunnerConfig {
  const WebRunnerConfig({
    this.engine = WebEngine.compiledModel,
    this.accelerator = 'auto',
    this.strictWebGpu = false,
    this.precision = Precision.fp16,
  });

  /// Which runtime compiles and runs the model.
  final WebEngine engine;

  /// Requested backend: `'auto'` (probe), `'webgpu'`, or `'wasm'`.
  final String accelerator;

  /// CompiledModel + WebGPU only: request `{gpu}` with no WASM fallback so a
  /// WebGPU compile failure surfaces instead of silently dropping to WASM.
  final bool strictWebGpu;

  /// CompiledModel WebGPU precision. Accepted for parity; LiteRT.js ignores it
  /// on the web, so it does not change runtime behaviour there.
  final Precision precision;

  WebRunnerConfig copyWith({
    WebEngine? engine,
    String? accelerator,
    bool? strictWebGpu,
    Precision? precision,
  }) => WebRunnerConfig(
    engine: engine ?? this.engine,
    accelerator: accelerator ?? this.accelerator,
    strictWebGpu: strictWebGpu ?? this.strictWebGpu,
    precision: precision ?? this.precision,
  );
}

/// Engine-agnostic wrapper the web model classes talk to instead of a raw
/// `LiteRtInterpreter`, so the same model code runs on either the LiteRT.js
/// interpreter or a LiteRT Next `CompiledModel`.
///
/// The surface is deliberately shape-free: `CompiledModel` on the web exposes
/// only byte sizes, not per-tensor shapes, so callers identify and size their
/// outputs from [outputElementCounts] and take input spatial dimensions from
/// model config (with [inputShape0] as an exact source when the engine
/// provides it).
abstract class WebModelRunner {
  /// The backend that actually compiled the model (`'webgpu'` / `'wasm'`).
  String get activeAccelerator;

  /// Element count of each output tensor, in output order.
  List<int> get outputElementCounts;

  /// Element count of input tensor 0.
  int get inputElementCount0;

  /// Shape of input tensor 0 when the engine exposes it (LiteRT.js
  /// interpreter), otherwise null (`CompiledModel` reports byte sizes only).
  List<int>? get inputShape0;

  /// Runs the model, filling each buffer in [outputs] (keyed by output index)
  /// from the matching result tensor.
  Future<void> run(List<Float32List> inputs, Map<int, Float32List> outputs);

  /// Releases the underlying model.
  void close();

  /// Compiles [bytes] on the engine and backend selected by [config].
  static Future<WebModelRunner> create(
    Uint8List bytes, {
    required WebRunnerConfig config,
    required String modelLabel,
  }) async {
    final String resolved = await resolveWebAccelerator(config.accelerator);
    switch (config.engine) {
      case WebEngine.liteRt:
        final itp = await LiteRtInterpreter.fromBytes(
          bytes,
          accelerator: resolved,
        );
        logCompileFallback(
          model: modelLabel,
          requested: resolved,
          actual: itp.activeAccelerator,
        );
        return _LiteRtRunner(itp);
      case WebEngine.compiledModel:
        final Set<Accelerator> accelerators = resolved == 'wasm'
            ? const {Accelerator.cpu}
            : (config.strictWebGpu
                  ? const {Accelerator.gpu}
                  : const {Accelerator.gpu, Accelerator.cpu});
        final model = await CompiledModel.fromBufferAsync(
          bytes,
          accelerators: accelerators,
          precision: config.precision,
        );
        final runner = _CompiledModelRunner(model);
        logCompileFallback(
          model: modelLabel,
          requested: resolved,
          actual: runner.activeAccelerator,
        );
        return runner;
    }
  }
}

class _LiteRtRunner implements WebModelRunner {
  _LiteRtRunner(this._itp);

  final LiteRtInterpreter _itp;

  @override
  String get activeAccelerator => _itp.activeAccelerator;

  @override
  List<int> get outputElementCounts => <int>[
    for (final t in _itp.getOutputTensors()) _product(t.shape),
  ];

  @override
  int get inputElementCount0 => _product(_itp.getInputTensor(0).shape);

  @override
  List<int>? get inputShape0 => List<int>.from(_itp.getInputTensor(0).shape);

  @override
  Future<void> run(List<Float32List> inputs, Map<int, Float32List> outputs) =>
      _itp.runForMultipleInputs(inputs, outputs);

  @override
  void close() => _itp.close();
}

class _CompiledModelRunner implements WebModelRunner {
  _CompiledModelRunner(this._model);

  final CompiledModel _model;

  @override
  String get activeAccelerator =>
      _model.accelerators.contains(Accelerator.gpu) ? 'webgpu' : 'wasm';

  @override
  List<int> get outputElementCounts => <int>[
    for (final bytes in _model.outputByteSizes)
      bytes ~/ Float32List.bytesPerElement,
  ];

  @override
  int get inputElementCount0 =>
      _model.inputByteSizes[0] ~/ Float32List.bytesPerElement;

  // CompiledModel exposes byte sizes only; spatial dimensions come from config.
  @override
  List<int>? get inputShape0 => null;

  @override
  Future<void> run(
    List<Float32List> inputs,
    Map<int, Float32List> outputs,
  ) async {
    final List<Float32List> result = await _model.runAsync(inputs);
    outputs.forEach((int index, Float32List dst) {
      final Float32List src = result[index];
      final int n = src.length < dst.length ? src.length : dst.length;
      dst.setRange(0, n, src);
    });
  }

  @override
  void close() => _model.close();
}

int _product(List<int> shape) {
  int n = 1;
  for (final int d in shape) {
    n *= d;
  }
  return n;
}
