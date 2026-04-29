import 'dart:typed_data';

/// Converts RGBA pixel data to a normalized RGB Float32List tensor in [0.0, 1.0].
void rgbaToRgbFloat32(Uint8List rgbaData, Float32List output) {
  const double norm = 1.0 / 255.0;
  int dst = 0;
  for (int src = 0; src < rgbaData.length; src += 4) {
    output[dst++] = rgbaData[src] * norm;
    output[dst++] = rgbaData[src + 1] * norm;
    output[dst++] = rgbaData[src + 2] * norm;
  }
}

/// Converts RGBA pixel data to a signed Float32List tensor in [-1.0, 1.0],
/// matching MediaPipe model expectations.
void rgbaToSignedRgbFloat32(Uint8List rgbaData, Float32List output) {
  const double norm = 1.0 / 127.5;
  int dst = 0;
  for (int src = 0; src < rgbaData.length; src += 4) {
    output[dst++] = rgbaData[src] * norm - 1.0;
    output[dst++] = rgbaData[src + 1] * norm - 1.0;
    output[dst++] = rgbaData[src + 2] * norm - 1.0;
  }
}
