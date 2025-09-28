#ifndef FACE_DETECTION_TFLITE_PUBLIC_PLUGIN_H_
#define FACE_DETECTION_TFLITE_PUBLIC_PLUGIN_H_

#include <flutter_windows.h>

#ifdef FLUTTER_PLUGIN_IMPL
#define FLUTTER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FLUTTER_PLUGIN_EXPORT __declspec(dllimport)
#endif

// C++ signature is fine (no extern "C") since the registrant is C++ and includes this header.
FLUTTER_PLUGIN_EXPORT void FaceDetectionTflitePluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar);

#endif  // FACE_DETECTION_TFLITE_PUBLIC_PLUGIN_H_
