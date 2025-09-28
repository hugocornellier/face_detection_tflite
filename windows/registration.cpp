#include "include/face_detection_tflite/face_detection_tflite_plugin.h"
#include "face_detection_tflite_plugin.h"
#include <flutter/plugin_registrar_windows.h>

void FaceDetectionTflitePluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  face_detection_tflite::FaceDetectionTflitePlugin::RegisterWithRegistrar(cpp_registrar);
}
