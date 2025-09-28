#include "include/face_detection_tflite/face_detection_tflite_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "face_detection_tflite_plugin.h"

void FaceDetectionTflitePluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  face_detection_tflite::FaceDetectionTflitePlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
