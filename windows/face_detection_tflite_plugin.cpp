#include "face_detection_tflite_plugin.h"
#include "include/face_detection_tflite/face_detection_tflite_plugin.h"  // ensures dllexport is seen here

// This must be included before many other Windows headers.
#include <windows.h>

// For getPlatformVersion; remove unless needed for your plugin implementation.
#include <VersionHelpers.h>

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <memory>
#include <sstream>

namespace face_detection_tflite {

// static
void FaceDetectionTflitePlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows *registrar) {
  auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "face_detection_tflite",
          &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<FaceDetectionTflitePlugin>();

  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto &call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

FaceDetectionTflitePlugin::FaceDetectionTflitePlugin() {}

FaceDetectionTflitePlugin::~FaceDetectionTflitePlugin() {}

void FaceDetectionTflitePlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue> &method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  if (method_call.method_name().compare("getPlatformVersion") == 0) {
    std::ostringstream version_stream;
    version_stream << "Windows ";
    if (IsWindows10OrGreater()) {
      version_stream << "10+";
    } else if (IsWindows8OrGreater()) {
      version_stream << "8";
    } else if (IsWindows7OrGreater()) {
      version_stream << "7";
    }
    result->Success(flutter::EncodableValue(version_stream.str()));
  } else {
    result->NotImplemented();
  }
}

}  // namespace face_detection_tflite

// Free function expected by generated_plugin_registrant on Windows.
// Uses the C API type, then converts to the C++ registrar.
void FaceDetectionTflitePluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  face_detection_tflite::FaceDetectionTflitePlugin::RegisterWithRegistrar(cpp_registrar);
}