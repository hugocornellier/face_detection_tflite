//
//  Generated file. Do not edit.
//

import FlutterMacOS
import Foundation

import camera_desktop
import face_detection_tflite
import file_selector_macos
import video_player_avfoundation

func RegisterGeneratedPlugins(registry: FlutterPluginRegistry) {
  CameraDesktopPlugin.register(with: registry.registrar(forPlugin: "CameraDesktopPlugin"))
  FaceDetectionTflitePlugin.register(with: registry.registrar(forPlugin: "FaceDetectionTflitePlugin"))
  FileSelectorPlugin.register(with: registry.registrar(forPlugin: "FileSelectorPlugin"))
  FVPVideoPlayerPlugin.register(with: registry.registrar(forPlugin: "FVPVideoPlayerPlugin"))
}
