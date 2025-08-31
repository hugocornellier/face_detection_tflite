Pod::Spec.new do |s|
  s.name                  = 'face_detection_tflite'
  s.version               = '0.0.1'
  s.summary               = 'Face detection via TensorFlow Lite (macOS)'
  s.description           = 'Flutter plugin that ships a TFLite C API dylib for macOS.'
  s.homepage              = 'https://github.com/your/repo'
  s.license               = { :type => 'MIT' }
  s.authors               = { 'You' => 'you@example.com' }
  s.source                = { :path => '.' }

  s.platform              = :osx, '10.15'
  s.swift_version         = '5.0'

  # ensure there is at least one file in Classes/ (e.g., FaceDetectionTflitePlugin.swift)
  s.source_files          = 'Classes/**/*'

  s.dependency            'FlutterMacOS'
  s.static_framework      = true

  # Keep the dylib in your repo here:
  # face_detection_tflite/macos/Resources/libtensorflowlite_c-mac.dylib
  # CocoaPods will copy this into the consuming app’s .app/Contents/Resources
  s.resources             = ['Resources/libtensorflowlite_c-mac.dylib']

  # (Optional) Keep source file around even if not compiled.
  s.preserve_paths        = ['Resources/libtensorflowlite_c-mac.dylib']
end