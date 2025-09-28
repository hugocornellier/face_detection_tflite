Pod::Spec.new do |s|
  s.name                  = 'face_detection_tflite'
  s.version               = '0.0.1'
  s.summary               = 'Face detection via TensorFlow Lite (macOS)'
  s.description           = 'Flutter plugin that ships a TFLite C API dylib for macOS.'
  s.homepage              = 'https://github.com/your/repo'
  s.license               = { :type => 'MIT' }
  s.authors               = { 'You' => 'you@example.com' }
  s.source                = { :path => '.' }

  s.platform              = :osx, '11.0'
  s.swift_version         = '5.0'

  s.source_files          = 'Classes/**/*'

  s.dependency            'FlutterMacOS'
  s.static_framework      = true

  s.resources             = ['Frameworks/libtensorflowlite_c-mac.dylib']
  s.preserve_paths        = ['Frameworks/libtensorflowlite_c-mac.dylib']
end