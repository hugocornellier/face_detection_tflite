Pod::Spec.new do |s|
  s.name                  = 'face_detection_tflite'
  s.version               = '6.2.5'
  s.summary               = 'Face detection via TensorFlow Lite (macOS)'
  s.description           = 'Flutter plugin for on-device face detection using TensorFlow Lite.'
  s.homepage              = 'https://github.com/your/repo'
  s.license               = { :type => 'MIT' }
  s.authors               = { 'You' => 'you@example.com' }
  s.source                = { :path => '.' }

  s.platform              = :osx, '11.0'

  # TFLite libraries are provided by flutter_litert dependency
end