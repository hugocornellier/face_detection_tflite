# face_detection_tflite

This project is a Flutter port of the Python library [patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite). It provides on-device face detection and landmark/iris estimation using TensorFlow Lite models, packaged for Flutter applications across mobile and desktop platforms.

## Features

- Face detection using TensorFlow Lite models
- Landmark detection (eyes, nose, mouth, tragion points)
- Iris detection for both eyes
- Works on Android, iOS, macOS, Windows, and Linux
- Example app included for testing and demonstration

## Installation

1. Add this package to your Flutter project by including it in pubspec.yaml:

```yaml
dependencies:
  face_detection_tflite:
    git:
      url: https://github.com/hugocornellier/face_detection_tflite.git
```

2. Run `flutter pub get` to fetch dependencies.

## Usage

The repository includes a working demo in the `example/` directory. You can run it with:

```bash
cd example
flutter run
```

A minimal usage pattern looks like this:

```dart
import 'package:face_detection_tflite/face_detection_tflite.dart';

final detector = await FaceDetection.create(FaceDetectionModel.backCamera);
final results = await detector(imageBytes);

// results contain bounding boxes, landmarks, and iris coordinates
```

See the example for a complete implementation, including image picking and overlay rendering.

## Desktop Support

On desktop platforms, you must ensure the TensorFlow Lite C library (libtensorflowlite_c) is bundled with your app:

- **macOS**: `libtensorflowlite_c.dylib` inside `Contents/Frameworks`
- **Windows**: `tensorflowlite_c.dll` next to your `.exe`
- **Linux**: `libtensorflowlite_c.so` available in the library path

## Credits

This project is based on the excellent work by [patlevin](https://github.com/patlevin) in the [face-detection-tflite](https://github.com/patlevin/face-detection-tflite) repository.
