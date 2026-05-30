# Face Detection TFLite - Example App

This example app demonstrates the full capabilities of the `face_detection_tflite` package with two comprehensive detection modes.

## Features

### 🖼️ Still Image Detection
- Pick images from gallery
- 5 detection models (Front, Back, Short, Full Range, Full Sparse)
- Full feature customization (bounding boxes, landmarks, mesh, iris)
- Color and size customization
- Detailed performance metrics
- Processing status indicators

### 📹 Live Camera Detection
- Real-time face detection from camera
- 5 detection models (Front, Back, Short, Full Range, Full Sparse)
- 3 detection modes (Fast, Standard, Full)
- Dynamic mode/model switching
- FPS monitoring
- Frame skip control (1/1 to 1/5)
- Cross-platform (iOS, Android, macOS, Windows, Linux)

## Quick Start

1. **Run the app:**
   ```bash
   flutter run
   ```

2. **Choose detection mode:**
   - Tap "Still Image Detection" for image analysis
   - Tap "Live Camera Detection" for real-time detection

## Model Selection Guide

| Model | Input Size | Best For | Speed |
|-------|------------|----------|-------|
| **Front** | 128×128 | Selfies, close-ups | ⚡⚡⚡ Fast |
| **Back** | 256×256 | General purpose | ⚡⚡ Medium |
| **Short** | 128×128 | Close-up faces | ⚡⚡⚡ Fast |
| **Full Range** | 192×192 | Wide shots, distant faces | ⚡ Slower |
| **Full Sparse** | 192×192 | Wide range (optimized) | ⚡⚡ Medium |

## Detection Mode Guide (Live Camera Only)

| Mode | Features | Speed | Use Case |
|------|----------|-------|----------|
| **Fast** | Box + 6 landmarks | ⚡⚡⚡ 30+ FPS | Tracking, filters |
| **Standard** | Fast + 468-point mesh | ⚡⚡ 15-20 FPS | AR masks, animation |
| **Full** | Standard + iris tracking | ⚡ 8-12 FPS | Gaze tracking, analysis |

## Performance Tips

### Still Image Detection
- Start with **Back** model for testing
- Use **Front** for selfies
- Use **Full Range** for group photos
- Disable unused features (mesh/iris) for faster processing

### Live Camera Detection
- Use **Fast** mode for smooth 30+ FPS
- Increase frame skip (1/4 or 1/5) when using Full mode
- Use **Front** or **Short** model for close faces
- Use **Full Range** for distant faces

## Color Customization (Still Image Only)

Default visualization colors:
- 🟦 **Cyan** - Bounding boxes
- 🔵 **Light Blue** - Landmarks (6 points)
- 🌸 **Baby Pink** - Face mesh (468 points)
- 🔷 **Blue** - Iris ovals and eye contours

All colors are customizable via the color picker.

## Platform Support

- ✅ **iOS** - Full support with camera package
- ✅ **Android** - Full support with camera package
- ✅ **macOS** - Full support with camera_desktop package
- ✅ **Windows/Linux** - Full support with camera_desktop package
- ✅ **Web** - Still image detection only

## Technical Details

### Performance Metrics

**Still Image Detection:**
- Detection: ~20-50ms (Fast mode)
- Mesh: +60-120ms (Standard mode)
- Iris: +40-80ms (Full mode)
- Total: 20-250ms depending on mode

**Live Camera Detection:**
- Fast: 20-50ms per frame (30+ FPS)
- Standard: 100-200ms per frame (15-20 FPS)
- Full: 200-400ms per frame (8-12 FPS)

### Face Detection Outputs

**Fast Mode (all models):**
- Bounding box (4 corners)
- 6 landmarks: leftEye, rightEye, noseTip, mouth, leftEyeTragion, rightEyeTragion

**Standard Mode:**
- All Fast mode outputs
- 468-point face mesh with 3D coordinates

**Full Mode:**
- All Standard mode outputs
- Left iris: center + 4 contour points + 71 eye mesh points
- Right iris: center + 4 contour points + 71 eye mesh points

## Documentation

- [📚 Main Package README](../README.md) - Package documentation

## Troubleshooting

### Camera Not Working
- Check camera permissions in app settings
- Restart the app
- Try different camera (front/back)

### Low FPS
- Switch to Fast mode
- Increase frame skip to 1/5
- Use Front or Short model for close faces

### Faces Not Detected
- Ensure good lighting
- Check if faces are visible and unobstructed
- Try Full Range or Full Sparse model
- Ensure faces are roughly front-facing

### High Detection Times (Still Image)
- Disable unused features
- Use appropriate model (don't use Full Range for selfies)
- Consider image resolution (very large images are slower)

## Contributing

Found a bug or have a feature request? Please file an issue on the GitHub repository.

## License

This example is part of the face_detection_tflite package and shares the same license.
