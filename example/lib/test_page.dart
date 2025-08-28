import 'dart:typed_data';
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_selector/file_selector.dart';
import 'package:face_detector_flutter/face_detector_flutter.dart';

class TestPage extends StatefulWidget {
  const TestPage({super.key});
  @override
  State<TestPage> createState() => _TestPageState();
}

class _TestPageState extends State<TestPage> {
  Uint8List? _imageBytes;
  Size? _imageSize;
  List<FaceResult> _results = [];
  FaceDetector? _fd;
  bool _loading = true;
  String? _loadError;

  @override
  void initState() {
    super.initState();
    _initModels();
  }

  Future<void> _initModels() async {
    try {
      final fd = await FaceDetector.create(model: FaceDetectionModel.backCamera);
      if (!mounted) return;
      setState(() {
        _fd = fd;
        _loading = false;
      });
      print("FaceDetector ready");
    } catch (e, st) {
      print("FaceDetector.create failed");
      print(e);
      print(st);
      if (!mounted) return;
      setState(() {
        _loadError = e.toString();
        _loading = false;
      });
    }
  }

  Future<Uint8List?> _pickImageBytes() async {
    try {
      if (!kIsWeb && (Platform.isMacOS || Platform.isWindows || Platform.isLinux)) {
        print("_pickImageBytes desktop branch called");

        final typeGroup = XTypeGroup(
          label: 'images',
          extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff'],
        );

        print("_pickImageBytes here1");

        final files = await openFiles(
          acceptedTypeGroups: [typeGroup],
          confirmButtonText: 'Select',
        );
        final file = files.isEmpty ? null : files.first;

        print("_pickImageBytes here2");

        if (file == null) {
          print("Returning null");
          return null;
        }

        return await file.readAsBytes();
      } else {
        final picker = ImagePicker();
        final picked = await picker.pickImage(
          source: ImageSource.gallery,
          imageQuality: 100,
        );
        if (picked == null) return null;
        return await picked.readAsBytes();
      }
    } catch (e) {
      print("_pickImageBytes failure");
      print(e);
      return null;
    }
  }

  Future<void> _pickAndRun() async {
    try {
      final bytes = await _pickImageBytes();
      if (bytes == null) {
        print("Bytes are null, returning");
        return;
      }

      print("Bytes not null..");

      final decoded = await decodeImageFromList(bytes);
      final imgSize = Size(decoded.width.toDouble(), decoded.height.toDouble());

      if (_fd == null) {
        print("_fd == null, returning");
        return;
      }

      final results = await _fd!.detect(
        bytes,
        withFaceLandmarks: false,
        withIris: true,
      );

      print(results);

      setState(() {
        _imageBytes = bytes;
        _imageSize = imgSize;
        _results = results;
      });
    } catch (e) {
      print("_pickAndRun failure");
      print(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    final hasImage = _imageBytes != null && _imageSize != null;
    return Scaffold(
      appBar: AppBar(title: const Text('face_detector_flutter • example')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _fd == null ? null : _pickAndRun,
        label: Text(_fd == null ? 'Loading…' : 'Pick Image'),
        icon: const Icon(Icons.image),
      ),
      body: Center(
        child: hasImage
            ? LayoutBuilder(
          builder: (context, constraints) {
            final fitted =
            _fitSize(_imageSize!, Size(constraints.maxWidth, constraints.maxHeight));
            return SizedBox(
              width: fitted.width,
              height: fitted.height,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.memory(_imageBytes!, fit: BoxFit.contain),
                  CustomPaint(
                    painter: _ResultsPainter(
                      results: _results,
                      originalSize: _imageSize!,
                    ),
                  ),
                ],
              ),
            );
          },
        )
            : (_loading
            ? const CircularProgressIndicator()
            : (_loadError != null
            ? Text('Init failed:\n$_loadError')
            : const Text('Pick an image to run detection'))),
      ),
    );
  }

  Size _fitSize(Size src, Size bound) {
    final scale = (bound.width / src.width < bound.height / src.height)
        ? bound.width / src.width
        : bound.height / src.height;
    return Size(src.width * scale, src.height * scale);
  }
}

class _ResultsPainter extends CustomPainter {
  final List<FaceResult> results;
  final Size originalSize;

  _ResultsPainter({required this.results, required this.originalSize});

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = const Color(0xFF00FFCC);

    final kpPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFF89CFF0);

    final irisPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = const Color(0xFFAA66CC);

    for (final r in results) {
      final rect = Rect.fromLTRB(
        r.bbox.xmin * size.width,
        r.bbox.ymin * size.height,
        r.bbox.xmax * size.width,
        r.bbox.ymax * size.height,
      );
      canvas.drawRect(rect, boxPaint);

      for (int i = 0; i < r.keypointsXY.length; i += 2) {
        final x = r.keypointsXY[i] * size.width;
        final y = r.keypointsXY[i + 1] * size.height;
        canvas.drawCircle(Offset(x, y), 3, kpPaint);
      }

      if (r.iris != null) {
        for (final p in [...r.iris!.leftEye, ...r.iris!.rightEye]) {
          final x = p[0] * size.width / originalSize.width;
          final y = p[1] * size.height / originalSize.height;
          canvas.drawCircle(Offset(x, y), 2, irisPaint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant _ResultsPainter old) =>
      old.results != results || old.originalSize != originalSize;
}