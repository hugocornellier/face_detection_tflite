import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('tracking IDs flow through the native detection pipeline', (
    WidgetTester tester,
  ) async {
    final FaceDetector detector = await FaceDetector.create(
      enableTracking: true,
    );
    try {
      final ByteData data = await rootBundle.load(
        'assets/samples/landmark-ex1.jpg',
      );
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Face> first = await detector.detectFacesFromBytes(
        bytes,
        mode: FaceDetectionMode.fast,
      );
      final List<Face> second = await detector.detectFacesFromBytes(
        bytes,
        mode: FaceDetectionMode.fast,
      );

      expect(first, isNotEmpty);
      expect(first.first.trackingId, isNotNull);
      expect(second.first.trackingId, first.first.trackingId);
    } finally {
      await detector.dispose();
    }
  });
}
