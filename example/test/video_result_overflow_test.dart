import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:face_detection_tflite_example/main.dart';

/// Mirrors the structure VideoFileScreen._buildBody puts the result card in:
/// a vertical SingleChildScrollView with a stretched Column.
Widget _harness(Widget child) {
  return MaterialApp(
    home: Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [child],
        ),
      ),
    ),
  );
}

Widget _resultCard({required double aspectRatio}) {
  return VideoResultCard(
    statusMessage: 'Done. Wrote 1234 frames to:\n'
        '/some/very/long/application/documents/path/face_1749700000000.mp4',
    summary: 'Total time: 02:13 (9.3 fps avg)',
    preview: VideoPlayerChrome(
      aspectRatio: aspectRatio,
      video: const ColoredBox(color: Colors.black),
      progress: const SizedBox(height: 24),
      isPlaying: true,
      positionLabel: '00:42 / 12:34',
      onTogglePlay: () {},
    ),
    onOpenOutput: () {},
  );
}

void main() {
  Future<void> pumpAt(
    WidgetTester tester,
    Size size, {
    required double aspectRatio,
  }) async {
    tester.view.physicalSize = size;
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(_harness(_resultCard(aspectRatio: aspectRatio)));
    await tester.pump();
  }

  for (final size in const [
    Size(1200, 800),
    Size(800, 300),
    Size(320, 700),
    Size(250, 250),
    Size(150, 150),
  ]) {
    for (final (name, ratio) in const [
      ('landscape 16:9', 16 / 9),
      ('portrait 9:16', 9 / 16),
      ('ultra-wide 21:9', 21 / 9),
      ('square 1:1', 1.0),
    ]) {
      testWidgets('video result does not overflow at $size with $name video',
          (tester) async {
        await pumpAt(tester, size, aspectRatio: ratio);
        // RenderFlex overflow reports through FlutterError and fails the
        // test, so reaching this point means no overflow occurred.
        expect(find.byType(VideoResultCard), findsOneWidget);
      });
    }
  }

  testWidgets('portrait preview height is capped to 45% of the screen',
      (tester) async {
    await pumpAt(tester, const Size(1200, 800), aspectRatio: 9 / 16);
    final previewSize = tester.getSize(find.byType(AspectRatio));
    expect(previewSize.height, lessThanOrEqualTo(800 * 0.45 + 0.01));
  });
}
