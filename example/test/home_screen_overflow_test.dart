import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:face_detection_tflite_example/main.dart';

void main() {
  Future<void> pumpHomeAt(WidgetTester tester, Size size) async {
    tester.view.physicalSize = size;
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(const MaterialApp(home: HomeScreen()));
    await tester.pumpAndSettle();
  }

  for (final size in const [
    Size(1200, 800),
    Size(900, 300),
    Size(320, 700),
    Size(250, 250),
    Size(150, 150),
  ]) {
    testWidgets('home screen does not overflow at $size', (tester) async {
      await pumpHomeAt(tester, size);
      expect(find.text('Choose Detection Mode'), findsOneWidget);
      // RenderFlex overflow reports through FlutterError and fails the test,
      // so reaching this point means no overflow occurred.
    });
  }

  testWidgets('mode cards are laid out in a single row', (tester) async {
    await pumpHomeAt(tester, const Size(1200, 800));
    final cards = find.byType(Card);
    expect(cards, findsNWidgets(3));
    final cardTops = <double>[
      for (final card in cards.evaluate())
        tester.getTopLeft(find.byWidget(card.widget)).dy,
    ];
    expect(cardTops.toSet().length, 1,
        reason: 'all three cards should share the same vertical offset');
  });

  testWidgets('live camera is first and segmentation is gone', (tester) async {
    await pumpHomeAt(tester, const Size(1200, 800));
    expect(find.text('Selfie Segmentation'), findsNothing);
    final lefts = {
      for (final title in const ['Live Camera', 'Still Image', 'Video File'])
        title: tester.getTopLeft(find.text(title)).dx,
    };
    expect(lefts['Live Camera']!, lessThan(lefts['Still Image']!));
    expect(lefts['Still Image']!, lessThan(lefts['Video File']!));
  });
}
