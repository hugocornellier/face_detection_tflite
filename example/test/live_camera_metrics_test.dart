import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite_example/main.dart';

void main() {
  group('formatInferenceMilliseconds', () {
    test('uses adaptive precision across fast and slow inference times', () {
      expect(formatInferenceMilliseconds(0), '0.000');
      expect(formatInferenceMilliseconds(6012), '6.012');
      expect(formatInferenceMilliseconds(54321), '54.32');
      expect(formatInferenceMilliseconds(120100), '120.1');
      expect(formatInferenceMilliseconds(987654), '987.7');
      expect(formatInferenceMilliseconds(1000000), '1000');
      expect(formatInferenceMilliseconds(1234000), '1234');
    });
  });

  group('LiveInferenceStats', () {
    test('averages accepted samples and rejects stale samples after reset', () {
      final stats = LiveInferenceStats();
      final firstGeneration = stats.beginSample();

      expect(stats.record(firstGeneration, 6012), isTrue);
      expect(stats.record(firstGeneration, 8012), isTrue);
      expect(stats.latestUs, 8012);
      expect(stats.averageUs, 7012);
      expect(stats.sampleCount, 2);

      final staleGeneration = stats.beginSample();
      stats.reset();

      expect(stats.latestUs, isNull);
      expect(stats.averageUs, isNull);
      expect(stats.sampleCount, 0);
      expect(stats.record(staleGeneration, 9999), isFalse);
      expect(stats.sampleCount, 0);

      final currentGeneration = stats.beginSample();
      expect(stats.record(currentGeneration, 120100), isTrue);
      expect(stats.latestUs, 120100);
      expect(stats.averageUs, 120100);
    });
  });

  group('LiveCameraMetrics', () {
    Future<void> pumpMetrics(
      WidgetTester tester, {
      required int latestUs,
      required double averageUs,
      double textScale = 1,
    }) async {
      tester.view.physicalSize = const Size(250, 80);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.reset);

      await tester.pumpWidget(
        MaterialApp(
          builder: (context, child) => MediaQuery(
            data: MediaQuery.of(
              context,
            ).copyWith(textScaler: TextScaler.linear(textScale)),
            child: child!,
          ),
          home: Scaffold(
            backgroundColor: Colors.black,
            body: Center(
              child: LiveCameraMetrics(
                fps: 30,
                latestInferenceUs: latestUs,
                averageInferenceUs: averageUs,
              ),
            ),
          ),
        ),
      );
    }

    testWidgets('fits a narrow mobile width with enlarged text', (
      tester,
    ) async {
      await pumpMetrics(
        tester,
        latestUs: 120100,
        averageUs: 1234000,
        textScale: 1.3,
      );

      expect(find.text('FPS'), findsOneWidget);
      expect(find.text('LAST'), findsOneWidget);
      expect(find.text('AVERAGE'), findsOneWidget);
      expect(find.text('120.1'), findsOneWidget);
      expect(find.text('1234'), findsOneWidget);
      expect(find.text('ms'), findsNWidgets(2));
    });

    testWidgets('keeps the unit fixed when the value width changes', (
      tester,
    ) async {
      await pumpMetrics(tester, latestUs: 6012, averageUs: 8012);
      final firstUnitX = tester.getTopLeft(find.text('ms').first).dx;

      await pumpMetrics(tester, latestUs: 120100, averageUs: 1234000);
      final secondUnitX = tester.getTopLeft(find.text('ms').first).dx;

      expect(secondUnitX, firstUnitX);
    });
  });
}
