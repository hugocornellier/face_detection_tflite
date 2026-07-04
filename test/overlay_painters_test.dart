import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_detection_tflite/src/native/face_native_lib.dart';

import 'test_config.dart';

/// Tests for the per-face info card (score/mesh score/head pose) added to the
/// overlay painters via `showPoseAndScores`.
void main() {
  globalTestSetup();

  // A face whose mesh spans both head axes so head pose is estimable.
  Face meshFace({double detectionScore = 0.87, double? meshScore = 0.9}) {
    final points = List.generate(468, (_) => const Point(320, 240, 0));
    points[10] = const Point(320, 140, 0); // forehead
    points[152] = const Point(320, 340, 0); // chin
    points[234] = const Point(220, 240, 0); // left cheek
    points[454] = const Point(420, 240, 0); // right cheek
    return Face(
      detection: Detection(
        boundingBox: const RectF(0.2, 0.2, 0.8, 0.8),
        score: detectionScore,
        keypointsXY: TestUtils.generateValidKeypoints(),
        imageSize: TestConstants.mediumImage,
      ),
      mesh: FaceMesh(points, score: meshScore),
      irises: const [],
      originalSize: TestConstants.mediumImage,
    );
  }

  Face fastFace() => Face(
    detection: Detection(
      boundingBox: const RectF(0.2, 0.2, 0.8, 0.8),
      score: 0.75,
      keypointsXY: TestUtils.generateValidKeypoints(),
      imageSize: TestConstants.mediumImage,
    ),
    mesh: null,
    irises: const [],
    originalSize: TestConstants.mediumImage,
  );

  group('faceInfoLabelText', () {
    test('mesh face lists both scores and all three angles', () {
      final text = faceInfoLabelText(meshFace());
      expect(text, contains('score 0.87'));
      expect(text, contains('mesh 0.90'));
      expect(text, contains('P '));
      expect(text, contains('Y '));
      expect(text, contains('R '));
    });

    test('omits mesh score when the mesh model does not report one', () {
      final text = faceInfoLabelText(meshFace(meshScore: null));
      expect(text, contains('score 0.87'));
      expect(text, isNot(contains('mesh')));
    });

    test('fast face shows detection score and roll only', () {
      final text = faceInfoLabelText(fastFace());
      expect(text, contains('score 0.75'));
      expect(text, isNot(contains('mesh')));
      expect(text, isNot(contains('P ')));
      expect(text, isNot(contains('Y ')));
      expect(text, contains('R '));
    });

    test('appends smile / eye-open only when showClassification is set', () {
      final scores = List<double>.filled(52, 0.0);
      scores[44] = 0.9; // mouthSmileLeft
      scores[45] = 0.8; // mouthSmileRight -> smile 0.85
      scores[9] = 0.05; // eyeBlinkLeft   -> eyeL open 0.95
      scores[10] = 0.7; // eyeBlinkRight  -> eyeR open 0.30
      final face = Face(
        detection: Detection(
          boundingBox: const RectF(0.2, 0.2, 0.8, 0.8),
          score: 0.9,
          keypointsXY: TestUtils.generateValidKeypoints(),
          imageSize: TestConstants.mediumImage,
        ),
        mesh: null,
        irises: const [],
        blendshapeScores: scores,
        originalSize: TestConstants.mediumImage,
      );
      final on = faceInfoLabelText(face, showClassification: true);
      expect(on, contains('smile 0.85'));
      expect(on, contains('eyeL 0.95'));
      expect(on, contains('eyeR 0.30'));
      // Off by default.
      expect(faceInfoLabelText(face), isNot(contains('smile')));
    });
  });

  group('painters with showPoseAndScores', () {
    testWidgets('DetectionsPainter paints the info card without errors', (
      tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Center(
            child: SizedBox(
              width: 400,
              height: 300,
              child: CustomPaint(
                painter: DetectionsPainter(
                  faces: [meshFace(), fastFace()],
                  imageRectOnCanvas: const Rect.fromLTWH(0, 0, 400, 300),
                  originalImageSize: TestConstants.mediumImage,
                  showBoundingBoxes: true,
                  showMesh: false,
                  showLandmarks: false,
                  showLandmarkLabels: false,
                  showIrises: false,
                  showEyeContours: false,
                  showEyeMesh: false,
                  showPoseAndScores: true,
                  boundingBoxColor: Colors.green,
                  landmarkColor: Colors.blue,
                  meshColor: Colors.pink,
                  irisColor: Colors.cyan,
                  eyeContourColor: Colors.cyan,
                  eyeMeshColor: Colors.orange,
                  boundingBoxThickness: 2,
                  landmarkSize: 3,
                  meshSize: 1,
                  eyeMeshSize: 1,
                ),
              ),
            ),
          ),
        ),
      );
      expect(tester.takeException(), isNull);
    });

    testWidgets('CameraDetectionPainter paints the info card without errors', (
      tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Center(
            child: SizedBox(
              width: 400,
              height: 300,
              child: CustomPaint(
                painter: CameraDetectionPainter(
                  faces: [meshFace()],
                  imageSize: TestConstants.mediumImage,
                  cameraAspectRatio: 4 / 3,
                  displayAspectRatio: 4 / 3,
                  detectionMode: FaceDetectionMode.standard,
                  sensorOrientation: 0,
                  deviceOrientation: Orientation.landscape,
                  isFrontCamera: false,
                  mirrorHorizontally: true,
                  showPoseAndScores: true,
                ),
              ),
            ),
          ),
        ),
      );
      expect(tester.takeException(), isNull);
    });
  });
}
