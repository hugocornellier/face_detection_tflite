import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart' show Size;
import 'dart:math' as math;

/// Global test setup and configuration for face_detection_tflite tests.
///
/// This file provides:
/// - Flutter test environment initialization
/// - Common test utilities and helpers
/// - Test data generators
/// - Assertion helpers
void globalTestSetup() {
  TestWidgetsFlutterBinding.ensureInitialized();
}

/// Test utilities for face detection tests
class TestUtils {
  /// Generates a valid normalized keypoints array with all 6 face landmarks
  ///
  /// Values are in normalized coordinates [0.0, 1.0]
  /// Order: leftEye, rightEye, noseTip, mouth, leftEyeTragion, rightEyeTragion
  static List<double> generateValidKeypoints({
    math.Point<double>? leftEye,
    math.Point<double>? rightEye,
    math.Point<double>? noseTip,
    math.Point<double>? mouth,
    math.Point<double>? leftEyeTragion,
    math.Point<double>? rightEyeTragion,
  }) {
    return [
      leftEye?.x ?? 0.3,
      leftEye?.y ?? 0.4,
      rightEye?.x ?? 0.7,
      rightEye?.y ?? 0.4,
      noseTip?.x ?? 0.5,
      noseTip?.y ?? 0.6,
      mouth?.x ?? 0.5,
      mouth?.y ?? 0.75,
      leftEyeTragion?.x ?? 0.1,
      leftEyeTragion?.y ?? 0.4,
      rightEyeTragion?.x ?? 0.9,
      rightEyeTragion?.y ?? 0.4,
    ];
  }

  /// Creates a dummy image byte array for testing
  ///
  /// Returns a minimal valid image in memory (1x1 pixel)
  /// For actual image processing tests, use real test fixtures
  static Uint8List createDummyImageBytes() {
    // Minimal PNG: 1x1 transparent pixel
    return Uint8List.fromList([
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
      0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
      0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // Width: 1, Height: 1
      0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, // Bit depth, color type
      0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
      0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
      0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, // Image data
      0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND chunk
      0x42, 0x60, 0x82
    ]);
  }

  /// Validates that a Point is within reasonable bounds
  static bool isValidPixelCoordinate(math.Point<double> point, Size imageSize) {
    return point.x >= 0 &&
        point.x <= imageSize.width &&
        point.y >= 0 &&
        point.y <= imageSize.height;
  }

  /// Checks if two doubles are approximately equal within tolerance
  static bool approximatelyEqual(double a, double b,
      {double epsilon = 0.0001}) {
    return (a - b).abs() < epsilon;
  }

  /// Checks if two Points are approximately equal
  static bool pointsApproximatelyEqual(
      math.Point<double> a, math.Point<double> b,
      {double epsilon = 0.0001}) {
    return approximatelyEqual(a.x, b.x, epsilon: epsilon) &&
        approximatelyEqual(a.y, b.y, epsilon: epsilon);
  }
}

/// Custom matchers for face detection tests
class FaceDetectionMatchers {
  /// Matcher for checking if a Point is approximately equal to expected
  static Matcher approximatelyEqualsPoint(math.Point<double> expected,
      {double epsilon = 0.0001}) {
    return _ApproximatePointMatcher(expected, epsilon);
  }

  /// Matcher for checking if a value is within a range
  static Matcher inRange(num min, num max) {
    return _InRangeMatcher(min, max);
  }

  /// Matcher for checking if a point is within image bounds
  static Matcher withinImageBounds(Size imageSize) {
    return _WithinImageBoundsMatcher(imageSize);
  }
}

/// Custom matcher for approximate point equality
class _ApproximatePointMatcher extends Matcher {
  final math.Point<double> expected;
  final double epsilon;

  _ApproximatePointMatcher(this.expected, this.epsilon);

  @override
  bool matches(Object? item, Map matchState) {
    if (item is! math.Point<double>) {
      matchState['error'] = 'Expected a Point<double>, got ${item.runtimeType}';
      return false;
    }

    final xMatch = (item.x - expected.x).abs() < epsilon;
    final yMatch = (item.y - expected.y).abs() < epsilon;

    if (!xMatch) {
      matchState['xDiff'] = (item.x - expected.x).abs();
    }
    if (!yMatch) {
      matchState['yDiff'] = (item.y - expected.y).abs();
    }

    return xMatch && yMatch;
  }

  @override
  Description describe(Description description) {
    return description.add(
        'approximately equals Point(${expected.x}, ${expected.y}) within $epsilon');
  }

  @override
  Description describeMismatch(
    Object? item,
    Description mismatchDescription,
    Map matchState,
    bool verbose,
  ) {
    if (matchState.containsKey('error')) {
      return mismatchDescription.add(matchState['error'] as String);
    }

    final buffer = StringBuffer('was Point(');
    if (item is math.Point<double>) {
      buffer.write('${item.x}, ${item.y}');
    } else {
      buffer.write(item.toString());
    }
    buffer.write(')');

    if (matchState.containsKey('xDiff') || matchState.containsKey('yDiff')) {
      buffer.write(' with differences: ');
      if (matchState.containsKey('xDiff')) {
        buffer.write('x: ${matchState['xDiff']}');
      }
      if (matchState.containsKey('yDiff')) {
        if (matchState.containsKey('xDiff')) buffer.write(', ');
        buffer.write('y: ${matchState['yDiff']}');
      }
    }

    return mismatchDescription.add(buffer.toString());
  }
}

/// Custom matcher for range checking
class _InRangeMatcher extends Matcher {
  final num min;
  final num max;

  _InRangeMatcher(this.min, this.max);

  @override
  bool matches(Object? item, Map matchState) {
    if (item is! num) {
      matchState['error'] = 'Expected a number, got ${item.runtimeType}';
      return false;
    }
    return item >= min && item <= max;
  }

  @override
  Description describe(Description description) {
    return description.add('in range [$min, $max]');
  }

  @override
  Description describeMismatch(
    Object? item,
    Description mismatchDescription,
    Map matchState,
    bool verbose,
  ) {
    if (matchState.containsKey('error')) {
      return mismatchDescription.add(matchState['error'] as String);
    }
    return mismatchDescription.add('was $item');
  }
}

/// Custom matcher for image bounds checking
class _WithinImageBoundsMatcher extends Matcher {
  final Size imageSize;

  _WithinImageBoundsMatcher(this.imageSize);

  @override
  bool matches(Object? item, Map matchState) {
    if (item is! math.Point<double>) {
      matchState['error'] = 'Expected a Point<double>, got ${item.runtimeType}';
      return false;
    }

    final xInBounds = item.x >= 0 && item.x <= imageSize.width;
    final yInBounds = item.y >= 0 && item.y <= imageSize.height;

    if (!xInBounds) {
      matchState['xOutOfBounds'] = item.x;
    }
    if (!yInBounds) {
      matchState['yOutOfBounds'] = item.y;
    }

    return xInBounds && yInBounds;
  }

  @override
  Description describe(Description description) {
    return description.add(
        'within image bounds (width: ${imageSize.width}, height: ${imageSize.height})');
  }

  @override
  Description describeMismatch(
    Object? item,
    Description mismatchDescription,
    Map matchState,
    bool verbose,
  ) {
    if (matchState.containsKey('error')) {
      return mismatchDescription.add(matchState['error'] as String);
    }

    final buffer = StringBuffer('Point(');
    if (item is math.Point<double>) {
      buffer.write('${item.x}, ${item.y}');
    }
    buffer.write(') is out of bounds');

    if (matchState.containsKey('xOutOfBounds')) {
      buffer.write(' (x: ${matchState['xOutOfBounds']})');
    }
    if (matchState.containsKey('yOutOfBounds')) {
      buffer.write(' (y: ${matchState['yOutOfBounds']})');
    }

    return mismatchDescription.add(buffer.toString());
  }
}

/// Test constants for consistent test data
class TestConstants {
  // Standard test image sizes
  static const Size smallImage = Size(100, 100);
  static const Size mediumImage = Size(640, 480);
  static const Size largeImage = Size(1920, 1080);
  static const Size veryLargeImage = Size(4096, 4096);

  // Common test scores
  static const double highConfidence = 0.95;
  static const double mediumConfidence = 0.7;
  static const double lowConfidence = 0.51;
  static const double veryLowConfidence = 0.3;

  // Standard test bounding boxes (normalized)
  static const double bboxXmin = 0.2;
  static const double bboxYmin = 0.3;
  static const double bboxXmax = 0.8;
  static const double bboxYmax = 0.7;

  // Epsilon values for floating point comparisons
  static const double defaultEpsilon = 0.0001;
  static const double relaxedEpsilon = 0.01;

  // Test keypoint positions (normalized)
  static const double leftEyeX = 0.35;
  static const double leftEyeY = 0.4;
  static const double rightEyeX = 0.65;
  static const double rightEyeY = 0.4;
  static const double noseTipX = 0.5;
  static const double noseTipY = 0.55;
  static const double mouthX = 0.5;
  static const double mouthY = 0.7;
  static const double leftEyeTragionX = 0.15;
  static const double leftEyeTragionY = 0.4;
  static const double rightEyeTragionX = 0.85;
  static const double rightEyeTragionY = 0.4;
}
