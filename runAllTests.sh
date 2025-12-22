#!/bin/bash

# Run all tests for pose_detection_tflite
# This script runs both regular unit tests and integration tests

set -e  # Exit on any error

echo "================================================"
echo "Running regular tests from root directory..."
echo "================================================"
flutter test

echo ""
echo "================================================"
echo "Running integration tests from example directory..."
echo "================================================"
cd example

# Run each integration test file separately to avoid debug connection issues
# when the macOS app restarts between test files
flutter test integration_test/face_detection_integration_test.dart -d macos

echo ""
echo "Running benchmark tests..."
flutter test integration_test/benchmark_test.dart -d macos

echo ""
echo "================================================"
echo "✓ All tests passed successfully!"
echo "================================================"
