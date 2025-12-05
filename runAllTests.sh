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
flutter test integration_test -d macos

echo ""
echo "================================================"
echo "✓ All tests passed successfully!"
echo "================================================"
