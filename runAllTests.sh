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
shopt -s nullglob
integration_tests=(integration_test/*.dart)
if [ ${#integration_tests[@]} -eq 0 ]; then
  echo "No integration tests found in example/integration_test."
  exit 1
fi

for test_file in "${integration_tests[@]}"; do
  echo ""
  echo "Running ${test_file}..."
  flutter test "${test_file}" -d macos
done

echo ""
echo "================================================"
echo "✓ All tests passed successfully!"
echo "================================================"
