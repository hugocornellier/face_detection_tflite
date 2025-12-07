#!/bin/bash

# Run benchmark tests for face_detection_tflite
# This script runs the performance benchmark integration tests

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/example/benchmark_results"

echo "================================================"
echo "Running benchmark tests from example directory..."
echo "================================================"

mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR/example"

# Run tests and capture output
TEMP_OUTPUT=$(mktemp)
flutter test integration_test/benchmark_test.dart -d macos --timeout 60m 2>&1 | tee "$TEMP_OUTPUT"

# Extract JSON blocks and save to files
echo ""
echo "Extracting benchmark results..."

# Use awk to extract JSON between markers and save to files
awk -v OUTPUT_DIR="$OUTPUT_DIR" '
/📊 BENCHMARK_JSON_START:/ {
    filename = $0
    sub(/.*BENCHMARK_JSON_START:/, "", filename)
    capturing = 1
    json = ""
    next
}
/📊 BENCHMARK_JSON_END:/ {
    if (capturing) {
        # Need parentheses for concatenation when redirecting output
        print json > (OUTPUT_DIR "/" filename)
        print "  Saved: " filename
    }
    capturing = 0
    next
}
capturing {
    json = json $0 "\n"
}
' "$TEMP_OUTPUT"

rm "$TEMP_OUTPUT"

echo ""
echo "================================================"
echo "✓ Benchmark tests completed successfully!"
echo "Results saved to: example/benchmark_results/"
echo "================================================"
