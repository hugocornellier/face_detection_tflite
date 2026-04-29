#!/bin/bash

# Run benchmark tests for face_detection_tflite on Web/Chrome.
# Requires ChromeDriver running on port 4444:
#   chromedriver --port=4444

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/example/benchmark_results"

echo "================================================"
echo "Running web benchmark tests in Chrome..."
echo "================================================"

if ! curl -s -o /dev/null http://localhost:4444/status; then
  echo "ERROR: ChromeDriver not detected on port 4444."
  echo "Start it in another terminal with:  chromedriver --port=4444"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR/example"

TEMP_OUTPUT=$(mktemp)
EXTRA_DEFINES=()
if [[ "${BENCH_USE_LITERT:-0}" == "1" ]]; then
  EXTRA_DEFINES+=("--dart-define=BENCH_USE_LITERT=true")
fi
if [[ -n "${BENCH_LITERT_ACCEL:-}" ]]; then
  EXTRA_DEFINES+=("--dart-define=BENCH_LITERT_ACCEL=${BENCH_LITERT_ACCEL}")
fi

flutter drive \
  --driver=test_driver/integration_test.dart \
  --target=integration_test/face_detector_web_benchmark_test.dart \
  -d chrome \
  --browser-name=chrome \
  --release \
  "${EXTRA_DEFINES[@]}" \
  2>&1 | tee "$TEMP_OUTPUT"

echo ""
echo "Extracting benchmark results..."

awk -v OUTPUT_DIR="$OUTPUT_DIR" '
/ BENCHMARK_JSON_START:/ {
    filename = $0
    sub(/.*BENCHMARK_JSON_START:/, "", filename)
    capturing = 1
    json = ""
    next
}
/ BENCHMARK_JSON_END:/ {
    if (capturing) {
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
echo "Web benchmark tests completed."
echo "Results saved to: example/benchmark_results/"
echo "================================================"
