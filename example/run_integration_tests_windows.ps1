# PowerShell script to run all integration tests on Windows
# Kills the test app between runs to avoid debug connection failures

$ErrorActionPreference = "Continue"

$testFiles = @(
    "opencv_helpers_test.dart",
    "performance_config_test.dart",
    "face_detection_integration_test.dart",
    "embedding_match_test.dart",
    "gpu_delegate_test.dart",
    "benchmark_test.dart",
    "error_recovery_test.dart",
    "edge_cases_test.dart",
    "all_model_variants_test.dart",
    "image_utils_test.dart",
    "concurrency_stress_test.dart"
)

$processName = "face_detection_tflite_example"
$passed = 0
$failed = 0
$failedTests = @()

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Running Integration Tests (Windows)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

foreach ($testFile in $testFiles) {
    Write-Host "=== $testFile ===" -ForegroundColor Yellow

    # Kill any existing app instance
    $existingProcess = Get-Process -Name $processName -ErrorAction SilentlyContinue
    if ($existingProcess) {
        Write-Host "Killing existing $processName process..." -ForegroundColor Gray
        Stop-Process -Name $processName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }

    # Small delay to ensure cleanup
    Start-Sleep -Seconds 1

    # Run the test
    $result = flutter test "integration_test/$testFile" -d windows

    if ($LASTEXITCODE -eq 0) {
        Write-Host "PASSED: $testFile" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "FAILED: $testFile" -ForegroundColor Red
        $failed++
        $failedTests += $testFile
    }

    Write-Host ""
}

# Final cleanup
Stop-Process -Name $processName -Force -ErrorAction SilentlyContinue

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "            Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Passed: $passed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })

if ($failedTests.Count -gt 0) {
    Write-Host "`nFailed tests:" -ForegroundColor Red
    foreach ($ft in $failedTests) {
        Write-Host "  - $ft" -ForegroundColor Red
    }
    exit 1
}

Write-Host "`nAll tests passed!" -ForegroundColor Green
exit 0
