// This is a host-VM driver script, not app code; it reports through stdout
// exactly like the stock integrationDriver it replaces.
// ignore_for_file: avoid_print

import 'package:flutter_driver/flutter_driver.dart';
import 'package:integration_test/common.dart';
import 'package:integration_test/integration_test_driver.dart'
    show writeResponseData;

/// Like the stock `integrationDriver()`, but writes the response data on
/// failure too.
///
/// Web drive runs report failure details as an empty string, so the
/// `reportData` the suites record (the WebGPU probe outcome, and which
/// assertion saw what) is the only diagnostic that survives a failed CI run.
/// The stock driver only writes it when everything passed, which is exactly
/// when nobody needs it.
Future<void> main() async {
  final FlutterDriver driver = await FlutterDriver.connect();
  final String jsonResult = await driver.requestData(
    null,
    timeout: const Duration(minutes: 20),
  );
  final Response response = Response.fromJson(jsonResult);
  await driver.close();

  await writeResponseData(response.data);

  if (response.allTestsPassed) {
    print('All tests passed.');
  } else {
    print('Failure Details:\n${response.formattedFailureDetails}');
    throw StateError('Web integration tests failed.');
  }
}
