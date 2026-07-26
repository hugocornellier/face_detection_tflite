# Testing and coverage

## The short version

The repository-wide coverage percentage is not a meaningful number for this
package, and raising it is not a goal. Read the per-component numbers instead.
`codecov.yml` defines the components and gates only the one that can be
measured honestly.

## What runs where

| Suite | Command | Platforms | Reports coverage |
| --- | --- | --- | --- |
| Host unit tests | `flutter test --coverage` | Dart VM | Yes |
| Browser unit tests | `flutter test --platform chrome` | Chrome (dart2js) | No |
| Integration tests | `flutter test integration_test/<file> -d <device>` | macOS, iOS, Android, Linux, Windows | No |
| Web integration | `flutter drive` via chromedriver | Chrome | No |

Only the first row produces `coverage/lcov.info`. Everything else executes real
code and verifies real behaviour while contributing nothing to the report.

## Why most of the package cannot be measured

Two separate mechanisms, both verified rather than assumed.

**Integration tests emit no coverage.** Running
`flutter test integration_test/gates_integration_test.dart -d macos --coverage`
passes its tests and writes a 0-byte `lcov.info`. Flutter (3.44.6) has no
coverage collection for device integration tests. This is why
`face_detector_core.dart`, `face_detection_model.dart`, and
`face_landmark.dart` report 0% despite being exercised by 49 integration test
files across five platforms.

**Files no host test imports are absent from the report entirely.** They do not
appear at 0%; they are not in the denominator at all. `dart:js_interop` cannot
compile for the Dart VM, so every file under `lib/src/web/` except
`detection_decode.dart` is invisible, along with
`lib/src/native/face_native_lib.dart` and the entry-point barrels. That is
about 780 executable lines outside the total, including the 1,059-line
`face_detector_web.dart`.

`flutter test --platform chrome --coverage` does not fix this. It runs the
browser tests and adds no web files to the report.

The two effects pull in opposite directions: the first makes the number look
worse than reality, the second makes it look better. They do not cancel in any
principled way, which is the whole reason the aggregate is not worth reading.

## The components

| Component | Paths | What the number means |
| --- | --- | --- |
| `logic` | `lib/src/shared/**`, `lib/src/web/detection_decode.dart` | Real. Pure Dart, no platform imports, shared by both pipelines, fully host-testable. **Gated at 99%.** |
| `orchestration` | `lib/src/models/**`, `lib/src/isolate/**`, `lib/src/face_detector.dart` | Nothing on its own. Covered by the integration suite, which cannot report. |
| `native-util` | `lib/src/util/**` | Partial. The pure-Dart half is unit-tested; the image ops need a native pipeline. |
| `ui` | `lib/src/ui/**` | Real but low. Presentation scaffolding, not detection logic. |

Measured on 2026-07-26:

```
logic           99.40%   822/827
native-util     33.85%    88/260
ui              21.54%   134/622
orchestration   11.07%   199/1798
TOTAL           35.44%  1244/3510
```

### Why the total is not a target

Those numbers were taken just after the native pipeline stopped carrying its
own copies of the shared geometry math and started calling
`lib/src/shared/face_geometry.dart` directly. That change deleted 64 executable
lines, 53 of which were covered, so the repository total went **down** (36.29%
to 35.44%) while the codebase got strictly better: one implementation instead
of four, guarded by the strictest test suite of the set.

A gated aggregate would have failed that commit. This is the concrete reason
every repo-level status in `codecov.yml` is informational.

## Where new tests are worth writing

Put logic in `lib/src/shared/` and it becomes testable for free on every
platform. This is already the established pattern:
`lib/src/web/detection_decode.dart` lives under `web/` but is pure Dart, so it
sits at 100% while the interop code around it is invisible.

Conversely, do not write unit tests to lift `face_detector_web.dart` or
`face_detector_core.dart`. The extractable logic has already been extracted
into `lib/src/shared/`. What remains is JS interop, isolate plumbing, async
lifecycle, and timing instrumentation. Mocking that surface is expensive and
finds close to nothing.

When adding tests to the shared layer, consider mutation-checking them: change
a constant or drop a branch in the source and confirm a test fails. The
`face_geometry` suite was built this way, and it caught one assertion that
passed against a deliberately broken implementation.
