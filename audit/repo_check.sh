#!/usr/bin/env bash
# Flutter 3.50 audit: analyze, format-check and test one package repo at its
# default branch, and try its example app's Android build.
#
# usage: repo_check.sh <repo-dir> <out-dir> <build-example-apk: 0|1>
set -u

REPO=$1
OUT=$2
BUILD_APK=$3
mkdir -p "$OUT"
RES="$OUT/results.tsv"
: > "$RES"
NAME=$(basename "$REPO")
FLUTTER_VER=$(flutter --version --machine 2>/dev/null | grep -o '"frameworkVersion": *"[^"]*"' | cut -d'"' -f4)

record() { # check status seconds note
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$FLUTTER_VER" "$1" "$2" "$3" "$4" >> "$RES"
  echo "RESULT $NAME | $FLUTTER_VER | $1 | $2 | ${3}s | $4"
}

step() { # check-name logfile dir command...
  local name=$1 log=$2 dir=$3 start rc note
  shift 3
  echo "::group::$name: $* (in $dir)"
  start=$SECONDS
  (cd "$dir" && "$@") > "$log" 2>&1
  rc=$?
  tail -n 120 "$log"
  echo "::endgroup::"
  if [ $rc -eq 0 ]; then
    note=$(grep -m1 -E "No issues found|All tests passed|issues? found|Formatted [0-9]+ files" "$log" | tr '\t' ' ' | cut -c1-200)
    record "$name" PASS $((SECONDS - start)) "$note"
  else
    note=$(awk '/\* What went wrong:/{f=1; n=0; next} f && /^\* /{exit} f && NF && n<4{gsub(/^[ >]+/,""); printf "%s%s", (n?" / ":""), $0; n++}' "$log")
    [ -z "$note" ] && note=$(grep -m1 -E "^ *(error|warning|info) [-•]|issues? found|Some tests failed|\[E\]|Error:|error:|Changed |version solving failed|Because " "$log")
    [ -z "$note" ] && note=$(grep -v '^[[:space:]]*$' "$log" | tail -n 1)
    record "$name" FAIL $((SECONDS - start)) "$(printf '%s' "$note" | tr '\t' ' ' | cut -c1-500)"
  fi
  return $rc
}

cd "$REPO" || exit 1

step pub-get "$OUT/pubget.log" . flutter pub get || exit 1
step analyze "$OUT/analyze.log" . flutter analyze .
step analyze-fatal-infos "$OUT/analyze-fatal-infos.log" . flutter analyze --fatal-infos .
step format "$OUT/format.log" . dart format --output=none --set-exit-if-changed .
if [ -d test ]; then
  step test "$OUT/test.log" . flutter test
else
  record test SKIP 0 "no test directory"
fi

if [ -f example/pubspec.yaml ]; then
  if step example-pub-get "$OUT/example-pubget.log" example flutter pub get; then
    step example-analyze "$OUT/example-analyze.log" example flutter analyze .
    if [ "$BUILD_APK" = "1" ] && [ -d example/android ]; then
      step example-apk-debug "$OUT/example-apk.log" example flutter build apk --debug
    fi
  fi
fi
exit 0
