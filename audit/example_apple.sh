#!/usr/bin/env bash
# Flutter 3.50 audit: build one repo's own example app for the Apple
# platforms the way the repo's CI does.
#
# usage: example_apple.sh <repo-dir> <out-dir> "<platforms: macos ios>" <ios-spm-off: 0|1>
set -u

REPO=$1
OUT=$2
PLATFORMS=$3
IOS_SPM_OFF=$4
mkdir -p "$OUT"
RES="$OUT/results.tsv"
: > "$RES"
NAME=$(basename "$REPO")
FLUTTER_VER=$(flutter --version --machine 2>/dev/null | grep -o '"frameworkVersion": *"[^"]*"' | cut -d'"' -f4)

record() { # check status seconds note tail
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$FLUTTER_VER" "$1" "$2" "$3" "$4" >> "$RES"
  echo "::notice title=RESULT::$NAME | $FLUTTER_VER | $1 | $2 | ${3}s | $4${5:+ ;; log tail: $5}"
}

step() { # check-name logfile command...
  local name=$1 log=$2 start rc note
  shift 2
  echo "::group::$name: $*"
  start=$SECONDS
  "$@" > "$log" 2>&1
  rc=$?
  tail -n 150 "$log"
  echo "::endgroup::"
  if [ $rc -eq 0 ]; then
    record "$name" PASS $((SECONDS - start)) "$(grep -m1 -E "Warning: .*Swift Package|CocoaPods" "$log" | tr '\t' ' ' | cut -c1-300)"
  else
    note=$(grep -m1 -E "(error:|Error:|ERROR:|Exception:|fatal error|Error \(Xcode\))" "$log" || true)
    [ -z "$note" ] && note=$(grep -v '^[[:space:]]*$' "$log" | tail -n 1)
    record "$name" FAIL $((SECONDS - start)) "$(printf '%s' "$note" | tr '\t' ' ' | cut -c1-500)" \
      "$(grep -v -E '^[[:space:]]*$' "$log" | tail -n 14 | tr '\t' ' ' | cut -c1-260 | awk '{printf "%s%s", (NR>1?" // ":""), $0}' | cut -c1-2600)"
  fi
  return $rc
}

cd "$REPO/example" || exit 1
step example-pub-get "$OUT/pubget.log" flutter pub get || exit 0
for p in $PLATFORMS; do
  case "$p" in
    macos) step example-macos-debug "$OUT/build-macos.log" flutter build macos --debug ;;
    ios)
      if [ "$IOS_SPM_OFF" = "1" ]; then
        flutter config --no-enable-swift-package-manager
        step example-ios-debug-cocoapods "$OUT/build-ios.log" flutter build ios --debug --no-codesign
      else
        step example-ios-debug "$OUT/build-ios.log" flutter build ios --debug --no-codesign
      fi
      ;;
  esac
done
exit 0
