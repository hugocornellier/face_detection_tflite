#!/usr/bin/env bash
# Flutter 3.50 audit: build a fresh `flutter create` app that depends on one
# PUBLISHED package, for every platform this host can build.
#
# usage: run.sh <package-key> <host: linux|macos|windows> <app-root> <out-dir>
set -u

KEY=$1
HOST=$2
APPROOT=$3
OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)

# shellcheck disable=SC1090
. "$HERE/pkgs/$KEY/conf.sh"

mkdir -p "$OUT" "$APPROOT"
RES="$OUT/results.tsv"
: > "$RES"
FLUTTER_VER=$(flutter --version --machine 2>/dev/null | grep -o '"frameworkVersion": *"[^"]*"' | cut -d'"' -f4)

case "$HOST" in
  linux) HOST_PLATFORMS="android linux web" ;;
  macos) HOST_PLATFORMS="ios macos" ;;
  windows) HOST_PLATFORMS="windows" ;;
esac

TODO=""
for p in $HOST_PLATFORMS; do
  case " $PLATFORMS " in *" $p "*) TODO="$TODO $p" ;; esac
done
TODO=${TODO# }
if [ -z "$TODO" ]; then
  echo "nothing to build for $KEY on $HOST"
  exit 0
fi

record() { # platform status seconds first-error
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$KEY" "$DEPS" "$FLUTTER_VER" "$1" "$2" "$3" "$4" >> "$RES"
  echo "RESULT $KEY | $FLUTTER_VER | $1 | $2 | ${3}s | $4"
}

first_error() { # logfile
  local f=$1 line
  # Gradle prints the cause on the line after "* What went wrong:".
  line=$(awk '/\* What went wrong:/{f=1; n=0; next} f && /^\* /{exit} f && NF && n<4{gsub(/^[ >]+/,""); printf "%s%s", (n?" / ":""), $0; n++}' "$f")
  if [ -z "$line" ]; then
    line=$(grep -m1 -E "(error:|Error:|ERROR:|FAILURE:|Exception:|fatal error|Error \(Xcode\)|error [A-Z]+[0-9]+)" "$f" || true)
  fi
  if [ -z "$line" ]; then
    line=$(grep -v '^[[:space:]]*$' "$f" | tail -n 1)
  fi
  printf '%s' "$line" | tr '\t' ' ' | cut -c1-500
}

cd "$APPROOT" || exit 1
CREATE_PLATFORMS=$(echo "$TODO" | tr " " ",")

echo "::group::flutter create ($CREATE_PLATFORMS)"
if ! flutter create --no-pub --org com.audit350 --platforms="$CREATE_PLATFORMS" audit_app > "$OUT/create.log" 2>&1; then
  cat "$OUT/create.log"
  echo "::endgroup::"
  for p in $TODO; do record "$p" "FAIL(create)" 0 "$(first_error "$OUT/create.log")"; done
  exit 1
fi
cat "$OUT/create.log"
echo "::endgroup::"
cd audit_app || exit 1

echo "::group::template versions"
for f in android/settings.gradle.kts android/gradle/wrapper/gradle-wrapper.properties android/gradle.properties android/app/build.gradle.kts; do
  [ -f "$f" ] && { echo "--- $f"; cat "$f"; }
done
grep -h "DEPLOYMENT_TARGET" ios/Runner.xcodeproj/project.pbxproj macos/Runner.xcodeproj/project.pbxproj 2>/dev/null | sort | uniq -c
echo "::endgroup::"

echo "::group::flutter pub add $DEPS"
# shellcheck disable=SC2086
if ! flutter pub add $DEPS > "$OUT/pubadd.log" 2>&1; then
  cat "$OUT/pubadd.log"
  echo "::endgroup::"
  for p in $TODO; do record "$p" "FAIL(pub)" 0 "$(first_error "$OUT/pubadd.log")"; done
  exit 1
fi
cat "$OUT/pubadd.log"
flutter pub deps --style=compact > "$OUT/pubdeps.txt" 2>&1
cat "$OUT/pubdeps.txt"
echo "::endgroup::"

cp "$HERE/pkgs/$KEY/main.dart" lib/main.dart
rm -rf test

# Setup steps each package's README documents for the app.
if [ "${IOS15:-0}" = "1" ] && [ -f ios/Runner.xcodeproj/project.pbxproj ]; then
  perl -pi -e 's/IPHONEOS_DEPLOYMENT_TARGET = [0-9.]+;/IPHONEOS_DEPLOYMENT_TARGET = 15.0;/' ios/Runner.xcodeproj/project.pbxproj
  echo "setup: iOS deployment target set to 15.0"
fi
if [ "${DARTCV_HOOK:-0}" = "1" ]; then
  printf "\nhooks:\n  user_defines:\n    dartcv4:\n      ios:\n        deployment_target: '15.0'\n" >> pubspec.yaml
  echo "setup: dartcv4 iOS deployment_target hook define added"
fi
if [ -n "${MINSDK:-}" ] && [ -f android/app/build.gradle.kts ]; then
  perl -pi -e "s/minSdk = flutter\.minSdkVersion/minSdk = $MINSDK/" android/app/build.gradle.kts
  grep -n "minSdk" android/app/build.gradle.kts
  echo "setup: minSdk set to $MINSDK"
fi

ANY_FAIL=0
for p in $TODO; do
  case "$p" in
    android) CMD="flutter build apk --release" ;;
    ios) CMD="flutter build ios --release --no-codesign" ;;
    macos) CMD="flutter build macos --release" ;;
    linux) CMD="flutter build linux --release" ;;
    windows) CMD="flutter build windows --release" ;;
    web) CMD="flutter build web --release" ;;
  esac
  LOG="$OUT/build-$p.log"
  echo "::group::$p: $CMD"
  START=$SECONDS
  $CMD > "$LOG" 2>&1
  RC=$?
  DUR=$((SECONDS - START))
  tail -n 150 "$LOG"
  echo "::endgroup::"
  if [ $RC -eq 0 ]; then
    record "$p" PASS "$DUR" ""
  else
    ANY_FAIL=1
    record "$p" FAIL "$DUR" "$(first_error "$LOG")"
  fi
  # Extra Android cell: the same app after the documented switch to AGP
  # built-in Kotlin and the new DSL (only when the app template itself no
  # longer applies the Kotlin Gradle plugin).
  if [ "$p" = "android" ] && [ $RC -eq 0 ] && [ -f android/gradle.properties ] \
      && ! grep -q -E "kotlin-android|org\.jetbrains\.kotlin\.android" android/app/build.gradle.kts; then
    cp android/gradle.properties "$OUT/gradle.properties.before-builtin-kotlin"
    perl -ni -e 'print unless /^android\.(builtInKotlin|newDsl)=/' android/gradle.properties
    printf 'android.builtInKotlin=true\nandroid.newDsl=true\n' >> android/gradle.properties
    LOG="$OUT/build-android-builtin-kotlin.log"
    echo "::group::android-builtin-kotlin: flutter build apk --release (android.builtInKotlin=true, android.newDsl=true)"
    START=$SECONDS
    flutter build apk --release > "$LOG" 2>&1
    RC=$?
    DUR=$((SECONDS - START))
    tail -n 80 "$LOG"
    echo "::endgroup::"
    if [ $RC -eq 0 ]; then
      record "android-builtin-kotlin" PASS "$DUR" ""
    else
      ANY_FAIL=1
      record "android-builtin-kotlin" FAIL "$DUR" "$(first_error "$LOG")"
    fi
    cp "$OUT/gradle.properties.before-builtin-kotlin" android/gradle.properties
  fi
  if [ "$p" = "web" ]; then
    LOG="$OUT/build-web-wasm.log"
    echo "::group::web-wasm: flutter build web --wasm"
    START=$SECONDS
    flutter build web --release --wasm > "$LOG" 2>&1
    RC=$?
    DUR=$((SECONDS - START))
    tail -n 80 "$LOG"
    echo "::endgroup::"
    if [ $RC -eq 0 ]; then
      record "web-wasm" PASS "$DUR" ""
    else
      ANY_FAIL=1
      record "web-wasm" FAIL "$DUR" "$(first_error "$LOG")"
    fi
    # Flutter 3.50 flag: fail the build on the deprecated dart:html family.
    if flutter build web -h 2>/dev/null | grep -q "deprecated-js-interop"; then
      LOG="$OUT/build-web-no-deprecated-js-interop.log"
      echo "::group::web-no-deprecated-js-interop: flutter build web --no-deprecated-js-interop"
      START=$SECONDS
      flutter build web --release --no-deprecated-js-interop > "$LOG" 2>&1
      RC=$?
      DUR=$((SECONDS - START))
      tail -n 80 "$LOG"
      echo "::endgroup::"
      if [ $RC -eq 0 ]; then
        record "web-no-deprecated-js-interop" PASS "$DUR" ""
      else
        ANY_FAIL=1
        record "web-no-deprecated-js-interop" FAIL "$DUR" "$(first_error "$LOG")"
      fi
    fi
  fi
done

# Keep what explains a failure, drop the rest.
for f in android/settings.gradle.kts android/gradle/wrapper/gradle-wrapper.properties android/gradle.properties android/app/build.gradle.kts ios/Podfile macos/Podfile pubspec.yaml pubspec.lock; do
  [ -f "$f" ] && { mkdir -p "$OUT/app/$(dirname "$f")"; cp "$f" "$OUT/app/$f"; }
done

exit $ANY_FAIL
