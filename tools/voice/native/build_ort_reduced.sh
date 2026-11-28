#!/usr/bin/env bash
# Builds a static ONNX Runtime 1.28.2 for Android arm64-v8a with only the kernels listed in
# ort-ops.config (full, non-minimal build: .onnx and .ort both load). build_sherpa.sh links it
# when OD_ORT_DIR points at the install dir this prints.
#
# Usage: tools/voice/native/build_ort_reduced.sh
# Env:   OD_ANDROID_NDK / OD_NDK_VERSION as build_sherpa.sh, OD_ORT_WORK (default /tmp/od-ort-build),
#        JOBS (default: cpu count). Needs python3, cmake >= 3.28 and preferably ninja on PATH.
set -euo pipefail

ORT_TAG=v1.28.2
ORT_COMMIT=33ca9628233dc8f002435e868d4c2e9f82766ca1
# Upstream is fetched by commit; the mirror carries the same object when upstream is unreachable.
ORT_REMOTES="https://github.com/microsoft/onnxruntime.git https://github.com/intel/onnxruntime.git"
# Static-lib bundling CMake used for sherpa-onnx's own prebuilt ORT archives.
WRAPPER_REPO=https://github.com/csukuangfj/onnxruntime-build.git
WRAPPER_COMMIT=23ec55ae5db21fb82be6c0f8a19365699137ecb7
PROTOC_VERSION=21.12
NDK_VERSION=${OD_NDK_VERSION:-26.1.10909125}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
WORK=${OD_ORT_WORK:-/tmp/od-ort-build}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
CONFIG=$HERE/ort-ops.config

ANDROID_NDK=${OD_ANDROID_NDK:-}
if [ -z "$ANDROID_NDK" ]; then
  SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}
  if [ -f "$REPO/local.properties" ]; then
    P=$(sed -n 's/^sdk\.dir=//p' "$REPO/local.properties" | tail -1)
    [ -n "$P" ] && SDK=$P
  fi
  ANDROID_NDK=$SDK/ndk/$NDK_VERSION
fi
[ -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ] || { echo "NDK not found: $ANDROID_NDK" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PROTOC_ZIP=protoc-$PROTOC_VERSION-osx-aarch_64.zip
    PROTOC_SHA256=96839af0caed64352442fc8236f4bdf7c1cd6efcfaa98fa5db37307a73fc7c70 ;;
  *) echo "add the protoc $PROTOC_VERSION zip + sha256 for this host" >&2; exit 1 ;;
esac
sha256() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || sha256sum "$1" | cut -d' ' -f1; }

mkdir -p "$WORK"
SRC=$WORK/onnxruntime
if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$ORT_COMMIT" ]; then
  rm -rf "$SRC"
  git init -q "$SRC"
  for r in $ORT_REMOTES; do
    git -C "$SRC" fetch -q --depth 1 "$r" "$ORT_COMMIT" && break || true
  done
  git -C "$SRC" checkout -q "$ORT_COMMIT"
fi
[ "$(cat "$SRC/VERSION_NUMBER")" = "${ORT_TAG#v}" ] || { echo "unexpected ORT version" >&2; exit 1; }
# As the wrapper's own build script: no SOVERSION on the (unused) shared target.
sed -i.bak '/SOVERSION/d' "$SRC/cmake/onnxruntime.cmake"

WRAPPER=$WORK/onnxruntime-build
if [ "$(git -C "$WRAPPER" rev-parse HEAD 2>/dev/null)" != "$WRAPPER_COMMIT" ]; then
  rm -rf "$WRAPPER"
  git init -q "$WRAPPER"
  git -C "$WRAPPER" fetch -q --depth 1 "$WRAPPER_REPO" "$WRAPPER_COMMIT"
  git -C "$WRAPPER" checkout -q "$WRAPPER_COMMIT"
fi

PROTOC_DIR=$WORK/protoc-$PROTOC_VERSION
if [ ! -x "$PROTOC_DIR/bin/protoc" ]; then
  mkdir -p "$PROTOC_DIR"
  curl -fsSL --retry 8 -o "$PROTOC_DIR/$PROTOC_ZIP" \
    "https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOC_VERSION/$PROTOC_ZIP"
  [ "$(sha256 "$PROTOC_DIR/$PROTOC_ZIP")" = "$PROTOC_SHA256" ] || { echo "protoc checksum mismatch" >&2; exit 1; }
  (cd "$PROTOC_DIR" && unzip -q -o "$PROTOC_ZIP")
fi

BUILD=$WORK/build-android-arm64-v8a-reduced
OUT=$WORK/onnxruntime-android-arm64-v8a-static_lib-${ORT_TAG#v}-reduced
mkdir -p "$BUILD"
# Comments out every kernel registration the config does not name (op_reduction.generated/).
python3 "$SRC/tools/ci_build/reduce_op_kernels.py" "$CONFIG" --cmake_build_dir "$BUILD" > "$BUILD/reduce.log" 2>&1

GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)
cmake "${GEN[@]}" -S "$WRAPPER/static_lib" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release \
  -DCMAKE_C_FLAGS=-g0 -DCMAKE_CXX_FLAGS=-g0 \
  -DCMAKE_INSTALL_PREFIX="$OUT" -DONNXRUNTIME_SOURCE_DIR="$SRC" -DCXX11_ABI=0 \
  --compile-no-warning-as-error \
  -Donnxruntime_BUILD_UNIT_TESTS=OFF -Donnxruntime_REDUCED_OPS_BUILD=ON \
  -DONNX_CUSTOM_PROTOC_EXECUTABLE="$PROTOC_DIR/bin/protoc" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_shared \
  > "$BUILD/configure.log" 2>&1
cmake --build "$BUILD" --config Release --parallel "$JOBS" > "$BUILD/build.log" 2>&1
cmake --install "$BUILD" --config Release > "$BUILD/install.log" 2>&1
[ -f "$OUT/lib/libonnxruntime.a" ] || { echo "libonnxruntime.a missing" >&2; exit 1; }
echo "ONNX Runtime ${ORT_TAG#v} (reduced ops) -> $OUT"
