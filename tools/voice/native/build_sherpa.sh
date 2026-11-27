#!/usr/bin/env bash
# Builds the slim libsherpa-onnx-jni.so (arm64-v8a) shipped in app/src/main/jniLibs.
# Only KWS, online/offline ASR, Silero VAD, speech denoiser JNI and the OrtSession runner
# are compiled; ONNX Runtime is the full static build (all ONNX + contrib ops).
#
# Usage: tools/voice/native/build_sherpa.sh [output.so]
# Env:   OD_ANDROID_NDK (else sdk.dir from local.properties / ANDROID_HOME + ndk/<OD_NDK_VERSION>)
#        OD_SHERPA_WORK (default /tmp/od-sherpa-build), JOBS (default: cpu count)
set -euo pipefail

SHERPA_TAG=v2.0.3
SHERPA_COMMIT=38016c6e0622e90bf417c35e8b47e0eb805af417
# ORT 1.28.2 is sherpa-onnx v2.0.3's own pairing; OD_ORT_VERSION picks another pinned build.
ORT_VERSION=${OD_ORT_VERSION:-1.28.2}
case $ORT_VERSION in
  1.28.2) ORT_SHA256=bc68410f35f13df82c260830bf476ee5565f5efd0ec7622297829a929613c6c0 ;;
  1.31.0) ORT_SHA256=25f8a9649e775af85d4216e624ef1a820b1e1f26d82d5fc5145657fa9789c685 ;;
  *) echo "no pinned checksum for ORT $ORT_VERSION" >&2; exit 1 ;;
esac
ORT_ZIP=onnxruntime-android-arm64-v8a-static_lib-${ORT_VERSION}.zip
ORT_URL=https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${ORT_VERSION}/${ORT_ZIP}
NDK_VERSION=${OD_NDK_VERSION:-26.1.10909125}  # app ndkVersion

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=${1:-$REPO/app/src/main/jniLibs/arm64-v8a/libsherpa-onnx-jni.so}
WORK=${OD_SHERPA_WORK:-/tmp/od-sherpa-build}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

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

sha256() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || sha256sum "$1" | cut -d' ' -f1; }

mkdir -p "$WORK"
SRC=$WORK/sherpa-onnx-$SHERPA_TAG
if [ ! -d "$SRC/.git" ]; then
  git clone -q --depth 1 --branch "$SHERPA_TAG" https://github.com/k2-fsa/sherpa-onnx.git "$SRC"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$SHERPA_COMMIT" ] || { echo "unexpected sherpa-onnx commit" >&2; exit 1; }
git -C "$SRC" checkout -q -- sherpa-onnx/jni/CMakeLists.txt cmake/onnxruntime-android-aarch64-static.cmake \
  sherpa-onnx/csrc/online-speech-denoiser-stft-impl.h

BUILD=$WORK/build-arm64-v8a-ort$ORT_VERSION
mkdir -p "$BUILD"
if [ ! -f "$BUILD/$ORT_ZIP" ] || [ "$(sha256 "$BUILD/$ORT_ZIP")" != "$ORT_SHA256" ]; then
  curl -fsSL --retry 8 --retry-all-errors -C - -o "$BUILD/$ORT_ZIP" "$ORT_URL" || curl -fsSL --retry 8 -o "$BUILD/$ORT_ZIP" "$ORT_URL"
fi
[ "$(sha256 "$BUILD/$ORT_ZIP")" = "$ORT_SHA256" ] || { echo "ORT checksum mismatch" >&2; exit 1; }

# Trim the JNI target to the wrappers OverDrive ships, add the OrtSession runner, and link
# with section GC + a version script that exports only the JNI entry points.
cp "$HERE/od-ort-session.cc" "$SRC/sherpa-onnx/jni/od-ort-session.cc"
cp "$HERE/od-jni-symbols.lds" "$SRC/sherpa-onnx/jni/od-jni-symbols.lds"
python3 -I - "$SRC/sherpa-onnx/jni/CMakeLists.txt" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
sources = """set(sources
  common.cc
  jni.cc
  keyword-spotter.cc
  offline-recognizer.cc
  offline-speech-denoiser.cc
  offline-stream.cc
  online-recognizer.cc
  online-speech-denoiser.cc
  online-stream.cc
  speech-denoiser.cc
  voice-activity-detector.cc
  od-ort-session.cc
)"""
s, n = re.subn(r"set\(sources\n.*?\n\)", sources, s, count=1, flags=re.S)
assert n == 1, "sources block not found"
s = s.replace("${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-symbols.lds",
              "${CMAKE_CURRENT_SOURCE_DIR}/od-jni-symbols.lds -Wl,--gc-sections -Wl,--icf=all "
              "-Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384 -Wl,--build-id=none")
assert "od-jni-symbols.lds" in s
open(p, "w").write(s)
PY
python3 -I - "$SRC/cmake/onnxruntime-android-aarch64-static.cmake" "$ORT_VERSION" "$ORT_SHA256" <<'PY'
import re, sys
p, ver, sha = sys.argv[1:4]
s = open(p).read()
s = s.replace("1.28.2", ver)
s, n = re.subn(r'SHA256=[0-9a-f]{64}', 'SHA256=' + sha, s)
assert n == 1, "ORT hash not found"
open(p, "w").write(s)
PY

# The streaming denoiser's STFT is an O(n^2) DFT (~30% of GTCRN's CPU); use kissfft instead.
python3 -I - "$SRC/sherpa-onnx/csrc/online-speech-denoiser-stft-impl.h" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
start = s.index("class StreamingDft {")
end = s.index("class OnlineSpeechDenoiserStftImpl {")
fft = """class StreamingDft {
 public:
  explicit StreamingDft(int32_t n_fft)
      : n_fft_(n_fft), fwd_(n_fft, false), inv_(n_fft, true), buf_(n_fft) {}

  // output: interleaved (re, im) for bins 0..n/2 of the forward DFT (e^{-i}).
  void Forward(const float *input, float *output) {
    std::copy(input, input + n_fft_, buf_.begin());
    fwd_.Compute(buf_.data());
    const int32_t half = n_fft_ / 2;
    output[0] = buf_[0];
    output[1] = 0;
    for (int32_t k = 1; k < half; ++k) {
      output[2 * k] = buf_[2 * k];
      output[2 * k + 1] = buf_[2 * k + 1];
    }
    output[2 * half] = buf_[1];
    output[2 * half + 1] = 0;
  }

  // Real inverse with 1/n scaling; the imaginary parts of DC and Nyquist are ignored.
  void Inverse(const float *input, float *output) {
    const int32_t half = n_fft_ / 2;
    buf_[0] = input[0];
    buf_[1] = input[2 * half];
    for (int32_t k = 1; k < half; ++k) {
      buf_[2 * k] = input[2 * k];
      buf_[2 * k + 1] = input[2 * k + 1];
    }
    inv_.Compute(buf_.data());
    const float scale = 1.0f / n_fft_;
    for (int32_t n = 0; n < n_fft_; ++n) output[n] = buf_[n] * scale;
  }

 private:
  int32_t n_fft_ = 0;
  knf::Rfft fwd_;
  knf::Rfft inv_;
  std::vector<float> buf_;
};

"""
s = s[:start] + fft + s[end:]
s = s.replace('#include "kaldi-native-fbank/csrc/feature-window.h"',
              '#include "kaldi-native-fbank/csrc/feature-window.h"\n#include "kaldi-native-fbank/csrc/rfft.h"', 1)
assert s.count("knf::Rfft fwd_") == 1
open(p, "w").write(s)
PY

# API 26: >= 27 pulls in NNAPI headers, which the static ORT build lacks.
SECTIONS="-ffunction-sections -fdata-sections"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="$SECTIONS" -DCMAKE_CXX_FLAGS="$SECTIONS" \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=ON -DSHERPA_ONNX_ENABLE_C_API=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_RKNN=OFF -DSHERPA_ONNX_ENABLE_QNN=OFF \
  -DSHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY=OFF > "$BUILD/configure.log"
cmake --build "$BUILD" --target sherpa-onnx-jni -j "$JOBS" > "$BUILD/build.log"

LIB=$(find "$BUILD" -name libsherpa-onnx-jni.so -type f | head -1)
STRIP=$(ls "$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip | head -1)
mkdir -p "$(dirname "$OUT")"
"$STRIP" --strip-all -o "$OUT" "$LIB"
echo "sherpa-onnx $SHERPA_TAG + ORT $ORT_VERSION -> $OUT ($(wc -c < "$OUT") bytes)"
