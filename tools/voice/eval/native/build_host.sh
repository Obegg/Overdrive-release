#!/bin/sh
# Builds the current SpeexDSP voice front-end from the repo sources as a host shared library.
# Usage: build_host.sh <repo-root> <out-dir>
set -eu
ROOT="$1"; OUT="$2"
CPP="$ROOT/app/src/main/cpp"; SPX="$CPP/third_party/speexdsp"
JH="${JAVA_HOME:-/Library/Java/JavaVirtualMachines/amazon-corretto-17.jdk/Contents/Home}"
case "$(uname)" in Darwin) EXT=dylib; JOS=darwin ;; *) EXT=so; JOS=linux ;; esac
mkdir -p "$OUT"
cc -O2 -shared -fPIC -fvisibility=hidden -DFLOATING_POINT -DUSE_KISS_FFT -DEXPORT= -DNDEBUG \
  -I"$CPP/voice" -I"$SPX/include" -I"$SPX/libspeexdsp" -I"$JH/include" -I"$JH/include/$JOS" \
  "$(dirname "$0")/speex_host.c" "$SPX/libspeexdsp/mdf.c" "$SPX/libspeexdsp/preprocess.c" \
  "$SPX/libspeexdsp/filterbank.c" "$SPX/libspeexdsp/fftwrap.c" "$SPX/libspeexdsp/kiss_fft.c" \
  "$SPX/libspeexdsp/kiss_fftr.c" -lm -o "$OUT/libod_speex.$EXT"
echo "$OUT/libod_speex.$EXT"
