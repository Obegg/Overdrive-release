#!/bin/sh
# Downloads the open evaluation corpora into $VOICE_EVAL_DATA (default ~/voice-eval-data), outside the repo.
# PY = a python with numpy/soundfile/pyarrow (the harness venv). Each corpus lands in its own directory; re-running skips what is already there.
set -eu
D="${VOICE_EVAL_DATA:-$HOME/voice-eval-data}"
SLR="${OPENSLR_MIRROR:-https://openslr.elda.org/resources}"
MUSAN_MB="${MUSAN_MUSIC_MB:-200}"  # optional extra music: openslr mirrors are slow
mkdir -p "$D"

libri() {  # LibriSpeech (CC BY 4.0), openslr 12, via the Hugging Face parquet export (much faster)
  [ -d "$D/librispeech-hf/LibriSpeech/$1" ] && return 0
  mkdir -p "$D/librispeech-hf"
  split=$(echo "$1" | tr - .)
  curl -fsSL --retry 5 -o "$D/librispeech-hf/$split.parquet" \
    "https://huggingface.co/api/datasets/openslr/librispeech_asr/parquet/all/$split/0.parquet"
  "${PY:-python3}" -I "$(dirname "$0")/libri_from_parquet.py" "$D/librispeech-hf/$split.parquet" "$1" "$D/librispeech-hf"
  rm -f "$D/librispeech-hf/$split.parquet"
}

musan_music() {  # MUSAN (CC BY 4.0), openslr 17: music is first in the archive, so fetch only its head
  [ -f "$D/musan/.done" ] && return 0
  mkdir -p "$D/musan"
  # resumable byte range to a file, then decode the truncated gzip stream locally
  python3 -I "$(dirname "$0")/fetch_range.py" "$SLR/17/musan.tar.gz" "$((MUSAN_MB * 1000000))" "$D/musan/head.tar.gz.part"
  (gzip -dc < "$D/musan/head.tar.gz.part" 2>/dev/null || true) | (cd "$D/musan" && tar x 2>/dev/null || true)
  rm -f "$D/musan/head.tar.gz.part"
  # the last file is cut by the byte range: drop anything shorter than its RIFF header says
  python3 -I "$(dirname "$0")/prune_truncated_wavs.py" "$D/musan"
  touch "$D/musan/.done"
}

music() {  # Free Music Archive, commercially licensed tracks at 16 kHz (HF dataset, CC BY 4.0 + per-track CC)
  [ -f "$D/fma16k/tracks/credits.jsonl" ] && return 0
  mkdir -p "$D/fma16k"
  curl -fsSL --retry 5 -o "$D/fma16k/shard0.parquet" \
    "https://huggingface.co/datasets/benjamin-paine/free-music-archive-commercial-16khz-full/resolve/main/data/train-00000-of-00013.parquet"
  "${PY:-python3}" -I "$(dirname "$0")/fma_from_parquet.py" "$D/fma16k/shard0.parquet" "$D/fma16k/tracks" 300 60
  rm -f "$D/fma16k/shard0.parquet"
}

demand() {  # DEMAND (CC BY 4.0), zenodo 1227121: TCAR = inside a car, TBUS = inside a bus
  [ -d "$D/demand/$1" ] && return 0
  mkdir -p "$D/demand"
  curl -fsSL --retry 10 --retry-all-errors -C - -o "$D/demand/$1_16k.zip" \
    "https://zenodo.org/records/1227121/files/$1_16k.zip?download=1"
  (cd "$D/demand" && unzip -q "$1_16k.zip" && rm "$1_16k.zip")
}

rirs() {  # RIRS_NOISES (Apache 2.0), openslr 28: only the real measured RIRs, fetched by HTTP range
  [ -f "$D/rirs/.done" ] && return 0
  mkdir -p "$D/rirs"
  python3 -I "$(dirname "$0")/remote_unzip.py" "$SLR/28/rirs_noises.zip" "RIRS_NOISES/real_rirs_isotropic_noises/" "$D/rirs" noise
  touch "$D/rirs/.done"
}

tts() {  # Piper voices (LibriTTS-R / VCTK speakers, CC BY 4.0) and Kokoro v1.0 (Apache 2.0)
  mkdir -p "$D/tts-models"; cd "$D/tts-models"
  HF=https://huggingface.co/rhasspy/piper-voices/resolve/main/en
  for v in en_US/libritts_r/medium/en_US-libritts_r-medium en_GB/vctk/medium/en_GB-vctk-medium; do
    b=$(basename $v)
    [ -f $b.onnx ] || { curl -fsSL -o $b.onnx "$HF/$v.onnx"; curl -fsSL -o $b.onnx.json "$HF/$v.onnx.json"; }
  done
  K=https://huggingface.co/fastrtc/kokoro-onnx/resolve/main
  [ -f kokoro-v1.0.onnx ] || curl -fsSL -o kokoro-v1.0.onnx $K/kokoro-v1.0.onnx
  [ -f voices-v1.0.bin ] || curl -fsSL -o voices-v1.0.bin $K/voices-v1.0.bin
  cd - >/dev/null
}

smart_turn() {  # Smart Turn v3.2 CPU (BSD-2-Clause, pipecat-ai) for the endpointing comparison
  mkdir -p "$D/models"
  [ -f "$D/models/smart-turn-v3.2-cpu.onnx" ] || curl -fsSL -o "$D/models/smart-turn-v3.2-cpu.onnx" \
    https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
}

case "${1:-all}" in
  libri) libri test-clean; libri test-other ;;
  music) music ;;
  musan) musan_music ;;
  demand) demand TCAR; demand TBUS ;;
  rirs) rirs ;;
  tts) tts ;;
  smart_turn) smart_turn ;;
  all) tts; smart_turn; libri test-clean; libri test-other; music; demand TCAR; demand TBUS; rirs ;;
esac
