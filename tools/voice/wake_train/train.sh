#!/bin/sh
# Stage-1 wake model retraining pipeline (microWakeWord, pinned). Recipe and results:
# private/docs/voice-eval/wake-retrain.md. Usage: train.sh <step>   (steps run in this order)
#   tts | music <shard.parquet> | pos | neg <libri.parquet> | val <dev-clean.parquet> | sayval | merge
#   mine <model.tflite> | prefix | train <run> | convert <run> | score <model.tflite>
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
EVAL_TOOLS=${EVAL_TOOLS:-$REPO/tools/voice/eval}   # the eval harness (carsim, front-ends, MicroFrontend port)
WT=${WT:-$HOME/voice-eval-data/wake-train}
PY=${PY:-$HOME/.venvs/od-wake-train/bin/python}    # tensorflow 2.21 + microwakeword @4665173 (see the doc)
MODEL="--pointwise_filters 64,64,64,64 --repeat_in_block 1,1,1,1 --mixconv_kernel_sizes [5],[7,11],[9,15],[23]
 --residual_connection 0,0,0,0 --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3"
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=4 TF_CPP_MIN_LOG_LEVEL=2
ulimit -n 10240 2>/dev/null || true
MF="nice -n 10 $PY -I $HERE/make_features.py --eval-tools $EVAL_TOOLS --repo $REPO"
step=$1; shift
case $step in
  tts) nice $PY -I "$HERE/gen_tts.py" --repo "$REPO" ;;
  music) nice $PY -I "$HERE/prep_music.py" "$1" "$WT/music" ;;
  pos) $MF --what pos --reps 4 && $MF --what pos --reps 3 --seed 78 --perturb --tag p && $MF --what hard ;;
  neg) $MF --what neg --libri "$1" --music-hours 3 --noise-hours 1 ;;
  val) $MF --what val --libri "$1" --music-hours 1 --noise-hours 0.5 &&
       nice $PY -I "$HERE/validate.py" build --eval-tools "$EVAL_TOOLS" --repo "$REPO" --per-cond 200 ;;
  sayval) nice $PY -I "$HERE/gen_sayval.py" --repo "$REPO" &&
       nice $PY -I "$HERE/validate.py" build --eval-tools "$EVAL_TOOLS" --repo "$REPO" --split sayval --per-cond 270 \
         --conds quiet,road10,road0,music_ser0,music_ser-10,roomir_road15 ;;
  merge) nice $PY -I "$HERE/merge_mmaps.py" "$WT/features" ;;
  mine) nice $PY -I "$HERE/mine.py" --eval-tools "$EVAL_TOOLS" --model "$1" --cutoff 0.2 --window 5 --tag m2 ;;
  prefix) nice $PY -I "$HERE/make_prefix.py" "$WT/features" --per 1 ;;
  train) R=$1; mkdir -p "$WT/runs"; cd "$WT/runs"
       $PY -I "$HERE/train_config.py" "$WT/features" "$WT/runs/$R" --steps 20000,10000 --lr 0.001,0.0002 \
         --neg-weight 5 --eval-every 10000 --time-mask 10 --freq-mask 5 > "$R.yaml"
       # shellcheck disable=SC2086
       nice -n 10 $PY -I -m microwakeword.model_train_eval --training_config "$R.yaml" --train 1 --restore_checkpoint 1 \
         --test_tf_nonstreaming 0 --test_tflite_nonstreaming 0 --test_tflite_nonstreaming_quantized 0 \
         --test_tflite_streaming 0 --test_tflite_streaming_quantized 0 --use_weights last_weights mixednet $MODEL > "$R.log" 2>&1 ;;
  convert) R=$1; cd "$WT/runs"
       # shellcheck disable=SC2086
       nice $PY -I "$HERE/convert.py" "$R.yaml" "$R/last_weights.weights.h5" "$R.tflite" mixednet $MODEL ;;
  score) nice $PY -I "$HERE/validate.py" score --eval-tools "$EVAL_TOOLS" --model "$1" \
         --cutoffs 0.15,0.2,0.25,0.3,0.4,0.5 --windows 5,7 &&
       nice $PY -I "$HERE/fire_timing.py" --eval-tools "$EVAL_TOOLS" "$1" 0.2 5 ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
