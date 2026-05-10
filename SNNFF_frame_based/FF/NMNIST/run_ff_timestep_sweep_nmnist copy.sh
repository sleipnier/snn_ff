#!/usr/bin/env bash
set -euo pipefail

# NMNIST FF timestep sweep
# - models: lif/alif/srm/dynsrm
# - T: 2/4/8/10/16
# - epochs: 300
# - parallel on multi-GPU
# Results: ./result/timestep/<model>/T<T>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/nmnist_ff_compare_refined.py}"

GPUS_CSV="${GPUS_CSV:-cuda:0,cuda:1}"
DATA_DIR="${DATA_DIR:-/home/public03/yhxu/spikingjelly/dataset/NMNIST}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/result/timestep}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-2026}"

TS=(8 10 16)
MODELS=(alif)

mkdir -p "$OUT_ROOT"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS_CSV is empty. Example: GPUS_CSV=cuda:0,cuda:1"
  exit 1
fi

run_one() {
  local model="$1"
  local T="$2"
  local gpu="$3"
  local out_dir="$OUT_ROOT/$model/T$T"
  mkdir -p "$out_dir"

  # Best-known v2 setup (same architecture/recipe as current best runs).
  local hidden_dim=500
  local depth=2
  local lr=1e-3
  local weight_decay=1e-3
  local theta=1.0
  local label_scale=1.0
  local tau=2.0
  local v_threshold=0.5
  local v_reset=0.0
  local tau_response=2.0
  local tau_refractory=10.0
  local pre_gain=5.0
  local goodness_mode=activity_l2
  local loss_mode=swish
  local swish_alpha=6.0
  local margin=1.0
  local neg_mode=hard
  local eval_every=1

  echo "======================================================================"
  echo "[FF-NMNIST] model=$model | T=$T | device=$gpu | out_dir=$out_dir"
  python "$TRAIN_SCRIPT" \
    -device "$gpu" \
    -T "$T" \
    -b "$BATCH_SIZE" \
    -epochs "$EPOCHS" \
    -j "$NUM_WORKERS" \
    -data-dir "$DATA_DIR" \
    -out-dir "$out_dir" \
    --model "$model" \
    --hidden-dim "$hidden_dim" \
    --depth "$depth" \
    --lr "$lr" \
    --weight-decay "$weight_decay" \
    --theta "$theta" \
    --label-scale "$label_scale" \
    --tau "$tau" \
    --v-threshold "$v_threshold" \
    --v-reset "$v_reset" \
    --tau-response "$tau_response" \
    --tau-refractory "$tau_refractory" \
    --pre-gain "$pre_gain" \
    --goodness-mode "$goodness_mode" \
    --loss-mode "$loss_mode" \
    --swish-alpha "$swish_alpha" \
    --margin "$margin" \
    --neg-mode "$neg_mode" \
    --eval-every "$eval_every" \
    --seed "$SEED"
}

job_idx=0
for model in "${MODELS[@]}"; do
  for T in "${TS[@]}"; do
    while (( $(jobs -rp | wc -l) >= ${#GPUS[@]} )); do
      wait -n
    done
    gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
    run_one "$model" "$T" "$gpu" &
    job_idx=$((job_idx + 1))
  done
done
wait

echo
echo "FF NMNIST timestep sweep finished."
echo "Results root: $OUT_ROOT"
