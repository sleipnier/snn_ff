#!/usr/bin/env bash
set -euo pipefail

# Sweep FF-SNN across time steps T for 4 neuron types.
# Results are written under: ./result/timestep/<model>/T<T>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/mnist_FF_train_v2.py}"

GPUS_CSV="${GPUS_CSV:-cuda:0,cuda:1}"
DATA_DIR="${DATA_DIR:-/home/public03/yhxu/spikingjelly/dataset/MNIST}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/result/timestep}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-2026}"

TS=(2 4 8 10)
MODELS=(lif alif srm dynsrm)

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

  # Best parameters from existing FF best runs.
  local hidden_dim=500
  local num_layers=2
  local weight_decay=0.0
  local input_gain=1.0
  local v_threshold=0.5
  local v_reset=0.0
  local tau=2.0
  local tau_response=2.0
  local alpha=6.0
  local label_scale=1.0

  local lr=0.0008
  local tau_refractory=2.0
  if [[ "$model" == "srm" || "$model" == "dynsrm" ]]; then
    lr=1e-6
    tau_refractory=12.0
  fi

  echo "=============================================================="
  echo "[FF] model=$model | T=$T | device=$gpu | out_dir=$out_dir"
  python "$TRAIN_SCRIPT" \
    -device "$gpu" \
    -data-dir "$DATA_DIR" \
    -out-dir "$out_dir" \
    -b "$BATCH_SIZE" \
    -epochs "$EPOCHS" \
    -j "$NUM_WORKERS" \
    -T "$T" \
    --seed "$SEED" \
    --model "$model" \
    --hidden-dim "$hidden_dim" \
    --num-layers "$num_layers" \
    --tau "$tau" \
    --v-threshold "$v_threshold" \
    --v-reset "$v_reset" \
    --tau-response "$tau_response" \
    --tau-refractory "$tau_refractory" \
    --input-gain "$input_gain" \
    --lr "$lr" \
    --weight-decay "$weight_decay" \
    --alpha "$alpha" \
    --label-scale "$label_scale"
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
echo "FF timestep sweep finished."
echo "Results root: $OUT_ROOT"
