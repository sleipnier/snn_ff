#!/usr/bin/env bash
set -euo pipefail

# NMNIST BP2 time-split finetune runner.
# Defaults reproduce the previous timestep sweep, but every *_CSV variable below
# can be widened to run a grid search.
#
# Examples:
#   bash run_bp2_time_split_finetune_nmnist.sh
#   LRS_CSV=5e-4,1e-3 V_THRESHOLDS_CSV=0.4,0.5 bash run_bp2_time_split_finetune_nmnist.sh
#
# Results: ./result/time_split_finetune/<model>/T<T>/<parameter-run>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/nmnist_bp2_time_split.py}"

GPUS_CSV="${GPUS_CSV:-cuda:0,cuda:1}"
DATA_DIR="${DATA_DIR:-/home/public03/yhxu/spikingjelly/dataset/NMNIST}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/result/time_split_finetune}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-2026}"
AMP="${AMP:-1}"
SCHEDULER="${SCHEDULER:-cosine}"
MOMENTUM="${MOMENTUM:-0.9}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TS_CSV="${TS_CSV:-2,4,8,10,16}"
MODELS_CSV="${MODELS_CSV:-lif,alif,srm,dynsrm}"
HIDDEN_DIMS_CSV="${HIDDEN_DIMS_CSV:-500}"
DEPTHS_CSV="${DEPTHS_CSV:-2}"
CRITERIA_CSV="${CRITERIA_CSV:-mse}"
OPTS_CSV="${OPTS_CSV:-adam}"
LRS_CSV="${LRS_CSV:-1e-3}"
LR_NEURONS_CSV="${LR_NEURONS_CSV:-5e-4}"
WEIGHT_DECAYS_CSV="${WEIGHT_DECAYS_CSV:-1e-3}"
TAUS_CSV="${TAUS_CSV:-2.0}"
V_THRESHOLDS_CSV="${V_THRESHOLDS_CSV:-0.5}"
V_RESETS_CSV="${V_RESETS_CSV:-0.0}"
TAU_RESPONSES_CSV="${TAU_RESPONSES_CSV:-2.0}"
TAU_REFRACTORIES_CSV="${TAU_REFRACTORIES_CSV:-10.0}"

read_csv() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$csv"
}

read_csv "$GPUS_CSV" GPUS
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS_CSV is empty. Use cuda:0 and/or cuda:1."
  exit 1
fi
for i in "${!GPUS[@]}"; do
  GPUS[$i]="${GPUS[$i]//[[:space:]]/}"
  case "${GPUS[$i]}" in
    cuda:0|cuda:1) ;;
    *)
      echo "ERROR: BP2 is limited to cuda:0 and cuda:1, got '${GPUS[$i]}'."
      exit 1
      ;;
  esac
done
if [[ -z "$MAX_PARALLEL" ]]; then
  MAX_PARALLEL="${#GPUS[@]}"
fi
if (( MAX_PARALLEL > ${#GPUS[@]} )); then
  MAX_PARALLEL="${#GPUS[@]}"
fi
if (( MAX_PARALLEL < 1 )); then
  echo "ERROR: MAX_PARALLEL must be >= 1."
  exit 1
fi

read_csv "$TS_CSV" TS
read_csv "$MODELS_CSV" MODELS
read_csv "$HIDDEN_DIMS_CSV" HIDDEN_DIMS
read_csv "$DEPTHS_CSV" DEPTHS
read_csv "$CRITERIA_CSV" CRITERIA
read_csv "$OPTS_CSV" OPTS
read_csv "$LRS_CSV" LRS
read_csv "$LR_NEURONS_CSV" LR_NEURONS
read_csv "$WEIGHT_DECAYS_CSV" WEIGHT_DECAYS
read_csv "$TAUS_CSV" TAUS
read_csv "$V_THRESHOLDS_CSV" V_THRESHOLDS
read_csv "$V_RESETS_CSV" V_RESETS
read_csv "$TAU_RESPONSES_CSV" TAU_RESPONSES
read_csv "$TAU_REFRACTORIES_CSV" TAU_REFRACTORIES

mkdir -p "$OUT_ROOT"

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
}

run_one() {
  local model="$1"
  local T="$2"
  local hidden_dim="$3"
  local depth="$4"
  local criterion="$5"
  local opt="$6"
  local lr="$7"
  local lr_neuron="$8"
  local weight_decay="$9"
  local tau="${10}"
  local v_threshold="${11}"
  local v_reset="${12}"
  local tau_response="${13}"
  local tau_refractory="${14}"
  local gpu="${15}"

  local run_name
  run_name="h${hidden_dim}_d${depth}_${criterion}_${opt}_lr${lr}_nlr${lr_neuron}_wd${weight_decay}_tau${tau}_thr${v_threshold}_reset${v_reset}_tr${tau_response}_tf${tau_refractory}_seed${SEED}"
  local out_dir="$OUT_ROOT/$model/T$T/$run_name"
  mkdir -p "$out_dir"

  local amp_flag=()
  if [[ "$AMP" == "1" ]]; then
    amp_flag=(--amp)
  fi

  echo "======================================================================"
  echo "[BP2-TimeSplit] model=$model | T=$T | device=$gpu | out_dir=$out_dir"
  "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    -device "$gpu" \
    -T "$T" \
    -b "$BATCH_SIZE" \
    -epochs "$EPOCHS" \
    -j "$NUM_WORKERS" \
    -data-dir "$DATA_DIR" \
    -out-dir "$out_dir" \
    --split-by time \
    --model "$model" \
    --hidden-dim "$hidden_dim" \
    --depth "$depth" \
    --criterion "$criterion" \
    --opt "$opt" \
    --lr "$lr" \
    --lr-neuron "$lr_neuron" \
    --weight-decay "$weight_decay" \
    --momentum "$MOMENTUM" \
    --scheduler "$SCHEDULER" \
    --tau "$tau" \
    --v-threshold "$v_threshold" \
    --v-reset "$v_reset" \
    --tau-response "$tau_response" \
    --tau-refractory "$tau_refractory" \
    --seed "$SEED" \
    "${amp_flag[@]}" \
    $EXTRA_ARGS
}

job_idx=0
for model in "${MODELS[@]}"; do
  for T in "${TS[@]}"; do
    for hidden_dim in "${HIDDEN_DIMS[@]}"; do
      for depth in "${DEPTHS[@]}"; do
        for criterion in "${CRITERIA[@]}"; do
          for opt in "${OPTS[@]}"; do
            for lr in "${LRS[@]}"; do
              for lr_neuron in "${LR_NEURONS[@]}"; do
                for weight_decay in "${WEIGHT_DECAYS[@]}"; do
                  for tau in "${TAUS[@]}"; do
                    for v_threshold in "${V_THRESHOLDS[@]}"; do
                      for v_reset in "${V_RESETS[@]}"; do
                        for tau_response in "${TAU_RESPONSES[@]}"; do
                          for tau_refractory in "${TAU_REFRACTORIES[@]}"; do
                            wait_for_slot
                            gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
                            run_one "$model" "$T" "$hidden_dim" "$depth" "$criterion" "$opt" "$lr" "$lr_neuron" "$weight_decay" "$tau" "$v_threshold" "$v_reset" "$tau_response" "$tau_refractory" "$gpu" &
                            job_idx=$((job_idx + 1))
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
wait

echo
echo "BP2 NMNIST time-split finetune finished."
echo "Results root: $OUT_ROOT"
