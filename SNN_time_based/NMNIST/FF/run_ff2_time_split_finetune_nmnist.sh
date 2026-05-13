#!/usr/bin/env bash
set -euo pipefail

# NMNIST FF2 time-split finetune runner.
# Defaults reproduce the previous timestep sweep, but every *_CSV variable below
# can be widened to run a grid search.
#
# Examples:
#   bash run_ff2_time_split_finetune_nmnist.sh
#   LRS_CSV=5e-4,1e-3 PRE_GAINS_CSV=4.0,5.0,6.0 bash run_ff2_time_split_finetune_nmnist.sh
#
# Results: ./result/time_split_finetune/<model>/T<T>/<parameter-run>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/nmnist_ff2_time_split.py}"

GPUS_CSV="${GPUS_CSV:-cuda:0,cuda:1}"
DATA_DIR="${DATA_DIR:-/home/public03/yhxu/spikingjelly/dataset/NMNIST}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/result/time_split_finetune}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-2026}"
EVAL_EVERY="${EVAL_EVERY:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TS_CSV="${TS_CSV:-2,4,8,10,16}"
MODELS_CSV="${MODELS_CSV:-srm,dynsrm}"
HIDDEN_DIMS_CSV="${HIDDEN_DIMS_CSV:-500}"
DEPTHS_CSV="${DEPTHS_CSV:-2}"
LRS_CSV="${LRS_CSV:-1e-3}"
WEIGHT_DECAYS_CSV="${WEIGHT_DECAYS_CSV:-1e-3}"
THETAS_CSV="${THETAS_CSV:-1.0}"
LABEL_SCALES_CSV="${LABEL_SCALES_CSV:-1.0}"
TAUS_CSV="${TAUS_CSV:-2.0}"
V_THRESHOLDS_CSV="${V_THRESHOLDS_CSV:-0.5}"
V_RESETS_CSV="${V_RESETS_CSV:-0.0}"
TAU_RESPONSES_CSV="${TAU_RESPONSES_CSV:-2.0}"
TAU_REFRACTORIES_CSV="${TAU_REFRACTORIES_CSV:-10.0}"
PRE_GAINS_CSV="${PRE_GAINS_CSV:-5.0}"
GOODNESS_MODES_CSV="${GOODNESS_MODES_CSV:-activity_l2}"
LOSS_MODES_CSV="${LOSS_MODES_CSV:-swish}"
SWISH_ALPHAS_CSV="${SWISH_ALPHAS_CSV:-6.0}"
MARGINS_CSV="${MARGINS_CSV:-1.0}"
NEG_MODES_CSV="${NEG_MODES_CSV:-hard}"
NORMALIZE_INPUT="${NORMALIZE_INPUT:-0}"

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
      echo "ERROR: FF2 is limited to cuda:0 and cuda:1, got '${GPUS[$i]}'."
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
read_csv "$LRS_CSV" LRS
read_csv "$WEIGHT_DECAYS_CSV" WEIGHT_DECAYS
read_csv "$THETAS_CSV" THETAS
read_csv "$LABEL_SCALES_CSV" LABEL_SCALES
read_csv "$TAUS_CSV" TAUS
read_csv "$V_THRESHOLDS_CSV" V_THRESHOLDS
read_csv "$V_RESETS_CSV" V_RESETS
read_csv "$TAU_RESPONSES_CSV" TAU_RESPONSES
read_csv "$TAU_REFRACTORIES_CSV" TAU_REFRACTORIES
read_csv "$PRE_GAINS_CSV" PRE_GAINS
read_csv "$GOODNESS_MODES_CSV" GOODNESS_MODES
read_csv "$LOSS_MODES_CSV" LOSS_MODES
read_csv "$SWISH_ALPHAS_CSV" SWISH_ALPHAS
read_csv "$MARGINS_CSV" MARGINS
read_csv "$NEG_MODES_CSV" NEG_MODES

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
  local lr="$5"
  local weight_decay="$6"
  local theta="$7"
  local label_scale="$8"
  local tau="$9"
  local v_threshold="${10}"
  local v_reset="${11}"
  local tau_response="${12}"
  local tau_refractory="${13}"
  local pre_gain="${14}"
  local goodness_mode="${15}"
  local loss_mode="${16}"
  local swish_alpha="${17}"
  local margin="${18}"
  local neg_mode="${19}"
  local gpu="${20}"

  local run_name
  run_name="h${hidden_dim}_d${depth}_lr${lr}_wd${weight_decay}_theta${theta}_label${label_scale}_tau${tau}_thr${v_threshold}_reset${v_reset}_tr${tau_response}_tf${tau_refractory}_gain${pre_gain}_${goodness_mode}_${loss_mode}_alpha${swish_alpha}_margin${margin}_${neg_mode}_seed${SEED}"
  local out_dir="$OUT_ROOT/$model/T$T/$run_name"
  mkdir -p "$out_dir"

  local normalize_flag=()
  if [[ "$NORMALIZE_INPUT" == "1" ]]; then
    normalize_flag=(--normalize-input)
  fi

  echo "======================================================================"
  echo "[FF2-TimeSplit] model=$model | T=$T | device=$gpu | out_dir=$out_dir"
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
    --eval-every "$EVAL_EVERY" \
    --seed "$SEED" \
    "${normalize_flag[@]}" \
    $EXTRA_ARGS
}

job_idx=0
for model in "${MODELS[@]}"; do
  for T in "${TS[@]}"; do
    for hidden_dim in "${HIDDEN_DIMS[@]}"; do
      for depth in "${DEPTHS[@]}"; do
        for lr in "${LRS[@]}"; do
          for weight_decay in "${WEIGHT_DECAYS[@]}"; do
            for theta in "${THETAS[@]}"; do
              for label_scale in "${LABEL_SCALES[@]}"; do
                for tau in "${TAUS[@]}"; do
                  for v_threshold in "${V_THRESHOLDS[@]}"; do
                    for v_reset in "${V_RESETS[@]}"; do
                      for tau_response in "${TAU_RESPONSES[@]}"; do
                        for tau_refractory in "${TAU_REFRACTORIES[@]}"; do
                          for pre_gain in "${PRE_GAINS[@]}"; do
                            for goodness_mode in "${GOODNESS_MODES[@]}"; do
                              for loss_mode in "${LOSS_MODES[@]}"; do
                                for swish_alpha in "${SWISH_ALPHAS[@]}"; do
                                  for margin in "${MARGINS[@]}"; do
                                    for neg_mode in "${NEG_MODES[@]}"; do
                                      wait_for_slot
                                      gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
                                      run_one "$model" "$T" "$hidden_dim" "$depth" "$lr" "$weight_decay" "$theta" "$label_scale" "$tau" "$v_threshold" "$v_reset" "$tau_response" "$tau_refractory" "$pre_gain" "$goodness_mode" "$loss_mode" "$swish_alpha" "$margin" "$neg_mode" "$gpu" &
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
        done
      done
    done
  done
done
wait

echo
echo "FF2 NMNIST time-split finetune finished."
echo "Results root: $OUT_ROOT"
