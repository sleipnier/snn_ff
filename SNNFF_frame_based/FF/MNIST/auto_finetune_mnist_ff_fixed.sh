#!/usr/bin/env bash
set -euo pipefail

# Resumable 2-GPU auto-finetune runner for mnist_FF_train_v2.py
# Search spaces are centered around the current MNIST FF defaults:
#   tau ~ 2, tau_response ~ 2, tau_refractory ~ 10.
# It creates per-run directories under ./result/fintune/, skips completed runs,
# and maintains a leaderboard.csv.

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-./mnist_FF_train_v2.py}
RESULT_ROOT=${RESULT_ROOT:-./result/fintune}
DATA_DIR=${DATA_DIR:-/home/public03/yhxu/spikingjelly/dataset/MNIST}
GPUS_CSV=${GPUS_CSV:-cuda:0,cuda:1}

# Trainer defaults / sweep defaults
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4096}
NUM_WORKERS=${NUM_WORKERS:-4}
TIME_STEPS=${TIME_STEPS:-10}
HIDDEN_DIM=${HIDDEN_DIM:-500}
NUM_LAYERS=${NUM_LAYERS:-2}
SEED=${SEED:-2026}

# Trial budgets per neuron type
TRIALS_LIF=${TRIALS_LIF:-36}
TRIALS_ALIF=${TRIALS_ALIF:-36}
TRIALS_SRM=${TRIALS_SRM:-48}
TRIALS_DYNSRM=${TRIALS_DYNSRM:-48}

# Resume / recovery behavior
STALE_HOURS=${STALE_HOURS:-12}
RETRY_FAILED=${RETRY_FAILED:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$RESULT_ROOT" "$RESULT_ROOT/configs" "$RESULT_ROOT/runs" "$RESULT_ROOT/logs"
MANIFEST="$RESULT_ROOT/manifest.csv"
RESULTS_CSV="$RESULT_ROOT/tuning_results.csv"
LEADERBOARD_CSV="$RESULT_ROOT/leaderboard.csv"
LOCK_FILE="$RESULT_ROOT/.scheduler.lock"
SUMMARY_TXT="$RESULT_ROOT/README_auto_finetune.txt"
export RESULT_ROOT MANIFEST RESULTS_CSV LEADERBOARD_CSV LOCK_FILE

cat > "$SUMMARY_TXT" <<TXT
Auto-finetune workspace for MNIST FF-SNN.
- Configs:      $RESULT_ROOT/configs/
- Run folders:  $RESULT_ROOT/runs/
- Logs:         $RESULT_ROOT/logs/
- Manifest:     $MANIFEST
- Results:      $RESULTS_CSV
- Leaderboard:  $LEADERBOARD_CSV
TXT

init_workspace() {
  RESULT_ROOT="$RESULT_ROOT" MANIFEST="$MANIFEST" RESULTS_CSV="$RESULTS_CSV" LEADERBOARD_CSV="$LEADERBOARD_CSV" \
  TRIALS_LIF="$TRIALS_LIF" TRIALS_ALIF="$TRIALS_ALIF" TRIALS_SRM="$TRIALS_SRM" TRIALS_DYNSRM="$TRIALS_DYNSRM" \
  EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" TIME_STEPS="$TIME_STEPS" HIDDEN_DIM="$HIDDEN_DIM" NUM_LAYERS="$NUM_LAYERS" SEED="$SEED" \
  "$PYTHON_BIN" - <<'PY'
import csv, itertools, json, math, os, random
from collections import OrderedDict

result_root = os.environ['RESULT_ROOT']
manifest = os.environ['MANIFEST']
results_csv = os.environ['RESULTS_CSV']
leaderboard_csv = os.environ['LEADERBOARD_CSV']
configs_dir = os.path.join(result_root, 'configs')
runs_dir = os.path.join(result_root, 'runs')
os.makedirs(configs_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)

if os.path.exists(manifest):
    print(f'Manifest already exists: {manifest}')
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'run_name','model','best_test_acc','best_test_macro_f1','best_test_event_synops',
                'final_test_acc','final_test_macro_f1','final_test_event_synops',
                'csv_path','summary_path','best_model_path','final_model_path','status'
            ])
            writer.writeheader()
    if not os.path.exists(leaderboard_csv):
        with open(leaderboard_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'rank','run_name','model','best_test_acc','best_test_macro_f1','best_test_event_synops',
                'final_test_acc','final_test_macro_f1','final_test_event_synops','status'
            ])
            writer.writeheader()
    raise SystemExit(0)

seed = int(os.environ['SEED'])
rng = random.Random(seed)

common = OrderedDict([
    ('epochs', int(os.environ['EPOCHS'])),
    ('batch_size', int(os.environ['BATCH_SIZE'])),
    ('num_workers', int(os.environ['NUM_WORKERS'])),
    ('time_steps', int(os.environ['TIME_STEPS'])),
    ('hidden_dim', int(os.environ['HIDDEN_DIM'])),
    ('num_layers', int(os.environ['NUM_LAYERS'])),
    ('seed', seed),
])

# Search spaces centered around theory / current defaults.
common_spaces = OrderedDict([
    ('lr', [5e-4, 8e-4, 1e-3]),
    ('alpha', [4.0, 6.0, 8.0]),
    ('input_gain', [0.8, 1.0, 1.2, 1.5]),
    ('label_scale', [0.8, 1.0, 1.2]),
])

model_spaces = {
    'lif': OrderedDict([
        ('tau', [1.5, 2.0, 2.5, 3.0, 4.0]),
        ('v_threshold', [0.3, 0.5, 0.7, 1.0]),
        ('v_reset', [0.0]),
    ]),
    'alif': OrderedDict([
        ('tau', [1.5, 2.0, 2.5, 3.0, 4.0]),
        ('v_threshold', [0.3, 0.5, 0.7, 1.0]),
        ('v_reset', [0.0]),
    ]),
    'srm': OrderedDict([
        ('tau_response', [1.5, 2.0, 2.5, 3.0, 4.0]),
        ('tau_refractory', [6.0, 8.0, 10.0, 12.0, 16.0]),
        ('v_threshold', [0.3, 0.5, 0.7, 1.0]),
        ('v_reset', [0.0]),
    ]),
    'dynsrm': OrderedDict([
        ('tau_response', [1.5, 2.0, 2.5, 3.0, 4.0]),
        ('tau_refractory', [6.0, 8.0, 10.0, 12.0, 16.0]),
        ('v_threshold', [0.3, 0.5, 0.7, 1.0]),
        ('v_reset', [0.0]),
    ]),
}

budgets = {
    'lif': int(os.environ['TRIALS_LIF']),
    'alif': int(os.environ['TRIALS_ALIF']),
    'srm': int(os.environ['TRIALS_SRM']),
    'dynsrm': int(os.environ['TRIALS_DYNSRM']),
}

canonical = {
    'lif':   dict(model='lif', tau=2.0, v_threshold=0.5, v_reset=0.0, lr=8e-4, alpha=6.0, input_gain=1.0, label_scale=1.0),
    'alif':  dict(model='alif', tau=2.0, v_threshold=0.5, v_reset=0.0, lr=8e-4, alpha=6.0, input_gain=1.0, label_scale=1.0),
    'srm':   dict(model='srm', tau_response=2.0, tau_refractory=10.0, v_threshold=0.5, v_reset=0.0, lr=8e-4, alpha=6.0, input_gain=1.0, label_scale=1.0),
    'dynsrm':dict(model='dynsrm', tau_response=2.0, tau_refractory=10.0, v_threshold=0.5, v_reset=0.0, lr=8e-4, alpha=6.0, input_gain=1.0, label_scale=1.0),
}

fieldnames = [
    'run_name','config_path','run_dir','model','tau','tau_response','tau_refractory','v_threshold','v_reset',
    'lr','alpha','input_gain','label_scale','hidden_dim','num_layers','time_steps','batch_size','epochs','seed'
]
rows = []


def fmt(x):
    if isinstance(x, float):
        s = f"{x:.6g}"
    else:
        s = str(x)
    return s.replace('-', 'm').replace('.', 'p')


def choose_configs(model):
    merged = OrderedDict()
    merged.update(common_spaces)
    merged.update(model_spaces[model])
    keys = list(merged.keys())
    values = [merged[k] for k in keys]
    all_cfgs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    rng.shuffle(all_cfgs)

    chosen = []
    seen = set()

    # canonical first
    base = canonical[model].copy()
    tup = tuple((k, base.get(k, None)) for k in keys)
    seen.add(tup)
    chosen.append(base)

    for cfg in all_cfgs:
        full = {'model': model, **cfg}
        tup = tuple((k, full.get(k, None)) for k in keys)
        if tup in seen:
            continue
        seen.add(tup)
        chosen.append(full)
        if len(chosen) >= budgets[model]:
            break
    return chosen

for model in ('lif','alif','srm','dynsrm'):
    configs = choose_configs(model)
    for idx, cfg in enumerate(configs):
        cfg_full = OrderedDict(common)
        cfg_full['model'] = model
        for key in ('tau','tau_response','tau_refractory','v_threshold','v_reset','lr','alpha','input_gain','label_scale'):
            if key in cfg:
                cfg_full[key] = cfg[key]
            else:
                cfg_full[key] = None
        parts = [
            model,
            f"id{idx:03d}",
            f"tau{fmt(cfg_full['tau'])}" if cfg_full['tau'] is not None else '',
            f"tr{fmt(cfg_full['tau_response'])}" if cfg_full['tau_response'] is not None else '',
            f"trefr{fmt(cfg_full['tau_refractory'])}" if cfg_full['tau_refractory'] is not None else '',
            f"vth{fmt(cfg_full['v_threshold'])}",
            f"lr{fmt(cfg_full['lr'])}",
            f"a{fmt(cfg_full['alpha'])}",
            f"ig{fmt(cfg_full['input_gain'])}",
            f"ls{fmt(cfg_full['label_scale'])}",
        ]
        run_name = '__'.join([p for p in parts if p])
        run_dir = os.path.join(runs_dir, run_name)
        cfg_full['run_name'] = run_name
        cfg_full['run_dir'] = run_dir
        config_path = os.path.join(configs_dir, f"{run_name}.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg_full, f, indent=2)
        row = {k: cfg_full.get(k, None) for k in fieldnames}
        row['config_path'] = config_path
        row['run_dir'] = run_dir
        row['run_name'] = run_name
        rows.append(row)

with open(manifest, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(results_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'run_name','model','best_test_acc','best_test_macro_f1','best_test_event_synops',
        'final_test_acc','final_test_macro_f1','final_test_event_synops',
        'csv_path','summary_path','best_model_path','final_model_path','status'
    ])
    writer.writeheader()

with open(leaderboard_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'rank','run_name','model','best_test_acc','best_test_macro_f1','best_test_event_synops',
        'final_test_acc','final_test_macro_f1','final_test_event_synops','status'
    ])
    writer.writeheader()

print(f'Created manifest with {len(rows)} runs: {manifest}')
PY
}

claim_next_config() {
  local gpu="$1"
  local now_epoch
  now_epoch=$(date +%s)
  (
    flock -x 200
    GPU_ID="$gpu" NOW_EPOCH="$now_epoch" STALE_HOURS="$STALE_HOURS" RETRY_FAILED="$RETRY_FAILED" \
    RESULT_ROOT="$RESULT_ROOT" MANIFEST="$MANIFEST" "$PYTHON_BIN" - <<'PY'
import csv, json, os, sys, time
from pathlib import Path

manifest = os.environ['MANIFEST']
result_root = os.environ['RESULT_ROOT']
gpu = os.environ['GPU_ID']
now_epoch = int(os.environ['NOW_EPOCH'])
stale_seconds = int(float(os.environ['STALE_HOURS']) * 3600)
retry_failed = int(os.environ['RETRY_FAILED'])

with open(manifest, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for row in rows:
    run_dir = Path(row['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / 'DONE.json'
    failed = run_dir / 'FAILED.json'
    running = run_dir / 'RUNNING.json'
    if done.exists():
        continue
    if failed.exists() and not retry_failed:
        continue
    if running.exists():
        try:
            info = json.loads(running.read_text(encoding='utf-8'))
            start_time = int(info.get('start_time_epoch', 0))
        except Exception:
            start_time = int(running.stat().st_mtime)
        if now_epoch - start_time < stale_seconds:
            continue
        try:
            running.unlink()
        except FileNotFoundError:
            pass

    cfg_path = row['config_path']
    payload = {
        'gpu': gpu,
        'cfg_path': cfg_path,
        'run_dir': row['run_dir'],
        'run_name': row['run_name'],
        'start_time_epoch': now_epoch,
    }
    running.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(cfg_path)
    sys.exit(0)

print('')
PY
  ) 200>"$LOCK_FILE"
}

run_training() {
  local cfg_path="$1"
  local gpu="$2"
  GPU_ID="$gpu" DATA_DIR="$DATA_DIR" TRAIN_SCRIPT="$TRAIN_SCRIPT" BATCH_SIZE="$BATCH_SIZE" EPOCHS="$EPOCHS" NUM_WORKERS="$NUM_WORKERS" TIME_STEPS="$TIME_STEPS" EXTRA_ARGS="$EXTRA_ARGS" \
  "$PYTHON_BIN" - "$cfg_path" <<'PY'
import json, os, shlex, subprocess, sys
cfg_path = sys.argv[1]
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

cmd = [
    sys.executable,
    os.environ['TRAIN_SCRIPT'],
    '-device', os.environ['GPU_ID'],
    '-data-dir', os.environ['DATA_DIR'],
    '-out-dir', cfg['run_dir'],
    '-b', str(cfg['batch_size']),
    '-epochs', str(cfg['epochs']),
    '-j', str(cfg['num_workers']),
    '-T', str(cfg['time_steps']),
    '--seed', str(cfg['seed']),
    '--model', cfg['model'],
    '--hidden-dim', str(cfg['hidden_dim']),
    '--num-layers', str(cfg['num_layers']),
    '--lr', str(cfg['lr']),
    '--alpha', str(cfg['alpha']),
    '--label-scale', str(cfg['label_scale']),
    '--input-gain', str(cfg['input_gain']),
    '--v-threshold', str(cfg['v_threshold']),
    '--v-reset', str(cfg['v_reset']),
]
if cfg.get('tau') is not None:
    cmd += ['--tau', str(cfg['tau'])]
if cfg.get('tau_response') is not None:
    cmd += ['--tau-response', str(cfg['tau_response'])]
if cfg.get('tau_refractory') is not None:
    cmd += ['--tau-refractory', str(cfg['tau_refractory'])]

extra_args = os.environ.get('EXTRA_ARGS', '').strip()
if extra_args:
    cmd += shlex.split(extra_args)

print('Launching:', ' '.join(shlex.quote(c) for c in cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

record_result() {
  local cfg_path="$1"
  local status="$2"
  (
    flock -x 200
    STATUS="$status" RESULTS_CSV="$RESULTS_CSV" LEADERBOARD_CSV="$LEADERBOARD_CSV" "$PYTHON_BIN" - "$cfg_path" <<'PY'
import csv, glob, json, os, pandas as pd, sys
cfg_path = sys.argv[1]
status = os.environ['STATUS']
results_csv = os.environ['RESULTS_CSV']
leaderboard_csv = os.environ['LEADERBOARD_CSV']

with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
run_dir = cfg['run_dir']
run_name = cfg['run_name']
model = cfg['model']

summary_files = sorted(glob.glob(os.path.join(run_dir, '*_summary.json')))
csv_files = sorted(glob.glob(os.path.join(run_dir, '*.csv')))
summary = {}
if summary_files:
    with open(summary_files[-1], 'r', encoding='utf-8') as f:
        summary = json.load(f)
run_csv = None
if csv_files:
    # prefer the training csv, ignore confusion matrices if present
    candidates = [p for p in csv_files if not p.endswith('_best_test_confusion.csv') and not p.endswith('_final_test_confusion.csv') and 'confusions' not in p]
    if candidates:
        run_csv = candidates[-1]
    else:
        run_csv = csv_files[-1]

best_acc = ''
best_f1 = ''
best_event_synops = ''
final_acc = ''
final_f1 = ''
final_event_synops = ''
if run_csv and os.path.exists(run_csv):
    df = pd.read_csv(run_csv)
    if len(df) > 0:
        best_idx = int(df['best_test_acc'].idxmax()) if 'best_test_acc' in df.columns else int(df['test_acc'].idxmax())
        best_row = df.iloc[best_idx]
        final_row = df.iloc[-1]
        best_acc = float(best_row.get('best_test_acc', best_row.get('test_acc', 0.0)))
        best_f1 = float(best_row.get('best_test_macro_f1', best_row.get('test_macro_f1', 0.0)))
        best_event_synops = float(best_row.get('test_event_synops', 0.0))
        final_acc = float(final_row.get('test_acc', 0.0))
        final_f1 = float(final_row.get('test_macro_f1', 0.0))
        final_event_synops = float(final_row.get('test_event_synops', 0.0))

record = {
    'run_name': run_name,
    'model': model,
    'best_test_acc': best_acc,
    'best_test_macro_f1': best_f1,
    'best_test_event_synops': best_event_synops,
    'final_test_acc': final_acc,
    'final_test_macro_f1': final_f1,
    'final_test_event_synops': final_event_synops,
    'csv_path': run_csv or '',
    'summary_path': summary_files[-1] if summary_files else '',
    'best_model_path': summary.get('best_model_path', ''),
    'final_model_path': summary.get('final_model_path', ''),
    'status': status,
}

fieldnames = list(record.keys())
rows = []
if os.path.exists(results_csv):
    with open(results_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('run_name') != run_name:
                rows.append(row)
rows.append(record)
with open(results_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

ok_rows = [r for r in rows if r.get('status') == 'DONE']
def as_float(x):
    try:
        return float(x)
    except Exception:
        return float('-inf')
ok_rows.sort(key=lambda r: (-as_float(r.get('best_test_acc')), -as_float(r.get('best_test_macro_f1')), as_float(r.get('best_test_event_synops')) if r.get('best_test_event_synops') not in ('', None) else float('inf'), r.get('run_name')))

leader_fields = ['rank','run_name','model','best_test_acc','best_test_macro_f1','best_test_event_synops','final_test_acc','final_test_macro_f1','final_test_event_synops','status']
with open(leaderboard_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=leader_fields)
    writer.writeheader()
    for rank, row in enumerate(ok_rows, start=1):
        writer.writerow({
            'rank': rank,
            'run_name': row['run_name'],
            'model': row['model'],
            'best_test_acc': row['best_test_acc'],
            'best_test_macro_f1': row['best_test_macro_f1'],
            'best_test_event_synops': row['best_test_event_synops'],
            'final_test_acc': row['final_test_acc'],
            'final_test_macro_f1': row['final_test_macro_f1'],
            'final_test_event_synops': row['final_test_event_synops'],
            'status': row['status'],
        })
PY
  ) 200>"$LOCK_FILE"
}

finish_run() {
  local cfg_path="$1"
  local status="$2"
  local gpu="$3"
  local message="$4"
  STATUS="$status" GPU_ID="$gpu" MSG="$message" "$PYTHON_BIN" - "$cfg_path" <<'PY'
import json, os, sys, time
cfg_path = sys.argv[1]
status = os.environ['STATUS']
gpu = os.environ['GPU_ID']
msg = os.environ['MSG']
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
run_dir = cfg['run_dir']
running = os.path.join(run_dir, 'RUNNING.json')
if os.path.exists(running):
    os.remove(running)
out = {
    'run_name': cfg['run_name'],
    'gpu': gpu,
    'status': status,
    'end_time_epoch': int(time.time()),
    'message': msg,
}
out_path = os.path.join(run_dir, f'{status}.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
PY
  record_result "$cfg_path" "$status"
}

worker_loop() {
  local gpu="$1"
  while true; do
    local cfg_path
    cfg_path=$(claim_next_config "$gpu")
    if [[ -z "$cfg_path" ]]; then
      echo "[$gpu] No pending runs left."
      break
    fi

    local run_name
    run_name=$("$PYTHON_BIN" - "$cfg_path" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg['run_name'])
PY
)

    local log_path="$RESULT_ROOT/logs/${run_name}.log"
    echo "[$gpu] Starting $run_name"
    set +e
    run_training "$cfg_path" "$gpu" > >(tee "$log_path") 2>&1
    local rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      finish_run "$cfg_path" "DONE" "$gpu" "completed successfully"
      echo "[$gpu] Finished $run_name"
    else
      finish_run "$cfg_path" "FAILED" "$gpu" "training exited with code $rc"
      echo "[$gpu] Failed $run_name (exit $rc)"
    fi
  done
}

init_workspace

IFS=',' read -r -a GPU_LIST <<< "$GPUS_CSV"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs specified in GPUS_CSV"
  exit 1
fi

pids=()
for gpu in "${GPU_LIST[@]}"; do
  gpu_trimmed="$(echo "$gpu" | xargs)"
  if [[ -z "$gpu_trimmed" ]]; then
    continue
  fi
  worker_loop "$gpu_trimmed" &
  pids+=("$!")
  sleep 2
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo
echo "Sweep finished."
echo "Manifest:    $MANIFEST"
echo "Results:     $RESULTS_CSV"
echo "Leaderboard: $LEADERBOARD_CSV"
