#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/yhxu/spikingjelly/spikingjelly/spikingjelly/SNNFF/BP/SHD/result/fintune/shd_bp}"
mkdir -p "$ROOT_DIR"
CONFIGS_DIR="$ROOT_DIR/configs"
RUNS_DIR="$ROOT_DIR/runs"
LOGS_DIR="$ROOT_DIR/logs"
mkdir -p "$CONFIGS_DIR" "$RUNS_DIR" "$LOGS_DIR"

MANIFEST="$ROOT_DIR/manifest.csv"
RESULTS="$ROOT_DIR/tuning_results.csv"
LEADERBOARD="$ROOT_DIR/leaderboard.csv"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-/home/yhxu/spikingjelly/spikingjelly/spikingjelly/SNNFF/BP/SHD/shd_bp_train_v1.py}"
DATA_DIR="${DATA_DIR:-/home/public03/yhxu/spikingjelly/dataset/SHD}"
GPUS_CSV="${GPUS_CSV:-cuda:0,cuda:1}"
SEED="${SEED:-2026}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
T_BINS="${T_BINS:-20}"
DOWNLOAD="${DOWNLOAD:-0}"

TRIALS_LIF="${TRIALS_LIF:-24}"
TRIALS_ALIF="${TRIALS_ALIF:-24}"
TRIALS_SRM="${TRIALS_SRM:-32}"
TRIALS_DYNSRM="${TRIALS_DYNSRM:-32}"

export ROOT_DIR TRAIN_SCRIPT DATA_DIR GPUS_CSV SEED EPOCHS BATCH_SIZE NUM_WORKERS T_BINS DOWNLOAD
export TRIALS_LIF TRIALS_ALIF TRIALS_SRM TRIALS_DYNSRM

python - <<'PY'
import csv, glob, json, os, random, subprocess, sys, time
from pathlib import Path

ROOT_DIR = os.environ["ROOT_DIR"]
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
RUNS_DIR = os.path.join(ROOT_DIR, "runs")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MANIFEST = os.path.join(ROOT_DIR, "manifest.csv")
RESULTS = os.path.join(ROOT_DIR, "tuning_results.csv")
LEADERBOARD = os.path.join(ROOT_DIR, "leaderboard.csv")

TRAIN_SCRIPT = os.environ["TRAIN_SCRIPT"]
DATA_DIR = os.environ["DATA_DIR"]
GPUS = [g.strip() for g in os.environ["GPUS_CSV"].split(",") if g.strip()]
SEED = int(os.environ["SEED"])
EPOCHS = int(os.environ["EPOCHS"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
NUM_WORKERS = int(os.environ["NUM_WORKERS"])
T_BINS = int(os.environ["T_BINS"])
DOWNLOAD = os.environ.get("DOWNLOAD", "0") == "1"

TRIALS = {
    "lif": int(os.environ["TRIALS_LIF"]),
    "alif": int(os.environ["TRIALS_ALIF"]),
    "srm": int(os.environ["TRIALS_SRM"]),
    "dynsrm": int(os.environ["TRIALS_DYNSRM"]),
}

random.seed(SEED)

def fmt(v):
    if isinstance(v, float):
        s = f"{v:g}".replace(".", "p").replace("-", "m")
        return s
    return str(v)

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sample_cfg(model):
    common = {
        "hidden_dim": random.choice([256, 384, 500]),
        "num_layers": random.choice([1, 2]),
        "lr": random.choice([3e-4, 8e-4, 1e-3, 2e-3]),
        "weight_decay": random.choice([0.0, 1e-5, 1e-4]),
        "input_gain": random.choice([0.8, 1.0, 1.2, 1.5]),
        "v_threshold": random.choice([0.3, 0.5, 0.7, 1.0]),
        "v_reset": 0.0,
    }
    if model in ("lif", "alif"):
        common["tau"] = random.choice([1.5, 2.0, 3.0, 4.0])
        common["tau_response"] = None
        common["tau_refractory"] = None
    else:
        common["tau"] = None
        common["tau_response"] = random.choice([1.5, 2.0, 3.0, 4.0])
        common["tau_refractory"] = random.choice([6.0, 8.0, 10.0, 12.0])
    return common


if not os.path.exists(MANIFEST):
    rows = []
    cfg_id = 0
    for model, n_trials in TRIALS.items():
        for _ in range(n_trials):
            cfg = sample_cfg(model)
            parts = [model, f"id{cfg_id:03d}"]
            for key in ["tau","tau_response","tau_refractory","v_threshold","lr","alpha","input_gain","label_scale","weight_decay","hidden_dim","num_layers","n_mels"]:
                if key in cfg and cfg[key] is not None:
                    tag = {"tau":"tau","tau_response":"tr","tau_refractory":"trefr","v_threshold":"vth","lr":"lr","alpha":"a","input_gain":"ig","label_scale":"ls","weight_decay":"wd","hidden_dim":"h","num_layers":"L","n_mels":"mel"}[key]
                    parts.append(f"{tag}{fmt(cfg[key])}")
            run_name = "__".join([p for p in parts if p])
            out_dir = os.path.join(RUNS_DIR, run_name)
            cfg_path = os.path.join(CONFIGS_DIR, run_name + ".json")
            payload = {
                "run_name": run_name,
                "model": model,
                "dataset": "shd",
                "method": "bp",
                "out_dir": out_dir,
                "cfg": cfg,
            }
            os.makedirs(out_dir, exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            rows.append({
                "run_name": run_name,
                "model": model,
                "cfg_path": cfg_path,
                "out_dir": out_dir,
                "status": "PENDING",
            })
            cfg_id += 1
    write_csv(MANIFEST, rows, ["run_name","model","cfg_path","out_dir","status"])

if not os.path.exists(RESULTS):
    write_csv(RESULTS, [], ["run_name","model","status","best_test_acc","best_test_macro_f1","best_test_event_synops","out_dir","csv_path","summary_path","cmd"])
if not os.path.exists(LEADERBOARD):
    write_csv(LEADERBOARD, [], ["run_name","model","status","best_test_acc","best_test_macro_f1","best_test_event_synops","out_dir","csv_path","summary_path","cmd"])

def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_manifest(rows):
    write_csv(MANIFEST, rows, ["run_name","model","cfg_path","out_dir","status"])

def parse_run_outputs(out_dir):
    csv_files = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    csv_files = [p for p in csv_files if not p.endswith("confusion.csv")]
    main_csv = None
    for p in csv_files:
        if "confusion" not in os.path.basename(p):
            main_csv = p
            break
    summary_files = sorted(glob.glob(os.path.join(out_dir, "*summary.json")))
    summary_path = summary_files[0] if summary_files else ""
    if main_csv is None:
        return None, summary_path
    import pandas as pd
    df = pd.read_csv(main_csv)
    if df.empty:
        return {"best_test_acc": None, "best_test_macro_f1": None, "best_test_event_synops": None}, summary_path
    sort_cols = [c for c in ["test_acc","test_macro_f1","test_event_synops"] if c in df.columns]
    row = df.sort_values(sort_cols, ascending=[False]*len(sort_cols)).iloc[0]
    return {
        "best_test_acc": float(row["test_acc"]) if "test_acc" in row else None,
        "best_test_macro_f1": float(row["test_macro_f1"]) if "test_macro_f1" in row else None,
        "best_test_event_synops": float(row["test_event_synops"]) if "test_event_synops" in row else None,
        "csv_path": main_csv,
    }, summary_path

def update_results():
    manifest = load_manifest()
    rows = []
    for m in manifest:
        out_dir = m["out_dir"]
        done_json = os.path.join(out_dir, "done.json")
        running_json = os.path.join(out_dir, "running.json")
        status = m["status"]
        if os.path.exists(done_json):
            status = "DONE"
        elif os.path.exists(running_json):
            try:
                age = time.time() - os.path.getmtime(running_json)
                if age > 12 * 3600:
                    os.remove(running_json)
                    status = "PENDING"
                else:
                    status = "RUNNING"
            except FileNotFoundError:
                status = "PENDING"
        else:
            status = "PENDING"
        m["status"] = status
        metrics, summary_path = parse_run_outputs(out_dir)
        with open(m["cfg_path"], "r", encoding="utf-8") as f:
            payload = json.load(f)
        cmd = " ".join(build_cmd(payload["cfg"], payload["out_dir"], payload["model"], "cuda:0")).replace("cuda:0", "<GPU>")
        rows.append({
            "run_name": m["run_name"],
            "model": m["model"],
            "status": status,
            "best_test_acc": "" if metrics is None or metrics["best_test_acc"] is None else metrics["best_test_acc"],
            "best_test_macro_f1": "" if metrics is None or metrics["best_test_macro_f1"] is None else metrics["best_test_macro_f1"],
            "best_test_event_synops": "" if metrics is None or metrics["best_test_event_synops"] is None else metrics["best_test_event_synops"],
            "out_dir": out_dir,
            "csv_path": "" if metrics is None else metrics.get("csv_path", ""),
            "summary_path": summary_path,
            "cmd": cmd,
        })
    write_csv(RESULTS, rows, ["run_name","model","status","best_test_acc","best_test_macro_f1","best_test_event_synops","out_dir","csv_path","summary_path","cmd"])
    done_rows = [r for r in rows if r["status"] == "DONE"]
    done_rows.sort(key=lambda r: (float(r["best_test_acc"]) if r["best_test_acc"] != "" else -1.0,
                                  float(r["best_test_macro_f1"]) if r["best_test_macro_f1"] != "" else -1.0,
                                  -(float(r["best_test_event_synops"]) if r["best_test_event_synops"] != "" else 1e30)), reverse=True)
    write_csv(LEADERBOARD, done_rows, ["run_name","model","status","best_test_acc","best_test_macro_f1","best_test_event_synops","out_dir","csv_path","summary_path","cmd"])
    save_manifest(manifest)

def build_cmd(cfg, out_dir, model, gpu):
    cmd = [sys.executable, TRAIN_SCRIPT, "-device", gpu, "-data-dir", DATA_DIR, "-out-dir", out_dir, "-b", str(BATCH_SIZE), "-epochs", str(EPOCHS), "-j", str(NUM_WORKERS), "-T", str(T_BINS), "--model", model]
    if "hidden_dim" in cfg: cmd += ["--hidden-dim", str(cfg["hidden_dim"])]
    if "num_layers" in cfg: cmd += ["--num-layers", str(cfg["num_layers"])]
    if "lr" in cfg: cmd += ['-lr', str(cfg["lr"])]
    if "weight_decay" in cfg: cmd += ["--weight-decay", str(cfg["weight_decay"])]
    if "input_gain" in cfg: cmd += ["--input-gain", str(cfg["input_gain"])]
    if "v_threshold" in cfg: cmd += ["--v-threshold", str(cfg["v_threshold"])]
    if "v_reset" in cfg: cmd += ["--v-reset", str(cfg["v_reset"])]
    if cfg.get("tau") is not None: cmd += ["--tau", str(cfg["tau"])]
    if cfg.get("tau_response") is not None: cmd += ["--tau-response", str(cfg["tau_response"])]
    if cfg.get("tau_refractory") is not None: cmd += ["--tau-refractory", str(cfg["tau_refractory"])]
    if DOWNLOAD:
        cmd.append("--download")
    return cmd

def load_cfg(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def next_pending(manifest):
    for m in manifest:
        if m["status"] == "PENDING":
            return m
    return None

update_results()
manifest = load_manifest()
procs = [None for _ in GPUS]

while True:
    manifest = load_manifest()
    all_done = True
    for i, gpu in enumerate(GPUS):
        proc = procs[i]
        if proc is not None and proc["p"].poll() is not None:
            rc = proc["p"].returncode
            out_dir = proc["out_dir"]
            running_json = os.path.join(out_dir, "running.json")
            if os.path.exists(running_json):
                os.remove(running_json)
            done_json = os.path.join(out_dir, "done.json")
            if rc == 0:
                with open(done_json, "w", encoding="utf-8") as f:
                    json.dump({"finished_at": time.time(), "returncode": rc}, f)
            procs[i] = None
            update_results()
            manifest = load_manifest()

        if procs[i] is None:
            pending = next_pending(manifest)
            if pending is not None:
                all_done = False
                cfg_payload = load_cfg(pending["cfg_path"])
                run_name = pending["run_name"]
                out_dir = pending["out_dir"]
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                log_path = os.path.join(LOGS_DIR, run_name + ".log")
                with open(os.path.join(out_dir, "running.json"), "w", encoding="utf-8") as f:
                    json.dump({"started_at": time.time(), "gpu": gpu}, f)
                cmd = build_cmd(cfg_payload["cfg"], out_dir, cfg_payload["model"], gpu)
                logf = open(log_path, "a", encoding="utf-8")
                logf.write("\n\n===== START " + time.strftime("%Y-%m-%d %H:%M:%S") + " =====\n")
                logf.write("CMD: " + " ".join(cmd) + "\n")
                logf.flush()
                p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
                procs[i] = {"p": p, "gpu": gpu, "out_dir": out_dir, "logf": logf}
                for m in manifest:
                    if m["run_name"] == run_name:
                        m["status"] = "RUNNING"
                        break
                save_manifest(manifest)
                update_results()
        else:
            all_done = False

    if all_done:
        break
    time.sleep(10)

for proc in procs:
    if proc is not None:
        proc["p"].wait()
        proc["logf"].close()

update_results()
print("Sweep finished.")
print("Manifest:   ", MANIFEST)
print("Results:    ", RESULTS)
print("Leaderboard:", LEADERBOARD)
PY
