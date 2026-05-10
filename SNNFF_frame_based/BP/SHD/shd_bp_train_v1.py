
#!/usr/bin/env python3
import argparse, json, os, sys, time, random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, random_split
from spikingjelly.activation_based import functional, neuron, surrogate

try:
    import psutil
except ImportError:
    psutil = None
try:
    import resource
except ImportError:
    resource = None

NUM_CLASSES = 20
INPUT_SIZE = 700

SPKINGJELLY_REPO_ROOT = "/home/yhxu/spikingjelly/spikingjelly"
if os.path.isdir(SPKINGJELLY_REPO_ROOT) and SPKINGJELLY_REPO_ROOT not in sys.path:
    sys.path.insert(0, SPKINGJELLY_REPO_ROOT)

try:
    from spikingjelly.datasets.shd import SpikingHeidelbergDigits
except Exception as e:
    raise ImportError(
        "Failed to import SpikingHeidelbergDigits from spikingjelly.datasets.shd. "
        f"Please check that your local package path exists: {SPKINGJELLY_REPO_ROOT}"
    ) from e


parser = argparse.ArgumentParser(description="SHD BP-SNN with four neuron types")
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/SHD")
parser.add_argument("-out-dir", type=str, default="./result")
parser.add_argument("-b", type=int, default=128)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-j", type=int, default=4)
parser.add_argument("-T", type=int, default=20)
parser.add_argument("--split-by", type=str, default="number", choices=["number","time"])
parser.add_argument("--val-ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=2026)

parser.add_argument("--model", type=str, default="lif", choices=["lif","alif","srm","dynsrm"])
parser.add_argument("--hidden-dim", type=int, default=500)
parser.add_argument("--num-layers", type=int, default=2, choices=[1,2,3])
parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v-threshold", type=float, default=0.5)
parser.add_argument("--v-reset", type=float, default=0.0)
parser.add_argument("--tau-response", type=float, default=2.0)
parser.add_argument("--tau-refractory", type=float, default=10.0)
parser.add_argument("--input-gain", type=float, default=1.0)

parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("-opt", type=str, default="adam", choices=["adam","sgd"])
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("--loss", type=str, default="mse", choices=["mse","ce"])
parser.add_argument("--cosine-scheduler", action="store_true")
parser.add_argument("-resume", type=str, default="")
parser.add_argument("-amp", action="store_true")
parser.add_argument("--save-test-confusion-every-epoch", action="store_true")
args, _ = parser.parse_known_args()
print(args)

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

def synchronize_if_needed(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

def get_process_memory_bytes():
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss) if sys.platform == "darwin" else int(rss)*1024
    return 0

def bytes_to_mb(x:int)->float:
    return float(x)/(1024.0*1024.0)

def update_confusion_matrix(confusion, pred, target):
    idx = target * NUM_CLASSES + pred
    confusion += torch.bincount(idx, minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

def macro_classification_metrics(confusion):
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    predicted = confusion.sum(0)
    actual = confusion.sum(1)
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2.0 * precision * recall / denom, torch.zeros_like(tp))
    return {"macro_precision": float(precision.mean().item()), "macro_recall": float(recall.mean().item()), "macro_f1": float(f1.mean().item())}

def init_activity_summary(num_layers):
    return {"layers":[{"spike_count":0.0,"num_elements":0.0,"input_nonzero_count":0.0,"input_num_elements":0.0,"dense_synops":0.0,"event_synops":0.0} for _ in range(num_layers)],
            "total_spike_count":0.0,"total_num_elements":0.0,"total_dense_synops":0.0,"total_event_synops":0.0}

def activity_summary_to_metrics(prefix, summary):
    metrics = {}
    if summary is None:
        return metrics
    metrics[f"{prefix}_total_spikes"] = float(summary["total_spike_count"])
    metrics[f"{prefix}_global_spike_rate"] = float(summary["total_spike_count"]) / max(float(summary["total_num_elements"]), 1.0)
    metrics[f"{prefix}_dense_synops"] = float(summary["total_dense_synops"])
    metrics[f"{prefix}_event_synops"] = float(summary["total_event_synops"])
    metrics[f"{prefix}_energy_proxy_synops"] = float(summary["total_event_synops"])
    metrics[f"{prefix}_event_to_dense_ratio"] = float(summary["total_event_synops"]) / max(float(summary["total_dense_synops"]), 1.0)
    for i, layer in enumerate(summary["layers"]):
        metrics[f"{prefix}_layer_{i}_spike_count"] = float(layer["spike_count"])
        metrics[f"{prefix}_layer_{i}_spike_rate"] = float(layer["spike_count"]) / max(float(layer["num_elements"]), 1.0)
        metrics[f"{prefix}_layer_{i}_active_input_rate"] = float(layer["input_nonzero_count"]) / max(float(layer["input_num_elements"]), 1.0)
        metrics[f"{prefix}_layer_{i}_dense_synops"] = float(layer["dense_synops"])
        metrics[f"{prefix}_layer_{i}_event_synops"] = float(layer["event_synops"])
        metrics[f"{prefix}_layer_{i}_event_to_dense_ratio"] = float(layer["event_synops"]) / max(float(layer["dense_synops"]), 1.0)
    return metrics

def save_confusion_matrix(path: str, confusion: torch.Tensor):
    pd.DataFrame(confusion.tolist(), index=[f"true_{i}" for i in range(NUM_CLASSES)], columns=[f"pred_{i}" for i in range(NUM_CLASSES)]).to_csv(path, index=True)

def save_json(path: str, payload: Dict[str, object]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def frame_tf(frames):
    if isinstance(frames, torch.Tensor):
        return frames.float().reshape(frames.shape[0], -1)
    return torch.from_numpy(frames).float().reshape(frames.shape[0], -1)


train_all = SpikingHeidelbergDigits(root=args.data_dir, train=True, data_type="frame", frames_number=args.T, split_by=args.split_by, transform=frame_tf)
test_dataset = SpikingHeidelbergDigits(root=args.data_dir, train=False, data_type="frame", frames_number=args.T, split_by=args.split_by, transform=frame_tf)
if args.val_ratio > 0:
    n_total = len(train_all)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_dataset, eval_dataset = random_split(train_all, [n_train, n_val], generator=g)
else:
    train_dataset = train_all
    eval_dataset = test_dataset

train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=max(256,args.b), shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=max(256,args.b), shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
print(f"SHD train={len(train_dataset)} eval={len(eval_dataset)} heldout_test={len(test_dataset)}")

def neuron_kwargs(out_features=None):
    base = dict(surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=args.v_threshold, v_reset=args.v_reset)
    if args.model == "lif":
        return neuron.LIFNode, dict(base, tau=args.tau)
    if args.model == "alif":
        cls = getattr(neuron, "ParametricLIFNode", neuron.LIFNode)
        return cls, dict(base, init_tau=args.tau)
    if args.model == "srm":
        return neuron.SRMNode, dict(base, tau_response=args.tau_response, tau_refractory=args.tau_refractory)
    cls = getattr(neuron, "DynamicSRMNode", neuron.SRMNode)
    kwargs = dict(base)
    if cls is neuron.SRMNode:
        kwargs.update(tau_response=args.tau_response, tau_refractory=args.tau_refractory)
    else:
        kwargs.update(init_tau_response=args.tau_response, init_tau_refractory=args.tau_refractory)
    return cls, kwargs

class BPSpikingNet(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int, num_classes: int):
        super().__init__()
        dims = [input_size] + [hidden_dim]*num_layers + [num_classes]
        self.fc_layers = nn.ModuleList()
        self.neuron_layers = nn.ModuleList()
        for i in range(len(dims)-1):
            fc = nn.Linear(dims[i], dims[i+1])
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fc_layers.append(fc)
            cls, kwargs = neuron_kwargs(dims[i+1])
            node = cls(**kwargs)
            if hasattr(node, "step_mode"):
                node.step_mode = "m"
            self.neuron_layers.append(node)
        self.layer_dims = dims

    def forward(self, x_tbf: torch.Tensor):
        h = x_tbf * float(args.input_gain)
        for fc, nd in zip(self.fc_layers, self.neuron_layers):
            h = nd(fc(h))
        return h

net = BPSpikingNet(INPUT_SIZE, args.hidden_dim, args.num_layers, NUM_CLASSES).to(device)
print(net)

if args.opt == "sgd":
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) if args.cosine_scheduler else None
scaler = amp.GradScaler() if args.amp else None

class SpikeStats:
    def __init__(self, net):
        self.counts = []
        self.elems = []
        self.handles = []
        for module in net.neuron_layers:
            idx = len(self.counts)
            self.counts.append(0.0); self.elems.append(0.0)
            self.handles.append(module.register_forward_hook(self.make_hook(idx)))
    def make_hook(self, idx):
        def hook(_m,_i,out):
            t = out[0] if isinstance(out,(tuple,list)) else out
            if torch.is_tensor(t):
                d = t.detach().float()
                self.counts[idx] += float(d.sum().item())
                self.elems[idx] += float(d.numel())
        return hook
    def close(self):
        for h in self.handles:
            h.remove()

def loss_from_rates(out_fr, y):
    if args.loss == "ce":
        return F.cross_entropy(out_fr, y)
    return F.mse_loss(out_fr, F.one_hot(y, NUM_CLASSES).float())

def run_epoch(loader, train: bool):
    net.train(train)
    stats = SpikeStats(net)
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_sum = correct = total = 0
    max_cpu = get_process_memory_bytes()
    input_nonzero_total = 0.0
    input_numel_total = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    start = time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x,y in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            x = x.float().to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_seq = x.permute(1,0,2).contiguous()
            input_nonzero_total += float(x_seq.ne(0).sum().item())
            input_numel_total += float(x_seq.numel())
            if train and scaler is not None:
                with amp.autocast():
                    out_seq = net(x_seq)
                    out_fr = out_seq.mean(0)
                    loss = loss_from_rates(out_fr, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out_seq = net(x_seq)
                out_fr = out_seq.mean(0)
                loss = loss_from_rates(out_fr, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
            pred = out_fr.argmax(1)
            n = y.numel()
            total += n
            loss_sum += float(loss.item()) * n
            correct += int((pred == y).sum().item())
            update_confusion_matrix(confusion, pred.detach().cpu(), y.detach().cpu())
            functional.reset_net(net)
            max_cpu = max(max_cpu, get_process_memory_bytes())
    synchronize_if_needed(device)
    elapsed = time.time() - start
    gpu_a = gpu_r = 0.0
    if device.type == "cuda":
        gpu_a = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        gpu_r = bytes_to_mb(torch.cuda.max_memory_reserved(device))
    cls = macro_classification_metrics(confusion)
    summary = init_activity_summary(len(net.neuron_layers))
    prev_spike, prev_numel, prev_features = input_nonzero_total, input_numel_total, INPUT_SIZE
    for i, out_features in enumerate(net.layer_dims[1:]):
        spike_count = stats.counts[i]; num_elements = stats.elems[i]
        dense_synops = float(args.T * total * prev_features * out_features)
        event_synops = float(prev_spike * out_features)
        layer = summary["layers"][i]
        layer["spike_count"] = spike_count
        layer["num_elements"] = num_elements
        layer["input_nonzero_count"] = prev_spike
        layer["input_num_elements"] = prev_numel
        layer["dense_synops"] = dense_synops
        layer["event_synops"] = event_synops
        summary["total_spike_count"] += spike_count
        summary["total_num_elements"] += num_elements
        summary["total_dense_synops"] += dense_synops
        summary["total_event_synops"] += event_synops
        prev_spike, prev_numel, prev_features = spike_count, num_elements, out_features
    stats.close()
    return {"loss":loss_sum/max(total,1),"acc":correct/max(total,1), "macro_precision":cls["macro_precision"], "macro_recall":cls["macro_recall"], "macro_f1":cls["macro_f1"],
            "samples":total,"time_sec":elapsed,"samples_per_sec":total/max(elapsed,1e-9),"latency_ms_per_sample":1000.0*elapsed/max(total,1),
            "cpu_memory_mb":bytes_to_mb(max_cpu),"gpu_memory_allocated_mb":gpu_a,"gpu_memory_reserved_mb":gpu_r,
            "confusion_matrix":confusion,"activity_summary":summary}

run_id = time.strftime("%Y%m%d_%H%M%S")
artifact_prefix = os.path.join(args.out_dir, f"SHD_{args.model}_BP_{run_id}")
csv_path = f"{artifact_prefix}.csv"
args_path = f"{artifact_prefix}_args.json"
summary_path = f"{artifact_prefix}_summary.json"
best_confusion_path = f"{artifact_prefix}_best_eval_confusion.csv"
final_confusion_path = f"{artifact_prefix}_final_eval_confusion.csv"
best_model_path = f"{artifact_prefix}_best_model.pth"
final_model_path = f"{artifact_prefix}_final_model.pth"
test_confusion_dir = f"{artifact_prefix}_eval_confusions"
if args.save_test_confusion_every_epoch:
    os.makedirs(test_confusion_dir, exist_ok=True)
save_json(args_path, vars(args))

csv_columns = ["run_id","dataset","method","epoch","model","hidden_dim","num_layers","T","split_by","val_ratio","tau","v_threshold","v_reset","tau_response","tau_refractory","input_gain","lr","weight_decay","train_loss","train_acc","train_macro_precision","train_macro_recall","train_macro_f1","test_loss","test_acc","test_macro_precision","test_macro_recall","test_macro_f1","best_test_acc","best_test_macro_f1","train_time_sec","test_time_sec","epoch_time_sec","train_speed","test_speed","train_latency_ms_per_sample","test_latency_ms_per_sample","train_cpu_memory_mb","test_cpu_memory_mb","train_gpu_memory_allocated_mb","train_gpu_memory_reserved_mb","test_gpu_memory_allocated_mb","test_gpu_memory_reserved_mb","test_confusion_path","best_test_confusion_path","final_test_confusion_path","best_model_path","final_model_path"] + [f"layer_{i}_loss" for i in range(len(net.neuron_layers))]
for prefix in ("train","test"):
    csv_columns.extend([f"{prefix}_total_spikes",f"{prefix}_global_spike_rate",f"{prefix}_dense_synops",f"{prefix}_event_synops",f"{prefix}_energy_proxy_synops",f"{prefix}_event_to_dense_ratio"])
    for i in range(len(net.neuron_layers)):
        csv_columns.extend([f"{prefix}_layer_{i}_spike_count",f"{prefix}_layer_{i}_spike_rate",f"{prefix}_layer_{i}_active_input_rate",f"{prefix}_layer_{i}_dense_synops",f"{prefix}_layer_{i}_event_synops",f"{prefix}_layer_{i}_event_to_dense_ratio"])
pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)

start_epoch = 0
best_test_acc = 0.0
best_test_macro_f1 = 0.0
best_heldout = None
if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    net.load_state_dict(ck["net"])
    optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and "lr_scheduler" in ck:
        scheduler.load_state_dict(ck["lr_scheduler"])
    start_epoch = int(ck["epoch"]) + 1
    best_test_acc = float(ck.get("best_test_acc", 0.0))
    best_test_macro_f1 = float(ck.get("best_test_macro_f1", 0.0))

last_eval_metrics = None
for epoch in range(start_epoch, args.epochs):
    epoch_start = time.time()
    train_metrics = run_epoch(train_loader, train=True)
    if scheduler is not None:
        scheduler.step()
    eval_metrics = run_epoch(eval_loader, train=False)
    heldout_metrics = run_epoch(test_loader, train=False)
    best_heldout = heldout_metrics
    last_eval_metrics = eval_metrics
    test_confusion_epoch_path = ""
    if args.save_test_confusion_every_epoch:
        test_confusion_epoch_path = os.path.join(test_confusion_dir, f"epoch_{epoch+1:03d}.csv")
        save_confusion_matrix(test_confusion_epoch_path, eval_metrics["confusion_matrix"])
    if eval_metrics["acc"] >= best_test_acc:
        best_test_acc = eval_metrics["acc"]
        save_confusion_matrix(best_confusion_path, eval_metrics["confusion_matrix"])
        torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_test_acc": best_test_acc, "best_test_macro_f1": max(best_test_macro_f1, eval_metrics["macro_f1"]), "args": vars(args)}, best_model_path)
    best_test_macro_f1 = max(best_test_macro_f1, eval_metrics["macro_f1"])
    row = {"run_id":run_id,"dataset":"shd","method":"bp","epoch":epoch+1,"model":args.model,"hidden_dim":args.hidden_dim,"num_layers":args.num_layers,"T":args.T,"split_by":args.split_by,"val_ratio":args.val_ratio,
           "tau":args.tau,"v_threshold":args.v_threshold,"v_reset":args.v_reset,"tau_response":args.tau_response,"tau_refractory":args.tau_refractory,"input_gain":args.input_gain,
           "lr":float(optimizer.param_groups[0]["lr"]),"weight_decay":args.weight_decay,"train_loss":train_metrics["loss"],"train_acc":train_metrics["acc"],"train_macro_precision":train_metrics["macro_precision"],"train_macro_recall":train_metrics["macro_recall"],"train_macro_f1":train_metrics["macro_f1"],
           "test_loss":eval_metrics["loss"],"test_acc":eval_metrics["acc"],"test_macro_precision":eval_metrics["macro_precision"],"test_macro_recall":eval_metrics["macro_recall"],"test_macro_f1":eval_metrics["macro_f1"],"best_test_acc":best_test_acc,"best_test_macro_f1":best_test_macro_f1,
           "train_time_sec":train_metrics["time_sec"],"test_time_sec":eval_metrics["time_sec"],"epoch_time_sec":time.time()-epoch_start,
           "train_speed":train_metrics["samples_per_sec"],"test_speed":eval_metrics["samples_per_sec"],"train_latency_ms_per_sample":train_metrics["latency_ms_per_sample"],"test_latency_ms_per_sample":eval_metrics["latency_ms_per_sample"],
           "train_cpu_memory_mb":train_metrics["cpu_memory_mb"],"test_cpu_memory_mb":eval_metrics["cpu_memory_mb"],"train_gpu_memory_allocated_mb":train_metrics["gpu_memory_allocated_mb"],"train_gpu_memory_reserved_mb":train_metrics["gpu_memory_reserved_mb"],"test_gpu_memory_allocated_mb":eval_metrics["gpu_memory_allocated_mb"],"test_gpu_memory_reserved_mb":eval_metrics["gpu_memory_reserved_mb"],
           "test_confusion_path":test_confusion_epoch_path,"best_test_confusion_path":best_confusion_path,"final_test_confusion_path":final_confusion_path,"best_model_path":best_model_path,"final_model_path":final_model_path}
    for i in range(len(net.neuron_layers)):
        row[f"layer_{i}_loss"] = 0.0
    row.update(activity_summary_to_metrics("train", train_metrics["activity_summary"]))
    row.update(activity_summary_to_metrics("test", eval_metrics["activity_summary"]))
    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)
    print(f"Epoch {epoch+1}/{args.epochs} | {args.model} | Train Acc/F1 {train_metrics['acc']*100:.2f}/{train_metrics['macro_f1']*100:.2f} | Eval Acc/F1 {eval_metrics['acc']*100:.2f}/{eval_metrics['macro_f1']*100:.2f} | Heldout {heldout_metrics['acc']*100:.2f} | Best Eval {best_test_acc*100:.2f}")

torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": args.epochs, "best_test_acc": best_test_acc, "best_test_macro_f1": best_test_macro_f1, "args": vars(args)}, final_model_path)
if last_eval_metrics is not None:
    save_confusion_matrix(final_confusion_path, last_eval_metrics["confusion_matrix"])
    save_json(summary_path, {"run_id": run_id, "dataset":"shd","method":"bp","csv_path":csv_path,"args_path":args_path,"best_model_path":best_model_path,"final_model_path":final_model_path,"best_eval_acc":best_test_acc,"best_eval_macro_f1":best_test_macro_f1,
                             "final_heldout_test_acc": best_heldout["acc"] if best_heldout else None, "final_heldout_test_macro_f1": best_heldout["macro_f1"] if best_heldout else None})
print("Done. CSV saved to:", csv_path)
