
#!/usr/bin/env python3
import argparse, json, os, sys, time, random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
SHD_INPUT_SIZE = 700
# v2: preserve all 700 SHD auditory channels and append 20 label channels.
INPUT_SIZE = SHD_INPUT_SIZE + NUM_CLASSES

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


parser = argparse.ArgumentParser(description="SHD FF-SNN v2: append label channels instead of overwriting SHD channels")
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/SHD")
parser.add_argument("-out-dir", type=str, default="./result")
parser.add_argument("-b", type=int, default=256)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-j", type=int, default=4)
parser.add_argument("-T", type=int, default=20, help="frames number")
parser.add_argument("--split-by", type=str, default="number", choices=["number", "time"])
parser.add_argument("--val-ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--download", action="store_true")

parser.add_argument("--model", type=str, default="lif", choices=["lif","alif","srm","dynsrm"])
parser.add_argument("--hidden-dim", type=int, default=500)
parser.add_argument("--num-layers", type=int, default=2, choices=[1,2,3])
parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v-threshold", type=float, default=0.5)
parser.add_argument("--v-reset", type=float, default=0.0)
parser.add_argument("--tau-response", type=float, default=2.0)
parser.add_argument("--tau-refractory", type=float, default=10.0)
parser.add_argument("--input-gain", type=float, default=1.0)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=6.0)
parser.add_argument("--label-scale", type=float, default=1.0)
parser.add_argument("--eval-subset", type=int, default=0)
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

def bytes_to_mb(x: int) -> float:
    return float(x)/(1024.0*1024.0)

def update_confusion_matrix(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
    idx = target * NUM_CLASSES + pred
    confusion += torch.bincount(idx, minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

def macro_classification_metrics(confusion: torch.Tensor):
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    predicted = confusion.sum(0)
    actual = confusion.sum(1)
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2.0 * precision * recall / denom, torch.zeros_like(tp))
    return {
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }

def init_activity_summary(num_layers: int):
    return {"layers": [{"spike_count":0.0,"num_elements":0.0,"input_nonzero_count":0.0,"input_num_elements":0.0,"dense_synops":0.0,"event_synops":0.0} for _ in range(num_layers)],
            "total_spike_count":0.0,"total_num_elements":0.0,"total_dense_synops":0.0,"total_event_synops":0.0}

def record_layer_activity(summary, layer_idx, x_seq, spk_seq, out_features):
    layer = summary["layers"][layer_idx]
    input_nonzero = float(x_seq.ne(0).sum().item())
    input_numel = float(x_seq.numel())
    spike_count = float(spk_seq.sum().item())
    num_elements = float(spk_seq.numel())
    dense_synops = float(x_seq.shape[0] * x_seq.shape[1] * x_seq.shape[2] * out_features)
    event_synops = input_nonzero * float(out_features)
    layer["spike_count"] += spike_count
    layer["num_elements"] += num_elements
    layer["input_nonzero_count"] += input_nonzero
    layer["input_num_elements"] += input_numel
    layer["dense_synops"] += dense_synops
    layer["event_synops"] += event_synops
    summary["total_spike_count"] += spike_count
    summary["total_num_elements"] += num_elements
    summary["total_dense_synops"] += dense_synops
    summary["total_event_synops"] += event_synops

def merge_activity_summaries(dst, src):
    for dl, sl in zip(dst["layers"], src["layers"]):
        for k in ("spike_count","num_elements","input_nonzero_count","input_num_elements","dense_synops","event_synops"):
            dl[k] += sl[k]
    for k in ("total_spike_count","total_num_elements","total_dense_synops","total_event_synops"):
        dst[k] += src[k]

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


def save_model_artifact(path: str, net_: nn.Module, extra_metadata: Optional[Dict[str, object]]=None):
    payload = {"model_state_dict": net_.state_dict(), "args": vars(args), "dims": dims, "model_class": net_.__class__.__name__, "raw_input_size": SHD_INPUT_SIZE, "input_size": INPUT_SIZE, "num_classes": NUM_CLASSES, "label_channels_appended": True}
    if extra_metadata is not None:
        payload["metadata"] = extra_metadata
    torch.save(payload, path)

def append_label_to_x_temporal(x: torch.Tensor, y: Union[torch.Tensor, int], num_classes=NUM_CLASSES, label_scale=1.0):
    """
    Append one-hot temporal label channels to SHD frames.

    v1 overwrote x[:, :, :NUM_CLASSES], which destroys the first 20 real SHD
    auditory channels. v2 preserves all original 700 SHD channels and appends
    20 additional label channels:

        x:       [B, T, 700]
        label:   [B, T, 20]
        return:  [B, T, 720]

    The active label channel is kept constant over all time frames. Its scale is
    matched to the per-sample maximum event-count/frame value, as in the v1
    overlay implementation.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x with shape [B, T, F], got {tuple(x.shape)}")

    B, T, F_in = x.shape
    if F_in != SHD_INPUT_SIZE:
        raise ValueError(
            f"Expected raw SHD input feature size {SHD_INPUT_SIZE}, got {F_in}. "
            "Do not pass already label-augmented tensors into this function."
        )

    if isinstance(y, int):
        y = torch.full((B,), y, device=x.device, dtype=torch.long)
    else:
        y = y.to(x.device, dtype=torch.long).view(-1)

    label = torch.zeros((B, T, num_classes), device=x.device, dtype=x.dtype)
    vmax = x.abs().amax(dim=(1, 2)) + 1e-6  # [B]
    b = torch.arange(B, device=x.device)
    label[b, :, y.clamp(0, num_classes - 1)] = vmax[b].unsqueeze(1) * float(label_scale)

    return torch.cat([x, label], dim=2)

def to_tbf(x_btf: torch.Tensor):
    return x_btf.permute(1,0,2).contiguous()

def goodness(h: torch.Tensor):
    return h.pow(2).mean(dim=1)

def swish_ff_loss(h_pos: torch.Tensor, h_neg: torch.Tensor, alpha: float):
    g_pos = goodness(h_pos)
    g_neg = goodness(h_neg)
    return F.silu(-alpha * (g_pos - g_neg)).mean()

def learning_rate_for_epoch(epoch: int, base_lr: float):
    if epoch >= 100:
        return 1e-6
    if epoch >= 75:
        return 1e-5
    if epoch >= 50:
        return 3e-4
    return float(base_lr)

def make_neuron_factory():
    common = dict(surrogate_function=surrogate.ATan(), detach_reset=True)
    if args.model == "lif":
        return neuron.LIFNode, dict(common, tau=args.tau, v_threshold=args.v_threshold, v_reset=args.v_reset)
    if args.model == "alif":
        cls = getattr(neuron, "ParametricLIFNode", neuron.LIFNode)
        return cls, dict(common, init_tau=args.tau, v_threshold=args.v_threshold, v_reset=args.v_reset)
    if args.model == "srm":
        return neuron.SRMNode, dict(common, tau_response=args.tau_response, tau_refractory=args.tau_refractory, v_threshold=args.v_threshold, v_reset=args.v_reset)
    cls = getattr(neuron, "DynamicSRMNode", neuron.SRMNode)
    kwargs = dict(common, v_threshold=args.v_threshold, v_reset=args.v_reset)
    if cls is neuron.SRMNode:
        kwargs.update(tau_response=args.tau_response, tau_refractory=args.tau_refractory)
    else:
        kwargs.update(init_tau_response=args.tau_response, init_tau_refractory=args.tau_refractory)
    return cls, kwargs

class FFSpikingLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, lr: float, weight_decay: float):
        super().__init__()
        node_cls, node_kwargs = make_neuron_factory()
        self.fc = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.neuron = node_cls(**node_kwargs)
        if hasattr(self.neuron, "step_mode"):
            self.neuron.step_mode = "m"
        self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def run(self, x_seq: torch.Tensor):
        functional.reset_net(self)
        cur_seq = self.fc(x_seq)
        spk_seq = self.neuron(cur_seq)
        count = spk_seq.sum(dim=0)
        return spk_seq, count

    def train_ff(self, x_pos_seq: torch.Tensor, x_neg_seq: torch.Tensor):
        self.train()
        _, h_pos = self.run(x_pos_seq)
        _, h_neg = self.run(x_neg_seq)
        loss = swish_ff_loss(h_pos, h_neg, alpha=args.alpha)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            spk_pos_seq, _ = self.run(x_pos_seq)
            spk_neg_seq, _ = self.run(x_neg_seq)
        return spk_pos_seq.detach(), spk_neg_seq.detach(), float(loss.item())

class FFSpikingNet(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList([FFSpikingLayer(dims[i], dims[i+1], lr=args.lr, weight_decay=args.weight_decay) for i in range(len(dims)-1)])

    def train_ff(self, x_pos_seq: torch.Tensor, x_neg_seq: torch.Tensor):
        h_pos, h_neg = x_pos_seq, x_neg_seq
        losses = []
        for layer in self.layers:
            h_pos, h_neg, lv = layer.train_ff(h_pos, h_neg)
            losses.append(lv)
        return losses

    def set_learning_rate(self, lr: float):
        for layer in self.layers:
            for g in layer.opt.param_groups:
                g["lr"] = float(lr)

    @torch.no_grad()
    def layer_goodnesses(self, x_btf: torch.Tensor, y_overlay: Union[torch.Tensor, int], activity_summary=None):
        x = append_label_to_x_temporal(x_btf, y_overlay, label_scale=args.label_scale)
        h_seq = to_tbf(x) * float(args.input_gain)
        gs = []
        for li, layer in enumerate(self.layers):
            spk_seq, h_count = layer.run(h_seq)
            if activity_summary is not None:
                record_layer_activity(activity_summary, li, h_seq, spk_seq, layer.fc.out_features)
            gs.append(goodness(h_count))
            h_seq = spk_seq.detach()
        return gs

    @torch.no_grad()
    def goodness_per_class(self, x_btf: torch.Tensor, collect_activity: bool=False):
        act = init_activity_summary(len(self.layers)) if collect_activity else None
        cols = []
        for label in range(NUM_CLASSES):
            cols.append(sum(self.layer_goodnesses(x_btf, label, act)).unsqueeze(1))
        g = torch.cat(cols, dim=1)
        if collect_activity:
            return g, act
        return g

    @torch.no_grad()
    def predict(self, x_btf: torch.Tensor, collect_activity: bool=False):
        if collect_activity:
            g, act = self.goodness_per_class(x_btf, collect_activity=True)
            return g.argmax(1), act
        return self.goodness_per_class(x_btf).argmax(1)

@torch.no_grad()
def make_examples(model: FFSpikingNet, x_btf: torch.Tensor, y_true: torch.Tensor, epsilon: float=1e-12):
    g = model.goodness_per_class(x_btf)
    g[torch.arange(x_btf.size(0), device=x_btf.device), y_true] = 0.0
    y_hard = torch.multinomial(torch.sqrt(g + epsilon), 1).squeeze(1)
    x_pos = append_label_to_x_temporal(x_btf, y_true, label_scale=args.label_scale)
    x_neg = append_label_to_x_temporal(x_btf, y_hard, label_scale=args.label_scale)
    return to_tbf(x_pos) * float(args.input_gain), to_tbf(x_neg) * float(args.input_gain)

# data
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
eval_loader = DataLoader(eval_dataset, batch_size=max(256, args.b), shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=max(256, args.b), shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
print(f"SHD train={len(train_dataset)} eval={len(eval_dataset)} heldout_test={len(test_dataset)}")

dims = [INPUT_SIZE] + [args.hidden_dim] * args.num_layers
print(f"Network dims: {dims} | raw SHD input={SHD_INPUT_SIZE}, appended label channels={NUM_CLASSES}")
net = FFSpikingNet(dims).to(device)
print(net)

run_id = time.strftime("%Y%m%d_%H%M%S")
artifact_prefix = os.path.join(args.out_dir, f"SHD_{args.model}_FF_v2_append_label_{run_id}")
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

csv_columns = ["run_id","dataset","method","epoch","model","hidden_dim","num_layers","T","split_by","val_ratio","tau","v_threshold","v_reset","tau_response","tau_refractory","input_gain","lr","weight_decay","alpha","label_scale","train_acc","train_macro_precision","train_macro_recall","train_macro_f1","test_acc","test_macro_precision","test_macro_recall","test_macro_f1","best_test_acc","best_test_macro_f1","train_time_sec","test_time_sec","epoch_time_sec","train_speed","test_speed","train_latency_ms_per_sample","test_latency_ms_per_sample","train_cpu_memory_mb","test_cpu_memory_mb","train_gpu_memory_allocated_mb","train_gpu_memory_reserved_mb","test_gpu_memory_allocated_mb","test_gpu_memory_reserved_mb","test_confusion_path","best_test_confusion_path","final_test_confusion_path","best_model_path","final_model_path"] + [f"layer_{i}_loss" for i in range(len(net.layers))]
for prefix in ("train","test"):
    csv_columns.extend([f"{prefix}_total_spikes",f"{prefix}_global_spike_rate",f"{prefix}_dense_synops",f"{prefix}_event_synops",f"{prefix}_energy_proxy_synops",f"{prefix}_event_to_dense_ratio"])
    for i in range(len(net.layers)):
        csv_columns.extend([f"{prefix}_layer_{i}_spike_count",f"{prefix}_layer_{i}_spike_rate",f"{prefix}_layer_{i}_active_input_rate",f"{prefix}_layer_{i}_dense_synops",f"{prefix}_layer_{i}_event_synops",f"{prefix}_layer_{i}_event_to_dense_ratio"])
pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)

@torch.no_grad()
def evaluate(loader: DataLoader, max_samples: int=0, collect_activity: bool=True):
    net.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    correct = 0
    total = 0
    activity_summary = init_activity_summary(len(net.layers)) if collect_activity else None
    max_cpu = get_process_memory_bytes()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    start = time.time()
    for x, y in loader:
        if max_samples > 0 and total >= max_samples:
            break
        x = x.float()
        if max_samples > 0 and total + y.numel() > max_samples:
            remain = max_samples - total
            x = x[:remain]
            y = y[:remain]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if collect_activity:
            pred, act = net.predict(x, collect_activity=True)
            merge_activity_summaries(activity_summary, act)
        else:
            pred = net.predict(x)
        correct += int((pred == y).sum().item())
        total += y.numel()
        update_confusion_matrix(confusion, pred.detach().cpu(), y.detach().cpu())
        max_cpu = max(max_cpu, get_process_memory_bytes())
    synchronize_if_needed(device)
    elapsed = time.time() - start
    gpu_a = gpu_r = 0.0
    if device.type == "cuda":
        gpu_a = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        gpu_r = bytes_to_mb(torch.cuda.max_memory_reserved(device))
    cls = macro_classification_metrics(confusion)
    return {"acc": correct/max(total,1), "macro_precision": cls["macro_precision"], "macro_recall": cls["macro_recall"], "macro_f1": cls["macro_f1"],
            "samples": total, "time_sec": elapsed, "samples_per_sec": total/max(elapsed,1e-9), "latency_ms_per_sample": 1000.0*elapsed/max(total,1),
            "cpu_memory_mb": bytes_to_mb(max_cpu), "gpu_memory_allocated_mb": gpu_a, "gpu_memory_reserved_mb": gpu_r,
            "confusion_matrix": confusion, "activity_summary": activity_summary}

best_test_acc = 0.0
best_test_macro_f1 = 0.0
last_eval_metrics = None
best_heldout = None

for epoch in range(args.epochs):
    epoch_start = time.time()
    lr_now = learning_rate_for_epoch(epoch, args.lr)
    net.set_learning_rate(lr_now)
    layer_losses = [[] for _ in range(len(net.layers))]
    train_samples = 0
    max_train_cpu = get_process_memory_bytes()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    train_start = time.time()
    net.train()
    for x, y in train_loader:
        x = x.float().to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        train_samples += y.numel()
        x_pos_seq, x_neg_seq = make_examples(net, x, y)
        losses = net.train_ff(x_pos_seq, x_neg_seq)
        for i, lv in enumerate(losses):
            layer_losses[i].append(lv)
        max_train_cpu = max(max_train_cpu, get_process_memory_bytes())
    synchronize_if_needed(device)
    train_time = time.time() - train_start
    train_gpu_a = train_gpu_r = 0.0
    if device.type == "cuda":
        train_gpu_a = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        train_gpu_r = bytes_to_mb(torch.cuda.max_memory_reserved(device))
    train_metrics = evaluate(train_loader, max_samples=args.eval_subset if args.eval_subset > 0 else 0, collect_activity=True)
    eval_metrics = evaluate(eval_loader, collect_activity=True)
    last_eval_metrics = eval_metrics
    heldout_metrics = evaluate(test_loader, collect_activity=False)
    best_heldout = heldout_metrics
    test_confusion_epoch_path = ""
    if args.save_test_confusion_every_epoch:
        test_confusion_epoch_path = os.path.join(test_confusion_dir, f"epoch_{epoch+1:03d}.csv")
        save_confusion_matrix(test_confusion_epoch_path, eval_metrics["confusion_matrix"])
    if eval_metrics["acc"] >= best_test_acc:
        best_test_acc = eval_metrics["acc"]
        save_confusion_matrix(best_confusion_path, eval_metrics["confusion_matrix"])
        save_model_artifact(best_model_path, net, {"epoch": epoch+1, "best_eval_acc": best_test_acc, "best_eval_macro_f1": max(best_test_macro_f1, eval_metrics["macro_f1"]), "heldout_test_acc_at_selection": heldout_metrics["acc"], "heldout_test_macro_f1_at_selection": heldout_metrics["macro_f1"]})
    best_test_macro_f1 = max(best_test_macro_f1, eval_metrics["macro_f1"])
    avg_losses = [float(np.mean(v)) if v else 0.0 for v in layer_losses]
    row = {"run_id": run_id, "dataset":"shd","method":"ff","epoch":epoch+1,"model":args.model,"hidden_dim":args.hidden_dim,"num_layers":args.num_layers,"T":args.T,"split_by":args.split_by,"val_ratio":args.val_ratio,
           "tau":args.tau,"v_threshold":args.v_threshold,"v_reset":args.v_reset,"tau_response":args.tau_response,"tau_refractory":args.tau_refractory,"input_gain":args.input_gain,
           "lr":lr_now,"weight_decay":args.weight_decay,"alpha":args.alpha,"label_scale":args.label_scale,
           "train_acc":train_metrics["acc"],"train_macro_precision":train_metrics["macro_precision"],"train_macro_recall":train_metrics["macro_recall"],"train_macro_f1":train_metrics["macro_f1"],
           "test_acc":eval_metrics["acc"],"test_macro_precision":eval_metrics["macro_precision"],"test_macro_recall":eval_metrics["macro_recall"],"test_macro_f1":eval_metrics["macro_f1"],
           "best_test_acc":best_test_acc,"best_test_macro_f1":best_test_macro_f1,
           "train_time_sec":train_time,"test_time_sec":eval_metrics["time_sec"],"epoch_time_sec":time.time()-epoch_start,
           "train_speed":train_samples/max(train_time,1e-9),"test_speed":eval_metrics["samples_per_sec"],
           "train_latency_ms_per_sample":1000.0*train_time/max(train_samples,1),"test_latency_ms_per_sample":eval_metrics["latency_ms_per_sample"],
           "train_cpu_memory_mb":bytes_to_mb(max_train_cpu),"test_cpu_memory_mb":eval_metrics["cpu_memory_mb"],
           "train_gpu_memory_allocated_mb":train_gpu_a,"train_gpu_memory_reserved_mb":train_gpu_r,
           "test_gpu_memory_allocated_mb":eval_metrics["gpu_memory_allocated_mb"],"test_gpu_memory_reserved_mb":eval_metrics["gpu_memory_reserved_mb"],
           "test_confusion_path":test_confusion_epoch_path,"best_test_confusion_path":best_confusion_path,"final_test_confusion_path":final_confusion_path,
           "best_model_path":best_model_path,"final_model_path":final_model_path}
    for i,v in enumerate(avg_losses):
        row[f"layer_{i}_loss"] = v
    row.update(activity_summary_to_metrics("train", train_metrics["activity_summary"]))
    row.update(activity_summary_to_metrics("test", eval_metrics["activity_summary"]))
    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)
    print(f"Epoch {epoch+1}/{args.epochs} | {args.model} | Train Acc/F1 {train_metrics['acc']*100:.2f}/{train_metrics['macro_f1']*100:.2f} | Eval Acc/F1 {eval_metrics['acc']*100:.2f}/{eval_metrics['macro_f1']*100:.2f} | Heldout {heldout_metrics['acc']*100:.2f} | Best Eval {best_test_acc*100:.2f}")

if last_eval_metrics is not None:
    save_confusion_matrix(final_confusion_path, last_eval_metrics["confusion_matrix"])
    save_model_artifact(final_model_path, net, {"epoch":args.epochs,"final_eval_acc":last_eval_metrics["acc"],"final_eval_macro_f1":last_eval_metrics["macro_f1"],"final_heldout_test_acc": best_heldout["acc"] if best_heldout else None, "final_heldout_test_macro_f1": best_heldout["macro_f1"] if best_heldout else None, "best_eval_acc": best_test_acc, "best_eval_macro_f1": best_test_macro_f1})
    save_json(summary_path, {"run_id":run_id,"dataset":"shd","method":"ff","csv_path":csv_path,"args_path":args_path,"best_model_path":best_model_path,"final_model_path":final_model_path,"best_eval_acc":best_test_acc,"best_eval_macro_f1":best_test_macro_f1,
                             "final_heldout_test_acc": best_heldout["acc"] if best_heldout else None, "final_heldout_test_macro_f1": best_heldout["macro_f1"] if best_heldout else None})
print("Done. CSV saved to:", csv_path)
