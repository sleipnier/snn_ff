import json
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, neuron, surrogate

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None


# -------------------------------------------------
# Args
# -------------------------------------------------
parser = argparse.ArgumentParser(description="MNIST local-greedy SNN (negative-free, FF-inspired)")

parser.add_argument("-device", default="cuda:0")
parser.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/MNIST")
parser.add_argument("-out-dir", type=str, default="./result/new/")
parser.add_argument("-b", type=int, default=4096)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-j", type=int, default=4)
parser.add_argument("-T", type=int, default=10)
parser.add_argument("--seed", type=int, default=2026)

# model
parser.add_argument("--model", type=str, default="lif", choices=["lif", "alif", "srm", "dynsrm"])
parser.add_argument("--hidden-dim", type=int, default=500)
parser.add_argument("--num-layers", type=int, default=2, choices=[1, 2, 3])
parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v-threshold", type=float, default=0.5)
parser.add_argument("--v-reset", type=float, default=0.0)
parser.add_argument("--tau-response", type=float, default=2.0)
parser.add_argument("--tau-refractory", type=float, default=10.0)
parser.add_argument("--input-gain", type=float, default=1.0)

# local greedy learning
parser.add_argument("--lr", type=float, default=8e-4)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--loss", type=str, default="ce", choices=["ce", "mse"])
parser.add_argument("--target-rate", type=float, default=0.15,
                    help="desired average spike rate per hidden neuron over T steps")
parser.add_argument("--rate-lambda", type=float, default=0.02,
                    help="penalty weight for hidden-layer rate regularization")
parser.add_argument("--confidence-lambda", type=float, default=0.0,
                    help="optional margin-style bonus on local heads")
parser.add_argument("--eval-subset", type=int, default=0)
parser.add_argument("--save-test-confusion-every-epoch", action="store_true")

args, _ = parser.parse_known_args()
print(args)

os.makedirs(args.out_dir, exist_ok=True)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# -------------------------------------------------
# Data
# -------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
])

train_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=False, transform=transform, download=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.b,
    shuffle=True,
    drop_last=True,
    num_workers=args.j,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=max(1024, args.b),
    shuffle=False,
    drop_last=False,
    num_workers=args.j,
    pin_memory=True,
)

print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

NUM_CLASSES = 10


def synchronize_if_needed(device_: torch.device):
    if device_.type == "cuda":
        torch.cuda.synchronize(device_)


def get_process_memory_bytes():
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss) * 1024
    return 0


def bytes_to_mb(value: int):
    return float(value) / (1024.0 * 1024.0)


def update_confusion_matrix(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
    indices = target * NUM_CLASSES + pred
    counts = torch.bincount(indices, minlength=NUM_CLASSES * NUM_CLASSES)
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)


def macro_classification_metrics(confusion: torch.Tensor):
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    predicted = confusion.sum(dim=0)
    actual = confusion.sum(dim=1)
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
    return {
        "layers": [
            {
                "spike_count": 0.0,
                "num_elements": 0.0,
                "input_nonzero_count": 0.0,
                "input_num_elements": 0.0,
                "dense_synops": 0.0,
                "event_synops": 0.0,
            }
            for _ in range(num_layers)
        ],
        "total_spike_count": 0.0,
        "total_num_elements": 0.0,
        "total_dense_synops": 0.0,
        "total_event_synops": 0.0,
    }


def record_layer_activity(summary: Dict[str, object], layer_idx: int, x_seq: torch.Tensor,
                          spk_seq: torch.Tensor, out_features: int):
    layer_summary = summary["layers"][layer_idx]
    input_nonzero_count = float(x_seq.ne(0).sum().item())
    input_num_elements = float(x_seq.numel())
    spike_count = float(spk_seq.sum().item())
    num_elements = float(spk_seq.numel())
    dense_synops = float(x_seq.shape[0] * x_seq.shape[1] * x_seq.shape[2] * out_features)
    event_synops = input_nonzero_count * float(out_features)

    layer_summary["spike_count"] += spike_count
    layer_summary["num_elements"] += num_elements
    layer_summary["input_nonzero_count"] += input_nonzero_count
    layer_summary["input_num_elements"] += input_num_elements
    layer_summary["dense_synops"] += dense_synops
    layer_summary["event_synops"] += event_synops

    summary["total_spike_count"] += spike_count
    summary["total_num_elements"] += num_elements
    summary["total_dense_synops"] += dense_synops
    summary["total_event_synops"] += event_synops


def merge_activity_summaries(dst: Dict[str, object], src: Dict[str, object]):
    for dst_layer, src_layer in zip(dst["layers"], src["layers"]):
        for key in ("spike_count", "num_elements", "input_nonzero_count", "input_num_elements", "dense_synops", "event_synops"):
            dst_layer[key] += src_layer[key]
    for key in ("total_spike_count", "total_num_elements", "total_dense_synops", "total_event_synops"):
        dst[key] += src[key]


def activity_summary_to_metrics(prefix: str, summary: Optional[Dict[str, object]]):
    metrics = {}
    if summary is None:
        return metrics

    total_spike_count = float(summary["total_spike_count"])
    total_num_elements = float(summary["total_num_elements"])
    total_dense_synops = float(summary["total_dense_synops"])
    total_event_synops = float(summary["total_event_synops"])

    metrics[f"{prefix}_total_spikes"] = total_spike_count
    metrics[f"{prefix}_global_spike_rate"] = total_spike_count / max(total_num_elements, 1.0)
    metrics[f"{prefix}_dense_synops"] = total_dense_synops
    metrics[f"{prefix}_event_synops"] = total_event_synops
    metrics[f"{prefix}_energy_proxy_synops"] = total_event_synops
    metrics[f"{prefix}_event_to_dense_ratio"] = total_event_synops / max(total_dense_synops, 1.0)

    for layer_idx, layer_summary in enumerate(summary["layers"]):
        spike_count = float(layer_summary["spike_count"])
        num_elements = float(layer_summary["num_elements"])
        input_nonzero_count = float(layer_summary["input_nonzero_count"])
        input_num_elements = float(layer_summary["input_num_elements"])
        dense_synops = float(layer_summary["dense_synops"])
        event_synops = float(layer_summary["event_synops"])

        metrics[f"{prefix}_layer_{layer_idx}_spike_count"] = spike_count
        metrics[f"{prefix}_layer_{layer_idx}_spike_rate"] = spike_count / max(num_elements, 1.0)
        metrics[f"{prefix}_layer_{layer_idx}_active_input_rate"] = input_nonzero_count / max(input_num_elements, 1.0)
        metrics[f"{prefix}_layer_{layer_idx}_dense_synops"] = dense_synops
        metrics[f"{prefix}_layer_{layer_idx}_event_synops"] = event_synops
        metrics[f"{prefix}_layer_{layer_idx}_event_to_dense_ratio"] = event_synops / max(dense_synops, 1.0)
    return metrics


def save_confusion_matrix(path: str, confusion: torch.Tensor):
    df = pd.DataFrame(
        confusion.tolist(),
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    )
    df.to_csv(path, index=True)


def save_json(path: str, payload: Dict[str, object]):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_model_artifact(path: str, net_: nn.Module, extra_metadata: Optional[Dict[str, object]] = None):
    payload = {
        "model_state_dict": net_.state_dict(),
        "args": vars(args),
        "dims": dims,
        "model_class": net_.__class__.__name__,
    }
    if extra_metadata is not None:
        payload["metadata"] = extra_metadata
    torch.save(payload, path)


@torch.no_grad()
def repeat_static_input(x: torch.Tensor, T: int, gain: float = 1.0):
    x = x * float(gain)
    return x.unsqueeze(0).repeat(T, 1, 1)


def make_neuron_factory():
    if args.model == "lif":
        node_cls = neuron.LIFNode
        node_kwargs = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            tau=args.tau,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    elif args.model == "alif":
        node_cls = getattr(neuron, "ParametricLIFNode", neuron.LIFNode)
        node_kwargs = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            init_tau=args.tau,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    elif args.model == "srm":
        node_cls = neuron.SRMNode
        node_kwargs = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            tau_response=args.tau_response,
            tau_refractory=args.tau_refractory,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    else:
        node_cls = getattr(neuron, "DynamicSRMNode", neuron.SRMNode)
        node_kwargs = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            init_tau_response=args.tau_response,
            init_tau_refractory=args.tau_refractory,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    return node_cls, node_kwargs


def classification_loss(logits: torch.Tensor, y: torch.Tensor):
    if args.loss == "ce":
        return F.cross_entropy(logits, y)
    return F.mse_loss(logits, F.one_hot(y, NUM_CLASSES).float())


def confidence_margin(logits: torch.Tensor, y: torch.Tensor):
    pos = logits[torch.arange(logits.size(0), device=logits.device), y]
    tmp = logits.clone()
    tmp[torch.arange(tmp.size(0), device=tmp.device), y] = -1e9
    neg = tmp.max(dim=1).values
    return pos - neg


class LocalGreedyLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, lr: float, weight_decay: float, is_final: bool):
        super().__init__()
        node_cls, node_kwargs = make_neuron_factory()
        self.fc = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.neuron = node_cls(**node_kwargs)
        if hasattr(self.neuron, "step_mode"):
            self.neuron.step_mode = "m"

        self.is_final = bool(is_final)
        self.aux_head = None if self.is_final else nn.Linear(out_features, NUM_CLASSES)
        if self.aux_head is not None:
            nn.init.xavier_uniform_(self.aux_head.weight)
            nn.init.zeros_(self.aux_head.bias)

        params = list(self.fc.parameters()) + list(self.neuron.parameters())
        if self.aux_head is not None:
            params += list(self.aux_head.parameters())
        self.opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def run(self, x_seq: torch.Tensor):
        functional.reset_net(self)
        cur_seq = self.fc(x_seq)
        spk_seq = self.neuron(cur_seq)
        count = spk_seq.sum(dim=0)
        return spk_seq, count

    def local_logits(self, count: torch.Tensor):
        if self.aux_head is None:
            return count
        return self.aux_head(count)

    def local_goodness(self, count: torch.Tensor, y: torch.Tensor):
        logits = self.local_logits(count)
        return confidence_margin(logits, y)

    def local_loss(self, count: torch.Tensor, y: torch.Tensor):
        logits = self.local_logits(count)
        cls = classification_loss(logits, y)
        rate = count.mean() / max(args.T, 1)
        rate_penalty = (rate - args.target_rate) ** 2 if not self.is_final else torch.zeros_like(cls)
        margin_bonus = -self.local_goodness(count, y).mean() if args.confidence_lambda > 0 else torch.zeros_like(cls)
        loss = cls + args.rate_lambda * rate_penalty + args.confidence_lambda * margin_bonus
        return loss, logits, cls.detach(), rate.detach(), (self.local_goodness(count, y).mean().detach())

    def train_local(self, x_seq: torch.Tensor, y: torch.Tensor):
        self.train()
        spk_seq, count = self.run(x_seq)
        loss, logits, cls_det, rate_det, goodness_det = self.local_loss(count, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            spk_seq, count = self.run(x_seq)
        return spk_seq.detach(), float(loss.item()), float(cls_det.item()), float(rate_det.item()), float(goodness_det.item())


class LocalGreedySNN(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalGreedyLayer(dims[i], dims[i + 1], lr=args.lr, weight_decay=args.weight_decay,
                             is_final=(i == len(dims) - 2))
            for i in range(len(dims) - 1)
        ])

    def set_learning_rate(self, lr: float):
        for layer in self.layers:
            for group in layer.opt.param_groups:
                group["lr"] = float(lr)

    def train_local_greedy(self, x_seq: torch.Tensor, y: torch.Tensor):
        h = x_seq
        losses, cls_losses, rates, goodnesses = [], [], [], []
        for layer in self.layers:
            h, lv, cv, rv, gv = layer.train_local(h, y)
            losses.append(lv)
            cls_losses.append(cv)
            rates.append(rv)
            goodnesses.append(gv)
        return losses, cls_losses, rates, goodnesses

    @torch.no_grad()
    def predict(self, x_flat: torch.Tensor, collect_activity: bool = False):
        h_seq = repeat_static_input(x_flat, T=args.T, gain=args.input_gain)
        activity_summary = init_activity_summary(len(self.layers)) if collect_activity else None
        final_count = None
        for layer_idx, layer in enumerate(self.layers):
            spk_seq, count = layer.run(h_seq)
            if activity_summary is not None:
                record_layer_activity(activity_summary, layer_idx, h_seq, spk_seq, layer.fc.out_features)
            h_seq = spk_seq.detach()
            final_count = count
        pred = final_count.argmax(dim=1)
        if collect_activity:
            return pred, activity_summary
        return pred


# -------------------------------------------------
# Build net
# -------------------------------------------------
dims = [784] + [args.hidden_dim] * args.num_layers + [NUM_CLASSES]
net = LocalGreedySNN(dims).to(device)
print(net)

run_id = time.strftime("%Y%m%d_%H%M%S")
artifact_prefix = os.path.join(args.out_dir, f"MNIST_{args.model}_LocalGreedy_v2_{run_id}")
csv_path = f"{artifact_prefix}.csv"
args_path = f"{artifact_prefix}_args.json"
summary_path = f"{artifact_prefix}_summary.json"
best_confusion_path = f"{artifact_prefix}_best_test_confusion.csv"
final_confusion_path = f"{artifact_prefix}_final_test_confusion.csv"
best_model_path = f"{artifact_prefix}_best_model.pth"
final_model_path = f"{artifact_prefix}_final_model.pth"
test_confusion_dir = f"{artifact_prefix}_test_confusions"

if args.save_test_confusion_every_epoch:
    os.makedirs(test_confusion_dir, exist_ok=True)

save_json(args_path, vars(args))

csv_columns = [
    "run_id", "epoch", "model", "hidden_dim", "num_layers", "T", "tau", "v_threshold", "v_reset",
    "tau_response", "tau_refractory", "input_gain", "lr", "weight_decay", "loss_name", "target_rate",
    "rate_lambda", "confidence_lambda", "eval_subset", "train_acc", "train_macro_precision",
    "train_macro_recall", "train_macro_f1", "test_acc", "test_macro_precision", "test_macro_recall",
    "test_macro_f1", "best_test_acc", "best_test_macro_f1", "train_time_sec", "test_time_sec",
    "epoch_time_sec", "train_speed", "test_speed", "train_latency_ms_per_sample", "test_latency_ms_per_sample",
    "train_cpu_memory_mb", "test_cpu_memory_mb", "train_gpu_memory_allocated_mb", "train_gpu_memory_reserved_mb",
    "test_gpu_memory_allocated_mb", "test_gpu_memory_reserved_mb", "test_confusion_path", "best_test_confusion_path",
    "final_test_confusion_path", "best_model_path", "final_model_path",
]
for i in range(len(net.layers)):
    csv_columns.extend([
        f"layer_{i}_loss",
        f"layer_{i}_cls_loss",
        f"layer_{i}_mean_rate",
        f"layer_{i}_goodness",
    ])

for prefix in ("train", "test"):
    csv_columns.extend([
        f"{prefix}_total_spikes",
        f"{prefix}_global_spike_rate",
        f"{prefix}_dense_synops",
        f"{prefix}_event_synops",
        f"{prefix}_energy_proxy_synops",
        f"{prefix}_event_to_dense_ratio",
    ])
    for i in range(len(net.layers)):
        csv_columns.extend([
            f"{prefix}_layer_{i}_spike_count",
            f"{prefix}_layer_{i}_spike_rate",
            f"{prefix}_layer_{i}_active_input_rate",
            f"{prefix}_layer_{i}_dense_synops",
            f"{prefix}_layer_{i}_event_synops",
            f"{prefix}_layer_{i}_event_to_dense_ratio",
        ])

pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)


@torch.no_grad()
def evaluate(loader: DataLoader, *, max_samples: int = 0, collect_activity: bool = True):
    net.eval()
    correct = 0
    total = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    activity_summary = init_activity_summary(len(net.layers)) if collect_activity else None
    max_cpu_memory_bytes = get_process_memory_bytes()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize_if_needed(device)
    start_time = time.time()

    for x, y in loader:
        if max_samples > 0 and total >= max_samples:
            break

        remaining = max_samples - total if max_samples > 0 else y.numel()
        if remaining <= 0:
            break
        if remaining < y.numel():
            x = x[:remaining]
            y = y[:remaining]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if collect_activity:
            pred, batch_activity = net.predict(x, collect_activity=True)
            merge_activity_summaries(activity_summary, batch_activity)
        else:
            pred = net.predict(x)

        correct += (pred == y).sum().item()
        total += y.numel()
        update_confusion_matrix(confusion, pred.detach().cpu(), y.detach().cpu())
        max_cpu_memory_bytes = max(max_cpu_memory_bytes, get_process_memory_bytes())

    synchronize_if_needed(device)
    elapsed = time.time() - start_time

    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0
    if device.type == "cuda":
        gpu_allocated_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        gpu_reserved_mb = bytes_to_mb(torch.cuda.max_memory_reserved(device))

    cls_metrics = macro_classification_metrics(confusion)
    metrics = {
        "acc": correct / max(total, 1),
        "macro_precision": cls_metrics["macro_precision"],
        "macro_recall": cls_metrics["macro_recall"],
        "macro_f1": cls_metrics["macro_f1"],
        "samples": total,
        "time_sec": elapsed,
        "samples_per_sec": total / max(elapsed, 1e-9),
        "latency_ms_per_sample": 1000.0 * elapsed / max(total, 1),
        "cpu_memory_mb": bytes_to_mb(max_cpu_memory_bytes),
        "gpu_memory_allocated_mb": gpu_allocated_mb,
        "gpu_memory_reserved_mb": gpu_reserved_mb,
        "confusion_matrix": confusion,
        "activity_summary": activity_summary,
    }
    return metrics


def learning_rate_for_epoch(epoch: int, base_lr: float) -> float:
    if epoch >= 100:
        return 1e-6
    if epoch >= 75:
        return 1e-5
    if epoch >= 50:
        return 3e-4
    return float(base_lr)


best_test_acc = 0.0
best_test_macro_f1 = 0.0
last_test_metrics = None

for epoch in range(args.epochs):
    epoch_start = time.time()
    current_lr = learning_rate_for_epoch(epoch, args.lr)
    net.set_learning_rate(current_lr)
    net.train()
    layer_losses_accum = [[] for _ in range(len(net.layers))]
    layer_cls_losses_accum = [[] for _ in range(len(net.layers))]
    layer_rates_accum = [[] for _ in range(len(net.layers))]
    layer_goodness_accum = [[] for _ in range(len(net.layers))]
    train_samples = 0
    max_train_cpu_memory_bytes = get_process_memory_bytes()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize_if_needed(device)
    train_start = time.time()

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        train_samples += y.numel()

        x_seq = repeat_static_input(x, T=args.T, gain=args.input_gain)
        losses, cls_losses, rates, goodnesses = net.train_local_greedy(x_seq, y)

        for i in range(len(net.layers)):
            layer_losses_accum[i].append(losses[i])
            layer_cls_losses_accum[i].append(cls_losses[i])
            layer_rates_accum[i].append(rates[i])
            layer_goodness_accum[i].append(goodnesses[i])

        max_train_cpu_memory_bytes = max(max_train_cpu_memory_bytes, get_process_memory_bytes())

    synchronize_if_needed(device)
    train_time = time.time() - train_start
    train_speed = train_samples / max(train_time, 1e-6)
    train_latency_ms_per_sample = 1000.0 * train_time / max(train_samples, 1)
    train_gpu_allocated_mb = 0.0
    train_gpu_reserved_mb = 0.0
    if device.type == "cuda":
        train_gpu_allocated_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        train_gpu_reserved_mb = bytes_to_mb(torch.cuda.max_memory_reserved(device))

    eval_train_loader = DataLoader(
        train_dataset,
        batch_size=max(1024, args.b),
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True,
    )

    train_metrics = evaluate(eval_train_loader, max_samples=args.eval_subset, collect_activity=True)

    test_start = time.time()
    test_metrics = evaluate(test_loader, collect_activity=True)
    synchronize_if_needed(device)
    test_time = time.time() - test_start
    test_speed = test_metrics["samples"] / max(test_time, 1e-6)
    test_latency_ms_per_sample = 1000.0 * test_time / max(test_metrics["samples"], 1)
    last_test_metrics = test_metrics

    test_confusion_epoch_path = ""
    if args.save_test_confusion_every_epoch:
        test_confusion_epoch_path = os.path.join(test_confusion_dir, f"epoch_{epoch + 1:03d}.csv")
        save_confusion_matrix(test_confusion_epoch_path, test_metrics["confusion_matrix"])

    current_best_test_macro_f1 = max(best_test_macro_f1, test_metrics["macro_f1"])
    if test_metrics["acc"] >= best_test_acc:
        best_test_acc = test_metrics["acc"]
        save_confusion_matrix(best_confusion_path, test_metrics["confusion_matrix"])
        save_model_artifact(best_model_path, net, {
            "run_id": run_id,
            "epoch": epoch + 1,
            "selection_metric": "best_test_acc",
            "test_acc": test_metrics["acc"],
            "test_macro_f1": test_metrics["macro_f1"],
            "best_test_acc": best_test_acc,
            "best_test_macro_f1": current_best_test_macro_f1,
        })
    best_test_macro_f1 = current_best_test_macro_f1

    avg_layer_losses = [float(np.mean(v)) if len(v) else 0.0 for v in layer_losses_accum]
    avg_layer_cls_losses = [float(np.mean(v)) if len(v) else 0.0 for v in layer_cls_losses_accum]
    avg_layer_rates = [float(np.mean(v)) if len(v) else 0.0 for v in layer_rates_accum]
    avg_layer_goodness = [float(np.mean(v)) if len(v) else 0.0 for v in layer_goodness_accum]

    row = {
        "run_id": run_id,
        "epoch": epoch + 1,
        "model": args.model,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "T": args.T,
        "tau": args.tau,
        "v_threshold": args.v_threshold,
        "v_reset": args.v_reset,
        "tau_response": args.tau_response,
        "tau_refractory": args.tau_refractory,
        "input_gain": args.input_gain,
        "lr": current_lr,
        "weight_decay": args.weight_decay,
        "loss_name": args.loss,
        "target_rate": args.target_rate,
        "rate_lambda": args.rate_lambda,
        "confidence_lambda": args.confidence_lambda,
        "eval_subset": args.eval_subset,
        "train_acc": train_metrics["acc"],
        "train_macro_precision": train_metrics["macro_precision"],
        "train_macro_recall": train_metrics["macro_recall"],
        "train_macro_f1": train_metrics["macro_f1"],
        "test_acc": test_metrics["acc"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_macro_f1,
        "train_time_sec": train_time,
        "test_time_sec": test_time,
        "epoch_time_sec": time.time() - epoch_start,
        "train_speed": train_speed,
        "test_speed": test_speed,
        "train_latency_ms_per_sample": train_latency_ms_per_sample,
        "test_latency_ms_per_sample": test_latency_ms_per_sample,
        "train_cpu_memory_mb": bytes_to_mb(max_train_cpu_memory_bytes),
        "test_cpu_memory_mb": test_metrics["cpu_memory_mb"],
        "train_gpu_memory_allocated_mb": train_gpu_allocated_mb,
        "train_gpu_memory_reserved_mb": train_gpu_reserved_mb,
        "test_gpu_memory_allocated_mb": test_metrics["gpu_memory_allocated_mb"],
        "test_gpu_memory_reserved_mb": test_metrics["gpu_memory_reserved_mb"],
        "test_confusion_path": test_confusion_epoch_path,
        "best_test_confusion_path": best_confusion_path,
        "final_test_confusion_path": final_confusion_path,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }
    for i in range(len(net.layers)):
        row[f"layer_{i}_loss"] = avg_layer_losses[i]
        row[f"layer_{i}_cls_loss"] = avg_layer_cls_losses[i]
        row[f"layer_{i}_mean_rate"] = avg_layer_rates[i]
        row[f"layer_{i}_goodness"] = avg_layer_goodness[i]
    row.update(activity_summary_to_metrics("train", train_metrics["activity_summary"]))
    row.update(activity_summary_to_metrics("test", test_metrics["activity_summary"]))

    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)

    print("\n" + "=" * 90)
    print(
        f"Epoch {epoch + 1}/{args.epochs} | model={args.model} | T={args.T} | "
        f"hidden={args.hidden_dim} | depth={args.num_layers} | lr={current_lr:g}"
    )
    print("-" * 90)
    print(
        f"Train Acc/F1: {train_metrics['acc'] * 100:.2f}% / {train_metrics['macro_f1'] * 100:.2f}% | "
        f"Test Acc/F1: {test_metrics['acc'] * 100:.2f}% / {test_metrics['macro_f1'] * 100:.2f}%"
    )
    print(
        f"Best Test Acc/F1: {best_test_acc * 100:.2f}% / {best_test_macro_f1 * 100:.2f}% | "
        f"Train Speed: {train_speed:.2f} samples/s | Test Speed: {test_speed:.2f} samples/s"
    )
    print(
        f"Layer cls losses: " + "  ".join([f"L{i}={v:.5f}" for i, v in enumerate(avg_layer_cls_losses)])
    )
    print(
        f"Layer mean rates: " + "  ".join([f"L{i}={v:.4f}" for i, v in enumerate(avg_layer_rates)])
    )
    print(
        f"Layer goodness: " + "  ".join([f"L{i}={v:.4f}" for i, v in enumerate(avg_layer_goodness)])
    )
    print("=" * 90)

if last_test_metrics is not None:
    save_confusion_matrix(final_confusion_path, last_test_metrics["confusion_matrix"])
    save_model_artifact(final_model_path, net, {
        "run_id": run_id,
        "epoch": args.epochs,
        "selection_metric": "final_epoch",
        "test_acc": last_test_metrics["acc"],
        "test_macro_f1": last_test_metrics["macro_f1"],
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_macro_f1,
    })
    save_json(summary_path, {
        "run_id": run_id,
        "csv_path": csv_path,
        "args_path": args_path,
        "best_test_confusion_path": best_confusion_path,
        "final_test_confusion_path": final_confusion_path,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_macro_f1,
    })

print("Done. CSV saved to:", csv_path)
