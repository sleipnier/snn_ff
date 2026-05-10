#!/usr/bin/env python3
# from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda import amp
from torch.utils.data import DataLoader

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import parametric_lif_net

NUM_CLASSES = 10
INPUT_SIZE = 784


parser = argparse.ArgumentParser(description="MNIST BP-SNN (SpikingJelly) v2")
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/MNIST")
parser.add_argument("-out-dir", type=str, default="./result")
parser.add_argument("-b", type=int, default=64, help="batch size")
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-j", type=int, default=4, help="dataloader workers")
parser.add_argument("-T", type=int, default=10, help="number of repeated static steps")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--model", type=str, default="lif", choices=["lif", "alif", "srm", "dynsrm"])
parser.add_argument("--hidden-dim", type=int, default=500)
parser.add_argument("--num-layers", type=int, default=2, choices=[1, 2])
parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v-threshold", type=float, default=0.5)
parser.add_argument("--v-reset", type=float, default=0.0)
parser.add_argument("--tau-response", type=float, default=2.0)
parser.add_argument("--tau-refractory", type=float, default=10.0)
parser.add_argument("--input-gain", type=float, default=1.0)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("-opt", type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("--adam-mode", type=str, default="plain", choices=["grouped", "plain"])
parser.add_argument("--loss", type=str, default="mse", choices=["mse", "ce"])
parser.add_argument("--cosine-scheduler", action="store_true")
parser.add_argument("-resume", type=str, default="")
parser.add_argument("-amp", action="store_true")
parser.add_argument("-cupy", action="store_true")
parser.add_argument("--max-train-batches", type=int, default=0)
parser.add_argument("--max-test-batches", type=int, default=0)
parser.add_argument("--save-test-confusion-every-epoch", action="store_true")
args, _ = parser.parse_known_args()
print(args)

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

NEURON_REGISTRY: Dict[str, Tuple[type, type]] = {
    "lif": (neuron.LIFNode, neuron.LIFNode),
    "alif": (getattr(neuron, "ALIFNode", getattr(neuron, "ParametricLIFNode2", neuron.ParametricLIFNode)), getattr(neuron, "ALIFNode", getattr(neuron, "ParametricLIFNode2", neuron.ParametricLIFNode))),
    "srm": (neuron.SRMNode, neuron.SRMNode),
    "dynsrm": (getattr(neuron, "DynamicSRMNode", neuron.SRMNode), getattr(neuron, "DynamicSRMNode", neuron.SRMNode)),
}


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
    pd.DataFrame(
        confusion.tolist(),
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    ).to_csv(path, index=True)


def save_json(path: str, payload: Dict[str, object]):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


@torch.no_grad()
def repeat_static_input(x: torch.Tensor, T: int, gain: float = 1.0):
    x = x * float(gain)
    return x.unsqueeze(0).repeat(T, 1, 1)


def initialize_lazy_neuron_parameters(net: torch.nn.Module, device_: torch.device):
    """Materialize lazy per-neuron tau parameters without running a forward pass."""
    prev_out_features = None

    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            prev_out_features = int(module.out_features)
            continue

        # ALIF-style lazy parameter: ParametricLIFNode2 / ALIFNode
        if hasattr(module, "w") and hasattr(module, "_init_w") and getattr(module, "w") is None:
            num_neurons = getattr(module, "num_neurons", None)
            if num_neurons is None:
                num_neurons = prev_out_features
            if num_neurons is None:
                continue
            w = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w), device=device_, dtype=torch.float32))
            module.w = w
            module.register_parameter("w", module.w)
            continue

        # DynamicSRM-style lazy parameters
        if (hasattr(module, "w_tau_response") and hasattr(module, "w_tau_refractory")
                and hasattr(module, "_init_w_response") and hasattr(module, "_init_w_refractory")
                and (getattr(module, "w_tau_response") is None or getattr(module, "w_tau_refractory") is None)):
            num_neurons = getattr(module, "num_neurons", None)
            if num_neurons is None:
                num_neurons = prev_out_features
            if num_neurons is None:
                continue
            w_resp = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w_response), device=device_, dtype=torch.float32))
            w_refr = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w_refractory), device=device_, dtype=torch.float32))
            module.w_tau_response = w_resp
            module.w_tau_refractory = w_refr
            module.register_parameter("w_tau_response", module.w_tau_response)
            module.register_parameter("w_tau_refractory", module.w_tau_refractory)


def spiking_layer_dims() -> List[int]:
    if args.num_layers == 1:
        return [NUM_CLASSES]
    return [args.hidden_dim, NUM_CLASSES]


def neuron_kwargs():
    base = dict(v_threshold=args.v_threshold, v_reset=args.v_reset, store_v_seq=False)
    if args.model == "lif":
        return {**base, "tau": args.tau}
    if args.model == "alif":
        return {**base, "init_tau": args.tau}
    if args.model == "srm":
        return {**base, "tau_response": args.tau_response, "tau_refractory": args.tau_refractory}
    if NEURON_REGISTRY["dynsrm"][0] is neuron.SRMNode:
        return {**base, "tau_response": args.tau_response, "tau_refractory": args.tau_refractory}
    return {**base, "init_tau_response": args.tau_response, "init_tau_refractory": args.tau_refractory}


def build_net() -> torch.nn.Module:
    cls, _ = NEURON_REGISTRY[args.model]
    common = dict(
        input_size=INPUT_SIZE,
        spiking_neuron=cls,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        output_spiking=True,
        **neuron_kwargs(),
    )
    net = parametric_lif_net.MNISTNet1(**common) if args.num_layers == 1 else parametric_lif_net.MNISTNet2(hidden_size=args.hidden_dim, **common)
    functional.set_step_mode(net, "m")
    if args.cupy:
        functional.set_backend(net, "cupy", instance=NEURON_REGISTRY[args.model][1])
    return net


def build_optimizer(net: torch.nn.Module):
    if args.opt == "sgd":
        return torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.adam_mode == "plain":
        return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


class SpikeStats:
    def __init__(self, net: torch.nn.Module, spike_cls: type):
        self.counts: List[float] = []
        self.elems: List[float] = []
        self._handles = []
        for _, module in net.named_modules():
            if isinstance(module, spike_cls):
                idx = len(self.counts)
                self.counts.append(0.0)
                self.elems.append(0.0)
                self._handles.append(module.register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx: int):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t):
                d = t.detach().float()
                self.counts[idx] += float(d.sum().item())
                self.elems[idx] += float(d.numel())

        return hook

    def close(self):
        for h in self._handles:
            h.remove()


def loss_from_rates(out_fr: torch.Tensor, y: torch.Tensor):
    if args.loss == "ce":
        return F.cross_entropy(out_fr, y)
    return F.mse_loss(out_fr, F.one_hot(y, NUM_CLASSES).float())


def run_epoch(net: torch.nn.Module, loader: DataLoader, *, train: bool, optimizer: Optional[torch.optim.Optimizer], scaler: Optional[amp.GradScaler]):
    net.train(train)
    stats = SpikeStats(net, NEURON_REGISTRY[args.model][0])
    dims = spiking_layer_dims()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_sum = 0.0
    correct = 0
    total = 0
    max_cpu = get_process_memory_bytes()
    input_nonzero_total = 0.0
    input_numel_total = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    t0 = time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, (x, y) in enumerate(loader):
            if train and args.max_train_batches and bi >= args.max_train_batches:
                break
            if (not train) and args.max_test_batches and bi >= args.max_test_batches:
                break
            if train:
                optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_seq = repeat_static_input(x, args.T, gain=args.input_gain)
            input_nonzero_total += float(x_seq.ne(0).sum().item())
            input_numel_total += float(x_seq.numel())

            if train and scaler is not None:
                with amp.autocast():
                    out_seq = net(x_seq)
                    out_fr = out_seq.mean(0) if out_seq.dim() >= 3 else out_seq
                    loss = loss_from_rates(out_fr, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # type: ignore[arg-type]
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)  # type: ignore[arg-type]
                scaler.update()
            else:
                out_seq = net(x_seq)
                out_fr = out_seq.mean(0) if out_seq.dim() >= 3 else out_seq
                loss = loss_from_rates(out_fr, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()  # type: ignore[union-attr]

            pred = out_fr.argmax(dim=1)
            n = y.numel()
            total += n
            loss_sum += float(loss.item()) * n
            correct += int((pred == y).sum().item())
            update_confusion_matrix(confusion, pred.detach().cpu(), y.detach().cpu())
            functional.reset_net(net)
            max_cpu = max(max_cpu, get_process_memory_bytes())

    synchronize_if_needed(device)
    elapsed = time.time() - t0
    gpu_alloc = 0.0
    gpu_res = 0.0
    if device.type == "cuda":
        gpu_alloc = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        gpu_res = bytes_to_mb(torch.cuda.max_memory_reserved(device))

    summary = init_activity_summary(len(dims))
    prev_spike = input_nonzero_total
    prev_numel = input_numel_total
    prev_features = INPUT_SIZE
    for i, out_features in enumerate(dims):
        spike_count = stats.counts[i] if i < len(stats.counts) else 0.0
        num_elements = stats.elems[i] if i < len(stats.elems) else 0.0
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
        prev_spike = spike_count
        prev_numel = num_elements
        prev_features = out_features
    stats.close()
    cls = macro_classification_metrics(confusion)
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "macro_precision": cls["macro_precision"],
        "macro_recall": cls["macro_recall"],
        "macro_f1": cls["macro_f1"],
        "samples": total,
        "time_sec": elapsed,
        "samples_per_sec": total / max(elapsed, 1e-9),
        "latency_ms_per_sample": 1000.0 * elapsed / max(total, 1),
        "cpu_memory_mb": bytes_to_mb(max_cpu),
        "gpu_memory_allocated_mb": gpu_alloc,
        "gpu_memory_reserved_mb": gpu_res,
        "confusion_matrix": confusion,
        "activity_summary": summary,
    }


transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=max(1024, args.b), shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | device={device}")

net = build_net().to(device)
initialize_lazy_neuron_parameters(net, device)
optimizer = build_optimizer(net)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) if args.cosine_scheduler else None
scaler = amp.GradScaler() if args.amp else None

run_id = time.strftime("%Y%m%d_%H%M%S")
artifact_prefix = os.path.join(args.out_dir, f"MNIST_{args.model}_BP_v2_{run_id}")
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
    "tau_response", "tau_refractory", "input_gain", "lr", "weight_decay", "alpha", "label_scale",
    "eval_subset", "train_acc", "train_macro_precision", "train_macro_recall", "train_macro_f1",
    "test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1", "best_test_acc",
    "best_test_macro_f1", "train_time_sec", "test_time_sec", "epoch_time_sec", "train_speed",
    "test_speed", "train_latency_ms_per_sample", "test_latency_ms_per_sample", "train_cpu_memory_mb",
    "test_cpu_memory_mb", "train_gpu_memory_allocated_mb", "train_gpu_memory_reserved_mb",
    "test_gpu_memory_allocated_mb", "test_gpu_memory_reserved_mb", "test_confusion_path",
    "best_test_confusion_path", "final_test_confusion_path", "best_model_path", "final_model_path",
] + [f"layer_{i}_loss" for i in range(len(spiking_layer_dims()))]
for prefix in ("train", "test"):
    csv_columns.extend([
        f"{prefix}_total_spikes", f"{prefix}_global_spike_rate", f"{prefix}_dense_synops",
        f"{prefix}_event_synops", f"{prefix}_energy_proxy_synops", f"{prefix}_event_to_dense_ratio",
    ])
    for i in range(len(spiking_layer_dims())):
        csv_columns.extend([
            f"{prefix}_layer_{i}_spike_count", f"{prefix}_layer_{i}_spike_rate",
            f"{prefix}_layer_{i}_active_input_rate", f"{prefix}_layer_{i}_dense_synops",
            f"{prefix}_layer_{i}_event_synops", f"{prefix}_layer_{i}_event_to_dense_ratio",
        ])
pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)

start_epoch = 0
best_test_acc = 0.0
best_test_macro_f1 = 0.0
last_test_metrics = None
if args.resume:
    ckpt = torch.load(args.resume, map_location="cpu")
    net.load_state_dict(ckpt["net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "lr_scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["lr_scheduler"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_test_acc = float(ckpt.get("best_test_acc", 0.0))
    best_test_macro_f1 = float(ckpt.get("best_test_macro_f1", 0.0))

for epoch in range(start_epoch, args.epochs):
    epoch_start = time.time()
    train_metrics = run_epoch(net, train_loader, train=True, optimizer=optimizer, scaler=scaler)
    if scheduler is not None:
        scheduler.step()
    lr_now = float(optimizer.param_groups[0]["lr"])
    test_metrics = run_epoch(net, test_loader, train=False, optimizer=None, scaler=None)
    last_test_metrics = test_metrics

    test_confusion_epoch_path = ""
    if args.save_test_confusion_every_epoch:
        test_confusion_epoch_path = os.path.join(test_confusion_dir, f"epoch_{epoch + 1:03d}.csv")
        save_confusion_matrix(test_confusion_epoch_path, test_metrics["confusion_matrix"])

    current_best_test_macro_f1 = max(best_test_macro_f1, test_metrics["macro_f1"])
    if test_metrics["acc"] >= best_test_acc:
        best_test_acc = test_metrics["acc"]
        save_confusion_matrix(best_confusion_path, test_metrics["confusion_matrix"])
        payload = {"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_test_acc": best_test_acc, "best_test_macro_f1": current_best_test_macro_f1, "args": vars(args)}
        if scheduler is not None:
            payload["lr_scheduler"] = scheduler.state_dict()
        torch.save(payload, best_model_path)
    best_test_macro_f1 = current_best_test_macro_f1

    row = {
        "run_id": run_id, "epoch": epoch + 1, "model": args.model, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "T": args.T, "tau": args.tau, "v_threshold": args.v_threshold,
        "v_reset": args.v_reset, "tau_response": args.tau_response, "tau_refractory": args.tau_refractory,
        "input_gain": args.input_gain, "lr": lr_now, "weight_decay": args.weight_decay, "alpha": 0.0,
        "label_scale": 0.0, "eval_subset": 0, "train_acc": train_metrics["acc"],
        "train_macro_precision": train_metrics["macro_precision"], "train_macro_recall": train_metrics["macro_recall"],
        "train_macro_f1": train_metrics["macro_f1"], "test_acc": test_metrics["acc"],
        "test_macro_precision": test_metrics["macro_precision"], "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"], "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_macro_f1, "train_time_sec": train_metrics["time_sec"],
        "test_time_sec": test_metrics["time_sec"], "epoch_time_sec": time.time() - epoch_start,
        "train_speed": train_metrics["samples_per_sec"], "test_speed": test_metrics["samples_per_sec"],
        "train_latency_ms_per_sample": train_metrics["latency_ms_per_sample"],
        "test_latency_ms_per_sample": test_metrics["latency_ms_per_sample"],
        "train_cpu_memory_mb": train_metrics["cpu_memory_mb"], "test_cpu_memory_mb": test_metrics["cpu_memory_mb"],
        "train_gpu_memory_allocated_mb": train_metrics["gpu_memory_allocated_mb"],
        "train_gpu_memory_reserved_mb": train_metrics["gpu_memory_reserved_mb"],
        "test_gpu_memory_allocated_mb": test_metrics["gpu_memory_allocated_mb"],
        "test_gpu_memory_reserved_mb": test_metrics["gpu_memory_reserved_mb"],
        "test_confusion_path": test_confusion_epoch_path, "best_test_confusion_path": best_confusion_path,
        "final_test_confusion_path": final_confusion_path, "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }
    for i in range(len(spiking_layer_dims())):
        row[f"layer_{i}_loss"] = 0.0
    row.update(activity_summary_to_metrics("train", train_metrics["activity_summary"]))
    row.update(activity_summary_to_metrics("test", test_metrics["activity_summary"]))
    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)

final_payload = {"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": args.epochs, "best_test_acc": best_test_acc, "best_test_macro_f1": best_test_macro_f1, "args": vars(args)}
if scheduler is not None:
    final_payload["lr_scheduler"] = scheduler.state_dict()
torch.save(final_payload, final_model_path)
if last_test_metrics is not None:
    save_confusion_matrix(final_confusion_path, last_test_metrics["confusion_matrix"])
    save_json(
        summary_path,
        {
            "run_id": run_id,
            "csv_path": csv_path,
            "args_path": args_path,
            "best_test_confusion_path": best_confusion_path,
            "final_test_confusion_path": final_confusion_path,
            "best_model_path": best_model_path,
            "final_model_path": final_model_path,
            "best_test_acc": best_test_acc,
            "best_test_macro_f1": best_test_macro_f1,
        },
    )
print("Done. CSV saved to:", csv_path)
sys.exit(0)
#!/usr/bin/env python3
"""
MNIST BP-SNN (SpikingJelly) v2 with FF-compatible logging schema.

Input encoding matches `mnist_FF_train_v2.py`:
`repeat_static_input(x, T, gain)` with flattened MNIST [B, 784].
"""
# from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda import amp
from torch.utils.data import DataLoader

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import parametric_lif_net

# -------------------------------------------------
# Args
# -------------------------------------------------
parser = argparse.ArgumentParser(description="MNIST BP-SNN (SpikingJelly) v2")
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/MNIST")
parser.add_argument("-out-dir", type=str, default="./result")
parser.add_argument("-b", type=int, default=64, help="batch size")
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-j", type=int, default=4, help="dataloader workers")
parser.add_argument("-T", type=int, default=10, help="time steps")
parser.add_argument("--input-gain", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=2026)

parser.add_argument("--model", type=str, default="lif", choices=["lif", "alif", "srm", "dynsrm"])
parser.add_argument("--num-layers", type=int, default=2, choices=[1, 2])
parser.add_argument("--hidden-dim", type=int, default=256)

parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v-threshold", type=float, default=0.4)
parser.add_argument("--v-reset", type=float, default=0.0)
parser.add_argument("--tau-response", type=float, default=2.0)
parser.add_argument("--tau-refractory", type=float, default=10.0)

parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("-opt", type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("--adam-mode", type=str, default="plain", choices=["grouped", "plain"])
parser.add_argument("--loss", type=str, default="mse", choices=["mse", "ce"])
parser.add_argument("--cosine-scheduler", action="store_true")
parser.add_argument("-resume", type=str, default="")
parser.add_argument("-amp", action="store_true")
parser.add_argument("-cupy", action="store_true")
parser.add_argument("--max-train-batches", type=int, default=0)
parser.add_argument("--max-test-batches", type=int, default=0)
parser.add_argument("--save-test-confusion-every-epoch", action="store_true")
args, _ = parser.parse_known_args()
print(args)

NUM_CLASSES = 10
INPUT_SIZE = 784

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

NEURON_REGISTRY: Dict[str, Tuple[type, type]] = {
    "lif": (neuron.LIFNode, neuron.LIFNode),
    "alif": (getattr(neuron, "ALIFNode", getattr(neuron, "ParametricLIFNode2", neuron.ParametricLIFNode)), getattr(neuron, "ALIFNode", getattr(neuron, "ParametricLIFNode2", neuron.ParametricLIFNode))),
    "srm": (neuron.SRMNode, neuron.SRMNode),
    "dynsrm": (
        getattr(neuron, "DynamicSRMNode", neuron.SRMNode),
        getattr(neuron, "DynamicSRMNode", neuron.SRMNode),
    ),
}


def synchronize_if_needed(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


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
    idx = target * NUM_CLASSES + pred
    confusion += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


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


def save_confusion_matrix(path: str, confusion: torch.Tensor):
    pd.DataFrame(
        confusion.tolist(),
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    ).to_csv(path, index=True)


def save_json(path: str, payload: Dict[str, object]):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


@torch.no_grad()
def repeat_static_input(x: torch.Tensor, T: int, gain: float = 1.0):
    x = x * float(gain)
    return x.unsqueeze(0).repeat(T, 1, 1)


def spiking_layer_dims() -> List[int]:
    if args.num_layers == 1:
        return [NUM_CLASSES]
    return [args.hidden_dim, NUM_CLASSES]


def dense_synops_per_sample() -> int:
    dims = [INPUT_SIZE] + spiking_layer_dims()
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1]
    return args.T * total


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

    for layer_idx, layer in enumerate(summary["layers"]):
        spike_count = float(layer["spike_count"])
        num_elements = float(layer["num_elements"])
        input_nonzero_count = float(layer["input_nonzero_count"])
        input_num_elements = float(layer["input_num_elements"])
        dense_synops = float(layer["dense_synops"])
        event_synops = float(layer["event_synops"])
        metrics[f"{prefix}_layer_{layer_idx}_spike_count"] = spike_count
        metrics[f"{prefix}_layer_{layer_idx}_spike_rate"] = spike_count / max(num_elements, 1.0)
        metrics[f"{prefix}_layer_{layer_idx}_active_input_rate"] = input_nonzero_count / max(input_num_elements, 1.0)
        metrics[f"{prefix}_layer_{layer_idx}_dense_synops"] = dense_synops
        metrics[f"{prefix}_layer_{layer_idx}_event_synops"] = event_synops
        metrics[f"{prefix}_layer_{layer_idx}_event_to_dense_ratio"] = event_synops / max(dense_synops, 1.0)
    return metrics


def neuron_kwargs():
    base = dict(v_threshold=args.v_threshold, v_reset=args.v_reset, store_v_seq=False)
    if args.model == "lif":
        return {**base, "tau": args.tau}
    if args.model == "alif":
        return {**base, "init_tau": args.tau}
    if args.model == "srm":
        return {**base, "tau_response": args.tau_response, "tau_refractory": args.tau_refractory}
    if NEURON_REGISTRY["dynsrm"][0] is neuron.SRMNode:
        return {**base, "tau_response": args.tau_response, "tau_refractory": args.tau_refractory}
    return {**base, "init_tau_response": args.tau_response, "init_tau_refractory": args.tau_refractory}


def build_net() -> torch.nn.Module:
    cls, _ = NEURON_REGISTRY[args.model]
    common = dict(
        input_size=INPUT_SIZE,
        spiking_neuron=cls,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        output_spiking=True,
        **neuron_kwargs(),
    )
    net = parametric_lif_net.MNISTNet1(**common) if args.num_layers == 1 else parametric_lif_net.MNISTNet2(hidden_size=args.hidden_dim, **common)
    functional.set_step_mode(net, "m")
    if args.cupy:
        functional.set_backend(net, "cupy", instance=NEURON_REGISTRY[args.model][1])
    return net



def initialize_lazy_neuron_parameters(net: torch.nn.Module, device_: torch.device) -> None:
    """Materialize lazy per-neuron tau parameters without running a forward pass."""
    prev_out_features = None

    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            prev_out_features = int(module.out_features)
            continue

        # ALIF-style lazy parameter: ParametricLIFNode2 / ALIFNode
        if hasattr(module, "w") and hasattr(module, "_init_w") and getattr(module, "w") is None:
            num_neurons = getattr(module, "num_neurons", None)
            if num_neurons is None:
                num_neurons = prev_out_features
            if num_neurons is None:
                continue
            w = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w), device=device_, dtype=torch.float32))
            module.w = w
            module.register_parameter("w", module.w)
            continue

        # DynamicSRM-style lazy parameters
        if (hasattr(module, "w_tau_response") and hasattr(module, "w_tau_refractory")
                and hasattr(module, "_init_w_response") and hasattr(module, "_init_w_refractory")
                and (getattr(module, "w_tau_response") is None or getattr(module, "w_tau_refractory") is None)):
            num_neurons = getattr(module, "num_neurons", None)
            if num_neurons is None:
                num_neurons = prev_out_features
            if num_neurons is None:
                continue
            w_resp = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w_response), device=device_, dtype=torch.float32))
            w_refr = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w_refractory), device=device_, dtype=torch.float32))
            module.w_tau_response = w_resp
            module.w_tau_refractory = w_refr
            module.register_parameter("w_tau_response", module.w_tau_response)
            module.register_parameter("w_tau_refractory", module.w_tau_refractory)


def param_groups(net: torch.nn.Module):
    weight_params, bias_params, other_params = [], [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name:
            bias_params.append(p)
        elif "weight" in name:
            weight_params.append(p)
        else:
            other_params.append(p)
    return weight_params, bias_params, other_params


def build_optimizer(net: torch.nn.Module):
    if args.opt == "sgd":
        w, b, o = param_groups(net)
        return torch.optim.SGD(
            [
                {"params": w, "weight_decay": args.weight_decay},
                {"params": b, "weight_decay": 0.0},
                {"params": o, "weight_decay": args.weight_decay if o else 0.0},
            ],
            lr=args.lr,
            momentum=args.momentum,
        )
    if args.adam_mode == "plain":
        return torch.optim.Adam(net.parameters(), lr=args.lr)
    w, b, o = param_groups(net)
    return torch.optim.Adam(
        [
            {"params": w, "weight_decay": args.weight_decay},
            {"params": b, "weight_decay": 0.0},
            {"params": o, "weight_decay": args.weight_decay if o else 0.0},
        ],
        lr=args.lr,
    )


class SpikeStats:
    def __init__(self, net: torch.nn.Module, spike_cls: type):
        self.names: List[str] = []
        self.counts: List[float] = []
        self.elems: List[float] = []
        self._handles = []
        for name, module in net.named_modules():
            if isinstance(module, spike_cls):
                idx = len(self.names)
                self.names.append(name)
                self.counts.append(0.0)
                self.elems.append(0.0)
                self._handles.append(module.register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx: int):
        def hook(_m, _i, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(tensor):
                detached = tensor.detach().float()
                self.counts[idx] += float(detached.sum().item())
                self.elems[idx] += float(detached.numel())

        return hook

    def close(self):
        for h in self._handles:
            h.remove()


def loss_from_rates(out_fr: torch.Tensor, y: torch.Tensor):
    if args.loss == "ce":
        return F.cross_entropy(out_fr, y)
    return F.mse_loss(out_fr, F.one_hot(y, NUM_CLASSES).float())


def run_epoch(
    net: torch.nn.Module,
    loader: DataLoader,
    *,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[amp.GradScaler],
):
    net.train(train)
    spike_cls = NEURON_REGISTRY[args.model][0]
    stats = SpikeStats(net, spike_cls)
    dims = spiking_layer_dims()

    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_sum = 0.0
    correct = 0
    total = 0
    max_cpu_memory_bytes = get_process_memory_bytes()
    input_nonzero_total = 0.0
    input_numel_total = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    start_time = time.time()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, (x, y) in enumerate(loader):
            if train and args.max_train_batches and bi >= args.max_train_batches:
                break
            if (not train) and args.max_test_batches and bi >= args.max_test_batches:
                break

            if train:
                optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_seq = repeat_static_input(x, args.T, gain=args.input_gain)

            input_nonzero_total += float(x_seq.ne(0).sum().item())
            input_numel_total += float(x_seq.numel())

            if train and scaler is not None:
                with amp.autocast():
                    out_seq = net(x_seq)
                    out_fr = out_seq.mean(0) if out_seq.dim() >= 3 else out_seq
                    loss = loss_from_rates(out_fr, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # type: ignore[arg-type]
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)  # type: ignore[arg-type]
                scaler.update()
            else:
                out_seq = net(x_seq)
                out_fr = out_seq.mean(0) if out_seq.dim() >= 3 else out_seq
                loss = loss_from_rates(out_fr, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()  # type: ignore[union-attr]

            pred = out_fr.argmax(dim=1)
            n = y.numel()
            total += n
            loss_sum += float(loss.item()) * n
            correct += int((pred == y).sum().item())
            update_confusion_matrix(confusion, pred.detach().cpu(), y.detach().cpu())
            functional.reset_net(net)
            max_cpu_memory_bytes = max(max_cpu_memory_bytes, get_process_memory_bytes())

    synchronize_if_needed(device)
    elapsed = time.time() - start_time
    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0
    if device.type == "cuda":
        gpu_allocated_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        gpu_reserved_mb = bytes_to_mb(torch.cuda.max_memory_reserved(device))

    cls_metrics = macro_classification_metrics(confusion)
    activity_summary = init_activity_summary(len(dims))
    prev_spike = input_nonzero_total
    prev_numel = input_numel_total
    prev_features = INPUT_SIZE
    for i, out_features in enumerate(dims):
        spike_count = stats.counts[i] if i < len(stats.counts) else 0.0
        num_elements = stats.elems[i] if i < len(stats.elems) else 0.0
        dense_synops = float(args.T * total * prev_features * out_features)
        event_synops = float(prev_spike * out_features)
        layer = activity_summary["layers"][i]
        layer["spike_count"] = spike_count
        layer["num_elements"] = num_elements
        layer["input_nonzero_count"] = prev_spike
        layer["input_num_elements"] = prev_numel
        layer["dense_synops"] = dense_synops
        layer["event_synops"] = event_synops
        activity_summary["total_spike_count"] += spike_count
        activity_summary["total_num_elements"] += num_elements
        activity_summary["total_dense_synops"] += dense_synops
        activity_summary["total_event_synops"] += event_synops
        prev_spike = spike_count
        prev_numel = num_elements
        prev_features = out_features
    stats.close()

    return {
        "loss": loss_sum / max(total, 1),
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


# -------------------------------------------------
# Data
# -------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
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
print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | device={device}")

net = build_net().to(device)
print(net)
initialize_lazy_neuron_parameters(net, device)
optimizer = build_optimizer(net)
scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
if args.cosine_scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
scaler: Optional[amp.GradScaler] = amp.GradScaler() if args.amp else None

run_id = time.strftime("%Y%m%d_%H%M%S")
artifact_prefix = os.path.join(args.out_dir, f"MNIST_{args.model}_BP_v2_{run_id}")
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
    "run_id",
    "epoch",
    "model",
    "hidden_dim",
    "num_layers",
    "T",
    "tau",
    "v_threshold",
    "v_reset",
    "tau_response",
    "tau_refractory",
    "input_gain",
    "lr",
    "weight_decay",
    "alpha",
    "label_scale",
    "eval_subset",
    "train_acc",
    "train_macro_precision",
    "train_macro_recall",
    "train_macro_f1",
    "test_acc",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "best_test_acc",
    "best_test_macro_f1",
    "train_time_sec",
    "test_time_sec",
    "epoch_time_sec",
    "train_speed",
    "test_speed",
    "train_latency_ms_per_sample",
    "test_latency_ms_per_sample",
    "train_cpu_memory_mb",
    "test_cpu_memory_mb",
    "train_gpu_memory_allocated_mb",
    "train_gpu_memory_reserved_mb",
    "test_gpu_memory_allocated_mb",
    "test_gpu_memory_reserved_mb",
    "test_confusion_path",
    "best_test_confusion_path",
    "final_test_confusion_path",
    "best_model_path",
    "final_model_path",
] + [f"layer_{i}_loss" for i in range(len(spiking_layer_dims()))]

for prefix in ("train", "test"):
    csv_columns.extend(
        [
            f"{prefix}_total_spikes",
            f"{prefix}_global_spike_rate",
            f"{prefix}_dense_synops",
            f"{prefix}_event_synops",
            f"{prefix}_energy_proxy_synops",
            f"{prefix}_event_to_dense_ratio",
        ]
    )
    for i in range(len(spiking_layer_dims())):
        csv_columns.extend(
            [
                f"{prefix}_layer_{i}_spike_count",
                f"{prefix}_layer_{i}_spike_rate",
                f"{prefix}_layer_{i}_active_input_rate",
                f"{prefix}_layer_{i}_dense_synops",
                f"{prefix}_layer_{i}_event_synops",
                f"{prefix}_layer_{i}_event_to_dense_ratio",
            ]
        )
pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)

start_epoch = 0
best_test_acc = 0.0
best_test_macro_f1 = 0.0
last_test_metrics = None

if args.resume:
    ckpt = torch.load(args.resume, map_location="cpu")
    net.load_state_dict(ckpt["net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "lr_scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["lr_scheduler"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_test_acc = float(ckpt.get("best_test_acc", 0.0))
    best_test_macro_f1 = float(ckpt.get("best_test_macro_f1", 0.0))

for epoch in range(start_epoch, args.epochs):
    epoch_start = time.time()
    train_metrics = run_epoch(net, train_loader, train=True, optimizer=optimizer, scaler=scaler)
    if scheduler is not None:
        scheduler.step()
    lr_now = float(optimizer.param_groups[0]["lr"])

    test_metrics = run_epoch(net, test_loader, train=False, optimizer=None, scaler=None)
    last_test_metrics = test_metrics

    test_confusion_epoch_path = ""
    if args.save_test_confusion_every_epoch:
        test_confusion_epoch_path = os.path.join(test_confusion_dir, f"epoch_{epoch + 1:03d}.csv")
        save_confusion_matrix(test_confusion_epoch_path, test_metrics["confusion_matrix"])

    current_best_test_macro_f1 = max(best_test_macro_f1, test_metrics["macro_f1"])
    if test_metrics["acc"] >= best_test_acc:
        best_test_acc = test_metrics["acc"]
        save_confusion_matrix(best_confusion_path, test_metrics["confusion_matrix"])
        payload = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_test_acc": best_test_acc,
            "best_test_macro_f1": current_best_test_macro_f1,
            "args": vars(args),
        }
        if scheduler is not None:
            payload["lr_scheduler"] = scheduler.state_dict()
        torch.save(payload, best_model_path)
    best_test_macro_f1 = current_best_test_macro_f1

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
        "lr": lr_now,
        "weight_decay": args.weight_decay,
        "alpha": 0.0,
        "label_scale": 0.0,
        "eval_subset": 0,
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
        "train_time_sec": train_metrics["time_sec"],
        "test_time_sec": test_metrics["time_sec"],
        "epoch_time_sec": time.time() - epoch_start,
        "train_speed": train_metrics["samples_per_sec"],
        "test_speed": test_metrics["samples_per_sec"],
        "train_latency_ms_per_sample": train_metrics["latency_ms_per_sample"],
        "test_latency_ms_per_sample": test_metrics["latency_ms_per_sample"],
        "train_cpu_memory_mb": train_metrics["cpu_memory_mb"],
        "test_cpu_memory_mb": test_metrics["cpu_memory_mb"],
        "train_gpu_memory_allocated_mb": train_metrics["gpu_memory_allocated_mb"],
        "train_gpu_memory_reserved_mb": train_metrics["gpu_memory_reserved_mb"],
        "test_gpu_memory_allocated_mb": test_metrics["gpu_memory_allocated_mb"],
        "test_gpu_memory_reserved_mb": test_metrics["gpu_memory_reserved_mb"],
        "test_confusion_path": test_confusion_epoch_path,
        "best_test_confusion_path": best_confusion_path,
        "final_test_confusion_path": final_confusion_path,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }
    for i in range(len(spiking_layer_dims())):
        row[f"layer_{i}_loss"] = 0.0
    row.update(activity_summary_to_metrics("train", train_metrics["activity_summary"]))
    row.update(activity_summary_to_metrics("test", test_metrics["activity_summary"]))
    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)

    print("\n" + "=" * 90)
    print(
        f"Epoch {epoch + 1}/{args.epochs} | model={args.model} | T={args.T} | "
        f"layers={args.num_layers} | hidden={args.hidden_dim} | lr={lr_now:g}"
    )
    print("-" * 90)
    print(
        f"Train Acc/F1: {train_metrics['acc'] * 100:.2f}% / {train_metrics['macro_f1'] * 100:.2f}% | "
        f"Test Acc/F1: {test_metrics['acc'] * 100:.2f}% / {test_metrics['macro_f1'] * 100:.2f}%"
    )
    print(
        f"Best Test Acc/F1: {best_test_acc * 100:.2f}% / {best_test_macro_f1 * 100:.2f}% | "
        f"Train Speed: {train_metrics['samples_per_sec']:.2f} samples/s | "
        f"Test Speed: {test_metrics['samples_per_sec']:.2f} samples/s"
    )
    print(
        f"Train/Test Time: {train_metrics['time_sec']:.2f}s / {test_metrics['time_sec']:.2f}s | "
        f"Train/Test Latency: {train_metrics['latency_ms_per_sample']:.4f} / "
        f"{test_metrics['latency_ms_per_sample']:.4f} ms/sample"
    )
    print(
        f"Test Spike Rate: {row['test_global_spike_rate']:.6f} | "
        f"Test Event SynOps: {row['test_event_synops']:.2f} | "
        f"Event/Dense Ratio: {row['test_event_to_dense_ratio']:.6f}"
    )
    print("Layer losses:", ", ".join([f"{row[f'layer_{i}_loss']:.6f}" for i in range(len(spiking_layer_dims()))]))
    print("=" * 90)

final_payload = {
    "net": net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": args.epochs,
    "best_test_acc": best_test_acc,
    "best_test_macro_f1": best_test_macro_f1,
    "args": vars(args),
}
if scheduler is not None:
    final_payload["lr_scheduler"] = scheduler.state_dict()
torch.save(final_payload, final_model_path)

if last_test_metrics is not None:
    save_confusion_matrix(final_confusion_path, last_test_metrics["confusion_matrix"])
    save_json(
        summary_path,
        {
            "run_id": run_id,
            "csv_path": csv_path,
            "args_path": args_path,
            "best_test_confusion_path": best_confusion_path,
            "final_test_confusion_path": final_confusion_path,
            "best_model_path": best_model_path,
            "final_model_path": final_model_path,
            "best_test_acc": best_test_acc,
            "best_test_macro_f1": best_test_macro_f1,
        },
    )

print("Done. CSV saved to:", csv_path)
#!/usr/bin/env python3
"""
MNIST backprop SNN (SpikingJelly) — BP v2.

Input encoding matches `mnist_FF_train_v2.py`: deterministic static repeat
  [T,N,784] = same scaled pixel vector every timestep (`repeat_static_input` + `--input-gain`),
for fair comparison with Forward–Forward runs. One forward over the sequence, mean rate vs label.

Still supports multiple neurons (lif / alif / srm / dynsrm) and 1–2 FC layers.

Per epoch logs (CSV + console):
  - Inference-style throughput on the test pass (forward-only, samples/s)
  - End-to-end train throughput (forward + backward)
  - Energy proxies: total hidden/output spikes, global spike rate, dense MAC-style op count
  - Peak CPU RSS and CUDA alloc/reserved memory

Example:
  python -u mnist_bp_train_v2.py --model lif -T 128 -b 256
"""
# from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda import amp
from torch.utils.data import DataLoader

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import parametric_lif_net

NUM_CLASSES = 10
INPUT_SIZE = 784

# -------------------------------------------------
# Args
# -------------------------------------------------
parser = argparse.ArgumentParser(description="MNIST BP-SNN (SpikingJelly) v2")
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/MNIST")
parser.add_argument("-out-dir", type=str, default="./result")
parser.add_argument("-b", type=int, default=64, help="batch size")
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-j", type=int, default=4, help="dataloader workers")
parser.add_argument("-T", type=int, default=10, help="time steps (static repeat, same as FF v2)")
parser.add_argument("--input-gain", type=float, default=1.0, help="scale flattened pixels before repeat (same as FF v2)")
parser.add_argument("--seed", type=int, default=2026)

parser.add_argument("--model", type=str, default="lif", choices=["lif", "alif", "srm", "dynsrm"])
parser.add_argument("--num-layers", type=int, default=2, choices=[1, 2])
parser.add_argument("--hidden-dim", type=int, default=256)

parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v-threshold", type=float, default=0.4)
parser.add_argument("--v-reset", type=float, default=0.0)
parser.add_argument("--tau-response", type=float, default=2.0)
parser.add_argument("--tau-refractory", type=float, default=10.0)

parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0)
parser.add_argument("-opt", type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("--adam-mode", type=str, default="plain", choices=["grouped", "plain"])
parser.add_argument("--loss", type=str, default="mse", choices=["mse", "ce"])
parser.add_argument("--cosine-scheduler", action="store_true", help="CosineAnnealingLR per epoch (off = fixed lr, like lif_fc_mnist)")
parser.add_argument("-resume", type=str, default="")
parser.add_argument("-amp", action="store_true")
parser.add_argument("-cupy", action="store_true")
parser.add_argument("--max-train-batches", type=int, default=0)
parser.add_argument("--max-test-batches", type=int, default=0)

args, _ = parser.parse_known_args()
print(args)

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

NEURON_REGISTRY: Dict[str, Tuple[type, type]] = {
    "lif": (neuron.LIFNode, neuron.LIFNode),
    "alif": (getattr(neuron, "ALIFNode", getattr(neuron, "ParametricLIFNode2", neuron.ParametricLIFNode)), getattr(neuron, "ALIFNode", getattr(neuron, "ParametricLIFNode2", neuron.ParametricLIFNode))),
    "srm": (neuron.SRMNode, neuron.SRMNode),
    "dynsrm": (
        getattr(neuron, "DynamicSRMNode", neuron.SRMNode),
        getattr(neuron, "DynamicSRMNode", neuron.SRMNode),
    ),
}


def synchronize_if_needed(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def get_process_memory_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss) if sys.platform == "darwin" else int(rss) * 1024
    return 0


def bytes_to_mb(n: int) -> float:
    return float(n) / (1024.0 * 1024.0)


def dense_mac_per_sample(T: int) -> int:
    """Rough MAC count per image for full unrolled sequence (matches FC depth)."""
    if args.num_layers == 1:
        return T * INPUT_SIZE * NUM_CLASSES
    h = args.hidden_dim
    return T * (INPUT_SIZE * h + h * NUM_CLASSES)


def update_confusion_matrix(cm: torch.Tensor, pred: torch.Tensor, tgt: torch.Tensor) -> None:
    idx = tgt * NUM_CLASSES + pred
    cm += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def macro_metrics(cm: torch.Tensor) -> Dict[str, float]:
    cm = cm.to(torch.float64)
    tp = cm.diag()
    pred_c = cm.sum(0)
    act_c = cm.sum(1)
    prec = torch.where(pred_c > 0, tp / pred_c, torch.zeros_like(tp))
    rec = torch.where(act_c > 0, tp / act_c, torch.zeros_like(tp))
    den = prec + rec
    f1 = torch.where(den > 0, 2 * prec * rec / den, torch.zeros_like(tp))
    return {
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
    }


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_confusion_csv(path: str, cm: torch.Tensor) -> None:
    pd.DataFrame(
        cm.tolist(),
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    ).to_csv(path, index=True)


@torch.no_grad()
def repeat_static_input(x: torch.Tensor, T: int, gain: float = 1.0) -> torch.Tensor:
    """Same as FF v2 / snntorch-style static encoding: x: [N,F] -> [T,N,F]."""
    x = x * float(gain)
    return x.unsqueeze(0).repeat(T, 1, 1)


def neuron_kwargs() -> dict:
    base = dict(v_threshold=args.v_threshold, v_reset=args.v_reset, store_v_seq=False)
    if args.model == "lif":
        return {**base, "tau": args.tau}
    if args.model == "alif":
        return {**base, "init_tau": args.tau}
    if args.model == "srm":
        return {**base, "tau_response": args.tau_response, "tau_refractory": args.tau_refractory}
    if args.model == "dynsrm":
        if NEURON_REGISTRY["dynsrm"][0] is neuron.SRMNode:
            return {**base, "tau_response": args.tau_response, "tau_refractory": args.tau_refractory}
        return {
            **base,
            "init_tau_response": args.tau_response,
            "init_tau_refractory": args.tau_refractory,
        }
    raise ValueError(args.model)


def build_net() -> torch.nn.Module:
    cls, _ = NEURON_REGISTRY[args.model]
    common = dict(
        input_size=INPUT_SIZE,
        spiking_neuron=cls,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        output_spiking=True,
        **neuron_kwargs(),
    )
    net = (
        parametric_lif_net.MNISTNet1(**common)
        if args.num_layers == 1
        else parametric_lif_net.MNISTNet2(hidden_size=args.hidden_dim, **common)
    )
    functional.set_step_mode(net, "m")
    if args.cupy:
        functional.set_backend(net, "cupy", instance=NEURON_REGISTRY[args.model][1])
    return net



def initialize_lazy_neuron_parameters(net: torch.nn.Module, device_: torch.device) -> None:
    """Materialize lazy per-neuron tau parameters without running a forward pass."""
    prev_out_features = None

    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            prev_out_features = int(module.out_features)
            continue

        # ALIF-style lazy parameter: ParametricLIFNode2 / ALIFNode
        if hasattr(module, "w") and hasattr(module, "_init_w") and getattr(module, "w") is None:
            num_neurons = getattr(module, "num_neurons", None)
            if num_neurons is None:
                num_neurons = prev_out_features
            if num_neurons is None:
                continue
            w = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w), device=device_, dtype=torch.float32))
            module.w = w
            module.register_parameter("w", module.w)
            continue

        # DynamicSRM-style lazy parameters
        if (hasattr(module, "w_tau_response") and hasattr(module, "w_tau_refractory")
                and hasattr(module, "_init_w_response") and hasattr(module, "_init_w_refractory")
                and (getattr(module, "w_tau_response") is None or getattr(module, "w_tau_refractory") is None)):
            num_neurons = getattr(module, "num_neurons", None)
            if num_neurons is None:
                num_neurons = prev_out_features
            if num_neurons is None:
                continue
            w_resp = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w_response), device=device_, dtype=torch.float32))
            w_refr = torch.nn.Parameter(torch.full((int(num_neurons),), float(module._init_w_refractory), device=device_, dtype=torch.float32))
            module.w_tau_response = w_resp
            module.w_tau_refractory = w_refr
            module.register_parameter("w_tau_response", module.w_tau_response)
            module.register_parameter("w_tau_refractory", module.w_tau_refractory)


def param_groups(net: torch.nn.Module) -> Tuple[List, List, List]:
    w, b, o = [], [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name:
            b.append(p)
        elif "weight" in name:
            w.append(p)
        elif name.endswith(".neuron.w") or "w_tau_response" in name or "w_tau_refractory" in name:
            b.append(p)
        else:
            o.append(p)
    return w, b, o


def build_optimizer(net: torch.nn.Module) -> torch.optim.Optimizer:
    if args.opt == "sgd":
        w, b, o = param_groups(net)
        return torch.optim.SGD(
            [
                {"params": w, "weight_decay": args.weight_decay},
                {"params": b, "weight_decay": 0.0},
                {"params": o, "weight_decay": args.weight_decay if o else 0.0},
            ],
            lr=args.lr,
            momentum=args.momentum,
        )
    if args.adam_mode == "plain":
        return torch.optim.Adam(net.parameters(), lr=args.lr)
    w, b, o = param_groups(net)
    return torch.optim.Adam(
        [
            {"params": w, "weight_decay": args.weight_decay},
            {"params": b, "weight_decay": 0.0},
            {"params": o, "weight_decay": args.weight_decay if o else 0.0},
        ],
        lr=args.lr,
    )


class SpikeStats:
    """Forward hooks on spiking neuron modules — same idea as counting spikes per layer."""

    def __init__(self, net: torch.nn.Module, spike_cls: type):
        self._hs = []
        self.tot_spikes = 0.0
        self.tot_elems = 0.0
        for _, m in net.named_modules():
            if isinstance(m, spike_cls):
                self._hs.append(m.register_forward_hook(self._hook))

    def _hook(self, _m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            d = t.detach().float()
            self.tot_spikes += float(d.sum().item())
            self.tot_elems += float(d.numel())

    def close(self) -> None:
        for h in self._hs:
            h.remove()


def loss_from_rates(out_fr: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if args.loss == "ce":
        return F.cross_entropy(out_fr, y)
    return F.mse_loss(out_fr, F.one_hot(y, NUM_CLASSES).float())


def mean_rate(net: torch.nn.Module, x_seq: torch.Tensor) -> torch.Tensor:
    out = net(x_seq)
    return out.mean(0) if out.dim() >= 3 else out


def run_epoch(
    net: torch.nn.Module,
    loader: DataLoader,
    *,
    train: bool,
    optim: Optional[torch.optim.Optimizer],
    scaler: Optional[amp.GradScaler],
    track_spikes: bool,
) -> Dict[str, object]:
    """
    One pass over `loader`. Test path is pure inference (forward only) — use
    `eval_inference_samples_per_sec` for “inference speed”. Train path includes backward.
    """
    spike_cls = NEURON_REGISTRY[args.model][0]
    stats = SpikeStats(net, spike_cls) if track_spikes else None
    net.train(train)

    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_sum = correct = samples = 0
    dense_mac_total = 0
    peak_cpu = get_process_memory_bytes()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize_if_needed(device)
    t0 = time.time()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, (x, y) in enumerate(loader):
            if train and args.max_train_batches and bi >= args.max_train_batches:
                break
            if (not train) and args.max_test_batches and bi >= args.max_test_batches:
                break

            if train:
                optim.zero_grad(set_to_none=True)  # type: ignore[union-attr]

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_seq = repeat_static_input(x, args.T, gain=args.input_gain)
            n = y.numel()
            dense_mac_total += dense_mac_per_sample(args.T) * n

            if train and scaler is not None:
                with amp.autocast():
                    out_fr = mean_rate(net, x_seq)
                    loss = loss_from_rates(out_fr, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)  # type: ignore[arg-type]
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optim)  # type: ignore[arg-type]
                scaler.update()
            else:
                out_fr = mean_rate(net, x_seq)
                loss = loss_from_rates(out_fr, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optim.step()  # type: ignore[union-attr]

            pred = out_fr.argmax(1)
            samples += n
            loss_sum += float(loss.item()) * n
            correct += int((pred == y).sum().item())
            update_confusion_matrix(cm, pred.cpu(), y.cpu())
            functional.reset_net(net)
            peak_cpu = max(peak_cpu, get_process_memory_bytes())

    synchronize_if_needed(device)
    elapsed = max(time.time() - t0, 1e-9)
    gpu_a = gpu_r = 0.0
    if device.type == "cuda":
        gpu_a = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        gpu_r = bytes_to_mb(torch.cuda.max_memory_reserved(device))

    mm = macro_metrics(cm)
    spikes = stats.tot_spikes if stats else 0.0
    elems = stats.tot_elems if stats else 0.0
    if stats:
        stats.close()

    sps = samples / elapsed
    out: Dict[str, object] = {
        "loss": loss_sum / max(samples, 1),
        "acc": correct / max(samples, 1),
        **mm,
        "samples": samples,
        "wall_time_sec": elapsed,
        "throughput_samples_per_sec": sps,
        "latency_ms_per_sample": 1000.0 * elapsed / max(samples, 1),
        "cpu_peak_rss_mb": bytes_to_mb(peak_cpu),
        "gpu_peak_allocated_mb": gpu_a,
        "gpu_peak_reserved_mb": gpu_r,
        "dense_mac_ops_total": float(dense_mac_total),
        "spike_count_total": spikes,
        "spike_tensor_elements_total": elems,
        "global_spike_rate": spikes / max(elems, 1.0),
        "spikes_per_dense_mac": spikes / max(dense_mac_total, 1.0),
        "confusion_matrix": cm,
    }
    return out


# --- data (flattened MNIST, same as FF v2) ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
train_ds = torchvision.datasets.MNIST(args.data_dir, train=True, transform=transform, download=True)
test_ds = torchvision.datasets.MNIST(args.data_dir, train=False, transform=transform, download=True)
train_loader = DataLoader(train_ds, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=max(1024, args.b), shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)

print(f"Train {len(train_ds)} | Test {len(test_ds)} | device={device}")

net = build_net().to(device)
print(net)
initialize_lazy_neuron_parameters(net, device)
optim = build_optimizer(net)
scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
if args.cosine_scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)
scaler = amp.GradScaler() if args.amp else None

run_id = time.strftime("%Y%m%d_%H%M%S")
prefix = os.path.join(args.out_dir, f"MNIST_{args.model}_BP_v2_{run_id}")
paths = {
    "csv": f"{prefix}.csv",
    "args": f"{prefix}_args.json",
    "summary": f"{prefix}_summary.json",
    "best_cm": f"{prefix}_best_test_confusion.csv",
    "final_cm": f"{prefix}_final_test_confusion.csv",
    "best_ckpt": f"{prefix}_best_checkpoint.pth",
    "final_ckpt": f"{prefix}_final_checkpoint.pth",
}
save_json(paths["args"], vars(args))

CSV_COLS = [
    "run_id",
    "epoch",
    "model",
    "num_layers",
    "hidden_dim",
    "T",
    "input_gain",
    "lr",
    "train_loss",
    "train_acc",
    "train_macro_f1",
    "test_loss",
    "test_acc",
    "test_macro_f1",
    "best_test_acc",
    "best_test_macro_f1",
    # speed
    "train_throughput_sps",
    "eval_inference_throughput_sps",
    "train_latency_ms_per_sample",
    "eval_latency_ms_per_sample",
    # memory
    "train_cpu_peak_mb",
    "eval_cpu_peak_mb",
    "train_gpu_alloc_peak_mb",
    "train_gpu_reserved_peak_mb",
    "eval_gpu_alloc_peak_mb",
    "eval_gpu_reserved_peak_mb",
    # energy-style proxies
    "train_spike_count",
    "train_global_spike_rate",
    "train_dense_mac_ops",
    "train_spikes_per_mac",
    "eval_spike_count",
    "eval_global_spike_rate",
    "eval_dense_mac_ops",
    "eval_spikes_per_mac",
    "epoch_wall_sec",
]
pd.DataFrame(columns=CSV_COLS).to_csv(paths["csv"], index=False)

start_epoch = 0
best_acc = 0.0
best_f1 = 0.0
last_eval: Optional[Dict] = None

if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    net.load_state_dict(ck["net"])
    optim.load_state_dict(ck["optimizer"])
    if scheduler is not None and "lr_scheduler" in ck:
        scheduler.load_state_dict(ck["lr_scheduler"])
    start_epoch = int(ck["epoch"]) + 1
    best_acc = float(ck.get("best_test_acc", 0.0))
    best_f1 = float(ck.get("best_test_macro_f1", 0.0))

for epoch in range(start_epoch, args.epochs):
    t_ep = time.time()
    tr = run_epoch(net, train_loader, train=True, optim=optim, scaler=scaler, track_spikes=True)
    if scheduler is not None:
        scheduler.step()
    lr_now = optim.param_groups[0]["lr"]

    ev = run_epoch(net, test_loader, train=False, optim=None, scaler=None, track_spikes=True)
    last_eval = ev

    best_f1 = max(best_f1, float(ev["macro_f1"]))
    if float(ev["acc"]) >= best_acc:
        best_acc = float(ev["acc"])
        save_confusion_csv(paths["best_cm"], ev["confusion_matrix"])  # type: ignore[arg-type]
        payload = {
            "net": net.state_dict(),
            "optimizer": optim.state_dict(),
            "epoch": epoch,
            "best_test_acc": best_acc,
            "best_test_macro_f1": best_f1,
            "args": vars(args),
        }
        if scheduler is not None:
            payload["lr_scheduler"] = scheduler.state_dict()
        torch.save(payload, paths["best_ckpt"])

    row = {
        "run_id": run_id,
        "epoch": epoch + 1,
        "model": args.model,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim if args.num_layers == 2 else 0,
        "T": args.T,
        "input_gain": args.input_gain,
        "lr": lr_now,
        "train_loss": tr["loss"],
        "train_acc": tr["acc"],
        "train_macro_f1": tr["macro_f1"],
        "test_loss": ev["loss"],
        "test_acc": ev["acc"],
        "test_macro_f1": ev["macro_f1"],
        "best_test_acc": best_acc,
        "best_test_macro_f1": best_f1,
        "train_throughput_sps": tr["throughput_samples_per_sec"],
        "eval_inference_throughput_sps": ev["throughput_samples_per_sec"],
        "train_latency_ms_per_sample": tr["latency_ms_per_sample"],
        "eval_latency_ms_per_sample": ev["latency_ms_per_sample"],
        "train_cpu_peak_mb": tr["cpu_peak_rss_mb"],
        "eval_cpu_peak_mb": ev["cpu_peak_rss_mb"],
        "train_gpu_alloc_peak_mb": tr["gpu_peak_allocated_mb"],
        "train_gpu_reserved_peak_mb": tr["gpu_peak_reserved_mb"],
        "eval_gpu_alloc_peak_mb": ev["gpu_peak_allocated_mb"],
        "eval_gpu_reserved_peak_mb": ev["gpu_peak_reserved_mb"],
        "train_spike_count": tr["spike_count_total"],
        "train_global_spike_rate": tr["global_spike_rate"],
        "train_dense_mac_ops": tr["dense_mac_ops_total"],
        "train_spikes_per_mac": tr["spikes_per_dense_mac"],
        "eval_spike_count": ev["spike_count_total"],
        "eval_global_spike_rate": ev["global_spike_rate"],
        "eval_dense_mac_ops": ev["dense_mac_ops_total"],
        "eval_spikes_per_mac": ev["spikes_per_dense_mac"],
        "epoch_wall_sec": time.time() - t_ep,
    }
    pd.DataFrame([row], columns=CSV_COLS).to_csv(paths["csv"], mode="a", header=False, index=False)

    print(
        f"Ep {epoch + 1}/{args.epochs} lr={lr_now:.2e} | "
        f"acc tr={tr['acc']:.4f} te={ev['acc']:.4f} best={best_acc:.4f} | "
        f"infer {ev['throughput_samples_per_sec']:.1f} samp/s | "
        f"spikes te={ev['spike_count_total']:.0f} | "
        f"GPU max {ev['gpu_peak_allocated_mb']:.0f} MB"
    )

final_payload = {
    "net": net.state_dict(),
    "optimizer": optim.state_dict(),
    "epoch": args.epochs - 1,
    "best_test_acc": best_acc,
    "best_test_macro_f1": best_f1,
    "args": vars(args),
}
if scheduler is not None:
    final_payload["lr_scheduler"] = scheduler.state_dict()
torch.save(final_payload, paths["final_ckpt"])

if last_eval is not None:
    save_confusion_csv(paths["final_cm"], last_eval["confusion_matrix"])  # type: ignore[arg-type]

if os.path.isfile(paths["best_ckpt"]):
    net.load_state_dict(torch.load(paths["best_ckpt"], map_location=device)["net"])
net.eval()

save_json(
    paths["summary"],
    {
        "run_id": run_id,
        "paths": paths,
        "best_test_acc": best_acc,
        "best_test_macro_f1": best_f1,
        "note": (
            "Input is FF v2 static repeat_static_input (deterministic), not Poisson. "
            "eval_inference_throughput_sps is test-set forward-only throughput; "
            "train_throughput_sps includes backward. "
            "dense_mac_ops_total is an FC MAC proxy; spike_count_total is from neuron outputs."
        ),
    },
)

print("Done.", paths["csv"], f"BEST_TEST_ACC={best_acc:.6f}")
