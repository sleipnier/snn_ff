#!/usr/bin/env python3
"""
SHD FF-SNN v3.

Main changes relative to v2:
  1) Keep the v2 append-label design by default: [700 SHD channels] + [20 label channels].
     Also supports --label-mode overwrite for direct comparison with the reference SNNFF_SHD.py.
  2) Use reference-inspired time-bin defaults: -T 10, --split-by time.
  3) Add optional current normalization after Linear, especially for hidden layers, matching the
     useful F.normalize(cur2) idea in the reference while keeping implementation clean.
  4) Use firing-rate goodness by default: goodness((spike_count / T)^2), which is more stable
     across T than raw spike count.
  5) Add multi-negative FF training: one positive can be contrasted against K hard negatives.
  6) Add configurable LR schedule. Default is constant LR because the reference effectively keeps
     Adam's LR fixed unless param_groups are explicitly updated.

Recommended first run:
  python -u shd_ff_train_v3_improved.py \
    -device cuda:0 -data-dir ./data/SHD -out-dir ./result \
    -T 10 --split-by time --label-mode append --label-scale 5 \
    --current-normalize hidden --goodness-mode rate --neg-k 3 \
    --lr 1e-3 --lr-schedule constant -epochs 300 \
    --model lif --hidden-dim 500 --num-layers 2 -b 256
"""

import argparse
import json
import os
import random
import sys
import time
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


# -------------------------------------------------
# Args
# -------------------------------------------------
parser = argparse.ArgumentParser(description="SHD FF-SNN v3 improved")
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/SHD")
parser.add_argument("-out-dir", type=str, default="./result")
parser.add_argument("-b", type=int, default=256, help="mini-batch size")
parser.add_argument("-epochs", type=int, default=300)
parser.add_argument("-j", type=int, default=4, help="num dataloader workers")
parser.add_argument("-T", type=int, default=10, help="number of SHD time frames")
parser.add_argument("--split-by", type=str, default="time", choices=["number", "time"])
parser.add_argument("--val-ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--download", action="store_true")

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

# FF improvements
parser.add_argument("--label-mode", type=str, default="append", choices=["append", "overwrite"],
                    help="append preserves all 700 SHD channels; overwrite matches the reference label injection")
parser.add_argument("--label-scale", type=float, default=5.0,
                    help="scale of the active label cue relative to per-sample max frame value")
parser.add_argument("--input-normalize", type=str, default="none", choices=["none", "binary", "max", "log1p", "log1p_max"],
                    help="normalization applied to SHD frame counts before label injection")
parser.add_argument("--current-normalize", type=str, default="hidden", choices=["none", "hidden", "all"],
                    help="normalize Linear current vectors before the spiking neuron; hidden means layer index > 0")
parser.add_argument("--goodness-mode", type=str, default="rate", choices=["count", "rate"],
                    help="count uses raw spike counts; rate divides spike count by T before computing goodness")
parser.add_argument("--ff-loss", type=str, default="swish", choices=["swish", "softplus", "hinton"],
                    help="local FF loss")
parser.add_argument("--theta", type=float, default=2.0, help="threshold for Hinton-style goodness loss")
parser.add_argument("--alpha", type=float, default=4.0, help="ranking strength for swish/softplus FF loss")
parser.add_argument("--neg-k", type=int, default=3, help="number of hard negatives sampled per positive example")
parser.add_argument("--predict-layers", type=str, default="all", choices=["all", "last", "weighted"],
                    help="which layer goodnesses are used for class-wise prediction")
parser.add_argument("--last-layer-weight", type=float, default=2.0,
                    help="extra last-layer weight when --predict-layers weighted")

# optimization
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "reference", "v2"],
                    help="constant is recommended first; reference is a slower decay; v2 reproduces earlier schedule")
parser.add_argument("--eval-subset", type=int, default=0,
                    help="if >0, only use this many training samples for train-acc evaluation")
parser.add_argument("--save-test-confusion-every-epoch", action="store_true")
args, _ = parser.parse_known_args()

if args.neg_k < 1:
    raise ValueError("--neg-k must be >= 1")

INPUT_SIZE = SHD_INPUT_SIZE + NUM_CLASSES if args.label_mode == "append" else SHD_INPUT_SIZE
print(args)
print(f"Effective input size = {INPUT_SIZE} (label_mode={args.label_mode})")

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def synchronize_if_needed(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def get_process_memory_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss) if sys.platform == "darwin" else int(rss) * 1024
    return 0


def bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def update_confusion_matrix(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
    idx = target * NUM_CLASSES + pred
    confusion += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


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


def record_layer_activity(summary, layer_idx: int, x_seq: torch.Tensor, spk_seq: torch.Tensor, out_features: int):
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
        for k in ("spike_count", "num_elements", "input_nonzero_count", "input_num_elements", "dense_synops", "event_synops"):
            dl[k] += sl[k]
    for k in ("total_spike_count", "total_num_elements", "total_dense_synops", "total_event_synops"):
        dst[k] += src[k]


def activity_summary_to_metrics(prefix: str, summary):
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
    pd.DataFrame(
        confusion.tolist(),
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    ).to_csv(path, index=True)


def save_json(path: str, payload: Dict[str, object]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -------------------------------------------------
# Data preprocessing
# -------------------------------------------------
def normalize_frames(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if args.input_normalize == "none":
        return x
    if args.input_normalize == "binary":
        return (x > 0).float()
    if args.input_normalize == "max":
        return x / (x.amax(dim=(0, 1), keepdim=True) + 1e-6)
    if args.input_normalize == "log1p":
        return torch.log1p(x)
    if args.input_normalize == "log1p_max":
        x = torch.log1p(x)
        return x / (x.amax(dim=(0, 1), keepdim=True) + 1e-6)
    raise ValueError(args.input_normalize)


def frame_tf(frames):
    if isinstance(frames, torch.Tensor):
        frames_t = frames.float().reshape(frames.shape[0], -1)
    else:
        frames_t = torch.from_numpy(frames).float().reshape(frames.shape[0], -1)
    return normalize_frames(frames_t)


# -------------------------------------------------
# Label injection and tensor layout
# -------------------------------------------------
def label_strength(x: torch.Tensor) -> torch.Tensor:
    """Per-sample label strength, x is [B, T, 700]."""
    return x.abs().amax(dim=(1, 2)) + 1e-6


def inject_label_temporal(x: torch.Tensor, y: Union[torch.Tensor, int], label_scale: float) -> torch.Tensor:
    """
    x: [B, T, 700]
    append mode:    return [B, T, 720]
    overwrite mode: return [B, T, 700] with x[:, :, :20] overwritten
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x with shape [B,T,F], got {tuple(x.shape)}")
    B, T, F_in = x.shape
    if F_in != SHD_INPUT_SIZE:
        raise ValueError(f"Expected raw SHD feature size {SHD_INPUT_SIZE}, got {F_in}")

    if isinstance(y, int):
        y_t = torch.full((B,), y, device=x.device, dtype=torch.long)
    else:
        y_t = y.to(x.device, dtype=torch.long).view(-1)
    y_t = y_t.clamp(0, NUM_CLASSES - 1)

    strength = label_strength(x) * float(label_scale)  # [B]
    b = torch.arange(B, device=x.device)

    if args.label_mode == "overwrite":
        out = x.clone()
        out[:, :, :NUM_CLASSES] = 0.0
        out[b, :, y_t] = strength[b].unsqueeze(1)
        return out

    label = torch.zeros((B, T, NUM_CLASSES), device=x.device, dtype=x.dtype)
    label[b, :, y_t] = strength[b].unsqueeze(1)
    return torch.cat([x, label], dim=2)


def to_tbf(x_btf: torch.Tensor) -> torch.Tensor:
    return x_btf.permute(1, 0, 2).contiguous()


# -------------------------------------------------
# Goodness and FF loss
# -------------------------------------------------
def goodness(h_count: torch.Tensor) -> torch.Tensor:
    """h_count: [B, H]."""
    h = h_count / float(args.T) if args.goodness_mode == "rate" else h_count
    return h.pow(2).mean(dim=1)


def ff_loss_from_counts(h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
    g_pos = goodness(h_pos)
    g_neg = goodness(h_neg)
    delta = g_pos - g_neg
    if args.ff_loss == "swish":
        return F.silu(-float(args.alpha) * delta).mean()
    if args.ff_loss == "softplus":
        return F.softplus(-float(args.alpha) * delta).mean()
    if args.ff_loss == "hinton":
        loss_pos = F.softplus(float(args.theta) - g_pos).mean()
        loss_neg = F.softplus(g_neg - float(args.theta)).mean()
        return loss_pos + loss_neg
    raise ValueError(args.ff_loss)


def learning_rate_for_epoch(epoch: int, base_lr: float) -> float:
    if args.lr_schedule == "constant":
        return float(base_lr)
    if args.lr_schedule == "reference":
        # A slower decay than v2. The uploaded reference changed a Python variable but did not
        # update Adam param_groups; this schedule is an explicit, conservative version.
        if epoch >= 250:
            return 1e-5
        if epoch >= 200:
            return 5e-5
        if epoch >= 150:
            return 1e-4
        if epoch >= 100:
            return 5e-4
        return float(base_lr)
    if args.lr_schedule == "v2":
        if epoch >= 100:
            return 1e-6
        if epoch >= 75:
            return 1e-5
        if epoch >= 50:
            return 3e-4
        return float(base_lr)
    raise ValueError(args.lr_schedule)


# -------------------------------------------------
# Model
# -------------------------------------------------
def make_neuron_factory(out_features: int):
    common = dict(surrogate_function=surrogate.ATan(), detach_reset=True)
    if args.model == "lif":
        return neuron.LIFNode, dict(common, tau=args.tau, v_threshold=args.v_threshold, v_reset=args.v_reset)

    if args.model == "alif":
        cls = getattr(neuron, "ALIFNode", getattr(neuron, "ParametricLIFNode2", getattr(neuron, "ParametricLIFNode", neuron.LIFNode)))
        kwargs = dict(common, init_tau=args.tau, v_threshold=args.v_threshold, v_reset=args.v_reset)
        if getattr(cls, "__name__", "") in {"ALIFNode", "ParametricLIFNode2"}:
            kwargs["num_neurons"] = out_features
        return cls, kwargs

    if args.model == "srm":
        return neuron.SRMNode, dict(
            common,
            tau_response=args.tau_response,
            tau_refractory=args.tau_refractory,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )

    cls = getattr(neuron, "DynamicSRMNode", neuron.SRMNode)
    kwargs = dict(common, v_threshold=args.v_threshold, v_reset=args.v_reset)
    if cls is neuron.SRMNode:
        kwargs.update(tau_response=args.tau_response, tau_refractory=args.tau_refractory)
    else:
        kwargs.update(init_tau_response=args.tau_response, init_tau_refractory=args.tau_refractory)
        kwargs["num_neurons"] = out_features
    return cls, kwargs


class FFSpikingLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, layer_idx: int, *, lr: float, weight_decay: float):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.fc = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        node_cls, node_kwargs = make_neuron_factory(out_features)
        self.neuron = node_cls(**node_kwargs)
        if hasattr(self.neuron, "step_mode"):
            self.neuron.step_mode = "m"

        self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def should_normalize_current(self) -> bool:
        if args.current_normalize == "all":
            return True
        if args.current_normalize == "hidden" and self.layer_idx > 0:
            return True
        return False

    def run(self, x_seq: torch.Tensor):
        """
        x_seq: [T, B, F]
        returns:
          spk_seq: [T, B, H]
          count:   [B, H]
        """
        functional.reset_net(self)
        cur_seq = self.fc(x_seq)
        if self.should_normalize_current():
            cur_seq = F.normalize(cur_seq, dim=-1)
        spk_seq = self.neuron(cur_seq)
        count = spk_seq.sum(dim=0)
        return spk_seq, count

    def train_ff(self, x_pos_seq: torch.Tensor, x_neg_seq: torch.Tensor):
        self.train()
        _, h_pos = self.run(x_pos_seq)
        _, h_neg = self.run(x_neg_seq)
        loss = ff_loss_from_counts(h_pos, h_neg)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.opt.step()

        with torch.no_grad():
            spk_pos_seq, _ = self.run(x_pos_seq)
            spk_neg_seq, _ = self.run(x_neg_seq)
        return spk_pos_seq.detach(), spk_neg_seq.detach(), float(loss.item())


class FFSpikingNet(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FFSpikingLayer(dims[i], dims[i + 1], i, lr=args.lr, weight_decay=args.weight_decay)
                for i in range(len(dims) - 1)
            ]
        )
        self.dims = dims

    def train_ff(self, x_pos_seq: torch.Tensor, x_neg_seq: torch.Tensor):
        h_pos, h_neg = x_pos_seq, x_neg_seq
        losses = []
        for layer in self.layers:
            h_pos, h_neg, loss_val = layer.train_ff(h_pos, h_neg)
            losses.append(loss_val)
        return losses

    def set_learning_rate(self, lr: float):
        for layer in self.layers:
            for group in layer.opt.param_groups:
                group["lr"] = float(lr)

    def combine_layer_goodnesses(self, gs: List[torch.Tensor]) -> torch.Tensor:
        if args.predict_layers == "last":
            return gs[-1]
        if args.predict_layers == "weighted":
            weights = [1.0] * len(gs)
            weights[-1] = float(args.last_layer_weight)
            return sum(w * g for w, g in zip(weights, gs))
        return sum(gs)

    @torch.no_grad()
    def layer_goodnesses(self, x_btf: torch.Tensor, y_overlay: Union[torch.Tensor, int], activity_summary=None):
        x_labeled = inject_label_temporal(x_btf, y_overlay, label_scale=args.label_scale)
        h_seq = to_tbf(x_labeled) * float(args.input_gain)
        gs = []
        for li, layer in enumerate(self.layers):
            spk_seq, h_count = layer.run(h_seq)
            if activity_summary is not None:
                record_layer_activity(activity_summary, li, h_seq, spk_seq, layer.fc.out_features)
            gs.append(goodness(h_count))
            h_seq = spk_seq.detach()
        return gs

    @torch.no_grad()
    def goodness_per_class(self, x_btf: torch.Tensor, collect_activity: bool = False):
        activity = init_activity_summary(len(self.layers)) if collect_activity else None
        cols = []
        for label in range(NUM_CLASSES):
            gs = self.layer_goodnesses(x_btf, label, activity_summary=activity)
            cols.append(self.combine_layer_goodnesses(gs).unsqueeze(1))
        g = torch.cat(cols, dim=1)
        if collect_activity:
            return g, activity
        return g

    @torch.no_grad()
    def predict(self, x_btf: torch.Tensor, collect_activity: bool = False):
        if collect_activity:
            g, activity = self.goodness_per_class(x_btf, collect_activity=True)
            return g.argmax(dim=1), activity
        return self.goodness_per_class(x_btf).argmax(dim=1)


# -------------------------------------------------
# Hard-negative mining
# -------------------------------------------------
@torch.no_grad()
def sample_hard_negative_labels(model: FFSpikingNet, x_btf: torch.Tensor, y_true: torch.Tensor, k: int, epsilon: float = 1e-12):
    g = model.goodness_per_class(x_btf)  # [B, C]
    g[torch.arange(x_btf.size(0), device=x_btf.device), y_true] = 0.0
    # Clamp prevents invalid probabilities if a local loss makes goodness slightly pathological.
    prob = torch.sqrt(torch.clamp(g, min=0.0) + epsilon)
    return torch.multinomial(prob, num_samples=k, replacement=True)  # [B, K]


@torch.no_grad()
def make_examples(model: FFSpikingNet, x_btf: torch.Tensor, y_true: torch.Tensor):
    """
    Returns positive and negative sequences in [T, B*K, F_eff].
    For K=1 this reduces to the original FF setting. For K>1, each positive
    example is repeated K times and contrasted against K hard negatives.
    """
    B, T, F_in = x_btf.shape
    k = int(args.neg_k)
    y_neg = sample_hard_negative_labels(model, x_btf, y_true, k=k)  # [B, K]

    x_rep = x_btf.unsqueeze(1).expand(B, k, T, F_in).reshape(B * k, T, F_in)
    y_pos_rep = y_true.unsqueeze(1).expand(B, k).reshape(B * k)
    y_neg_flat = y_neg.reshape(B * k)

    x_pos = inject_label_temporal(x_rep, y_pos_rep, label_scale=args.label_scale)
    x_neg = inject_label_temporal(x_rep, y_neg_flat, label_scale=args.label_scale)

    return to_tbf(x_pos) * float(args.input_gain), to_tbf(x_neg) * float(args.input_gain)


# -------------------------------------------------
# Data
# -------------------------------------------------
train_all = SpikingHeidelbergDigits(
    root=args.data_dir,
    train=True,
    data_type="frame",
    frames_number=args.T,
    split_by=args.split_by,
    transform=frame_tf,
)
test_dataset = SpikingHeidelbergDigits(
    root=args.data_dir,
    train=False,
    data_type="frame",
    frames_number=args.T,
    split_by=args.split_by,
    transform=frame_tf,
)

if args.val_ratio > 0:
    n_total = len(train_all)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val
    g_split = torch.Generator().manual_seed(args.seed)
    train_dataset, eval_dataset = random_split(train_all, [n_train, n_val], generator=g_split)
else:
    train_dataset = train_all
    eval_dataset = test_dataset

train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=max(256, args.b), shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=max(256, args.b), shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
print(f"SHD train={len(train_dataset)} eval={len(eval_dataset)} heldout_test={len(test_dataset)}")

# -------------------------------------------------
# Build net
# -------------------------------------------------
dims = [INPUT_SIZE] + [args.hidden_dim] * args.num_layers
net = FFSpikingNet(dims).to(device)
print(net)


def save_model_artifact(path: str, net_: nn.Module, extra_metadata: Optional[Dict[str, object]] = None):
    payload = {
        "model_state_dict": net_.state_dict(),
        "args": vars(args),
        "dims": dims,
        "model_class": net_.__class__.__name__,
        "raw_input_size": SHD_INPUT_SIZE,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "label_mode": args.label_mode,
        "note": "v3 improved FF-SNN: optional append/overwrite label, current normalization, multi-negative training, rate goodness.",
    }
    if extra_metadata is not None:
        payload["metadata"] = extra_metadata
    torch.save(payload, path)


# -------------------------------------------------
# Artifacts
# -------------------------------------------------
run_id = time.strftime("%Y%m%d_%H%M%S")
artifact_prefix = os.path.join(args.out_dir, f"SHD_{args.model}_FF_v3_{run_id}")
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

csv_columns = [
    "run_id", "dataset", "method", "epoch", "model", "hidden_dim", "num_layers", "T", "split_by",
    "val_ratio", "label_mode", "input_normalize", "current_normalize", "goodness_mode", "ff_loss",
    "neg_k", "predict_layers", "tau", "v_threshold", "v_reset", "tau_response", "tau_refractory",
    "input_gain", "lr", "lr_schedule", "weight_decay", "alpha", "theta", "label_scale",
    "train_acc", "train_macro_precision", "train_macro_recall", "train_macro_f1",
    "test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "heldout_test_acc", "heldout_test_macro_f1", "best_test_acc", "best_test_macro_f1",
    "train_time_sec", "test_time_sec", "epoch_time_sec", "train_speed", "test_speed",
    "train_latency_ms_per_sample", "test_latency_ms_per_sample", "train_cpu_memory_mb", "test_cpu_memory_mb",
    "train_gpu_memory_allocated_mb", "train_gpu_memory_reserved_mb", "test_gpu_memory_allocated_mb",
    "test_gpu_memory_reserved_mb", "test_confusion_path", "best_test_confusion_path", "final_test_confusion_path",
    "best_model_path", "final_model_path",
] + [f"layer_{i}_loss" for i in range(len(net.layers))]

for prefix in ("train", "test"):
    csv_columns.extend([
        f"{prefix}_total_spikes", f"{prefix}_global_spike_rate", f"{prefix}_dense_synops",
        f"{prefix}_event_synops", f"{prefix}_energy_proxy_synops", f"{prefix}_event_to_dense_ratio",
    ])
    for i in range(len(net.layers)):
        csv_columns.extend([
            f"{prefix}_layer_{i}_spike_count", f"{prefix}_layer_{i}_spike_rate",
            f"{prefix}_layer_{i}_active_input_rate", f"{prefix}_layer_{i}_dense_synops",
            f"{prefix}_layer_{i}_event_synops", f"{prefix}_layer_{i}_event_to_dense_ratio",
        ])

pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
@torch.no_grad()
def evaluate(loader: DataLoader, max_samples: int = 0, collect_activity: bool = True):
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
            pred = net.predict(x, collect_activity=False)

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
    return {
        "acc": correct / max(total, 1),
        "macro_precision": cls["macro_precision"],
        "macro_recall": cls["macro_recall"],
        "macro_f1": cls["macro_f1"],
        "samples": total,
        "time_sec": elapsed,
        "samples_per_sec": total / max(elapsed, 1e-9),
        "latency_ms_per_sample": 1000.0 * elapsed / max(total, 1),
        "cpu_memory_mb": bytes_to_mb(max_cpu),
        "gpu_memory_allocated_mb": gpu_a,
        "gpu_memory_reserved_mb": gpu_r,
        "confusion_matrix": confusion,
        "activity_summary": activity_summary,
    }


# -------------------------------------------------
# Training
# -------------------------------------------------
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

    # Evaluation. train_metrics can be restricted by --eval-subset to save time.
    train_metrics = evaluate(train_loader, max_samples=args.eval_subset if args.eval_subset > 0 else 0, collect_activity=True)
    eval_metrics = evaluate(eval_loader, collect_activity=True)
    heldout_metrics = evaluate(test_loader, collect_activity=False)
    best_heldout = heldout_metrics
    last_eval_metrics = eval_metrics

    test_confusion_epoch_path = ""
    if args.save_test_confusion_every_epoch:
        test_confusion_epoch_path = os.path.join(test_confusion_dir, f"epoch_{epoch + 1:03d}.csv")
        save_confusion_matrix(test_confusion_epoch_path, eval_metrics["confusion_matrix"])

    if eval_metrics["acc"] >= best_test_acc:
        best_test_acc = eval_metrics["acc"]
        save_confusion_matrix(best_confusion_path, eval_metrics["confusion_matrix"])
        save_model_artifact(
            best_model_path,
            net,
            {
                "epoch": epoch + 1,
                "best_eval_acc": best_test_acc,
                "best_eval_macro_f1": max(best_test_macro_f1, eval_metrics["macro_f1"]),
                "heldout_test_acc_at_selection": heldout_metrics["acc"],
                "heldout_test_macro_f1_at_selection": heldout_metrics["macro_f1"],
            },
        )
    best_test_macro_f1 = max(best_test_macro_f1, eval_metrics["macro_f1"])
    avg_losses = [float(np.mean(v)) if v else 0.0 for v in layer_losses]

    row = {
        "run_id": run_id,
        "dataset": "shd",
        "method": "ff_v3",
        "epoch": epoch + 1,
        "model": args.model,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "T": args.T,
        "split_by": args.split_by,
        "val_ratio": args.val_ratio,
        "label_mode": args.label_mode,
        "input_normalize": args.input_normalize,
        "current_normalize": args.current_normalize,
        "goodness_mode": args.goodness_mode,
        "ff_loss": args.ff_loss,
        "neg_k": args.neg_k,
        "predict_layers": args.predict_layers,
        "tau": args.tau,
        "v_threshold": args.v_threshold,
        "v_reset": args.v_reset,
        "tau_response": args.tau_response,
        "tau_refractory": args.tau_refractory,
        "input_gain": args.input_gain,
        "lr": lr_now,
        "lr_schedule": args.lr_schedule,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "theta": args.theta,
        "label_scale": args.label_scale,
        "train_acc": train_metrics["acc"],
        "train_macro_precision": train_metrics["macro_precision"],
        "train_macro_recall": train_metrics["macro_recall"],
        "train_macro_f1": train_metrics["macro_f1"],
        "test_acc": eval_metrics["acc"],
        "test_macro_precision": eval_metrics["macro_precision"],
        "test_macro_recall": eval_metrics["macro_recall"],
        "test_macro_f1": eval_metrics["macro_f1"],
        "heldout_test_acc": heldout_metrics["acc"],
        "heldout_test_macro_f1": heldout_metrics["macro_f1"],
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_macro_f1,
        "train_time_sec": train_time,
        "test_time_sec": eval_metrics["time_sec"],
        "epoch_time_sec": time.time() - epoch_start,
        "train_speed": train_samples / max(train_time, 1e-9),
        "test_speed": eval_metrics["samples_per_sec"],
        "train_latency_ms_per_sample": 1000.0 * train_time / max(train_samples, 1),
        "test_latency_ms_per_sample": eval_metrics["latency_ms_per_sample"],
        "train_cpu_memory_mb": bytes_to_mb(max_train_cpu),
        "test_cpu_memory_mb": eval_metrics["cpu_memory_mb"],
        "train_gpu_memory_allocated_mb": train_gpu_a,
        "train_gpu_memory_reserved_mb": train_gpu_r,
        "test_gpu_memory_allocated_mb": eval_metrics["gpu_memory_allocated_mb"],
        "test_gpu_memory_reserved_mb": eval_metrics["gpu_memory_reserved_mb"],
        "test_confusion_path": test_confusion_epoch_path,
        "best_test_confusion_path": best_confusion_path,
        "final_test_confusion_path": final_confusion_path,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }
    for i, v in enumerate(avg_losses):
        row[f"layer_{i}_loss"] = v
    row.update(activity_summary_to_metrics("train", train_metrics["activity_summary"]))
    row.update(activity_summary_to_metrics("test", eval_metrics["activity_summary"]))
    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)

    print("\n" + "=" * 100)
    print(
        f"Epoch {epoch + 1}/{args.epochs} | model={args.model} | T={args.T} | split={args.split_by} | "
        f"label={args.label_mode}:{args.label_scale:g} | norm={args.current_normalize} | "
        f"neg_k={args.neg_k} | lr={lr_now:g}"
    )
    print("-" * 100)
    print(
        f"Train Acc/F1: {train_metrics['acc'] * 100:.2f}% / {train_metrics['macro_f1'] * 100:.2f}% | "
        f"Eval Acc/F1: {eval_metrics['acc'] * 100:.2f}% / {eval_metrics['macro_f1'] * 100:.2f}% | "
        f"Heldout Acc/F1: {heldout_metrics['acc'] * 100:.2f}% / {heldout_metrics['macro_f1'] * 100:.2f}%"
    )
    print(
        f"Best Eval Acc/F1: {best_test_acc * 100:.2f}% / {best_test_macro_f1 * 100:.2f}% | "
        f"Train Speed: {row['train_speed']:.2f} samples/s | Eval Speed: {row['test_speed']:.2f} samples/s"
    )
    print(
        f"Eval Spike Rate: {row['test_global_spike_rate']:.6f} | "
        f"Event/Dense Ratio: {row['test_event_to_dense_ratio']:.6f} | "
        f"Layer losses: " + ", ".join([f"{v:.6f}" for v in avg_losses])
    )
    print("=" * 100)

if last_eval_metrics is not None:
    save_confusion_matrix(final_confusion_path, last_eval_metrics["confusion_matrix"])
    save_model_artifact(
        final_model_path,
        net,
        {
            "epoch": args.epochs,
            "final_eval_acc": last_eval_metrics["acc"],
            "final_eval_macro_f1": last_eval_metrics["macro_f1"],
            "final_heldout_test_acc": best_heldout["acc"] if best_heldout else None,
            "final_heldout_test_macro_f1": best_heldout["macro_f1"] if best_heldout else None,
            "best_eval_acc": best_test_acc,
            "best_eval_macro_f1": best_test_macro_f1,
        },
    )
    save_json(
        summary_path,
        {
            "run_id": run_id,
            "dataset": "shd",
            "method": "ff_v3",
            "csv_path": csv_path,
            "args_path": args_path,
            "best_model_path": best_model_path,
            "final_model_path": final_model_path,
            "best_eval_acc": best_test_acc,
            "best_eval_macro_f1": best_test_macro_f1,
            "final_heldout_test_acc": best_heldout["acc"] if best_heldout else None,
            "final_heldout_test_macro_f1": best_heldout["macro_f1"] if best_heldout else None,
            "important_note": "Eval metrics use validation split when --val-ratio > 0; heldout_test is logged separately.",
        },
    )
print("Done. CSV saved to:", csv_path)
