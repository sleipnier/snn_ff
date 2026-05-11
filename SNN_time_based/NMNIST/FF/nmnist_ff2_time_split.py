import os
import csv
import time
import math
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.datasets.n_mnist import NMNIST

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def resolve_device(device_name: str) -> torch.device:
    """Allow CPU for debugging, but restrict CUDA training to cuda:0/cuda:1."""
    requested = torch.device(device_name)
    if requested.type != "cuda":
        return requested

    index = 0 if requested.index is None else requested.index
    if index not in (0, 1):
        raise ValueError(f"Only cuda:0 and cuda:1 are allowed, got {device_name!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{device_name} was requested, but CUDA is not available")
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"{device_name} was requested, but only {torch.cuda.device_count()} CUDA device(s) are visible")
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def to_tbf(x: torch.Tensor) -> torch.Tensor:
    """[B, T, C, H, W] -> [T, B, F]"""
    b, t, c, h, w = x.shape
    return x.permute(1, 0, 2, 3, 4).contiguous().view(t, b, -1)


@torch.no_grad()
def overlay_y_on_x_nmnist(x: torch.Tensor, y: torch.Tensor,
                          num_classes: int = 10,
                          label_scale: float = 1.0) -> torch.Tensor:
    """Overlay label cue into the first row of channel 0, matching FF style."""
    x = x.clone()
    y = y.to(x.device, dtype=torch.long).view(-1)
    b = x.size(0)
    x[:, :, 0, 0, :num_classes] = 0.0
    vmax = x.abs().amax(dim=(1, 2, 3, 4)) + 1e-6
    idx = torch.arange(b, device=x.device)
    y = y.clamp(0, num_classes - 1)
    x[idx, :, 0, 0, y] = vmax[idx].view(b, 1) * float(label_scale)
    return x


@torch.no_grad()
def make_negative_labels_random(y_true: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    offset = torch.randint(1, num_classes, y_true.shape, device=y_true.device)
    return (y_true + offset) % num_classes


def confusion_matrix_from_preds(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1).cpu(), preds.view(-1).cpu()):
        cm[t.long(), p.long()] += 1
    return cm


def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    f1s = []
    for k in range(cm.size(0)):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(sum(f1s) / len(f1s))


# ============================================================
# FF components
# ============================================================

class FFSpikingLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 neuron_factory,
                 *,
                 theta: float = 1.0,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-3,
                 normalize_input: bool = False,
                 pre_gain: float = 5.0,
                 goodness_mode: str = "activity_l2",
                 loss_mode: str = "swish",
                 swish_alpha: float = 6.0,
                 margin: float = 1.0,
                 neuron_kwargs: Optional[dict] = None):
        super().__init__()
        neuron_kwargs = dict(neuron_kwargs or {})
        self.theta = float(theta)
        self.normalize_input = bool(normalize_input)
        self.pre_gain = float(pre_gain)
        self.goodness_mode = goodness_mode
        self.loss_mode = loss_mode
        self.swish_alpha = float(swish_alpha)
        self.margin = float(margin)
        self.out_features = int(out_features)

        self.fc = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)

        self.neuron = neuron_factory(**neuron_kwargs)
        if hasattr(self.neuron, "step_mode"):
            self.neuron.step_mode = "m"
        if hasattr(self.neuron, "store_v_seq"):
            self.neuron.store_v_seq = True

        self.opt = torch.optim.Adam(self.fc.parameters(), lr=lr, weight_decay=weight_decay)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _activity(pre: torch.Tensor, v_seq: Optional[torch.Tensor]) -> torch.Tensor:
        return v_seq if v_seq is not None else pre

    def run(self, x_tbf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        pre = self.fc(x_tbf) * self.pre_gain
        spk = self.neuron(pre)
        v_seq = getattr(self.neuron, "v_seq", None)
        act = self._activity(pre, v_seq)
        return pre, spk, v_seq, act

    def goodness(self, act: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        if self.goodness_mode == "activity_l2":
            return (act ** 2).mean(dim=(0, 2))
        if self.goodness_mode == "activity_l1":
            return act.abs().mean(dim=(0, 2))
        if self.goodness_mode == "spike_count":
            return spk.mean(dim=(0, 2))
        raise ValueError(f"Unknown goodness_mode={self.goodness_mode}")

    def prefix_goodness(self, act: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """Return [T, B] prefix goodness for latency analysis."""
        if self.goodness_mode == "activity_l2":
            per_t = (act ** 2).mean(dim=2)
        elif self.goodness_mode == "activity_l1":
            per_t = act.abs().mean(dim=2)
        elif self.goodness_mode == "spike_count":
            per_t = spk.mean(dim=2)
        else:
            raise ValueError(f"Unknown goodness_mode={self.goodness_mode}")
        denom = torch.arange(1, per_t.size(0) + 1, device=per_t.device, dtype=per_t.dtype).view(-1, 1)
        return per_t.cumsum(dim=0) / denom

    def ff_loss(self, g_pos: torch.Tensor, g_neg: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "swish":
            delta = g_pos - g_neg
            return F.silu(-self.swish_alpha * delta).mean()
        if self.loss_mode == "threshold":
            return (F.softplus(self.theta - g_pos).mean() +
                    F.softplus(g_neg - self.theta).mean())
        if self.loss_mode == "margin":
            return F.relu(self.margin - (g_pos - g_neg)).mean()
        raise ValueError(f"Unknown loss_mode={self.loss_mode}")

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        self.train()
        if self.normalize_input:
            x_pos = self._l2_normalize(x_pos)
            x_neg = self._l2_normalize(x_neg)

        functional.reset_net(self.neuron)
        _, spk_pos, _, act_pos = self.run(x_pos)
        functional.reset_net(self.neuron)
        _, spk_neg, _, act_neg = self.run(x_neg)

        g_pos = self.goodness(act_pos, spk_pos)
        g_neg = self.goodness(act_neg, spk_neg)
        loss = self.ff_loss(g_pos, g_neg)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            functional.reset_net(self.neuron)
            _, _, _, act_pos2 = self.run(x_pos)
            h_pos = self._l2_normalize(act_pos2)
            functional.reset_net(self.neuron)
            _, _, _, act_neg2 = self.run(x_neg)
            h_neg = self._l2_normalize(act_neg2)

        return h_pos.detach(), h_neg.detach(), float(loss.detach())


class FFSpikingNet(nn.Module):
    def __init__(self, dims: Tuple[int, ...], neuron_factory, *, theta=1.0, lr=1e-3,
                 weight_decay=1e-3, normalize_input=False, pre_gain=5.0,
                 goodness_mode="activity_l2", loss_mode="swish", swish_alpha=6.0,
                 margin=1.0, neuron_kwargs=None):
        super().__init__()
        self.layers = nn.ModuleList([
            FFSpikingLayer(
                dims[i], dims[i + 1], neuron_factory,
                theta=(theta[i] if isinstance(theta, (list, tuple)) else theta),
                lr=lr,
                weight_decay=weight_decay,
                normalize_input=(normalize_input if i == 0 else False),
                pre_gain=pre_gain,
                goodness_mode=goodness_mode,
                loss_mode=loss_mode,
                swish_alpha=swish_alpha,
                margin=margin,
                neuron_kwargs=neuron_kwargs,
            )
            for i in range(len(dims) - 1)
        ])
        self.dims = dims

    def train_ff(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> List[float]:
        h_pos, h_neg = x_pos, x_neg
        losses = []
        for layer in self.layers:
            h_pos, h_neg, loss_value = layer.train_step(h_pos, h_neg)
            losses.append(loss_value)
        return losses

    def set_learning_rate(self, lr: float):
        for layer in self.layers:
            for group in layer.opt.param_groups:
                group["lr"] = float(lr)


# ============================================================
# Build neuron factory
# ============================================================

def build_model(args, device: torch.device):
    if args.model == "lif":
        factory = neuron.LIFNode
        nkw = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            tau=args.tau,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    elif args.model == "alif":
        factory = getattr(neuron, "ParametricLIFNode", neuron.LIFNode)
        nkw = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            init_tau=args.tau,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    elif args.model == "srm":
        factory = getattr(neuron, "SRMNode")
        nkw = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            tau_response=args.tau_response,
            tau_refractory=args.tau_refractory,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    elif args.model == "dynsrm":
        factory = getattr(neuron, "DynamicSRMNode")
        nkw = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            num_neurons=None,
            init_tau_response=args.tau_response,
            init_tau_refractory=args.tau_refractory,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    else:
        raise ValueError(args.model)

    in_dim = 2 * 34 * 34
    dims = [in_dim] + [args.hidden_dim] * max(1, args.depth)
    net = FFSpikingNet(
        dims=tuple(dims),
        neuron_factory=factory,
        theta=args.theta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        normalize_input=args.normalize_input,
        pre_gain=args.pre_gain,
        goodness_mode=args.goodness_mode,
        loss_mode=args.loss_mode,
        swish_alpha=args.swish_alpha,
        margin=args.margin,
        neuron_kwargs=nkw,
    ).to(device)
    return net


# ============================================================
# FF evaluation / prediction / metrics
# ============================================================

@dataclass
class EvalStats:
    preds: torch.Tensor
    total_goodness: torch.Tensor
    mean_total_spikes: float
    mean_spike_rate: float
    mean_event_synops_est: float
    mean_dense_synops_est: float
    mean_event_to_dense_ratio: float
    mean_input_events: float
    latency_mean_all: float
    latency_mean_correct: float
    latency_std_correct: float


@torch.no_grad()
def evaluate_candidates(net: FFSpikingNet,
                        x_btc_hw: torch.Tensor,
                        *,
                        label_scale: float,
                        num_classes: int,
                        collect_stats: bool) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Run all label candidates and return G [B,C].
    If collect_stats=True, also return tensors for latency and energy estimates.
    """
    device = x_btc_hw.device
    b = x_btc_hw.size(0)
    t_steps = x_btc_hw.size(1)
    G = torch.zeros(b, num_classes, device=device)

    prefix_G = None
    if collect_stats:
        prefix_G = torch.zeros(b, num_classes, t_steps, device=device)
        total_spikes = torch.zeros(b, device=device)
        total_synops = torch.zeros(b, device=device)
        total_dense_synops = torch.zeros(b, device=device)
        input_events_acc = torch.zeros(b, device=device)
        dense_per_candidate = float(
            t_steps * sum(net.dims[i] * net.dims[i + 1] for i in range(len(net.dims) - 1))
        )

    for lab in range(num_classes):
        labels = torch.full((b,), lab, device=device, dtype=torch.long)
        x_lab = overlay_y_on_x_nmnist(x_btc_hw, labels, num_classes=num_classes, label_scale=label_scale)
        h = to_tbf(x_lab)

        if collect_stats:
            input_events = (x_lab != 0).sum(dim=(1, 2, 3, 4)).float()
            input_events_acc += input_events
            total_synops += input_events * net.layers[0].out_features
            total_dense_synops += dense_per_candidate

        total_g = torch.zeros(b, device=device)
        total_prefix = torch.zeros(t_steps, b, device=device) if collect_stats else None

        for li, layer in enumerate(net.layers):
            if layer.normalize_input:
                h = layer._l2_normalize(h)

            functional.reset_net(layer.neuron)
            _, spk, _, act = layer.run(h)

            g = layer.goodness(act, spk)
            total_g += g
            if collect_stats:
                total_prefix += layer.prefix_goodness(act, spk)
                spike_count = spk.sum(dim=(0, 2)).float()
                total_spikes += spike_count
                if li + 1 < len(net.layers):
                    total_synops += spike_count * net.layers[li + 1].out_features

            h = layer._l2_normalize(act)

        G[:, lab] = total_g
        if collect_stats:
            prefix_G[:, lab, :] = total_prefix.transpose(0, 1)

    if not collect_stats:
        return G, None

    pred_prefix = prefix_G.argmax(dim=1)  # [B,T]
    final_pred = G.argmax(dim=1)          # [B]

    stable = torch.zeros_like(pred_prefix, dtype=torch.bool)
    stable[:, -1] = pred_prefix[:, -1] == final_pred
    for t in range(t_steps - 2, -1, -1):
        stable[:, t] = (pred_prefix[:, t] == final_pred) & stable[:, t + 1]

    latency = torch.full((b,), float(t_steps), device=device)
    has_stable = stable.any(dim=1)
    if has_stable.any():
        latency[has_stable] = stable[has_stable].float().argmax(dim=1).float() + 1.0

    total_neurons = sum(layer.out_features for layer in net.layers)
    denom = float(num_classes * t_steps * max(total_neurons, 1))
    spike_rate = total_spikes / denom

    stats = {
        "latency": latency,
        "total_spikes": total_spikes / float(num_classes),
        "spike_rate": spike_rate,
        "event_synops_est": total_synops / float(num_classes),
        "dense_synops_est": total_dense_synops / float(num_classes),
        "event_to_dense_ratio": total_synops / torch.clamp(total_dense_synops, min=1.0),
        "input_events": input_events_acc / float(num_classes),
    }
    return G, stats


@torch.no_grad()
def predict_ff_nmnist(net: FFSpikingNet, x_btc_hw: torch.Tensor, *, label_scale: float,
                      num_classes: int = 10) -> torch.Tensor:
    G, _ = evaluate_candidates(net, x_btc_hw, label_scale=label_scale, num_classes=num_classes, collect_stats=False)
    return G.argmax(dim=1)


@torch.no_grad()
def goodness_per_class(net: FFSpikingNet, x_btc_hw: torch.Tensor, *, label_scale: float,
                       num_classes: int = 10) -> torch.Tensor:
    G, _ = evaluate_candidates(net, x_btc_hw, label_scale=label_scale, num_classes=num_classes, collect_stats=False)
    return G


@torch.no_grad()
def make_examples(net: FFSpikingNet,
                  x: torch.Tensor,
                  y_true: torch.Tensor,
                  *,
                  label_scale: float,
                  neg_mode: str,
                  num_classes: int = 10,
                  epsilon: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    if neg_mode == "random":
        y_neg = make_negative_labels_random(y_true, num_classes=num_classes)
    elif neg_mode == "hard":
        g = goodness_per_class(net, x, label_scale=label_scale, num_classes=num_classes)
        g[torch.arange(x.size(0), device=x.device), y_true] = 0
        y_neg = torch.multinomial(torch.sqrt(g.clamp_min(0.0)) + epsilon, 1).squeeze(1)
    else:
        raise ValueError(f"Unknown neg_mode={neg_mode}")

    x_pos = overlay_y_on_x_nmnist(x, y_true, num_classes=num_classes, label_scale=label_scale)
    x_neg = overlay_y_on_x_nmnist(x, y_neg, num_classes=num_classes, label_scale=label_scale)
    return x_pos, x_neg


@torch.no_grad()
def eval_loader(net: FFSpikingNet,
                loader: DataLoader,
                *,
                device: torch.device,
                label_scale: float,
                num_classes: int,
                collect_stats: bool) -> EvalStats:
    net.eval()
    all_preds = []
    all_targets = []
    lat_all = []
    lat_correct = []
    spikes_all = []
    spike_rates_all = []
    synops_all = []
    dense_synops_all = []
    event_to_dense_ratio_all = []
    input_events_all = []
    total_goodness = []

    for x, y in loader:
        x = x.float().to(device)
        y = y.to(device, dtype=torch.long)
        G, stats = evaluate_candidates(net, x, label_scale=label_scale, num_classes=num_classes, collect_stats=collect_stats)
        preds = G.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        total_goodness.append(G.max(dim=1).values.cpu())

        if collect_stats and stats is not None:
            latency = stats["latency"]
            lat_all.append(latency.cpu())
            correct_mask = preds.eq(y)
            if correct_mask.any():
                lat_correct.append(latency[correct_mask].cpu())
            spikes_all.append(stats["total_spikes"].cpu())
            spike_rates_all.append(stats["spike_rate"].cpu())
            synops_all.append(stats["event_synops_est"].cpu())
            dense_synops_all.append(stats["dense_synops_est"].cpu())
            event_to_dense_ratio_all.append(stats["event_to_dense_ratio"].cpu())
            input_events_all.append(stats["input_events"].cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    total_goodness = torch.cat(total_goodness)

    def _mean(xs: List[torch.Tensor]) -> float:
        return float(torch.cat(xs).float().mean().item()) if xs else float("nan")

    def _std(xs: List[torch.Tensor]) -> float:
        return float(torch.cat(xs).float().std(unbiased=False).item()) if xs else float("nan")

    out = EvalStats(
        preds=preds,
        total_goodness=total_goodness,
        mean_total_spikes=_mean(spikes_all),
        mean_spike_rate=_mean(spike_rates_all),
        mean_event_synops_est=_mean(synops_all),
        mean_dense_synops_est=_mean(dense_synops_all),
        mean_event_to_dense_ratio=_mean(event_to_dense_ratio_all),
        mean_input_events=_mean(input_events_all),
        latency_mean_all=_mean(lat_all),
        latency_mean_correct=_mean(lat_correct),
        latency_std_correct=_std(lat_correct),
    )
    net.train()
    return out


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="NMNIST time-split FF-SNN training with 4 neurons + metrics")
    p.add_argument("-device", default="cuda:0")
    p.add_argument("-T", type=int, default=10)
    p.add_argument("-b", type=int, default=4096)
    p.add_argument("-epochs", type=int, default=300)
    p.add_argument("-j", type=int, default=4)
    p.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/NMNIST")
    p.add_argument("-out-dir", type=str, default="./result")
    p.add_argument("--split-by", type=str, default="time", choices=["time"],
                   help="NMNIST event-to-frame split strategy. FF2 fixes this to equal-duration time bins.")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--label-scale", type=float, default=1.0)
    p.add_argument("--model", type=str, default="lif", choices=["lif", "alif", "srm", "dynsrm"])
    p.add_argument("--hidden-dim", type=int, default=500)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--v-threshold", type=float, default=0.5)
    p.add_argument("--v-reset", type=float, default=0.0)
    p.add_argument("--tau-response", type=float, default=2.0)
    p.add_argument("--tau-refractory", type=float, default=10.0)
    p.add_argument("--pre-gain", type=float, default=5.0)
    p.add_argument("--normalize-input", action="store_true")
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--goodness-mode", type=str, default="activity_l2",
                   choices=["activity_l2", "activity_l1", "spike_count"])
    p.add_argument("--loss-mode", type=str, default="swish",
                   choices=["swish", "threshold", "margin"])
    p.add_argument("--swish-alpha", type=float, default=6.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--neg-mode", type=str, default="hard", choices=["random", "hard"])
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=10)
    return p.parse_args()


def append_row(csv_path: str, fieldnames: List[str], row: Dict):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def learning_rate_for_epoch(epoch: int, base_lr: float) -> float:
    """
    Piecewise decay schedule following SNNFF_NMNIST.py (0-based epoch):
    [0,49]: base_lr; [50,74]: 5e-4; [75,99]: 1e-4; [100,124]: 5e-5;
    [125,149]: 1e-6; [150, ...]: 1e-7.
    """
    if epoch >= 150:
        return 1e-7
    if epoch >= 125:
        return 1e-6
    if epoch >= 100:
        return 5e-5
    if epoch >= 75:
        return 1e-4
    if epoch >= 50:
        return 5e-4
    return float(base_lr)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_set = NMNIST(root=args.data_dir, train=True, data_type="frame",
                       frames_number=args.T, split_by=args.split_by)
    test_set = NMNIST(root=args.data_dir, train=False, data_type="frame",
                      frames_number=args.T, split_by=args.split_by)

    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True,
                              drop_last=True, num_workers=args.j, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=False,
                             drop_last=False, num_workers=args.j, pin_memory=True)

    sample_frame, sample_label = train_set[0]
    assert tuple(sample_frame.shape) == (args.T, 2, 34, 34), tuple(sample_frame.shape)
    print(args)
    print(f"Sample frame shape: {tuple(sample_frame.shape)}, label={sample_label}, split_by={args.split_by}")

    net = build_model(args, device)
    print(net)

    csv_path = os.path.join(args.out_dir, f"NMNIST_{args.model}_FF_compare.csv")
    fieldnames = [
        "epoch", "train_acc", "train_macro_f1", "test_acc", "test_macro_f1", "best_test_acc",
        "mean_total_spikes", "mean_spike_rate", "mean_event_synops_est", "mean_dense_synops_est",
        "mean_event_to_dense_ratio", "mean_input_events",
        "latency_mean_all", "latency_mean_correct", "latency_std_correct",
        "train_speed_samples_per_s", "test_speed_samples_per_s",
        "train_time_sec", "test_time_sec", "epoch_time_sec",
        "train_latency_ms_per_sample", "test_latency_ms_per_sample",
        "train_cpu_memory_mb", "test_cpu_memory_mb",
        "train_gpu_memory_allocated_mb", "train_gpu_memory_reserved_mb",
        "test_gpu_memory_allocated_mb", "test_gpu_memory_reserved_mb",
        "lr"
    ] + [f"layer_{i}_loss" for i in range(len(net.layers))]

    best_test_acc = 0.0
    best_ckpt_path = os.path.join(args.out_dir, f"NMNIST_{args.model}_FF_best.pth")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = learning_rate_for_epoch(epoch - 1, args.lr)
        net.set_learning_rate(current_lr)
        net.train()
        layer_losses_accum = [[] for _ in range(len(net.layers))]
        max_train_cpu_memory_bytes = get_process_memory_bytes()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize_if_needed(device)
        train_start = time.time()

        for x, y in train_loader:
            x = x.float().to(device)
            y = y.to(device, dtype=torch.long)
            x_pos, x_neg = make_examples(
                net, x, y,
                label_scale=args.label_scale,
                neg_mode=args.neg_mode,
                num_classes=args.num_classes,
            )
            losses = net.train_ff(to_tbf(x_pos), to_tbf(x_neg))
            for i, lv in enumerate(losses):
                layer_losses_accum[i].append(lv)
            max_train_cpu_memory_bytes = max(max_train_cpu_memory_bytes, get_process_memory_bytes())

        synchronize_if_needed(device)
        train_elapsed = time.time() - train_start
        train_speed = len(train_loader.dataset) / max(train_elapsed, 1e-9)
        train_latency_ms = 1000.0 * train_elapsed / max(len(train_loader.dataset), 1)
        train_gpu_allocated_mb = 0.0
        train_gpu_reserved_mb = 0.0
        if device.type == "cuda":
            train_gpu_allocated_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device))
            train_gpu_reserved_mb = bytes_to_mb(torch.cuda.max_memory_reserved(device))
        avg_losses = [float(sum(v) / max(len(v), 1)) for v in layer_losses_accum]

        if epoch % args.eval_every != 0:
            row = {"epoch": epoch, "train_acc": "", "train_macro_f1": "", "test_acc": "", "test_macro_f1": "",
                   "best_test_acc": best_test_acc, "mean_total_spikes": "", "mean_spike_rate": "",
                   "mean_event_synops_est": "", "mean_dense_synops_est": "", "mean_event_to_dense_ratio": "",
                   "mean_input_events": "", "latency_mean_all": "",
                   "latency_mean_correct": "", "latency_std_correct": "", "train_speed_samples_per_s": train_speed,
                   "test_speed_samples_per_s": "", "train_time_sec": train_elapsed, "test_time_sec": "",
                   "epoch_time_sec": time.time() - epoch_start,
                   "train_latency_ms_per_sample": train_latency_ms, "test_latency_ms_per_sample": "",
                   "train_cpu_memory_mb": bytes_to_mb(max_train_cpu_memory_bytes), "test_cpu_memory_mb": "",
                   "train_gpu_memory_allocated_mb": train_gpu_allocated_mb,
                   "train_gpu_memory_reserved_mb": train_gpu_reserved_mb,
                   "test_gpu_memory_allocated_mb": "", "test_gpu_memory_reserved_mb": "",
                   "lr": current_lr}
            for i, lv in enumerate(avg_losses):
                row[f"layer_{i}_loss"] = lv
            append_row(csv_path, fieldnames, row)
            print(f"Epoch {epoch}/{args.epochs} | train only | layer losses: {avg_losses}")
            continue

        # Compute train metrics in a dedicated pass; this keeps targets aligned with
        # predictions while avoiding cached dataset-sized tensors.
        train_cm = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.to(device, dtype=torch.long)
            preds = predict_ff_nmnist(net, x, label_scale=args.label_scale, num_classes=args.num_classes)
            train_cm += confusion_matrix_from_preds(preds, y, args.num_classes)
        train_acc = float(train_cm.diag().sum().item() / max(train_cm.sum().item(), 1))
        train_macro_f1 = macro_f1_from_confusion(train_cm)

        test_start = time.time()
        test_preds = []
        test_targets = []
        stats_accum = {
            "spikes": [], "spike_rate": [], "event_synops": [], "dense_synops": [],
            "event_to_dense_ratio": [], "input_events": [],
            "lat_all": [], "lat_correct": []
        }
        test_cm = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)
        max_test_cpu_memory_bytes = get_process_memory_bytes()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize_if_needed(device)

        net.eval()
        for x, y in test_loader:
            x = x.float().to(device)
            y = y.to(device, dtype=torch.long)
            G, stats = evaluate_candidates(net, x, label_scale=args.label_scale,
                                           num_classes=args.num_classes, collect_stats=True)
            preds = G.argmax(dim=1)
            test_cm += confusion_matrix_from_preds(preds, y, args.num_classes)
            stats_accum["spikes"].append(stats["total_spikes"].cpu())
            stats_accum["spike_rate"].append(stats["spike_rate"].cpu())
            stats_accum["event_synops"].append(stats["event_synops_est"].cpu())
            stats_accum["dense_synops"].append(stats["dense_synops_est"].cpu())
            stats_accum["event_to_dense_ratio"].append(stats["event_to_dense_ratio"].cpu())
            stats_accum["input_events"].append(stats["input_events"].cpu())
            stats_accum["lat_all"].append(stats["latency"].cpu())
            correct_mask = preds.eq(y)
            if correct_mask.any():
                stats_accum["lat_correct"].append(stats["latency"][correct_mask].cpu())
            max_test_cpu_memory_bytes = max(max_test_cpu_memory_bytes, get_process_memory_bytes())
        net.train()
        synchronize_if_needed(device)
        test_elapsed = time.time() - test_start
        test_speed = len(test_loader.dataset) / max(test_elapsed, 1e-9)
        test_latency_ms = 1000.0 * test_elapsed / max(len(test_loader.dataset), 1)
        test_gpu_allocated_mb = 0.0
        test_gpu_reserved_mb = 0.0
        if device.type == "cuda":
            test_gpu_allocated_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device))
            test_gpu_reserved_mb = bytes_to_mb(torch.cuda.max_memory_reserved(device))

        test_acc = float(test_cm.diag().sum().item() / max(test_cm.sum().item(), 1))
        test_macro_f1 = macro_f1_from_confusion(test_cm)
        is_best = test_acc >= best_test_acc
        if is_best:
            best_test_acc = test_acc

        def cat_mean(xs):
            return float(torch.cat(xs).float().mean().item()) if xs else float("nan")

        def cat_std(xs):
            return float(torch.cat(xs).float().std(unbiased=False).item()) if xs else float("nan")

        row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_macro_f1": train_macro_f1,
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "best_test_acc": best_test_acc,
            "mean_total_spikes": cat_mean(stats_accum["spikes"]),
            "mean_spike_rate": cat_mean(stats_accum["spike_rate"]),
            "mean_event_synops_est": cat_mean(stats_accum["event_synops"]),
            "mean_dense_synops_est": cat_mean(stats_accum["dense_synops"]),
            "mean_event_to_dense_ratio": cat_mean(stats_accum["event_to_dense_ratio"]),
            "mean_input_events": cat_mean(stats_accum["input_events"]),
            "latency_mean_all": cat_mean(stats_accum["lat_all"]),
            "latency_mean_correct": cat_mean(stats_accum["lat_correct"]),
            "latency_std_correct": cat_std(stats_accum["lat_correct"]),
            "train_speed_samples_per_s": train_speed,
            "test_speed_samples_per_s": test_speed,
            "train_time_sec": train_elapsed,
            "test_time_sec": test_elapsed,
            "epoch_time_sec": time.time() - epoch_start,
            "train_latency_ms_per_sample": train_latency_ms,
            "test_latency_ms_per_sample": test_latency_ms,
            "train_cpu_memory_mb": bytes_to_mb(max_train_cpu_memory_bytes),
            "test_cpu_memory_mb": bytes_to_mb(max_test_cpu_memory_bytes),
            "train_gpu_memory_allocated_mb": train_gpu_allocated_mb,
            "train_gpu_memory_reserved_mb": train_gpu_reserved_mb,
            "test_gpu_memory_allocated_mb": test_gpu_allocated_mb,
            "test_gpu_memory_reserved_mb": test_gpu_reserved_mb,
            "lr": current_lr,
        }
        for i, lv in enumerate(avg_losses):
            row[f"layer_{i}_loss"] = lv
        append_row(csv_path, fieldnames, row)

        if is_best:
            torch.save({
                "epoch": epoch,
                "model": net.state_dict(),
                "args": vars(args),
                "best_test_acc": best_test_acc,
                "lr": current_lr,
                "row": row,
            }, best_ckpt_path)

        print("\n" + "=" * 96)
        print(f"Epoch {epoch}/{args.epochs} | model={args.model} | neg={args.neg_mode} | loss={args.loss_mode} | goodness={args.goodness_mode} | lr={current_lr:.2e}")
        print("-" * 96)
        print(f"Train Acc: {train_acc * 100:.2f}% | Train Macro-F1: {train_macro_f1:.4f}")
        print(f"Test  Acc: {test_acc * 100:.2f}% | Test  Macro-F1: {test_macro_f1:.4f} | Best: {best_test_acc * 100:.2f}%")
        print(f"Mean total spikes: {row['mean_total_spikes']:.3f} | Mean spike rate: {row['mean_spike_rate']:.6f}")
        print(f"Mean event/dense SynOps est.: {row['mean_event_synops_est']:.3f} / {row['mean_dense_synops_est']:.3f} "
              f"| Ratio: {row['mean_event_to_dense_ratio']:.6f}")
        print(f"Mean input events: {row['mean_input_events']:.3f}")
        print(f"Latency mean(all): {row['latency_mean_all']:.3f} | Latency mean(correct): {row['latency_mean_correct']:.3f} +/- {row['latency_std_correct']:.3f}")
        print(f"Train/Test speed: {train_speed:.1f}/{test_speed:.1f} samples/s | "
              f"Train/Test memory (CPU MB): {row['train_cpu_memory_mb']:.1f}/{row['test_cpu_memory_mb']:.1f}")
        print(f"Train/Test GPU alloc MB: {row['train_gpu_memory_allocated_mb']:.1f}/{row['test_gpu_memory_allocated_mb']:.1f}")
        print("Layer losses: " + "  ".join(f"L{i}={lv:.5f}" for i, lv in enumerate(avg_losses)))
        print(f"CSV: {csv_path}")
        print(f"Best ckpt: {best_ckpt_path}")
        print("=" * 96)

    print("Done.")
