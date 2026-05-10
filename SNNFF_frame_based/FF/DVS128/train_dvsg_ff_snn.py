#!/usr/bin/env python3
"""
Train an 8-layer MLP SNN on DVS128 Gesture with a Forward-Forward (FF) style
local layer-wise objective.

This script is adapted from:
  1) SpikingJelly DVS128 BP example: dataset loading / frame shape conventions.
  2) NMNIST FF-SNN script: label overlay, positive/negative examples,
     layer-wise goodness, local layer optimizers, candidate-label inference.

Default network:
  input  = 2 * 128 * 128 = 32768 flattened event frame
  first four transitions = 32768 -> 10000 -> 5000 -> 1000 -> --hidden-dim
  remaining transitions  = --hidden-dim -> --hidden-dim until --depth layers

Important:
  FF inference tests all class labels. For DVS128 Gesture this is 11 forward
  candidate passes per batch, so evaluation is much slower than standard BP.
"""

import argparse
import csv
import datetime
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tbf(x: torch.Tensor) -> torch.Tensor:
    """Convert DVS frames from [B, T, C, H, W] to [T, B, F]."""
    if x.dim() != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got shape {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    return x.permute(1, 0, 2, 3, 4).contiguous().view(t, b, c * h * w)


@torch.no_grad()
def overlay_y_on_x_dvsg(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_classes: int = 11,
    label_scale: float = 1.0,
    channel: int = 0,
    row: int = 0,
) -> torch.Tensor:
    """
    Overlay a label cue into the first row of one polarity channel.

    x: [B, T, 2, 128, 128]
    y: [B]

    The first num_classes pixels of x[:, :, channel, row, :] are cleared, then
    the pixel corresponding to the candidate label is set to per-sample vmax.
    This follows the FF label-overlay trick used in the NMNIST reference, but
    uses num_classes=11 for DVS128 Gesture.
    """
    if x.dim() != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got shape {tuple(x.shape)}")
    if x.size(2) <= channel:
        raise ValueError(f"Input has {x.size(2)} channels, but channel={channel}")
    if x.size(-1) < num_classes:
        raise ValueError(f"Input width {x.size(-1)} is smaller than num_classes={num_classes}")

    out = x.clone()
    y = y.to(out.device, dtype=torch.long).view(-1).clamp(0, num_classes - 1)
    b = out.size(0)
    idx = torch.arange(b, device=out.device)

    out[:, :, channel, row, :num_classes] = 0.0
    vmax = out.abs().amax(dim=(1, 2, 3, 4)) + 1e-6
    out[idx, :, channel, row, y] = vmax[idx].view(b, 1) * float(label_scale)
    return out


@torch.no_grad()
def make_negative_labels_random(y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    offset = torch.randint(1, num_classes, y_true.shape, device=y_true.device)
    return (y_true + offset) % num_classes


def confusion_matrix_from_preds(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1).cpu(), preds.view(-1).cpu()):
        cm[t.long(), p.long()] += 1
    return cm


def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    f1s: List[float] = []
    for k in range(cm.size(0)):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(sum(f1s) / max(len(f1s), 1))


def append_row(csv_path: str, fieldnames: Sequence[str], row: Dict) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def maybe_float(x: Optional[str]) -> Optional[float]:
    return None if x is None else float(x)


# ============================================================
# Forward-Forward SNN layers
# ============================================================


class FFSpikingLayer(nn.Module):
    """
    One locally trained FF layer: Linear -> spiking neuron.

    The optimizer is local to this layer. During train_step(), only this layer's
    parameters are updated from its positive/negative goodness loss.
    """

    def __init__(
        self,
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
        neuron_kwargs: Optional[dict] = None,
        train_neuron_params: bool = False,
    ):
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

        # Some custom neuron implementations need num_neurons; fill it if the
        # user passed num_neurons=None.
        if "num_neurons" in neuron_kwargs and neuron_kwargs["num_neurons"] is None:
            neuron_kwargs["num_neurons"] = out_features

        self.neuron = neuron_factory(**neuron_kwargs)
        if hasattr(self.neuron, "step_mode"):
            self.neuron.step_mode = "m"
        if hasattr(self.neuron, "store_v_seq"):
            self.neuron.store_v_seq = True

        params = list(self.fc.parameters())
        if train_neuron_params:
            params += list(self.neuron.parameters())
        self.opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _activity(pre: torch.Tensor, v_seq: Optional[torch.Tensor]) -> torch.Tensor:
        # Prefer membrane trace when available; otherwise use pre-activation.
        return v_seq if v_seq is not None else pre

    def run(self, x_tbf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        pre = self.fc(x_tbf) * self.pre_gain
        spk = self.neuron(pre)
        v_seq = getattr(self.neuron, "v_seq", None)
        act = self._activity(pre, v_seq)
        return pre, spk, v_seq, act

    def goodness(self, act: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """Return per-sample goodness [B]."""
        if self.goodness_mode == "activity_l2":
            return (act ** 2).mean(dim=(0, 2))
        if self.goodness_mode == "activity_l1":
            return act.abs().mean(dim=(0, 2))
        if self.goodness_mode == "spike_count":
            return spk.mean(dim=(0, 2))
        raise ValueError(f"Unknown goodness_mode={self.goodness_mode}")

    def prefix_goodness(self, act: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """Return prefix goodness [T, B], useful for latency analysis."""
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
            return F.softplus(self.theta - g_pos).mean() + F.softplus(g_neg - self.theta).mean()
        if self.loss_mode == "margin":
            return F.relu(self.margin - (g_pos - g_neg)).mean()
        raise ValueError(f"Unknown loss_mode={self.loss_mode}")

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
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

        # Recompute local layer outputs after the update, then detach before
        # sending to the next layer. This is the layer-wise FF training rule.
        with torch.no_grad():
            functional.reset_net(self.neuron)
            _, _, _, act_pos2 = self.run(x_pos)
            h_pos = self._l2_normalize(act_pos2)
            functional.reset_net(self.neuron)
            _, _, _, act_neg2 = self.run(x_neg)
            h_neg = self._l2_normalize(act_neg2)

        return h_pos.detach(), h_neg.detach(), float(loss.detach().item())


class FFSpikingNet(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, ...],
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
        neuron_kwargs: Optional[dict] = None,
        train_neuron_params: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FFSpikingLayer(
                dims[i],
                dims[i + 1],
                neuron_factory,
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
                train_neuron_params=train_neuron_params,
            )
            for i in range(len(dims) - 1)
        ])
        self.dims = dims

    def train_ff(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> List[float]:
        h_pos, h_neg = x_pos, x_neg
        losses: List[float] = []
        for layer in self.layers:
            h_pos, h_neg, loss_value = layer.train_step(h_pos, h_neg)
            losses.append(loss_value)
        return losses

    def optimizer_state_dicts(self) -> List[Dict]:
        return [layer.opt.state_dict() for layer in self.layers]

    def load_optimizer_state_dicts(self, states: List[Dict]) -> None:
        for layer, state in zip(self.layers, states):
            layer.opt.load_state_dict(state)


# ============================================================
# Model builder
# ============================================================


def build_neuron(args):
    if args.model == "lif":
        factory = neuron.LIFNode
        kwargs = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            tau=args.tau,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    elif args.model == "plif":
        factory = neuron.ParametricLIFNode
        kwargs = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            init_tau=args.tau,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    elif args.model == "srm":
        if not hasattr(neuron, "SRMNode"):
            raise RuntimeError("spikingjelly.activation_based.neuron.SRMNode is not available in this environment.")
        factory = getattr(neuron, "SRMNode")
        kwargs = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            tau_response=args.tau_response,
            tau_refractory=args.tau_refractory,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    elif args.model == "dynsrm":
        if not hasattr(neuron, "DynamicSRMNode"):
            raise RuntimeError("spikingjelly.activation_based.neuron.DynamicSRMNode is not available in this environment.")
        factory = getattr(neuron, "DynamicSRMNode")
        kwargs = dict(
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            num_neurons=None,  # filled per layer as out_features
            init_tau_response=args.tau_response,
            init_tau_refractory=args.tau_refractory,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    else:
        raise ValueError(args.model)
    return factory, kwargs


def build_model(args, sample_shape: Tuple[int, ...], device: torch.device) -> FFSpikingNet:
    # sample_shape is [T, C, H, W]. For DVS128 Gesture with two polarities,
    # the flattened input dimension should be 2 * 128 * 128 = 32768.
    if len(sample_shape) != 4:
        raise ValueError(f"Expected sample shape [T,C,H,W], got {sample_shape}")
    _, c, h, w = sample_shape
    in_dim = int(c * h * w)

    if in_dim != 32768:
        print(f"Warning: computed input dim is {in_dim}, not 32768. "
              f"The wide-stem architecture will still use the computed input dim.")

    # Wide decreasing stem requested for DVS128:
    #   layer 1: input_dim -> 10000
    #   layer 2: 10000     -> 5000
    #   layer 3: 5000      -> 1000
    #   layer 4: 1000      -> hidden_dim
    # Remaining layers, if any, are hidden_dim -> hidden_dim.
    depth = int(args.depth)
    if depth < 4:
        raise ValueError("This DVS128 wide-stem architecture requires --depth >= 4, "
                         "because the first four layers are input->10000->5000->1000->hidden_dim.")

    hidden_dim = int(args.hidden_dim)
    dims = [in_dim, 10000, 5000, 1000, hidden_dim] + [hidden_dim] * (depth - 4)
    factory, kwargs = build_neuron(args)

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
        neuron_kwargs=kwargs,
        train_neuron_params=args.train_neuron_params,
    ).to(device)
    return net

# ============================================================
# FF inference / examples / metrics
# ============================================================


@dataclass
class EvalStats:
    cm: torch.Tensor
    acc: float
    macro_f1: float
    mean_total_spikes: float = float("nan")
    mean_spike_rate: float = float("nan")
    mean_event_synops_est: float = float("nan")
    mean_input_events: float = float("nan")
    latency_mean_all: float = float("nan")
    latency_mean_correct: float = float("nan")
    latency_std_correct: float = float("nan")


@torch.no_grad()
def evaluate_candidates(
    net: FFSpikingNet,
    x_btc_hw: torch.Tensor,
    *,
    label_scale: float,
    num_classes: int,
    collect_stats: bool,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Evaluate all candidate labels. Return total goodness G with shape [B,C].
    Predicted class = argmax_c G[:, c].
    """
    device = x_btc_hw.device
    b = x_btc_hw.size(0)
    t_steps = x_btc_hw.size(1)
    G = torch.zeros(b, num_classes, device=device)

    if collect_stats:
        prefix_G = torch.zeros(b, num_classes, t_steps, device=device)
        total_spikes = torch.zeros(b, device=device)
        total_synops = torch.zeros(b, device=device)
        input_events_acc = torch.zeros(b, device=device)
    else:
        prefix_G = None
        total_spikes = total_synops = input_events_acc = None

    for lab in range(num_classes):
        labels = torch.full((b,), lab, device=device, dtype=torch.long)
        x_lab = overlay_y_on_x_dvsg(
            x_btc_hw,
            labels,
            num_classes=num_classes,
            label_scale=label_scale,
        )
        h = to_tbf(x_lab)

        if collect_stats:
            input_events = (x_lab != 0).sum(dim=(1, 2, 3, 4)).float()
            input_events_acc += input_events
            total_synops += input_events * net.layers[0].out_features

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
    for tt in range(t_steps - 2, -1, -1):
        stable[:, tt] = (pred_prefix[:, tt] == final_pred) & stable[:, tt + 1]

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
        "input_events": input_events_acc / float(num_classes),
    }
    return G, stats


@torch.no_grad()
def goodness_per_class(
    net: FFSpikingNet,
    x_btc_hw: torch.Tensor,
    *,
    label_scale: float,
    num_classes: int,
) -> torch.Tensor:
    G, _ = evaluate_candidates(
        net,
        x_btc_hw,
        label_scale=label_scale,
        num_classes=num_classes,
        collect_stats=False,
    )
    return G


@torch.no_grad()
def predict_ff(
    net: FFSpikingNet,
    x_btc_hw: torch.Tensor,
    *,
    label_scale: float,
    num_classes: int,
) -> torch.Tensor:
    return goodness_per_class(net, x_btc_hw, label_scale=label_scale, num_classes=num_classes).argmax(dim=1)


@torch.no_grad()
def make_examples(
    net: FFSpikingNet,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    label_scale: float,
    neg_mode: str,
    num_classes: int,
    hard_neg_epsilon: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if neg_mode == "random":
        y_neg = make_negative_labels_random(y_true, num_classes=num_classes)
    elif neg_mode == "hard":
        g = goodness_per_class(net, x, label_scale=label_scale, num_classes=num_classes)
        weights = torch.sqrt(g.clamp_min(0.0)) + hard_neg_epsilon
        weights[torch.arange(x.size(0), device=x.device), y_true] = 0.0
        bad = weights.sum(dim=1) <= 0
        if bad.any():
            weights[bad] = 1.0
            weights[bad, y_true[bad]] = 0.0
        y_neg = torch.multinomial(weights, 1).squeeze(1)
    else:
        raise ValueError(f"Unknown neg_mode={neg_mode}")

    x_pos = overlay_y_on_x_dvsg(x, y_true, num_classes=num_classes, label_scale=label_scale)
    x_neg = overlay_y_on_x_dvsg(x, y_neg, num_classes=num_classes, label_scale=label_scale)
    return x_pos, x_neg


def _cat_mean(xs: List[torch.Tensor]) -> float:
    return float(torch.cat(xs).float().mean().item()) if xs else float("nan")


def _cat_std(xs: List[torch.Tensor]) -> float:
    return float(torch.cat(xs).float().std(unbiased=False).item()) if xs else float("nan")


@torch.no_grad()
def eval_loader(
    net: FFSpikingNet,
    loader: DataLoader,
    *,
    device: torch.device,
    label_scale: float,
    num_classes: int,
    collect_stats: bool,
    input_scale: float,
) -> EvalStats:
    net.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    spikes_all: List[torch.Tensor] = []
    spike_rates_all: List[torch.Tensor] = []
    synops_all: List[torch.Tensor] = []
    input_events_all: List[torch.Tensor] = []
    lat_all: List[torch.Tensor] = []
    lat_correct: List[torch.Tensor] = []

    for x, y in loader:
        x = x.float().to(device) * float(input_scale)
        y = y.to(device, dtype=torch.long)
        G, stats = evaluate_candidates(
            net,
            x,
            label_scale=label_scale,
            num_classes=num_classes,
            collect_stats=collect_stats,
        )
        preds = G.argmax(dim=1)
        cm += confusion_matrix_from_preds(preds, y, num_classes)

        if collect_stats and stats is not None:
            spikes_all.append(stats["total_spikes"].cpu())
            spike_rates_all.append(stats["spike_rate"].cpu())
            synops_all.append(stats["event_synops_est"].cpu())
            input_events_all.append(stats["input_events"].cpu())
            lat_all.append(stats["latency"].cpu())
            correct_mask = preds.eq(y)
            if correct_mask.any():
                lat_correct.append(stats["latency"][correct_mask].cpu())

    acc = float(cm.diag().sum().item() / max(cm.sum().item(), 1))
    macro_f1 = macro_f1_from_confusion(cm)
    out = EvalStats(cm=cm, acc=acc, macro_f1=macro_f1)
    if collect_stats:
        out.mean_total_spikes = _cat_mean(spikes_all)
        out.mean_spike_rate = _cat_mean(spike_rates_all)
        out.mean_event_synops_est = _cat_mean(synops_all)
        out.mean_input_events = _cat_mean(input_events_all)
        out.latency_mean_all = _cat_mean(lat_all)
        out.latency_mean_correct = _cat_mean(lat_correct)
        out.latency_std_correct = _cat_std(lat_correct)
    net.train()
    return out


# ============================================================
# CLI / main
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description="DVS128 Gesture FF-SNN with configurable-depth MLP")

    # DVS / runtime
    p.add_argument("-device", default="cuda:0")
    p.add_argument("-T", type=int, default=16, help="simulation time-steps / DVS frames")
    p.add_argument("-b", type=int, default=256, help="training batch size")
    p.add_argument("--eval-b", type=int, default=0, help="evaluation batch size; 0 means same as -b")
    p.add_argument("-epochs", type=int, default=64)
    p.add_argument("-j", type=int, default=4, help="DataLoader workers")
    p.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/DVSGesture")
    p.add_argument("-out-dir", type=str, default="./result")
    p.add_argument("--resume", type=str, default="", help="checkpoint path")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--input-scale", type=float, default=1.0, help="multiply input frames by this value")
    p.add_argument("--num-classes", type=int, default=11, help="DVS128 Gesture has 11 classes")

    # FF optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--label-scale", type=float, default=1.0)
    p.add_argument("--pre-gain", type=float, default=5.0)
    p.add_argument("--normalize-input", action="store_true")
    p.add_argument("--goodness-mode", type=str, default="activity_l2", choices=["activity_l2", "activity_l1", "spike_count"])
    p.add_argument("--loss-mode", type=str, default="swish", choices=["swish", "threshold", "margin"])
    p.add_argument("--swish-alpha", type=float, default=6.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--neg-mode", type=str, default="hard", choices=["random", "hard"])
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--train-eval", action="store_true", help="also evaluate the whole train loader each eval epoch")

    # MLP-SNN architecture
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=8, help="number of FF Linear+spiking layers; must be >=4 for the DVS128 wide stem")
    p.add_argument("--model", type=str, default="lif", choices=["lif", "plif", "srm", "dynsrm"])
    p.add_argument("--train-neuron-params", action="store_true", help="include trainable neuron params in each local optimizer")
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--v-threshold", type=float, default=0.5)
    p.add_argument("--v-reset", type=float, default=0.0)
    p.add_argument("--tau-response", type=float, default=2.0)
    p.add_argument("--tau-refractory", type=float, default=10.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    run_name = (
        f"DVS128_FF_T{args.T}_b{args.b}_depth{args.depth}_hid{args.hidden_dim}_"
        f"{args.model}_{args.goodness_mode}_{args.loss_mode}_neg{args.neg_mode}_lr{args.lr}"
    )
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as f:
        f.write(str(args) + "\n")
        f.write(" ".join(sys.argv) + "\n")

    train_set = DVS128Gesture(
        root=args.data_dir,
        train=True,
        data_type="frame",
        frames_number=args.T,
        split_by="number",
    )
    test_set = DVS128Gesture(
        root=args.data_dir,
        train=False,
        data_type="frame",
        frames_number=args.T,
        split_by="number",
    )

    eval_b = args.eval_b if args.eval_b and args.eval_b > 0 else args.b
    train_loader = DataLoader(
        train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        train_set,
        batch_size=eval_b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True,
    )

    sample_frame, sample_label = train_set[0]
    sample_shape = tuple(sample_frame.shape)  # [T, C, H, W]
    if len(sample_shape) != 4 or sample_shape[0] != args.T:
        raise RuntimeError(f"Unexpected sample frame shape: {sample_shape}; expected [T,C,H,W] with T={args.T}")
    if sample_shape[-1] < args.num_classes:
        raise RuntimeError(f"Input width {sample_shape[-1]} < num_classes {args.num_classes}; label overlay cannot fit.")

    net = build_model(args, sample_shape, device)
    print(args)
    print(f"Output directory: {out_dir}")
    print(f"Sample frame shape: {sample_shape}, sample label: {sample_label}")
    print(f"Network dims: {net.dims}")
    print(net)

    writer = SummaryWriter(out_dir)
    csv_path = os.path.join(out_dir, "metrics.csv")
    fieldnames = [
        "epoch",
        "train_loss_mean",
        "train_acc",
        "train_macro_f1",
        "test_acc",
        "test_macro_f1",
        "best_test_acc",
        "mean_total_spikes",
        "mean_spike_rate",
        "mean_event_synops_est",
        "mean_input_events",
        "latency_mean_all",
        "latency_mean_correct",
        "latency_std_correct",
        "train_speed_samples_per_s",
        "eval_speed_samples_per_s",
    ] + [f"layer_{i}_loss" for i in range(len(net.layers))]

    start_epoch = 1
    best_test_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(checkpoint["model"])
        if "layer_optimizer_states" in checkpoint:
            net.load_optimizer_state_dicts(checkpoint["layer_optimizer_states"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_test_acc = float(checkpoint.get("best_test_acc", 0.0))
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, best_test_acc={best_test_acc:.4f}")

    best_ckpt_path = os.path.join(out_dir, "checkpoint_best.pth")
    latest_ckpt_path = os.path.join(out_dir, "checkpoint_latest.pth")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        net.train()
        layer_losses_accum: List[List[float]] = [[] for _ in range(len(net.layers))]
        train_samples = 0

        for x, y in train_loader:
            x = x.float().to(device) * float(args.input_scale)
            y = y.to(device, dtype=torch.long)

            x_pos, x_neg = make_examples(
                net,
                x,
                y,
                label_scale=args.label_scale,
                neg_mode=args.neg_mode,
                num_classes=args.num_classes,
            )
            losses = net.train_ff(to_tbf(x_pos), to_tbf(x_neg))
            for i, lv in enumerate(losses):
                layer_losses_accum[i].append(lv)
            train_samples += y.numel()

        train_elapsed = time.time() - epoch_start
        train_speed = train_samples / max(train_elapsed, 1e-9)
        avg_losses = [float(sum(v) / max(len(v), 1)) for v in layer_losses_accum]
        train_loss_mean = float(sum(avg_losses) / max(len(avg_losses), 1))

        row = {
            "epoch": epoch,
            "train_loss_mean": train_loss_mean,
            "train_acc": "",
            "train_macro_f1": "",
            "test_acc": "",
            "test_macro_f1": "",
            "best_test_acc": best_test_acc,
            "mean_total_spikes": "",
            "mean_spike_rate": "",
            "mean_event_synops_est": "",
            "mean_input_events": "",
            "latency_mean_all": "",
            "latency_mean_correct": "",
            "latency_std_correct": "",
            "train_speed_samples_per_s": train_speed,
            "eval_speed_samples_per_s": "",
        }
        for i, lv in enumerate(avg_losses):
            row[f"layer_{i}_loss"] = lv

        writer.add_scalar("train/loss_mean", train_loss_mean, epoch)
        for i, lv in enumerate(avg_losses):
            writer.add_scalar(f"train/layer_{i}_loss", lv, epoch)
        writer.add_scalar("speed/train_samples_per_s", train_speed, epoch)

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            eval_start = time.time()
            train_stats = None
            if args.train_eval:
                train_stats = eval_loader(
                    net,
                    train_eval_loader,
                    device=device,
                    label_scale=args.label_scale,
                    num_classes=args.num_classes,
                    collect_stats=False,
                    input_scale=args.input_scale,
                )
            test_stats = eval_loader(
                net,
                test_loader,
                device=device,
                label_scale=args.label_scale,
                num_classes=args.num_classes,
                collect_stats=True,
                input_scale=args.input_scale,
            )
            eval_elapsed = time.time() - eval_start
            eval_speed = (len(test_loader.dataset) + (len(train_eval_loader.dataset) if args.train_eval else 0)) / max(eval_elapsed, 1e-9)

            best_before = best_test_acc
            best_test_acc = max(best_test_acc, test_stats.acc)
            save_best = test_stats.acc >= best_before

            row.update({
                "train_acc": train_stats.acc if train_stats is not None else "",
                "train_macro_f1": train_stats.macro_f1 if train_stats is not None else "",
                "test_acc": test_stats.acc,
                "test_macro_f1": test_stats.macro_f1,
                "best_test_acc": best_test_acc,
                "mean_total_spikes": test_stats.mean_total_spikes,
                "mean_spike_rate": test_stats.mean_spike_rate,
                "mean_event_synops_est": test_stats.mean_event_synops_est,
                "mean_input_events": test_stats.mean_input_events,
                "latency_mean_all": test_stats.latency_mean_all,
                "latency_mean_correct": test_stats.latency_mean_correct,
                "latency_std_correct": test_stats.latency_std_correct,
                "eval_speed_samples_per_s": eval_speed,
            })

            if train_stats is not None:
                writer.add_scalar("train/acc", train_stats.acc, epoch)
                writer.add_scalar("train/macro_f1", train_stats.macro_f1, epoch)
            writer.add_scalar("test/acc", test_stats.acc, epoch)
            writer.add_scalar("test/macro_f1", test_stats.macro_f1, epoch)
            writer.add_scalar("test/best_acc", best_test_acc, epoch)
            writer.add_scalar("test/mean_total_spikes", test_stats.mean_total_spikes, epoch)
            writer.add_scalar("test/mean_spike_rate", test_stats.mean_spike_rate, epoch)
            writer.add_scalar("test/mean_event_synops_est", test_stats.mean_event_synops_est, epoch)
            writer.add_scalar("test/latency_mean_correct", test_stats.latency_mean_correct, epoch)
            writer.add_scalar("speed/eval_samples_per_s", eval_speed, epoch)

            if save_best:
                torch.save({
                    "epoch": epoch,
                    "model": net.state_dict(),
                    "layer_optimizer_states": net.optimizer_state_dicts(),
                    "args": vars(args),
                    "best_test_acc": best_test_acc,
                    "row": row,
                }, best_ckpt_path)

            print("\n" + "=" * 110)
            print(f"Epoch {epoch}/{args.epochs} | model={args.model} | depth={args.depth} | hidden={args.hidden_dim} | neg={args.neg_mode}")
            print("-" * 110)
            if train_stats is not None:
                print(f"Train Acc: {train_stats.acc * 100:.2f}% | Train Macro-F1: {train_stats.macro_f1:.4f}")
            print(f"Test  Acc: {test_stats.acc * 100:.2f}% | Test  Macro-F1: {test_stats.macro_f1:.4f} | Best: {best_test_acc * 100:.2f}%")
            print(f"Mean total spikes: {test_stats.mean_total_spikes:.3f} | Mean spike rate: {test_stats.mean_spike_rate:.6f}")
            print(f"Mean event SynOps est.: {test_stats.mean_event_synops_est:.3f} | Mean input events: {test_stats.mean_input_events:.3f}")
            print(f"Latency mean(all): {test_stats.latency_mean_all:.3f} | correct: {test_stats.latency_mean_correct:.3f} ± {test_stats.latency_std_correct:.3f}")
            print(f"Train speed: {train_speed:.1f} samples/s | Eval speed: {eval_speed:.1f} samples/s")
            print("Layer losses: " + "  ".join(f"L{i}={lv:.5f}" for i, lv in enumerate(avg_losses)))
            print(f"CSV: {csv_path}")
            print(f"Best ckpt: {best_ckpt_path}")
            eta = datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - epoch_start) * max(args.epochs - epoch, 0))
            print(f"Estimated finish: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 110)
        else:
            print(f"Epoch {epoch}/{args.epochs} | train only | mean loss={train_loss_mean:.5f} | "
                  f"losses=" + " ".join(f"L{i}:{lv:.5f}" for i, lv in enumerate(avg_losses)))

        append_row(csv_path, fieldnames, row)
        torch.save({
            "epoch": epoch,
            "model": net.state_dict(),
            "layer_optimizer_states": net.optimizer_state_dicts(),
            "args": vars(args),
            "best_test_acc": best_test_acc,
            "row": row,
        }, latest_ckpt_path)

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
