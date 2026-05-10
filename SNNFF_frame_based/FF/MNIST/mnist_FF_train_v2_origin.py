import os
import time
import argparse
import numpy as np
import pandas as pd
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, neuron, surrogate


# -------------------------------------------------
# Args
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Improved MNIST FF-SNN (SpikingJelly)")

parser.add_argument("-device", default="cuda:0")
parser.add_argument("-data-dir", type=str, default="/home/public03/yhxu/spikingjelly/dataset/MNIST")
parser.add_argument("-out-dir", type=str, default="./result")
parser.add_argument("-b", type=int, default=2048, help="batch size")
parser.add_argument("-epochs", type=int, default=300)
parser.add_argument("-j", type=int, default=4, help="num workers")
parser.add_argument("-T", type=int, default=10, help="number of repeated static steps")
parser.add_argument("--seed", type=int, default=2026)

# model
parser.add_argument("--model", type=str, default="lif", choices=["lif", "alif", "srm", "dynsrm"])
parser.add_argument("--hidden-dim", type=int, default=500)
parser.add_argument("--num-layers", type=int, default=2, choices=[1, 2, 3])
parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v-threshold", type=float, default=0.5)
parser.add_argument("--v-reset", type=float, default=0.0)
parser.add_argument("--tau-response", type=float, default=2.0)
parser.add_argument("--tau-refractory", type=float, default=2.0)
parser.add_argument("--input-gain", type=float, default=1.0)

# ff
parser.add_argument("--lr", type=float, default=8e-4)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=6.0, help="ranking strength in swish-style FF loss")
parser.add_argument("--label-scale", type=float, default=1.0)
parser.add_argument("--eval-subset", type=int, default=0,
                    help="if >0, only use this many training samples for train-acc evaluation")

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


# -------------------------------------------------
# Helpers
# -------------------------------------------------
@torch.no_grad()
def superimpose_label(x: torch.Tensor, y: Union[torch.Tensor, int], num_classes: int = 10, label_scale: float = 1.0):
    """
    x: [B, 784] in [0,1]
    y: [B] or int
    overwrite first 10 pixels with one-hot label cue
    """
    x = x.clone()
    B = x.size(0)

    if isinstance(y, int):
        y = torch.full((B,), y, device=x.device, dtype=torch.long)
    else:
        y = y.to(x.device, dtype=torch.long).view(-1)

    x[:, :num_classes] = 0.0
    vmax = x.max(dim=1, keepdim=True).values + 1e-6
    x[torch.arange(B, device=x.device), y.clamp(0, num_classes - 1)] = (vmax[:, 0] * float(label_scale))
    return x


@torch.no_grad()
def repeat_static_input(x: torch.Tensor, T: int, gain: float = 1.0):
    """
    Deterministic static encoding, closer to the snntorch reference.
    x: [B, F] -> [T, B, F]
    """
    x = x * float(gain)
    return x.unsqueeze(0).repeat(T, 1, 1)


def goodness(h: torch.Tensor):
    """
    h: [B, F]
    """
    return h.pow(2).mean(dim=1)


def swish_ff_loss(h_pos: torch.Tensor, h_neg: torch.Tensor, alpha: float = 6.0):
    g_pos = goodness(h_pos)
    g_neg = goodness(h_neg)
    delta = g_pos - g_neg
    return F.silu(-alpha * delta).mean()


# -------------------------------------------------
# Model
# -------------------------------------------------
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
            tau=args.tau,
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
            tau_response=args.tau_response,
            tau_refractory=args.tau_refractory,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
        )
    return node_cls, node_kwargs


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
        """
        x_seq: [T, B, F]
        returns
          spk_seq: [T, B, H]
          count:   [B, H]
        """
        functional.reset_net(self)
        cur_seq = self.fc(x_seq)
        spk_seq = self.neuron(cur_seq)
        count = spk_seq.sum(dim=0)
        return spk_seq, count

    def train_ff(self, x_pos_seq: torch.Tensor, x_neg_seq: torch.Tensor):
        self.train()
        spk_pos_seq, h_pos = self.run(x_pos_seq)
        spk_neg_seq, h_neg = self.run(x_neg_seq)

        loss = swish_ff_loss(h_pos, h_neg, alpha=args.alpha)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            spk_pos_seq, _ = self.run(x_pos_seq)
            spk_neg_seq, _ = self.run(x_neg_seq)

        return spk_pos_seq.detach(), spk_neg_seq.detach(), float(loss.item())


class FFSpikingNet(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.layers = nn.ModuleList([
            FFSpikingLayer(dims[i], dims[i + 1], lr=args.lr, weight_decay=args.weight_decay)
            for i in range(len(dims) - 1)
        ])

    def train_ff(self, x_pos_seq: torch.Tensor, x_neg_seq: torch.Tensor):
        h_pos, h_neg = x_pos_seq, x_neg_seq
        losses = []
        for layer in self.layers:
            h_pos, h_neg, loss_val = layer.train_ff(h_pos, h_neg)
            losses.append(loss_val)
        return losses

    @torch.no_grad()
    def layer_goodnesses(self, x_flat: torch.Tensor, y_overlay: Union[torch.Tensor, int]):
        x = superimpose_label(x_flat, y_overlay, label_scale=args.label_scale)
        h_seq = repeat_static_input(x, T=args.T, gain=args.input_gain)

        gs = []
        for layer in self.layers:
            spk_seq, h_count = layer.run(h_seq)
            gs.append(goodness(h_count))
            h_seq = spk_seq.detach()
        return gs

    @torch.no_grad()
    def goodness_per_class(self, x_flat: torch.Tensor, num_classes: int = 10):
        all_g = []
        for label in range(num_classes):
            g_label = sum(self.layer_goodnesses(x_flat, label))
            all_g.append(g_label.unsqueeze(1))
        return torch.cat(all_g, dim=1)

    @torch.no_grad()
    def predict(self, x_flat: torch.Tensor):
        return self.goodness_per_class(x_flat).argmax(dim=1)


# -------------------------------------------------
# Hard-negative mining (borrowed from reference idea)
# -------------------------------------------------
@torch.no_grad()
def make_examples(model: FFSpikingNet, x: torch.Tensor, y_true: torch.Tensor, epsilon: float = 1e-12):
    g = model.goodness_per_class(x)
    g[torch.arange(x.size(0), device=x.device), y_true] = 0.0
    y_hard = torch.multinomial(torch.sqrt(g + epsilon), 1).squeeze(1)

    x_pos = superimpose_label(x, y_true, label_scale=args.label_scale)
    x_neg = superimpose_label(x, y_hard, label_scale=args.label_scale)

    x_pos_seq = repeat_static_input(x_pos, T=args.T, gain=args.input_gain)
    x_neg_seq = repeat_static_input(x_neg, T=args.T, gain=args.input_gain)
    return x_pos_seq, x_neg_seq


# -------------------------------------------------
# Build net
# -------------------------------------------------
dims = [784] + [args.hidden_dim] * args.num_layers
net = FFSpikingNet(dims).to(device)
print(net)

csv_path = os.path.join(args.out_dir, f"MNIST_{args.model}_FF_v2.csv")

csv_columns = [
    "epoch", "train_acc", "test_acc", "best_test_acc", "train_speed", "test_speed"
] + [f"layer_{i}_loss" for i in range(len(net.layers))]

if not os.path.exists(csv_path):
    pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
@torch.no_grad()
def evaluate(loader: DataLoader, *, max_samples: int = 0):
    net.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = net.predict(x)
        correct += (pred == y).sum().item()
        total += y.numel()
        if max_samples > 0 and total >= max_samples:
            break

    if max_samples > 0:
        total = min(total, max_samples)
    return correct / max(total, 1)


# -------------------------------------------------
# Training
# -------------------------------------------------
best_test_acc = 0.0

for epoch in range(args.epochs):
    net.train()
    layer_losses_accum = [[] for _ in range(len(net.layers))]
    train_start = time.time()

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_pos_seq, x_neg_seq = make_examples(net, x, y)
        losses = net.train_ff(x_pos_seq, x_neg_seq)

        for i, lv in enumerate(losses):
            layer_losses_accum[i].append(lv)

    train_time = time.time() - train_start
    train_speed = len(train_loader.dataset) / max(train_time, 1e-6)

    eval_train_loader = train_loader if args.eval_subset <= 0 else DataLoader(
        train_dataset,
        batch_size=max(1024, args.b),
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True,
    )

    train_acc = evaluate(eval_train_loader, max_samples=args.eval_subset)

    test_start = time.time()
    test_acc = evaluate(test_loader)
    test_time = time.time() - test_start
    test_speed = len(test_loader.dataset) / max(test_time, 1e-6)

    best_test_acc = max(best_test_acc, test_acc)
    avg_layer_losses = [float(np.mean(v)) if len(v) else 0.0 for v in layer_losses_accum]

    row = {
        "epoch": epoch + 1,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "best_test_acc": best_test_acc,
        "train_speed": train_speed,
        "test_speed": test_speed,
    }
    for i, v in enumerate(avg_layer_losses):
        row[f"layer_{i}_loss"] = v

    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)

    print("\n" + "=" * 90)
    print(
        f"Epoch {epoch + 1}/{args.epochs} | model={args.model} | T={args.T} | "
        f"layers={args.num_layers} | hidden={args.hidden_dim}"
    )
    print("-" * 90)
    print(
        f"Train Acc: {train_acc * 100:.2f}% | Test Acc: {test_acc * 100:.2f}% | "
        f"Best: {best_test_acc * 100:.2f}%"
    )
    print(f"Train Speed: {train_speed:.2f} samples/s | Test Speed: {test_speed:.2f} samples/s")
    print("Layer losses:", ", ".join([f"{v:.6f}" for v in avg_layer_losses]))
    print("=" * 90)

print("Done. CSV saved to:", csv_path)
