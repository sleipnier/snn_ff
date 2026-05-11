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
from torch.cuda import amp
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
    if device_.type == 'cuda':
        torch.cuda.synchronize(device_)


def get_process_memory_bytes():
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return int(rss)
        return int(rss) * 1024
    return 0


def bytes_to_mb(value: int):
    return float(value) / (1024.0 * 1024.0)


def resolve_device(device_name: str) -> torch.device:
    """Allow CPU for debugging, but restrict CUDA training to cuda:0/cuda:1."""
    requested = torch.device(device_name)
    if requested.type != 'cuda':
        return requested

    index = 0 if requested.index is None else requested.index
    if index not in (0, 1):
        raise ValueError(f'Only cuda:0 and cuda:1 are allowed, got {device_name!r}')
    if not torch.cuda.is_available():
        raise RuntimeError(f'{device_name} was requested, but CUDA is not available')
    if index >= torch.cuda.device_count():
        raise RuntimeError(f'{device_name} was requested, but only {torch.cuda.device_count()} CUDA device(s) are visible')
    torch.cuda.set_device(index)
    return torch.device(f'cuda:{index}')


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
# Neuron construction (use library neurons, same style as FF code)
# ============================================================

# ============================================================
# Network
# ============================================================

class LinearSpikingLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, neuron_kind: str, args):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)
        self.out_features = int(out_features)
        self.neuron = self._build_neuron(neuron_kind, out_features, args)

    @staticmethod
    def _build_neuron(neuron_kind: str, out_features: int, args):
        if neuron_kind == 'lif':
            return neuron.LIFNode(
                tau=args.tau,
                decay_input=False,
                v_threshold=args.v_threshold,
                v_reset=args.v_reset,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                step_mode='s',
                backend='torch',
                store_v_seq=False,
            )
        if neuron_kind == 'alif':
            # ParametricLIFNode = learnable tau, keeping threshold/reset fixed
            return neuron.ParametricLIFNode(
                init_tau=args.tau,
                decay_input=False,
                v_threshold=args.v_threshold,
                v_reset=args.v_reset,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                step_mode='s',
                backend='torch',
                store_v_seq=False,
            )
        if neuron_kind == 'srm':
            return neuron.SRMNode(
                tau_response=args.tau_response,
                tau_refractory=args.tau_refractory,
                v_rest=0.0,
                v_threshold=args.v_threshold,
                v_reset=args.v_reset,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                step_mode='s',
                backend='torch',
                store_v_seq=False,
            )
        if neuron_kind == 'dynsrm':
            return neuron.DynamicSRMNode(
                num_neurons=out_features,
                init_tau_response=args.tau_response,
                init_tau_refractory=args.tau_refractory,
                v_rest=0.0,
                v_threshold=args.v_threshold,
                v_reset=args.v_reset,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                step_mode='s',
                backend='torch',
                store_v_seq=False,
            )
        raise ValueError(neuron_kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron(self.linear(x))


class BPSpikingNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, num_classes: int, neuron_kind: str, args):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * max(1, depth) + [num_classes]
        self.layers = nn.ModuleList([
            LinearSpikingLayer(dims[i], dims[i + 1], neuron_kind, args)
            for i in range(len(dims) - 1)
        ])
        self.fanouts = [dims[i + 2] if i + 2 < len(dims) else 0 for i in range(len(dims) - 1)]
        self.num_classes = int(num_classes)
        self.total_spiking_neurons = sum(dims[1:])

    def forward(self, x_tbf: torch.Tensor, collect_spikes: bool = False):
        """
        x_tbf: [T,B,C,H,W] or [T,B,F]
        Returns:
          out_seq: [T,B,num_classes]
          spike_seqs: optional list of [T,B,N_l]
        """
        T = x_tbf.shape[0]
        per_layer = [[] for _ in range(len(self.layers))] if collect_spikes else None
        out_seq = []

        for t in range(T):
            h = x_tbf[t]
            if h.dim() > 2:
                h = h.flatten(1)
            for li, layer_i in enumerate(self.layers):
                h = layer_i(h)
                if collect_spikes:
                    per_layer[li].append(h)
            out_seq.append(h)

        out_seq = torch.stack(out_seq, dim=0)
        if not collect_spikes:
            return out_seq, None
        spike_seqs = [torch.stack(seq, dim=0) for seq in per_layer]
        return out_seq, spike_seqs


# ============================================================
# Metrics
# ============================================================

@dataclass
class BatchStats:
    preds: torch.Tensor
    loss: float
    total_spikes: torch.Tensor
    spike_rate: torch.Tensor
    event_synops_est: torch.Tensor
    dense_synops_est: torch.Tensor
    event_to_dense_ratio: torch.Tensor
    input_events: torch.Tensor
    latency: torch.Tensor


def loss_fn_from_output(out_seq: torch.Tensor, labels: torch.Tensor, criterion: str) -> torch.Tensor:
    logits = out_seq.mean(dim=0)
    if criterion == 'mse':
        y_onehot = F.one_hot(labels, out_seq.size(-1)).float()
        return F.mse_loss(logits, y_onehot)
    if criterion == 'ce':
        return F.cross_entropy(logits, labels)
    raise ValueError(criterion)


@torch.no_grad()
def batch_metrics_from_output(out_seq: torch.Tensor,
                              spike_seqs: List[torch.Tensor],
                              labels: torch.Tensor,
                              fanouts: List[int],
                              dense_synops_per_sample: float) -> BatchStats:
    T, B, C = out_seq.shape
    logits = out_seq.mean(dim=0)
    preds = logits.argmax(dim=1)

    total_spikes = torch.zeros(B, device=out_seq.device)
    synops = torch.zeros(B, device=out_seq.device)
    for li, spk in enumerate(spike_seqs):
        spk_count = spk.sum(dim=(0, 2)).float()
        total_spikes += spk_count
        if fanouts[li] > 0:
            synops += spk_count * fanouts[li]

    total_neuron_time = float(T * sum(spk.size(2) for spk in spike_seqs))
    spike_rate = total_spikes / max(total_neuron_time, 1.0)

    cumulative = out_seq.cumsum(dim=0)
    final_pred = logits.argmax(dim=1)
    prefix_pred = cumulative.argmax(dim=2).transpose(0, 1)  # [B,T]
    emitted = (cumulative.sum(dim=2).transpose(0, 1) > 0)

    stable = torch.zeros_like(prefix_pred, dtype=torch.bool)
    stable[:, -1] = (prefix_pred[:, -1] == final_pred) & emitted[:, -1]
    for t in range(T - 2, -1, -1):
        stable[:, t] = (prefix_pred[:, t] == final_pred) & emitted[:, t] & stable[:, t + 1]

    latency = torch.full((B,), float(T), device=out_seq.device)
    has_stable = stable.any(dim=1)
    if has_stable.any():
        latency[has_stable] = stable[has_stable].float().argmax(dim=1).float() + 1.0

    dense_synops = torch.full((B,), float(dense_synops_per_sample), device=out_seq.device)
    return BatchStats(
        preds=preds,
        loss=0.0,
        total_spikes=total_spikes,
        spike_rate=spike_rate,
        event_synops_est=synops,
        dense_synops_est=dense_synops,
        event_to_dense_ratio=synops / torch.clamp(dense_synops, min=1.0),
        input_events=torch.tensor(0.0, device=out_seq.device),
        latency=latency,
    )


def current_neuron_param_summary(net: BPSpikingNet, model_name: str) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if model_name == 'alif':
        taus = []
        for layer_i in net.layers:
            n = layer_i.neuron
            if hasattr(n, 'w'):
                taus.append((1.0 / torch.sigmoid(n.w.detach())).mean().item())
        if taus:
            summary['tau_mean'] = float(sum(taus) / len(taus))
    elif model_name == 'dynsrm':
        tau_resp = []
        tau_ref = []
        for layer_i in net.layers:
            n = layer_i.neuron
            if hasattr(n, 'w_tau_response') and getattr(n, 'w_tau_response') is not None:
                tau_resp.append((1.0 + torch.exp(-n.w_tau_response.detach())).mean().item())
            if hasattr(n, 'w_tau_refractory') and getattr(n, 'w_tau_refractory') is not None:
                tau_ref.append((1.0 + torch.exp(-n.w_tau_refractory.detach())).mean().item())
        if tau_resp:
            summary['tau_response_mean'] = float(sum(tau_resp) / len(tau_resp))
        if tau_ref:
            summary['tau_refractory_mean'] = float(sum(tau_ref) / len(tau_ref))
    else:
        summary['tau_mean'] = float('nan')
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='NMNIST time-split SNN + Backpropagation (BP)')
    parser.add_argument('-device', default='cuda:0')
    parser.add_argument('-T', type=int, default=10)
    parser.add_argument('-b', type=int, default=128)
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-j', type=int, default=4)
    parser.add_argument('-data-dir', type=str, default='/home/public03/yhxu/spikingjelly/dataset/NMNIST')
    parser.add_argument('-out-dir', type=str, default='./result')
    parser.add_argument('--split-by', type=str, default='time', choices=['time'],
                        help='NMNIST event-to-frame split strategy. BP2 fixes this to equal-duration time bins.')
    parser.add_argument('--model', type=str, default='lif', choices=['lif', 'alif', 'srm', 'dynsrm'])
    parser.add_argument('--hidden-dim', type=int, default=500)
    parser.add_argument('--depth', type=int, default=2, help='number of hidden spiking layers before output')
    parser.add_argument('--criterion', type=str, default='mse', choices=['mse', 'ce'])
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-neuron', type=float, default=5e-4,
                        help='used only for learnable neuron parameters (ALIF/DynSRM)')
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'multistep'])
    parser.add_argument('--milestones', type=str, default='50,75,100,125')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--v-threshold', type=float, default=0.5)
    parser.add_argument('--v-reset', type=float, default=0.0)
    parser.add_argument('--tau-response', type=float, default=2.0)
    parser.add_argument('--tau-refractory', type=float, default=10.0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    print(args)

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_set = NMNIST(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by=args.split_by)
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by=args.split_by)
    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, drop_last=True,
                              num_workers=args.j, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, drop_last=False,
                             num_workers=args.j, pin_memory=True)

    sample_frame, sample_label = train_set[0]
    print(f'Validating NMNIST frame format: {tuple(sample_frame.shape)} label={sample_label} split_by={args.split_by}')
    assert tuple(sample_frame.shape) == (args.T, 2, 34, 34)
    print('NMNIST format OK\n')

    net = BPSpikingNet(input_dim=2 * 34 * 34,
                       hidden_dim=args.hidden_dim,
                       depth=args.depth,
                       num_classes=10,
                       neuron_kind=args.model,
                       args=args).to(device)
    print(net)

    weight_params, bias_params, neuron_params = [], [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if 'linear.weight' in name:
            weight_params.append(p)
        elif 'linear.bias' in name:
            bias_params.append(p)
        else:
            neuron_params.append(p)
    print(f'Parameter groups: weights={len(weight_params)}, bias={len(bias_params)}, neuron={len(neuron_params)}')

    param_groups = [
        {'params': weight_params, 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': bias_params, 'weight_decay': 0.0, 'lr': args.lr},
    ]
    if neuron_params:
        param_groups.append({'params': neuron_params, 'weight_decay': 0.0, 'lr': args.lr_neuron})

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'multistep':
        milestones = [int(x) for x in args.milestones.split(',') if x.strip()]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    else:
        scheduler = None

    scaler = amp.GradScaler(enabled=args.amp)
    dense_synops_per_sample = float(
        args.T * ((2 * 34 * 34) * args.hidden_dim + (max(1, args.depth) - 1) * (args.hidden_dim ** 2) + args.hidden_dim * 10)
    )

    csv_path = os.path.join(args.out_dir, f'NMNIST_BP_{args.model}_results.csv')
    csv_fields = [
        'epoch', 'train_loss', 'test_loss', 'train_acc', 'train_macro_f1', 'test_acc', 'test_macro_f1',
        'best_test_acc', 'mean_total_spikes', 'mean_spike_rate', 'mean_event_synops_est', 'mean_dense_synops_est',
        'mean_event_to_dense_ratio', 'mean_input_events', 'latency_mean_all', 'latency_mean_correct',
        'latency_std_correct', 'train_speed_samples_per_s', 'test_speed_samples_per_s',
        'train_time_sec', 'test_time_sec', 'epoch_time_sec',
        'train_latency_ms_per_sample', 'test_latency_ms_per_sample',
        'train_cpu_memory_mb', 'test_cpu_memory_mb',
        'train_gpu_memory_allocated_mb', 'train_gpu_memory_reserved_mb',
        'test_gpu_memory_allocated_mb', 'test_gpu_memory_reserved_mb',
        'lr', 'tau_mean', 'tau_response_mean', 'tau_refractory_mean'
    ]
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # ------------------------
        # Train
        # ------------------------
        net.train()
        train_loss_sum = 0.0
        train_samples = 0
        train_cm = torch.zeros(10, 10, dtype=torch.long)
        max_train_cpu_memory_bytes = get_process_memory_bytes()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        synchronize_if_needed(device)
        t0 = time.time()

        for frame, label in train_loader:
            frame = frame.to(device).transpose(0, 1).float()   # [T,B,C,H,W]
            label = label.to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=args.amp):
                out_seq, _ = net(frame, collect_spikes=False)
                loss = loss_fn_from_output(out_seq, label, args.criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                logits = out_seq.mean(dim=0)
                preds = logits.argmax(dim=1)
                train_cm += confusion_matrix_from_preds(preds, label, 10)
                train_loss_sum += loss.item() * label.numel()
                train_samples += label.numel()
                max_train_cpu_memory_bytes = max(max_train_cpu_memory_bytes, get_process_memory_bytes())

            functional.reset_net(net)

        synchronize_if_needed(device)
        train_time = time.time() - t0
        train_loss = train_loss_sum / max(train_samples, 1)
        train_acc = train_cm.diag().sum().item() / max(train_cm.sum().item(), 1)
        train_macro_f1 = macro_f1_from_confusion(train_cm)
        train_speed = len(train_loader.dataset) / max(train_time, 1e-9)
        train_latency_ms = 1000.0 * train_time / max(len(train_loader.dataset), 1)
        train_gpu_allocated_mb = 0.0
        train_gpu_reserved_mb = 0.0
        if device.type == 'cuda':
            train_gpu_allocated_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device))
            train_gpu_reserved_mb = bytes_to_mb(torch.cuda.max_memory_reserved(device))

        # ------------------------
        # Test + comparison stats
        # ------------------------
        net.eval()
        test_loss_sum = 0.0
        test_samples = 0
        test_cm = torch.zeros(10, 10, dtype=torch.long)
        spikes_all, spike_rate_all, event_synops_all, dense_synops_all, event_to_dense_ratio_all, input_events_all = [], [], [], [], [], []
        latency_all, correct_latency_all = [], []
        max_test_cpu_memory_bytes = get_process_memory_bytes()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        synchronize_if_needed(device)
        t1 = time.time()

        with torch.no_grad():
            for frame, label in test_loader:
                frame_b = frame.float()
                input_events = (frame_b != 0).sum(dim=(1, 2, 3, 4)).float()
                frame = frame_b.to(device).transpose(0, 1)  # [T,B,C,H,W]
                label = label.to(device, dtype=torch.long)

                out_seq, spike_seqs = net(frame, collect_spikes=True)
                loss = loss_fn_from_output(out_seq, label, args.criterion)
                stats = batch_metrics_from_output(
                    out_seq, spike_seqs, label, net.fanouts, dense_synops_per_sample
                )
                stats.input_events = input_events.to(device)
                stats.event_synops_est += stats.input_events * net.layers[0].out_features
                stats.event_to_dense_ratio = stats.event_synops_est / torch.clamp(stats.dense_synops_est, min=1.0)

                test_cm += confusion_matrix_from_preds(stats.preds, label, 10)
                test_loss_sum += loss.item() * label.numel()
                test_samples += label.numel()

                spikes_all.append(stats.total_spikes.cpu())
                spike_rate_all.append(stats.spike_rate.cpu())
                event_synops_all.append(stats.event_synops_est.cpu())
                dense_synops_all.append(stats.dense_synops_est.cpu())
                event_to_dense_ratio_all.append(stats.event_to_dense_ratio.cpu())
                input_events_all.append(stats.input_events.cpu())
                latency_all.append(stats.latency.cpu())
                if (stats.preds == label).any():
                    correct_latency_all.append(stats.latency[stats.preds == label].cpu())
                max_test_cpu_memory_bytes = max(max_test_cpu_memory_bytes, get_process_memory_bytes())

                functional.reset_net(net)

        synchronize_if_needed(device)
        test_time = time.time() - t1
        test_loss = test_loss_sum / max(test_samples, 1)
        test_acc = test_cm.diag().sum().item() / max(test_cm.sum().item(), 1)
        test_macro_f1 = macro_f1_from_confusion(test_cm)
        test_speed = len(test_loader.dataset) / max(test_time, 1e-9)
        test_latency_ms = 1000.0 * test_time / max(len(test_loader.dataset), 1)
        test_gpu_allocated_mb = 0.0
        test_gpu_reserved_mb = 0.0
        if device.type == 'cuda':
            test_gpu_allocated_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device))
            test_gpu_reserved_mb = bytes_to_mb(torch.cuda.max_memory_reserved(device))
        is_best = test_acc >= best_test_acc
        if is_best:
            best_test_acc = test_acc

        if scheduler is not None:
            scheduler.step()

        def cat_mean(xs: List[torch.Tensor]) -> float:
            return torch.cat(xs).float().mean().item() if xs else float('nan')

        latency_tensor = torch.cat(latency_all).float() if latency_all else torch.empty(0)
        correct_latency_tensor = torch.cat(correct_latency_all).float() if correct_latency_all else torch.empty(0)

        neuron_summary = current_neuron_param_summary(net, args.model)
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'train_macro_f1': train_macro_f1,
            'test_acc': test_acc,
            'test_macro_f1': test_macro_f1,
            'best_test_acc': best_test_acc,
            'mean_total_spikes': cat_mean(spikes_all),
            'mean_spike_rate': cat_mean(spike_rate_all),
            'mean_event_synops_est': cat_mean(event_synops_all),
            'mean_dense_synops_est': cat_mean(dense_synops_all),
            'mean_event_to_dense_ratio': cat_mean(event_to_dense_ratio_all),
            'mean_input_events': cat_mean(input_events_all),
            'latency_mean_all': latency_tensor.mean().item() if latency_tensor.numel() > 0 else float('nan'),
            'latency_mean_correct': correct_latency_tensor.mean().item() if correct_latency_tensor.numel() > 0 else float('nan'),
            'latency_std_correct': correct_latency_tensor.std(unbiased=False).item() if correct_latency_tensor.numel() > 0 else float('nan'),
            'train_speed_samples_per_s': train_speed,
            'test_speed_samples_per_s': test_speed,
            'train_time_sec': train_time,
            'test_time_sec': test_time,
            'epoch_time_sec': time.time() - epoch_start,
            'train_latency_ms_per_sample': train_latency_ms,
            'test_latency_ms_per_sample': test_latency_ms,
            'train_cpu_memory_mb': bytes_to_mb(max_train_cpu_memory_bytes),
            'test_cpu_memory_mb': bytes_to_mb(max_test_cpu_memory_bytes),
            'train_gpu_memory_allocated_mb': train_gpu_allocated_mb,
            'train_gpu_memory_reserved_mb': train_gpu_reserved_mb,
            'test_gpu_memory_allocated_mb': test_gpu_allocated_mb,
            'test_gpu_memory_reserved_mb': test_gpu_reserved_mb,
            'lr': optimizer.param_groups[0]['lr'],
            'tau_mean': neuron_summary.get('tau_mean', float('nan')),
            'tau_response_mean': neuron_summary.get('tau_response_mean', float('nan')),
            'tau_refractory_mean': neuron_summary.get('tau_refractory_mean', float('nan')),
        }

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        ckpt = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_test_acc': best_test_acc,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, f'checkpoint_latest_{args.model}.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(args.out_dir, f'checkpoint_best_{args.model}.pth'))

        print('\n' + '=' * 88)
        print(f'Epoch {epoch}/{args.epochs} | model={args.model} | criterion={args.criterion}')
        print('-' * 88)
        print(f'Train Acc: {train_acc * 100:.2f}% | Train Macro-F1: {train_macro_f1:.4f} | Train Loss: {train_loss:.5f}')
        print(f'Test  Acc: {test_acc * 100:.2f}% | Test  Macro-F1: {test_macro_f1:.4f} | Test  Loss: {test_loss:.5f} | Best: {best_test_acc * 100:.2f}%')
        print(f'Mean total spikes: {row["mean_total_spikes"]:.3f} | Mean spike rate: {row["mean_spike_rate"]:.6f}')
        print(f'Mean event/dense SynOps est.: {row["mean_event_synops_est"]:.3f} / {row["mean_dense_synops_est"]:.3f} '
              f'| Ratio: {row["mean_event_to_dense_ratio"]:.6f}')
        print(f'Mean input events: {row["mean_input_events"]:.3f}')
        print(f'Latency mean(all): {row["latency_mean_all"]:.3f} | Latency mean(correct): {row["latency_mean_correct"]:.3f} +/- {row["latency_std_correct"]:.3f}')
        if args.model == 'alif':
            print(f'ALIF tau mean: {row["tau_mean"]:.4f}')
        if args.model == 'dynsrm':
            print(f'DynSRM tau_response mean: {row["tau_response_mean"]:.4f} | tau_refractory mean: {row["tau_refractory_mean"]:.4f}')
        print(f'Train/Test speed: {train_speed:.1f}/{test_speed:.1f} samples/s | '
              f'Train/Test memory (CPU MB): {row["train_cpu_memory_mb"]:.1f}/{row["test_cpu_memory_mb"]:.1f}')
        print(f'Train/Test GPU alloc MB: {row["train_gpu_memory_allocated_mb"]:.1f}/{row["test_gpu_memory_allocated_mb"]:.1f}')
        print(f'CSV -> {csv_path}')
        print('=' * 88)

    print('Done.')


if __name__ == '__main__':
    main()
