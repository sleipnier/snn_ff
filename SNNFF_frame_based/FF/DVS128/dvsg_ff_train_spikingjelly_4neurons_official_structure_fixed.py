#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

NUM_CLASSES = 11


def update_confusion_matrix(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
    idx = target * NUM_CLASSES + pred
    counts = torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)
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


class SpikeStats:
    def __init__(self, net: nn.Module, spike_cls: type):
        self.counts = 0.0
        self.elems = 0.0
        self.handles = []
        for _, module in net.named_modules():
            if isinstance(module, spike_cls):
                self.handles.append(module.register_forward_hook(self._hook))

    def _hook(self, _m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            d = t.detach().float()
            self.counts += float(d.sum().item())
            self.elems += float(d.numel())

    def close(self):
        for h in self.handles:
            h.remove()


def make_neuron(model: str, tau: float, v_threshold: float, v_reset: float,
                tau_response: float, tau_refractory: float):
    # Only return neuron-SPECIFIC kwargs here.
    # DVSGestureNet already forwards surrogate_function and detach_reset to the
    # spiking neuron constructor, so including them again here causes
    # `got multiple values for keyword argument` errors.
    common = dict(
        v_threshold=v_threshold,
        v_reset=v_reset,
    )
    if model == "lif":
        return neuron.LIFNode, dict(common, tau=tau)
    if model == "alif":
        node = getattr(neuron, "ParametricLIFNode", neuron.LIFNode)
        return node, dict(common, init_tau=tau)
    if model == "srm":
        return neuron.SRMNode, dict(common, tau_response=tau_response, tau_refractory=tau_refractory)
    node = getattr(neuron, "DynamicSRMNode", neuron.SRMNode)
    if node is neuron.SRMNode:
        return node, dict(common, tau_response=tau_response, tau_refractory=tau_refractory)
    return node, dict(common, init_tau_response=tau_response, init_tau_refractory=tau_refractory)


@torch.no_grad()
def overlay_label(x_ntchw: torch.Tensor, y: torch.Tensor, label_scale: float = 1.0):
    x = x_ntchw.clone()
    B = x.size(0)
    y = y.to(x.device, dtype=torch.long).view(-1)
    x[:, :, 0, 0, :NUM_CLASSES] = 0.0
    vmax = x.abs().amax(dim=(1, 2, 3, 4)) + 1e-6
    b = torch.arange(B, device=x.device)
    x[b, :, 0, 0, y.clamp(0, NUM_CLASSES - 1)] = vmax[b].view(B, 1) * float(label_scale)
    return x


def goodness(h: torch.Tensor):
    return h.pow(2).mean(dim=1)


def ff_loss(h_pos: torch.Tensor, h_neg: torch.Tensor, alpha: float):
    g_pos = goodness(h_pos)
    g_neg = goodness(h_neg)
    delta = g_pos - g_neg
    return F.silu(-alpha * delta).mean()


class DVSFFNet(nn.Module):
    """
    Forward-Forward style training on the official SpikingJelly DVSGestureNet backbone.

    This keeps the same network structure as the original SpikingJelly DVS example,
    but uses a global FF objective on the final output rates rather than BP with MSE.
    """
    def __init__(self, args):
        super().__init__()
        spike_cls, spike_kwargs = make_neuron(
            model=args.model,
            tau=args.tau,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
            tau_response=args.tau_response,
            tau_refractory=args.tau_refractory,
        )
        self.spike_cls = spike_cls
        self.net = parametric_lif_net.DVSGestureNet(
            channels=args.channels,
            spiking_neuron=spike_cls,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            **spike_kwargs
        )
        functional.set_step_mode(self.net, 'm')
        if args.cupy:
            functional.set_backend(self.net, 'cupy', instance=spike_cls)

    def forward_rates(self, x_ntchw: torch.Tensor):
        x_tnchw = x_ntchw.transpose(0, 1)
        functional.reset_net(self.net)
        out_seq = self.net(x_tnchw)
        return out_seq.mean(0)

    @torch.no_grad()
    def goodness_per_class(self, x_ntchw: torch.Tensor, label_scale: float):
        all_g = []
        for lab in range(NUM_CLASSES):
            labels = torch.full((x_ntchw.size(0),), lab, device=x_ntchw.device, dtype=torch.long)
            x_lab = overlay_label(x_ntchw, labels, label_scale=label_scale)
            out = self.forward_rates(x_lab)
            all_g.append(goodness(out).unsqueeze(1))
        return torch.cat(all_g, dim=1)

    @torch.no_grad()
    def predict(self, x_ntchw: torch.Tensor, label_scale: float):
        return self.goodness_per_class(x_ntchw, label_scale).argmax(dim=1)


@torch.no_grad()
def make_examples(net: DVSFFNet, x: torch.Tensor, y_true: torch.Tensor, label_scale: float,
                  neg_mode: str = "hard", epsilon: float = 1e-12):
    if neg_mode == "hard":
        g = net.goodness_per_class(x, label_scale=label_scale)
        g[torch.arange(x.size(0), device=x.device), y_true] = 0.0
        y_neg = torch.multinomial(torch.sqrt(g + epsilon), 1).squeeze(1)
    else:
        offset = torch.randint(1, NUM_CLASSES, y_true.shape, device=y_true.device)
        y_neg = (y_true + offset) % NUM_CLASSES
    x_pos = overlay_label(x, y_true, label_scale=label_scale)
    x_neg = overlay_label(x, y_neg, label_scale=label_scale)
    return x_pos, x_neg


@torch.no_grad()
def evaluate(net: DVSFFNet, loader: DataLoader, device: torch.device, label_scale: float):
    net.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    total = 0
    correct = 0
    spike_stats = SpikeStats(net.net, net.spike_cls)
    t0 = time.time()
    for frame, label in loader:
        frame = frame.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True)
        pred = net.predict(frame, label_scale=label_scale)
        update_confusion_matrix(confusion, pred.detach().cpu(), label.detach().cpu())
        correct += (pred == label).sum().item()
        total += label.numel()
    elapsed = max(time.time() - t0, 1e-9)
    cls = macro_classification_metrics(confusion)
    out = {
        "acc": correct / max(total, 1),
        "macro_precision": cls["macro_precision"],
        "macro_recall": cls["macro_recall"],
        "macro_f1": cls["macro_f1"],
        "samples": total,
        "time_sec": elapsed,
        "samples_per_sec": total / elapsed,
        "latency_ms_per_sample": 1000.0 * elapsed / max(total, 1),
        "spike_count": spike_stats.counts,
        "global_spike_rate": spike_stats.counts / max(spike_stats.elems, 1.0),
        "confusion_matrix": confusion,
    }
    spike_stats.close()
    return out


def main():
    parser = argparse.ArgumentParser(description='FF train DVS Gesture with 4 neuron types on official SpikingJelly structure')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-data-dir', type=str,default="/home/public03/yhxu/spikingjelly/dataset/DVSGesture",  help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./result', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, default='', help='resume from checkpoint')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of official DVSGestureNet')
    parser.add_argument('--model', type=str, default='lif', choices=['lif', 'alif', 'srm', 'dynsrm'])
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--v-threshold', type=float, default=0.5)
    parser.add_argument('--v-reset', type=float, default=0.0)
    parser.add_argument('--tau-response', type=float, default=2.0)
    parser.add_argument('--tau-refractory', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=6.0)
    parser.add_argument('--label-scale', type=float, default=1.0)
    parser.add_argument('--neg-mode', type=str, default='hard', choices=['hard', 'random'])
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')

    train_loader = DataLoader(dataset=train_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)

    model = DVSFFNet(args).to(device)
    print(model.net)

    scaler = amp.GradScaler() if args.amp else None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    start_epoch = 0
    best_test_acc = -1.0
    best_test_f1 = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_acc = checkpoint.get('best_test_acc', -1.0)
        best_test_f1 = checkpoint.get('best_test_f1', -1.0)

    out_dir = os.path.join(args.out_dir, f'FF_{args.model}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')
    if args.amp:
        out_dir += '_amp'
    if args.cupy:
        out_dir += '_cupy'
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    csv_path = os.path.join(out_dir, 'metrics.csv')
    if start_epoch == 0 and not os.path.exists(csv_path):
        pd.DataFrame(columns=[
            'epoch', 'train_loss', 'train_acc', 'train_macro_f1', 'test_acc', 'test_macro_f1',
            'best_test_acc', 'best_test_f1', 'train_speed', 'test_speed',
            'train_latency_ms_per_sample', 'test_latency_ms_per_sample',
            'train_spike_count', 'train_global_spike_rate',
            'test_spike_count', 'test_global_spike_rate'
        ]).to_csv(csv_path, index=False)

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0
        train_correct = 0
        train_confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        train_spikes = SpikeStats(model.net, model.spike_cls)

        t_train = time.time()
        for frame, label in train_loader:
            optimizer.zero_grad()
            frame = frame.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True)

            x_pos, x_neg = make_examples(model, frame, label, label_scale=args.label_scale, neg_mode=args.neg_mode)

            if scaler is not None:
                with amp.autocast():
                    out_pos = model.forward_rates(x_pos)
                    out_neg = model.forward_rates(x_neg)
                    loss = ff_loss(out_pos, out_neg, alpha=args.alpha)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_pos = model.forward_rates(x_pos)
                out_neg = model.forward_rates(x_neg)
                loss = ff_loss(out_pos, out_neg, alpha=args.alpha)
                loss.backward()
                optimizer.step()

            pred = model.predict(frame, label_scale=args.label_scale)
            update_confusion_matrix(train_confusion, pred.detach().cpu(), label.detach().cpu())
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_correct += (pred == label).sum().item()

        train_time = max(time.time() - t_train, 1e-9)
        train_loss /= max(train_samples, 1)
        train_acc = train_correct / max(train_samples, 1)
        train_cls = macro_classification_metrics(train_confusion)
        train_spike_count = train_spikes.counts
        train_spike_rate = train_spikes.counts / max(train_spikes.elems, 1.0)
        train_spikes.close()
        lr_scheduler.step()

        test_metrics = evaluate(model, test_loader, device=device, label_scale=args.label_scale)
        test_acc = test_metrics['acc']
        test_f1 = test_metrics['macro_f1']

        save_max = False
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = max(best_test_f1, test_f1)
            save_max = True
        else:
            best_test_f1 = max(best_test_f1, test_f1)

        checkpoint = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_test_acc': best_test_acc,
            'best_test_f1': best_test_f1,
            'args': vars(args),
        }
        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
            pd.DataFrame(
                test_metrics['confusion_matrix'].tolist(),
                index=[f"true_{i}" for i in range(NUM_CLASSES)],
                columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
            ).to_csv(os.path.join(out_dir, 'best_test_confusion.csv'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_macro_f1': train_cls['macro_f1'],
            'test_acc': test_acc,
            'test_macro_f1': test_f1,
            'best_test_acc': best_test_acc,
            'best_test_f1': best_test_f1,
            'train_speed': train_samples / train_time,
            'test_speed': test_metrics['samples_per_sec'],
            'train_latency_ms_per_sample': 1000.0 * train_time / max(train_samples, 1),
            'test_latency_ms_per_sample': test_metrics['latency_ms_per_sample'],
            'train_spike_count': train_spike_count,
            'train_global_spike_rate': train_spike_rate,
            'test_spike_count': test_metrics['spike_count'],
            'test_global_spike_rate': test_metrics['global_spike_rate'],
        }
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)

        remaining = (time.time() - t_epoch) * max(args.epochs - epoch - 1, 0)
        print(args)
        print(out_dir)
        print(
            f'epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
            f'train_f1={train_cls["macro_f1"]:.4f}, test_acc={test_acc:.4f}, '
            f'test_f1={test_f1:.4f}, best_test_acc={best_test_acc:.4f}'
        )
        print(f'train speed={train_samples / train_time:.4f} images/s, test speed={test_metrics["samples_per_sec"]:.4f} images/s')
        print(f'train spike rate={train_spike_rate:.6f}, test spike rate={test_metrics["global_spike_rate"]:.6f}')
        print(f'escape time={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining))}\n')


if __name__ == '__main__':
    main()
