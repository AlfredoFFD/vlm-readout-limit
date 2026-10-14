"""Training and evaluation loop for the Siamese CNN baseline.

Uses BCEWithLogitsLoss with class-weighting (our dataset is 50/50, so balanced) and
Adam (1e-3, betas=0.9/0.99). Trains for a small number of epochs; full course report
typically ~15-25 epochs on the n=4200 train set (6 set sizes x 100 trials x 0.7 split
x 2 conditions if combined).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import LuckVogelDataset, split_trials_by_set_size
from .model import SiameseCNN


@dataclass
class TrainConfig:
    manifest_path: str
    out_dir: str
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    num_workers: int = 2
    seed: int = 0
    device: str = "cuda"
    augment: bool = True
    log_every: int = 50


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def _epoch_loop(model: nn.Module, loader: DataLoader, loss_fn, optim, device: str, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    n_total = 0
    n_correct = 0
    if train:
        ctx = torch.enable_grad()
    else:
        ctx = torch.inference_mode()
    with ctx:
        for sample, test, label, _meta in loader:
            sample = sample.to(device, non_blocking=True)
            test = test.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model(sample, test)
            loss = loss_fn(logits, label)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            total_loss += loss.item() * sample.size(0)
            n_total += sample.size(0)
            n_correct += int(((logits > 0) == (label > 0.5)).sum().item())
    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


def train_siamese(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    train_ids, val_ids, test_ids = split_trials_by_set_size(cfg.manifest_path, seed=cfg.seed)
    train_ds = LuckVogelDataset(cfg.manifest_path, train_ids, augment=cfg.augment)
    val_ds = LuckVogelDataset(cfg.manifest_path, val_ids, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(device == "cuda"))

    model = SiameseCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                             betas=(0.9, 0.99))
    loss_fn = nn.BCEWithLogitsLoss()

    history = TrainHistory()
    best_val_acc = -1.0
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "siamese_best.pt"

    t0 = time.monotonic()
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _epoch_loop(model, train_loader, loss_fn, optim, device, train=True)
        va_loss, va_acc = _epoch_loop(model, val_loader, loss_fn, optim, device, train=False)
        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)
        print(f"  epoch {ep:2d}/{cfg.epochs}: train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(), "epoch": ep, "val_acc": va_acc}, best_path)
    elapsed = time.monotonic() - t0

    summary = {
        "best_val_acc": best_val_acc,
        "elapsed_s": elapsed,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ids),
        "test_ids": test_ids,
        "history": history.__dict__,
        "ckpt": str(best_path),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_siamese(
    manifest_path: str,
    ckpt_path: str,
    eval_trial_ids: list[str],
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    device = device if torch.cuda.is_available() else "cpu"
    model = SiameseCNN().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.train(False)

    ds = LuckVogelDataset(manifest_path, eval_trial_ids, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=(device == "cuda"))

    n_total = 0
    n_correct = 0
    by_n: dict[int, list[int]] = {}
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for sample, test, label, meta in loader:
            sample = sample.to(device, non_blocking=True)
            test = test.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model(sample, test)
            pred = (logits > 0).int().cpu().tolist()
            true = (label > 0.5).int().cpu().tolist()
            for n, p, t in zip(meta["set_size"].tolist(), pred, true):
                by_n.setdefault(int(n), []).append(int(p == t))
                n_total += 1
                n_correct += int(p == t)
                y_pred.append(p); y_true.append(t)

    acc = n_correct / max(n_total, 1)
    per_n = {n: sum(v) / len(v) for n, v in sorted(by_n.items())}
    return {
        "test_size": n_total,
        "accuracy": acc,
        "accuracy_by_set_size": per_n,
        "y_true": y_true,
        "y_pred": y_pred,
    }
