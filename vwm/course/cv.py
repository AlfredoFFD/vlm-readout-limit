"""5-fold cross-validation for the Siamese CNN.

Addresses the reviewer concern: a single 70/15/15 stratified split with 100%
test accuracy is suspicious. Better evidence is 5-fold CV showing the result
is stable across data splits.

We perform stratified 5-fold over (set_size x change_flag), training a fresh
model for each fold, then report mean +/- SD test accuracy and the per-fold
accuracy-by-set-size matrix.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LuckVogelDataset
from .model import SiameseCNN
from .train import TrainConfig, _epoch_loop


def _stratified_kfold_ids(manifest_path: Path, k: int = 5, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    trials = manifest["trials"]
    rng = random.Random(seed)
    by_strata: dict[tuple[int, bool], list[str]] = {}
    for t in trials:
        by_strata.setdefault((int(t["set_size"]), bool(t["change"])), []).append(t["trial_id"])

    # Build k folds, each stratified.
    folds: list[list[str]] = [[] for _ in range(k)]
    for key, ids in by_strata.items():
        rng.shuffle(ids)
        for i, tid in enumerate(ids):
            folds[i % k].append(tid)

    splits: list[tuple[list[str], list[str]]] = []
    for f in range(k):
        test_ids = folds[f]
        train_ids = [tid for i, fold in enumerate(folds) if i != f for tid in fold]
        splits.append((train_ids, test_ids))
    return splits


@dataclass
class CVConfig:
    manifest_path: str
    out_dir: str
    k: int = 5
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    num_workers: int = 2
    seed: int = 0
    device: str = "cuda"
    augment: bool = True


def run_cv(cfg: CVConfig) -> dict:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = _stratified_kfold_ids(Path(cfg.manifest_path), k=cfg.k, seed=cfg.seed)

    per_fold: list[dict] = []
    for fold_i, (train_ids, test_ids) in enumerate(splits):
        torch.manual_seed(cfg.seed + fold_i)
        train_ds = LuckVogelDataset(cfg.manifest_path, train_ids, augment=cfg.augment)
        test_ds = LuckVogelDataset(cfg.manifest_path, test_ids, augment=False)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers,
                                  pin_memory=(device == "cuda"))
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers,
                                 pin_memory=(device == "cuda"))

        model = SiameseCNN().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                 betas=(0.9, 0.99))
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for ep in range(1, cfg.epochs + 1):
            tr_loss, tr_acc = _epoch_loop(model, train_loader, loss_fn, optim, device, train=True)
        te_loss, te_acc = _epoch_loop(model, test_loader, loss_fn, optim, device, train=False)

        # Per-set-size breakdown
        by_n: dict[int, list[int]] = {}
        model.train(False)
        with torch.inference_mode():
            for sample, test, label, meta in test_loader:
                sample = sample.to(device, non_blocking=True)
                test = test.to(device, non_blocking=True)
                logits = model(sample, test)
                pred = (logits > 0).int().cpu().tolist()
                true = (label > 0.5).int().cpu().tolist()
                for n, p, t in zip(meta["set_size"].tolist(), pred, true):
                    by_n.setdefault(int(n), []).append(int(p == t))
        per_set_size = {n: sum(v) / len(v) for n, v in sorted(by_n.items())}
        per_fold.append({
            "fold": fold_i,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "test_acc": te_acc,
            "accuracy_by_set_size": per_set_size,
        })
        print(f"  fold {fold_i+1}/{cfg.k}: test_acc={te_acc:.3f}  per_n={per_set_size}")
        del model

    accs = np.array([f["test_acc"] for f in per_fold], dtype=np.float64)
    summary = {
        "k": cfg.k,
        "mean_acc": float(accs.mean()),
        "std_acc": float(accs.std(ddof=1)) if cfg.k > 1 else 0.0,
        "min_acc": float(accs.min()),
        "max_acc": float(accs.max()),
        "per_fold": per_fold,
    }
    (out_dir / "cv_report.json").write_text(json.dumps(summary, indent=2))
    print(f"CV summary: {summary['mean_acc']:.4f} +/- {summary['std_acc']:.4f}  "
          f"(min {summary['min_acc']:.3f}, max {summary['max_acc']:.3f}, k={cfg.k})")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/stimuli/manifest_C2.json")
    p.add_argument("--out-dir", default="results/course")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args(argv)
    cfg = CVConfig(
        manifest_path=args.manifest, out_dir=args.out_dir,
        k=args.k, epochs=args.epochs,
    )
    run_cv(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
