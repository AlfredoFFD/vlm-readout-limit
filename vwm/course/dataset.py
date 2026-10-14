"""PyTorch Dataset for the Luck-Vogel stimuli (CCOM-6070 course deliverable).

Each item returns:
    sample : Tensor[3, 300, 400]   # in [0, 1]
    test   : Tensor[3, 300, 400]
    label  : Tensor[]  (0.0 = no-change, 1.0 = change)
    meta   : dict with set_size, trial_id, etc. (for stratified eval)

We use the C2 manifest by default because C2's separate sample/test pair is what a
classical change detector consumes naturally; the same images also appear in C1's
combined view. Train/val/test splits are stratified by set_size to avoid evaluating
on a different distribution than we trained on.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_pil_as_tensor(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HxWx3
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # 3xHxW


class LuckVogelDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        trial_ids: Iterable[str] | None = None,
        augment: bool = False,
    ):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        trials = manifest["trials"]
        if trial_ids is not None:
            ids = set(trial_ids)
            trials = [t for t in trials if t["trial_id"] in ids]
        self.trials = trials
        self.augment = augment
        self.manifest_root = Path(manifest_path).parent

    def __len__(self) -> int:
        return len(self.trials)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.exists():
            return path
        alt = self.manifest_root.parent / p
        return alt if alt.exists() else path

    def __getitem__(self, idx: int):
        t = self.trials[idx]
        sample = _load_pil_as_tensor(self._resolve(t["sample_path"]))
        test = _load_pil_as_tensor(self._resolve(t["test_path"]))
        if self.augment:
            # Horizontal flip together (keeps the change consistent)
            if random.random() < 0.5:
                sample = torch.flip(sample, dims=(2,))
                test = torch.flip(test, dims=(2,))
        label = torch.tensor(1.0 if t["change"] else 0.0, dtype=torch.float32)
        meta = {
            "trial_id": t["trial_id"],
            "set_size": int(t["set_size"]),
            "change": bool(t["change"]),
        }
        return sample, test, label, meta


def split_trials_by_set_size(
    manifest_path: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified train/val/test split, balanced across set sizes AND change/no-change."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    trials = manifest["trials"]
    rng = random.Random(seed)
    by_strata: dict[tuple[int, bool], list[str]] = {}
    for t in trials:
        key = (int(t["set_size"]), bool(t["change"]))
        by_strata.setdefault(key, []).append(t["trial_id"])

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for key, ids in by_strata.items():
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        train += ids[:n_tr]
        val += ids[n_tr:n_tr + n_va]
        test += ids[n_tr + n_va:]
    return train, val, test
