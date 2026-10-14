"""Classical baselines for the CCOM-6070 change-detection task.

1. Pixel-diff + threshold: mean absolute pixel difference between sample and test,
   threshold tuned on the validation set.
2. Random chance: 50/50 coin flip (sanity floor).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _load(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


# -------- pixel-diff baseline --------

def pixel_diff_score(sample_path: str | Path, test_path: str | Path) -> float:
    """Returns mean absolute pixel difference in [0, 1]."""
    s = _load(sample_path)
    t = _load(test_path)
    return float(np.abs(s - t).mean())


def fit_pixel_diff_threshold(
    trials: list[dict],
    manifest_root: Path,
) -> tuple[float, float]:
    """Choose the threshold that maximizes balanced accuracy on `trials`.

    Returns (best_threshold, best_balanced_accuracy).
    """
    scores: list[tuple[float, bool]] = []
    for t in trials:
        s_path = manifest_root.parent / t["sample_path"]
        t_path = manifest_root.parent / t["test_path"]
        if not s_path.exists():
            s_path = Path(t["sample_path"])
        if not t_path.exists():
            t_path = Path(t["test_path"])
        score = pixel_diff_score(s_path, t_path)
        scores.append((score, bool(t["change"])))

    scores.sort()
    best_thresh = 0.0
    best_bacc = -1.0
    pos = [s for s, c in scores if c]
    neg = [s for s, c in scores if not c]
    # Sweep candidate thresholds at all unique score midpoints
    uniq = sorted(set(s for s, _ in scores))
    if not uniq:
        return 0.0, 0.0
    for i in range(len(uniq)):
        thr = uniq[i]
        tp = sum(1 for s in pos if s > thr)
        fn = len(pos) - tp
        fp = sum(1 for s in neg if s > thr)
        tn = len(neg) - fp
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        bacc = 0.5 * (sens + spec)
        if bacc > best_bacc:
            best_bacc = bacc
            best_thresh = thr
    return best_thresh, best_bacc


def pixel_diff_change_detector(
    manifest_path: str | Path,
    val_trial_ids: Iterable[str],
    eval_trial_ids: Iterable[str],
) -> dict:
    """Fit a threshold on val, evaluate on eval. Returns metrics + threshold."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_root = Path(manifest_path)
    by_id = {t["trial_id"]: t for t in manifest["trials"]}
    val_trials = [by_id[i] for i in val_trial_ids if i in by_id]
    eval_trials = [by_id[i] for i in eval_trial_ids if i in by_id]

    thr, val_bacc = fit_pixel_diff_threshold(val_trials, manifest_root)
    metrics = evaluate_detector(
        eval_trials, manifest_root,
        predict_fn=lambda s, t: pixel_diff_score(s, t) > thr,
    )
    metrics["threshold"] = thr
    metrics["val_balanced_accuracy"] = val_bacc
    return metrics


# -------- random baseline --------

def random_change_detector(
    manifest_path: str | Path,
    eval_trial_ids: Iterable[str],
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_root = Path(manifest_path)
    by_id = {t["trial_id"]: t for t in manifest["trials"]}
    eval_trials = [by_id[i] for i in eval_trial_ids if i in by_id]
    return evaluate_detector(
        eval_trials, manifest_root,
        predict_fn=lambda s, t: bool(rng.random() < 0.5),
    )


# -------- evaluator --------

def evaluate_detector(
    trials: list[dict],
    manifest_root: Path,
    predict_fn,
) -> dict:
    """predict_fn(sample_path, test_path) -> bool (True = change). Returns metrics."""
    y_true: list[bool] = []
    y_pred: list[bool] = []
    by_n: dict[int, list[tuple[bool, bool]]] = {}
    for t in trials:
        s_path = manifest_root.parent / t["sample_path"]
        if not s_path.exists():
            s_path = Path(t["sample_path"])
        t_path = manifest_root.parent / t["test_path"]
        if not t_path.exists():
            t_path = Path(t["test_path"])
        pred = bool(predict_fn(s_path, t_path))
        y_true.append(bool(t["change"]))
        y_pred.append(pred)
        by_n.setdefault(int(t["set_size"]), []).append((bool(t["change"]), pred))

    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    pos = [(a, b) for a, b in zip(y_true, y_pred) if a]
    neg = [(a, b) for a, b in zip(y_true, y_pred) if not a]
    sens = float(np.mean([b for _, b in pos])) if pos else float("nan")
    spec = float(1 - np.mean([b for _, b in neg])) if neg else float("nan")
    bacc = 0.5 * (sens + spec)

    per_set_size = {}
    for n, items in sorted(by_n.items()):
        correct = sum(1 for a, b in items if a == b)
        per_set_size[n] = correct / len(items)

    return {
        "n": len(trials),
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": bacc,
        "accuracy_by_set_size": per_set_size,
    }
