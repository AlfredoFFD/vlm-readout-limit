"""End-to-end CCOM-6070 driver.

1. Split the C2 trial manifest into stratified train/val/test (70/15/15).
2. Train the Siamese CNN.
3. Evaluate Siamese, pixel-diff, and random baselines on the held-out test set.
4. Write a JSON report + a comparison bar chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train import TrainConfig, train_siamese, evaluate_siamese
from .baselines import pixel_diff_change_detector, random_change_detector
from .dataset import split_trials_by_set_size


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/stimuli/manifest_C2.json")
    p.add_argument("--out-dir", default="results/course")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--skip-train", action="store_true",
                   help="If a checkpoint already exists, reuse it instead of retraining.")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = split_trials_by_set_size(args.manifest, seed=0)

    ckpt_path = out_dir / "siamese_best.pt"
    if args.skip_train and ckpt_path.exists():
        print("[siamese] reusing existing checkpoint")
        train_summary = json.loads((out_dir / "train_summary.json").read_text())
    else:
        cfg = TrainConfig(manifest_path=args.manifest, out_dir=str(out_dir), epochs=args.epochs)
        train_summary = train_siamese(cfg)

    # Evaluate three detectors on the held-out test set.
    siam_metrics = evaluate_siamese(args.manifest, str(ckpt_path), test_ids)
    px_metrics = pixel_diff_change_detector(args.manifest, val_ids, test_ids)
    rnd_metrics = random_change_detector(args.manifest, test_ids, seed=0)

    report = {
        "splits": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "train_summary": {k: v for k, v in train_summary.items() if k != "test_ids"},
        "siamese": {
            "accuracy": siam_metrics["accuracy"],
            "accuracy_by_set_size": siam_metrics["accuracy_by_set_size"],
            "test_size": siam_metrics["test_size"],
        },
        "pixel_diff": px_metrics,
        "random": rnd_metrics,
    }
    (out_dir / "course_report.json").write_text(json.dumps(report, indent=2))

    # ---- Comparison bar chart ----
    detectors = [("Random", rnd_metrics), ("Pixel-diff", px_metrics), ("Siamese CNN", siam_metrics)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = [d[0] for d in detectors]
    accs = [d[1]["accuracy"] for d in detectors]
    bars = ax.bar(names, accs, color=["#bdbdbd", "#4c72b0", "#dd8452"], edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.015, f"{a:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("CCOM-6070: Change-detection baselines on held-out test set")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_course_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_course_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Accuracy by set size figure ----
    set_sizes = sorted(siam_metrics["accuracy_by_set_size"].keys(),
                       key=lambda x: int(x))
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    for name, m, color in [
        ("Siamese CNN", siam_metrics, "#dd8452"),
        ("Pixel-diff", px_metrics, "#4c72b0"),
        ("Random", rnd_metrics, "#bdbdbd"),
    ]:
        ys = [m["accuracy_by_set_size"][int(n)] for n in set_sizes]
        ax2.plot([int(n) for n in set_sizes], ys, "-o", label=name, color=color, lw=2)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Set size N")
    ax2.set_ylabel("Test accuracy")
    ax2.set_xticks([int(n) for n in set_sizes])
    ax2.set_ylim(0.4, 1.05)
    ax2.set_title("Accuracy vs set size, by detector")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_course_by_set_size.png", dpi=300, bbox_inches="tight")
    fig2.savefig(out_dir / "fig_course_by_set_size.pdf", bbox_inches="tight")
    plt.close(fig2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
