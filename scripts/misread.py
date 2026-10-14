"""DOES A MODEL READ THE OTHER CONVENTION'S NUMBERS IN ITS OWN UNITS?

Qwen2.5-VL-7B scores below chance on normalized 0-999 coordinates; Qwen3-VL-4B
scores near chance on absolute pixels. Each was trained on the other convention.
If a model silently reads the option numbers in its training units, then its
answer should be the option whose numbers, so misread, land nearest the object.

For each locate trial we compute the option a model would pick under the misread
(normalized numbers taken as pixels; pixel numbers divided by 1000 and scaled to
W x H), and count how often the model's argmax equals that option. Chance is the
rate expected from the model's own marginal answer distribution against the
distribution of misread-predicted slots (the same bias-aware null the paper uses).
An exact binomial p-value against 1/K is also reported.

Usage: .venv/Scripts/python.exe scripts/misread.py   (writes results/MISREAD.txt)
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

RUNS = {
    "Qwen2.5-VL-7B": ("results/coco/convention_Qwen2_5-VL-7B-Instruct.jsonl", "norm1000", "as_pixels"),
    "Qwen3-VL-4B":   ("results/coco/convention_qwen3vl4b.jsonl", "pixel", "as_norm1000"),
    "InternVL3-2B":  ("results/coco/convention_InternVL3-2B-hf.jsonl", "norm1000", "as_pixels"),
    # frontier logs carry no geometry; prompts are byte-identical replays of the Qwen3 run,
    # so geometry is joined from that log by (trial_id, desc). Both misread directions are
    # tested because neither vendor documents a training convention.
    "Gemini (pixel)":    ("results/frontier/gemini-flash-latest.jsonl", "pixel", "as_norm1000"),
    "Gemini (norm)":     ("results/frontier/gemini-flash-latest.jsonl", "norm1000", "as_pixels"),
    "GPT-4o (pixel)":    ("results/frontier/gpt-4o.jsonl", "pixel", "as_norm1000"),
    "GPT-4o (norm)":     ("results/frontier/gpt-4o.jsonl", "norm1000", "as_pixels"),
}
GEOM = "results/coco/convention_qwen3vl4b.jsonl"
OPT = re.compile(r"(\d+)=\((\d+), (\d+)\)")


def misread_slot(r, mode):
    opts = OPT.findall(r["prompt"])
    assert len(opts) == r["n_options"], r["trial_id"]
    x1, y1, x2, y2 = r["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    W, H = r["W"], r["H"]
    best, bd = None, 1e18
    for k, x, y in opts:
        x, y = float(x), float(y)
        if mode == "as_norm1000":          # pixel numbers read as 0-999 fractions
            x, y = x / 1000 * W, y / 1000 * H
        d = (x - cx) ** 2 + (y - cy) ** 2  # numbers read as pixels
        if d < bd:
            best, bd = int(k) - 1, d
    return best


def binom_tail(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    out = ["MISREAD  does the model read the other convention's numbers in its own units?",
           "=" * 78,
           f"{'model':16} {'condition':10} {'misread':10} {'n':>4} {'target%':>8} {'misread%':>9} {'bias-null%':>11} {'p(>1/K)':>10}"]
    for model, (path, desc, mode) in RUNS.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if r["desc"] == desc and r["direction"] == "locate"]
        if "prompt" not in rows[0]:
            geom = {(g["trial_id"], g["desc"]): g for g in map(json.loads, open(GEOM, encoding="utf-8"))
                    if g["direction"] == "locate"}
            for r in rows:
                g = geom[(r["trial_id"], r["desc"])]
                assert g["target_idx"] == r["target_idx"], r["trial_id"]
                r.update(prompt=g["prompt"], W=g["W"], H=g["H"], bbox=g["bbox"])
            rows = [r for r in rows if r["argmax"] is not None]  # unparseable answers carry no slot
        n = len(rows)
        pred = [misread_slot(r, mode) for r in rows]
        hit_target = sum(r["argmax"] == r["target_idx"] for r in rows) / n
        hit_mis = sum(a["argmax"] == p for a, p in zip(rows, pred)) / n
        # bias-aware null: P(argmax = s) * P(predicted = s) summed over slots
        pa = Counter(r["argmax"] for r in rows)
        pp = Counter(pred)
        null = sum(pa[s] / n * pp[s] / n for s in range(rows[0]["n_options"]))
        p = binom_tail(round(hit_mis * n), n, 1 / rows[0]["n_options"])
        same = sum(p_ == r["target_idx"] for p_, r in zip(pred, rows)) / n
        out.append(f"{model:16} {desc:10} {mode:10} {n:4d} {100*hit_target:8.1f} {100*hit_mis:9.1f} {100*null:11.1f} {p:10.1e}")
        out.append(f"{'':16} misread-predicted slot coincides with the true target on {100*same:.1f}% of trials")
        # accurate models inflate the misread hit rate through those coincidences; restrict to
        # trials where the misread slot is NOT the target, with the bias-null on that subset
        sub = [(a, p_) for a, p_ in zip(rows, pred) if p_ != a["target_idx"]]
        m = len(sub)
        hit_sub = sum(a["argmax"] == p_ for a, p_ in sub) / m
        pa2, pp2 = Counter(a["argmax"] for a, _ in sub), Counter(p_ for _, p_ in sub)
        null_sub = sum(pa2[s_] / m * pp2[s_] / m for s_ in range(rows[0]["n_options"]))
        p_sub = binom_tail(round(hit_sub * m), m, 1 / rows[0]["n_options"])
        out.append(f"{'':16} excluding those: n={m}, misread {100*hit_sub:.1f}%, bias-null {100*null_sub:.1f}%, p(>1/K) {p_sub:.1e}")
    txt = "\n".join(out)
    print(txt)
    Path("results/MISREAD.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
