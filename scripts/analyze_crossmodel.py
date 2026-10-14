"""CROSS-MODEL SUMMARY: calibration yield, capability gate, and the convention gap.

The capability gate comes first and is not negotiable. A convention gap is a
difference between two readouts of the same competence; where there is no
competence there is nothing for a readout to lose, and a null contrast in a model
that cannot do the task is uninformative rather than disconfirming. The gate is
the word rung's best available readout clearing chance by a stated margin.

Usage: .venv/Scripts/python.exe scripts/analyze_crossmodel.py
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

GATE_MARGIN = 20.0        # points above chance the word rung must clear

def acc(rows, rung, field):
    v = [r[field] for r in rows if r["rung"] == rung and field in r]
    return (100 * np.mean(v), len(v)) if v else (float("nan"), 0)

def paired(rows, a, b, field):
    A = {r["trial_id"]: r[field] for r in rows if r["rung"] == a and field in r}
    B = {r["trial_id"]: r[field] for r in rows if r["rung"] == b and field in r}
    sh = sorted(set(A) & set(B))
    if not sh:
        return float("nan"), float("nan"), 0
    x = sum(1 for t in sh if A[t] and not B[t]); y = sum(1 for t in sh if B[t] and not A[t])
    d = 100 * (np.mean([B[t] for t in sh]) - np.mean([A[t] for t in sh]))
    p = binomtest(min(x, y), x + y, 0.5).pvalue if (x + y) else 1.0
    return d, p, len(sh)

def main():
    print(f"{'model':<26}{'K':>3}{'chance':>8}{'word(gen)':>11}{'word(ft)':>10}"
          f"{'best':>7}{'gate':>7}{'lbl-num':>9}{'p':>10}")
    rows_by = {}
    for f in sorted(glob.glob("results/crossmodel/*_contrast.jsonl")):
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not rows:
            continue
        name = Path(f).name.replace("_contrast.jsonl", "")
        rows_by[name] = rows
        K = rows[0]["K"]; ch = 100 / K
        wg, _ = acc(rows, "word", "visible_correct")
        wf, nf = acc(rows, "word", "visible_ft_correct")
        # amendment 3: the gate is evaluated on the readout the CONTRAST uses,
        # which is generate-and-parse. Scoring the gate on a readout the contrast
        # does not use let LLaVA pass on first-token while its contrast was
        # computed on an instrument sitting below chance.
        best = wg
        gate = "PASS" if best - ch >= GATE_MARGIN else "fail"
        d, p, n = paired(rows, "label", "numeric", "visible_correct")
        print(f"{name:<26}{K:>3}{ch:>7.1f}%{wg:>10.1f}%"
              f"{(f'{wf:.1f}%' if nf else '-'):>10}{best:>6.1f}%{gate:>7}"
              f"{d:>+8.1f}{p:>10.2e}")
    print()
    print(f"gate = best word-rung visible accuracy at least {GATE_MARGIN:.0f} points "
          f"above chance")
    print("lbl->num is the matched contrast: same stimuli, same alternatives, same")
    print("integer output, only the option DESCRIPTION swapped. Read it only for")
    print("models that PASS the gate.")
    print()
    print("CALIBRATED VOCABULARY PER MODEL (an outcome, not a setting):")
    for f in sorted(glob.glob("results/crossmodel/*_calibration.json")):
        c = json.load(open(f, encoding="utf-8"))
        y = c["yield"]
        print(f"  {Path(f).name.replace('_calibration.json',''):<26} "
              f"{y['stable_angles']:>3} stable angles, {y['anchors']} anchors: "
              f"{[(d['wheel'], d['name']) for d in c['named']]}")

if __name__ == "__main__":
    main()
