"""DOES THE READOUT PENALTY GENERALISE FROM COLOUR TO SPACE?

Accuracy per (direction, description, condition), with the paired contrasts that
matter tested against cellname. Pairing is per trial: the same image, the same
target square and the same option order are scored under all three descriptions,
so an exact McNemar test on the discordant pairs is the right test and no
bootstrap over stimuli is needed for the contrast.

Usage: .venv/Scripts/python.exe scripts/analyze_coordinates.py [path.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

SRC = Path(sys.argv[1] if len(sys.argv) > 1
           else "results/gpu_p3/coordinate_grounding.jsonl")
DESCS = ["cellname", "pixel", "norm1000"]
CONDS = ["visible", "cached"]


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    print(f"{len(rows)} cells from {SRC}")
    errs = sum(1 for r in rows if any(k.startswith("error") for k in r))
    if errs:
        print(f"!! {errs} cells carry an error field")

    # (direction, desc, cond, trial) -> correct
    ok = {}
    mass = defaultdict(list); ptok = defaultdict(list); nopt = {}
    for r in rows:
        for c in CONDS:
            if f"correct_{c}" in r:
                ok[(r["direction"], r["desc"], c, r["trial_id"])] = r[f"correct_{c}"]
                mass[(r["direction"], r["desc"], c)].append(r[f"mass_{c}"])
        ptok[(r["direction"], r["desc"])].append(r["prompt_tokens"])
        nopt[r["direction"]] = r["n_options"]

    dirs = sorted({k[0] for k in ok})
    print()
    for d in dirs:
        print(f"=== {d}   ({nopt[d]} options, chance {100/nopt[d]:.1f}%)")
        print(f"{'desc':<11}{'ptok':>6}" +
              "".join(f"{c:>12}" for c in CONDS) + f"{'mass':>8}")
        for de in DESCS:
            cells = []
            for c in CONDS:
                v = [ok[k] for k in ok if k[:3] == (d, de, c)]
                cells.append(f"{100*np.mean(v):>11.1f}%" if v else f"{'-':>12}")
            m = np.mean([x for c in CONDS for x in mass.get((d, de, c), [])] or [np.nan])
            print(f"{de:<11}{int(np.median(ptok[(d, de)])):>6}" + "".join(cells) +
                  f"{m:>8.2f}")

        print(f"  paired contrasts vs cellname (exact McNemar on discordant trials)")
        for c in CONDS:
            base = {k[3]: ok[k] for k in ok if k[:3] == (d, "cellname", c)}
            for de in DESCS[1:]:
                other = {k[3]: ok[k] for k in ok if k[:3] == (d, de, c)}
                sh = sorted(set(base) & set(other))
                if not sh:
                    continue
                b = sum(1 for t in sh if base[t] and not other[t])   # cellname only
                cc = sum(1 for t in sh if other[t] and not base[t])  # other only
                delta = 100 * (np.mean([other[t] for t in sh])
                               - np.mean([base[t] for t in sh]))
                p = binomtest(min(b, cc), b + cc, 0.5).pvalue if (b + cc) else 1.0
                print(f"    {c:<8} {de:<9} {delta:>+7.1f} pts   "
                      f"cellname-only {b:>3}  {de}-only {cc:>3}   p={p:.2e}")
        print()

    print("If pixel and norm1000 both fall far below cellname, the readout penalty")
    print("is about coordinate systems and not about colour, and it reaches the")
    print("pixel conventions localisation benchmarks are scored in. If norm1000")
    print("holds while pixel falls, the claim sharpens instead: a model is usable")
    print("in a coordinate system it was TRAINED in and not in one merely described")
    print("to it, which is also why a CIELAB hue angle failed.")


if __name__ == "__main__":
    main()
