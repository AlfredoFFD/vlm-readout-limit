"""READ THE INTERFACE LADDER: where on the ladder does the crossover die?

Unit of inference is the TRIAL, not the cell. Each trial contributes exactly one
q1 DiD and one q2 DiD per rung, and the statistic is their paired difference -- the
crossover interaction. Testing cells independently would pseudoreplicate, since q1
and q2 within a trial share the same two cache states and the same option list.

Reported per rung:
  visible acc     can the model do this interface at all, array on screen. A rung
                  at floor here cannot support any claim about its cached arm.
  cand mass       softmax mass on the scored alphabet. Low mass means the model is
                  answering off-alphabet and the margin is measuring a sideshow.
  DiD q1/q2/q3    predicted +, -, 0
  interaction     mean(DiD q1 - DiD q2) with a percentile bootstrap over trials,
                  and a Wilcoxon signed-rank test on the per-trial differences
  %free           interaction as a fraction of the `free` rung, the ladder's top

Usage: .venv/Scripts/python.exe scripts/analyze_ladder.py [path.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SRC = Path(sys.argv[1] if len(sys.argv) > 1
           else "results/gpu_p3/interface_ladder.jsonl")
RUNG_ORDER = ["free", "freepad", "word", "label", "numeric_gloss", "numeric",
              "numeric_bare"]


def boot(x, n=10000, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = r.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    print(f"{len(rows)} cells from {SRC}")
    errs = [r for r in rows if any(k.startswith("error") for k in r)]
    if errs:
        print(f"!! {len(errs)} cells carry an error field; first: "
              f"{ {k: v for k, v in errs[0].items() if k.startswith('error')} }")

    by = defaultdict(dict)                      # (rung, trial) -> {query: did}
    byz = defaultdict(dict)                     # same, spread-normalised
    vis = defaultdict(list); mass = defaultdict(list)
    cacc = defaultdict(list)                    # cached accuracy, q1/q2 only
    viscol = defaultdict(lambda: [0, 0])        # (rung, colour) -> [correct, n]
    ptok = defaultdict(list); spread = defaultdict(list)
    for r in rows:
        k = (r["rung"], r["trial_id"])
        if "did" in r:
            by[k][r["query"]] = r["did"]
            # Displaying the option list may simply inflate the whole scored range
            # rather than supply more evidence, and a raw margin cannot separate
            # those. Dividing each cell's margin by the SD of that cell's candidate
            # scores removes any per-cell gain, so a rung that only got louder
            # normalises back to its neighbour while a rung that got more
            # informative does not.
            sd = 0.5 * (np.std(r["scores_A"]) + np.std(r["scores_B"]))
            if sd > 0:
                byz[k][r["query"]] = ((r["margin_A"] - r["margin_B"]) / sd)
            spread[r["rung"]].append(sd)
        for ck in ("A", "B"):
            if f"argmax_{ck}" in r and r["query"] in ("q1", "q2"):
                cacc[r["rung"]].append(r[f"argmax_{ck}"] == r[f"target_idx_{ck}"])
        if "visible_correct" in r:
            vis[r["rung"]].append(r["visible_correct"])
            c = viscol[(r["rung"], r["target_A"])]
            c[0] += int(r["visible_correct"]); c[1] += 1
        for ck in ("A", "B"):
            if f"mass_{ck}" in r:
                mass[r["rung"]].append(r[f"mass_{ck}"])
        ptok[r["rung"]].append(r["prompt_tokens"])

    rungs = [x for x in RUNG_ORDER if any(k[0] == x for k in by)]
    print()
    print("RAW margin (logit units). Comparable within a rung, inflated across rungs")
    print("by any change in candidate-score gain.")
    print(f"{'rung':<9}{'ptok':>6}{'visacc':>9}{'cacheacc':>10}{'mass':>7}{'spread':>8}"
          f"{'DiD q1':>9}{'DiD q2':>9}{'DiD q3':>9}{'interact':>10}{'95% CI':>18}"
          f"{'wilcox p':>11}{'%free':>7}")

    ref = None
    nt = 0
    for rg in rungs:
        ks = [k for k in by if k[0] == rg and "q1" in by[k] and "q2" in by[k]]
        nt = len(ks)
        d1 = np.array([by[k]["q1"] for k in ks])
        d2 = np.array([by[k]["q2"] for k in ks])
        d3 = np.array([by[k]["q3"] for k in ks if "q3" in by[k]])
        inter = d1 - d2
        lo, hi = boot(inter)
        p = wilcoxon(inter).pvalue if len(inter) > 5 and np.any(inter != 0) else float("nan")
        mi = float(inter.mean())
        if ref is None:
            ref = mi
        pct = 100 * mi / ref if ref else float("nan")
        print(f"{rg:<9}{int(np.median(ptok[rg])):>6}"
              f"{100*np.mean(vis[rg]):>8.1f}%{100*np.mean(cacc[rg]):>9.1f}%"
              f"{np.mean(mass[rg]):>7.2f}{np.mean(spread[rg]):>8.1f}"
              f"{d1.mean():>+9.2f}{d2.mean():>+9.2f}{d3.mean():>+9.2f}"
              f"{mi:>+10.2f}{f'[{lo:+.1f},{hi:+.1f}]':>18}{p:>11.1e}{pct:>6.0f}%")

    print()
    print("SPREAD-NORMALISED margin (each cell divided by the SD of its own candidate")
    print("scores). This is the cross-rung comparison; the raw one is not.")
    print(f"{'rung':<9}{'DiD q1':>10}{'DiD q2':>10}{'DiD q3':>10}{'interact':>11}"
          f"{'95% CI':>18}{'wilcox p':>11}{'%free':>7}")
    refz = None
    for rg in rungs:
        ks = [k for k in byz if k[0] == rg and "q1" in byz[k] and "q2" in byz[k]]
        d1 = np.array([byz[k]["q1"] for k in ks])
        d2 = np.array([byz[k]["q2"] for k in ks])
        d3 = np.array([byz[k]["q3"] for k in ks if "q3" in byz[k]])
        inter = d1 - d2
        lo, hi = boot(inter)
        p = wilcoxon(inter).pvalue if len(inter) > 5 and np.any(inter != 0) else float("nan")
        mi = float(inter.mean())
        if refz is None:
            refz = mi
        print(f"{rg:<9}{d1.mean():>+10.3f}{d2.mean():>+10.3f}{d3.mean():>+10.3f}"
              f"{mi:>+11.3f}{f'[{lo:+.2f},{hi:+.2f}]':>18}{p:>11.1e}"
              f"{100*mi/refz if refz else float('nan'):>6.0f}%")

    print()
    print(f"n trials per rung: {nt}   interaction = mean(DiD q1 - DiD q2), paired "
          f"within trial (the trial is the unit of inference, not the cell)")
    print("free->freepad isolates prompt LENGTH; freepad->word isolates PRESENCE OF")
    print("ALTERNATIVES; word->label isolates OUTPUT CODE; label->numeric isolates")
    print("the RESPONSE ALPHABET with stimuli, options and output code all held fixed.")
    print("cacheacc is first-token accuracy on q1/q2 cached cells; chance is "
          f"{100/max(1,rows[0]['K']):.1f}%.")

    print()
    print("visible accuracy by target colour (the naming check):")
    cols = sorted({c for _, c in viscol})
    print(f"{'rung':<9}" + "".join(f"{c:>9}" for c in cols))
    for rg in rungs:
        cells = []
        for c in cols:
            k, n = viscol.get((rg, c), [0, 0])
            cells.append(f"{100*k/n:>8.0f}%" if n else f"{'-':>9}")
        print(f"{rg:<9}" + "".join(cells))


if __name__ == "__main__":
    main()
