"""ROBUSTNESS: turn each 'not verified' item into a verified result or a stated limit.

The audits establish that the numbers are what the runs produced. They do not
establish that the effect survives the obvious slicings of the data, and a reviewer
will ask for exactly those. Everything here is computed from data already collected,
so nothing can be tuned by running it again.

  1. OBJECT SIZE       does the gap depend on how big the target object is?
  2. BOUNDARY DISTANCE does it depend on how unambiguous the correct cell is?
  3. CATEGORY          is it driven by a handful of COCO classes?
  4. THE ANOMALY       Qwen2.5-VL scores 5.0% on normalised coordinates, below
                       chance. Transposition was already rejected; this tests all
                       eight dihedral transforms of the grid plus a shift family.

Usage: .venv/Scripts/python.exe scripts/robustness.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = {
    "Qwen3-VL-4B": "results/coco/convention_qwen3vl4b.jsonl",
    "Qwen2.5-VL-7B": "results/coco/convention_Qwen2_5-VL-7B-Instruct.jsonl",
    "InternVL3-2B": "results/coco/convention_InternVL3-2B-hf.jsonl",
    "gemini": "results/frontier/gemini-flash-latest.jsonl",
    "gpt-4o": "results/frontier/gpt-4o.jsonl",
}
CELLS = [(r, c) for r in range(3) for c in range(3)]


def load(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] \
        if p.exists() else []


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def gap(rows, keep=lambda r: True, a="cellname", b="pixel"):
    A = {r["trial_id"]: r["correct"] for r in rows
         if r["desc"] == a and keep(r) and "correct" in r}
    B = {r["trial_id"]: r["correct"] for r in rows
         if r["desc"] == b and keep(r) and "correct" in r}
    sh = sorted(set(A) & set(B))
    if len(sh) < 15:
        return None
    x = sum(1 for t in sh if A[t] and not B[t])
    y = sum(1 for t in sh if B[t] and not A[t])
    d = 100 * (np.mean([B[t] for t in sh]) - np.mean([A[t] for t in sh]))
    p = binomtest(min(x, y), x + y, 0.5).pvalue if (x + y) else 1.0
    return d, p, len(sh), 100 * np.mean([A[t] for t in sh]), 100 * np.mean([B[t] for t in sh])


def geom(rows):
    """Per-image object size fraction and distance of its centre from the nearest
    cell boundary, in units of cell width. Computed from the raw boxes."""
    g = {}
    for r in rows:
        if "bbox" not in r or r["trial_id"] in g:
            continue
        W, H = r["W"], r["H"]
        x0, y0, x1, y1 = r["bbox"]
        area = (x1 - x0) * (y1 - y0) / (W * H)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        fx, fy = cx / W * 3, cy / H * 3
        d = min(abs(fx - round(fx)), abs(fy - round(fy)))   # 0 = on a boundary
        g[r["trial_id"]] = (area, d)
    return g


def slice_report(title, rows, g, key, edges, labels):
    print(f"\n  {title}")
    print(f"    {'slice':<18}{'cellname':>10}{'pixel':>8}{'gap':>9}{'p':>10}{'n':>6}")
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        res = gap(rows, keep=lambda r: r["trial_id"] in g
                  and lo <= key(g[r["trial_id"]]) < hi)
        if res is None:
            print(f"    {lab:<18}{'too few':>43}")
            continue
        d, p, n, ca, pa = res
        print(f"    {lab:<18}{ca:>9.1f}%{pa:>7.1f}%{d:>+9.1f}{p:>10.1e}{n:>6}")


def main():
    base = load(MODELS["Qwen3-VL-4B"])
    g = geom(base)
    areas = sorted(a for a, _ in g.values())
    dists = sorted(d for _, d in g.values())
    a33, a66 = np.percentile(areas, [33, 66])
    d33, d66 = np.percentile(dists, [33, 66])

    hdr("1-2. DOES THE GAP DEPEND ON OBJECT SIZE OR ON HOW CLEAR THE ANSWER IS?")
    print("  If the gap only appeared for big or borderline objects it would be an")
    print("  artefact of the task rather than of the convention.")
    for m, f in MODELS.items():
        rows = [r for r in load(f) if r.get("direction") == "locate"]
        if not rows:
            continue
        print(f"\n  --- {m}")
        slice_report("by object size (fraction of the image)", rows, g,
                     lambda t: t[0], [0, a33, a66, 1.01],
                     [f"small <{a33:.3f}", "medium", f"large >{a66:.3f}"])
        slice_report("by distance of the centre from a cell boundary", rows, g,
                     lambda t: t[1], [0, d33, d66, 1.01],
                     [f"near edge <{d33:.2f}", "middling", f"well inside >{d66:.2f}"])

    hdr("3. IS THE GAP DRIVEN BY A FEW COCO CATEGORIES?")
    rows = [r for r in base if r.get("direction") == "locate"]
    cat = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r["desc"] in ("cellname", "pixel") and "correct" in r:
            i = 0 if r["desc"] == "cellname" else 1
            cat[r["category"]][i] += int(r["correct"])
            if i == 0:
                cat[r["category"]][2] += 1
    big = [(k, v) for k, v in cat.items() if v[2] >= 5]
    pos = sum(1 for _, v in big if (v[0] - v[1]) > 0)
    print(f"  {len(big)} categories with at least 5 images; the gap favours cell "
          f"names in {pos}/{len(big)} of them")
    worst = sorted(big, key=lambda kv: -(kv[1][0] - kv[1][1]))[:6]
    print(f"    {'category':<16}{'cellname':>10}{'pixel':>8}{'n':>5}")
    for k, v in worst:
        print(f"    {k:<16}{100*v[0]/v[2]:>9.0f}%{100*v[1]/v[2]:>7.0f}%{v[2]:>5}")
    res = gap(rows, keep=lambda r: r["category"] != "person")
    if res:
        d, p, n, ca, pa = res
        print(f"  excluding 'person' (the most frequent class): gap {d:+.1f}, "
              f"p={p:.1e}, n={n}")

    hdr("4. THE ANOMALY: Qwen2.5-VL at 5.0% on normalised coordinates")
    print("  Transposition was rejected earlier. Testing all eight dihedral")
    print("  transforms of the 3x3 grid, plus row and column shifts.")
    import importlib.util
    spec = importlib.util.spec_from_file_location("cc", "scripts/coco_convention.py")
    cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
    R = [r for r in load(MODELS["Qwen2.5-VL-7B"])
         if r.get("desc") == "norm1000" and r.get("direction") == "locate"
         and "argmax" in r]
    T = {
        "identity": lambda r, c: (r, c),
        "transpose": lambda r, c: (c, r),
        "flip rows": lambda r, c: (2 - r, c),
        "flip cols": lambda r, c: (r, 2 - c),
        "rot 180": lambda r, c: (2 - r, 2 - c),
        "rot 90": lambda r, c: (c, 2 - r),
        "rot 270": lambda r, c: (2 - c, r),
        "anti-transpose": lambda r, c: (2 - c, 2 - r),
        "row +1": lambda r, c: ((r + 1) % 3, c),
        "col +1": lambda r, c: (r, (c + 1) % 3),
    }
    print(f"    {'transform':<16}{'hit rate':>10}{'vs 11.1% chance':>18}")
    for nm, fn in T.items():
        hits = []
        for r in R:
            order = sorted(range(9), key=lambda j: cc.hsh(r["trial_id"], "opt", j))
            opts = [CELLS[j] for j in order]
            tgt = fn(*r["cell"])
            hits.append(r["argmax"] == opts.index(tgt))
        a, n = int(np.sum(hits)), len(hits)
        p = binomtest(a, n, 1 / 9, alternative="greater").pvalue
        flag = "  <-- explains it" if (a / n > 0.25 and p < 1e-3) else ""
        print(f"    {nm:<16}{100*a/n:>9.1f}%{p:>18.1e}{flag}")
    print("\n  A transform hitting well above chance would identify a systematic")
    print("  misreading. None above means the behaviour stays unexplained, and the")
    print("  paper reports it that way.")


if __name__ == "__main__":
    main()
