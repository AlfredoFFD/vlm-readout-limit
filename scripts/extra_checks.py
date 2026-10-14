"""TWO REVIEWER-REQUESTED CHECKS THAT NEED NO NEW MODEL RUNS.

1. Gate sensitivity. The capability gate admits a model when its control
   condition is at least 20 points over chance. Which models would enter or leave
   at other thresholds, and does the set of models showing the penalty change?
   Margins are read from the generated tables (T4 word rung, T5 cell-name column).

2. Cross-instrument agreement. For the local models scored by first token, the
   crossmodel logs also hold a generate-and-parse answer for every visible trial.
   Per-trial agreement between the two readouts, by model and rung.

Usage: .venv/Scripts/python.exe scripts/extra_checks.py   (writes results/EXTRA.txt)
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

TABLES = Path("results/TABLES.txt").read_text(encoding="utf-8")

# which models show the penalty, per task (from T4 p-values and T5 gap CIs)
COLOR_PENALTY = {"Qwen3-VL-4B*", "InternVL3-2B-hf", "Qwen2_5-VL-7B-Instruct"}
COCO_PENALTY = {"Qwen3-VL-4B", "InternVL3-2B", "Gemini"}


def section(name):
    m = re.search(rf"^{name} .*?\n=+\n(.*?)\n=+", TABLES, re.S | re.M)
    return m.group(1)


def gate_lines():
    out = ["GATE SENSITIVITY  control margin over chance, per model", "-" * 78]
    # T4: model K chance word ...
    rows = []
    for ln in section("T4").splitlines()[1:]:
        m = re.match(r"(\S+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%", ln)
        if m:
            rows.append((m.group(1), float(m.group(4)) - float(m.group(3)), m.group(1) in COLOR_PENALTY))
    out.append("color task (word rung over 1/K chance):")
    for name, margin, pen in sorted(rows, key=lambda r: -r[1]):
        out.append(f"  {name:26} margin {margin:5.1f}  {'penalty' if pen else 'no penalty / null'}")
    pen_min = min(m for _, m, p in rows if p)
    others = [m for _, m, p in rows if not p]
    out.append(f"  set of penalty models unchanged for every threshold from 0 to {pen_min:.1f}; "
               f"non-penalty models enter or leave at {', '.join(f'{m:.1f}' for m in sorted(others))}")
    rows = []
    for ln in section("T5").splitlines()[1:]:
        m = re.match(r"(.+?)\s{2,}([\d.]+) \[", ln)
        if m and "8-bit" not in m.group(1):
            name = m.group(1).strip()
            rows.append((name, float(m.group(2)) - 100 / 9, name in COCO_PENALTY))
    out.append("COCO locate (cell-name accuracy over 1/9 chance):")
    for name, margin, pen in sorted(rows, key=lambda r: -r[1]):
        out.append(f"  {name:26} margin {margin:5.1f}  {'penalty' if pen else 'no penalty / null'}")
    pen_min = min(m for _, m, p in rows if p)
    others = [m for _, m, p in rows if not p]
    out.append(f"  set of penalty models unchanged for every threshold from 0 to {pen_min:.1f}; "
               f"non-penalty models leave at {', '.join(f'{m:.1f}' for m in sorted(others))}")
    return out


def concordance_lines():
    out = ["", "CROSS-INSTRUMENT AGREEMENT  first-token argmax vs generate-and-parse, visible trials", "-" * 78]
    for f in sorted(glob.glob("results/crossmodel/*_contrast.jsonl")):
        name = Path(f).name.replace("_contrast.jsonl", "")
        by = defaultdict(lambda: [0, 0])
        for r in map(json.loads, open(f, encoding="utf-8")):
            if not r.get("ft_available") or r.get("visible_offvocab"):
                continue
            gi, ft = r.get("visible_gen_idx"), r.get("visible_ft_argmax")
            if gi is None or ft is None:
                continue
            by[r["rung"]][0] += 1
            by[r["rung"]][1] += int(int(gi) == int(ft))
        for rung, (n, a) in sorted(by.items()):
            out.append(f"  {name:26} {rung:8} n={n:3d}  agree {100*a/n:5.1f}%")
    return out


def main():
    lines = gate_lines() + concordance_lines()
    txt = "\n".join(lines)
    print(txt)
    Path("results/EXTRA.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
