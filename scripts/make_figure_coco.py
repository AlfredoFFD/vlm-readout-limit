"""FIGURE 1: COCO locate accuracy, five models x three conventions, Wilson CIs.

Reads the T5 block of results/TABLES.txt (itself regenerated from raw logs) so
the bars and intervals are exactly the numbers printed in the paper. The legend
sits above the axes so it can never overlap a bar.

Usage: .venv/Scripts/python.exe scripts/make_figure_coco.py
Writes paper/figs/coco_locate.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLES = Path("results/TABLES.txt").read_text(encoding="utf-8")
OUT = Path("paper/figs/coco_locate.pdf")
ORDER = ["Qwen3-VL-4B", "InternVL3-2B", "Qwen2.5-VL-7B", "Gemini", "GPT-4o"]
CONVS = [("cellname", "English cell name", "#3a3a3a"),
         ("norm1000", "normalized 0-999", "#8a8a8a"),
         ("pixel", "absolute pixels", "#d0d0d0")]
CELL = re.compile(r"([\d.]+) \[\s*([\d.]+),\s*([\d.]+)\]")


def t5():
    block = re.search(r"^T5 .*?\n=+\n(.*?)\n=+", TABLES, re.S | re.M).group(1)
    rows = {}
    for ln in block.splitlines()[1:]:
        cells = CELL.findall(ln)
        if len(cells) >= 3:
            name = ln[:ln.index(cells[0][0])].strip()
            rows[name] = [tuple(map(float, c)) for c in cells[:3]]  # cellname, normalized, pixel
    return rows


def main():
    rows = t5()
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 2.25))
    w = 0.26
    for j, (key, label, color) in enumerate(CONVS):
        xs = [i + (j - 1) * w for i in range(len(ORDER))]
        acc = [rows[m][j][0] for m in ORDER]
        lo = [rows[m][j][0] - rows[m][j][1] for m in ORDER]
        hi = [rows[m][j][2] - rows[m][j][0] for m in ORDER]
        ax.bar(xs, acc, w, yerr=[lo, hi], color=color, edgecolor="black", linewidth=0.5,
               capsize=2, error_kw={"linewidth": 0.8}, label=label)
    ax.axhline(100 / 9, color="black", linestyle="--", linewidth=0.8)
    # label sits outside the axes, right of the line, so it can never cover a bar
    ax.text(1.01, 100 / 9, "chance", fontsize=7, ha="left", va="center",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.4, columnspacing=1.6)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT, {m: [r[0] for r in rows[m]] for m in ORDER})


if __name__ == "__main__":
    main()
