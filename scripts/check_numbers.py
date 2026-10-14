"""EVERY DECIMAL IN THE PAPER MUST EXIST IN A REGENERATED RESULT FILE.

The reproducibility statement claims that no number in the paper is entered by
hand. This makes that claim checkable: extract every one-decimal number from the
main text (bibliography excluded) and confirm it appears in the output of the
table generator or the robustness scripts. A number that appears nowhere was
typed from memory and is a defect, whether or not it happens to be right.

Usage: .venv/Scripts/python.exe scripts/check_numbers.py
Exit 1 if any number is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TEX = Path("paper/main.tex")
SOURCES = ["results/TABLES.txt", "results/ROBUSTNESS.txt", "results/COORD.txt",
           "results/GRAN.txt", "results/AUDIT.txt", "results/AUDIT2.txt",
           "results/MISREAD.txt", "results/EXTRA.txt"]


def main():
    tex = TEX.read_text(encoding="utf-8")
    body = tex.split(r"\bibliography{refs}")[0]
    # appendix files pulled in with \input{...} are part of the manuscript too
    for name in re.findall(r"\\input\{([^}]+)\}", tex):
        body += (TEX.parent / (name if name.endswith(".tex") else name + ".tex")).read_text(encoding="utf-8")
    corpus = " ".join(Path(p).read_text(encoding="utf-8", errors="replace")
                      for p in SOURCES if Path(p).exists())
    nums = sorted(set(re.findall(r"(?<![\d.])(\d{1,3}\.\d)(?![\d])", body)))
    missing = [n for n in nums if n not in corpus]
    print(f"{len(nums)} distinct decimals in the paper body; "
          f"{len(nums) - len(missing)} present in regenerated result files")
    if missing:
        print("NOT FOUND:", missing)
        sys.exit(1)
    print("every decimal in the paper is reproduced by a script")


if __name__ == "__main__":
    main()
