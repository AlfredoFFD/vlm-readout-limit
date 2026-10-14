"""Emit a publication-ready LaTeX K-table from the per-model K CSV.

Reads `results/figures/k_table.csv` (produced by `vwm.analysis.k_estimation`),
generates `paper/results_table.tex` that the preprint can pull in via `\input`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HEADER = r"""\begin{table}[h]
\centering
\small
\caption{Pashler's K with bootstrap 95\% CIs per (model, condition, set size).
H is hit rate, F false-alarm rate, K = N(H-F).}
\label{tab:k_table}
\begin{tabular}{lllrrrrrr}
\toprule
Provider & Model & Condition & N & Trials & H & F & K & 95\% CI \\
\midrule
"""

FOOTER = r"""\bottomrule
\end{tabular}
\end{table}
"""


def to_latex(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["condition", "provider", "model", "set_size"])
    rows: list[str] = []
    for _, r in df.iterrows():
        provider = str(r["provider"]).replace("_", r"\_")
        model = str(r["model"]).replace("_", r"\_")
        ci_low = float(r["K_low"]) if pd.notna(r["K_low"]) else float("nan")
        ci_high = float(r["K_high"]) if pd.notna(r["K_high"]) else float("nan")
        ci = f"[{ci_low:.2f}, {ci_high:.2f}]" if pd.notna(ci_low) else "n/a"
        rows.append(
            f"{provider} & {model} & {r['condition']} & {int(r['set_size'])} & "
            f"{int(r['n_trials'])} & {float(r['hit_rate']):.2f} & "
            f"{float(r['fa_rate']):.2f} & {float(r['K']):.2f} & {ci} \\\\"
        )
    body = "\n".join(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HEADER + body + "\n" + FOOTER, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("results/figures/k_table.csv"))
    p.add_argument("--out", type=Path, default=Path("paper/results_table.tex"))
    args = p.parse_args(argv)
    out = to_latex(args.csv, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
