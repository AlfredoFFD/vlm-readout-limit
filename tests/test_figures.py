"""End-to-end test: synthesize a fake results frame and verify figures render."""

from __future__ import annotations

import zlib

from pathlib import Path

import numpy as np
import pandas as pd

from vwm.figures.plots import (
    plot_psychometric_curves,
    plot_k_bar_chart,
    plot_c1_c2_delta,
    make_all,
)
from vwm.analysis.k_estimation import per_model_k_table
from vwm.utils.io import write_jsonl


def _synth_trial_results(
    model: str, provider: str, condition: str, k_target: float,
    set_sizes=(1, 2, 3, 4, 6, 8), trials_per_size=100, seed: int = 0,
) -> list[dict]:
    """Synthesize per-trial rows whose H/FA roughly produce K ~= k_target."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in set_sizes:
        # Implied H, FA from K = N*(H-F): pick FA=0.05, H = k_target/N + 0.05
        H = min(1.0, max(0.5, k_target / n + 0.05))
        FA = 0.05
        for t in range(trials_per_size):
            change = bool(t % 2)
            yes_p = H if change else FA
            parsed_yes = bool(rng.random() < yes_p)
            rows.append({
                "trial_id": f"{condition}_n{n}_t{t}",
                "set_size": n,
                "change": change,
                "changed_idx": 0 if change else -1,
                "new_color": "red" if change else "",
                "colors": ["red", "green", "blue", "yellow"][:n] + ["magenta"] * max(0, n - 4),
                "positions": [[10, 10]] * n,
                "condition": condition,
                "provider": provider,
                "model": model,
                "raw": "YES" if parsed_yes else "NO",
                "parsed_yes": parsed_yes,
                "parsable": True,
                "correct": parsed_yes == change,
                "latency_s": 1.0,
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "error": None,
                "timestamp": "2026-05-21T00:00:00Z",
            })
    return rows


def test_make_all_renders_three_figures(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # 3 fake models, K ~ 2.5 in C1 and ~1.8 in C2 (memory hit)
    models = [("openai", "gpt-4o", 2.5, 1.8),
              ("anthropic", "claude", 2.6, 2.0),
              ("gemini", "gemini-2.5-pro", 2.3, 1.5)]
    for provider, model, k1, k2 in models:
        write_jsonl(raw_dir / f"{provider}_C1.jsonl",
                    _synth_trial_results(model, provider, "C1", k1, seed=zlib.crc32(model.encode()) & 0xFFFF))
        write_jsonl(raw_dir / f"{provider}_C2.jsonl",
                    _synth_trial_results(model, provider, "C2", k2, seed=zlib.crc32(model.encode()) & 0xFFFF))

    figs_dir = tmp_path / "figs"
    paths = make_all(raw_dir, figs_dir, n_boot=200)
    for k in ("fig1", "fig2", "fig3"):
        assert k in paths, f"missing {k}"
        assert paths[k].exists()
    csv = figs_dir / "k_table.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    # Sanity: at N=8, gpt-4o K should be in a plausible band
    g = df[(df.model == "gpt-4o") & (df.condition == "C1") & (df.set_size == 8)]
    assert not g.empty
    assert 1.0 < g.iloc[0]["K"] < 4.5
