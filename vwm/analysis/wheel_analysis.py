"""Behavioral analysis for the CIELAB-wheel design (Tier-2 SDT / precision metrics).

Consumes the JSONL rows written by ``vwm.local.wheel_runner`` and produces the
falsifier-bearing summaries the research plan calls for:

- **miss-rate vs delta** per set size -- THE resource/TCC (smooth decline) vs slot
  (magnitude-independent step) signature.
- **curved SDT ROC vs the linear high-threshold chord** + AUC per set size; plus
  d'(N) and Cowan's K(N) (K reported but NOT trusted -- it bakes in the high-threshold
  slot model; see plan guardrail 5).
- **report circular precision(N)**: bias, circular SD, RMSE from the signed report
  error, and a Bays-style **swap rate** (report nearer a non-probed item than the
  target) when the manifest is provided.

Pure NumPy/stdlib; the model never enters here. Signed report error is recomputed from
``reported_index`` + ``report_hue_deg`` in the canonical (offset-invariant) frame.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .roc import roc_auc, roc_curve, high_threshold_roc, dprime
from ..stimuli.generate import legend_anchor_canonical_hue, WHEEL_REPORT_M


# ---------------- small circular helpers ----------------

def _signed_diff(a_deg: float, b_deg: float) -> float:
    """Signed circular difference a-b wrapped to (-180, 180]."""
    return ((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0


def _abs_diff(a_deg: float, b_deg: float) -> float:
    d = abs(float(a_deg) - float(b_deg)) % 360.0
    return min(d, 360.0 - d)


def _circular_stats(signed_errors_deg: list[float]) -> dict:
    """Circular mean (bias) and circular SD in degrees, plus RMSE."""
    if not signed_errors_deg:
        return {"n": 0, "bias_deg": None, "circular_sd_deg": None, "rmse_deg": None}
    th = [math.radians(e) for e in signed_errors_deg]
    C = sum(math.cos(t) for t in th) / len(th)
    S = sum(math.sin(t) for t in th) / len(th)
    R = math.hypot(C, S)
    bias = math.degrees(math.atan2(S, C))
    csd = float("inf") if R <= 0.0 else math.degrees(math.sqrt(-2.0 * math.log(R)))
    rmse = math.sqrt(sum(e * e for e in signed_errors_deg) / len(signed_errors_deg))
    return {"n": len(signed_errors_deg), "bias_deg": bias,
            "circular_sd_deg": csd, "rmse_deg": rmse}


def _trapezoid(fpr, tpr) -> float:
    return float(sum((fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2.0
                     for i in range(len(fpr) - 1)))


# ---------------- change task ----------------

def miss_rate_by_delta(rows: Iterable[dict],
                       task: str = "change") -> dict[int, dict[float, float]]:
    """{set_size: {delta_deg: miss_rate}} over parsable change trials of the
    given task (miss = said NO). ``task`` may be "change" (sequential) or
    "simultaneous" (perception baseline) — both write the same row schema."""
    miss: dict[int, dict[float, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("task") != task or not r.get("change"):
            continue
        if r.get("parsed_yes") is None:
            continue
        miss[r["set_size"]][r["delta_deg"]].append(0 if r["parsed_yes"] else 1)
    return {n: {d: sum(v) / len(v) for d, v in by_d.items()}
            for n, by_d in miss.items()}


def detection_by_set_size(rows: Iterable[dict], task: str = "change") -> dict[int, dict]:
    """{set_size: {hit, fa, d_prime, k, n_change, n_nochange, auc, auc_ht, curvature}}.

    hit/fa come from the hard YES/NO; AUC/curvature from graded P(YES) when present.
    curvature = empirical AUC - high-threshold-chord AUC (>0 => curved/SDT-like).
    ``task``: "change" (sequential) or "simultaneous" (perception baseline)."""
    rows = [r for r in rows if r.get("task") == task]
    out: dict[int, dict] = {}
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_n[r["set_size"]].append(r)

    for n, rs in by_n.items():
        chg = [r for r in rs if r.get("change") and r.get("parsed_yes") is not None]
        noc = [r for r in rs if not r.get("change") and r.get("parsed_yes") is not None]
        n_chg, n_noc = len(chg), len(noc)
        hit = sum(1 for r in chg if r["parsed_yes"]) / n_chg if n_chg else None
        fa = sum(1 for r in noc if r["parsed_yes"]) / n_noc if n_noc else None

        d_prime = k = k_pashler = None
        if hit is not None and fa is not None:
            d_prime = dprime(hit, fa, n_signal=n_chg, n_noise=n_noc)
            k = n * (hit - fa)
            # Whole-display task -> Pashler's form N(H-F)/(1-F) is the
            # appropriate estimator (Rouder et al. 2011); Cowan's N(H-F) is
            # kept for cross-paper comparability but is single-probe-derived.
            # Both remain descriptive only (guardrail 5: SDT-ROC is primary).
            k_pashler = n * (hit - fa) / (1 - fa) if fa < 1 else None

        # graded ROC from P(YES). curvature compares the empirical AUC against
        # the high-threshold chord built from hit/fa — those two statistics MUST
        # come from the SAME trial population, so restrict the AUC to rows that
        # are BOTH graded (p_yes) and parsed (parsed_yes); otherwise parse
        # failures (which plausibly cluster on hard trials) bias the
        # slot-vs-SDT discriminator. auc_all is reported separately for
        # completeness over every graded row.
        ps, ys, ps_all, ys_all = [], [], [], []
        for r in rs:
            if r.get("p_yes") is None:
                continue
            ps_all.append(r["p_yes"])
            ys_all.append(bool(r.get("change")))
            if r.get("parsed_yes") is not None:
                ps.append(r["p_yes"])
                ys.append(bool(r.get("change")))
        auc = auc_ht = curvature = auc_all = None
        if ps_all and any(ys_all) and not all(ys_all):
            auc_all = roc_auc(ps_all, ys_all)
        if ps and any(ys) and not all(ys):
            auc = roc_auc(ps, ys)
            if hit is not None and fa is not None:
                hf, ht = high_threshold_roc(hit, fa)
                auc_ht = _trapezoid(hf, ht)
                curvature = auc - auc_ht

        out[n] = {
            "hit": hit, "fa": fa, "d_prime": d_prime, "k": k,
            "k_pashler": k_pashler,
            "n_change": n_chg, "n_nochange": n_noc,
            "auc": auc, "auc_all": auc_all, "auc_ht": auc_ht, "curvature": curvature,
        }
    return out


def auc_by_delta(rows: Iterable[dict]) -> dict[int, dict[float, float]]:
    """{set_size: {delta_deg: AUC}} — delta-conditioned ROCs (assembly review):
    each change-magnitude cell scored against the SAME no-change trials of that
    set size, over rows that are both graded and parsed. Pooled-delta ROCs mix
    heterogeneous signal strengths and can fake curvature; report both."""
    by_n: dict[int, dict] = defaultdict(lambda: {"noise": [], "signal": defaultdict(list)})
    for r in rows:
        if r.get("task") != "change" or r.get("p_yes") is None \
                or r.get("parsed_yes") is None:
            continue
        cell = by_n[r["set_size"]]
        if r.get("change"):
            cell["signal"][r["delta_deg"]].append(r["p_yes"])
        else:
            cell["noise"].append(r["p_yes"])
    out: dict[int, dict[float, float]] = {}
    for n, cell in by_n.items():
        if not cell["noise"]:
            continue
        out[n] = {}
        for d, ps in sorted(cell["signal"].items()):
            if ps:
                labels = [True] * len(ps) + [False] * len(cell["noise"])
                out[n][d] = roc_auc(ps + cell["noise"], labels)
    return out


def roc_points_by_set_size(rows: Iterable[dict]) -> dict[int, dict]:
    """{set_size: {fpr, tpr, thresholds}} for plotting, from graded P(YES).
    Same trial population as detection_by_set_size (graded AND parsed), so the
    plotted curve and the high-threshold chord are comparable."""
    by_n: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r in rows:
        if r.get("task") == "change" and r.get("p_yes") is not None \
                and r.get("parsed_yes") is not None:
            ps, ys = by_n[r["set_size"]]
            ps.append(r["p_yes"])
            ys.append(bool(r.get("change")))
    out: dict[int, dict] = {}
    for n, (ps, ys) in by_n.items():
        if ps and any(ys) and not all(ys):
            fpr, tpr, thr = roc_curve(ps, ys)
            out[n] = {"fpr": list(fpr), "tpr": list(tpr), "thresholds": list(thr)}
    return out


# ---------------- report task ----------------

def report_signed_errors(rows: Iterable[dict]) -> dict[int, list[float]]:
    """{set_size: [signed report error in (-180,180]]} recomputed in the canonical frame
    from reported_index + report_hue_deg (offset-invariant). Public for plotting."""
    errs: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("task") != "report" or r.get("reported_index") is None:
            continue
        reported = legend_anchor_canonical_hue(r["reported_index"] - 1, WHEEL_REPORT_M)
        errs[r["set_size"]].append(_signed_diff(reported, r["report_hue_deg"]))
    return errs


def report_error_by_set_size(rows: Iterable[dict]) -> dict[int, dict]:
    """{set_size: {n, bias_deg, circular_sd_deg, rmse_deg}} from signed report error."""
    return {n: _circular_stats(es) for n, es in report_signed_errors(rows).items()}


def report_swap_rate(rows: Iterable[dict], manifest: dict) -> dict[int, float]:
    """{set_size: swap_rate}: fraction of reports closer to a NON-probed item's hue than
    to the probed item's hue (Bays-style misbinding signature). Needs the manifest to
    recover every item's canonical hue per trial."""
    by_id = {t["trial_id"]: t for t in manifest.get("trials", [])}
    counts: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("task") != "report" or r.get("reported_index") is None:
            continue
        tr = by_id.get(r.get("trial_id"))
        if tr is None or tr["set_size"] < 2:
            continue
        reported = legend_anchor_canonical_hue(r["reported_index"] - 1, WHEEL_REPORT_M)
        hues = tr["target_hues_deg"]
        probe = tr["probe_idx"]
        d_target = _abs_diff(reported, hues[probe])
        d_others = min(_abs_diff(reported, h) for j, h in enumerate(hues) if j != probe)
        counts[r["set_size"]].append(1 if d_others < d_target else 0)
    return {n: sum(v) / len(v) for n, v in counts.items() if v}


def report_swap_permutation_null(rows: Iterable[dict], manifest: dict,
                                 n_perm: int = 2000, seed: int = 0
                                 ) -> dict[int, dict]:
    """Chance baseline for report_swap_rate (assembly review): at N=6 a pure
    GUESSER lands closer to some non-probed item ~(N-1)/N of the time, so a
    raw swap rate means nothing without this null. For each set size, draw
    ``n_perm`` uniform-random legend indices per real report and recompute the
    swap statistic -> {set_size: {observed, null_mean, null_sd, z, p_greater}}.
    A MISBINDING claim needs observed >> null? No — the diagnostic direction
    is the opposite: a memory-driven reporter has swap rate BELOW the guesser
    null (it usually hits the target); Bays-style misbinding shows up as a
    swap rate above the ERROR-MATCHED expectation, so also compare against
    reports rotated by each trial's signed error magnitude before claiming
    misbinding (p_greater is versus the uniform-guess null only)."""
    import random as _random

    from ..stimuli.generate import WHEEL_REPORT_M, legend_anchor_canonical_hue

    rng = _random.Random(seed)
    by_id = {t["trial_id"]: t for t in manifest.get("trials", [])}
    trials_by_n: dict[int, list[tuple[list[float], int]]] = defaultdict(list)
    observed = report_swap_rate(rows, manifest)
    for r in rows:
        if r.get("task") != "report" or r.get("reported_index") is None:
            continue
        tr = by_id.get(r.get("trial_id"))
        if tr is None or tr["set_size"] < 2:
            continue
        trials_by_n[r["set_size"]].append((tr["target_hues_deg"], tr["probe_idx"]))

    out: dict[int, dict] = {}
    for n, trials in trials_by_n.items():
        if n not in observed or not trials:
            continue
        null_rates = []
        for _p in range(n_perm):
            hits = 0
            for hues, probe in trials:
                reported = legend_anchor_canonical_hue(
                    rng.randrange(WHEEL_REPORT_M), WHEEL_REPORT_M)
                d_t = _abs_diff(reported, hues[probe])
                d_o = min(_abs_diff(reported, h)
                          for j, h in enumerate(hues) if j != probe)
                hits += 1 if d_o < d_t else 0
            null_rates.append(hits / len(trials))
        mu = sum(null_rates) / len(null_rates)
        var = sum((x - mu) ** 2 for x in null_rates) / max(len(null_rates) - 1, 1)
        sd = var ** 0.5
        obs = observed[n]
        out[n] = {
            "observed": obs, "null_mean": mu, "null_sd": sd,
            "z": (obs - mu) / sd if sd > 0 else None,
            "p_greater": sum(1 for x in null_rates if x >= obs) / len(null_rates),
        }
    return out


# ---------------- bundles ----------------

def summarize_change(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    return {"by_delta": miss_rate_by_delta(rows),
            "by_set_size": detection_by_set_size(rows),
            "roc": roc_points_by_set_size(rows)}


def summarize_report(rows: Iterable[dict], manifest: dict | None = None) -> dict:
    rows = list(rows)
    out = {"by_set_size": report_error_by_set_size(rows)}
    if manifest is not None:
        out["swap_rate"] = report_swap_rate(rows, manifest)
    return out
