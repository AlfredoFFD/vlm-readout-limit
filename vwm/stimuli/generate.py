"""Stimulus generator for Luck-Vogel change detection.

Generates two image variants per trial:
- `sample`: array of colored squares to be remembered.
- `test`:   either identical to sample (no-change) or with exactly one square's color
            changed to a *different* color drawn from the unused colors in the array.

For Condition C1 (Simultaneous) we render a *combined* image with sample on the left half
and test on the right half, separated by a thin divider line. For C2/C3 we use the two
separate images.

Trial manifest schema (`manifest_<condition>.json`):
{
  "seed": 20260521,
  "condition": "C1" | "C2" | "C3",
  "n_trials": 600,                 # 6 set sizes * 100 trials
  "trials": [
    {
      "trial_id": "C1_n4_t0042",
      "set_size": 4,
      "colors": ["red", "green", "blue", "yellow"],
      "positions": [[x,y], ...],
      "change": true,
      "changed_idx": 2,
      "new_color": "magenta",
      "sample_path": "stimuli/C1/sample_C1_n4_t0042.png",
      "test_path":   "stimuli/C1/test_C1_n4_t0042.png",
      "combined_path": "stimuli/C1/combined_C1_n4_t0042.png"
    }, ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .palette import COLORS, COLOR_NAMES, BACKGROUND_RGB
from .colorwheel import (
    make_wheel_trial,
    wheel_rgb,
    angular_distance,
    FOIL_DISTANCES_DEG,
)
from .geometry import (
    sample_positions,
    SQUARE_PX,
    CANVAS_W,
    CANVAS_H,
)

# 6 set sizes × 100 trials × 50% change rate (per PROJECT.md R-STIM-01)
SET_SIZES: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
TRIALS_PER_SET_SIZE: int = 100
DIVIDER_PX: int = 8     # gap between sample/test halves in combined image


@dataclass
class Trial:
    trial_id: str
    set_size: int
    colors: list[str]
    positions: list[tuple[int, int]]
    change: bool
    changed_idx: int                 # -1 if no change
    new_color: str                   # "" if no change
    sample_path: str = ""
    test_path: str = ""
    combined_path: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = [list(p) for p in self.positions]
        return d


# ---------- core trial construction ----------

def make_trial(
    trial_id: str,
    set_size: int,
    change: bool,
    rng: random.Random,
) -> Trial:
    """Build a single Trial spec (no rendering yet)."""
    if set_size < 1 or set_size > 8:
        raise ValueError(f"set_size {set_size} out of range [1, 8]")
    chosen = rng.sample(COLOR_NAMES, set_size)
    positions = sample_positions(set_size, rng)
    if change:
        changed_idx = rng.randrange(set_size)
        unused = [c for c in COLOR_NAMES if c not in chosen]
        new_color = rng.choice(unused)
    else:
        changed_idx = -1
        new_color = ""
    return Trial(
        trial_id=trial_id,
        set_size=set_size,
        colors=list(chosen),
        positions=positions,
        change=change,
        changed_idx=changed_idx,
        new_color=new_color,
    )


# ---------- rendering ----------

def _draw_array(img: Image.Image, colors: Iterable[str], positions: Iterable[tuple[int, int]]) -> None:
    draw = ImageDraw.Draw(img)
    half = SQUARE_PX // 2
    for name, (cx, cy) in zip(colors, positions):
        rgb = COLORS[name]
        draw.rectangle(
            [cx - half, cy - half, cx + half - 1, cy + half - 1],
            fill=rgb,
            outline=(64, 64, 64),
            width=1,
        )


def _make_canvas() -> Image.Image:
    return Image.new("RGB", (CANVAS_W, CANVAS_H), BACKGROUND_RGB)


def render_trial(trial: Trial, out_dir: Path, condition: str) -> Trial:
    """Render sample.png, test.png, and combined.png. Returns trial with paths set."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample
    sample_img = _make_canvas()
    _draw_array(sample_img, trial.colors, trial.positions)

    # Test (mutate one color if change trial)
    test_colors = list(trial.colors)
    if trial.change:
        test_colors[trial.changed_idx] = trial.new_color
    test_img = _make_canvas()
    _draw_array(test_img, test_colors, trial.positions)

    sample_path = out_dir / f"sample_{trial.trial_id}.png"
    test_path = out_dir / f"test_{trial.trial_id}.png"
    sample_img.save(sample_path, optimize=True)
    test_img.save(test_path, optimize=True)

    # Combined (for C1 Simultaneous): sample | divider | test, horizontally.
    combined_w = CANVAS_W * 2 + DIVIDER_PX
    combined = Image.new("RGB", (combined_w, CANVAS_H), (255, 255, 255))
    combined.paste(sample_img, (0, 0))
    combined.paste(test_img, (CANVAS_W + DIVIDER_PX, 0))
    # Black divider line
    div = ImageDraw.Draw(combined)
    div.rectangle(
        [CANVAS_W, 0, CANVAS_W + DIVIDER_PX - 1, CANVAS_H - 1],
        fill=(0, 0, 0),
    )
    combined_path = out_dir / f"combined_{trial.trial_id}.png"
    combined.save(combined_path, optimize=True)

    # Store paths *relative* to data/stimuli so manifests are portable
    base = Path(out_dir).as_posix()
    trial.sample_path = f"{base}/{sample_path.name}"
    trial.test_path = f"{base}/{test_path.name}"
    trial.combined_path = f"{base}/{combined_path.name}"
    return trial


# ---------- dataset generation ----------

def generate_dataset(
    out_root: Path,
    condition: str,
    seed: int,
    set_sizes: tuple[int, ...] = SET_SIZES,
    trials_per_set_size: int = TRIALS_PER_SET_SIZE,
) -> dict:
    """Generate the full trial set for one condition. Writes images + manifest JSON.

    Half of trials at each set size are change-trials (rounded).
    Deterministic given (seed, condition) — EXCEPT that C2 and C3 share one
    stimulus stream: C3 is the sample-withheld presentation of the SAME
    physical trials as C2, so the central C2-vs-C3 contrast is paired. (Fixed
    2026-08-07: the old per-condition seed made C2/C3 disjoint stimulus sets —
    584/600 trials differed — while ids and docs claimed pairing. Manifests
    generated before this fix are NOT paired across those conditions.)
    """
    seed_condition = "C2" if condition == "C3" else condition
    rng = random.Random(f"{seed}:{seed_condition}")
    out_dir = Path(out_root) / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: list[Trial] = []
    for n in set_sizes:
        half = trials_per_set_size // 2
        # First `half` are change=True, rest are change=False, then shuffled.
        change_flags = [True] * half + [False] * (trials_per_set_size - half)
        rng.shuffle(change_flags)
        for idx, change in enumerate(change_flags):
            tid = f"{condition}_n{n}_t{idx:04d}"
            trial = make_trial(tid, n, change, rng)
            trial = render_trial(trial, out_dir, condition)
            trials.append(trial)

    manifest = {
        "seed": seed,
        "condition": condition,
        "n_trials": len(trials),
        "set_sizes": list(set_sizes),
        "trials_per_set_size": trials_per_set_size,
        "schema_version": 1,
        "trials": [t.to_dict() for t in trials],
    }
    manifest_path = Path(out_root) / f"manifest_{condition}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"manifest_path": str(manifest_path), "n_trials": len(trials), "out_dir": str(out_dir)}


# ================================================================================
# CIELAB color-wheel design (manifest schema v2, design="wheel")
# ================================================================================
# Phase 0.5(a): a graded change-magnitude dataset over the gamut-safe CIELAB wheel
# (`colorwheel.py`). Replaces the 1-bit categorical recolor with parametric angular
# foils so miss-rate-vs-magnitude can adjudicate slot / variable-precision / TCC.
# Durable spec: research wf_f41b1976-1e4 (see RESEARCH_PLAN.md Phase 0.5).

# Set size 8 is dropped for the wheel design: 8 items each pairwise >=40 deg apart on a
# 360 deg wheel leaves only ~40 deg total slack, making both the sample and (especially)
# the rotated-test configuration geometrically marginal. The graded color-wheel WM
# literature (Zhang & Luck 2008; Bays; Schurgin) uses set sizes <=6 for the same reason.
WHEEL_SET_SIZES: tuple[int, ...] = (1, 2, 3, 4, 6)
WHEEL_FOIL_DISTANCES: tuple[int, ...] = FOIL_DISTANCES_DEG   # (15,30,60,90,135,180)
WHEEL_CATCH_RATE: float = 0.30                               # fraction of no-change trials
WHEEL_TRIALS_PER_CELL: int = 55                              # ~50-60 per (set_size, delta)
WHEEL_MIN_SAMPLE_SEP_DEG: float = 40.0                       # items individuable
WHEEL_OUTLINE_RGB: tuple[int, int, int] = (64, 64, 64)

# ---- M3 continuous-report (anchor-legend ring) geometry ----
WHEEL_REPORT_M: int = 12                                     # number of legend anchors (>=8-AFC)
LEGEND_H: int = 220                                          # legend panel height (below the array)
LEGEND_RING_R: int = 80                                      # radius of the anchor ring
LEGEND_DISC_R: int = 16                                      # radius of each anchor disc


@dataclass
class WheelDatasetTrial:
    trial_id: str
    set_size: int
    target_hues_deg: list[float]          # CANONICAL sample hues (pre-offset)
    positions: list[tuple[int, int]]
    change: bool
    changed_idx: int                      # -1 if no change (catch)
    delta_deg: float                      # 0.0 on catch trials
    wheel_zero_offset: int                # U{0..359}; applied at render time only
    test_hue_deg: float | None            # canonical hue of changed square; None on catch
    test_hues_deg: list[float]            # full canonical test array (for rendering)
    probe_idx: int                        # M3: which square the continuous-report cue marks
    sample_path: str = ""
    test_path: str = ""
    combined_path: str = ""
    report_path: str = ""

    def _changed_nn_gap(self) -> float | None:
        if not self.change or self.changed_idx < 0 or self.set_size < 2:
            return None
        from .colorwheel import angular_distance
        changed = self.test_hues_deg[self.changed_idx]
        others = [h for i, h in enumerate(self.test_hues_deg) if i != self.changed_idx]
        return round(min(angular_distance(changed, o) for o in others), 3)

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "set_size": self.set_size,
            "design": "wheel",
            "stim_space": "cielab_wheel",
            "target_hues_deg": [round(x, 3) for x in self.target_hues_deg],
            "positions": [list(p) for p in self.positions],
            "change": self.change,
            "changed_idx": self.changed_idx,
            "delta_deg": round(self.delta_deg, 3) if self.delta_deg else 0,
            "wheel_zero_offset": self.wheel_zero_offset,
            "test_hue_deg": (round(self.test_hue_deg, 3)
                             if self.test_hue_deg is not None else None),
            "test_hues_deg": [round(x, 3) for x in self.test_hues_deg],
            "probe_idx": self.probe_idx,
            "report_hue_deg": round(self.target_hues_deg[self.probe_idx], 3),
            # audit field (2026-08-07 review): the changed item's nearest-
            # neighbor gap in the TEST array — lets the analysis check that
            # min-gap construction did not correlate gap statistics with delta
            "changed_nn_gap_deg": self._changed_nn_gap(),
            "sample_path": self.sample_path,
            "test_path": self.test_path,
            "combined_path": self.combined_path,
            "report_path": self.report_path,
        }


def _draw_wheel_array(
    img: Image.Image,
    hues_deg: Iterable[float],
    positions: Iterable[tuple[int, int]],
    offset: int,
) -> None:
    """Render squares filled with the CIELAB-wheel color at (hue + offset) mod 360.

    The per-trial ``offset`` rotates the absolute colors so the model cannot anchor
    a stored canonical angle to a fixed perceptual category across trials.
    """
    draw = ImageDraw.Draw(img)
    half = SQUARE_PX // 2
    for hue, (cx, cy) in zip(hues_deg, positions):
        rgb = wheel_rgb((float(hue) + offset) % 360.0)
        draw.rectangle(
            [cx - half, cy - half, cx + half - 1, cy + half - 1],
            fill=rgb,
            outline=WHEEL_OUTLINE_RGB,
            width=1,
        )


def render_wheel_trial(trial: WheelDatasetTrial, out_dir: Path) -> WheelDatasetTrial:
    """Render sample/test/combined PNGs for one wheel trial. Sets *_path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    off = trial.wheel_zero_offset

    sample_img = _make_canvas()
    _draw_wheel_array(sample_img, trial.target_hues_deg, trial.positions, off)
    test_img = _make_canvas()
    _draw_wheel_array(test_img, trial.test_hues_deg, trial.positions, off)

    sample_path = out_dir / f"sample_{trial.trial_id}.png"
    test_path = out_dir / f"test_{trial.trial_id}.png"
    sample_img.save(sample_path, optimize=True)
    test_img.save(test_path, optimize=True)

    combined_w = CANVAS_W * 2 + DIVIDER_PX
    combined = Image.new("RGB", (combined_w, CANVAS_H), (255, 255, 255))
    combined.paste(sample_img, (0, 0))
    combined.paste(test_img, (CANVAS_W + DIVIDER_PX, 0))
    ImageDraw.Draw(combined).rectangle(
        [CANVAS_W, 0, CANVAS_W + DIVIDER_PX - 1, CANVAS_H - 1], fill=(0, 0, 0)
    )
    combined_path = out_dir / f"combined_{trial.trial_id}.png"
    combined.save(combined_path, optimize=True)

    base = out_dir.as_posix()
    trial.sample_path = f"{base}/{sample_path.name}"
    trial.test_path = f"{base}/{test_path.name}"
    trial.combined_path = f"{base}/{combined_path.name}"
    return trial


# ---------- M3: continuous-report probe (anchor-legend ring + scorer) ----------

def legend_anchor_canonical_hue(k: int, M: int = WHEEL_REPORT_M) -> float:
    """Canonical (pre-offset) hue of legend anchor ``k`` (0-based), evenly spaced."""
    return 360.0 * (k % M) / M


def generate_swap_dataset(
    out_root: Path | str,
    seed: int = 20260521,
    set_sizes: tuple[int, ...] = (2, 3, 4, 6),
    trials_per_cell: int = 55,
    catch_rate: float = 0.3,
    render: bool = True,
    min_swap_sep_deg: float = 40.0,
) -> dict:
    """SWAP-FOIL dataset (PLAN_TOPTIER P2 keystone): change trials exchange two
    squares' colors (color SET preserved -> only binding detects); catch trials
    are identical arrays. Runner-compatible with the wheel change task
    (design=wheel); ``delta_deg`` on swap trials = angular distance between the
    two swapped colors (the discriminability of the swap, analogous to foil
    distance). Deterministic given ``seed``.
    """
    from .colorwheel import make_swap_trial, angular_distance

    rng = random.Random(f"{seed}:swap")
    out_root = Path(out_root)
    out_dir = out_root / "swap"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    def build(trial_id: str, n: int, change: bool) -> dict:
        positions = sample_positions(n, rng)
        st = make_swap_trial(trial_id, n, rng, positions,
                             min_swap_sep_deg=min_swap_sep_deg, change=change)
        offset = rng.randrange(360)
        if change:
            i, j = st.swapped_pair
            delta = angular_distance(st.angles_deg[i], st.angles_deg[j])
        else:
            delta = 0.0
        row = {
            "trial_id": trial_id, "set_size": n, "design": "wheel",
            "stim_space": "cielab_wheel", "condition": "swap",
            "target_hues_deg": [round(x, 3) for x in st.angles_deg],
            "test_hues_deg": [round(x, 3) for x in st.test_angles_deg],
            "positions": [list(p) for p in st.positions],
            "change": change, "swap": bool(change),
            "swapped_pair": list(st.swapped_pair),
            "delta_deg": round(delta, 3), "changed_idx": -1,
            "wheel_zero_offset": offset, "probe_idx": 0,
            "report_hue_deg": round(st.angles_deg[0], 3),
        }
        if render:
            sample_img = _make_canvas()
            _draw_wheel_array(sample_img, st.angles_deg, st.positions, offset)
            test_img = _make_canvas()
            _draw_wheel_array(test_img, st.test_angles_deg, st.positions, offset)
            sp = out_dir / f"sample_{trial_id}.png"
            tp = out_dir / f"test_{trial_id}.png"
            sample_img.save(sp, optimize=True)
            test_img.save(tp, optimize=True)
            # combined (side-by-side) so the SIMULTANEOUS perception gate can
            # run on swap trials too (cross-model review requirement)
            combined_w = CANVAS_W * 2 + DIVIDER_PX
            combined = Image.new("RGB", (combined_w, CANVAS_H), (255, 255, 255))
            combined.paste(sample_img, (0, 0))
            combined.paste(test_img, (CANVAS_W + DIVIDER_PX, 0))
            ImageDraw.Draw(combined).rectangle(
                [CANVAS_W, 0, CANVAS_W + DIVIDER_PX - 1, CANVAS_H - 1], fill=(0, 0, 0))
            cp = out_dir / f"combined_{trial_id}.png"
            combined.save(cp, optimize=True)
            row["sample_path"] = f"{out_dir.as_posix()}/{sp.name}"
            row["test_path"] = f"{out_dir.as_posix()}/{tp.name}"
            row["combined_path"] = f"{out_dir.as_posix()}/{cp.name}"
        else:
            row["sample_path"] = row["test_path"] = row["combined_path"] = ""
        return row

    for n in set_sizes:
        for k in range(trials_per_cell):
            rows.append(build(f"swap_n{n}_t{k:04d}", n, True))
        n_catch = max(1, round(trials_per_cell * catch_rate / (1 - catch_rate)))
        for k in range(n_catch):
            rows.append(build(f"swap_n{n}_catch_t{k:04d}", n, False))

    # Deterministic shuffle so any --n-trials smoke/resume slice contains both
    # swap and catch trials (else no FA estimate; cross-model review finding).
    random.Random(f"{seed}:swap-shuffle").shuffle(rows)

    manifest = {
        "design": "wheel", "condition": "swap", "schema_version": 2,
        "seed": seed, "set_sizes": list(set_sizes),
        "trials_per_cell": trials_per_cell, "catch_rate": catch_rate,
        "min_swap_sep_deg": min_swap_sep_deg, "trials": rows,
    }
    mp = out_root / "manifest_swap.json"
    mp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {"n_trials": len(rows), "manifest_path": str(mp)}


def legend_ring_positions(
    M: int = WHEEL_REPORT_M,
    canvas_w: int = CANVAS_W,
    legend_top: int = CANVAS_H,
    legend_h: int = LEGEND_H,
    ring_r: int = LEGEND_RING_R,
) -> list[tuple[int, int]]:
    """Pixel centers of the M anchor discs, arranged in a ring (k=0 at top, clockwise)."""
    cx = canvas_w // 2
    cy = legend_top + legend_h // 2
    out: list[tuple[int, int]] = []
    for k in range(M):
        ang = 2.0 * math.pi * k / M - math.pi / 2.0
        out.append((round(cx + ring_r * math.cos(ang)), round(cy + ring_r * math.sin(ang))))
    return out


def score_report(reported_index: int, target_hue_deg: float, M: int = WHEEL_REPORT_M) -> float:
    """Circular report error (degrees) for a 1..M index answer.

    Offset-invariant: the legend is rendered with the same per-trial wheel offset as the
    squares, so both the reported anchor hue and ``target_hue_deg`` live in the canonical
    frame and the offset cancels. Error is the wrap-around angular distance in [0, 180].
    """
    if not (1 <= reported_index <= M):
        raise ValueError(f"reported_index {reported_index} out of range [1, {M}]")
    reported_hue = legend_anchor_canonical_hue(reported_index - 1, M)
    return angular_distance(reported_hue, target_hue_deg)


def render_report_probe(
    trial: WheelDatasetTrial, out_dir: Path, M: int = WHEEL_REPORT_M,
) -> WheelDatasetTrial:
    """Render the M3 report probe: a neutral (color-free) array with the probed square
    cued by a double black/white border ring, plus an M-anchor legend ring rotated by
    the trial's wheel_zero_offset. Sets ``report_path``. NO answer color is shown.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    H = CANVAS_H + LEGEND_H
    img = Image.new("RGB", (CANVAS_W, H), BACKGROUND_RGB)
    draw = ImageDraw.Draw(img)
    half = SQUARE_PX // 2

    # Neutral array: empty boxes (no item color -> no leak). Cue the probed square.
    for i, (cx, cy) in enumerate(trial.positions):
        draw.rectangle([cx - half, cy - half, cx + half - 1, cy + half - 1],
                       fill=BACKGROUND_RGB, outline=(80, 80, 80), width=2)
        if i == trial.probe_idx:
            draw.rectangle([cx - half - 2, cy - half - 2, cx + half + 1, cy + half + 1],
                           outline=(0, 0, 0), width=3)          # outer black
            draw.rectangle([cx - half + 2, cy - half + 2, cx + half - 3, cy + half - 3],
                           outline=(255, 255, 255), width=2)    # inner white

    draw.line([(0, CANVAS_H), (CANVAS_W, CANVAS_H)], fill=(0, 0, 0), width=1)

    # Anchor-legend ring (rotated WITH the trial offset; labels placed OUTSIDE the discs
    # so disc centers stay pure color for scoring/inspection).
    off = trial.wheel_zero_offset
    ring_cx, ring_cy = CANVAS_W // 2, CANVAS_H + LEGEND_H // 2
    for k, (lx, ly) in enumerate(legend_ring_positions(M)):
        rgb = wheel_rgb((legend_anchor_canonical_hue(k, M) + off) % 360.0)
        draw.ellipse([lx - LEGEND_DISC_R, ly - LEGEND_DISC_R,
                      lx + LEGEND_DISC_R, ly + LEGEND_DISC_R],
                     fill=rgb, outline=(0, 0, 0), width=1)
        dx, dy = lx - ring_cx, ly - ring_cy
        norm = math.hypot(dx, dy) or 1.0
        tx = round(lx + dx / norm * (LEGEND_DISC_R + 9))
        ty = round(ly + dy / norm * (LEGEND_DISC_R + 9))
        draw.text((tx, ty), str(k + 1), fill=(0, 0, 0), anchor="mm")

    report_path = out_dir / f"report_{trial.trial_id}.png"
    img.save(report_path, optimize=True)
    trial.report_path = f"{out_dir.as_posix()}/{report_path.name}"
    return trial


def _build_wheel_trial(
    trial_id: str, set_size: int, change: bool, delta_deg: float | None,
    rng: random.Random,
) -> WheelDatasetTrial:
    positions = sample_positions(set_size, rng)
    wt = make_wheel_trial(
        trial_id, set_size, change=change, rng=rng, positions=positions,
        change_distance_deg=delta_deg,
        min_sample_sep_deg=WHEEL_MIN_SAMPLE_SEP_DEG,
    )
    offset = rng.randrange(360)
    test_hue = wt.test_angles[wt.changed_idx] if change else None
    probe_idx = rng.randrange(set_size)
    return WheelDatasetTrial(
        trial_id=trial_id,
        set_size=set_size,
        target_hues_deg=list(wt.angles),
        positions=list(positions),
        change=change,
        changed_idx=wt.changed_idx,
        delta_deg=wt.change_distance_deg,
        wheel_zero_offset=offset,
        test_hue_deg=test_hue,
        test_hues_deg=list(wt.test_angles),
        probe_idx=probe_idx,
    )


def generate_wheel_dataset(
    out_root: Path,
    seed: int,
    set_sizes: tuple[int, ...] = WHEEL_SET_SIZES,
    trials_per_cell: int = WHEEL_TRIALS_PER_CELL,
    foil_distances: tuple[int, ...] = WHEEL_FOIL_DISTANCES,
    catch_rate: float = WHEEL_CATCH_RATE,
    render: bool = True,
) -> dict:
    """Generate the graded-magnitude CIELAB-wheel change-detection dataset.

    Structure: a full (set_size x delta) grid of ``trials_per_cell`` change trials,
    plus catch (no-change, delta=0) trials so catch ~= ``catch_rate`` per set size.
    Deterministic given ``seed``. Writes images (if ``render``) + manifest_wheel.json.
    """
    rng = random.Random(f"{seed}:wheel")
    out_dir = Path(out_root) / "wheel"
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: list[WheelDatasetTrial] = []
    for n in set_sizes:
        for d in foil_distances:
            for idx in range(trials_per_cell):
                tid = f"wheel_n{n}_d{int(d)}_t{idx:04d}"
                trials.append(_build_wheel_trial(tid, n, True, float(d), rng))
        n_change = len(foil_distances) * trials_per_cell
        n_catch = round(n_change * catch_rate / (1.0 - catch_rate))
        for idx in range(n_catch):
            tid = f"wheel_n{n}_catch_t{idx:04d}"
            trials.append(_build_wheel_trial(tid, n, False, None, rng))

    rng.shuffle(trials)   # deterministic given seed; de-blocks delta presentation order

    if render:
        for tr in trials:
            render_wheel_trial(tr, out_dir)
            render_report_probe(tr, out_dir)

    manifest = {
        "seed": seed,
        "condition": "wheel",
        "design": "wheel",
        "schema_version": 2,
        "n_trials": len(trials),
        "set_sizes": list(set_sizes),
        "trials_per_cell": trials_per_cell,
        "foil_distances_deg": list(foil_distances),
        "catch_rate": catch_rate,
        "wheel": {"L": 70.0, "center": [0.0, 0.0], "radius": 38.0, "n_hues": 360},
        "trials": [t.to_dict() for t in trials],
    }
    manifest_path = Path(out_root) / "manifest_wheel.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"manifest_path": str(manifest_path), "n_trials": len(trials), "out_dir": str(out_dir)}


# ---------- CLI ----------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Luck-Vogel change-detection stimuli.")
    p.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_SEED", "20260521")))
    p.add_argument("--out", type=Path, default=Path(os.environ.get("STIMULI_DIR", "./data/stimuli")))
    p.add_argument(
        "--conditions",
        type=str,
        default="C1,C2,C3",
        help="Comma-separated subset of {C1,C2,C3}. C2 and C3 reuse the same physical "
             "sample/test pairs; only the runner differs.",
    )
    p.add_argument("--trials-per-set-size", type=int, default=TRIALS_PER_SET_SIZE)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for cond in conds:
        if cond not in {"C1", "C2", "C3"}:
            raise SystemExit(f"Unknown condition '{cond}'")
        info = generate_dataset(
            out_root=args.out,
            condition=cond,
            seed=args.seed,
            trials_per_set_size=args.trials_per_set_size,
        )
        print(f"[{cond}] wrote {info['n_trials']} trials -> {info['out_dir']}")
        print(f"[{cond}] manifest -> {info['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
