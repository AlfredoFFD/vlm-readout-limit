"""CIELAB color-wheel stimuli with a GRADED change-magnitude axis.

The 9-color categorical palette (`palette.py`) collapses every theory-separating
signature to a single bit (change is always a large categorical recolor). To
adjudicate slot vs variable-precision vs rate-distortion vs TCC accounts we need a
*continuous* color space where "change magnitude" is a parametric, perceptually
meaningful angular distance -- the canonical Zhang & Luck (2008) / Bays / Schurgin
color wheel.

Construction (Zhang & Luck 2008 / Bays / Schurgin-Wixted-Brady 2020 lineage, with a
gamut fix verified by design-research `wf_f41b1976-1e4`): colors lie on a circle in
CIELAB at fixed lightness L* = 70, centered at (a*, b*) = (0, 0), radius 38, sampled
at 360 evenly spaced hue angles. We use r = 38 (NOT the literal Zhang-Luck r=49/60),
because r >= 49 at L* = 70 pushes 129+/360 points OUT of sRGB; clipping them destroys
the perceptual uniformity that TCC's parameter-free prediction depends on. r = 38 is
the largest radius keeping all 360 hues in-gamut. Change magnitude = angular distance.

Conversion is CIELAB(D65) -> XYZ -> linear sRGB -> gamma, clipped to gamut (some
wheel points at r=60 fall slightly outside sRGB and are clipped, as in the
canonical PsychoPy implementations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# ---- canonical wheel parameters (Zhang & Luck 2008) ----
WHEEL_L: Final[float] = 70.0
WHEEL_CENTER_A: Final[float] = 0.0
WHEEL_CENTER_B: Final[float] = 0.0
WHEEL_RADIUS: Final[float] = 38.0   # largest r at L*=70 with ALL 360 hues IN sRGB gamut
WHEEL_N: Final[int] = 360           # (verified 0/360 out-of-gamut; r>=49 clips -> kills TCC uniformity)

BACKGROUND_RGB: Final[tuple[int, int, int]] = (128, 128, 128)

# D65 reference white
_XN, _YN, _ZN = 95.047, 100.0, 108.883
# XYZ -> linear sRGB (IEC 61966-2-1), XYZ scaled to [0,1]
_M = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _lab_to_rgb(L: float, a: float, b: float) -> tuple[int, int, int]:
    """CIELAB(D65) -> sRGB 8-bit, gamut-clipped."""
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0          # 0.008856
    kappa = 24389.0 / 27.0         # 903.3

    def _f_inv(t: float) -> float:
        t3 = t ** 3
        return t3 if t3 > eps else (116.0 * t - 16.0) / kappa

    X = _f_inv(fx) * _XN
    Y = _f_inv(fy) * _YN
    Z = _f_inv(fz) * _ZN
    lin = _M @ (np.array([X, Y, Z]) / 100.0)
    lin = np.clip(lin, 0.0, 1.0)
    # gamma encode
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return tuple(int(round(c * 255)) for c in np.clip(srgb, 0.0, 1.0))


def wheel_rgb(angle_deg: float) -> tuple[int, int, int]:
    """sRGB for a hue at ``angle_deg`` on the canonical CIELAB wheel."""
    th = np.deg2rad(float(angle_deg))
    a = WHEEL_CENTER_A + WHEEL_RADIUS * np.cos(th)
    b = WHEEL_CENTER_B + WHEEL_RADIUS * np.sin(th)
    return _lab_to_rgb(WHEEL_L, a, b)


def wheel_lab(angle_deg: float) -> tuple[float, float, float]:
    th = np.deg2rad(float(angle_deg))
    return (WHEEL_L,
            WHEEL_CENTER_A + WHEEL_RADIUS * np.cos(th),
            WHEEL_CENTER_B + WHEEL_RADIUS * np.sin(th))


def angular_distance(a_deg: float, b_deg: float) -> float:
    """Smallest absolute angular distance in [0, 180]."""
    d = abs((float(a_deg) - float(b_deg)) % 360.0)
    return min(d, 360.0 - d)


def delta_e(a_deg: float, b_deg: float) -> float:
    """CIE76 Delta-E between two wheel hues (lightness equal, so it is the
    Euclidean chord distance in the a*b* plane)."""
    la = np.array(wheel_lab(a_deg))
    lb = np.array(wheel_lab(b_deg))
    return float(np.linalg.norm(la - lb))


# Graded change-magnitude foil set. 0 = catch (no real change among "change" trials
# would be degenerate, so 0 is reserved for no-change trials). The non-zero set
# spans near->far to expose magnitude-dependence of the miss rate (the
# resource-vs-slot signature). Refine from `wf_f41b1976-1e4` if it recommends a
# different grid.
FOIL_DISTANCES_DEG: Final[tuple[int, ...]] = (15, 30, 60, 90, 135, 180)


@dataclass
class WheelTrial:
    trial_id: str
    set_size: int
    angles: list[float]            # sample hue angle per square
    positions: list[tuple[int, int]]
    change: bool
    changed_idx: int               # -1 if no change
    change_distance_deg: float     # 0.0 if no change
    test_angles: list[float]       # sample with one rotated (== angles if no change)

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "set_size": self.set_size,
            "angles": [round(x, 3) for x in self.angles],
            "positions": self.positions,
            "change": self.change,
            "changed_idx": self.changed_idx,
            "change_distance_deg": round(self.change_distance_deg, 3),
            "test_angles": [round(x, 3) for x in self.test_angles],
            "stim_space": "cielab_wheel",
        }


def _draw_separated_hues(set_size: int, min_sep_deg: float, rng) -> list[float]:
    """Draw ``set_size`` hue angles whose circular pairwise distances are all
    >= ``min_sep_deg``.

    Uses a direct minimum-gap construction instead of rejection sampling: distribute
    the slack ``R = 360 - set_size * min_sep`` uniformly across the inter-point gaps,
    then lay points out with a random global rotation. This is O(n), never gets stuck
    (rejection can dead-end at n=6/sep=40 when an early prefix leaves no valid slot),
    and is uniform over valid circular arrangements. Because every *consecutive* gap
    is >= min_sep, every *pairwise* distance is too (non-adjacent points are farther)."""
    R = 360.0 - set_size * min_sep_deg
    if R < 0:
        raise RuntimeError(
            f"set_size {set_size} infeasible at sep {min_sep_deg} (need set_size*sep <= 360)"
        )
    cuts = sorted(rng.uniform(0.0, R) for _ in range(set_size - 1))
    excess: list[float] = []
    prev = 0.0
    for c in cuts:
        excess.append(c - prev)
        prev = c
    excess.append(R - prev)                       # set_size non-negative parts summing to R
    phi = rng.uniform(0.0, 360.0)                 # random global rotation
    angles: list[float] = []
    acc = phi
    for i in range(set_size):
        angles.append(acc % 360.0)
        acc += min_sep_deg + excess[i]            # gap to next point is min_sep + excess >= min_sep
    return angles


@dataclass
class SwapTrial:
    """Set-preserving swap: two squares EXCHANGE colors. The color multiset is
    identical between sample and test, so set-membership observers see 'no
    change'; only color-location binding detects it (PLAN_TOPTIER keystone)."""
    trial_id: str
    set_size: int
    angles_deg: list[float]
    positions: list[tuple[int, int]]
    change: bool                      # True for swap trials, False for catch
    swap: bool
    swapped_pair: tuple[int, int]     # (-1, -1) on catch trials
    test_angles_deg: list[float]


def make_swap_trial(
    trial_id: str,
    set_size: int,
    rng,
    positions: list[tuple[int, int]],
    min_sample_sep_deg: float = 40.0,
    min_swap_sep_deg: float = 40.0,
    change: bool = True,
) -> SwapTrial:
    """One swap-foil trial. Swapped colors are guaranteed >= min_swap_sep_deg
    apart so the swap is discriminable by ANY color-sensitive observer; the
    failure mode being tested is binding, not discrimination. Catch trials
    (change=False) use the same construction with no swap applied."""
    if set_size < 2:
        raise ValueError("swap trials need at least 2 items")
    for _attempt in range(2000):
        angles = _draw_separated_hues(set_size, min_sample_sep_deg, rng)
        pairs = [(i, j) for i in range(set_size) for j in range(i + 1, set_size)
                 if angular_distance(angles[i], angles[j]) >= min_swap_sep_deg]
        if not pairs:
            continue
        if not change:
            return SwapTrial(trial_id, set_size, angles, positions, False, False,
                             (-1, -1), list(angles))
        i, j = pairs[rng.randrange(len(pairs))]
        test = list(angles)
        test[i], test[j] = test[j], test[i]
        return SwapTrial(trial_id, set_size, angles, positions, True, True,
                         (i, j), test)
    raise RuntimeError(
        f"could not build swap trial at set_size={set_size}, "
        f"sep={min_sample_sep_deg}/{min_swap_sep_deg} after 2000 redraws")


def make_wheel_trial(
    trial_id: str,
    set_size: int,
    change: bool,
    rng,
    positions: list[tuple[int, int]],
    change_distance_deg: float | None = None,
    min_sample_sep_deg: float = 40.0,
) -> WheelTrial:
    """One graded-magnitude change-detection trial on the CIELAB wheel.

    Sample hues are drawn at least ``min_sample_sep_deg`` apart (so items are
    individuable). On change trials exactly one square is rotated by
    ``change_distance_deg`` (drawn from FOIL_DISTANCES_DEG if None) to a hue that is
    GUARANTEED to stay >= ``min_sample_sep_deg`` from every other item -- so the test
    array is as individuable as the sample and the change magnitude is not confounded
    by a near-collision. If a sample configuration admits no valid rotation we redraw
    it; if the (set_size, distance, sep) cell is genuinely infeasible we raise rather
    than silently emit a colliding trial.
    """
    if change_distance_deg is None:
        change_distance_deg = float(rng.choice(FOIL_DISTANCES_DEG))
    # Catch trials draw a PHANTOM distance and pass the SAME feasibility gate
    # as change trials (a valid rotation must exist) without applying it —
    # otherwise catch samples come from the unconditioned configuration
    # distribution while change samples are feasibility-conditioned, and the
    # miss-vs-delta axis is confounded by configuration statistics
    # (2026-08-07 assembly review).
    dist = float(change_distance_deg)

    for _attempt in range(2000):
        angles = _draw_separated_hues(set_size, min_sample_sep_deg, rng)

        # collect, per item, the rotated hues (+/- dist) that clear all OTHER items
        feasible: dict[int, list[float]] = {}
        for idx in range(set_size):
            base = angles[idx]
            others = [a for j, a in enumerate(angles) if j != idx]
            opts: list[float] = []
            for sign in (-1.0, 1.0):
                new_ang = (base + sign * dist) % 360.0
                if all(angular_distance(new_ang, o) >= min_sample_sep_deg for o in others):
                    opts.append(new_ang)
            # dedup (dist==180 makes +/- coincide)
            opts = list({round(o, 6): o for o in opts}.values())
            if opts:
                feasible[idx] = opts

        if feasible:
            if not change:
                # phantom passed -> accept the configuration, change nothing
                return WheelTrial(trial_id, set_size, angles, positions, False,
                                  -1, 0.0, list(angles))
            changed_idx = rng.choice(list(feasible.keys()))          # uniform over feasible items
            choices = feasible[changed_idx]
            new_ang = choices[rng.randrange(len(choices))]           # uniform over non-colliding signs
            test_angles = list(angles)
            test_angles[changed_idx] = new_ang
            return WheelTrial(trial_id, set_size, angles, positions, True,
                              changed_idx, dist, test_angles)

    raise RuntimeError(
        f"could not place change dist={dist} at set_size={set_size}, "
        f"sep={min_sample_sep_deg} after 2000 sample redraws (cell likely infeasible)"
    )
