"""Position sampling for Luck-Vogel change-detection arrays.

Constraints (Colin Quirk / Awh lab canon, confirmed 2026-04-15):
- canvas 400 x 300 px
- square side ~55 px (1.5 degrees of visual angle at the canonical viewing distance)
- minimum center-to-center distance >= 92 px (2.5 degrees)
- max 2 items per quadrant

Algorithm: stratified slot-grid with bounded jitter.

  We use a fixed 4 x 2 lattice of slot centers (8 slots total) inside the 400 x 300
  canvas. Slot spacing is 100 px horizontal, 150 px vertical; both exceed the 92 px
  minimum, so any subset of slots respects the min-distance rule. The lattice is laid
  out so each quadrant contains exactly 2 slots, automatically satisfying the
  max-2-per-quadrant rule for any chosen set size n <= 8.

  Per trial, we pick `n` slots uniformly without replacement and apply a small uniform
  jitter (+/- MAX_JITTER_PX) in x and y. Jitter is bounded so the worst-case
  inter-slot distance after jitter is still >= MIN_DIST_PX.

  This is deterministic given the (seed, condition) RNG state — verifying
  determinism is part of the test suite. It also runs in O(n) (no rejection
  sampling) so dataset generation is fast.
"""

from __future__ import annotations

import random
from typing import Final

CANVAS_W: Final[int] = 400
CANVAS_H: Final[int] = 300
SQUARE_PX: Final[int] = 55
MIN_DIST_PX: Final[int] = 92            # ~2.5 degrees visual angle
MAX_PER_QUADRANT: Final[int] = 2
MAX_JITTER_PX: Final[int] = 4           # +/- per axis. With slot spacing 100, post-jitter min = 92.

# 4 x 2 slot lattice. Columns at x in {50, 150, 250, 350}, rows at y in {75, 225}.
# Adjacent horizontal spacing = 100 px, vertical = 150 px, diagonal ~180 px.
# Each quadrant (200 x 150 half-canvas) contains exactly two slots.
SLOTS: Final[tuple[tuple[int, int], ...]] = (
    (50,  75), (150,  75), (250,  75), (350,  75),     # top row    (quadrants 0,0,1,1)
    (50, 225), (150, 225), (250, 225), (350, 225),     # bottom row (quadrants 2,2,3,3)
)


def sample_positions(n: int, rng: random.Random) -> list[tuple[int, int]]:
    if n < 1:
        return []
    if n > len(SLOTS):
        raise ValueError(f"n={n} exceeds the {len(SLOTS)}-slot lattice capacity")
    chosen = rng.sample(list(SLOTS), n)
    j = MAX_JITTER_PX
    return [(cx + rng.randint(-j, j), cy + rng.randint(-j, j)) for cx, cy in chosen]
