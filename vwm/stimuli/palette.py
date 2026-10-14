"""Colin Quirk / Awh-lab 9-color palette.

Colors are the highly-discriminable set used in `colinquirk/PsychopyChangeDetection`
(Awh lab canonical implementation). All values are sRGB 8-bit. White is included as
a discriminable color; the canvas background is a neutral gray (128, 128, 128) so
white is visually distinct from background.
"""

from __future__ import annotations

from typing import Final

COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "red":     (255,   0,   0),
    "green":   (  0, 255,   0),
    "blue":    (  0,   0, 255),
    "yellow":  (255, 255,   0),
    "magenta": (255,   0, 255),
    "cyan":    (  0, 255, 255),
    "orange":  (255, 128,   0),
    "white":   (255, 255, 255),
    "black":   (  0,   0,   0),
}

COLOR_NAMES: Final[tuple[str, ...]] = tuple(COLORS.keys())

BACKGROUND_RGB: Final[tuple[int, int, int]] = (128, 128, 128)


def color_by_name(name: str) -> tuple[int, int, int]:
    if name not in COLORS:
        raise KeyError(f"Unknown color '{name}'. Known: {COLOR_NAMES}")
    return COLORS[name]
