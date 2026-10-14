"""Phase-0 KV-surgery core: object-level mask plans, selective rescue, and
value-only interventions for the post-encoding cache-ablation experiments.

Motivation (RESEARCH_PLAN.md, guardrail 6 + concurrent-work note): after
arXiv:2607.25467 (Chen et al. 2026, causal visual-KV audit) the full-span
masking collapse is replication; the contribution is the CONTROL MATRIX this
module makes possible:

  full-span          mask every sample image token          (validated baseline)
  target-only        mask only the changed square's patch tokens
  nontarget-matched  mask an equal COUNT of non-target patch tokens
  rescue-target      mask the span EXCEPT the target's tokens (selective restore)
  rescue-random      mask the span except an equal-count random subset
  value-zero         zero the V vectors at chosen positions/layers, keys intact
                     (attendable-but-contentless; dissociates from the attention
                     mask, which makes positions unattendable on the K side)

Everything in this file is CPU-testable (tests/test_kv_surgery.py). The GPU
forward that consumes these plans lives in kv_truncation_runner.py; hardware
validation checklist: results/M4_VALIDATION.md.

Geometry: Qwen-VL processors patchify the (resized) image into
``grid_thw = (t, h, w)`` patches of ``patch_size`` px, then spatially merge
``merge_size x merge_size`` patches into one token, row-major over the merged
grid. So token offset (relative to the image span) for merged cell (r, c) is
``r * (w // merge_size) + c``, and each cell covers a square of
``patch_size * merge_size`` px in the RESIZED image. Original-image pixel
coordinates are mapped through the resize scale. Row-major merge order is
asserted against real Qwen tokenization on the A100 (M4 checklist item)."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

PATCH_SIZE = 14
MERGE_SIZE = 2


# ------------------------------------------------------------------ geometry

def merged_grid_shape(grid_thw: Sequence[int],
                      merge_size: int = MERGE_SIZE) -> tuple[int, int]:
    """(rows, cols) of the merged token grid for a single-frame image."""
    _, h, w = grid_thw
    if h % merge_size or w % merge_size:
        raise ValueError(f"grid {grid_thw} not divisible by merge {merge_size}")
    return h // merge_size, w // merge_size


def bbox_to_token_offsets(bbox_xyxy: Sequence[float],
                          orig_size: Sequence[int],
                          grid_thw: Sequence[int],
                          patch_size: int = PATCH_SIZE,
                          merge_size: int = MERGE_SIZE) -> list[int]:
    """Token offsets (relative to the image span start, row-major, sorted)
    whose merged cells overlap ``bbox_xyxy`` = (x0, y0, x1, y1) in ORIGINAL
    image pixels. ``orig_size`` = (width, height) of the pre-processor image.

    The bbox is clamped to the image; a degenerate (zero-area after clamping)
    bbox raises ValueError rather than silently returning an empty mask.
    """
    x0, y0, x1, y1 = bbox_xyxy
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"degenerate bbox {bbox_xyxy}")
    ow, oh = orig_size
    rows, cols = merged_grid_shape(grid_thw, merge_size)
    rw = grid_thw[2] * patch_size          # resized width in px
    rh = grid_thw[1] * patch_size          # resized height in px
    cell = patch_size * merge_size

    # original -> resized coordinates, clamped into the image
    sx, sy = rw / ow, rh / oh
    rx0 = min(max(x0 * sx, 0.0), rw)
    rx1 = min(max(x1 * sx, 0.0), rw)
    ry0 = min(max(y0 * sy, 0.0), rh)
    ry1 = min(max(y1 * sy, 0.0), rh)
    if rx1 <= rx0 or ry1 <= ry0:
        # Entirely outside the image after resize: a silent edge-cell snap
        # would mask the WRONG tokens without any error. Fail loud instead.
        raise ValueError(f"bbox {bbox_xyxy} lies entirely outside the "
                         f"{ow}x{oh} image")

    c0 = int(rx0 // cell)
    c1 = min(int(math.ceil(rx1 / cell)) - 1, cols - 1)
    r0 = int(ry0 // cell)
    r1 = min(int(math.ceil(ry1 / cell)) - 1, rows - 1)
    return sorted(r * cols + c for r in range(r0, r1 + 1)
                  for c in range(c0, c1 + 1))


def infer_grid_from_span(n_tokens: int, orig_size: Sequence[int],
                         merge_size: int = MERGE_SIZE) -> tuple[int, int, int]:
    """Infer grid_thw = (1, h, w) in PATCH units from an image span's token
    count and the original image's aspect ratio (Qwen smart-resize roughly
    preserves aspect; each token = a merge_size^2 patch block, row-major).

    Needed because inside a drop_positions_fn only (ids1, span) are available
    — hardcoding a square sqrt-grid mis-maps every non-square image
    (cross-model review: C2 canvases are 400x300, not square)."""
    ow, oh = orig_size
    target_ratio = oh / ow
    best = None
    for h_m in range(1, n_tokens + 1):
        if n_tokens % h_m:
            continue
        w_m = n_tokens // h_m
        err = abs((h_m / w_m) - target_ratio)
        if best is None or err < best[0]:
            best = (err, h_m, w_m)
    if best is None:
        raise ValueError(f"cannot factor {n_tokens} tokens into a grid")
    _, h_m, w_m = best
    return (1, h_m * merge_size, w_m * merge_size)


def square_token_indices(center: Sequence[float], square_px: float,
                         orig_size: Sequence[int], grid_thw: Sequence[int],
                         span_start: int,
                         patch_size: int = PATCH_SIZE,
                         merge_size: int = MERGE_SIZE) -> list[int]:
    """ABSOLUTE sequence positions of the patch tokens covering a stimulus
    square drawn centered at ``center`` with side ``square_px`` (the manifest's
    `positions` + SQUARE_PX convention), given the image span's start index."""
    cx, cy = center
    half = square_px / 2.0
    offs = bbox_to_token_offsets((cx - half, cy - half, cx + half, cy + half),
                                 orig_size, grid_thw, patch_size, merge_size)
    return [span_start + o for o in offs]


# ---------------------------------------------------------------- conditions

def surgery_conditions(span: tuple[int, int], target_positions: Sequence[int],
                       rng: random.Random) -> dict[str, list[int]]:
    """The mask-control matrix: condition name -> sorted DROP positions
    (absolute sequence indices whose KV becomes unattendable).

    ``span`` is the inclusive (start, end) of the sample image span;
    ``target_positions`` are the target square's absolute token indices.
    Draws are reproducible from ``rng``.
    """
    s0, s1 = span
    span_set = set(range(s0, s1 + 1))
    tgt = set(target_positions)
    if not tgt or not tgt <= span_set:
        raise ValueError("target positions must be non-empty and inside the span")
    pool = sorted(span_set - tgt)
    if len(pool) < len(tgt):
        raise ValueError("span too small for a matched non-target draw")
    nontarget = rng.sample(pool, len(tgt))
    # rescue-random restores an equal COUNT of non-target tokens: drawing from
    # span-minus-target keeps the contrast exact (a draw containing target
    # tokens would partially BE the rescue-target condition).
    rescue_keep = rng.sample(pool, len(tgt))
    return {
        "full-span": sorted(span_set),
        "target-only": sorted(tgt),
        "nontarget-matched": sorted(nontarget),
        "rescue-target": sorted(span_set - tgt),
        "rescue-random": sorted(span_set - set(rescue_keep)),
    }


# ------------------------------------------------------------- attention row

def build_attention_row(cache_len: int, new_len: int,
                        drop_positions: Iterable[int]):
    """(1, cache_len + new_len) long-tensor attention mask with the drop
    positions zeroed. Drops must lie inside the cached prefix — the new turn's
    own tokens are never maskable this way."""
    import torch

    drops = sorted(set(int(p) for p in drop_positions))
    if drops and (drops[0] < 0 or drops[-1] >= cache_len):
        raise ValueError(f"drop positions {drops[:3]}..{drops[-3:]} outside "
                         f"cached prefix [0, {cache_len})")
    row = torch.ones((1, cache_len + new_len), dtype=torch.long)
    if drops:
        row[0, drops] = 0
    return row


# ------------------------------------------------------------- value zeroing

def _layer_kv_pairs(cache):
    """Yield (key_tensor, value_tensor) per layer for legacy tuple caches or
    transformers DynamicCache-likes (key_cache/value_cache lists)."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        yield from zip(cache.key_cache, cache.value_cache)
    else:
        for layer in cache:
            yield layer[0], layer[1]


def swap_value_positions(cache_dst, cache_src, positions: Sequence[int],
                         layers: Sequence[int] | None = None) -> None:
    """In-place: overwrite ``cache_dst``'s VALUE vectors at ``positions`` with
    ``cache_src``'s values at the SAME positions (keys untouched).

    The wrong-content rescue arm (assembly review): re-injecting a DIFFERENT
    trial's sample-object KV at the same positions separates content
    restoration from mere format/position restoration — if behavior recovers
    with wrong content, the rescue effect was structural, not informational.
    Both caches must share sequence layout (same prompt template + span)."""
    pos = sorted(set(int(p) for p in positions))
    dst_pairs = list(_layer_kv_pairs(cache_dst))
    src_pairs = list(_layer_kv_pairs(cache_src))
    if len(dst_pairs) != len(src_pairs):
        raise ValueError(f"layer count mismatch: {len(dst_pairs)} vs {len(src_pairs)}")
    chosen = range(len(dst_pairs)) if layers is None else layers
    for li in chosen:
        _, v_dst = dst_pairs[li]
        _, v_src = src_pairs[li]
        seq = min(v_dst.shape[2], v_src.shape[2])
        if pos and (pos[0] < 0 or pos[-1] >= seq):
            raise IndexError(f"position {pos[-1]} outside shared seq len {seq}")
        v_dst[:, :, pos, :] = v_src[:, :, pos, :]


def zero_value_positions(cache, positions: Sequence[int],
                         layers: Sequence[int] | None = None) -> None:
    """In-place: zero the VALUE vectors at ``positions`` (sequence axis) for
    the given ``layers`` (all layers if None), leaving keys untouched.

    This is the attendable-but-contentless intervention: attention structure
    over the sample survives, its content does not — the V-side counterpart to
    the K-side attention mask."""
    pos = sorted(set(int(p) for p in positions))
    pairs = list(_layer_kv_pairs(cache))
    chosen = range(len(pairs)) if layers is None else layers
    for li in chosen:
        _, v = pairs[li]
        seq = v.shape[2]
        if pos and (pos[0] < 0 or pos[-1] >= seq):
            raise IndexError(f"position {pos[-1]} outside cache seq len {seq}")
        v[:, :, pos, :] = 0
