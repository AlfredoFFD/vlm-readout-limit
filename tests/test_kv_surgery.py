"""CPU tests for the Phase-0 KV-surgery core (vwm/local/kv_surgery.py).

Everything here is geometry / plan / tensor logic that must be exactly right
BEFORE burning an A100 session. Hardware validation of the full path stays in
results/M4_VALIDATION.md (Colab checklist).
"""

from __future__ import annotations

import random

import pytest
import torch

from vwm.local.kv_surgery import (
    merged_grid_shape,
    bbox_to_token_offsets,
    square_token_indices,
    surgery_conditions,
    build_attention_row,
    zero_value_positions,
)


# ---------------------------------------------------------------- geometry --

# Canonical synthetic setup: original image 476x476, processor resize keeps it
# 476x476 -> grid_thw = (1, 34, 34) patches of 14px -> merged grid 17x17 cells
# of 28px -> 289 image tokens.
GRID = (1, 34, 34)
ORIG = (476, 476)


def test_merged_grid_shape():
    assert merged_grid_shape(GRID) == (17, 17)


def test_full_image_bbox_covers_all_tokens():
    offs = bbox_to_token_offsets((0, 0, 476, 476), ORIG, GRID)
    assert offs == list(range(17 * 17))


def test_single_aligned_cell_maps_to_one_token():
    # Cell (row 2, col 3) in the merged grid covers pixels x:[84,112) y:[56,84).
    offs = bbox_to_token_offsets((84, 56, 112, 84), ORIG, GRID)
    assert offs == [2 * 17 + 3]


def test_straddling_bbox_takes_all_overlapped_cells():
    # x:[80,116) overlaps merged cols 2,3,4 ; y:[50,86) overlaps rows 1,2,3.
    offs = bbox_to_token_offsets((80, 50, 116, 86), ORIG, GRID)
    expect = sorted(r * 17 + c for r in (1, 2, 3) for c in (2, 3, 4))
    assert offs == expect


def test_bbox_scaling_when_processor_resized():
    # Original 952x952 downscaled to the same 476x476 grid: pixel coords halve.
    offs_big = bbox_to_token_offsets((168, 112, 224, 168), (952, 952), GRID)
    offs_small = bbox_to_token_offsets((84, 56, 112, 84), ORIG, GRID)
    assert offs_big == offs_small


def test_out_of_bounds_bbox_is_clamped_not_crashed():
    offs = bbox_to_token_offsets((-10, -10, 20, 20), ORIG, GRID)
    assert offs == [0]  # partial overlap clamps into the top-left cell


def test_fully_outside_bbox_raises():
    with pytest.raises(ValueError):
        bbox_to_token_offsets((-100, -100, -50, -50), ORIG, GRID)
    with pytest.raises(ValueError):
        bbox_to_token_offsets((500, 500, 600, 600), ORIG, GRID)


def test_degenerate_bbox_raises():
    with pytest.raises(ValueError):
        bbox_to_token_offsets((100, 100, 100, 120), ORIG, GRID)


def test_square_token_indices_absolute_positions():
    # A stimulus square drawn centered at (100, 70) with side 40 -> bbox
    # (80, 50, 120, 90); span starts at sequence position 20.
    idx = square_token_indices(
        center=(100, 70), square_px=40, orig_size=ORIG, grid_thw=GRID,
        span_start=20,
    )
    rel = bbox_to_token_offsets((80, 50, 120, 90), ORIG, GRID)
    assert idx == [20 + o for o in rel]
    assert all(20 <= i < 20 + 289 for i in idx)


# --------------------------------------------------------------- conditions --

SPAN = (20, 308)  # inclusive, 289 tokens


def _target():
    return square_token_indices(center=(100, 70), square_px=40, orig_size=ORIG,
                                grid_thw=GRID, span_start=SPAN[0])


def test_surgery_conditions_shapes_and_disjointness():
    tgt = _target()
    conds = surgery_conditions(SPAN, tgt, rng=random.Random(0))
    span_set = set(range(SPAN[0], SPAN[1] + 1))
    tgt_set = set(tgt)

    assert set(conds["full-span"]) == span_set
    assert set(conds["target-only"]) == tgt_set
    # matched non-target control: same COUNT, drawn outside the target, inside span
    nt = set(conds["nontarget-matched"])
    assert len(nt) == len(tgt_set) and nt.isdisjoint(tgt_set) and nt <= span_set
    # rescue-target: mask everything EXCEPT the target
    assert set(conds["rescue-target"]) == span_set - tgt_set
    # rescue-random: same restored COUNT drawn from NON-target tokens, so the
    # target stays masked — the exact contrast for selective rescue.
    rr = set(conds["rescue-random"])
    assert len(rr) == len(span_set) - len(tgt_set) and rr <= span_set
    assert tgt_set <= rr                       # target is among the DROPPED
    assert (span_set - rr).isdisjoint(tgt_set)  # restored subset is non-target


def test_surgery_conditions_reproducible_by_seed():
    tgt = _target()
    a = surgery_conditions(SPAN, tgt, rng=random.Random(7))
    b = surgery_conditions(SPAN, tgt, rng=random.Random(7))
    assert a == b


def test_surgery_conditions_rejects_target_outside_span():
    with pytest.raises(ValueError):
        surgery_conditions(SPAN, [5], rng=random.Random(0))


# ----------------------------------------------------------- attention row --

def test_build_attention_row():
    row = build_attention_row(cache_len=10, new_len=4, drop_positions=[2, 3, 7])
    assert row.shape == (1, 14)
    assert row.dtype == torch.long
    assert row[0, [2, 3, 7]].tolist() == [0, 0, 0]
    assert int(row.sum()) == 14 - 3


def test_build_attention_row_rejects_drops_in_new_tokens():
    with pytest.raises(ValueError):
        build_attention_row(cache_len=10, new_len=4, drop_positions=[11])


# ------------------------------------------------------------ value zeroing --

def _fake_cache(n_layers=3, seq=12, heads=2, dim=4):
    torch.manual_seed(0)
    return [(torch.randn(1, heads, seq, dim), torch.randn(1, heads, seq, dim))
            for _ in range(n_layers)]


def test_zero_value_positions_zeroes_only_values_at_positions():
    cache = _fake_cache()
    before = [(k.clone(), v.clone()) for k, v in cache]
    zero_value_positions(cache, positions=[3, 5])
    for (k, v), (k0, v0) in zip(cache, before):
        assert torch.equal(k, k0)                       # keys untouched
        assert torch.all(v[:, :, [3, 5], :] == 0)       # dropped V zeroed
        keep = [i for i in range(v.shape[2]) if i not in (3, 5)]
        assert torch.equal(v[:, :, keep, :], v0[:, :, keep, :])


def test_zero_value_positions_layer_subset():
    cache = _fake_cache()
    before = [(k.clone(), v.clone()) for k, v in cache]
    zero_value_positions(cache, positions=[1], layers=[0, 2])
    assert torch.all(cache[0][1][:, :, 1, :] == 0)
    assert torch.equal(cache[1][1], before[1][1])       # layer 1 untouched
    assert torch.all(cache[2][1][:, :, 1, :] == 0)


def test_zero_value_positions_out_of_range_raises():
    cache = _fake_cache(seq=8)
    with pytest.raises(IndexError):
        zero_value_positions(cache, positions=[8])


def test_swap_value_positions_wrong_content_rescue():
    from vwm.local.kv_surgery import swap_value_positions
    dst = _fake_cache()
    src = _fake_cache()
    # regenerate src with different values
    torch.manual_seed(99)
    src = [(torch.randn_like(k), torch.randn_like(v)) for k, v in src]
    dst_before = [(k.clone(), v.clone()) for k, v in dst]
    swap_value_positions(dst, src, positions=[2, 4])
    for (k, v), (k0, v0), (_, vs) in zip(dst, dst_before, src):
        assert torch.equal(k, k0)                              # keys untouched
        assert torch.equal(v[:, :, [2, 4], :], vs[:, :, [2, 4], :])  # src content
        keep = [i for i in range(v.shape[2]) if i not in (2, 4)]
        assert torch.equal(v[:, :, keep, :], v0[:, :, keep, :])


def test_swap_value_positions_layer_mismatch_raises():
    from vwm.local.kv_surgery import swap_value_positions
    with pytest.raises(ValueError):
        swap_value_positions(_fake_cache(n_layers=3), _fake_cache(n_layers=2),
                             positions=[0])


def test_infer_grid_from_span_nonsquare():
    from vwm.local.kv_surgery import infer_grid_from_span
    # 400x300 C2 canvas smart-resized: e.g. 140 merged tokens -> 10x14 merged
    assert infer_grid_from_span(140, (400, 300)) == (1, 20, 28)
    # square image: 289 tokens -> 17x17 merged
    assert infer_grid_from_span(289, (476, 476)) == (1, 34, 34)
