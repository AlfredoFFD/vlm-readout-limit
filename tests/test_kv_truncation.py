"""CPU unit tests for the KV-truncation runner's image-token span detection.

The span detector is the load-bearing, GPU-independent part of the literal
KV-cache truncation experiment: it decides exactly which cached positions are
"the sample image" and therefore which positions get dropped. These tests use
synthetic token sequences (adversarial layouts) so they run with no model.

The mapping to real Qwen3-VL tokenization (image_token_id=151655, sample image =
108 contiguous tokens at positions [47..154] flanked by vision_start/end) was
verified manually against the processor; that wiring is exercised at runtime by
`kv_truncation_runner --validate` on a CUDA host.
"""

from vwm.local.kv_truncation_runner import find_image_token_spans

IMG = 151655  # Qwen3-VL config.image_token_id


def test_empty():
    assert find_image_token_spans([], IMG) == []


def test_no_image_tokens():
    assert find_image_token_spans([1, 2, 3, 4], IMG) == []


def test_single_contiguous_span():
    assert find_image_token_spans([1, IMG, IMG, IMG, 2], IMG) == [(1, 3)]


def test_span_at_start():
    assert find_image_token_spans([IMG, IMG, 1], IMG) == [(0, 1)]


def test_span_at_end():
    assert find_image_token_spans([1, IMG, IMG], IMG) == [(1, 2)]


def test_two_spans_sample_then_test():
    # The realistic case: sample image span first, test image span second,
    # separated by vision markers / prompt text tokens.
    ids = [0] * 47 + [IMG] * 108 + [1, 2, 3, 4] + [IMG] * 108 + [9, 9]
    spans = find_image_token_spans(ids, IMG)
    # sample: 47..154 (108 tokens); then 4 text tokens (155..158); test: 159..266.
    assert spans == [(47, 154), (159, 266)]
    # The SAMPLE image is always the first span -> the one we drop.
    assert spans[0] == (47, 154)


def test_separated_singletons_are_distinct_spans():
    # A non-image token between image tokens splits the run.
    assert find_image_token_spans([IMG, 5, IMG], IMG) == [(0, 0), (2, 2)]


def test_contiguous_run_is_one_span():
    # Qwen separates two images with vision_start/end, so a single contiguous run
    # is exactly one image -> one span (never silently merged across a gap).
    assert find_image_token_spans([IMG, IMG, IMG], IMG) == [(0, 2)]


def test_accepts_int_coercible_iterable():
    # tensor.tolist() yields plain ints; also tolerate numpy/int-like values.
    assert find_image_token_spans([float(IMG), float(IMG)], IMG) == [(0, 1)]
