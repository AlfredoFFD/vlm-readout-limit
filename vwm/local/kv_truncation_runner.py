"""Literal post-encoding KV-cache truncation runner (open models only).

THE decisive storage-vs-retrieval experiment, and the one the paper's "KV-cache
ablation" framing actually requires. It differs from C3 (`c3_runner.py`), which
merely *withholds* the sample. Here we:

  Phase 1 -- present the sample (turn 1) so the model genuinely ENCODES it and
            the assistant acknowledgment ("READY") attends over it. We run a
            forward pass with use_cache=True and keep the resulting KV cache.
  Surgery -- drop ONLY the sample image's cached key/value positions, KEEPING
            the acknowledgment and the system / sample-prompt text positions.
            (Two methods: 'mask' = make those key positions un-attendable in the
            test-turn forward; 'remove' = physically drop them via
            DynamicCache.batch_select_indices.)
  Phase 2 -- present the test turn and generate the YES/NO answer on the edited
            cache.

Interpretation:
  * If K survives the surgery -> sample information was *smuggled forward* into the
    retained acknowledgment/context tokens during encoding (a form of storage).
  * If K collapses to ~0 like C3 -> the representation lived entirely in the
    now-dropped sample positions (pure in-context retrieval).

============================ VALIDATION STATUS ============================
The sample image-token SPAN DETECTION (`find_image_token_spans`) is unit-tested
on CPU (tests/test_kv_truncation.py) and verified against real Qwen3-VL
tokenization (sample span = contiguous run of config.image_token_id between
vision_start/vision_end).

VALIDATED (method='mask') on a Colab A100 with Qwen3-VL-4B (transformers 5.10.1),
2026-06-04: across 16 mixed-set-size trials the sample-VISIBLE control scored
16/16 (100%) change detection while the sample-DROPPED condition (sample image-
token KV masked, acknowledgment retained) collapsed to 9/16 (~56%, chance),
answering "NO" on 15/16 trials. The surgery changed the output on 7/16 trials.
This reproduces the C2->C3 retrieval-not-storage collapse via a genuine post-
encoding cache ablation -- the sample information did not survive in the retained
acknowledgment tokens. (The 'remove' / physical-crop method is NOT yet validated;
prefer 'mask'.) The `--validate` mode re-runs these sample-visible vs. sample-
dropped controls; do not report sweep numbers from a run whose controls did not
pass. The full canonical sweep (600 trials x 6 set sizes, seed-20260521 stimuli,
both Qwens) is the remaining step for publication-grade numbers.
=========================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ..utils import (
    load_manifest,
    append_jsonl,
    ensure_dir,
    SYSTEM_INSTRUCTION,
    PROMPT_C2_SEQUENTIAL_TURN1,
    PROMPT_C2_SEQUENTIAL_TURN2,
)
from ..utils.parsing import normalize_response
from .qwen3_vl import Qwen3VLClient
from .qwen25_vl import Qwen25VLClient


LOCAL_REGISTRY = {
    "qwen3-vl-4b": Qwen3VLClient,
    "qwen2.5-vl-7b": Qwen25VLClient,
}

FIXED_ACK = "READY"


# --------------------------------------------------------------------------
# CPU-testable: locate image-token spans in a tokenized sequence.
# --------------------------------------------------------------------------

def find_image_token_spans(input_ids, image_token_id: int) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index spans of each contiguous run of
    ``image_token_id`` in ``input_ids`` (a list / 1-D sequence of ints).

    Qwen-VL expands every image into a contiguous block of ``image_token_id``
    placeholders flanked by ``vision_start``/``vision_end``. The first span is
    the sample image (turn 1); the second, if present, is the test image.
    Validated on CPU against real Qwen3-VL tokenization (see tests/).
    """
    ids = list(int(x) for x in input_ids)
    spans: list[tuple[int, int]] = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == image_token_id:
            j = i
            while j + 1 < n and ids[j + 1] == image_token_id:
                j += 1
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


# --------------------------------------------------------------------------
# GPU forward pass (requires CUDA). Validate with --validate before trusting.
# --------------------------------------------------------------------------

def _truncated_forward(model, processor, sample_img, test_img, *, system,
                       sample_prompt, test_prompt, method: str, drop_sample: bool,
                       drop_positions_fn=None, max_new_tokens: int = 8) -> str:
    """Two-phase forward; returns the raw generated answer text.

    drop_sample=False  -> sample stays visible to the test turn (the ~C2 control).
    drop_sample=True   -> sample image KV positions are dropped after encoding
                          (the experimental, literal-truncation condition).
    method in {'mask','remove'} (only used when drop_sample=True).

    drop_positions_fn  -> optional callable ``(ids1, (s0, s1)) -> list[int]`` of
        ABSOLUTE phase-1 positions to make unattendable in the test turn. When
        given it OVERRIDES drop_sample and enables the guardrail-6 control
        matrix (target-only / matched / rescue-* / graded-fraction masks built
        from vwm.local.kv_surgery.surgery_conditions) through the SAME
        validated forward path — the Colab session must not reimplement this.
    """
    import torch
    from PIL import Image

    device = model.device
    image_token_id = model.config.image_token_id

    def _pil(x):
        return x if isinstance(x, Image.Image) else Image.open(x).convert("RGB")

    sample = _pil(sample_img)
    test = _pil(test_img)

    # ---- Phase 1: encode [system, sample, sample_prompt, assistant(READY)] ----
    m1 = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample},
            {"type": "text", "text": sample_prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": FIXED_ACK}]},
    ]
    text1 = processor.apply_chat_template(m1, tokenize=False, add_generation_prompt=False)
    enc1 = processor(text=[text1], images=[sample], return_tensors="pt")
    enc1 = {k: v.to(device) for k, v in enc1.items()}
    with torch.inference_mode():
        out1 = model(**enc1, use_cache=True)
    cache = out1.past_key_values
    ids1 = enc1["input_ids"][0].tolist()
    L1 = len(ids1)
    spans = find_image_token_spans(ids1, image_token_id)
    if not spans:
        raise RuntimeError("no sample image span found in phase-1 tokens")
    s0, s1 = spans[0]                    # sample image span (inclusive)

    # ---- Phase 2: present the test turn, optionally dropping the sample KV ----
    m2 = [{"role": "user", "content": [
        {"type": "image", "image": test},
        {"type": "text", "text": test_prompt},
    ]}]
    # Tokenize the continuation turn on its own (so pixel_values carries only the
    # TEST image; the sample's pixels were consumed in phase 1).
    text2 = processor.apply_chat_template(m2, tokenize=False, add_generation_prompt=True)
    enc2 = processor(text=[text2], images=[test], return_tensors="pt")
    enc2 = {k: v.to(device) for k, v in enc2.items()}
    new_len = enc2["input_ids"].shape[1]

    if drop_sample and method == "remove":
        # Physically drop the sample image positions from the cache.
        # NOTE: VALIDATED method is 'mask' (below). Physical removal shrinks the
        # cache, so the retained tokens' original (post-RoPE) positions no longer
        # match the cache length and generate() will renumber them -- this branch
        # needs explicit position_ids handling and is NOT yet GPU-validated. Prefer
        # method='mask'.
        keep = [i for i in range(L1) if not (s0 <= i <= s1)]
        keep_t = torch.tensor(keep, device=device)
        cache.batch_select_indices(keep_t)
        attn = torch.ones((1, len(keep) + new_len), dtype=torch.long, device=device)
    else:
        # 'mask' method (default) or the visible control: keep the full cache,
        # build an attention mask over (L1 + new_len). For the experimental
        # condition, zero out the sample image key positions so the test/answer
        # queries cannot attend to them (the acknowledgment, which DID attend to
        # the sample, stays attendable -> tests smuggling-forward).
        # VALIDATED on an A100 with Qwen3-VL-4B (transformers 5.10.1): the
        # sample-visible control scores 100% change-detection while the
        # sample-dropped condition collapses to chance (~56%), reproducing the
        # C2->C3 retrieval-not-storage result via genuine cache ablation.
        attn = torch.ones((1, L1 + new_len), dtype=torch.long, device=device)
        if drop_positions_fn is not None:
            drops = sorted(set(int(p) for p in drop_positions_fn(ids1, (s0, s1))))
            if drops and (drops[0] < 0 or drops[-1] >= L1):
                raise ValueError(f"drop positions outside phase-1 prefix [0,{L1})")
            if drops:
                attn[0, drops] = 0
        elif drop_sample:
            attn[0, s0:s1 + 1] = 0

    # NB: do NOT pass cache_position to generate() -- it is an internal forward
    # arg and generate() rejects it (ValueError: model_kwargs not used). With a
    # past_key_values + attention_mask spanning (cache + new), generate() derives
    # the positions itself.
    gen_kwargs = dict(
        input_ids=enc2["input_ids"],
        attention_mask=attn,
        past_key_values=cache,
        pixel_values=enc2.get("pixel_values"),
        image_grid_thw=enc2.get("image_grid_thw"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    with torch.inference_mode():
        out2 = model.generate(**gen_kwargs)
    gen = out2[:, enc2["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def _guard_manifest(manifest: dict, path: Path) -> None:
    """Refuse manifests this runner was not built for. Every sibling runner
    guards its manifest; this one previously accepted C1/wheel manifests
    silently (both carry sample_path/test_path) and wrote mislabeled rows."""
    design = manifest.get("design", "palette")
    cond = manifest.get("condition")
    if design != "palette" or (cond is not None and cond != "C2"):
        raise ValueError(
            f"kv_truncation_runner needs the C2 palette manifest (two-turn "
            f"sample/test trials); got design={design!r} condition={cond!r} "
            f"from {path}")


def _stratified_sample(trials: list[dict], n: int) -> list[dict]:
    """Round-robin across set sizes. trials[:n] on a set-size-ordered manifest
    validated ONLY set_size=1, so regressions that bite at larger image-token
    spans shipped under a passed gate."""
    from collections import defaultdict
    by_n: dict[int, list[dict]] = defaultdict(list)
    for t in trials:
        by_n[t["set_size"]].append(t)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by_n.values()):
        for ss in sorted(by_n):
            if i < len(by_n[ss]) and len(out) < n:
                out.append(by_n[ss][i])
        i += 1
    return out


def surgery_forward(model, processor, sample_img, test_img, *, system,
                    sample_prompt, test_prompt, drop_positions_fn,
                    max_new_tokens: int = 8) -> str:
    """Public seam for the guardrail-6 control matrix: run the validated
    two-phase forward with an arbitrary drop-position rule (see
    `_truncated_forward`). ``drop_positions_fn(ids1, span) -> list[int]``."""
    return _truncated_forward(
        model, processor, sample_img, test_img, system=system,
        sample_prompt=sample_prompt, test_prompt=test_prompt, method="mask",
        drop_sample=False, drop_positions_fn=drop_positions_fn,
        max_new_tokens=max_new_tokens)


def layer_zero_forward(model, processor, sample_img, test_img, *, system,
                       sample_prompt, test_prompt, positions_fn,
                       layers=None, max_new_tokens: int = 8) -> str:
    """Layer-selective V-zeroing seam (it-25 localization sweep).

    Runs the SAME validated two-phase forward as `surgery_forward`, but instead
    of making positions un-attendable via the attention mask (which acts at
    every layer at once — the maximal intervention), it zeroes the cached VALUE
    vectors at ``positions_fn(ids1, span)`` for only the given ``layers``
    (all layers if None), leaving keys and the attention mask intact. Queries
    still attend to the positions; at the chosen layers they read zeros.

    Validation contract (enforced by the sweep script, not here):
      layers=None + full-span positions  -> must reproduce the known erase;
      layers=[]                          -> must equal the unmasked baseline.
    """
    import torch
    from PIL import Image

    from .kv_surgery import zero_value_positions

    device = model.device
    image_token_id = model.config.image_token_id

    def _pil(x):
        return x if isinstance(x, Image.Image) else Image.open(x).convert("RGB")

    sample = _pil(sample_img)
    test = _pil(test_img)

    m1 = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample},
            {"type": "text", "text": sample_prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": FIXED_ACK}]},
    ]
    text1 = processor.apply_chat_template(m1, tokenize=False, add_generation_prompt=False)
    enc1 = processor(text=[text1], images=[sample], return_tensors="pt")
    enc1 = {k: v.to(device) for k, v in enc1.items()}
    with torch.inference_mode():
        out1 = model(**enc1, use_cache=True)
    cache = out1.past_key_values
    ids1 = enc1["input_ids"][0].tolist()
    L1 = len(ids1)
    spans = find_image_token_spans(ids1, image_token_id)
    if not spans:
        raise RuntimeError("no sample image span found in phase-1 tokens")
    s0, s1 = spans[0]

    drops = sorted(set(int(p) for p in positions_fn(ids1, (s0, s1))))
    if drops and (drops[0] < 0 or drops[-1] >= L1):
        raise ValueError(f"zero positions outside phase-1 prefix [0,{L1})")
    if drops and (layers is None or len(list(layers)) > 0):
        with torch.inference_mode():
            zero_value_positions(cache, drops, layers=layers)

    m2 = [{"role": "user", "content": [
        {"type": "image", "image": test},
        {"type": "text", "text": test_prompt},
    ]}]
    text2 = processor.apply_chat_template(m2, tokenize=False, add_generation_prompt=True)
    enc2 = processor(text=[text2], images=[test], return_tensors="pt")
    enc2 = {k: v.to(device) for k, v in enc2.items()}
    new_len = enc2["input_ids"].shape[1]
    attn = torch.ones((1, L1 + new_len), dtype=torch.long, device=device)

    gen_kwargs = dict(
        input_ids=enc2["input_ids"],
        attention_mask=attn,
        past_key_values=cache,
        pixel_values=enc2.get("pixel_values"),
        image_grid_thw=enc2.get("image_grid_thw"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    with torch.inference_mode():
        out2 = model.generate(**gen_kwargs)
    gen = out2[:, enc2["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def cache_layer_count(model) -> int:
    """Number of decoder layers (VL models keep it on the nested text_config)."""
    cfg = getattr(model.config, "text_config", model.config)
    return int(cfg.num_hidden_layers)


@dataclass
class KVTruncConfig:
    model_key: str
    manifest_path: Path
    out_path: Path
    method: str = "mask"               # 'mask' | 'remove'
    n_trials: int | None = None
    set_sizes: tuple[int, ...] | None = None
    dtype: str = "bfloat16"            # A100 regime; pass '4bit' on small GPUs


def _require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "kv_truncation_runner requires a CUDA GPU (4-bit bitsandbytes is "
            "CUDA-only) and torch.cuda.is_available() is False. Fix the GPU "
            "driver (reboot / reinstall) or run on a CUDA host, then retry."
        )


def validate(model_key: str, manifest_path: Path, n: int = 12,
             dtype: str = "bfloat16") -> dict:
    """Sanity controls before any real sweep. Returns a dict of pass/fail checks.

    C-visible : sample stays visible in phase 2 -> should track C2 (well above 50%).
    C-dropped : sample KV dropped -> the test of the hypothesis. We only assert it
                DIFFERS from C-visible (the surgery must bite); its *direction* is
                the scientific result, not a sanity check.
    """
    _require_cuda()
    load_dotenv()
    cls = LOCAL_REGISTRY[model_key.lower()]
    client = cls(dtype=dtype)
    client.ensure_loaded()
    model, processor = client._model, client._processor
    manifest = load_manifest(manifest_path)
    _guard_manifest(manifest, manifest_path)
    trials = _stratified_sample(manifest["trials"], n)

    vis_correct = drop_correct = differ = 0
    for t in trials:
        kw = dict(system=SYSTEM_INSTRUCTION, sample_prompt=PROMPT_C2_SEQUENTIAL_TURN1,
                  test_prompt=PROMPT_C2_SEQUENTIAL_TURN2)
        vis = _truncated_forward(model, processor, t["sample_path"], t["test_path"],
                                 method="mask", drop_sample=False, **kw)
        drp = _truncated_forward(model, processor, t["sample_path"], t["test_path"],
                                 method="mask", drop_sample=True, **kw)
        nv, nd = normalize_response(vis), normalize_response(drp)
        if nv["parsable"] and nv["parsed_yes"] == t["change"]:
            vis_correct += 1
        if nd["parsable"] and nd["parsed_yes"] == t["change"]:
            drop_correct += 1
        if vis != drp:
            differ += 1
    client.unload()
    res = {
        "n": len(trials),
        "visible_acc": vis_correct / max(len(trials), 1),
        "dropped_acc": drop_correct / max(len(trials), 1),
        "frac_outputs_differ": differ / max(len(trials), 1),
        "PASS_visible_above_chance": (vis_correct / max(len(trials), 1)) > 0.6,
        "PASS_surgery_bites": (differ / max(len(trials), 1)) > 0.0,
    }
    print("[kv-trunc validate]", res)
    if not (res["PASS_visible_above_chance"] and res["PASS_surgery_bites"]):
        print("  !! VALIDATION FAILED -- do not trust sweep numbers; debug the "
              "phase-2 forward (position_ids / pixel_values / mask shape).")
    return res


def run_kv_truncation(cfg: KVTruncConfig) -> dict:
    _require_cuda()
    load_dotenv()
    if cfg.method == "remove":
        # 'remove' physically crops the cache; generate() then renumbers RoPE
        # positions and corrupts relative-position geometry. NOT GPU-validated.
        raise SystemExit(
            "--method remove is NOT validated (position_ids renumbering after "
            "physical cache crop corrupts RoPE geometry) and its rows would "
            "enter the K table as valid data. Use --method mask. If you are "
            "actively developing 'remove', edit this guard consciously.")
    cls = LOCAL_REGISTRY.get(cfg.model_key.lower())
    if cls is None:
        raise ValueError(f"unknown model {cfg.model_key!r}; known: {list(LOCAL_REGISTRY)}")
    manifest = load_manifest(cfg.manifest_path)
    _guard_manifest(manifest, cfg.manifest_path)
    trials = manifest["trials"]
    if cfg.set_sizes:
        trials = [t for t in trials if t["set_size"] in cfg.set_sizes]
    if cfg.n_trials is not None:
        trials = trials[: cfg.n_trials]

    client = cls(dtype=cfg.dtype)
    client.ensure_loaded()
    model, processor = client._model, client._processor
    ensure_dir(cfg.out_path.parent)

    n_ok = n_err = 0
    t0 = time.monotonic()
    for trial in tqdm(trials, desc=f"kv-trunc/{cfg.model_key}/{cfg.method}"):
        row = {
            "trial_id": trial["trial_id"], "set_size": trial["set_size"],
            "change": trial["change"], "condition": f"C3-truncate-{cfg.method}",
            "provider": "local", "model": client.model_id,
            "dtype": getattr(client, "effective_dtype", cfg.dtype),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            raw = _truncated_forward(
                model, processor, trial["sample_path"], trial["test_path"],
                system=SYSTEM_INSTRUCTION, sample_prompt=PROMPT_C2_SEQUENTIAL_TURN1,
                test_prompt=PROMPT_C2_SEQUENTIAL_TURN2, method=cfg.method, drop_sample=True,
            )
            norm = normalize_response(raw)
            row.update({
                "raw": raw, "parsed_yes": norm["parsed_yes"], "parsable": norm["parsable"],
                "correct": (norm["parsed_yes"] == trial["change"]) if norm["parsable"] else None,
                "error": None,
            })
            n_ok += 1
        except Exception as e:                                          # noqa: BLE001
            row.update({"raw": "", "parsed_yes": None, "parsable": False,
                        "correct": None, "error": f"{type(e).__name__}: {e}"})
            n_err += 1
        append_jsonl(cfg.out_path, row)

    client.unload()
    dur = time.monotonic() - t0
    summary = {"model": client.model_id, "method": cfg.method, "n_trials": len(trials),
               "n_ok": n_ok, "n_err": n_err, "duration_s": dur, "out_path": str(cfg.out_path)}
    print(f"[kv-trunc/{cfg.model_key}] {n_ok} ok / {n_err} err in {dur:.1f}s")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Literal post-encoding KV-cache truncation.")
    p.add_argument("--model", required=True, help=f"one of {list(LOCAL_REGISTRY)}")
    p.add_argument("--manifest", type=Path, default=Path("data/stimuli/manifest_C2.json"),
                   help="C2 manifest (needs sample_path + test_path per trial)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--method", choices=["mask", "remove"], default="mask")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--set-sizes", type=str, default=None)
    p.add_argument("--dtype", choices=["bfloat16", "4bit"], default="bfloat16",
                   help="bf16 = A100 regime (default here); pass 4bit on "
                        "VRAM-constrained GPUs")
    p.add_argument("--validate", action="store_true",
                   help="run the sanity controls (visible-vs-dropped) and exit")
    args = p.parse_args(argv)

    if args.validate:
        validate(args.model, args.manifest, dtype=args.dtype)
        return 0

    out = args.out or Path(f"results/raw/{args.model}_C3-truncate-{args.method}.jsonl")
    set_sizes = tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None
    print(run_kv_truncation(KVTruncConfig(
        model_key=args.model, manifest_path=args.manifest, out_path=out,
        method=args.method, n_trials=args.n_trials, set_sizes=set_sizes,
        dtype=args.dtype,
    )))
    return 0


if __name__ == "__main__":
    sys.exit(main())
