"""THE CROSSOVER: is the cached evidence addressed to the CUED LOCATION?

Everything measured so far shows that swapping remembered content moves the option
distribution. It does not show the effect is bound to the location asked about. A
query-independent spatial weighting of the image would produce the same average.
This tests location-addressing directly, with a prediction that a nuisance factor
cannot imitate: the effect must REVERSE SIGN when the query moves.

DESIGN. Two arrays differing only by swapping hues h1 and h2 between locations q1
and q2. Every other item is untouched, so the palette is identical and three of four
bindings are shared.

  cache A:  q1 -> h1,  q2 -> h2
  cache B:  q1 -> h2,  q2 -> h1
  the diagnostic contrast is FIXED as score(h1) - score(h2) in every cell

  ask about q1:  A is h1, B is h2   ->  DiD should be POSITIVE
  ask about q2:  A is h2, B is h1   ->  DiD should be NEGATIVE
  ask about q3:  untouched by the swap -> DiD should be ZERO  (negative control)

A location-blind sensitivity to "which array is cached" gives the SAME sign at q1
and q2. Only location-addressed retrieval flips it. The q3 control catches any
residual whole-image effect. This is a within-array, palette-identical,
option-identical design, and the prediction is an interaction, not a mean.

WHY THIS AND NOT MORE SELECTION-BIAS WORK. Option-position and option-token bias in
VLM multiple choice are established, including for Qwen-family models, which shift
from token bias to positional bias as difficulty rises (Benchmarking and Mitigating
MCQA Selection Bias of LVLMs, EMNLP 2025; PriDe, ICLR 2024; MMBench, ECCV 2024).
None of that literature manipulates the remembered CONTENT with the option set held
fixed. The crossover is about what the evidence is ABOUT, which those protocols
cannot ask.

COLOUR MAPPING BUG, FIXED HERE. Earlier prompts declared "0 is red, 60 is yellow,
120 is green, 180 is cyan, 240 is blue, 300 is magenta" while wheel_rgb renders a
CIELAB circle at L*=70, r=38, whose 240 deg is cyan and whose 60 deg is orange -- a
systematic 35-50 deg mismatch. Any accuracy computed under the old prompt partly
measured compliance with a wrong instruction. Anchors here are MEASURED from the
renderer. A visible-target control runs the identical interface with the array on
screen, so mapping competence is established before any memory claim.

Usage: VWM_DTYPE=4bit python scripts/cue_crossover.py
Env: VWM_XO_N (stimuli, default 120), VWM_XO_K (default 8)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw

from vwm.stimuli.colorwheel import wheel_rgb, angular_distance
from vwm.stimuli.geometry import CANVAS_W, CANVAS_H, SQUARE_PX
from vwm.local.qwen3_vl import Qwen3VLClient
from vwm.local.kv_truncation_runner import FIXED_ACK

DTYPE = os.environ.get("VWM_DTYPE", "4bit")
N_STIM = int(os.environ.get("VWM_XO_N", "120"))
K = int(os.environ.get("VWM_XO_K", "8"))
SET_SIZE = 4
MIN_SEP = 30.0
BG = (127, 127, 127)

OUT_PATH = Path(os.environ.get("VWM_XO_OUT", "results/gpu_p3/cue_crossover.jsonl"))
IMG_DIR = Path("results/gpu_p3/_xo_imgs"); IMG_DIR.mkdir(parents=True, exist_ok=True)

SYS = ("You are a careful visual working memory experiment participant. "
       "Follow the instructions exactly and answer concisely.")
T1 = ("I will show you an array of colored squares. Study the exact color and "
      "location of each. Reply with only the word READY.")
# Anchors MEASURED from wheel_rgb, not assumed. See module docstring.
SCALE = ("Colors are described by a hue angle from 0 to 359 on a color wheel where "
         "0 is pink, 45 is orange, 90 is yellow, 135 is green, 180 is teal, "
         "225 is cyan, 270 is sky blue, and 315 is violet.")


def hsh(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, p)).encode()).digest()[:4], "big")


def cell_of(cx, cy):
    def band(v, hi):
        t = v / hi
        return 0 if t < 1 / 3 else (1 if t < 2 / 3 else 2)
    return (band(cy, CANVAS_H), band(cx, CANVAS_W))


def phrase(cell):
    nv = ("top", "middle", "bottom"); nh = ("left", "center", "right")
    v, h = cell
    return f"{nv[v]}-{nh[h]}" if (v, h) != (1, 1) else "center"


def render(positions, hues, path, visible=True):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(img); h = SQUARE_PX // 2
    for (cx, cy), hue in zip(positions, hues):
        d.rectangle([cx - h, cy - h, cx + h - 1, cy + h - 1],
                    fill=wheel_rgb(hue) if visible else None,
                    outline=(64, 64, 64), width=2)
    img.save(path)
    return img


def build_options(required, K, seed):
    """Every hue in `required` is GUARANTEED to be an option, at its exact value.

    Previously only h1 and h2 were forced in. h3 -- the hue at the unchanged
    control location -- was not, so scoring q3's visible answer against the
    NEAREST offered option was measuring proximity to a decoy, not correctness.
    The q3 DiD was never affected (its contrast is score(h1)-score(h2), and both
    of those were always present), but every visible-accuracy number computed on
    a q3 row was invalid. Forcing all three in repairs the accuracy measure and
    costs nothing elsewhere."""
    for a in range(len(required)):
        for b in range(a + 1, len(required)):
            if angular_distance(required[a], required[b]) < MIN_SEP:
                return None
    opts = list(required)
    step = 360.0 / (K * 2)
    for i in range(K * 6):
        if len(opts) >= K:
            break
        c = (required[0] + 11.0 + i * step) % 360.0
        if all(angular_distance(c, o) >= MIN_SEP for o in opts):
            opts.append(c)
    if len(opts) < K:
        return None
    order = sorted(range(K), key=lambda j: hsh(seed, "opt", j))
    return [opts[j] for j in order]


def exact_index(options, hue):
    """Index of `hue` in `options`, by exact identity rather than nearest match.

    Returns None if absent or ambiguous. Callers must fail loudly instead of
    silently substituting a decoy -- that substitution is the q3 defect."""
    hits = [j for j, o in enumerate(options) if angular_distance(o, hue) < 1e-6]
    return hits[0] if len(hits) == 1 else None


def token_ids(tok, K):
    ids = []
    for j in range(1, K + 1):
        e = tok.encode(str(j), add_special_tokens=False)
        if len(e) != 1:
            return None
        ids.append(e[0])
    return ids if len(set(ids)) == K else None


def ask(model, proc, sample_img, prompt, ids, cached):
    """cached=True: two-phase, sample encoded once then a text-only query against the
    cache. cached=False: single pass with the array visible alongside the question --
    the mapping-competence control. Fresh encode every call; generate() mutates."""
    dev = model.device
    if not cached:
        m = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
             {"role": "user", "content": [
                 {"type": "image", "image": sample_img},
                 {"type": "text", "text": prompt}]}]
        t = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        e = proc(text=[t], images=[sample_img], return_tensors="pt")
        e = {k: v.to(dev) for k, v in e.items()}
        with torch.inference_mode():
            g = model.generate(**e, max_new_tokens=1, do_sample=False,
                               temperature=None, top_p=None,
                               return_dict_in_generate=True, output_scores=True)
        lg = g.scores[0][0].float()
    else:
        m1 = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
              {"role": "user", "content": [
                  {"type": "image", "image": sample_img},
                  {"type": "text", "text": T1}]},
              {"role": "assistant", "content": [{"type": "text", "text": FIXED_ACK}]}]
        t1 = proc.apply_chat_template(m1, tokenize=False, add_generation_prompt=False)
        e1 = proc(text=[t1], images=[sample_img], return_tensors="pt")
        e1 = {k: v.to(dev) for k, v in e1.items()}
        with torch.inference_mode():
            o1 = model(**e1, use_cache=True)
        L1 = e1["input_ids"].shape[1]
        m2 = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        t2 = proc.apply_chat_template(m2, tokenize=False, add_generation_prompt=True)
        e2 = proc(text=[t2], return_tensors="pt")
        e2 = {k: v.to(dev) for k, v in e2.items()}
        L2 = e2["input_ids"].shape[1]
        attn = torch.ones((1, L1 + L2), dtype=torch.long, device=dev)
        kw = {k: v for k, v in e2.items() if k != "attention_mask"}
        with torch.inference_mode():
            g = model.generate(**kw, attention_mask=attn,
                               past_key_values=o1.past_key_values,
                               max_new_tokens=1, do_sample=False,
                               temperature=None, top_p=None,
                               return_dict_in_generate=True, output_scores=True)
        lg = g.scores[0][0].float()
    sel = lg[torch.tensor(ids, device=lg.device)]
    return [float(x) for x in sel], int(torch.argmax(sel))


def main():
    manifest = json.load(open("data/stimuli/manifest_wheel.json", encoding="utf-8"))
    client = Qwen3VLClient(dtype=DTYPE); client.ensure_loaded()
    model, proc = client._model, client._processor
    tok = getattr(proc, "tokenizer", proc)
    ids = token_ids(tok, K)
    assert ids is not None

    trials, seen = [], set()
    for t in manifest["trials"]:
        if t["set_size"] != SET_SIZE:
            continue
        cells = [cell_of(*p) for p in t["positions"]]
        if len(set(cells)) != SET_SIZE:        # all four in distinct coarse cells
            continue
        trials.append(t)
        if len(trials) >= N_STIM:
            break

    f = OUT_PATH.open("w", encoding="utf-8")
    n = skipped = 0
    t0 = time.time()

    for t in trials:
        off = t["wheel_zero_offset"]
        positions = [tuple(p) for p in t["positions"]]
        cells = [cell_of(*p) for p in positions]
        hues = [(x + off) % 360.0 for x in t["target_hues_deg"]]
        q1, q2, q3 = 0, 1, 2                    # swap q1<->q2, q3 is the control
        h1, h2, h3 = hues[q1], hues[q2], hues[q3]
        base = build_options([h1, h2, h3], K, t["trial_id"])
        if base is None:
            skipped += 1; continue
        iH1, iH2, iH3 = (exact_index(base, h1), exact_index(base, h2),
                         exact_index(base, h3))
        if None in (iH1, iH2, iH3) or len({iH1, iH2, iH3}) != 3:
            raise AssertionError(
                f"gate 1 violated on {t['trial_id']}: a queried target hue is "
                f"missing or ambiguous in the option set "
                f"(h1={h1:.2f}->{iH1}, h2={h2:.2f}->{iH2}, h3={h3:.2f}->{iH3})")
        tgt_idx = {"q1": iH1, "q2": iH2, "q3": iH3}

        huesA = list(hues)
        huesB = list(hues); huesB[q1], huesB[q2] = huesB[q2], huesB[q1]
        imgs = {"A": render(positions, huesA, IMG_DIR / f"{t['trial_id']}_A.png"),
                "B": render(positions, huesB, IMG_DIR / f"{t['trial_id']}_B.png")}

        listing = ", ".join(f"{j+1}={int(round(base[j]))}" for j in range(K))
        for qname, qi in (("q1", q1), ("q2", q2), ("q3", q3)):
            prompt = (f"{SCALE} In the array you memorized, the square at the "
                      f"{phrase(cells[qi])} of the image had one of these {K} hues: "
                      f"{listing}. Which one? Answer with a single integer 1 to {K}.")
            vis_prompt = prompt.replace("you memorized", "shown")
            rec = {"trial_id": t["trial_id"], "query": qname, "K": K,
                   "iH1": iH1, "iH2": iH2, "iH3": iH3,
                   "h1": round(h1, 2), "h2": round(h2, 2), "h3": round(h3, 2),
                   "options": [round(x, 2) for x in base],
                   "target_idx_visible": tgt_idx[qname],
                   "where": phrase(cells[qi]), "prompt": prompt,
                   # HF generate() scores under greedy decoding with no sampling
                   # warpers configured. NOT verified identical to unprocessed
                   # logits -- scripts/score_provenance.py does that separately.
                   "score_source": "hf_generate_scores_greedy", "dtype": DTYPE}
            good = True
            for ck in ("A", "B"):
                try:
                    lg, am = ask(model, proc, imgs[ck], prompt, ids, cached=True)
                except Exception as e:                        # noqa: BLE001
                    rec[f"error_{ck}"] = f"{type(e).__name__}:{e}"; good = False; continue
                rec[f"gap_{ck}"] = lg[iH1] - lg[iH2]      # FIXED contrast h1 vs h2
                rec[f"argmax_{ck}"] = am
                n += 1
            # mapping-competence control: same interface, array on screen
            try:
                lgv, amv = ask(model, proc, imgs["A"], vis_prompt, ids, cached=False)
                # Cache A holds the unpermuted array, so the on-screen hue at this
                # location IS hues[qi], whose exact option index is tgt_idx[qname].
                # No nearest-neighbour substitution anywhere.
                tgt = tgt_idx[qname]
                assert abs(angular_distance(base[tgt], huesA[qi])) < 1e-6
                rec["visible_argmax"] = amv
                rec["visible_correct"] = bool(amv == tgt)
                n += 1
            except Exception as e:                            # noqa: BLE001
                rec["error_visible"] = f"{type(e).__name__}:{e}"
            if good:
                rec["did"] = rec["gap_A"] - rec["gap_B"]
            f.write(json.dumps(rec) + chr(10)); f.flush()
        if n % 150 == 0 and n:
            print(f"  {n} forwards  {(time.time()-t0)/n:.2f}s each", flush=True)

    f.close()
    print(f"XODONE wrote {n} forwards, skipped {skipped} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
