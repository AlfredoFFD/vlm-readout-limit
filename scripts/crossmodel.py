"""THE SAME PROTOCOL, ON ANY VISION-LANGUAGE MODEL.

H2: the response-vocabulary penalty is not a Qwen3-VL quirk. Testing that needs a
second model family, and it needs the SAME procedure rather than a second protocol
that happens to point the same way. So this file is self-contained and takes a
model id: it calibrates that model's own colour vocabulary from scratch, samples
held-out angles by the frozen rule, and runs the matched word-versus-number
contrast. Nothing about the Qwen calibration carries over -- if another model names
this wheel differently, that is a result, not a problem to be configured away.

WHY THE VISIBLE ARM COMES FIRST. Two days of this project went into making a
cross-turn KV continuation behave on Qwen3-VL (mm_token_type_ids, M-RoPE, a
generate() call that mutates the cache it is handed). There is no reason to expect
a different architecture to be quicker, and no reason to let it gate the claim,
because the claim does not need it:

    response vocabulary affects measured performance        visible arm suffices
    the effect survives held-out stimuli                    visible arm suffices
    the effect crosses model families                       visible arm suffices
    cached readout is affected the same way                 needs the cache path

So the visible contrast runs first and always. The cached arm is attempted second
and its failure is logged as a fact about the port, never allowed to void the run.

WHAT IS HELD IDENTICAL ACROSS MODELS
  the renderer, the calibration grid (5 deg), the three renders per angle, the
  unanimity criterion, the minimum category separation, the held-out sampling rule,
  the option count, the candidate order, the output format, and the scoring (first
  generated token, restricted to a verified-distinct candidate set).

WHAT IS ALLOWED TO DIFFER
  the vocabulary itself, and how much of it survives. Calibration yield is an
  OUTCOME and is reported: a model that admits two stable colour names rather than
  six constrains the method, and saying so is the honest version of this result.

Usage:
  VWM_XM_MODEL=google/gemma-3-4b-it .venv/Scripts/python.exe scripts/crossmodel.py
Env: VWM_XM_MODEL, VWM_XM_N (trials, default 120), VWM_XM_STEP (calib grid, 5),
     VWM_XM_STAGES (calibrate,contrast), VWM_XM_CACHED (1 to attempt the cache arm)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from vwm.stimuli.colorwheel import wheel_rgb
from vwm.stimuli.geometry import CANVAS_W, CANVAS_H, SQUARE_PX

MODEL_ID = os.environ.get("VWM_XM_MODEL", "google/gemma-3-4b-it")
N_TRIALS = int(os.environ.get("VWM_XM_N", "120"))
STEP = int(os.environ.get("VWM_XM_STEP", "5"))
STAGES = os.environ.get("VWM_XM_STAGES", "calibrate,contrast").split(",")
TRY_CACHED = os.environ.get("VWM_XM_CACHED", "1") == "1"
SET_SIZE = 4
BG = (127, 127, 127)

SLUG = MODEL_ID.split("/")[-1].replace(".", "_")
OUT_DIR = Path("results/crossmodel"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CAL_PATH = OUT_DIR / f"{SLUG}_calibration.json"
OUT_PATH = OUT_DIR / f"{SLUG}_contrast.jsonl"
IMG_DIR = Path(f"results/crossmodel/_imgs_{SLUG}"); IMG_DIR.mkdir(parents=True, exist_ok=True)

SYS = ("You are a careful visual working memory experiment participant. "
       "Follow the instructions exactly and answer concisely.")
T1 = ("I will show you an array of colored squares. Study the exact color and "
      "location of each. Reply with only the word READY.")
ACK = "READY"
NAME_Q = ("What color is the square in this image? Answer with a single common "
          "color word and nothing else.")

VARIANTS = [(SQUARE_PX, 0.5, 0.5), (int(SQUARE_PX * 1.4), 0.32, 0.36),
            (int(SQUARE_PX * 0.8), 0.68, 0.64)]
MIN_WHEEL_SEP = 40.0
MIN_NAME_GAP = 2
CAL_STEP = STEP
GUARD_OTHER = 20.0

import hashlib


def hsh(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, p)).encode()).digest()[:4], "big")


def circ(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def cell_of(cx, cy):
    def band(v, hi):
        t = v / hi
        return 0 if t < 1 / 3 else (1 if t < 2 / 3 else 2)
    return (band(cy, CANVAS_H), band(cx, CANVAS_W))


def phrase(cell):
    nv = ("top", "middle", "bottom"); nh = ("left", "center", "right")
    v, h = cell
    return f"{nv[v]}-{nh[h]}" if (v, h) != (1, 1) else "center"


def swatch(hue, size, fx, fy, path):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(img)
    cx, cy = int(CANVAS_W * fx), int(CANVAS_H * fy)
    h = size // 2
    d.rectangle([cx - h, cy - h, cx + h - 1, cy + h - 1],
                fill=wheel_rgb(hue), outline=(64, 64, 64), width=2)
    img.save(path)
    return img


def render(positions, hues, path):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(img); h = SQUARE_PX // 2
    for (cx, cy), hue in zip(positions, hues):
        d.rectangle([cx - h, cy - h, cx + h - 1, cy + h - 1],
                    fill=wheel_rgb(hue), outline=(64, 64, 64), width=2)
    img.save(path)
    return img


def load(model_id):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=q, device_map="cuda:0")
    model.eval()
    return model, proc


def build(proc, msgs, img, gen_prompt=True):
    txt = proc.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=gen_prompt)
    e = proc(text=[txt], images=[img] if img is not None else None,
             return_tensors="pt")
    return e


def generate_text(model, proc, img, prompt, max_new_tokens=6):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": prompt}]}]
    e = build(proc, msgs, img)
    e = {k: v.to(model.device) for k, v in e.items()}
    L = e["input_ids"].shape[1]
    with torch.inference_mode():
        g = model.generate(**e, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(g[:, L:], skip_special_tokens=True)[0].strip()


def score_visible(model, proc, img, prompt, ids):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": prompt}]}]
    e = build(proc, msgs, img)
    e = {k: v.to(model.device) for k, v in e.items()}
    with torch.inference_mode():
        g = model.generate(**e, max_new_tokens=1, do_sample=False,
                           return_dict_in_generate=True, output_scores=True)
    lg = g.scores[0][0].float()
    t = torch.tensor(ids, device=lg.device)
    return ([float(x) for x in lg[t]], int(torch.argmax(lg[t])),
            float(torch.softmax(lg, -1)[t].sum()))


def score_cached(model, proc, img, prompt, ids):
    """Two-phase: encode the array once, continue with a text-only query. Portable
    version -- no model-specific kwargs. Raises on any architecture that will not
    take a text continuation against an image-built cache; the caller records it."""
    dev = model.device
    m1 = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
          {"role": "user", "content": [{"type": "image", "image": img},
                                       {"type": "text", "text": T1}]},
          {"role": "assistant", "content": [{"type": "text", "text": ACK}]}]
    e1 = build(proc, m1, img, gen_prompt=False)
    e1 = {k: v.to(dev) for k, v in e1.items()}
    with torch.inference_mode():
        o1 = model(**e1, use_cache=True)
    L1 = e1["input_ids"].shape[1]
    m2 = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    e2 = build(proc, m2, None)
    e2 = {k: v.to(dev) for k, v in e2.items()}
    L2 = e2["input_ids"].shape[1]
    attn = torch.ones((1, L1 + L2), dtype=torch.long, device=dev)
    kw = {k: v for k, v in e2.items() if k != "attention_mask"}
    with torch.inference_mode():
        g = model.generate(**kw, attention_mask=attn,
                           past_key_values=o1.past_key_values,
                           max_new_tokens=1, do_sample=False,
                           return_dict_in_generate=True, output_scores=True)
    lg = g.scores[0][0].float()
    t = torch.tensor(ids, device=lg.device)
    return ([float(x) for x in lg[t]], int(torch.argmax(lg[t])),
            float(torch.softmax(lg, -1)[t].sum()))


def calibrate(model, proc):
    table = {}
    for a in range(0, 360, STEP):
        names = []
        for i, (sz, fx, fy) in enumerate(VARIANTS):
            im = swatch(a, sz, fx, fy, IMG_DIR / f"cal_{a:03d}_{i}.png")
            out = generate_text(model, proc, im, NAME_Q)
            w = re.findall(r"[a-zA-Z]+", out.lower())
            names.append(w[0] if w else "")
        c = Counter(names); modal, k = c.most_common(1)[0]
        table[a] = {"names": names, "modal": modal, "agree": k / len(names),
                    "rgb": wheel_rgb(a)}
        print(f"  {a:3d} {wheel_rgb(a)} -> {names} modal={modal}", flush=True)

    # anchors: self + both neighbours unanimous on the same name, then greedily
    # spaced at least MIN_WHEEL_SEP apart, exactly the Qwen3-VL rule
    cand = []
    for a, v in table.items():
        if v["agree"] < 1.0:
            continue
        nb = [table.get((a - STEP) % 360), table.get((a + STEP) % 360)]
        if any(x is None or x["agree"] < 1.0 or x["modal"] != v["modal"] for x in nb):
            continue
        cand.append((a, v["modal"]))
    byname = {}
    for a, nm in cand:
        byname.setdefault(nm, []).append(a)
    reps = []
    for nm, aa in byname.items():
        reps.append((aa[len(aa) // 2], nm))          # centre of the stable run
    reps.sort(key=lambda x: -len(byname[x[1]]))      # widest runs first
    chosen = []
    for a, nm in reps:
        if all(circ(a, b) >= MIN_WHEEL_SEP for b, _ in chosen):
            chosen.append((a, nm))
    chosen.sort()
    out = {"model": MODEL_ID, "step": STEP, "table": table,
           "named": [{"wheel": a, "name": nm} for a, nm in chosen],
           "yield": {"stable_angles": len(cand), "stable_names": len(byname),
                     "anchors": len(chosen)}}
    json.dump(out, open(CAL_PATH, "w", encoding="utf-8"), indent=1)
    print(f"\nCALIBRATION YIELD for {MODEL_ID}")
    print(f"  unanimous angles      {len(cand)}/{len(table)}")
    print(f"  stable names          {sorted(byname)}")
    print(f"  anchors >= {MIN_WHEEL_SEP:.0f} deg apart  "
          f"{[(a, nm) for a, nm in chosen]}")
    print(f"  wrote {CAL_PATH}")
    return chosen


def heldout_for(named):
    out = {}
    for w, nm in named:
        picks = []
        for side in (-1, 1):
            cands = [int((w + side * d) % 360) for d in range(10, 21)]
            cands = [t for t in cands if t % CAL_STEP != 0
                     and all(circ(t, ow) >= GUARD_OTHER
                             for ow, onm in named if onm != nm)]
            for target in (12, 18):
                if cands:
                    p = min(cands, key=lambda t: abs(circ(t, w) - target))
                    if p not in picks:
                        picks.append(p)
        out[nm] = sorted(picks) or [w]
    return out


# Surface variants a model might actually emit first. Assuming one of these is
# how the harness measured SmolVLM at chance on a task it was answering correctly:
# it emits " Green" (id 4910) and the candidate set said "green" (id 9298), so
# candidate mass was 0.00 and every cell scored as a coin flip. The model was
# right and the readout was wrong -- which is this paper's own thesis, committed
# against a model being used to test it. The fix is to stop assuming and measure.
VARIANT_FAMILIES = [
    ("bare", lambda w: w),
    ("space", lambda w: " " + w),
    ("cap", lambda w: w.capitalize()),
    ("space_cap", lambda w: " " + w.capitalize()),
    ("upper", lambda w: w.upper()),
    ("space_upper", lambda w: " " + w.upper()),
]


def _ids_for(tok, words, fn):
    ids = [tok.encode(fn(w), add_special_tokens=False)[0] for w in words]
    return ids if len(set(ids)) == len(ids) else None


def resolve_tokens(model, proc, words, probes, min_mass=0.5):
    """Pick the surface family this model actually emits, by measuring it.

    `probes` are (image, prompt) pairs whose answers lie in `words`. For each
    family the candidate ids are scored on every probe and the family with the
    most probability mass on the candidate set wins. Below `min_mass` the caller
    must abort rather than report numbers: a chance-level score with no mass on
    the alphabet is a broken readout, not a model that cannot do the task."""
    tok = getattr(proc, "tokenizer", proc)
    best = (None, -1.0, None)
    for nm, fn in VARIANT_FAMILIES:
        ids = _ids_for(tok, words, fn)
        if ids is None:
            print(f"    {nm:<12} skipped, candidates share a first token")
            continue
        ms = []
        for img, prompt in probes:
            try:
                _, _, m = score_visible(model, proc, img, prompt, ids)
                ms.append(m)
            except Exception:                                 # noqa: BLE001
                pass
        mm = float(sum(ms) / len(ms)) if ms else 0.0
        print(f"    {nm:<12} candidate mass {mm:.3f}")
        if mm > best[1]:
            best = (nm, mm, ids)
    nm, mm, ids = best
    print(f"  -> using {nm!r}, mass {mm:.3f}")
    if ids is None or mm < min_mass:
        raise RuntimeError(
            f"no surface family puts mass on the candidate set (best {nm}={mm:.3f} "
            f"< {min_mass}). Refusing to report scores read off a token set this "
            f"model does not use.")
    return ids


def answer_by_generation(model, proc, img, prompt, kind, choices, cached=False):
    """Generate freely and parse, the protocol real benchmarks actually use.

    First-token scoring is not universally available. SmolVLM emits a shared
    leading-space token before every integer, so all six labels collide on their
    first token and no restricted first-token readout of a label answer exists for
    that model at all -- which is worth reporting, and is also why accuracy here is
    primarily measured by generating and parsing. Where a first-token readout does
    resolve, both are recorded and their agreement is reported."""
    txt = generate_text(model, proc, img, prompt, max_new_tokens=8)
    low = txt.lower()
    if kind == "word":
        hits = [(low.find(w), j) for j, w in enumerate(choices) if w in low]
        hits = [h for h in hits if h[0] >= 0]
        return (min(hits)[1] if hits else None), txt
    m = re.search(r"\d+", txt)
    if not m:
        return None, txt
    v = int(m.group()) - 1
    return (v if 0 <= v < len(choices) else None), txt


def first_tokens(tok, words):
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in words]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"colour words share a first token: {list(zip(words, ids))}")
    return ids


def contrast(model, proc, named):
    tok = getattr(proc, "tokenizer", proc)
    K = len(named)
    if K < 3:
        print(f"!! only {K} stable anchors; the contrast needs at least 3. "
              f"Reporting calibration yield only.")
        return
    WORDS = [nm for _, nm in named]
    ho = heldout_for(named)

    # Resolve the readout against this model before measuring anything with it.
    probe_imgs = []
    for i, (w, nm) in enumerate(named[:4]):
        probe_imgs.append((swatch(w, SQUARE_PX, 0.5, 0.5,
                                  IMG_DIR / f"probe_{nm}.png"), nm))
    wprobes = [(im, f"In this image, the square was one of these colors: "
                    f"{', '.join(WORDS)}. Which of these is it closest to? Answer "
                    f"with a single color word and nothing else.")
               for im, _ in probe_imgs]
    lprobes = [(im, f"In this image, the square was one of these colors: "
                    f"{', '.join(f'{j+1}={w}' for j, w in enumerate(WORDS))}. Which "
                    f"of these is it closest to? Answer with a single integer 1 to "
                    f"{K} and nothing else.")
               for im, _ in probe_imgs]
    print("resolving the WORD readout for this model:")
    WORD_IDS = resolve_tokens(model, proc, WORDS, wprobes)
    print("resolving the LABEL readout for this model:")
    try:
        DIGIT_IDS = resolve_tokens(model, proc, [str(j) for j in range(1, K + 1)],
                                   lprobes)
    except RuntimeError as e:                                 # noqa: BLE001
        DIGIT_IDS = None
        print(f"  label first-token readout UNAVAILABLE: {e}")
        print("  -> label and numeric rungs scored by generate-and-parse only.")
    ANG = {nm: w for w, nm in named}
    NUM_SCALE = ("Colors are described by a hue angle from 0 to 359 on a color "
                 "wheel where " + ", ".join(f"{w} is {nm}" for w, nm in named) + ".")
    print(f"held-out angles: { {k: v for k, v in ho.items()} }", flush=True)

    cells = [(r, c) for r in range(3) for c in range(3)]
    f = OUT_PATH.open("w", encoding="utf-8")
    n = 0; t0 = time.time(); cache_fail = None

    for ti in range(N_TRIALS):
        tid = f"xm{ti:04d}"
        co = sorted(range(9), key=lambda j: hsh(tid, "cell", j))[:SET_SIZE]
        pos = [(int(CANVAS_W * (cells[j][1] + .5) / 3),
                int(CANVAS_H * (cells[j][0] + .5) / 3)) for j in co]
        po = sorted(range(K), key=lambda j: hsh(tid, "pick", j))[:SET_SIZE]
        if min(abs(po[0] - po[1]), K - abs(po[0] - po[1])) < MIN_NAME_GAP:
            continue
        names = [WORDS[i] for i in po]
        hues = [ho[nm][hsh(tid, "ho", i) % len(ho[nm])] for i, nm in enumerate(names)]
        img = render(pos, hues, IMG_DIR / f"{tid}.png")
        qi = hsh(tid, "q") % SET_SIZE
        where = phrase(cell_of(*pos[qi]))
        oo = sorted(range(K), key=lambda j: hsh(tid, "opt", j))
        opt = [WORDS[j] for j in oo]
        tgt_word = WORDS.index(names[qi])
        tgt_slot = opt.index(names[qi])

        for rung in ("word", "label", "numeric"):
            ids = WORD_IDS if rung == "word" else DIGIT_IDS
            if rung == "word":
                p = (f"In the array you memorized, the square at the {where} of the "
                     f"image was one of these colors: {', '.join(opt)}. Which of "
                     f"these is it closest to? Answer with a single color word and "
                     f"nothing else.")
                tgt = tgt_word
            elif rung == "label":
                p = (f"In the array you memorized, the square at the {where} of the "
                     f"image was one of these colors: "
                     f"{', '.join(f'{j+1}={w}' for j, w in enumerate(opt))}. Which "
                     f"of these is it closest to? Answer with a single integer 1 to "
                     f"{K} and nothing else.")
                tgt = tgt_slot
            else:
                p = (f"{NUM_SCALE} In the array you memorized, the square at the "
                     f"{where} of the image had one of these hues: "
                     f"{', '.join(f'{j+1}={ANG[w]}' for j, w in enumerate(opt))}. "
                     f"Which of these is it closest to? Answer with a single "
                     f"integer 1 to {K} and nothing else.")
                tgt = tgt_slot
            vis_p = p.replace("In the array you memorized", "In this image")
            kind = "word" if rung == "word" else "label"
            # INDEX-SPACE BUG, fixed. This was `opt`, the SHUFFLED option list, so
            # the parse returned a slot index while tgt for the word rung is an
            # index into the canonical alphabet WORDS. The two spaces differ by a
            # per-trial shuffle, which put every cross-model word(gen) score at
            # chance -- 15.0, 19.7, 14.1, 22.9 against 16.7 -- and made four
            # capable models look incapable. Same class as the SmolVLM token bug:
            # the readout and the key it is scored against were not the same thing.
            # The label rungs were never affected; both sides used slot indices.
            choices = WORDS if rung == "word" else [str(j + 1) for j in range(K)]
            rec = {"model": MODEL_ID, "trial_id": tid, "rung": rung, "K": K,
                   "where": where, "target": names[qi], "target_idx": tgt,
                   "hue": hues[qi], "options": opt, "heldout": True,
                   "ft_available": ids is not None}
            # PRIMARY: generate and parse, available on every model
            try:
                a, txt = answer_by_generation(model, proc, img, vis_p, kind, choices)
                rec.update(visible_gen=txt, visible_gen_idx=a,
                           visible_correct=bool(a == tgt),
                           visible_offvocab=bool(a is None))
                n += 1
            except Exception as e:                            # noqa: BLE001
                rec["error_visible_gen"] = f"{type(e).__name__}:{e}"
            # SECONDARY, where the model admits it: restricted first-token score
            if ids is not None:
                try:
                    sel, am, mass = score_visible(model, proc, img, vis_p, ids)
                    rec.update(visible_ft_argmax=am,
                               visible_ft_correct=bool(am == tgt),
                               visible_mass=mass,
                               visible_scores=[round(x, 4) for x in sel])
                    n += 1
                except Exception as e:                        # noqa: BLE001
                    rec["error_visible_ft"] = f"{type(e).__name__}:{e}"
            if TRY_CACHED and cache_fail is None and ids is not None:
                try:
                    sel, am, mass = score_cached(model, proc, img, p, ids)
                    rec.update(cached_ft_argmax=am, cached_correct=bool(am == tgt),
                               cached_mass=mass,
                               cached_scores=[round(x, 4) for x in sel])
                    n += 1
                except Exception as e:                        # noqa: BLE001
                    cache_fail = f"{type(e).__name__}: {e}"
                    print(f"!! cached arm unavailable on {MODEL_ID}: {cache_fail}",
                          flush=True)
                    print("   continuing with the visible arm, which carries H2.",
                          flush=True)
            if cache_fail:
                rec["cache_unavailable"] = cache_fail
            f.write(json.dumps(rec) + chr(10)); f.flush()
        if ti and ti % 20 == 0:
            el = time.time() - t0
            print(f"  {ti} trials  {n} forwards  {el/max(1,n):.2f}s  {el/60:.1f}m",
                  flush=True)
    f.close()
    print(f"XMDONE {MODEL_ID}: {n} forwards -> {OUT_PATH}")
    if cache_fail:
        print(f"CACHE ARM NOT AVAILABLE: {cache_fail}")


def main():
    print(f"=== {MODEL_ID}", flush=True)
    model, proc = load(MODEL_ID)
    named = None
    if "calibrate" in STAGES:
        named = calibrate(model, proc)
    if named is None and CAL_PATH.exists():
        named = [(d["wheel"], d["name"])
                 for d in json.load(open(CAL_PATH, encoding="utf-8"))["named"]]
    if "contrast" in STAGES and named:
        contrast(model, proc, named)


if __name__ == "__main__":
    main()
