"""IS THE PENALTY ABOUT COLOUR, OR ABOUT COORDINATES?

The colour result narrows to something sharper than it first looked. Changing the
ANSWER from a colour word to an integer cost 1%. Changing how the OPTIONS were
DESCRIBED, from "blue" to "245", cost 35 points. So the deficit is not in choosing
or in formatting. It is in grounding a described coordinate to a perceptual
quality: the model was told that hue runs 0-359 and could not use it.

If that is what is happening, it is not a fact about colour. It is a fact about
coordinate systems, and it should appear anywhere an evaluation elicits an answer
in units the model has to be told about. That includes the other half of binding,
WHERE, and it includes how the entire grounding and detection literature scores
vision-language models: in pixels.

So run the same manipulation on space.

TWO DIRECTIONS, because a coordinate can sit on either side of the question.

  locate    "where was the <colour> square?"  options are the nine grid cells
  identify  "what colour was the square at <location>?"  options are the six
            frozen colour words, and the CUE carries the coordinate

THREE DESCRIPTIONS of the same nine locations, everything else identical:

  cellname   top-left, top-center, ... plain English spatial phrases
  pixel      (74, 74) ... absolute pixel centres in this canvas
  norm1000   (133, 133) ... the 0-999 normalised convention Qwen-VL grounding
             data actually uses

THE THIRD RUNG IS WHY THIS IS WORTH RUNNING. The two outcomes are both useful and
they say different things.

  If pixel and norm1000 both collapse the way hue degrees did, the coordinate
  grounding deficit is general, and localisation benchmarks understate VLM spatial
  representation for the same reason our hue wheel understated colour memory.

  If norm1000 holds up while pixel and hue degrees collapse, the claim gets
  SHARPER rather than weaker: models are fine in coordinate systems they were
  TRAINED in and fail in ones merely described in context. That explains the hue
  result -- nothing in pretraining maps "245 degrees on a CIELAB wheel at L*=70"
  to an appearance -- and it turns a measurement complaint into a prediction about
  which readouts will and will not work for a given model.

CONTROLS
  * output format is held fixed within a direction: an integer label in `locate`,
    a colour word in `identify`. Only the DESCRIPTION of the locations varies, so
    the 1%-costing output-code factor cannot contribute.
  * the queried colour in `locate` is always named by the frozen calibrated
    vocabulary, so the cue side of that direction is the readout already shown to
    work and cannot be what fails.
  * visible arm for every cell: the same interface with the array on screen. A
    description that fails with the image present has not failed at memory.
  * cached arm is the same two-phase path used throughout: encode once, answer
    from the KV in a text-only continuation.

Usage: VWM_DTYPE=4bit .venv/Scripts/python.exe scripts/coordinate_grounding.py
Env: VWM_CG_N (trials, default 120), VWM_CG_SETSIZE (default 4)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw

from vwm.stimuli.colorwheel import wheel_rgb
from vwm.stimuli.geometry import CANVAS_W, CANVAS_H, SQUARE_PX
from vwm.local.qwen3_vl import Qwen3VLClient
from vwm.local.kv_truncation_runner import FIXED_ACK

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_ladder import (SYS, T1, NAMED, K, check_anchors, first_tokens, hsh)

DTYPE = os.environ.get("VWM_DTYPE", "4bit")
N_TRIALS = int(os.environ.get("VWM_CG_N", "120"))
SET_SIZE = int(os.environ.get("VWM_CG_SETSIZE", "4"))
DIRECTIONS = os.environ.get("VWM_CG_DIRS", "locate,identify").split(",")
DESCS = os.environ.get("VWM_CG_DESCS", "cellname,pixel,norm1000").split(",")
# H6 in the spatial domain. pixel_gloss attaches the cell name to each pixel
# option; pixel_mismatch attaches a DERANGED one. If the model reads the pixels,
# mispairing the names should not move it. If it reads the names, accuracy will
# follow the name and abandon the coordinate, as it did for hue.
GLOSSED = ("pixel_gloss", "pixel_mismatch")
CONDS = os.environ.get("VWM_CG_CONDS", "visible,cached").split(",")
BG = (127, 127, 127)

OUT_PATH = Path(os.environ.get("VWM_CG_OUT", "results/gpu_p3/coordinate_grounding.jsonl"))
IMG_DIR = Path("results/gpu_p3/_cg2_imgs"); IMG_DIR.mkdir(parents=True, exist_ok=True)

VNAME = ("top", "middle", "bottom")
HNAME = ("left", "center", "right")
CELLS = [(r, c) for r in range(3) for c in range(3)]


def centre(r, c):
    return int(CANVAS_W * (c + 0.5) / 3), int(CANVAS_H * (r + 0.5) / 3)


def derange(items, seed):
    n = len(items)
    for a in range(200):
        o = sorted(range(n), key=lambda j: hsh(seed, "der", a, j))
        if all(o[i] != i for i in range(n)):
            return [items[j] for j in o]
    return items[1:] + items[:1]


def describe(r, c, desc):
    cx, cy = centre(r, c)
    if desc in GLOSSED:
        desc = "pixel"
    if desc == "cellname":
        return "center" if (r, c) == (1, 1) else f"{VNAME[r]}-{HNAME[c]}"
    if desc == "pixel":
        return f"({cx}, {cy})"
    if desc == "norm1000":
        return (f"({round(1000 * cx / CANVAS_W)}, {round(1000 * cy / CANVAS_H)})")
    raise ValueError(desc)


PREAMBLE = {
    "cellname": "",
    "pixel": (f"Positions are given as (x, y) pixel coordinates in this "
              f"{CANVAS_W} by {CANVAS_H} image, where (0, 0) is the top-left "
              f"corner and x increases to the right and y increases downward. "),
    "norm1000": ("Positions are given as (x, y) coordinates normalized to a 0 to "
                 "999 range, where (0, 0) is the top-left corner and x increases "
                 "to the right and y increases downward. "),
}


def render(cells, hues, path):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(img); h = SQUARE_PX // 2
    for (r, c), hue in zip(cells, hues):
        cx, cy = centre(r, c)
        d.rectangle([cx - h, cy - h, cx + h - 1, cy + h - 1],
                    fill=wheel_rgb(hue), outline=(64, 64, 64), width=2)
    img.save(path)
    return img


def ask(model, proc, img, prompt, ids, cached):
    dev = model.device
    if not cached:
        m = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
             {"role": "user", "content": [{"type": "image", "image": img},
                                          {"type": "text", "text": prompt}]}]
        t = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        e = proc(text=[t], images=[img], return_tensors="pt")
        e = {k: v.to(dev) for k, v in e.items()}
        with torch.inference_mode():
            g = model.generate(**e, max_new_tokens=1, do_sample=False,
                               temperature=None, top_p=None,
                               return_dict_in_generate=True, output_scores=True)
    else:
        m1 = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
              {"role": "user", "content": [{"type": "image", "image": img},
                                           {"type": "text", "text": T1}]},
              {"role": "assistant", "content": [{"type": "text", "text": FIXED_ACK}]}]
        t1 = proc.apply_chat_template(m1, tokenize=False, add_generation_prompt=False)
        e1 = proc(text=[t1], images=[img], return_tensors="pt")
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
                               max_new_tokens=1, do_sample=False, temperature=None,
                               top_p=None, return_dict_in_generate=True,
                               output_scores=True)
    lg = g.scores[0][0].float()
    sel = lg[torch.tensor(ids, device=lg.device)]
    mass = float(torch.softmax(lg, -1)[torch.tensor(ids, device=lg.device)].sum())
    return [float(x) for x in sel], int(torch.argmax(sel)), mass


def main():
    check_anchors()
    client = Qwen3VLClient(dtype=DTYPE); client.ensure_loaded()
    model, proc = client._model, client._processor
    tok = getattr(proc, "tokenizer", proc)
    WORDS = [nm for _, nm in NAMED]
    WORD_IDS = first_tokens(tok, WORDS)
    DIGIT_IDS = []
    for j in range(1, 10):
        e = tok.encode(str(j), add_special_tokens=False)
        assert len(e) == 1, f"label {j} is not a single token"
        DIGIT_IDS.append(e[0])
    assert len(set(DIGIT_IDS)) == 9
    print(f"alphabets ok: words {WORDS}, 9 digit labels", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = OUT_PATH.open("w", encoding="utf-8")
    n = 0; t0 = time.time()

    for ti in range(N_TRIALS):
        tid = f"cg{ti:04d}"
        cell_order = sorted(range(9), key=lambda j: hsh(tid, "cell", j))
        cells = [CELLS[j] for j in cell_order[:SET_SIZE]]
        col_order = sorted(range(K), key=lambda j: hsh(tid, "col", j))
        present = col_order[:SET_SIZE]
        hues = [NAMED[i][0] for i in present]
        names = [NAMED[i][1] for i in present]
        img = render(cells, hues, IMG_DIR / f"{tid}.png")

        qi = hsh(tid, "q") % SET_SIZE            # which square is asked about
        tgt_cell = cells[qi]
        tgt_name = names[qi]

        # option orders, fixed within a trial so every rung sees the same layout
        loc_order = sorted(range(9), key=lambda j: hsh(tid, "lopt", j))
        loc_opts = [CELLS[j] for j in loc_order]
        loc_tgt = loc_opts.index(tgt_cell)
        col_order2 = sorted(range(K), key=lambda j: hsh(tid, "copt", j))
        col_opts = [WORDS[j] for j in col_order2]

        for desc in DESCS:
            if desc in GLOSSED:
                names = [describe(r, c, "cellname") for (r, c) in loc_opts]
                gl = derange(names, tid) if desc == "pixel_mismatch" else names
                listed = ", ".join(f"{j+1}={describe(r, c, 'pixel')} [{gl[j]}]"
                                   for j, (r, c) in enumerate(loc_opts))
            else:
                gl = None
                listed = ", ".join(f"{j+1}={describe(r, c, desc)}"
                                   for j, (r, c) in enumerate(loc_opts))
            cue = describe(*tgt_cell, desc)
            pre = PREAMBLE["pixel" if desc in GLOSSED else desc]

            prompts = {
                "locate": (f"{pre}In the array you memorized, the {tgt_name} square "
                           f"was at one of these positions: {listed}. Which one? "
                           f"Answer with a single integer 1 to 9 and nothing else."),
                "identify": (f"{pre}In the array you memorized, what color was the "
                             f"square at position {cue}? Answer with one of these "
                             f"words: {', '.join(col_opts)}. Answer with a single "
                             f"color word and nothing else."),
            }
            for direction in DIRECTIONS:
                prompt = prompts[direction]
                vis_prompt = prompt.replace("In the array you memorized",
                                            "In this image")
                ids = DIGIT_IDS if direction == "locate" else WORD_IDS
                tgt = loc_tgt if direction == "locate" else WORDS.index(tgt_name)
                if gl is not None and direction == "locate":
                    # key_word: the slot whose CELL NAME is the target's. Under
                    # pixel_gloss it coincides with key_coord by construction; under
                    # pixel_mismatch it never does.
                    tname = describe(*tgt_cell, "cellname")
                    rec_kw = gl.index(tname) if tname in gl else None
                else:
                    rec_kw = None
                rec = {"trial_id": tid, "direction": direction, "desc": desc,
                       "key_word": rec_kw,
                       "target_cell": list(tgt_cell), "target_color": tgt_name,
                       "target_idx": tgt, "set_size": SET_SIZE,
                       "n_options": 9 if direction == "locate" else K,
                       "prompt": prompt,
                       "prompt_tokens": len(tok.encode(prompt,
                                                       add_special_tokens=False)),
                       "score_source": "hf_generate_scores_greedy", "dtype": DTYPE}
                for cond in CONDS:
                    p = vis_prompt if cond == "visible" else prompt
                    try:
                        sel, am, mass = ask(model, proc, img, p, ids,
                                            cached=(cond == "cached"))
                    except Exception as e:                    # noqa: BLE001
                        rec[f"error_{cond}"] = f"{type(e).__name__}:{e}"
                        continue
                    rec[f"argmax_{cond}"] = am
                    rec[f"correct_{cond}"] = bool(am == tgt)
                    rec[f"mass_{cond}"] = mass
                    rec[f"scores_{cond}"] = [round(x, 4) for x in sel]
                    n += 1
                f.write(json.dumps(rec) + chr(10)); f.flush()
        if ti and ti % 20 == 0:
            el = time.time() - t0
            print(f"  {ti} trials  {n} forwards  {el/max(1,n):.2f}s each  "
                  f"{el/60:.1f}m", flush=True)

    f.close()
    print(f"CGDONE wrote {n} forwards -> {OUT_PATH}")


if __name__ == "__main__":
    main()
