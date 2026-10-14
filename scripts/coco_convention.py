"""THE CONVENTION GAP ON REAL PHOTOGRAPHS.

Every result so far lives on coloured squares against a grey field, and the fair
objection is that nobody evaluates anything in CIELAB hue degrees. H3 answered half
of that by moving to space. This answers the rest by moving to real images and to
the coordinate convention the localisation literature actually scores models in.

THE TASK. One COCO object per trial. Where is it? Choose among the nine cells of a
3x3 grid over the real photograph. Only the DESCRIPTION of those nine cells varies:

  cellname         top-left, center, ...
  norm1000         (167, 500)          the 0-999 convention
  pixel            (107, 213)          absolute pixels, explained in a preamble
  pixel_gloss      (107, 213) [top-left]
  pixel_mismatch   (107, 213) [bottom-right]     deranged, no fixed point

and, in the other direction, the coordinate becomes the CUE and the options become
COCO category names: "what object is at (107, 213)?"

WHY THE MISMATCH RUNG IS HERE AND NOT JUST THE GRADIENT. On synthetic colour,
attaching a word to a coordinate appeared to repair the readout, and the deranged
control showed it did nothing of the kind: the model read the word and selected
AGAINST the coordinate, key_angle 5.4% against 16.7% chance. If real images behave
the same way, the substitution account is general. If mispairing the name leaves
accuracy sitting on the coordinate, then coordinates ARE read on natural images and
the synthetic conclusion is domain-limited. Both outcomes are worth the ten minutes.

TRIAL ADMISSION, fixed in PROTOCOL_FREEZE amendment 6 before this file existed:
exactly one instance of the queried category in the image; box area under 25% of
the image; box centre at least 8% of width and height inside its cell's border, so
a centre grazing a boundary cannot turn a correct answer into a scored error; COCO
categories used as given, not renamed and not filtered by how nameable they are.

Usage: VWM_DTYPE=4bit .venv/Scripts/python.exe scripts/coco_convention.py
Env: VWM_CC_N (trials, default 200), VWM_CC_DIRS, VWM_CC_DESCS
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
import torch
from PIL import Image

from vwm.local.qwen3_vl import Qwen3VLClient
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          BitsAndBytesConfig)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_ladder import SYS, hsh

# COCO val2017 parquet shards from the Hugging Face cache; HF_HOME overrides the default location.
HF_HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")

DTYPE = os.environ.get("VWM_DTYPE", "4bit")
# H4 across models. Empty means the primary Qwen3-VL client; any other value is
# loaded through the generic AutoModelForImageTextToText path used by crossmodel.py,
# so the replication shares the task code and differs only in the model.
MODEL_ID = os.environ.get("VWM_CC_MODEL", "")
N_TRIALS = int(os.environ.get("VWM_CC_N", "200"))
DIRECTIONS = os.environ.get("VWM_CC_DIRS", "locate,identify").split(",")
DESCS = os.environ.get("VWM_CC_DESCS",
                       "cellname,norm1000,pixel,pixel_gloss,pixel_mismatch").split(",")
# H8: two conventions never tested before, for the out-of-sample check declared in
# PROTOCOL_FREEZE amendment 8. `grid` is a very familiar FORMAT (A1..C3) that still
# requires the same image-to-coordinate mapping; `percent` uses familiar units but
# is, like absolute pixels, a convention described only in the prompt.
BASES = ("pixel", "norm1000", "grid", "percent")
GLOSSED = tuple(f"{b}_{k}" for b in BASES for k in ("gloss", "mismatch"))
MISMATCHED = tuple(f"{b}_mismatch" for b in BASES)


def base_of(desc):
    """The coordinate system a glossed rung is built on."""
    for b in BASES:
        if desc.startswith(b):
            return b
    return "pixel"

MAX_AREA_FRAC = 0.25      # box must not dominate the image
MARGIN_FRAC = 0.08        # box centre must sit this far inside its cell's border
N_CAT_OPTS = 6            # options in the `identify` direction

_SLUG = MODEL_ID.split("/")[-1].replace(".", "_") if MODEL_ID else "qwen3vl4b"
OUT_PATH = Path(os.environ.get("VWM_CC_OUT",
                               f"results/coco/convention_{_SLUG}.jsonl"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

VNAME = ("top", "middle", "bottom")
HNAME = ("left", "center", "right")
CELLS = [(r, c) for r in range(3) for c in range(3)]

# COCO 2017 category names in the contiguous 0-79 order this dataset uses.
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light "
        "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
        "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
        "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
        "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
        "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase "
        "scissors teddy_bear hair_drier toothbrush").split()


def catname(i):
    return COCO[i].replace("_", " ") if 0 <= i < len(COCO) else f"category{i}"


def derange(items, seed):
    n = len(items)
    for a in range(200):
        o = sorted(range(n), key=lambda j: hsh(seed, "der", a, j))
        if all(o[i] != i for i in range(n)):
            return [items[j] for j in o]
    return items[1:] + items[:1]


def cellname(r, c):
    return "center" if (r, c) == (1, 1) else f"{VNAME[r]}-{HNAME[c]}"


def describe(r, c, desc, W, H):
    cx, cy = int(W * (c + 0.5) / 3), int(H * (r + 0.5) / 3)
    if desc in GLOSSED:
        desc = base_of(desc)
    if desc == "pixel":
        return f"({cx}, {cy})"
    if desc == "cellname":
        return cellname(r, c)
    if desc == "norm1000":
        return f"({round(1000 * cx / W)}, {round(1000 * cy / H)})"
    if desc == "grid":
        return f"{'ABC'[c]}{r + 1}"
    if desc == "percent":
        return f"({round(100 * cx / W)}, {round(100 * cy / H)})"
    raise ValueError(desc)


def preamble(desc, W, H):
    if desc in GLOSSED:
        desc = base_of(desc)
    if desc == "cellname":
        return ""
    if desc == "norm1000":
        return ("Positions are given as (x, y) coordinates normalized to a 0 to 999 "
                "range, where (0, 0) is the top-left corner and x increases to the "
                "right and y increases downward. ")
    if desc == "grid":
        return ("The image is divided into a 3 by 3 grid. Columns are lettered A, "
                "B, C from left to right and rows are numbered 1, 2, 3 from top to "
                "bottom, so A1 is the top-left cell. ")
    if desc == "percent":
        return ("Positions are given as (x, y) percentages of the image width and "
                "height, where (0, 0) is the top-left corner and x increases to "
                "the right and y increases downward. ")
    return (f"Positions are given as (x, y) pixel coordinates in this {W} by {H} "
            f"image, where (0, 0) is the top-left corner and x increases to the "
            f"right and y increases downward. ")


def admit(row):
    """Returns (cell, category_index, box) for an admissible trial, else None.

    Every criterion here was fixed in PROTOCOL_FREEZE amendment 6 before this file
    was written, so none of them can be tuned once the numbers are visible."""
    W, H = row["width"], row["height"]
    o = row["objects"]
    cats, boxes, areas = o["category"], o["bbox"], o["area"]
    counts = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1
    for ci, (cat, box, area) in enumerate(zip(cats, boxes, areas)):
        if counts[cat] != 1:                       # must be the only one of its kind
            continue
        if area > MAX_AREA_FRAC * W * H:           # must not dominate the frame
            continue
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        col, r_ = int(cx / W * 3), int(cy / H * 3)
        col, r_ = min(col, 2), min(r_, 2)
        # centre must sit clear of its cell's borders
        lo_x, hi_x = col * W / 3, (col + 1) * W / 3
        lo_y, hi_y = r_ * H / 3, (r_ + 1) * H / 3
        if not (lo_x + MARGIN_FRAC * W <= cx <= hi_x - MARGIN_FRAC * W):
            continue
        if not (lo_y + MARGIN_FRAC * H <= cy <= hi_y - MARGIN_FRAC * H):
            continue
        return (r_, col), cat, [round(v, 1) for v in box]
    return None


def score(model, proc, img, prompt, ids):
    m = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
         {"role": "user", "content": [{"type": "image", "image": img},
                                      {"type": "text", "text": prompt}]}]
    t = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    e = proc(text=[t], images=[img], return_tensors="pt")
    e = {k: v.to(model.device) for k, v in e.items()}
    with torch.inference_mode():
        g = model.generate(**e, max_new_tokens=1, do_sample=False, temperature=None,
                           top_p=None, return_dict_in_generate=True,
                           output_scores=True)
    lg = g.scores[0][0].float()
    t_ = torch.tensor(ids, device=lg.device)
    return (int(torch.argmax(lg[t_])),
            float(torch.softmax(lg, -1)[t_].sum()),
            [round(float(x), 4) for x in lg[t_]])


def resolve_digits(model, proc, img, probe_prompt, k):
    """Pick the digit surface this model actually emits, by measuring it.

    Scoring a model on tokens it never produces is the failure this whole paper is
    about, and it was committed four times in this project before the check existed.
    A model whose digits all share a leading token has no first-token label readout
    at all and must be refused rather than reported at chance."""
    tok = getattr(proc, "tokenizer", proc)
    best = (None, -1.0, None)
    for nm, fn in (("bare", lambda w: w), ("space", lambda w: " " + w)):
        ids = [tok.encode(fn(str(j)), add_special_tokens=False)[0]
               for j in range(1, k + 1)]
        if len(set(ids)) != k:
            print(f"    {nm:<7} skipped: digits share a first token")
            continue
        _, m, _ = score(model, proc, img, probe_prompt, ids)
        print(f"    {nm:<7} candidate mass {m:.3f}")
        if m > best[1]:
            best = (nm, m, ids)
    nm, m, ids = best
    print(f"  -> digit readout {nm!r}, mass {m:.3f}")
    if ids is None or m < 0.5:
        raise RuntimeError(f"no digit surface carries mass for {MODEL_ID} "
                           f"(best {nm}={m:.3f}); refusing to report scores")
    return ids


def main():
    shards = sorted(glob.glob(os.path.join(HF_HUB, "datasets--"
                              "detection-datasets--coco/snapshots/*/data/val-*.parquet")))
    assert shards, "COCO val parquet not found"
    if MODEL_ID:
        # VWM_CC_BITS=8 runs the same model at higher precision, so that the
        # obvious reviewer question -- is the convention gap an artefact of 4-bit
        # quantisation -- is answered by measurement rather than by argument.
        if os.environ.get("VWM_CC_BITS") == "8":
            q = BitsAndBytesConfig(load_in_8bit=True)
        else:
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, quantization_config=q, device_map="cuda:0")
        model.eval()
    else:
        client = Qwen3VLClient(dtype=DTYPE); client.ensure_loaded()
        model, proc = client._model, client._processor
    tok = getattr(proc, "tokenizer", proc)
    print(f"model: {MODEL_ID or 'Qwen3-VL-4B (primary)'}", flush=True)

    from PIL import Image as _I
    probe = _I.new("RGB", (64, 64), (127, 127, 127))
    digits = resolve_digits(model, proc, probe,
                            "Which is it: 1=a, 2=b, 3=c, 4=d, 5=e, 6=f, 7=g, 8=h, "
                            "9=i? Answer with a single integer 1 to 9.", 9)
    print(f"shards {len(shards)}", flush=True)

    f = OUT_PATH.open("w", encoding="utf-8")
    n = seen = kept = 0
    t0 = time.time()

    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                if kept >= N_TRIALS:
                    break
                seen += 1
                a = admit(row)
                if a is None:
                    continue
                cell, cat, box = a
                W, H = row["width"], row["height"]
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                tid = f"coco{row['image_id']}"
                kept += 1

                order = sorted(range(9), key=lambda j: hsh(tid, "opt", j))
                opts = [CELLS[j] for j in order]
                tgt_loc = opts.index(cell)

                # distractor categories for the `identify` direction: present in
                # neither this image nor the answer, drawn deterministically
                present = set(row["objects"]["category"])
                pool = [i for i in range(len(COCO)) if i not in present]
                dis = [pool[hsh(tid, "cat", j) % len(pool)] for j in range(30)]
                seenc, cats = {cat}, [cat]
                for d in dis:
                    if d not in seenc and len(cats) < N_CAT_OPTS:
                        seenc.add(d); cats.append(d)
                corder = sorted(range(len(cats)), key=lambda j: hsh(tid, "co", j))
                cat_opts = [cats[j] for j in corder]
                tgt_cat = cat_opts.index(cat)

                for desc in DESCS:
                    names = [cellname(r, c) for (r, c) in opts]
                    gl = (derange(names, tid) if desc in MISMATCHED
                          else names if desc in GLOSSED else None)
                    if gl is not None:
                        bs = base_of(desc)
                        listed = ", ".join(
                            f"{j+1}={describe(r, c, bs, W, H)} [{gl[j]}]"
                            for j, (r, c) in enumerate(opts))
                    else:
                        listed = ", ".join(f"{j+1}={describe(r, c, desc, W, H)}"
                                           for j, (r, c) in enumerate(opts))
                    pre = preamble(desc, W, H)
                    cue = describe(*cell, desc, W, H)
                    if gl is not None:
                        cue = (f"{describe(*cell, base_of(desc), W, H)} "
                               f"[{gl[tgt_loc]}]")

                    prompts = {
                        "locate": (f"{pre}In this image, the {catname(cat)} is at "
                                   f"one of these positions: {listed}. Which one? "
                                   f"Answer with a single integer 1 to 9 and "
                                   f"nothing else."),
                        "identify": (f"{pre}In this image, what object is at "
                                     f"position {cue}? It is one of these: "
                                     + ", ".join(f"{j+1}={catname(c)}"
                                                 for j, c in enumerate(cat_opts))
                                     + f". Answer with a single integer 1 to "
                                       f"{len(cat_opts)} and nothing else."),
                    }
                    for d in DIRECTIONS:
                        ids = digits[:9] if d == "locate" else digits[:len(cat_opts)]
                        tgt = tgt_loc if d == "locate" else tgt_cat
                        rec = {"trial_id": tid, "direction": d, "desc": desc,
                               "W": W, "H": H, "category": catname(cat),
                               "cell": list(cell), "bbox": box,
                               "n_options": len(ids), "target_idx": tgt,
                               "prompt": prompts[d]}
                        if desc in MISMATCHED and d == "locate":
                            # key_name: the slot carrying the target's CELL NAME
                            # after derangement. Never equal to target_idx.
                            rec["key_name"] = gl.index(cellname(*cell))
                        try:
                            am, mass, sel = score(model, proc, img, prompts[d], ids)
                            rec.update(argmax=am, correct=bool(am == tgt),
                                       mass=mass, scores=sel)
                            n += 1
                        except Exception as e:                 # noqa: BLE001
                            rec["error"] = f"{type(e).__name__}:{e}"
                        f.write(json.dumps(rec) + chr(10)); f.flush()
                if kept % 25 == 0:
                    el = time.time() - t0
                    print(f"  {kept} trials ({seen} scanned)  {n} forwards  "
                          f"{el/max(1,n):.2f}s  {el/60:.1f}m", flush=True)
            if kept >= N_TRIALS:
                break
        if kept >= N_TRIALS:
            break

    f.close()
    print(f"CCDONE {kept} trials from {seen} images, {n} forwards -> {OUT_PATH}")


if __name__ == "__main__":
    main()
