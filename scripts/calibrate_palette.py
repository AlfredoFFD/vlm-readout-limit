"""LET THE MODEL NAME THE RENDERER, INSTEAD OF NAMING IT FOR THE MODEL.

The project's worst bug was describing the CIELAB wheel with conventional HSV
anchors the renderer does not honour. The first repair measured the anchors from
the renderer instead of assuming them -- better, but still a coordinate-to-word map
written by hand. The smoke run showed that map failing in exactly the predictable
place: wheel 23 is HSV hue 0.0, so a band table calls it "red", but it renders as
rgb(238,146,146), which is salmon, and the model named it correctly as something
else on 0% of trials. Saturation and lightness were dropped on the way from pixels
to a word, and the dropped information is what decides the name.

There is no table that fixes this in general, because the question "what is this
colour called" is not answerable from coordinates. So stop answering it. This asks
the model, with the swatch on screen, in free report, and keeps its modal answer.
The response alphabet becomes the model's own naming of the renderer's output.

  sweep       one swatch per wheel angle, rendered exactly as the stimuli are
  free report "what color is this square? one word"
  modal name  over REPS renders that differ in square size and position, so a name
              only survives if it is stable to nuisance variation rather than to
              one particular crop

  anchors     the selected eight: distinct modal names, every name held on all REPS
              renders, and the wheel angles as far apart as that allows

This is calibration, not scoring, and the distinction matters for what may later be
claimed. Choosing words the model produces makes the readout usable; it does not by
itself demonstrate colour memory. Visible accuracy measured through an alphabet
calibrated this way is partly true by construction, and must be reported as the
interface control it is rather than as a perception result.

Usage: VWM_DTYPE=4bit .venv/Scripts/python.exe scripts/calibrate_palette.py
Env: VWM_CP_STEP (wheel sampling step in deg, default 10), VWM_CP_REPS (default 3)
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw

from vwm.stimuli.colorwheel import wheel_rgb, angular_distance
from vwm.stimuli.geometry import CANVAS_W, CANVAS_H, SQUARE_PX
from vwm.local.qwen3_vl import Qwen3VLClient

DTYPE = os.environ.get("VWM_DTYPE", "4bit")
STEP = int(os.environ.get("VWM_CP_STEP", "10"))
REPS = int(os.environ.get("VWM_CP_REPS", "3"))
N_ANCHORS = int(os.environ.get("VWM_CP_ANCHORS", "8"))
BG = (127, 127, 127)

OUT = Path(os.environ.get("VWM_CP_OUT", "results/gpu_p3/palette_calibration.json"))
IMG_DIR = Path("results/gpu_p3/_cp_imgs"); IMG_DIR.mkdir(parents=True, exist_ok=True)

SYS = ("You are a careful visual working memory experiment participant. "
       "Follow the instructions exactly and answer concisely.")
Q = ("What color is the square in this image? Answer with a single common color "
     "word and nothing else.")

# Nuisance variation the name must survive. Same renderer, same background, only
# the square's size and placement move.
VARIANTS = [(SQUARE_PX, 0.5, 0.5), (int(SQUARE_PX * 1.4), 0.32, 0.36),
            (int(SQUARE_PX * 0.8), 0.68, 0.64)][:max(1, REPS)]


def swatch(hue, size, fx, fy, path):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(img)
    cx, cy = int(CANVAS_W * fx), int(CANVAS_H * fy)
    h = size // 2
    d.rectangle([cx - h, cy - h, cx + h - 1, cy + h - 1],
                fill=wheel_rgb(hue), outline=(64, 64, 64), width=2)
    img.save(path)
    return img


def name_it(client, img):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": Q}]}]
    out = client._generate(msgs, max_new_tokens=6)
    w = re.findall(r"[a-zA-Z]+", out.lower())
    return w[0] if w else ""


def main():
    client = Qwen3VLClient(dtype=DTYPE); client.ensure_loaded()

    angles = list(range(0, 360, STEP))
    table = {}
    for a in angles:
        names = []
        for i, (sz, fx, fy) in enumerate(VARIANTS):
            im = swatch(a, sz, fx, fy, IMG_DIR / f"h{a:03d}_{i}.png")
            names.append(name_it(client, im))
        c = Counter(names)
        modal, k = c.most_common(1)[0]
        table[a] = {"names": names, "modal": modal, "agree": k / len(names),
                    "rgb": wheel_rgb(a)}
        print(f"  wheel {a:3d}  rgb {wheel_rgb(a)}  -> {names}  modal={modal} "
              f"({k}/{len(names)})", flush=True)

    # unanimous angles only: a name that flips with square size is not a name
    solid = {a: v["modal"] for a, v in table.items() if v["agree"] == 1.0}
    print(f"\n{len(solid)}/{len(angles)} angles name unanimously across "
          f"{len(VARIANTS)} renders")

    byname = {}
    for a, nm in solid.items():
        byname.setdefault(nm, []).append(a)
    print("stable names and their wheel support:")
    for nm, aa in sorted(byname.items(), key=lambda x: -len(x[1])):
        print(f"  {nm:<12} {len(aa):>2} angles  {aa}")

    # greedy pick: one angle per name, maximising the minimum pairwise wheel
    # separation, so the chosen anchors are both nameably distinct and far apart
    cands = []
    for nm, aa in byname.items():
        # the angle furthest from that name's edges is its most typical instance
        best = max(aa, key=lambda a: min(
            (angular_distance(a, b) for b in solid if solid[b] != nm), default=180))
        cands.append((nm, best))
    chosen = []
    pool = list(cands)
    while pool and len(chosen) < N_ANCHORS:
        if not chosen:
            pick = max(pool, key=lambda c: min(
                (angular_distance(c[1], d[1]) for d in pool if d is not c), default=180))
        else:
            pick = max(pool, key=lambda c: min(
                angular_distance(c[1], d[1]) for d in chosen))
        chosen.append(pick); pool.remove(pick)
    chosen.sort(key=lambda c: c[1])

    seps = [angular_distance(chosen[i][1], chosen[(i + 1) % len(chosen)][1])
            for i in range(len(chosen))]
    print(f"\nCHOSEN {len(chosen)} anchors (min separation {min(seps):.0f} deg):")
    print("NAMED = [" + ", ".join(f"({a}, {nm!r})" for nm, a in chosen) + "]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"step": STEP, "variants": VARIANTS, "table": table,
               "chosen": [{"wheel": a, "name": nm} for nm, a in chosen]},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
