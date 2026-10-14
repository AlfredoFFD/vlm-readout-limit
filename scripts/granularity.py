"""DOES THE CONVENTION GAP SURVIVE A FINER SPATIAL GRID?

The fair objection to the COCO result is that a 3x3 grid is coarse enough that a
model might answer from priors about where objects usually sit, and that pixel
coordinates are being punished for a precision the task never needed. The
robustness slicings already argue against it -- the gap holds for the smallest
objects and for centres furthest from any boundary -- but the direct test is to
make the grid finer and look again.

4x4, sixteen cells, chance 6.25%. Row and column names stay ordinary English
("upper-middle", "center-left"), so the control condition is still a phrase a model
can produce rather than a coordinate in disguise.

SCORING IS GENERATE-AND-PARSE, NOT FIRST-TOKEN, and deliberately so: labels 10-16
are not single tokens, so a restricted first-token readout over sixteen integers
does not exist. Rather than quietly score a broken alphabet -- the failure this
paper is about -- the run uses the generate-and-parse measure that is already this
project's primary one. Numbers here are therefore compared to each other, not to
the 3x3 first-token table.

Written as a separate file on purpose. coco_convention.py has audited outputs and
is not edited to support this.

Usage: VWM_DTYPE=4bit .venv/Scripts/python.exe scripts/granularity.py
Env: VWM_GR_N (images, default 200), VWM_GR_GRID (default 4)
"""

from __future__ import annotations

import glob
import io
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq
import torch
from PIL import Image

from vwm.local.qwen3_vl import Qwen3VLClient
from interface_ladder import SYS, hsh
from coco_convention import COCO, catname

# COCO val2017 parquet shards from the Hugging Face cache; HF_HOME overrides the default location.
HF_HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")

DTYPE = os.environ.get("VWM_DTYPE", "4bit")
N = int(os.environ.get("VWM_GR_N", "200"))
G = int(os.environ.get("VWM_GR_GRID", "4"))
MAX_AREA_FRAC = 0.25
CELL_MARGIN = 0.24            # of a cell's own width, matching the 3x3 setting
OUT = Path(os.environ.get("VWM_GR_OUT", f"results/coco/granularity_{G}x{G}.jsonl"))
OUT.parent.mkdir(parents=True, exist_ok=True)

ROWS4 = ("top", "upper-middle", "lower-middle", "bottom")
COLS4 = ("left", "center-left", "center-right", "right")
ROWS3 = ("top", "middle", "bottom")
COLS3 = ("left", "center", "right")
CELLS = [(r, c) for r in range(G) for c in range(G)]


def cellname(r, c):
    R, C = (ROWS4, COLS4) if G == 4 else (ROWS3, COLS3)
    return f"{R[r]}-{C[c]}"


def centre(r, c, W, H):
    return int(W * (c + 0.5) / G), int(H * (r + 0.5) / G)


def describe(r, c, desc, W, H):
    if desc == "cellname":
        return cellname(r, c)
    cx, cy = centre(r, c, W, H)
    return f"({cx}, {cy})"


def admit(row):
    W, H = row["width"], row["height"]
    o = row["objects"]
    counts = {}
    for c in o["category"]:
        counts[c] = counts.get(c, 0) + 1
    for cat, box, area in zip(o["category"], o["bbox"], o["area"]):
        if counts[cat] != 1 or area > MAX_AREA_FRAC * W * H:
            continue
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        col, r_ = min(int(cx / W * G), G - 1), min(int(cy / H * G), G - 1)
        mw, mh = CELL_MARGIN * W / G, CELL_MARGIN * H / G
        if not (col * W / G + mw <= cx <= (col + 1) * W / G - mw):
            continue
        if not (r_ * H / G + mh <= cy <= (r_ + 1) * H / G - mh):
            continue
        return (r_, col), cat, [round(v, 1) for v in box]
    return None


def generate(client, img, prompt):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": prompt}]}]
    return client._generate(msgs, max_new_tokens=8)


def main():
    shards = sorted(glob.glob(os.path.join(HF_HUB, "datasets--"
                              "detection-datasets--coco/snapshots/*/data/val-*.parquet")))
    client = Qwen3VLClient(dtype=DTYPE); client.ensure_loaded()
    K = G * G
    print(f"{G}x{G} grid, {K} options, chance {100/K:.2f}%", flush=True)

    f = OUT.open("w", encoding="utf-8")
    n = kept = seen = 0; t0 = time.time()
    for shard in shards:
        for b in pq.ParquetFile(shard).iter_batches(batch_size=16):
            for row in b.to_pylist():
                if kept >= N:
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
                order = sorted(range(K), key=lambda j: hsh(tid, "opt", j))
                opts = [CELLS[j] for j in order]
                tgt = opts.index(cell)
                for desc in ("cellname", "pixel"):
                    listed = ", ".join(f"{j+1}={describe(r, c, desc, W, H)}"
                                       for j, (r, c) in enumerate(opts))
                    pre = "" if desc == "cellname" else (
                        f"Positions are given as (x, y) pixel coordinates in this "
                        f"{W} by {H} image, where (0, 0) is the top-left corner and "
                        f"x increases to the right and y increases downward. ")
                    prompt = (f"{pre}In this image, the {catname(cat)} is at one of "
                              f"these positions: {listed}. Which one? Answer with a "
                              f"single integer 1 to {K} and nothing else.")
                    rec = {"trial_id": tid, "grid": G, "desc": desc, "n_options": K,
                           "target_idx": tgt, "cell": list(cell), "bbox": box,
                           "W": W, "H": H, "category": catname(cat),
                           "prompt": prompt, "scoring": "generate_and_parse"}
                    try:
                        txt = generate(client, img, prompt)
                        m = re.search(r"\d+", txt or "")
                        v = int(m.group()) - 1 if m else None
                        if v is not None and not (0 <= v < K):
                            v = None
                        rec.update(raw=txt, argmax=v, correct=bool(v == tgt),
                                   offvocab=bool(v is None))
                        n += 1
                    except Exception as e:                      # noqa: BLE001
                        rec["error"] = f"{type(e).__name__}:{e}"
                    f.write(json.dumps(rec) + chr(10)); f.flush()
                if kept % 40 == 0:
                    print(f"  {kept} images ({seen} scanned)  {n} calls  "
                          f"{(time.time()-t0)/60:.1f}m", flush=True)
            if kept >= N:
                break
        if kept >= N:
            break
    f.close()
    print(f"GRDONE {kept} images, {n} calls -> {OUT}")


if __name__ == "__main__":
    main()
