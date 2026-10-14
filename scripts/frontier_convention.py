"""THE CONVENTION SWAP ON FRONTIER API MODELS.

The obvious objection to every result here is that the models are 2--7B open
checkpoints. This replays the SAME experiment against hosted frontier models.

Exact replication, not a re-implementation. Trial ids, prompts, option orders and
target indices are read back out of the local run's own jsonl, and the images are
reloaded from the same COCO parquet by image id. A frontier model therefore sees
byte-identical prompts over the identical 200 photographs, and every comparison is
paired per image.

SCORING DIFFERS, AND THE PAPER SAYS SO. Hosted APIs expose no usable first-token
distribution, so these rows are scored by generate-and-parse only. That is already
this project's primary measure and is what real benchmarks use, but the
first-token cross-check available locally is absent here and the absence is
reported rather than glossed.

WHAT THE QUANTISATION CONTROL MAKES THIS A TEST OF. Running Qwen3-VL-4B at 8-bit
instead of 4-bit improved every condition and WIDENED the convention gap, 48.5 to
52.0 points. If fidelity and gap move together, a stronger model should show a
larger gap, not a smaller one. That is the opposite of what most readers expect and
it is what this run tests.

Usage: GEMINI_API_KEY=... VWM_FR_MODEL=gemini-flash-latest \
       python scripts/frontier_convention.py
       (for GPT-4o: OPENAI_API_KEY=... VWM_FR_MODEL=gpt-4o)
Env: VWM_FR_MODEL, VWM_FR_N (images, default 200), VWM_FR_DESCS,
     GEMINI_API_KEY or OPENAI_API_KEY (read from the environment, never stored)
"""

from __future__ import annotations

import base64
import glob
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

# COCO val2017 parquet shards from the Hugging Face cache; HF_HOME overrides the default location.
HF_HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")

MODEL = os.environ.get("VWM_FR_MODEL", "gemini-flash-latest")
N = int(os.environ.get("VWM_FR_N", "200"))
DESCS = os.environ.get("VWM_FR_DESCS",
                       "cellname,norm1000,pixel,pixel_mismatch").split(",")
SRC = "results/coco/convention_qwen3vl4b.jsonl"
OUT = Path(f"results/frontier/{MODEL.replace('/', '_')}.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
IS_OPENAI = MODEL.startswith("gpt") or MODEL.startswith("o")


def post(url, body, hdr, tries=5):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers=hdr)
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            code = e.code
            msg = e.read()[:200].decode(errors="replace")
            if code in (429, 500, 502, 503, 529) and a < tries - 1:
                time.sleep(2 ** a * 2)
                continue
            raise RuntimeError(f"HTTP {code}: {msg}")
        except Exception:
            if a < tries - 1:
                time.sleep(2 ** a * 2)
                continue
            raise
    raise RuntimeError("retries exhausted")


def ask(img_bytes, prompt):
    b64 = base64.b64encode(img_bytes).decode()
    if IS_OPENAI:
        r = post("https://api.openai.com/v1/chat/completions",
                 {"model": MODEL, "max_tokens": 8, "temperature": 0,
                  "messages": [{"role": "user", "content": [
                      {"type": "text", "text": prompt},
                      {"type": "image_url",
                       "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]},
                 {"Content-Type": "application/json",
                  "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        txt = r["choices"][0]["message"]["content"]
        u = r.get("usage", {})
        return txt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    k = os.environ["GEMINI_API_KEY"]
    r = post(f"https://generativelanguage.googleapis.com/v1beta/models/"
             f"{MODEL}:generateContent?key={k}",
             {"contents": [{"role": "user", "parts": [
                 {"text": prompt},
                 {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
              # maxOutputTokens must NOT be tight on a reasoning model: Gemini 3.x
              # spends the budget on internal thinking tokens before emitting
              # anything, so a cap of 8 returned empty strings on 10/10 smoke
              # calls and would have recorded a capable model as producing
              # nothing. thinkingBudget 0 disables the reasoning pass; the cap is
              # left generous anyway. This is the paper own failure mode, met for
              # the fifth time, and caught only by reading the raw output.
              "generationConfig": {"temperature": 0, "maxOutputTokens": 512,
                                   "thinkingConfig": {"thinkingBudget": 0}}},
             {"Content-Type": "application/json"})
    c = r.get("candidates", [{}])[0]
    parts = c.get("content", {}).get("parts", [])
    txt = "".join(p.get("text", "") for p in parts)
    u = r.get("usageMetadata", {})
    return txt, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    plan = {}
    for r in rows:
        if r.get("direction") != "locate" or r.get("desc") not in DESCS:
            continue
        plan.setdefault(r["trial_id"], {})[r["desc"]] = r
    ids = sorted(plan)[:N]
    want = {int(t[4:]) for t in ids}
    print(f"{MODEL}: {len(ids)} images x {len(DESCS)} conditions = "
          f"{len(ids)*len(DESCS)} calls", flush=True)

    imgs = {}
    for shard in sorted(glob.glob(os.path.join(HF_HUB, "datasets--"
                                  "detection-datasets--coco/snapshots/*/data/val-*.parquet"))):
        for b in pq.ParquetFile(shard).iter_batches(batch_size=32):
            for row in b.to_pylist():
                if row["image_id"] in want:
                    imgs[f"coco{row['image_id']}"] = row["image"]["bytes"]
        if len(imgs) >= len(want):
            break
    print(f"loaded {len(imgs)} images", flush=True)

    f = OUT.open("w", encoding="utf-8")
    n = 0; pt = ct = 0; t0 = time.time(); errs = 0
    for tid in ids:
        if tid not in imgs:
            continue
        for desc in DESCS:
            src = plan[tid].get(desc)
            if src is None:
                continue
            rec = {"model": MODEL, "trial_id": tid, "desc": desc,
                   "direction": "locate", "n_options": src["n_options"],
                   "target_idx": src["target_idx"], "category": src["category"],
                   "cell": src["cell"], "scoring": "generate_and_parse"}
            if "key_name" in src:
                rec["key_name"] = src["key_name"]
            try:
                txt, a, b = ask(imgs[tid], src["prompt"])
                pt += a; ct += b
                m = re.search(r"\d+", txt or "")
                v = int(m.group()) - 1 if m else None
                if v is not None and not (0 <= v < src["n_options"]):
                    v = None
                rec.update(raw=txt, argmax=v, correct=bool(v == src["target_idx"]),
                           offvocab=bool(v is None))
                n += 1
            except Exception as e:                              # noqa: BLE001
                rec["error"] = f"{type(e).__name__}:{str(e)[:160]}"
                errs += 1
            f.write(json.dumps(rec) + chr(10)); f.flush()
        if n and n % 80 == 0:
            el = time.time() - t0
            print(f"  {n} calls  {errs} errors  {pt/1000:.0f}k in / {ct} out  "
                  f"{el/60:.1f}m", flush=True)
    f.close()
    rate = 2.50 if IS_OPENAI else 0.30
    print(f"FRDONE {MODEL}: {n} calls, {errs} errors -> {OUT}")
    print(f"  tokens: {pt} in, {ct} out   approx cost ${pt/1e6*rate:.2f}")


if __name__ == "__main__":
    main()
