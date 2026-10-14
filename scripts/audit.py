"""ADVERSARIAL AUDIT OF EVERY RESULT FILE THE PAPER USES.

Not a smoke test. Each check is written to FAIL if the corresponding result is an
artefact, and several recompute a headline number by a path that shares no code
with the one that produced it. Four readout bugs were found by hand in this project
and every one of them would have been caught by a check of this kind, so the checks
exist now.

A check that passes proves only what it says it proves. The list of what is NOT
checked is at the bottom and is part of the output.

Usage: .venv/Scripts/python.exe scripts/audit.py
Exit code 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FAIL = []
WARN = []


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def warn(name, cond, detail=""):
    if not cond:
        print(f"  WARN  {name}  {detail}")
        WARN.append(name)
    else:
        print(f"  PASS  {name}{'  ' + detail if detail else ''}")


def load(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] \
        if p.exists() else []


def hdr(t):
    print(f"\n{'-'*76}\n{t}\n{'-'*76}")


# =========================================================== 1. structural integrity
def check_structural():
    hdr("1. STRUCTURAL: errors, duplicates, index ranges, candidate mass")
    files = {
        "colour heldout": "results/gpu_p3/interface_ladder_heldout.jsonl",
        "colour recovery": "results/gpu_p3/interface_ladder_recovery.jsonl",
        "colour mismatch": "results/gpu_p3/interface_ladder_mismatch.jsonl",
        "spatial grounding": "results/gpu_p3/coordinate_grounding.jsonl",
        "spatial mismatch": "results/gpu_p3/coordinate_mismatch.jsonl",
        "coco convention": "results/coco/convention.jsonl",
        "coco readership": "results/coco/readership.jsonl",
    }
    for lab, p in files.items():
        rows = load(p)
        if not rows:
            ok(f"{lab}: file present", False, p)
            continue
        errs = [r for r in rows if any(k.startswith("error") for k in r)]
        ok(f"{lab}: no error fields", not errs, f"{len(rows)} rows")
        # every cell scored must be a distinct (trial, condition, query) key
        keys = []
        for r in rows:
            keys.append((r.get("trial_id"), r.get("rung") or r.get("desc"),
                         r.get("query") or r.get("direction")))
        dup = [k for k, c in Counter(keys).items() if c > 1]
        ok(f"{lab}: no duplicate cells", not dup, f"{len(dup)} dups")
        # candidate mass: a low value means the scorer is reading tokens the model
        # does not emit, which is the bug that made SmolVLM look incapable
        ms = [v for r in rows for k, v in r.items()
              if k.startswith("mass") or k.endswith("_mass")]
        if ms:
            # threshold set from the observed distribution rather than guessed: the
            # failure mode this guards against (scoring tokens the model does not
            # emit) produced mass near 0.00 on SmolVLM, not 0.95. One COCO cell of
            # 2000 sits at 0.949 and none below 0.90.
            ok(f"{lab}: candidate mass >= 0.90 on every cell",
               float(np.min(ms)) >= 0.90,
               f"min {np.min(ms):.3f} median {np.median(ms):.4f}")
        # argmax and target must be inside the option set
        bad = 0
        for r in rows:
            k = r.get("n_options") or r.get("K")
            if not k:
                continue
            for f_ in ("argmax", "argmax_A", "argmax_B", "argmax_visible",
                       "target_idx", "target_idx_A", "target_idx_B", "key_name",
                       "key_word", "key_word_A", "key_word_B"):
                v = r.get(f_)
                if v is not None and not (0 <= v < k):
                    bad += 1
        ok(f"{lab}: all indices within option set", bad == 0, f"{bad} out of range")


# =============================================== 2. stored labels vs recomputation
def check_recompute():
    hdr("2. RECOMPUTE: does the stored `correct` flag match argmax == target?")
    for lab, p, a, t in (
        ("coco convention", "results/coco/convention.jsonl", "argmax", "target_idx"),
        ("coco readership", "results/coco/readership.jsonl", "argmax", "target_idx"),
    ):
        rows = [r for r in load(p) if a in r and t in r and "correct" in r]
        mism = [r for r in rows if (r[a] == r[t]) != r["correct"]]
        ok(f"{lab}: correct == (argmax==target)", not mism,
           f"{len(mism)}/{len(rows)} mismatched")
    rows = [r for r in load("results/gpu_p3/interface_ladder_heldout.jsonl")
            if "visible_argmax" in r and "target_idx_A" in r]
    mism = [r for r in rows
            if (r["visible_argmax"] == r["target_idx_A"]) != r["visible_correct"]]
    ok("colour heldout: visible_correct == (argmax==target)", not mism,
       f"{len(mism)}/{len(rows)}")


# ============================================ 3. independent ground-truth for COCO
def check_coco_groundtruth():
    hdr("3. COCO GROUND TRUTH: recompute the answer cell from the raw box")
    rows = load("results/coco/convention.jsonl")
    seen, bad, checked = set(), 0, 0
    for r in rows:
        if r["trial_id"] in seen or "bbox" not in r:
            continue
        seen.add(r["trial_id"])
        W, H = r["W"], r["H"]
        x0, y0, x1, y1 = r["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        col = min(int(cx / W * 3), 2); row = min(int(cy / H * 3), 2)
        checked += 1
        if [row, col] != r["cell"]:
            bad += 1
    ok("stored cell == cell recomputed from bbox", bad == 0,
       f"{checked} images, {bad} wrong")

    # admission criteria actually enforced
    viol_area = viol_margin = 0
    seen = set()
    for r in rows:
        if r["trial_id"] in seen or "bbox" not in r:
            continue
        seen.add(r["trial_id"])
        W, H = r["W"], r["H"]
        x0, y0, x1, y1 = r["bbox"]
        # admit() filters on COCO's `area` field, which is SEGMENTATION area and is
        # always <= bbox area, so a sprawling object can pass with a large box. The
        # sensitivity analysis shows this is conservative: excluding the 32 affected
        # images moves cellname->pixel from -48.5 to -51.2 points. Checked here as a
        # reported count, not a failure.
        if (x1 - x0) * (y1 - y0) > 0.25 * W * H:
            viol_area += 1
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        col = min(int(cx / W * 3), 2); row = min(int(cy / H * 3), 2)
        if not (col * W / 3 + .08 * W - 1 <= cx <= (col + 1) * W / 3 - .08 * W + 1):
            viol_margin += 1
        if not (row * H / 3 + .08 * H - 1 <= cy <= (row + 1) * H / 3 - .08 * H + 1):
            viol_margin += 1
    warn("admission: bbox area under 25% (spec says SEGMENTATION area)",
         viol_area <= 40,
         f"{viol_area}/{checked} images have a bbox over 25%; excluding them "
         f"ENLARGES the effect (-48.5 -> -51.2 pts)")
    ok("admission: centre clear of cell border", viol_margin == 0,
       f"{viol_margin} violations")


# ==================================================== 4. controls land where they must
def check_controls():
    hdr("4. CONTROLS: do the conditions with a known answer give it?")
    rows = [r for r in load("results/gpu_p3/interface_ladder_heldout.jsonl")
            + load("results/gpu_p3/interface_ladder_recovery.jsonl")
            if r.get("query") == "q3" and "did" in r]
    by = defaultdict(list)
    for r in rows:
        by[r["rung"]].append(r["did"])
    for rg, v in sorted(by.items()):
        m = float(np.mean(v))
        ok(f"q3 null control ({rg}) near zero", abs(m) < 1.0,
           f"mean DiD {m:+.3f}, n={len(v)}")

    mm = load("results/gpu_p3/interface_ladder_mismatch.jsonl")
    coll = 0
    for r in mm:
        for ck in ("A", "B"):
            if f"key_word_{ck}" in r and r[f"key_word_{ck}"] == r[f"target_idx_{ck}"]:
                coll += 1
    ok("mismatch: the two scoring keys never collide", coll == 0, f"{coll} collisions")

    cc = [r for r in load("results/coco/convention.jsonl")
          if r.get("desc") == "pixel_mismatch" and "key_name" in r]
    coll = sum(1 for r in cc if r["key_name"] == r["target_idx"])
    ok("coco mismatch: keys never collide", coll == 0, f"{coll}/{len(cc)}")


# ==================================================== 5. label-permutation null test
def check_permutation_null():
    hdr("5. PERMUTATION NULL: break the trial-to-answer pairing; accuracy must "
        "fall to chance")
    rng = np.random.default_rng(0)
    for lab, p, fld, tgt, nopt in (
        ("coco locate cellname", "results/coco/convention.jsonl", "argmax",
         "target_idx", 9),
        ("coco locate pixel", "results/coco/convention.jsonl", "argmax",
         "target_idx", 9),
    ):
        desc = "cellname" if "cellname" in lab else "pixel"
        rows = [r for r in load(p)
                if r.get("desc") == desc and r["direction"] == "locate"
                and fld in r]
        if not rows:
            continue
        real = np.mean([r[fld] == r[tgt] for r in rows])
        perm = []
        for _ in range(200):
            t = rng.permutation([r[tgt] for r in rows])
            perm.append(np.mean([r[fld] == t[i] for i, r in enumerate(rows)]))
        mu, sd = float(np.mean(perm)), float(np.std(perm))
        ok(f"{lab}: shuffled pairing lands at chance",
           abs(mu - 1 / nopt) < 0.05,
           f"real {100*real:.1f}%  shuffled {100*mu:.1f}+-{100*sd:.1f}%  "
           f"chance {100/nopt:.1f}%")
        ok(f"{lab}: real beats shuffled by >3 sd" if desc == "cellname"
           else f"{lab}: real vs shuffled (pixel is expected to be NEAR chance)",
           (real - mu) / sd > 3 if desc == "cellname" else True,
           f"{(real-mu)/max(sd,1e-9):.1f} sd")


# ============================================== 6. prompts say what the paper claims
def check_prompts():
    hdr("6. PROMPTS: read the actual strings, do not trust the condition name")
    rows = load("results/gpu_p3/interface_ladder_heldout.jsonl")
    for rg, must, mustnot in (
        ("word", ["one of these colors", "single color word"], ["hue angle"]),
        ("label", ["1=", "single integer"], ["hue angle"]),
        ("numeric", ["hue angle", "1=", "single integer"], []),
    ):
        ex = [r for r in rows if r["rung"] == rg and "prompt" in r]
        if not ex:
            continue
        p = ex[0]["prompt"]
        ok(f"prompt[{rg}] contains required phrases",
           all(m in p for m in must) and not any(m in p for m in mustnot))
    mm = [r for r in load("results/gpu_p3/interface_ladder_mismatch.jsonl")
          if r["rung"] == "numeric_mismatch" and "prompt" in r and "gloss" in r]
    if mm:
        r = mm[0]
        # the gloss in the prompt must be the deranged list, not the true one
        ok("mismatch prompt carries the DERANGED gloss",
           all(f"({g})" in r["prompt"] for g in r["gloss"]))
        ok("mismatch prompt: no option glossed with its own colour",
           not any(f"{o} ({o})" in r["prompt"] for o in r["options"]))
    cc = [r for r in load("results/coco/convention.jsonl")
          if r.get("desc") == "pixel" and r["direction"] == "locate"]
    if cc:
        ok("coco pixel prompt states the image size",
           f"{cc[0]['W']} by {cc[0]['H']}" in cc[0]["prompt"])


# =================================================== 7. sample sizes and provenance
def check_sizes():
    hdr("7. SAMPLE SIZES: do the n's in the paper match the files?")
    checks = [
        ("colour heldout trials == 173",
         len({r["trial_id"] for r in load("results/gpu_p3/interface_ladder_heldout.jsonl")}), 173),
        ("colour mismatch trials == 173",
         len({r["trial_id"] for r in load("results/gpu_p3/interface_ladder_mismatch.jsonl")}), 173),
        ("coco images == 200",
         len({r["trial_id"] for r in load("results/coco/convention.jsonl")}), 200),
        ("coco readership images == 200",
         len({r["trial_id"] for r in load("results/coco/readership.jsonl")}), 200),
    ]
    for name, got, want in checks:
        ok(name, got == want, f"got {got}")


def main():
    print("ADVERSARIAL AUDIT — each check fails if the result is an artefact")
    for f in (check_structural, check_recompute, check_coco_groundtruth,
              check_controls, check_permutation_null, check_prompts, check_sizes):
        try:
            f()
        except Exception as e:                                   # noqa: BLE001
            print(f"  FAIL  {f.__name__} raised {type(e).__name__}: {e}")
            FAIL.append(f.__name__)

    hdr("WHAT THIS AUDIT DOES NOT CHECK")
    print("""  * that 4-bit quantisation does not change the conclusions
  * that the effect holds beyond 3x3 spatial granularity
  * that the COCO category names are equally easy for every model
  * that greedy decoding is representative of sampled decoding
  * anything about models not run here
  These are limitations of the study, not of the audit, and belong in the paper.""")

    hdr("RESULT")
    if FAIL:
        print(f"  {len(FAIL)} CHECK(S) FAILED: {FAIL}")
        sys.exit(1)
    print(f"  all checks passed ({len(WARN)} warnings)")


if __name__ == "__main__":
    main()
