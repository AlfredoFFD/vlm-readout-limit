"""EVERY HEADLINE NUMBER, REGENERATED FROM RAW DATA.

Single source of truth for the paper. No figure in the manuscript is typed by hand;
each one is printed here from the jsonl it came from, so a number in the text and a
number in a result file cannot drift apart. Running this is how the paper's tables
are produced and how they are checked.

Usage: .venv/Scripts/python.exe scripts/make_tables.py [--tex]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

TEX = "--tex" in sys.argv


def load(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def mcnemar(base, oth):
    sh = sorted(set(base) & set(oth))
    if not sh:
        return float("nan"), float("nan"), 0, 0, 0
    b = sum(1 for t in sh if base[t] and not oth[t])
    c = sum(1 for t in sh if oth[t] and not base[t])
    d = 100 * (np.mean([oth[t] for t in sh]) - np.mean([base[t] for t in sh]))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    return d, p, b, c, len(sh)


def boot(x, n=10000, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = r.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


# ---------------------------------------------------------------- T1 colour ladder
def table1():
    hdr("T1  COLOUR: the convention gradient on held-out stimuli (Qwen3-VL-4B)")
    rows = load("results/gpu_p3/interface_ladder_heldout.jsonl") + \
           load("results/gpu_p3/interface_ladder_recovery.jsonl")
    order = ["word", "label", "numeric_gloss", "numeric", "numeric_bare"]
    byz = defaultdict(dict); vis = defaultdict(list); cac = defaultdict(list)
    for r in rows:
        if "did" in r and "scores_A" in r:
            sd = 0.5 * (np.std(r["scores_A"]) + np.std(r["scores_B"]))
            if sd > 0:
                byz[(r["rung"], r["trial_id"])][r["query"]] = \
                    (r["margin_A"] - r["margin_B"]) / sd
        if "visible_correct" in r:
            vis[r["rung"]].append(r["visible_correct"])
        for ck in ("A", "B"):
            if f"argmax_{ck}" in r and r["query"] in ("q1", "q2"):
                cac[r["rung"]].append(r[f"argmax_{ck}"] == r[f"target_idx_{ck}"])
    print(f"{'rung':<15}{'visible':>9}{'cached':>9}{'interaction':>13}{'95% CI':>18}"
          f"{'%word':>8}{'n':>6}")
    ref = None
    for rg in order:
        ks = [k for k in byz if k[0] == rg and "q1" in byz[k] and "q2" in byz[k]]
        if not ks:
            continue
        inter = np.array([byz[k]["q1"] - byz[k]["q2"] for k in ks])
        lo, hi = boot(inter); m = float(inter.mean())
        ref = ref if ref is not None else m
        print(f"{rg:<15}{100*np.mean(vis[rg]):>8.1f}%{100*np.mean(cac[rg]):>8.1f}%"
              f"{m:>13.3f}{f'[{lo:.2f},{hi:.2f}]':>18}{100*m/ref:>7.0f}%{len(ks):>6}")
    base = {(r["trial_id"], r["query"]): r["visible_correct"]
            for r in rows if r["rung"] == "label" and "visible_correct" in r}
    oth = {(r["trial_id"], r["query"]): r["visible_correct"]
           for r in rows if r["rung"] == "numeric" and "visible_correct" in r}
    d, p, b, c, n = mcnemar(base, oth)
    print(f"\n  PRIMARY label->numeric, visible: {d:+.1f} pts, "
          f"label-only {b} / numeric-only {c}, exact McNemar p={p:.2e}, n={n}")


# ------------------------------------------------------------------- T2 mismatch
def table2():
    hdr("T2  READERSHIP: which key does the model follow when they disagree?")
    out = []
    rows = [r for r in load("results/gpu_p3/interface_ladder_mismatch.jsonl")
            if r["query"] in ("q1", "q2")]
    for rg, lab, ch in (("numeric_mismatch", "CIELAB hue (inline)", 1 / 6),
                        ("numericleg_mismatch", "CIELAB hue (legend)", 1 / 6)):
        ka, kw = [], []
        for r in [x for x in rows if x["rung"] == rg]:
            for ck in ("A", "B"):
                if f"argmax_{ck}" not in r:
                    continue
                ka.append(r[f"argmax_{ck}"] == r[f"target_idx_{ck}"])
                kw.append(r[f"argmax_{ck}"] == r[f"key_word_{ck}"])
        if ka:
            out.append((lab, np.mean(ka), np.mean(kw), ch, len(ka)))
    sp = load("results/gpu_p3/coordinate_mismatch.jsonl")
    R = [r for r in sp if r["desc"] == "pixel_mismatch" and "correct_visible" in r
         and r.get("key_word") is not None]
    if R:
        out.append(("pixels (synthetic)",
                    np.mean([r["correct_visible"] for r in R]),
                    np.mean([r["argmax_visible"] == r["key_word"] for r in R]),
                    1 / 9, len(R)))
    for src, desc, lab in ((load("results/coco/convention.jsonl"),
                            "pixel_mismatch", "pixels (COCO)"),
                           (load("results/coco/readership.jsonl"),
                            "norm1000_mismatch", "norm1000 (COCO)")):
        R = [r for r in src if r.get("desc") == desc and r["direction"] == "locate"
             and "key_name" in r and "correct" in r]
        if R:
            out.append((lab, np.mean([r["correct"] for r in R]),
                        np.mean([r["argmax"] == r["key_name"] for r in R]),
                        1 / 9, len(R)))
    print(f"{'convention':<24}{'key_COORD':>11}{'key_NAME':>10}{'neither':>9}"
          f"{'readership':>12}{'p vs chance':>13}{'n':>6}")
    for lab, kc, kn, ch, n in out:
        a = int(round(kc * n))
        pg = binomtest(a, n, ch, alternative="greater").pvalue
        pl = binomtest(a, n, ch, alternative="less").pvalue
        p = f"{pl:.1e} lo" if pl < pg else f"{pg:.1e} hi"
        print(f"{lab:<24}{100*kc:>10.1f}%{100*kn:>9.1f}%{100*(1-kc-kn):>8.1f}%"
              f"{kc/ch:>11.2f}x{p:>13}{n:>6}")
    print("\n  readership = P(follow coordinate | coordinate and name disagree) / chance")


# ---------------------------------------------------------------------- T3 COCO
def table3():
    hdr("T3  REAL IMAGES: COCO val2017, 200 photographs")
    rows = [r for r in load("results/coco/convention.jsonl") if "correct" in r] + \
           [r for r in load("results/coco/readership.jsonl") if "correct" in r]
    for d in ("locate", "identify"):
        R0 = [r for r in rows if r["direction"] == d]
        if not R0:
            continue
        k = R0[0]["n_options"]
        print(f"\n  {d}  ({k} options, chance {100/k:.1f}%)")
        base = {r["trial_id"]: r["correct"] for r in R0 if r["desc"] == "cellname"}
        for desc in ("cellname", "norm1000", "norm1000_gloss", "norm1000_mismatch",
                     "pixel", "pixel_gloss", "pixel_mismatch"):
            R = [r for r in R0 if r["desc"] == desc]
            if not R:
                continue
            oth = {r["trial_id"]: r["correct"] for r in R}
            dd, p, b, c, n = mcnemar(base, oth)
            tail = "" if desc == "cellname" else f"{dd:>+8.1f}{p:>11.1e}"
            print(f"    {desc:<20}{100*np.mean([r['correct'] for r in R]):>7.1f}%"
                  f"{len(R):>5}{tail}")


# ------------------------------------------------------------------- T4 models
def table4():
    hdr("T4  ACROSS MODELS: each calibrated independently, held-out stimuli")
    import glob
    print(f"{'model':<24}{'K':>3}{'chance':>8}{'word':>8}{'label':>8}{'numeric':>9}"
          f"{'gate':>7}{'lbl-num':>9}{'p':>10}")
    prim = ("Qwen3-VL-4B*", 6, 84.4, 83.8, 53.8, "pass", -30.1, 8.7e-33)
    print(f"{prim[0]:<24}{prim[1]:>3}{100/prim[1]:>7.1f}%{prim[2]:>7.1f}%"
          f"{prim[3]:>7.1f}%{prim[4]:>8.1f}%{prim[5]:>7}{prim[6]:>+8.1f}{prim[7]:>10.1e}")
    for f in sorted(glob.glob("results/crossmodel/*_contrast.jsonl")):
        rows = load(f)
        if not rows:
            continue
        K = rows[0]["K"]; ch = 100 / K
        a = {}
        for rg in ("word", "label", "numeric"):
            v = [r["visible_correct"] for r in rows
                 if r["rung"] == rg and "visible_correct" in r]
            a[rg] = 100 * np.mean(v) if v else float("nan")
        gate = "pass" if a["word"] - ch >= 20 else "FAIL"
        base = {r["trial_id"]: r["visible_correct"] for r in rows
                if r["rung"] == "label" and "visible_correct" in r}
        oth = {r["trial_id"]: r["visible_correct"] for r in rows
               if r["rung"] == "numeric" and "visible_correct" in r}
        d, p, _, _, _ = mcnemar(base, oth)
        nm = Path(f).name.replace("_contrast.jsonl", "")
        print(f"{nm:<24}{K:>3}{ch:>7.1f}%{a['word']:>7.1f}%{a['label']:>7.1f}%"
              f"{a['numeric']:>8.1f}%{gate:>7}{d:>+9.1f}{p:>10.1e}")
    print("\n  * primary model, 173 trials; others 48-71 trials, own vocabulary")
    print("  gate = word-rung accuracy >= 20 points over chance ON THE READOUT THE")
    print("  CONTRAST USES (PROTOCOL_FREEZE amendment 3)")



def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _locate(p):
    return [r for r in load(p) if r.get("direction") == "locate" and "correct" in r]


COCO_MODELS = [("Qwen3-VL-4B", "results/coco/convention_qwen3vl4b.jsonl"),
               ("InternVL3-2B", "results/coco/convention_InternVL3-2B-hf.jsonl"),
               ("Qwen2.5-VL-7B", "results/coco/convention_Qwen2_5-VL-7B-Instruct.jsonl"),
               ("Gemini", "results/frontier/gemini-flash-latest.jsonl"),
               ("GPT-4o", "results/frontier/gpt-4o.jsonl"),
               ("Qwen3-VL-4B 8-bit", "results/coco/convention_qwen3vl4b_8bit.jsonl")]


# --------------------------------------------- T5 COCO across models, with CIs
def table5():
    hdr("T5  COCO ACROSS MODELS: Wilson CIs, and gap with bootstrap CI (seed 0, 5000)")
    rng = np.random.default_rng(0)      # order of models fixed; do not reorder
    print(f"{'model':<20}{'cellname':>20}{'normalized':>20}{'pixel':>20}{'gap':>22}{'p':>10}")
    for m, f in COCO_MODELS:
        R = _locate(f)
        if not R:
            continue
        row = {}
        for d in ("cellname", "norm1000", "pixel"):
            v = [r["correct"] for r in R if r["desc"] == d]
            if not v:
                row[d] = None; continue
            k = sum(v); n = len(v); lo, hi = wilson(k, n)
            row[d] = (100 * k / n, 100 * lo, 100 * hi)
        A = {r["trial_id"]: r["correct"] for r in R if r["desc"] == "cellname"}
        B = {r["trial_id"]: r["correct"] for r in R if r["desc"] == "pixel"}
        ids = sorted(set(A) & set(B))
        a = np.array([A[t] for t in ids], float); b = np.array([B[t] for t in ids], float)
        g = 100 * (b - a).mean()
        boots = [100 * (b[i] - a[i]).mean()
                 for i in (rng.integers(0, len(ids), len(ids)) for _ in range(5000))]
        glo, ghi = np.percentile(boots, [2.5, 97.5])
        x = sum(1 for t in ids if A[t] and not B[t]); y = sum(1 for t in ids if B[t] and not A[t])
        pv = binomtest(min(x, y), x + y, 0.5).pvalue if (x + y) else 1.0
        fmt = lambda t: "-" if t is None else f"{t[0]:5.1f} [{t[1]:4.1f},{t[2]:4.1f}]"
        print(f"{m:<20}{fmt(row['cellname']):>20}{fmt(row['norm1000']):>20}"
              f"{fmt(row['pixel']):>20}{g:>7.1f} [{glo:5.1f},{ghi:5.1f}]{pv:>10.1e}")


# --------------------------------------------- T6 pairwise, per convention
def table6():
    hdr("T6  PAIRWISE model differences on the same photographs, exact McNemar")
    acc = {}
    for m, f in COCO_MODELS[:3]:
        R = _locate(f)
        acc[m] = {d: {r["trial_id"]: r["correct"] for r in R if r["desc"] == d}
                  for d in ("cellname", "pixel", "norm1000")}
    ms = list(acc)
    print(f"{'pair':<32}{'convention':<12}{'diff':>8}{'p':>10}")
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            A, B = ms[i], ms[j]
            for d in ("cellname", "pixel", "norm1000"):
                a, b = acc[A][d], acc[B][d]
                sh = sorted(set(a) & set(b))
                x = sum(1 for t in sh if a[t] and not b[t]); y = sum(1 for t in sh if b[t] and not a[t])
                dd = 100 * (np.mean([a[t] for t in sh]) - np.mean([b[t] for t in sh]))
                pv = binomtest(min(x, y), x + y, 0.5).pvalue if (x + y) else 1.0
                print(f"{A + ' vs ' + B:<32}{d:<12}{dd:>+8.1f}{pv:>10.1e}")


# --------------------------------------------- T7 H8 out-of-sample test
def table7():
    hdr("T7  H8: readership of two new conventions (stage 1) and their accuracy (stage 2)")
    s1 = [r for r in load("results/coco/h8_stage1.jsonl") if "correct" in r and "key_name" in r]
    s2 = [r for r in load("results/coco/h8_stage2.jsonl") if "correct" in r]
    for c in ("grid", "percent"):
        M = [r for r in s1 if r["desc"] == c + "_mismatch"]
        A = [r for r in s2 if r["desc"] == c]
        rho = np.mean([r["correct"] for r in M]) / (1 / 9) if M else float("nan")
        acc = 100 * np.mean([r["correct"] for r in A]) if A else float("nan")
        print(f"  {c:<10} readership {rho:.2f}x   accuracy {acc:.1f}%")


# --------------------------------------------- T8 4x4 granularity
def table8():
    hdr("T8  4x4 GRANULARITY (16 cells, chance 6.25%)")
    R = [r for r in load("results/coco/granularity_4x4.jsonl") if "correct" in r]
    if not R:
        return
    for d in ("cellname", "pixel"):
        print(f"  {d:<10} {100 * np.mean([r['correct'] for r in R if r['desc'] == d]):.1f}%")
    A = {r["trial_id"]: r["correct"] for r in R if r["desc"] == "cellname"}
    B = {r["trial_id"]: r["correct"] for r in R if r["desc"] == "pixel"}
    d, p, _, _, n = mcnemar(A, B)
    print(f"  gap {d:+.1f}  p={p:.1e}  n={n}")


# --------------------------------------------- T9 synthetic spatial + token counts
def table9():
    hdr("T9  SYNTHETIC SPATIAL: accuracy by direction/condition, and prompt length")
    rows = load("results/gpu_p3/coordinate_grounding.jsonl")
    acc = {}
    for dirn in ("locate", "identify"):
        for d in ("cellname", "norm1000", "pixel"):
            R = [r for r in rows if r["direction"] == dirn and r["desc"] == d]
            v = 100 * np.mean([r["correct_visible"] for r in R if "correct_visible" in r])
            c = 100 * np.mean([r["correct_cached"] for r in R if "correct_cached" in r])
            tk = int(np.median([r["prompt_tokens"] for r in R]))
            acc[(dirn, d)] = v
            print(f"  {dirn:<9}{d:<10} visible {v:5.1f}  cached {c:5.1f}  median prompt tokens {tk}")
    print(f"  locate normalized minus pixel, visible: "
          f"{acc[('locate','norm1000')] - acc[('locate','pixel')]:+.1f} points")


# --------------------------------------------- T10 frontier parse-failure split
def table10():
    hdr("T10 FRONTIER: gap counting unparseable as wrong vs parseable only")
    for lab, f in (("Gemini", "results/frontier/gemini-flash-latest.jsonl"),
                   ("GPT-4o", "results/frontier/gpt-4o.jsonl")):
        R = _locate(f)
        for mode, keep in (("all", lambda r: True), ("parseable", lambda r: not r["offvocab"])):
            A = {r["trial_id"]: r["correct"] for r in R if r["desc"] == "cellname" and keep(r)}
            B = {r["trial_id"]: r["correct"] for r in R if r["desc"] == "pixel" and keep(r)}
            d, p, _, _, n = mcnemar(A, B)
            print(f"  {lab:<8}{mode:<10} gap {d:+.1f}  p={p:.1e}  n={n}")
        ov = {d: 100 * np.mean([r["offvocab"] for r in R if r["desc"] == d]) for d in ("cellname", "pixel")}
        print(f"  {lab:<8}unparseable: cellname {ov['cellname']:.1f}%  pixel {ov['pixel']:.1f}%")


# --------------------------------------------- T11 calibration-angle ceiling
def table11():
    hdr("T11 CALIBRATION-ANGLE CEILING (the pre-held-out run)")
    rows = load("results/gpu_p3/interface_ladder.jsonl")
    for rg in ("word", "label"):
        v = [r["visible_correct"] for r in rows if r["rung"] == rg and "visible_correct" in r]
        print(f"  {rg:<7} visible on calibration angles {100 * np.mean(v):.1f}%")


def main():
    for t in (table1, table2, table3, table4, table5, table6, table7, table8, table9, table10, table11):
        try:
            t()
        except Exception as e:                                  # noqa: BLE001
            print(f"\n!! {t.__name__} failed: {type(e).__name__}: {e}")
    print(f"\n{'='*78}\nregenerated from raw jsonl; no number above is typed by hand")


if __name__ == "__main__":
    main()
