"""WHERE DOES CUE-ADDRESSED EVIDENCE STOP BEING USABLE: RETRIEVAL, OR RESPONSE?

The crossover established that swapping two remembered colours moves the answer
distribution in a direction that depends on WHICH LOCATION is asked about (sign
reverses q1 vs q2, null at the untouched q3). That locates the evidence. It does
not say why a model carrying that evidence still answers at chance.

Two accounts survive, and they have been conflated all project:

  RETRIEVAL INTERFERENCE  displaying alternatives degrades access to the cached
                          binding itself, before any answer is formatted
  RESPONSE CODING         access is fine; the loss happens converting retrieved
                          content into a discrete label the grader accepts

They are separated by an interface ladder in which each rung changes exactly one
thing, and the SAME cue-crossover cache manipulation runs at every rung:

  free     no alternatives shown; answer is a colour word
  freepad  no alternatives shown; answer is a colour word; prompt padded to the
           token length of the option listing
  word     alternatives shown as colour words; answer is a colour word
  label    alternatives shown as colour words; answer is the integer label
  numeric  the SAME alternatives, described as hue angles on a legended wheel;
           answer is the integer label

  free -> freepad   isolates PROMPT LENGTH (nothing else changes)
  freepad -> word   isolates PRESENCE OF ALTERNATIVES (length now matched)
  word -> label     isolates OUTPUT CODE (same alternatives displayed)
  label -> numeric  isolates the RESPONSE ALPHABET, and is the control the whole
                    headline rests on

THE CONFOUND `numeric` EXISTS TO KILL. The four word rungs score at ceiling, while
the same model on the same renderer under a numeric CIELAB hue scale scored 26.9%
with the array on screen. That comparison is not clean: the numeric task also asked
a FINER discrimination, options spaced 30 deg apart rather than six categorically
distinct colours. Alphabet and difficulty are confounded in it, so it cannot carry
the claim that the alphabet was the cause.

`numeric` unconfounds them. Identical stimuli, identical six options, identical
integer output code, and a legend built from the calibrated anchors so the numeric
interface is presented at its best rather than in the broken form that produced the
26.9%. The ONLY difference from `label` is whether each option is named by a word
or by a number. Whatever separates those two rungs is the alphabet and nothing
else.

freepad is the rung the external reviews did not ask for and the design needs.
Without it, any free->word drop is equally explained by "more tokens sit between
the cache and the answer position", which is a property of the harness rather
than of retrieval. With it, that explanation is measured instead of assumed.

PREDICTIONS
  response coding     interaction survives free, freepad and word at similar
                      magnitude, then falls at label
  retrieval interference
                      interaction falls at freepad -> word, i.e. merely showing
                      the alternatives costs evidence before any label exists

DV, comparable across every rung. Within a trial the diagnostic contrast is FIXED
as score(c1) - score(c2) where c1 and c2 are the two swapped colours: their first
answer tokens in the word rungs, the digits at their listed slots in the label
rung. DiD = margin(cache A) - margin(cache B). Predicted POSITIVE at q1, NEGATIVE
at q2, ZERO at q3.

COLOUR NAMING, AND WHY THIS RETIRES THE HUE-MAPPING BUG ENTIRELY. Earlier runs
described a CIELAB wheel to the model with conventional HSV anchors that the
renderer does not honour, a 35-50 deg mismatch that invalidated every accuracy
number computed under it. There is no numeric wheel here at all. Stimuli are
drawn from six anchors whose names were obtained by SHOWING the model each
swatch and asking it to name one, so the response alphabet is the model's own
naming of the renderer's output rather than a coordinate system laid over it
(scripts/calibrate_palette.py). Every queried target is an option by
construction: the option set is the four colours present plus the two absent
ones, i.e. the whole alphabet, so no target can be missing and none is resolved
by nearest-neighbour.

CONTROLS
  * visible arm at every rung: identical interface with the array on screen.
    Establishes that the model can name these colours through this interface at
    all, before any memory claim. A rung whose visible accuracy is at floor
    cannot support an inference about its cached arm.
  * the two swapped colours must be at least two alphabet slots and 40 deg of
    wheel apart, so the diagnostic pair is never a boundary call between
    neighbouring names.
  * caches A and B are rendered by the same code at the same positions and hold
    the IDENTICAL colour multiset. Only the binding differs.
  * q3 is untouched by the swap and its contrast is still score(c1)-score(c2),
    both of which are always options, so the null control is well posed.

Usage: VWM_DTYPE=4bit .venv/Scripts/python.exe scripts/interface_ladder.py
Env: VWM_IL_N (stimuli, default 100), VWM_IL_RUNGS (default free,freepad,word,label)
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

from vwm.stimuli.colorwheel import wheel_rgb
from vwm.stimuli.geometry import CANVAS_W, CANVAS_H, SQUARE_PX
from vwm.local.qwen3_vl import Qwen3VLClient
from vwm.local.kv_truncation_runner import FIXED_ACK

DTYPE = os.environ.get("VWM_DTYPE", "4bit")
N_STIM = int(os.environ.get("VWM_IL_N", "100"))
RUNGS = os.environ.get("VWM_IL_RUNGS",
                       "free,freepad,word,label,numeric").split(",")
SET_SIZE = 4
BG = (127, 127, 127)

OUT_PATH = Path(os.environ.get("VWM_IL_OUT", "results/gpu_p3/interface_ladder.jsonl"))
IMG_DIR = Path("results/gpu_p3/_il_imgs"); IMG_DIR.mkdir(parents=True, exist_ok=True)

# The response alphabet, CALIBRATED not declared. scripts/calibrate_palette.py
# renders one swatch per wheel angle and asks the model to name it in free report,
# three times per angle under different square sizes and placements; an angle
# counts only if all three agree. Each anchor below is the centre of a contiguous
# unanimously-named run, and main() re-verifies every one against the saved
# calibration before running.
#
# The first hand-written version of this table was wrong on two of eight anchors
# in the way this project has been wrong three times: wheel 23 is HSV hue 0.0 and
# a coordinate table calls it "red", but it renders rgb(238,146,146), and the model
# calls it pink -- correctly. This wheel has no red on it and no cyan either; the
# whole 210-290 arc is one very wide "blue". Two further names the model does use,
# teal and brown, are excluded. Teal's unanimous run is 190-195, too narrow for an
# anchor with two agreeing neighbours. Brown survived that test and was dropped for
# a different reason, found by running it: the model names an isolated brown swatch
# brown on 3/3 renders but names the SAME hue correctly on 0% of trials once it sits
# among three other squares. Simultaneous contrast, presumably, and a real effect --
# but a response alphabet has to be stable in the context the stimuli actually use,
# so brown is calibrated out rather than reported as a memory failure.
NAMED = [(5, "pink"), (50, "orange"), (95, "yellow"), (145, "green"),
         (245, "blue"), (310, "purple")]
CALIB = Path("results/gpu_p3/palette_calibration.json")
K = len(NAMED)

# HELD-OUT MODE (VWM_IL_HELDOUT=1). The vocabulary above was chosen by asking the
# model, at those exact angles, so evaluating at those same angles cannot separate
# "the model has this representation" from "we picked the colours it likes". The
# sampling rule below was declared in PROTOCOL_FREEZE.md and committed (b721c98)
# BEFORE this code existed, and the vocabulary is frozen: no category is added,
# removed or moved after any held-out number is seen.
#
# Rule: integer angles 10-20 deg from the anchor, off the 5 deg calibration grid,
# and at least 20 deg from every other category's anchor. Two per side per
# category, nearest to anchor +-12 and anchor +-18.
#
# Both rungs then ask which listed option the remembered square is CLOSEST to.
# Without "closest", held-out mode would quietly favour the word rung: a held-out
# pink is still exactly "pink", but it is not exactly "5", so the numeric rung
# alone would carry the burden of the offset. Same phrasing in both keeps the
# manipulation to the one thing it is supposed to be.
HELDOUT = os.environ.get("VWM_IL_HELDOUT") == "1"
CAL_STEP = 5
GUARD_OTHER = 20.0


def _circ(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def heldout_angles():
    """The declared held-out set, computed from the rule rather than listed."""
    out = {}
    for w, nm in NAMED:
        picks = []
        for side in (-1, 1):
            cands = [int((w + side * d) % 360) for d in range(10, 21)]
            cands = [t for t in cands if t % CAL_STEP != 0
                     and all(_circ(t, ow) >= GUARD_OTHER
                             for ow, onm in NAMED if onm != nm)]
            for target in (12, 18):
                if not cands:
                    break
                pick = min(cands, key=lambda t: abs(_circ(t, w) - target))
                if pick not in picks:
                    picks.append(pick)
        out[nm] = sorted(picks)
    return out
MIN_NAME_GAP = 2          # alphabet slots between the two swapped colours
MIN_WHEEL_SEP = 40.0      # degrees between them, so the DV is never a boundary call

SYS = ("You are a careful visual working memory experiment participant. "
       "Follow the instructions exactly and answer concisely.")
T1 = ("I will show you an array of colored squares. Study the exact color and "
      "location of each. Reply with only the word READY.")
# Padding for the freepad rung. Names no colour and offers no alternatives; its
# only job is to occupy the token budget the option listing would occupy. The
# realised token counts of both are logged per trial so the match is auditable
# rather than asserted.
PAD = ("Take your time and consider the array carefully before you answer. Recall "
       "which position was asked about, then recall the appearance of the square "
       "that occupied that position in the array you studied.")


def hsh(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, p)).encode()).digest()[:4], "big")


def check_anchors():
    """The anchor AND both of its sampled neighbours must name unanimously, as the
    declared name.

    An earlier version demanded 20 deg of clearance to the nearest differently
    named angle. That rejected orange, brown and yellow, which sit in a crowded
    45-105 arc, and it was the wrong test: clearance matters when interpolating,
    and nothing here interpolates -- the stimulus IS rendered at the anchor angle,
    where the model's naming was measured directly at 3/3. Requiring the two
    neighbours to agree keeps what that guard was really for, a name that does not
    sit on a boundary, and tests it against data instead of against a margin.
    Teal is the one name this rejects: its unanimous run is 190-195, so no anchor
    in it has two agreeing neighbours.

    Failing loudly here is the whole point. Both previous colour-mapping bugs
    shipped because the mapping was asserted in a comment instead of checked
    against data at run time."""
    cal = json.load(open(CALIB, encoding="utf-8"))
    tab = {int(a): v for a, v in cal["table"].items()}
    step = cal["step"]
    for w, nm in NAMED:
        for d in (-step, 0, step):
            a = (w + d) % 360
            if a not in tab:
                raise AssertionError(f"anchor {w}: angle {a} not in {CALIB}")
            v = tab[a]
            if v["agree"] < 1.0 or v["modal"] != nm:
                raise AssertionError(
                    f"anchor {w} declared {nm!r}: at {a} the model said "
                    f"{v['modal']!r} with agreement {v['agree']:.2f} {v['names']}")
    print(f"anchors verified against {CALIB} (self + both {step} deg neighbours "
          f"unanimous): {NAMED}", flush=True)


def cell_of(cx, cy):
    def band(v, hi):
        t = v / hi
        return 0 if t < 1 / 3 else (1 if t < 2 / 3 else 2)
    return (band(cy, CANVAS_H), band(cx, CANVAS_W))


def phrase(cell):
    nv = ("top", "middle", "bottom"); nh = ("left", "center", "right")
    v, h = cell
    return f"{nv[v]}-{nh[h]}" if (v, h) != (1, 1) else "center"


def render(positions, hues, path):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(img); h = SQUARE_PX // 2
    for (cx, cy), hue in zip(positions, hues):
        d.rectangle([cx - h, cy - h, cx + h - 1, cy + h - 1],
                    fill=wheel_rgb(hue), outline=(64, 64, 64), width=2)
    img.save(path)
    return img


def first_tokens(tok, words):
    """First token id of each word as it appears at the start of an assistant turn.

    Qwen's template puts assistant content immediately after the header newline, so
    the answer's first token carries no leading space. Requiring the ids to be
    DISTINCT is what makes a first-token margin a valid contrast between two colour
    words; if two names shared a first token the DV would be undefined."""
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in words]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"colour words share a first token: {list(zip(words, ids))}")
    return ids


def digit_tokens(tok, k):
    ids = []
    for j in range(1, k + 1):
        e = tok.encode(str(j), add_special_tokens=False)
        if len(e) != 1:
            raise AssertionError(f"label {j} is not a single token")
        ids.append(e[0])
    if len(set(ids)) != k:
        raise AssertionError("label tokens are not distinct")
    return ids


ANGLE_OF = {nm: w for w, nm in NAMED}
# Legend for the numeric rung, built from the calibrated anchors rather than from
# convention -- this is the numeric interface presented as favourably as the
# calibration allows, so a collapse here cannot be blamed on a bad legend.
NUM_SCALE = ("Colors are described by a hue angle from 0 to 359 on a color wheel "
             "where " + ", ".join(f"{w} is {nm}" for w, nm in NAMED) + ".")


def derange(words, seed):
    """A permutation with NO fixed point, deterministic per trial.

    Every angle and every word still appears exactly once; only the pairing between
    them is broken. A fixed point would leave one option correctly glossed and give
    the two scoring keys a shared answer on that slot, which would blunt exactly the
    separation this control exists to make."""
    n = len(words)
    for attempt in range(200):
        order = sorted(range(n), key=lambda j: hsh(seed, "der", attempt, j))
        if all(order[i] != i for i in range(n)):
            return [words[j] for j in order]
    return words[1:] + words[:1]        # cyclic shift: also has no fixed point


def build_prompt(rung, where, opt_names, memorized=True, gloss=None):
    """One rung of the ladder. Exactly one thing changes between adjacent rungs."""
    seen = "you memorized" if memorized else "shown to you"
    if rung == "free":
        return (f"In the array {seen}, what color was the square at the {where} of "
                f"the image? Answer with a single color word and nothing else.")
    if rung == "freepad":
        return (f"{PAD} In the array {seen}, what color was the square at the "
                f"{where} of the image? Answer with a single color word and "
                f"nothing else.")
    which = "Which of these is it closest to?" if HELDOUT else "Which one?"
    listing = ", ".join(opt_names)
    if rung == "word":
        return (f"In the array {seen}, the square at the {where} of the image was "
                f"one of these colors: {listing}. {which} Answer with a single "
                f"color word and nothing else.")
    if rung == "label":
        numbered = ", ".join(f"{j+1}={w}" for j, w in enumerate(opt_names))
        return (f"In the array {seen}, the square at the {where} of the image was "
                f"one of these colors: {numbered}. {which} Answer with a single "
                f"integer 1 to {len(opt_names)} and nothing else.")
    if rung == "numeric":
        numbered = ", ".join(f"{j+1}={ANGLE_OF[w]}" for j, w in enumerate(opt_names))
        return (f"{NUM_SCALE} In the array {seen}, the square at the {where} of the "
                f"image had one of these hues: {numbered}. {which} Answer with a "
                f"single integer 1 to {len(opt_names)} and nothing else.")
    # --- H5, the recovery gradient. Declared in PROTOCOL_FREEZE amendment 2 before
    # this code was written. Stimulus, alternatives and output format are identical
    # across all four rungs; only how much grounding the coordinate carries changes.
    if rung == "numeric_bare":
        numbered = ", ".join(f"{j+1}={ANGLE_OF[w]}" for j, w in enumerate(opt_names))
        return (f"In the array {seen}, the square at the {where} of the image had "
                f"one of these hues: {numbered}. {which} Answer with a single "
                f"integer 1 to {len(opt_names)} and nothing else.")
    if rung == "numeric_mismatch":
        # H6. Identical to numeric_gloss except the gloss is DERANGED, so the word
        # beside each angle is never that angle's word. Scoring the same response
        # against key_angle and key_word separates 'the coordinate was read' from
        # 'the coordinate was ignored and the word was read'.
        numbered = ", ".join(f"{j+1}={ANGLE_OF[w]} ({gloss[j]})"
                             for j, w in enumerate(opt_names))
        return (f"In the array {seen}, the square at the {where} of the image had "
                f"one of these hues: {numbered}. {which} Answer with a single "
                f"integer 1 to {len(opt_names)} and nothing else.")
    if rung == "numericleg_mismatch":
        # the same derangement, but delivered as a separate legend sentence -- the
        # `numeric` rung's format. Tests whether the weak two-hop path the separated
        # legend supports is following the number or the word.
        scale = ("Colors are described by a hue angle from 0 to 359 on a color "
                 "wheel where " + ", ".join(f"{ANGLE_OF[w]} is {gloss[j]}"
                                            for j, w in enumerate(opt_names)) + ".")
        numbered = ", ".join(f"{j+1}={ANGLE_OF[w]}" for j, w in enumerate(opt_names))
        return (f"{scale} In the array {seen}, the square at the {where} of the "
                f"image had one of these hues: {numbered}. {which} Answer with a "
                f"single integer 1 to {len(opt_names)} and nothing else.")
    if rung == "numeric_gloss":
        # the SAME legend content as `numeric`, but bound to each option instead of
        # stated once and separately. Separating this from `numeric` is what tells
        # "the grounding was absent" apart from "the grounding was not attached".
        numbered = ", ".join(f"{j+1}={ANGLE_OF[w]} ({w})"
                             for j, w in enumerate(opt_names))
        return (f"In the array {seen}, the square at the {where} of the image had "
                f"one of these hues: {numbered}. {which} Answer with a single "
                f"integer 1 to {len(opt_names)} and nothing else.")
    raise ValueError(rung)


def ask(model, proc, img, prompt, score_ids, cached):
    """cached: sample encoded once, then a TEXT-ONLY query continued against the KV.
    not cached: single pass with the array on screen beside the question.

    Fresh encode on every call -- generate() mutates the cache it is handed, which
    silently leaked the previous condition's tokens when a cache was reused."""
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
                               max_new_tokens=1, do_sample=False,
                               temperature=None, top_p=None,
                               return_dict_in_generate=True, output_scores=True)
    lg = g.scores[0][0].float()
    sel = lg[torch.tensor(score_ids, device=lg.device)]
    # candidate mass: how much of the distribution the scored alphabet captures at
    # all. A rung where the model answers off-alphabet is not measuring what the
    # margin claims to measure, so this is logged rather than inferred.
    mass = float(torch.softmax(lg, -1)[torch.tensor(score_ids, device=lg.device)].sum())
    top = int(torch.argmax(lg))
    return [float(x) for x in sel], int(torch.argmax(sel)), mass, top


def main():
    check_anchors()
    if "--names" in sys.argv:
        for w, nm in NAMED:
            print(f"wheel {w:3d} -> rgb {wheel_rgb(w)} -> {nm}")
        return

    manifest = json.load(open("data/stimuli/manifest_wheel.json", encoding="utf-8"))
    client = Qwen3VLClient(dtype=DTYPE); client.ensure_loaded()
    model, proc = client._model, client._processor
    tok = getattr(proc, "tokenizer", proc)

    WORDS = [nm for _, nm in NAMED]
    WORD_IDS = first_tokens(tok, WORDS)
    DIGIT_IDS = digit_tokens(tok, K)
    print(f"colour alphabet OK: {list(zip(WORDS, WORD_IDS))}", flush=True)

    trials = []
    for t in manifest["trials"]:
        if t["set_size"] != SET_SIZE:
            continue
        cells = [cell_of(*p) for p in t["positions"]]
        if len(set(cells)) != SET_SIZE:
            continue
        trials.append(t)
        if len(trials) >= N_STIM:
            break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = OUT_PATH.open("w", encoding="utf-8")
    n = skipped = 0
    t0 = time.time()

    for t in trials:
        tid = t["trial_id"]
        positions = [tuple(p) for p in t["positions"]]
        cells = [cell_of(*p) for p in positions]

        # four distinct named colours for this array, deterministic per trial
        order = sorted(range(K), key=lambda j: hsh(tid, "pick", j))
        present = order[:SET_SIZE]
        q1, q2, q3 = 0, 1, 2
        gap = min(abs(present[q1] - present[q2]), K - abs(present[q1] - present[q2]))
        wsep = min((NAMED[present[q1]][0] - NAMED[present[q2]][0]) % 360,
                   (NAMED[present[q2]][0] - NAMED[present[q1]][0]) % 360)
        if gap < MIN_NAME_GAP or wsep < MIN_WHEEL_SEP:
            skipped += 1; continue

        if HELDOUT:
            ho = heldout_angles()
            hues = [ho[NAMED[i][1]][hsh(tid, "ho", i) % len(ho[NAMED[i][1]])]
                    for i in present]
        else:
            hues = [NAMED[i][0] for i in present]
        names = [NAMED[i][1] for i in present]
        c1, c2 = names[q1], names[q2]

        # option set = the four present colours plus the four absent ones, i.e. the
        # whole alphabet, shuffled. Every queried target is present by construction.
        opt_order = sorted(range(K), key=lambda j: hsh(tid, "opt", j))
        opt_names = [WORDS[j] for j in opt_order]
        i1, i2 = opt_names.index(c1), opt_names.index(c2)
        slot = {nm: opt_names.index(nm) for nm in names}

        huesA = list(hues)
        huesB = list(hues); huesB[q1], huesB[q2] = huesB[q2], huesB[q1]
        namesB = list(names); namesB[q1], namesB[q2] = namesB[q2], namesB[q1]
        imgs = {"A": render(positions, huesA, IMG_DIR / f"{tid}_A.png"),
                "B": render(positions, huesB, IMG_DIR / f"{tid}_B.png")}

        for rung in RUNGS:
            LABELLED = ("label", "numeric", "numeric_bare", "numeric_gloss",
                        "numeric_mismatch", "numericleg_mismatch")
            MISMATCH = ("numeric_mismatch", "numericleg_mismatch")
            ids = DIGIT_IDS if rung in LABELLED else WORD_IDS
            # the FIXED diagnostic contrast, expressed in this rung's alphabet
            if rung in LABELLED:
                j1, j2 = i1, i2
            else:
                j1, j2 = WORDS.index(c1), WORDS.index(c2)
            gloss = derange(opt_names, tid) if rung in MISMATCH else None
            for qname, qi in (("q1", q1), ("q2", q2), ("q3", q3)):
                where = phrase(cells[qi])
                prompt = build_prompt(rung, where, opt_names, memorized=True,
                                      gloss=gloss)
                vis_prompt = build_prompt(rung, where, opt_names, memorized=False,
                                          gloss=gloss)
                rec = {"trial_id": tid, "rung": rung, "query": qname, "K": K,
                       "heldout": HELDOUT, "hues": hues,
                       "where": where, "c1": c1, "c2": c2,
                       "present": names, "options": opt_names,
                       "i1": i1, "i2": i2, "j1": j1, "j2": j2,
                       "prompt_tokens": len(tok.encode(prompt, add_special_tokens=False)),
                       "target_A": names[qi], "target_B": namesB[qi],
                       "target_slot_A": slot[names[qi]],
                       "score_source": "hf_generate_scores_greedy",
                       "dtype": DTYPE}
                # correct answer index in THIS rung's alphabet, per cache. Lets the
                # analysis report cached accuracy beside the DiD: the crossover says
                # the evidence is there, accuracy says whether it decides the answer,
                # and the project has repeatedly conflated the two.
                def tgt_of(nm):
                    return (slot[nm] if rung in LABELLED else WORDS.index(nm))
                rec["target_idx_A"] = tgt_of(names[qi])
                rec["target_idx_B"] = tgt_of(namesB[qi])
                if gloss is not None:
                    # key_angle is target_idx_* above: the slot whose ANGLE belongs
                    # to the remembered colour. key_word is the slot whose deranged
                    # GLOSS WORD names it. They never coincide, by construction.
                    rec["gloss"] = gloss
                    rec["key_word_A"] = gloss.index(names[qi])
                    rec["key_word_B"] = gloss.index(namesB[qi])
                    assert rec["key_word_A"] != rec["target_idx_A"],                         "derangement left a fixed point; the two keys collided"
                good = True
                for ck in ("A", "B"):
                    try:
                        sel, am, mass, top = ask(model, proc, imgs[ck], prompt,
                                                 ids, cached=True)
                    except Exception as e:                    # noqa: BLE001
                        rec[f"error_{ck}"] = f"{type(e).__name__}:{e}"
                        good = False; continue
                    rec[f"margin_{ck}"] = sel[j1] - sel[j2]
                    rec[f"argmax_{ck}"] = am
                    rec[f"mass_{ck}"] = mass
                    rec[f"toptok_{ck}"] = top
                    # every candidate score, not just the diagnostic pair. Showing
                    # the option list could inflate the whole scored range rather
                    # than add evidence, and a margin alone cannot tell those apart;
                    # the spread of this vector is what normalises the comparison
                    # across rungs.
                    rec[f"scores_{ck}"] = [round(x, 4) for x in sel]
                    n += 1
                try:
                    sel, am, mass, top = ask(model, proc, imgs["A"], vis_prompt,
                                             ids, cached=False)
                    tgt = rec["target_idx_A"]
                    rec["visible_argmax"] = am
                    rec["visible_correct"] = bool(am == tgt)
                    rec["visible_mass"] = mass
                    rec["visible_margin"] = sel[j1] - sel[j2]
                    rec["visible_scores"] = [round(x, 4) for x in sel]
                    rec["target_scored_idx"] = tgt
                    n += 1
                except Exception as e:                        # noqa: BLE001
                    rec["error_visible"] = f"{type(e).__name__}:{e}"
                if good:
                    rec["did"] = rec["margin_A"] - rec["margin_B"]
                f.write(json.dumps(rec) + chr(10)); f.flush()
        if n and n % 216 == 0:
            el = time.time() - t0
            print(f"  {n} forwards  {el/n:.2f}s each  elapsed {el/60:.1f}m", flush=True)

    f.close()
    print(f"ILDONE wrote {n} forwards, skipped {skipped} trials -> {OUT_PATH}")


if __name__ == "__main__":
    main()
