# Frozen analysis plan

**Frozen 2026-09-19 21:0x EDT, before any held-out stimulus was generated.**
Everything below is fixed. Changes after this commit must be added as dated
amendments at the bottom, never by editing the text above them.

Written because the calibration that rescued this project also created its main
vulnerability: the response vocabulary was chosen by asking the model, and the
evaluation then used the very angles the calibration selected. That is fine as
calibration and fatal as evidence, unless the split is declared before the
held-out data exists. Hence the order: this file, then the generator.

---

## 1. The claim, and its scope

The measured capability of a vision-language model depends on the vocabulary the
evaluation uses to elicit an answer. When options are described in a coordinate
system the model has not grounded, the score reports the grounding failure and
not the underlying representation.

Scope discipline. "A vision-language model" until a second family replicates;
"vision-language models" only after. No claim of perception-memory equivalence.
No claim of human-comparable precision.

### Terminology
"Response vocabulary", not "response alphabet". The manipulation changes semantic
grounding, lexical status, tokenization and symbol priors at once, and "alphabet"
implies a narrower token-encoding story than what is actually varied.

---

## 2. Primary hypotheses

- **H1 (held-out, within model).** On colour angles never used to select the
  vocabulary, word-described options outperform numerically described options,
  with stimuli, candidate set, candidate order and output format held fixed.
- **H2 (cross family).** The same qualitative penalty appears in one non-Qwen VLM,
  each model calibrated independently under an identical procedure.
- **H3 (generalisation beyond colour).** The penalty appears for SPATIAL readout
  too: naming a location beats giving pixel coordinates for the same target in the
  same image. If it holds, the finding is about coordinate grounding rather than
  about colour.
- **H4 (transfer to a real benchmark).** On an existing multiple-choice VLM
  benchmark, re-describing the same options changes the score materially.

H1 is the gate. H2, H3 and H4 each widen the claim; none of them rescues it if H1
fails.

---

## 3. Primary endpoints

1. Held-out visible accuracy, word minus numeric, paired per stimulus.
2. Held-out cached accuracy, word minus numeric, paired per stimulus.
3. Cue-crossover interaction: `mean(DiD q1 - DiD q2)`, spread-normalised, the
   trial as the unit of inference.

Spread normalisation is mandatory for any cross-condition comparison of margins.
Raw margins are reported only within a condition. (Established: displaying options
raised the raw interaction 24% while candidate-score SD rose 7.5 to 9.5; the
apparent gain was entirely scale.)

## 4. Secondary endpoints

Candidate probability mass; off-vocabulary generation rate; calibration yield per
model; per-category accuracy; load-dependent transition width; cache-eviction
results.

---

## 5. Held-out sampling rule — declared before generation

For each retained category with anchor `A`:

- **Calibration angles**: the 5-degree grid used by `calibrate_palette.py`,
  including `A` and its two neighbours. Used to select the vocabulary. Never used
  for evaluation of H1.
- **Guard**: `A ± 5` degrees, part of calibration, excluded from evaluation.
- **Held-out evaluation angles**: integer degrees `theta` with
  `10 <= |theta - A| <= 20`, `theta` not on the 5-degree calibration grid, and
  `theta` at least 20 degrees from any other category's anchor. Two per category
  per side where the rule admits them; if a side admits none, that side is empty
  and the yield is reported rather than the interval widened.
- **Boundary angles**: reported descriptively only, never in a primary endpoint.

The vocabulary is FROZEN at
`pink 5, orange 50, yellow 95, green 145, blue 245, purple 310`
and is not revised after any held-out result is seen.

Diagnostic-pair constraints carry over unchanged: at least 2 alphabet slots and
40 degrees of wheel between the two swapped colours.

---

## 6. Exclusions — evidence that may not be cited

- Old numeric-hue-wheel accuracy, as evidence about capacity or precision. The
  prompt's anchors did not describe the renderer.
- `q3` visible accuracy from any run before `3fd177d`; the candidate set did not
  contain `h3`. The q3 *DiD* from those runs remains valid, because its contrast
  is `score(h1) - score(h2)` and both were always present.
- The set-size-1 precision estimate as a number. The sweep's step is 4.5-10
  degrees and the fit returned 0.6; it is reported as "finer than the sampling
  resolution" and appears in no table, abstract or caption as a value.
- Any comparison of a fitted transition width to human continuous-report SD. Human
  VWM may be cited as motivation, never as a common metric.
- "Raw logits" as a description of `generate().scores` until
  `score_provenance.py` has been run and reports zero difference.
- Any suggestion that 50% cache retention is safe. The preregistered binding-first
  dissociation was REJECTED; both binding and feature-set evidence collapsed
  together. The surviving claims are severity at routine budgets and that
  sink-plus-recent underperformed random at matched budget.

---

## 7. Decision rule, evaluated Monday night

| Outcome | Action |
|---|---|
| H1 passes, non-Qwen replicates | Submit as a cross-model measurement paper |
| H1 passes, non-Qwen runs but differs | Submit only if the difference is interpretable, scoped explicitly |
| H1 passes, non-Qwen cannot be run cleanly | Submit with title and abstract naming a single model, never "VLMs" |
| H1 fails | Do not submit this headline. Narrow to a preprint or workshop |

A failed second model narrows the title. Only a failed held-out test kills the
paper.

---

## 8. Fixed configuration

Qwen3-VL-4B-Instruct, 4-bit NF4, greedy decoding (`do_sample=False`,
temperature and top_p unset), one generated token scored, first-token contrast
over a candidate set whose members are verified to be distinct single tokens.
Stimuli: CIELAB wheel, L* = 70, r = 38. Calibration: 5-degree grid, 3 renders per
angle differing in square size and placement, unanimity required.

Logged per cell: full candidate score vector, prompt string, option list and
order, target index, candidate mass, top token overall, seed, model and dtype.

---

## Amendments

_(none yet)_

### Amendment 1 — 2026-09-19, after the held-out run, before any re-run

**Defect in the held-out sampling rule of section 5.** The guard was "at least 20
degrees from any other category's anchor". It should have been "at least 20 degrees
from the nearest DIFFERENTLY NAMED angle in the calibration table". The calibration
table contains names that were deliberately dropped from the vocabulary — brown and
teal — and boundary angles that name inconsistently, and nothing in the rule as
written guarded against a held-out angle landing in one.

The stricter guard was written first and relaxed when it rejected orange, brown and
yellow as anchors. That relaxation was justified for anchors, whose naming was
measured directly at 3/3. It was carried into the held-out rule, where it is not
justified, because held-out angles were never measured.

Seven of 24 held-out angles are affected: orange 32 (calibration: pink at 30),
orange 68 (brown at 70), yellow 77 (brown at 75), yellow 113 (green at 115),
purple 292 (blue at 290), purple 328 (pink at 330), and orange 38 sits next to a
0.67-agreement angle.

**The primary analysis stands as frozen and is reported as frozen.** The defect is
conservative with respect to H1: a held-out angle the model would name differently
penalises the word rung at least as much as the numeric rung. Visible label-minus-
numeric is +30.1 points as frozen and +33.2 points with yellow removed, so the
frozen figure understates the effect.

**Post-hoc sensitivity** (labelled as such wherever it appears): excluding yellow
raises word visible accuracy from 84.4% to 95.4% and label from 83.8% to 93.1%,
while numeric rises from 53.8% to 60.0%.

**Corrected rule for any future run**, including all cross-model work: a held-out
angle must be at least 20 degrees from every calibration-grid angle whose modal
name differs from the category's, and at least 20 degrees from every angle with
agreement below 1.0. Anchors keep the self-plus-neighbours criterion unchanged.

**Methodological point this earns, and it belongs in the paper.** Held-out
generalisation of a model-calibrated vocabulary is bounded by the categories the
calibration DISCARDED, not only by the ones it kept. Dropping a name from the
response set does not remove the region of stimulus space the model assigns to it.

### Amendment 2 — 2026-09-19 ~23:00 EDT, BEFORE the recovery experiment is written

Two non-Qwen models have now been attempted and neither clears the capability gate.
SmolVLM-2B: best visible accuracy 30.4% against 16.7% chance. LLaVA-1.5-7B: four
anchors survive calibration, best visible accuracy 45.8% against 25% chance, and
generate-and-parse puts it at 22.9%, below chance. Both contrasts are null in both
models. **H2 is UNTESTED, not refuted** — a convention gap cannot be measured in a
model with no competence to lose, which is a scoping fact about the method and is
reported as one.

#### H5 — the recovery prediction, declared before implementation

A convention gap caused by missing grounding should be REPAIRABLE by supplying the
grounding, without changing the stimulus, the alternatives, or the output format.
If instead the gap is caused by the numeral itself — tokenization, digit priors,
symbol arbitrariness — then supplying grounding changes nothing.

Colour, four rungs forming a grounding gradient. Identical held-out stimuli,
identical six alternatives, identical integer output, frozen vocabulary:

| rung | option list | grounding |
|---|---|---|
| `numeric_bare` | `1=5, 2=50, ...` | none at all |
| `numeric` | legend sentence, then `1=5, 2=50, ...` | present but separated |
| `numeric_gloss` | `1=5 (pink), 2=50 (orange), ...` | present and inline |
| `label` | `1=pink, 2=orange, ...` | the coordinate removed |

**Predicted ordering: `numeric_bare` < `numeric` < `numeric_gloss` <= `label`.**

Spatial, the same gradient: `pixel_bare` < `pixel` < `pixel_gloss` <= `cellname`,
where `pixel_gloss` is `1=(66, 50) [top-left], ...`.

#### What each outcome means, committed in advance

- Ordering holds and `numeric_gloss` reaches `label`: the deficit is a convention
  TRANSLATION bottleneck, fully repairable in the prompt. The diagnosis predicts
  its own fix and the method has an intervention.
- `numeric_gloss` recovers part of the gap: partially repairable; the remainder is
  attributable to the numeral rather than to grounding.
- `numeric_gloss` does not beat `numeric`: **H5 is refuted.** The gap is not a
  grounding failure, and the paper must drop the grounding interpretation and
  report the gap as unexplained.
- `numeric` beats `numeric_gloss`: unpredicted, reported as such, no post-hoc story.

The separated-versus-inline comparison (`numeric` vs `numeric_gloss`) is the part
no reviewer has asked for and the part that distinguishes "the grounding was
absent" from "the grounding was present but not bound to the option".

#### Scope discipline for the write-up, fixed now

Claims that may NOT be made, having been considered and rejected as unsupported:
- that every benchmark using degrees, percentages, timestamps, Likert scales,
  pixels or boxes measures mapping rather than capability;
- that no model-agnostic response vocabulary can exist. What two calibrations
  support is only: Qwen3-VL-4B and SmolVLM-2B partitioned the same rendered wheel
  into different stable verbal categories, so a fixed textual answer vocabulary
  may not have equivalent perceptual grounding across models;
- that the 0-999 result is explained by Qwen's training data. "A convention the
  model can use" is supportable; "a convention the model was trained on" is not,
  absent documentation. The alternative accounts — tokenization, numeric range,
  formatting familiarity — are not excluded;
- that the model "has the representation but lacks the mapping". The evidence is a
  behavioural dissociation; the representational reading stays a hypothesis.

### Amendment 3 — 2026-09-19 ~23:55 EDT. A gate defect, flagged because correcting it helps me

The capability gate in amendment 2 was "best word-rung visible accuracy at least 20
points above chance". It is underspecified in a way that matters, and I am
recording the correction rather than applying it quietly, because applying it
quietly would move a model from disconfirming to uninformative.

**The defect.** "Best" ranges over readouts. LLaVA-1.5-7B clears it on the
first-token readout (45.8% against 25% chance, margin 20.8) but the
label-versus-numeric contrast is computed on generate-and-parse, where LLaVA is at
22.9% — below chance. The gate passed on one instrument and the contrast was
measured on another that has no signal in its own baseline condition. A difference
between two conditions cannot be measured with an instrument that reads noise in
the reference condition.

**The correction.** The gate must be evaluated on THE SAME READOUT the contrast is
computed on. Under the corrected gate LLaVA fails and is uninformative rather than
disconfirming.

**Both readings are reported, always, side by side.** Suppressing the lenient one
would be the dishonest version of this.

| model | gate (as declared) | gate (readout-matched) | label→numeric | reading |
|---|---|---|---|---|
| Qwen3-VL-4B | pass | pass | −30.1, p=8.7e-33 | supports H1 |
| Qwen2.5-VL-7B | pass | pass | −23.9, p=9.1e-4 | supports, within family |
| LLaVA-1.5-7B | pass (margin 0.8) | **fail** | −2.1, p=1.0 | null under the declared gate; uninformative under the corrected one |
| SmolVLM-2B | fail | fail | −11.3, p=.077 | uninformative |

Even under the lenient reading the count is two supporting against one null, and
the null is a 2023 model whose two readouts disagree by 23 points. That is worth
stating as a limitation and not worth explaining away.

**Consequence for the write-up.** Any future model is gated on the readout used for
its contrast, decided before its contrast is inspected.

### Amendment 4 — 2026-09-20 ~01:00 EDT. H2 result, and a fourth readout bug

**A fourth bug, same shape as the other three.** `answer_by_generation` returned an
index into `opt`, the per-trial SHUFFLED option list, while the word rung's target
was an index into `WORDS`, the canonical alphabet. Two index spaces differing by a
shuffle. Every cross-model word-rung generate-and-parse score sat at chance —
15.0, 19.7, 14.1, 22.9 against 16.7 — and four capable models read as incapable.
The label rungs were never affected; both sides used slot indices, which is why
only one rung looked broken and why it looked like a property of the models.

After the fix, generate-and-parse and first-token agree on 100% of word-rung cells
in every model, which is what a resolved candidate mass near 1.0 predicts and what
the bug was masking. That agreement is itself the cross-readout check the reviews
asked for, and it now passes.

**H2: supported in three models of five, with one genuine null.**

| model | K | chance | word | label | numeric | gate | label→numeric |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B (primary) | 6 | 16.7% | 84.4% | 83.8% | 53.8% | pass | −30.1, p=8.7e-33 |
| Qwen2.5-VL-7B | 6 | 16.7% | 64.8% | 73.2% | 49.3% | pass | −23.9, p=9.1e-4 |
| InternVL3-2B | 6 | 16.7% | 54.9% | 47.9% | 28.2% | pass | −19.7, p=1.6e-2 |
| LLaVA-1.5-7B | 4 | 25.0% | 45.8% | 33.3% | 31.2% | pass | **−2.1, p=1.0** |
| SmolVLM-2B | 6 | 16.7% | 29.6% | 25.4% | 14.1% | **fail** | uninformative |

**Architecture, verified from the configs rather than assumed.** InternVL3-2B is
`internvl_vision` with a `qwen2` text backbone: different vision encoder, different
lab, different training recipe, SHARED language-model lineage with the two Qwen
models. It is a partial cross-family replication and must be described as one. The
two fully independent stacks — SmolVLM (`idefics3` + `llama`) and LLaVA
(`clip_vision_model` + `llama`) — are respectively below the gate and null.

**LLaVA-1.5-7B is a real disconfirming case and is reported as one.** It clears the
readout-matched gate (45.8% against 25% chance) and shows no penalty. It is the
oldest model tested, has the lowest input resolution, and only four anchors survived
its calibration. None of that is offered as an explanation, only as description; the
honest sentence is that the effect is not universal among models that can do the
task.

**Consequence for the claim.** "Three of four models that clear the capability gate
show a substantial word-to-numeric penalty; one does not." Not "VLMs". The title
names the finding, not a universal.

### Amendment 5 — 2026-09-20 ~01:15 EDT, BEFORE the mismatch control is written

**H5 has an alternative explanation I have not excluded, and it is the obvious one.**
`numeric_gloss` shows `1=5 (pink)`. The model may be reading "pink" and ignoring
"5" entirely, in which case the 91% "recovery" is not recovery: it is the word rung
wearing a numeric costume. The numbers are consistent with exactly that —
`numeric_gloss` 78.2% sits 5.6 points under `label` 83.8%, which is what pure
word-reading minus a small distraction cost would look like.

If I ship H5 without excluding this, the intervention claim is worthless.

#### H6 — the mispaired-gloss control, declared before implementation

Derange the gloss so no option carries its true pairing: `1=5 (green),
2=50 (purple), 3=95 (pink), ...`. Every angle and every word still appears exactly
once; only the pairing is broken. Two scoring keys then exist for the same response:

- **key_angle** — the slot whose ANGLE is the remembered colour's angle
- **key_word** — the slot whose GLOSS WORD is the remembered colour's name

Run it on both glossed rungs: `numeric_mismatch` (gloss inline, the H5 rung) and
`numericleg_mismatch` (legend stated separately, the `numeric` rung).

#### The three hypotheses and what each predicts

| account | key_word accuracy | key_angle accuracy |
|---|---|---|
| **Gloss substitution** — the coordinate is never read | near the clean glossed rung (~78%) | near chance |
| **Genuine integration** — coordinate and gloss both contribute | both depressed, conflict | both depressed |
| **Coordinate grounding installed** — the gloss taught the mapping | near chance | near the clean glossed rung |

#### My prediction, on the record, and what it costs me

I expect **gloss substitution**: key_word high, key_angle at chance. The `numeric`
rung's 53.8% shows the model can extract *something* from a separated legend, so a
two-hop number→legend→word path exists and is weak; but with the word attached to
the option there is no reason to traverse it.

**If that is what happens, H5 as currently written is wrong and the paper's
recommendation inverts.** It would no longer be "ground your coordinates with a
gloss and the readout is repaired". It would be:

> A coordinate convention cannot be repaired by explanation. It can only be
> bypassed. What looked like grounding a coordinate was supplying an alternative
> readout beside it, and the coordinate went unread. A benchmark that requires a
> coordinate answer therefore measures the convention, and no amount of in-prompt
> description changes that — only training the convention, or abandoning it, does.

That is a stronger and more surprising claim than the one it replaces, and it is
the one I expect the data to support. Recording the expectation now so that
confirming it cannot be mistaken for a post-hoc reframing, and so that failing to
confirm it is visibly a failure.

**If instead key_angle stays high**, the coordinate IS being read and H5's original
grounding interpretation survives intact and strengthened.

### Amendment 6 — 2026-09-20 ~01:50 EDT, BEFORE the real-image experiment is written

Everything so far is coloured squares on a grey field. The fair objection is "you
showed a model cannot use CIELAB hue angles, and nobody uses those." H3 answered
half of it with space; this answers the rest with real photographs and the
convention the localisation literature actually scores models in.

#### H4 — the convention gap on real images

COCO val2017 (`detection-datasets/coco`, 2476 images with boxes). One object per
trial, its location asked for as a choice among the nine cells of a 3x3 grid over
the real image. Only the DESCRIPTION of those nine cells varies:

| rung | options read |
|---|---|
| `cellname` | `top-left`, `center`, ... |
| `norm1000` | `(167, 500)` — the 0-999 convention |
| `pixel` | `(107, 213)` — absolute pixels, explained in a preamble |
| `pixel_gloss` | `(107, 213) [top-left]` |
| `pixel_mismatch` | `(107, 213) [bottom-right]` — deranged, no fixed point |

Both directions, as in H3: `locate` (coordinate is the answer) and `identify`
(coordinate is the cue, options are COCO category names).

#### Trial admission, fixed now so it cannot be tuned to the result

- exactly ONE instance of the queried category in the image, so the answer is unique
- the box's area is under 25% of the image, so "which cell" is well posed
- the box centre lies at least 8% of the image's width and height inside its cell's
  border, so a centre near a boundary cannot make a correct answer look wrong
- categories used as given by COCO, no renaming, no filtering by how nameable they are

#### Predictions, committed

1. `cellname` > `norm1000` > `pixel`, reproducing the synthetic ordering on real
   photographs.
2. `pixel_mismatch` follows the deranged NAME, not the pixels, with key_coord at or
   below chance — the H6 substitution signature, on real images.
3. `pixel_gloss` ≈ `cellname`, and for the same reason: the coordinate goes unread.

If 1 fails, the convention gap is an artefact of synthetic stimuli and the paper
must say so and shrink. If 2 fails — if mispairing the name leaves accuracy on the
coordinate — then coordinates ARE read on real images and the H6 conclusion is
domain-limited rather than general.

#### What this cannot show

COCO categories are frequent English nouns and the 3x3 grid is coarse. This tests
whether the convention gap survives on natural images with real objects. It does
not measure fine-grained localisation, and no claim about box-regression benchmarks
follows from a nine-cell choice.

### Amendment 7 — 2026-09-20 ~02:30 EDT, BEFORE the norm1000 mismatch is written

H4 confirmed all three predictions on real photographs, and H6 came apart by domain
in a way that suggests a single quantity underneath both.

**Convention readership.** Mispair a coordinate with a name and ask which the model
follows. The fraction following the COORDINATE, measured against chance, is how much
that convention is actually being read. Values so far:

| convention | key_coord | chance | readership |
|---|---|---|---|
| CIELAB hue degrees (synthetic) | 5.4% | 16.7% | 0.32 — below chance |
| absolute pixels (synthetic space) | 21.7% | 11.1% | 1.95 |
| absolute pixels (real COCO) | 16.0% | 11.1% | 1.44 |

#### H7 — readership tracks trained familiarity

`norm1000` is the 0-999 convention Qwen-VL's own grounding data is written in, and
it already outperforms absolute pixels in accuracy on both synthetic space and real
images. If readership is the mechanism behind that accuracy gradient rather than a
separate fact, then:

**Predicted: readership(norm1000) > readership(pixel) > readership(hue), on the
same images, same nine options, same deranged-name design.**

Run `norm1000_gloss` and `norm1000_mismatch` on COCO alongside the pixel rungs
already collected.

- If norm1000 readership exceeds pixel readership, one quantity explains the whole
  accuracy ordering across two feature domains and both stimulus types, and it is
  measurable for any model and any candidate convention with a single cheap test.
- If norm1000 readership is at or below pixel readership, readership does NOT
  explain the accuracy gradient, H7 fails, and the accuracy ordering and the
  mismatch results must be reported as two separate findings rather than one.

That failure mode is the likely one if the accuracy advantage of norm1000 comes
from something other than the coordinate being read — smaller integers, fewer
digits, or range familiarity independent of grounding. Recording it now so the
distinction is not blurred afterwards.

### Amendment 8 — 2026-09-20 ~15:30 EDT, BEFORE the out-of-sample conventions are run

Readership currently DESCRIBES three conventions already measured. A diagnostic has
to PREDICT one it has not seen, or it is a curve fitted to its own points.

#### H8 — out-of-sample prediction, run in two locked stages

Two conventions never tested, both naming the same nine COCO cells:

- `grid` — `A1` ... `C3`, a letter-number grid reference. A very familiar FORMAT
  that still requires the same image-to-coordinate mapping.
- `percent` — `(33, 67)`, position as a percentage of width and height. Familiar
  units, but a described convention like absolute pixels.

**Stage 1.** Run the mispaired-gloss version of each and compute readership. Nothing
else is run and no accuracy is inspected.

**Stage 2.** With those two readership values in hand and accuracy still unmeasured,
write the predicted accuracy ORDER into this file as a dated entry. Then run the
plain version and check.

The prediction rule, fixed now so stage 2 cannot invent one that fits: accuracy is
predicted to be monotone in readership, and each new convention is placed relative
to the two COCO anchors already measured --- `pixel` at readership $1.44\times$ and
accuracy $20.0\%$, `norm1000` at $4.05\times$ and $59.5\%$. A convention whose
readership lands between them is predicted to land between them in accuracy; below
$1.44\times$ predicts under $20.0\%$; above $4.05\times$ predicts over $59.5\%$.

**What failure looks like.** If a convention with higher readership scores lower in
accuracy, or if both new conventions land outside the predicted intervals, then
readership does not predict out of sample. It would remain a real measurement of
which token stream the model follows, but the paper could not call it a diagnostic
and the claim would shrink to a description of three conventions.

Two conventions is a weak test in absolute terms and is reported as one. It is a
genuine out-of-sample test nonetheless, because the prediction is written before
the accuracy run exists.

### Amendment 8, stage 2 — 2026-09-20 ~17:15 EDT. THE PREDICTION, written before the accuracy run

Stage 1 is complete. Readership measured on 200 COCO images per convention, with
no accuracy arm run and none inspected:

| convention | key_coord | key_name | neither | readership |
|---|---|---|---|---|
| `grid` (A1..C3) | 14.0% | 40.0% | 46.0% | **1.26x** |
| `percent` ((33,67)) | 14.5% | 32.0% | 53.5% | **1.30x** |

Applying the placement rule fixed in amendment 8 — monotone in readership, anchored
on `pixel` at 1.44x/20.0% and `norm1000` at 4.05x/59.5% — both fall below the
`pixel` anchor, so:

**PREDICTED: `grid` scores under 20.0%. `percent` scores under 20.0%.**
**PREDICTED ORDER: `percent` >= `grid`, since 1.30x > 1.26x.**

This prediction is deliberately counter-intuitive and that is the point of making
it. `A1`/`B2`/`C3` is a format almost every reader knows from chess and
battleship, and a reasonable person would expect it to work well. Readership says
it will not, because familiarity with the FORMAT is not the same as having a
grounded mapping from the image to that format — which is the paper's whole claim,
now stated as a falsifiable number before the test.

**Failure conditions, restated so stage 2 cannot soften them.** If either
convention scores at or above 20.0%, the placement rule is wrong and readership
does not predict out of sample. If `grid` beats `percent` materially, the monotone
ordering is wrong. Either outcome removes the word "diagnostic" from the paper and
reduces readership to a description of which token stream a model follows.

### Amendment 8, outcome — 2026-09-20 ~17:40 EDT. H8 FAILS. Reported as declared.

| convention | readership | predicted | observed | verdict |
|---|---|---|---|---|
| `grid` | 1.26x | under 20.0% | **31.0%** | MISS |
| `percent` | 1.30x | under 20.0% | **23.0%** | MISS |
| order | `percent` >= `grid` | — | grid 31.0% > percent 23.0% | MISS |

Every part of the prediction failed. Per the failure condition written before the
run: **readership does not predict accuracy out of sample, and the paper does not
call it a diagnostic.**

**Why, stated as a flaw in the quantity rather than as bad luck.** Readership
measures which stream wins when a coordinate and a name are both present and
disagree. It therefore mixes two things the placement rule treated as one: how
usable the coordinate system is, and how strongly a competing name pulls. `grid`
has key_name 40.0%, the strongest name pull of any convention tested, which drags
its readership to 1.26x even though the bare grid convention works at 31.0%. A
convention can be readable and still lose an argument with an English phrase.

**What survives, and is unaffected by this failure.**

1. The convention gap itself. cellname 68.5% against pixel 20.0% on real
   photographs, chance 11.1%, replicated in a second model at 46.0% against 13.5%.
   Nothing here depends on readership.
2. The producing-versus-consuming asymmetry, -48.5 points against -3.5.
3. The mismatch design as a causal test of which stream is followed when both are
   present. That is what it measures and it measures it correctly.
4. The two extremes, which remain striking and are not in dispute: CIELAB hue
   angles are followed BELOW chance (0.30x, p = 7.7e-21) while a normalised
   convention is followed well above it (4.05x, p = 1.2e-33), and the practical
   consequence holds -- glossing coordinate options silently converts the task into
   a word task.

**What dies.** "Readership predicts how much a benchmark understates a model." Not
supported. The relationship holds at the extremes and fails across the middle, and
five points ordered by readership do not order by accuracy.

**Not permitted after this entry**: searching for a different statistic that fits
these five points and presenting it as though it had been predicted. Any revised
quantity is post-hoc, is labelled post-hoc, and requires its own out-of-sample test
before it may be called predictive.

### Amendment 9 — 2026-09-20 ~17:50 EDT, BEFORE the remaining models are run on COCO

H8's failure removed the predictive claim. What remains is a phenomenon without a
consequence, and the consequence worth testing is already visible in two models:

| convention | Qwen3-VL-4B | InternVL3-2B | measured gap |
|---|---|---|---|
| cellname | 68.5% | 46.0% | 22.5 pts |
| norm1000 | 59.5% | 19.0% | 40.5 pts |
| pixel | 20.0% | 13.5% | 6.5 pts |

The apparent difference between two models swings 6.5 to 40.5 points on identical
images and identical choices, decided by the answer convention alone.

#### H9 — convention choice changes model COMPARISONS, not only scores

Run the COCO convention swap on every remaining model: Qwen2.5-VL-7B,
LLaVA-1.5-7B, SmolVLM-2B. Same 200 images, same admission criteria, each model's
digit readout resolved before scoring, capability gate applied on the readout the
comparison uses.

**Predeclared, so that finding one is not cherry-picking:** we are looking for a
RANK INVERSION — a pair of models whose order reverses between two conventions on
the same items. We are also recording, for every model pair, the range of the
measured gap across conventions.

**Reported regardless of outcome.**
- An inversion exists: convention choice can reverse a published comparison. This
  is the strongest form of the claim and the one being searched for.
- No inversion, but gaps vary widely: convention choice changes the magnitude of
  every comparison without reversing it. Weaker, still a leaderboard result, and
  the 6x swing above already establishes it for one pair.
- Gaps are stable across conventions: H9 fails, the effect is a uniform offset,
  and the paper claims only that absolute scores move.

A model failing the capability gate is reported as failing and is excluded from
comparisons, not dropped from the table.

**Not permitted**: reporting only the model pairs that happen to invert. Every pair
tested appears, inverting or not.

### Amendment 10 — 2026-09-21. Documentation found for the training-convention attribution

Amendment 2 barred attributing the normalized-coordinate result to training
"absent documentation". Documentation now exists and is cited: the Qwen2.5-VL
technical report (arXiv:2502.13923) states that boxes and points are represented
"based on the actual dimensions of the input images", and the Qwen3-VL technical
report (arXiv:2511.21631) describes a normalized [0, 1000] grounding system. The
paper attributes the direction of the Qwen2.5 / Qwen3 rank inversion to these
documented conventions and nothing more; InternVL3, Gemini and GPT-4o receive no
such attribution because no equivalent statement was located for them.

### Amendment 11 — 2026-09-21. The below-chance number is explained (post hoc, prompted by an external review)

Amendment 10 left Qwen2.5-VL-7B's 5.0% under the normalized convention as
unexplained after ten grid-transform hypotheses. An automated review
(paperreview.ai) asked for a diagnostic plan. The eleventh hypothesis follows
directly from Amendment 10's documentation: a model trained on absolute pixel
coordinates may read the 0-999 numbers as pixels. Test (scripts/misread.py,
results/MISREAD.txt): for each locate trial, the option whose numbers, read as
pixels, lie nearest the object's bbox centre is the "misread-predicted" slot.

- Qwen2.5-VL-7B, normalized: argmax equals the misread-predicted slot on 53.0%
  of 200 trials; the bias-aware null (product of the model's marginal answer
  distribution and the predicted-slot distribution) is 11.8%; exact binomial
  p = 7.8e-48 against 1/9. The misread-predicted slot coincides with the true
  target on only 2.5% of trials, which is why the accuracy falls below chance.
- Qwen3-VL-4B, pixel numbers read as 0-999 fractions: 28.0% vs 11.4% null,
  p = 4.6e-11. Weaker, same direction, mirror of the above.
- InternVL3-2B, normalized read as pixels: 10.5% vs 10.2% null, p = 0.64. Null,
  as expected for a model with no documented pixel convention.

Status: post hoc, one hypothesis, not pre-registered; it survives any correction
for the eleven tests. The paper reports it as the resolution of the Amendment 10
gap and does not extend it beyond the three models tested. The limitation
sentence "unexplained after ten hypotheses" is replaced.

Addendum, same day: (a) a coincidence-excluded statistic was added (trials where
the misread option is the true target are dropped): Qwen2.5-VL 54.4% vs 11.9%
null, Qwen3-VL 26.4% vs 11.6%, InternVL3 10.8% vs 10.2%. (b) The test was applied
to Gemini and GPT-4o in both directions using geometry joined from the
byte-identical Qwen3 prompts; all four are null after exclusion (0.0% to 15.0%
against 10.9% to 11.5%, smallest p = 0.07). The misreading explains the open-model
inversions and not the frontier gaps. Seven tests in total; the two positives
survive any correction.
