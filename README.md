# What Looks Like a Capability Limit in Vision-Language Models Is a Readout Limit

Code, per-trial logs and the pre-registered analysis plan for the paper

> Alfredo F. Frontera Del Valle. *What Looks Like a Capability Limit in Vision-Language
> Models Is a Readout Limit.* arXiv:2609.27408, 2026. https://arxiv.org/abs/2609.27408

## Summary

Benchmarks for vision-language models write their answer options in some convention:
a color word, a hue angle, a grid cell name, a pixel coordinate. This work holds the
image, the question, the set of alternatives, their order and the output format fixed,
and changes only how the alternatives are described. On 200 COCO val2017 photographs,
Qwen3-VL-4B picks the correct one of nine cells for a named object 68.5% of the time
when the cells are named in English and 20.0% when the same cells are given as pixel
coordinates (chance 11.1%). Putting the coordinate in the question instead of the
answer options costs 3.5 points and is not significant. The gap holds on a 4x4 grid,
at 8-bit, and across object size, boundary distance and category, and it reverses the
ranking of two models depending on the coordinate system. A mispaired-name test
(each coordinate glossed with the wrong cell name) measures whether a convention is
read at all. The paper also reports five scoring errors in its own pipeline where a
capable model was measured as incapable.

## Quick check (CPU only, no GPU, no API keys)

Every number in the paper is computed from the per-trial logs in `results/`, so all
tables can be regenerated on a laptop in about a minute.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                               # 223 tests
python scripts/make_tables.py        # prints every table; compare with results/TABLES.txt
python scripts/check_numbers.py      # every one-decimal number in paper/main.tex appears in results/*.txt
python scripts/misread.py            # rewrites results/MISREAD.txt
python scripts/extra_checks.py       # rewrites results/EXTRA.txt
python scripts/audit.py              # adversarial checks on the color and first COCO logs
python scripts/make_figure_coco.py   # rewrites paper/figs/coco_locate.pdf from results/TABLES.txt
```

Run all commands from the repository root. `make_tables.py`, `analyze_coordinates.py`,
`robustness.py`, `audit.py` and `audit2.py` print to stdout; redirect to the matching
file in `results/` to regenerate it.

`scripts/robustness.py` and `scripts/audit2.py` also run on CPU, but they import
`scripts/coco_convention.py`, which imports `torch`, `transformers` and `pyarrow` at
module level, so they need `requirements-gpu.txt` installed. Check 3 of `audit2.py`
(do the frontier models' images match their prompts) also needs the COCO parquet files
described below.

## Where each result comes from

"Regenerate" runs on CPU from the included logs. "Collect" re-runs the model and needs a
CUDA GPU (open models) or paid API keys (frontier models).

| Paper | Regenerate (CPU) | Log(s) in `results/` | Collect |
|---|---|---|---|
| Table 1, color convention gradient | `make_tables.py` T1, `analyze_ladder.py` | `gpu_p3/interface_ladder_heldout.jsonl`, `gpu_p3/interface_ladder_recovery.jsonl` | `calibrate_palette.py`, `interface_ladder.py` (GPU) |
| Table 2, COCO locate / identify | `make_tables.py` T3 | `coco/convention.jsonl` | `coco_convention.py` (GPU, COCO) |
| Table 3, readership | `make_tables.py` T2 | `gpu_p3/interface_ladder_mismatch.jsonl`, `gpu_p3/coordinate_mismatch.jsonl`, `coco/convention.jsonl`, `coco/readership.jsonl` | `interface_ladder.py`, `coordinate_grounding.py`, `coco_convention.py` (GPU) |
| Table 4, rank inversions | `make_tables.py` T6 | `coco/convention_{qwen3vl4b,InternVL3-2B-hf,Qwen2_5-VL-7B-Instruct}.jsonl` | `coco_convention.py` (GPU) |
| Figure 1 | `make_figure_coco.py` (reads T5 of `TABLES.txt`) | as Table 6 | |
| Table 5, color vocabularies | `analyze_crossmodel.py` (four models); the Qwen3-VL-4B row is the vocabulary frozen in `PROTOCOL_FREEZE.md` section 5, which `interface_ladder.py` checks against its calibration file at run time | `gpu_p3/palette_calibration.json`, `crossmodel/*_calibration.json` | `calibrate_palette.py`, `crossmodel.py` (GPU) |
| Table 6, COCO across five models | `make_tables.py` T5, T10 | as Table 4, plus `coco/convention_qwen3vl4b_8bit.jsonl`, `frontier/*.jsonl` | `frontier_convention.py` (paid API) |
| Table 7, slices by size and boundary | `robustness.py` | as Table 4 | |
| Table 8, synthetic spatial arrays | `make_tables.py` T9, `analyze_coordinates.py` | `gpu_p3/coordinate_grounding.jsonl` | `coordinate_grounding.py` (GPU) |
| Section 7, color gap across models | `make_tables.py` T4, `analyze_crossmodel.py` | `crossmodel/*_contrast.jsonl` | `crossmodel.py` (GPU) |
| Section 6, readership as a predictor (H8) | `make_tables.py` T7 | `coco/h8_stage1.jsonl`, `coco/h8_stage2.jsonl` | `coco_convention.py` (GPU) |
| Section 5, 4x4 grid | `make_tables.py` T8 | `coco/granularity_4x4.jsonl` | `granularity.py` (GPU) |
| Section 4, calibration-angle ceiling | `make_tables.py` T11 | `gpu_p3/interface_ladder.jsonl` | `interface_ladder.py` (GPU) |
| Appendix A, the below-chance result | `misread.py`, `robustness.py` section 4 | as Table 4 | |
| Appendix A, readership per model on COCO pixels | no script prints these yet: 9 x the share of `pixel_mismatch` rows with `correct` true, per model | as Table 6 | |
| Appendix B, gate sensitivity and instrument agreement | `extra_checks.py` | `crossmodel/*_contrast.jsonl`, `TABLES.txt` | |
| Section 8 and the reproducibility statement, audits | `audit.py`, `audit2.py` | all of the above | |

Generated summaries in `results/`: `TABLES.txt` (make_tables), `COORD.txt`
(analyze_coordinates), `ROBUSTNESS.txt` (robustness), `MISREAD.txt` (misread),
`EXTRA.txt` (extra_checks), `AUDIT.txt` (audit), `AUDIT2.txt` (audit2). `GRAN.txt` is a
saved output of an inline check; every value in it is also printed by `make_tables.py`
(T8, T10, T11). `coco/convention.jsonl` and `coco/convention_qwen3vl4b.jsonl` are the
same run under two names, both read by the scripts.

## Re-collecting the data

### Models

Open weights, all from Hugging Face, loaded in 4-bit NF4 with greedy decoding unless
stated: `Qwen/Qwen3-VL-4B-Instruct` (primary), `Qwen/Qwen2.5-VL-7B-Instruct`,
`OpenGVLab/InternVL3-2B-hf`, `HuggingFaceTB/SmolVLM-Instruct`,
`llava-hf/llava-1.5-7b-hf`. Install `requirements-gpu.txt`; the logged runs used
`torch 2.11.0+cu128` and `transformers 5.9.0` on a single CUDA GPU.

Frontier models were called through their REST APIs on 20 September 2026 as
`gemini-flash-latest` and `gpt-4o`. Both names are aliases that the providers update,
so a re-run today may reach a different model. Keys are read from the environment
(`GEMINI_API_KEY`, `OPENAI_API_KEY`) and are never written to disk.

### COCO images

The photographs are not redistributed here. The scripts read the COCO 2017 validation
split as parquet shards from the Hugging Face dataset `detection-datasets/coco`, at
`$HF_HOME/hub/datasets--detection-datasets--coco/snapshots/*/data/val-*.parquet`
(`HF_HOME` defaults to `~/.cache/huggingface`). One way to fetch them:

```python
from huggingface_hub import snapshot_download
snapshot_download("detection-datasets/coco", repo_type="dataset",
                  allow_patterns=["data/val-*.parquet"])
```

COCO annotations are licensed CC BY 4.0 (Lin et al., 2014); the images carry their own
Flickr licenses. The logs in `results/coco/` contain COCO image ids, category names and
boxes, but no image data.

### Commands

The settings below are inferred from the conditions recorded in each log; the exact
command lines were not saved. `VWM_DTYPE=4bit` is the default.

```bash
# color vocabulary for Qwen3-VL-4B (5-degree grid, 3 renders per angle)
VWM_CP_STEP=5 python scripts/calibrate_palette.py
# color ladder: calibration angles, then held-out angles (Table 1), the gloss rungs, and the
# mispaired gloss (Table 3). The logs hold 173 trials per rung, so set VWM_IL_N to at least 173
# (default 100).
VWM_IL_RUNGS=free,freepad,word,label python scripts/interface_ladder.py
VWM_IL_HELDOUT=1 VWM_IL_RUNGS=word,label,numeric VWM_IL_OUT=results/gpu_p3/interface_ladder_heldout.jsonl python scripts/interface_ladder.py
VWM_IL_HELDOUT=1 VWM_IL_RUNGS=numeric_gloss,numeric_bare VWM_IL_OUT=results/gpu_p3/interface_ladder_recovery.jsonl python scripts/interface_ladder.py
VWM_IL_HELDOUT=1 VWM_IL_RUNGS=numeric_mismatch,numericleg_mismatch VWM_IL_OUT=results/gpu_p3/interface_ladder_mismatch.jsonl python scripts/interface_ladder.py
# synthetic spatial arrays (Table 8) and their mispaired gloss
python scripts/coordinate_grounding.py
VWM_CG_DIRS=locate VWM_CG_DESCS=pixel_gloss,pixel_mismatch VWM_CG_OUT=results/gpu_p3/coordinate_mismatch.jsonl python scripts/coordinate_grounding.py
# COCO, primary model (Table 2), then the normalized-coordinate readership arm and H8
python scripts/coco_convention.py
VWM_CC_DIRS=locate VWM_CC_DESCS=norm1000_gloss,norm1000_mismatch VWM_CC_OUT=results/coco/readership.jsonl python scripts/coco_convention.py
VWM_CC_DIRS=locate VWM_CC_DESCS=grid_mismatch,percent_mismatch VWM_CC_OUT=results/coco/h8_stage1.jsonl python scripts/coco_convention.py
VWM_CC_DIRS=locate VWM_CC_DESCS=grid,percent VWM_CC_OUT=results/coco/h8_stage2.jsonl python scripts/coco_convention.py
# COCO, other open models and the 8-bit control
VWM_CC_MODEL=OpenGVLab/InternVL3-2B-hf python scripts/coco_convention.py
VWM_CC_MODEL=Qwen/Qwen2.5-VL-7B-Instruct python scripts/coco_convention.py
VWM_CC_MODEL=Qwen/Qwen3-VL-4B-Instruct VWM_CC_BITS=8 VWM_CC_DIRS=locate VWM_CC_DESCS=cellname,norm1000,pixel VWM_CC_OUT=results/coco/convention_qwen3vl4b_8bit.jsonl python scripts/coco_convention.py
# 4x4 grid
python scripts/granularity.py
# color task across models (calibrates each model's own vocabulary first)
VWM_XM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct python scripts/crossmodel.py   # likewise for the other three
# frontier models (paid API)
GEMINI_API_KEY=... VWM_FR_MODEL=gemini-flash-latest python scripts/frontier_convention.py
OPENAI_API_KEY=... VWM_FR_MODEL=gpt-4o python scripts/frontier_convention.py
```

`scripts/cue_crossover.py` is the original cue-crossover experiment that
`interface_ladder.py` extends to every rung; it is kept for reference and no number in
the paper is read from its output.

## Layout

```
paper/        arXiv source (main.tex, app_prompts.tex, main.bbl, style files), figure, and the arXiv PDF
scripts/      experiment drivers, table and figure generators, audits
results/      per-trial JSONL logs behind every number, and the generated summaries
vwm/          shared library: stimuli, local model clients, parsing
tests/        unit tests for vwm/ (CPU, no network)
PROTOCOL_FREEZE.md   the frozen analysis plan with dated amendments 1 to 11, as of the arXiv submission
```

`vwm/` began as the library for an earlier visual-working-memory study by the same
author and still contains its modules (KV-cache surgery, target-confusability
analysis, ROC fitting); the tests cover them. The experiments in this paper use
`vwm.local` (model clients), `vwm.stimuli` (color wheel and array geometry) and
`vwm.utils`.

A good place to start reading is `scripts/coco_convention.py`: `describe()` and
`preamble()` write the same nine cells in each convention, `resolve_digits()` refuses
to score a model when no tokenization of the answer digits carries at least half of
the next-token probability, and the main loop builds the mispaired-name condition and
records both answer keys.

## Pre-registration

`PROTOCOL_FREEZE.md` holds the hypotheses, endpoints, exclusions and decision rules,
frozen before the held-out data existed, with every later change added as a dated
amendment. It includes the two predictions that failed (H5 and H8) and marks the
below-chance analysis of Appendix A as post hoc (amendment 11).

## Citation

```bibtex
@misc{fronteradelvalle2026readout,
  title         = {What Looks Like a Capability Limit in Vision-Language Models Is a Readout Limit},
  author        = {Frontera Del Valle, Alfredo F.},
  year          = {2026},
  eprint        = {2609.27408},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

Code and logs: MIT, see `LICENSE`. The LaTeX style files in `paper/`
(`iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty`, `fancyhdr.sty`)
are third-party files under their own licenses and are included only so the paper
source compiles. COCO-derived fields in the logs are under the COCO annotation license
(CC BY 4.0).
