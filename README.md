# Scoped Policy Memory (SPM)

Keep an agent's stable policy in LoRA weights, and serve each later rule edit as a small,
dependency-closed context, so that a small model follows both its memory and the edit.

## Method

```
                 policy v0 (all rules)                              rule store (versioned, scoped)
                        |                                           v1: edit task 9   v2: edit task 4 ...
                        v                                                   |
   +---------------------------------------------+                         |
   | 1. WRITE  conflict-aware LoRA internalisation|                         |
   |    memory rows        (no policy in context)|                         |
   |  + Override rows      (context changes one  |                         |
   |                        rule; label follows  |                         |
   |                        the context)         |                         |
   |  + Substitute rows    (foreign policy in    |                         |
   |                        context)             |                         |
   +---------------------------------------------+                         |
                        |                                                   |
   +---------------------------------------------+                         |
   | 2. REPAIR 120-step continuation in the      |                         |
   |    closure context format (conflict rows)   |                         |
   +---------------------------------------------+                         |
                        |                                                   v
                        v                         request --> 3. ROUTE (task-ID oracle | BM25 | dense)
             base model + LoRA  <---- context ----            |  top-1 section changed? -> CLOSURE
                        |                                     |     = shared constants + changed section
                        v                                     |  unchanged             -> no context
                 one-line tool call                           |  ambiguous / low margin -> full policy
```

* **Write.** The policy is internalised with memory practice plus conflict samples: the context states a
  rule that differs from what is being memorised and the label follows the context (Override on a
  subset of tasks, Substitute with foreign policies). Plain SFT learns the policy but then ignores
  in-context edits; conflict samples keep both.
* **Store.** Edits are versioned and scoped (which task changed, at which version). Only edits newer than
  the adapter are served via context.
* **Closure.** An edited rule is served together with the shared definitions it depends on
  (dependency closure), not as a bare diff; unaffected requests get no context at all.
* **Route.** The request's scope is found by task-ID parsing, BM25 or a dense retriever; ambiguous or
  low-margin decisions fall back to the full current policy (margin threshold calibrated on dev data).
* **Repair.** A 120-step continuation teaches the adapter the closure context format.

Benchmark: CC-Gen-ST-Lite, a single-turn, program-scored derivative of CC-Gen (CAP-CPT,
arXiv 2510.11588) with the official Override / Substitute constructions.

## Install

```bash
pip install -e .                 # core: data, routing, scoring, CLI (standard library only)
pip install -e ".[test]"         # + pytest
pip install -r requirements.txt  # full GPU stack for training / evaluation (torch, peft, vLLM, IFEval deps)
```

Environment variables: `SPM_MODEL_DIR` (base model, e.g. Qwen3.5-4B), `SPM_OUTPUT_DIR` (default
`./outputs`), optional `SPM_DENSE_MODEL` (bge-small-en-v1.5), `SPM_GEN_MODEL` and `CAPCPT_PROMPT_FILE`
(CAP-CPT baseline), `SPM_IFEVAL_DIR` (defaults to the vendored `third_party/ifeval`).

## Quick start: write -> update -> serve

The mock backend runs on CPU and only demonstrates the data flow (its scores are not results).
Use `--backend real` with `SPM_MODEL_DIR` on a GPU for real training and inference.

```bash
spm init   --ws ws/demo --rule-type ccgen_t12a3s1_lite --seed 0 --backend mock
spm write  --ws ws/demo                  # conflict-aware data -> LoRA (800 steps on real backend)
spm repair --ws ws/demo                  # 120-step closure-format continuation
spm update --ws ws/demo --task 9 --show  # edit one rule; stored as version 1
spm serve  --ws ws/demo --context-policy closed --router oracle
spm serve  --ws ws/demo --context-policy closed --router bm25 --tau $TAU   # TAU: routing/routing.json of `spm-exp route`
spm serve  --ws ws/demo --context-policy full                             # full-policy baseline
spm status --ws ws/demo
```

An already trained adapter can be adopted instead of training: `spm write --ws ws/demo --adopt RUN_DIR
--adopt-data prec-s0.jsonl` (and likewise `spm repair --adopt`).

## Reproduce the paper

All data are regenerated deterministically on CPU in seconds (`spm-exp prepare`); every frozen file is
byte-identical to the one used in the paper (checked by `tests/test_golden_data.py`).

```bash
export SPM_MODEL_DIR=/path/to/Qwen3.5-4B SPM_OUTPUT_DIR=outputs
bash scripts/reproduce_main.sh       # training-recipe table (P / M / ctrl / cons / prec) + system table + IFEval
bash scripts/reproduce_ablation.sh   # dev selection, conflict vs consistent repair, raw/full/closed views,
                                     # BM25 / dense routing with fallback, few-shot prompting, expression re-scoring
bash scripts/reproduce_capcpt.sh     # CAP-CPT (sftA / capcpt / capcpt_nosim / capcpt_conf)
bash scripts/reproduce_analysis.sh   # appendix: synthetic-world exposure (E7/E8) and conflict-vs-exposure gate
```

Each script writes per-item `eval.jsonl` files (the source of every number) and an audit JSON with the
frozen gates and paired bootstrap intervals. Reported numbers, tables and figures will be added
under `results/` once every number has been re-checked against its audit file.

## Tests

```bash
pytest                    # CPU only; ~20 s
```

* `test_golden_data.py`, `test_golden_analysis.py`: regenerated data files against SHA-256 recorded by
  the original experiments (`tests/golden/frozen_sha256.json`).
* `test_reference_parity.py`: item-by-item comparison with the original implementation on unfrozen
  seeds and rule types (runs only if `SPM_REFERENCE_REPO` points to it).
* scoring and expression re-scoring, dependency closure, routing and fallback, conflict-sample
  construction, versioned store and CLI.

## Layout

```
spm/
  data/        worlds (CC-Gen-ST, synthetic mixed), policy variants, prompts, memory-practice recipe
  training/    conflict rows (Override / Substitute / closure repair), frozen datasets, LoRA trainer
  runtime/     versioned store, dependency closure, routers + fallback, backends, `spm` CLI
  eval/        scoring (+ expression re-check), probe set, confirmation sets and gates, vLLM runner,
               IFEval wrapper, paired statistics
  baselines/   P / M / ctrl / cons recipes, CAP-CPT data, prompting (full / matched / few-shot)
  experiments.py   `spm-exp`: prepare data, routing views, CAP-CPT data, audits, re-scoring
analysis/      synthetic-world mechanism experiments (exposure arms, conflict-vs-exposure gate)
configs/       frozen protocols of the reported experiments (no machine paths)
scripts/       one entry point per table
tests/         CPU tests and golden SHA-256
third_party/ifeval   official IFEval evaluation code (Apache-2.0)
```

## Citation

```bibtex
@misc{spm2026,
  title  = {Scoped Policy Memory: Conflict-Aware Internalization with Dependency-Closed Policy Edits},
  author = {<AUTHORS>},
  year   = {2026},
  note   = {<VENUE / ARXIV ID>}
}
```

License: MIT (see `LICENSE`).
