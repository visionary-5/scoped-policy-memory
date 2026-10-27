#!/usr/bin/env bash
# CAP-CPT baseline under identical conditions (run after reproduce_main.sh's prepare step).
#   CAPCPT_PROMPT_FILE  official Utils/Policy_Analysis/Final_Prompting.md from the CAP-CPT release
#   SPM_GEN_MODEL       data-generation LLM (paper: Qwen3-32B)
# Arms: sftA, capcpt, capcpt_nosim, capcpt_conf (our 128 conflict rows stacked on CAP-CPT).
# Approximate compute: 21-29 GPU-hours for three seeds.
source "$(dirname "$0")/common.sh"
: "${CAPCPT_PROMPT_FILE:?path to Final_Prompting.md}"
: "${SPM_GEN_MODEL:?data-generation LLM (e.g. Qwen/Qwen3-32B)}"
CD=$DATA/capcpt/data/$RT

"$PY" -m spm.experiments capcpt-prepare --root "$DATA" --prompt-file "$CAPCPT_PROMPT_FILE" --seeds $SEEDS
for s in $SEEDS; do
  "$PY" -m spm.experiments capcpt-gen --root "$DATA" --seed "$s" --gen-model "$SPM_GEN_MODEL"
  "$PY" -m spm.experiments capcpt-assemble --root "$DATA" --seed "$s" --model "$SPM_MODEL_DIR"
  for corpus in cpt_full cpt_nosim; do   # continued pre-training: all tokens, one pass
    "$PY" -m spm.training.lora --stage cpt --model "$SPM_MODEL_DIR" --type "$RT" --seed "$s" --K 12 \
      --train-data-file "$CD/$corpus-s$s.jsonl" --out "$RUNS/$corpus-s$s" --bs 32 --micro-bs 16 --epochs 1 --max-len 2048 --lr 2e-4
  done
  train "sftA-s$s"         "$CD/P_anchor-s$s.jsonl"    "$s" 800 2e-4
  train "capcpt-s$s"       "$CD/P_anchor-s$s.jsonl"    "$s" 800 2e-4 "$RUNS/cpt_full-s$s"
  train "capcpt_nosim-s$s" "$CD/P_anchor-s$s.jsonl"    "$s" 800 2e-4 "$RUNS/cpt_nosim-s$s"
  train "capcpt_conf-s$s"  "$CD/prec_anchor-s$s.jsonl" "$s" 800 2e-4 "$RUNS/cpt_full-s$s"
  for arm in sftA capcpt capcpt_nosim capcpt_conf; do
    evaluate "$DATA/recipe/data/$RT/test-s$s.jsonl" "$EVALS/recipe/probe/$arm-s$s" "$RUNS/$arm-s$s"
    evaluate "$DATA/capcpt/pid/data/$RT/test-s$s.jsonl" "$EVALS/recipe/pid/$arm-s$s" "$RUNS/$arm-s$s"
  done
done
"$PY" -m spm.experiments audit-recipe --evals "$EVALS" --out "$OUT/audit_capcpt.json" --seeds $SEEDS \
  --extra-arms sftA capcpt capcpt_nosim capcpt_conf
echo "done: $OUT/audit_capcpt.json"
