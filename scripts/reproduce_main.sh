#!/usr/bin/env bash
# Main tables: (1) training recipes on the probe set, (2) the deployed system on the confirmation set.
#   1. frozen data (CPU, seconds; byte-identical to the paper's files, see tests/golden)
#   2. write: P / M / ctrl / cons / prec adapters (800 steps each) + probe-set evaluation
#   3. repair: 120-step closure-format continuation from prec
#   4. system vs baselines on 3 x 1500 confirmation requests; IFEval guard; audits
# Approximate compute (A100 80GB, Qwen3.5-4B): 15 x 1.8 h training + 3 x 0.1 h repair + evaluation.
source "$(dirname "$0")/common.sh"

"$PY" -m spm.experiments prepare --root "$DATA"
ifeval "$EVALS/ifeval/base"

for s in $SEEDS; do
  evaluate "$DATA/recipe/data/$RT/test-s$s.jsonl" "$EVALS/recipe/probe/base-s$s"
  for arm in P M ctrl cons prec; do
    train "$arm-s$s" "$DATA/recipe/data/$RT/$arm-s$s.jsonl" "$s" 800 2e-4
    evaluate "$DATA/recipe/data/$RT/test-s$s.jsonl" "$EVALS/recipe/probe/$arm-s$s" "$RUNS/$arm-s$s"
  done
  ifeval "$EVALS/ifeval/prec-s$s" "$RUNS/prec-s$s"

  train "repair-s$s" "$DATA/closeout/repair/data/repair-s$s.jsonl" "$s" 120 1e-4 "$RUNS/prec-s$s"
  ifeval "$EVALS/ifeval/repair-s$s" "$RUNS/repair-s$s"

  for cell in repair:closed P:closed cons:closed prec:full base:matched base:full; do
    arm=${cell%%:*}; v=${cell##*:}
    adapter=""; [ "$arm" != base ] && adapter="$RUNS/$arm-s$s"
    evaluate "$(view confirm "$v" "$s")" "$EVALS/closeout/$v/$arm-s$s" $adapter
  done
done

"$PY" -m spm.experiments audit-recipe --evals "$EVALS" --out "$OUT/audit_recipe.json" --seeds $SEEDS
"$PY" -m spm.experiments audit-closeout --evals "$EVALS" --out "$OUT/audit_closeout.json" --seeds $SEEDS
"$PY" -m spm.experiments rescore "$EVALS/closeout/*/*/eval.jsonl" --out "$OUT/rescore_closeout.json"
echo "done: $OUT/audit_recipe.json $OUT/audit_closeout.json"
