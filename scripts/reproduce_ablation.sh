#!/usr/bin/env bash
# Runtime and repair ablations (run after reproduce_main.sh; reuses its adapters):
#   A. dev-set selection (seed 0): prec raw / closed / typed / full, P closed / typed, base matched / full
#   B. conflict vs consistent repair (120 steps each from prec), and repair under raw / full / closed views
#   C. real routing (BM25, optional dense) with dev-calibrated tau and full-policy fallback
#   D. prompting baselines with 3 worked examples (full_fs3 / matched_fs3) and expression-aware re-scoring
source "$(dirname "$0")/common.sh"

# A. development selection
for cell in prec:raw prec:closed prec:typed prec:full P:closed P:typed base:matched base:full repair:closed; do
  arm=${cell%%:*}; v=${cell##*:}
  adapter=""; [ "$arm" != base ] && adapter="$RUNS/$arm-s0"
  evaluate "$(view dev "$v" 0)" "$EVALS/closeout-dev/$v/$arm-s0" $adapter
done
"$PY" -m spm.experiments audit-dev --evals "$EVALS" --out "$OUT/audit_dev.json" \
  --cells prec:closed prec:typed repair:closed

# B. repair arms and views
for s in $SEEDS; do
  train "consistent-s$s" "$DATA/closeout/repair/data/consistent-s$s.jsonl" "$s" 120 1e-4 "$RUNS/prec-s$s"
  ifeval "$EVALS/ifeval/consistent-s$s" "$RUNS/consistent-s$s"
  for cell in consistent:closed repair:raw repair:full prec:closed; do
    arm=${cell%%:*}; v=${cell##*:}
    evaluate "$(view confirm "$v" "$s")" "$EVALS/closeout/$v/$arm-s$s" "$RUNS/$arm-s$s"
  done
done
"$PY" -m spm.experiments audit-closeout --evals "$EVALS" --out "$OUT/audit_ablation.json" --seeds $SEEDS \
  --cells repair:closed consistent:closed repair:raw repair:full prec:closed prec:full base:matched base:full

# C. routing
"$PY" -m spm.experiments route --root "$DATA" ${SPM_DENSE_MODEL:+--dense-model "$SPM_DENSE_MODEL"}
configs="bm25-explicit-k1 bm25-para-k1 bm25-para-k3 bm25-mask-k1"
[ -n "${SPM_DENSE_MODEL:-}" ] && configs="$configs dense-explicit-k1 dense-para-k1 dense-para-k3 dense-mask-k1"
for s in $SEEDS; do
  for c in $configs; do
    evaluate "$DATA/routing/views/$c/data/$RT/test-s$s.jsonl" "$EVALS/closeout/$c/repair-s$s" "$RUNS/repair-s$s"
  done
done
"$PY" -m spm.experiments audit-closeout --evals "$EVALS" --out "$OUT/audit_routing.json" --seeds $SEEDS \
  --cells repair:closed $(for c in $configs; do echo "repair:$c"; done) prec:full base:matched base:full

# D. few-shot prompting baselines (base model) and expression-aware re-scoring
for s in $SEEDS; do
  for v in full_fs3 matched_fs3; do
    evaluate "$(view confirm "$v" "$s")" "$EVALS/closeout/$v/base-s$s"
  done
done
"$PY" -m spm.experiments rescore "$EVALS/closeout/*/*/eval.jsonl" --out "$OUT/rescore_closeout.json"
echo "done: $OUT/audit_dev.json $OUT/audit_ablation.json $OUT/audit_routing.json $OUT/rescore_closeout.json"
