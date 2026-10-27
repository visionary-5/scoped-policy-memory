#!/usr/bin/env bash
# Appendix mechanism analyses on the synthetic mixed world (K=16):
#   E7/E8 exposure arms W / C / X / M / R (seeds 0-2) and gate V2 conf_k05 vs cons_u05 (seed 0).
source "$(dirname "$0")/common.sh"
A=$OUT/analysis
export TYPE=mixed K=16

"$PY" -m analysis.exposure prepare --root "$A"
"$PY" -m analysis.gate_v2 prepare --root "$A/gate"
for s in $SEEDS; do
  for arm in W C X M R; do
    train "mixed-$arm-s$s" "$A/data/$arm-s$s.jsonl" "$s" 800 2e-4
    evaluate "$A/data/test-s$s.jsonl" "$EVALS/analysis/$arm-s$s" "$RUNS/mixed-$arm-s$s"
  done
done
for arm in conf_k05 cons_u05; do
  train "mixed-$arm-s0" "$A/gate/data/$arm-s0.jsonl" 0 800 2e-4
  evaluate "$A/data/test-s0.jsonl" "$EVALS/analysis/$arm-s0" "$RUNS/mixed-$arm-s0"
done
"$PY" -m analysis.exposure audit --root "$A" --evals "$EVALS/analysis" --seeds $SEEDS
"$PY" -m analysis.gate_v2 audit --root "$A" --evals "$EVALS/analysis"
echo "done: $A/audit.json $A/audit_v2.json"
