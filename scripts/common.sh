# Shared settings for the reproduction scripts. Source it; do not run it.
#   SPM_MODEL_DIR   base model directory or HF id (required for GPU steps), e.g. Qwen/Qwen3.5-4B
#   SPM_OUTPUT_DIR  where data, runs and evaluations go (default: ./outputs)
#   SEEDS           seeds to run (default: "0 1 2")
#   PYTHON          interpreter (default: python)
# Hyperparameters mirror configs/train.json.
set -euo pipefail
: "${SPM_MODEL_DIR:?set SPM_MODEL_DIR to the base model (e.g. Qwen/Qwen3.5-4B)}"
OUT=${SPM_OUTPUT_DIR:-outputs}
DATA=$OUT/data
RUNS=$OUT/runs
EVALS=$OUT/evals
PY=${PYTHON:-python}
RT=ccgen_t12a3s1_lite
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p "$DATA" "$RUNS" "$EVALS"

# train NAME DATA_FILE SEED STEPS LR [INIT_RUN_DIR]   (answer-token SFT; skips if already trained)
train() {
  "$PY" -m spm.training.lora --model "$SPM_MODEL_DIR" --type "${TYPE:-$RT}" --seed "$3" --K "${K:-12}" \
    --train-data-file "$2" --out "$RUNS/$1" --max-steps "$4" --lr "$5" --bs 16 --micro-bs 4 ${6:+--init-adapter "$6"}
}

# evaluate TEST_FILE OUT_DIR [ADAPTER_RUN_DIR]   (greedy vLLM; skips if metrics.json exists)
evaluate() {
  "$PY" -m spm.eval.generate --model "$SPM_MODEL_DIR" --rule-type "${TYPE:-$RT}" --test "$1" --out "$2" ${3:+--adapter "$3"}
}

# ifeval OUT_DIR [ADAPTER_RUN_DIR]   (needs $EVALS/ifeval/base first for adapters)
ifeval() {
  if [ -n "${2:-}" ]; then
    "$PY" -m spm.eval.ifeval --model "$SPM_MODEL_DIR" --out "$1" --adapter "$2" --base-dir "$EVALS/ifeval/base"
  else
    "$PY" -m spm.eval.ifeval --model "$SPM_MODEL_DIR" --out "$1"
  fi
}

view() {  # view PHASE VIEW SEED -> path of a closeout evaluation file
  echo "$DATA/closeout/$1/views/$2/data/$RT/test-s$3.jsonl"
}
