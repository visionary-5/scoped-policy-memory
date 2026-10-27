"""Greedy generation + scoring of a frozen evaluation file with vLLM (GPU).

Input rows carry ``context`` (system-side policy text or None) and optional few-shot ``shots``.
Each record stores the output, the primary verdict, the expression-aware verdict, truncation, prompt
tokens and (when ``old_answer`` exists) whether the output equals the stale memory answer.

    python -m spm.eval.generate --test FILE --model BASE [--adapter RUN_DIR] --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..baselines.prompting import messages
from ..data.world import max_new_tokens
from ..io import read_jsonl, save_json, sha256_file, write_jsonl
from .probes import summarize, summarize_expr
from .score import correct_expr, score_call


def load_llm(model: str, adapter: str | None, max_model_len: int = 32768, gpu_memory_utilization: float = .85):
    from vllm import LLM
    if adapter:
        from vllm.lora.request import LoRARequest
        llm = LLM(model, enable_lora=True, max_lora_rank=64, max_loras=1, max_model_len=max_model_len,
                  enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization)
        return llm, {'lora_request': LoRARequest('spm', 1, str(Path(adapter) / 'adapter'))}
    return LLM(model, max_model_len=max_model_len, enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization), {}


def generate_and_score(llm, tokenizer, rows, rule_type, **kwargs):
    from vllm import SamplingParams
    M = max_new_tokens(rule_type)
    prompts = [tokenizer.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for r in rows]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=M), **kwargs)
    records = []
    for r, o in zip(rows, outs):
        c = o.outputs[0]; tr = c.finish_reason == 'length' or len(c.token_ids) >= M
        s = score_call(c.text, r['answer'], truncated=tr)
        rec = {**{k: v for k, v in r.items() if k not in ('context', 'shots')}, 'output': c.text, **s,
               'correct_expr': correct_expr(c.text, r['answer'], tr, s['correct']), 'prompt_tokens': len(o.prompt_token_ids)}
        if r.get('old_answer') is not None:
            rec['old'] = score_call(c.text, r['old_answer'], truncated=tr)['correct']
        records.append(rec)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', type=Path, required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', help='training run directory containing adapter/')
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--rule-type', default='ccgen_t12a3s1_lite')
    ap.add_argument('--max-model-len', type=int, default=32768); ap.add_argument('--gpu-mem', type=float, default=.85)
    a = ap.parse_args(argv)
    if (a.out / 'metrics.json').exists():
        print('SKIP (metrics.json exists)', a.out); return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    llm, kw = load_llm(a.model, a.adapter, a.max_model_len, a.gpu_mem)
    rows = read_jsonl(a.test)
    records = generate_and_score(llm, tok, rows, a.rule_type, **kw)
    write_jsonl(a.out / 'eval.jsonl', records)
    seed = None
    if a.adapter and (Path(a.adapter) / 'train_meta.json').exists():
        seed = json.loads((Path(a.adapter) / 'train_meta.json').read_text()).get('seed')
    save_json(a.out / 'metrics.json', {'rows': summarize(records), 'rows_expr': summarize_expr(records), 'seed': seed,
                                       'adapter': a.adapter, 'model': a.model, 'test_sha256': sha256_file(a.test)})
    print(json.dumps(summarize(records)), flush=True)


if __name__ == '__main__':
    main()
