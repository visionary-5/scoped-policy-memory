"""IFEval capability guard: the official 541-prompt IFEval, greedy, 1024 new tokens, strict and loose.

The official evaluation code (google-research, Apache-2.0) is vendored under ``third_party/ifeval``
with file hashes in ``SOURCE.json``; every run verifies them. Override the location with
``SPM_IFEVAL_DIR``. Requires vllm, nltk, langdetect, absl-py.

    python -m spm.eval.ifeval --model BASE --out DIR                         # base model
    python -m spm.eval.ifeval --model BASE --adapter RUN --base-dir DIR --out DIR   # adapter + delta
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from ..io import save_json, write_jsonl
from .stats import acc_stats

VENDOR = Path(os.environ.get("SPM_IFEVAL_DIR", Path(__file__).resolve().parents[2] / "third_party" / "ifeval"))
MAX_NEW = 1024


def preflight():
    source = json.loads((VENDOR / "SOURCE.json").read_text())
    for name, expected in source["sha256"].items():
        actual = hashlib.sha256((VENDOR / "instruction_following_eval" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"IFEval vendored file hash mismatch: {name}")
    sys.path.insert(0, str(VENDOR))
    from instruction_following_eval import evaluation_lib, instructions_util
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    random.seed(0)
    if instructions_util.count_sentences("First sentence. Second sentence.") != 2:
        raise RuntimeError("IFEval NLTK sentence splitter check failed")
    examples = evaluation_lib.read_prompt_list(VENDOR / "instruction_following_eval/data/input_data.jsonl")
    if len(examples) != 541 or len({x.key for x in examples}) != 541:
        raise RuntimeError("IFEval must use the full 541 prompts")
    return evaluation_lib, examples, source


def evaluate(llm, tokenizer, out, lora=None):
    from vllm import SamplingParams
    lib, examples, source = preflight()
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": x.prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for x in examples]
    kwargs = {"lora_request": lora} if lora is not None else {}
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=MAX_NEW), **kwargs)
    if len(outputs) != len(examples):
        raise RuntimeError("IFEval generation did not return all 541 outputs")
    records = []
    for example, output in zip(examples, outputs):
        completion = output.outputs[0]
        truncated = completion.finish_reason == "length" or len(completion.token_ids) >= MAX_NEW
        response = {example.prompt: completion.text}
        strict = lib.test_instruction_following_strict(example, response)
        loose = lib.test_instruction_following_loose(example, response)
        records.append({"key": example.key, "prompt": example.prompt, "output": completion.text,
                        "instruction_ids": example.instruction_id_list,
                        "strict": strict.follow_all_instructions and not truncated,
                        "correct": loose.follow_all_instructions and not truncated,
                        "strict_instructions": [bool(v) and not truncated for v in strict.follow_instruction_list],
                        "loose_instructions": [bool(v) and not truncated for v in loose.follow_instruction_list],
                        "truncated": truncated, "prompt_tokens": len(output.prompt_token_ids),
                        "output_tokens": len(completion.token_ids)})
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "ifeval.jsonl", records)
    metrics = acc_stats("eval/general/ifeval_acc", sum(r["correct"] for r in records), len(records))
    metrics.update(acc_stats("eval/general/ifeval_strict_acc", sum(r["strict"] for r in records), len(records)))
    for mode in ("strict", "loose"):
        judgments = [v for r in records for v in r[f"{mode}_instructions"]]
        metrics.update(acc_stats(f"eval/general/ifeval_{mode}_instruction_acc", sum(judgments), len(judgments)))
    metrics.update({"eval/general/ifeval_truncation_rate": sum(r["truncated"] for r in records) / len(records),
                    "eval/general/ifeval_mean_output_tokens": sum(r["output_tokens"] for r in records) / len(records),
                    "ifeval_revision": source["revision"], "ifeval_max_new_tokens": MAX_NEW,
                    "ifeval_data_sha256": source["sha256"]["data/input_data.jsonl"]})
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", help="training run directory containing adapter/ (requires --base-dir)")
    ap.add_argument("--base-dir", help="IFEval output directory of the base model")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "metrics.json").exists():
        print("SKIP (metrics.json exists)", out); return
    preflight()
    from transformers import AutoTokenizer
    from vllm import LLM
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if not a.adapter:
        llm = LLM(a.model, max_model_len=16384, enable_prefix_caching=True, gpu_memory_utilization=0.85)
        metrics = evaluate(llm, tokenizer, out)
        metrics["model"] = a.model
    else:
        if not a.base_dir:
            raise SystemExit("--adapter requires --base-dir")
        from vllm.lora.request import LoRARequest
        llm = LLM(a.model, enable_lora=True, max_lora_rank=64, max_loras=1, max_model_len=16384,
                  enable_prefix_caching=True, gpu_memory_utilization=0.85)
        metrics = evaluate(llm, tokenizer, out, LoRARequest("ife", 1, str(Path(a.adapter) / "adapter")))
        base = json.loads((Path(a.base_dir) / "metrics.json").read_text())
        for key in ("ifeval_revision", "ifeval_max_new_tokens", "ifeval_data_sha256"):
            if metrics[key] != base[key]:
                raise RuntimeError(f"IFEval base/adapter configuration differs: {key}")
        if base["model"] != a.model:
            raise RuntimeError("IFEval base result used a different model")
        metrics.update(model=a.model, adapter=a.adapter,
                       **{"eval/general/ifeval_delta": metrics["eval/general/ifeval_acc"] - base["eval/general/ifeval_acc"]})
    save_json(out / "metrics.json", metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
