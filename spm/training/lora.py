"""LoRA trainer for frozen training files (GPU; needs torch, transformers, peft).

Stages:
  sft  rows ``{messages, oracle, item, ...}``; loss on answer tokens only. Every non-statement row is
       re-verified against the world oracle before training (split must be ``train``).
  cpt  rows ``{"text"}``; loss on all tokens (CAP-CPT continued pre-training), one pass by default.

``--init-adapter RUN_DIR`` continues training an existing LoRA (same rank / alpha / target modules,
fresh optimizer). This is how the 120-step closure repair continues from the prec adapter and how
CAP-CPT's SFT stage continues from its CPT adapter.

Fixed recipe (paper): rank 16, alpha 32, dropout 0, all linear layers of the language model, AdamW
(weight decay 0), linear warmup over 5% then linear decay, grad-clip 1.0, batch 16 with micro-batch
4 and answer-token-weighted gradient accumulation, greedy shuffling with ``random.Random(seed)``.

    python -m spm.training.lora --model BASE --type ccgen_t12a3s1_lite --seed 0 --K 12 \\
        --train-data-file prec-s0.jsonl --out runs/prec-s0 --max-steps 800 --lr 2e-4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lora_targets(model) -> list[str]:
    """Every nn.Linear inside the language-model layers (vision tower and lm_head excluded)."""
    import torch.nn as nn
    names = set()
    multimodal = hasattr(model.config, "vision_config")
    for n, m in model.named_modules():
        in_language = ".language_model.layers." in n if multimodal else ".layers." in n
        if isinstance(m, nn.Linear) and in_language and "visual" not in n and "vision" not in n:
            names.add(n)
    return sorted(names)


def check_adapter_config(have, wanted, module_names):
    """Compare effective target modules (PEFT may store suffix specifications), rank and alpha."""
    def expand(spec):
        if isinstance(spec, str):
            return {n for n in module_names if re.fullmatch(spec, n)}
        return {n for n in module_names if any(n == t or n.endswith('.' + t) for t in spec)}
    actual, expected = expand(have.target_modules), expand(wanted.target_modules)
    if have.r != wanted.r or have.lora_alpha != wanted.lora_alpha or not expected or actual != expected:
        raise RuntimeError("init adapter rank/alpha/target modules differ from this run's configuration")


def load_sft_rows(path, world, K, seed, rule_type):
    from ..data.world import lookup_solve
    from ..eval.score import score_call
    data = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not data:
        raise RuntimeError("training file is empty")
    for d in data:
        if (d["world_hash"], d["K"], d["seed"], d["rule_type"]) != (world.hash(), K, seed, rule_type):
            raise RuntimeError("training file does not match the world")
        if d["kind"] != "statement":
            if d["item"]["split"] != "train" or not score_call(d["oracle"], lookup_solve(world, K, d["item"]))["correct"]:
                raise RuntimeError("training file has a non-train item or a wrong oracle label")
        d["label"] = d["oracle"]
    return data


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["sft", "cpt"], default="sft")
    ap.add_argument("--model", required=True); ap.add_argument("--type", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--train-data-file", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro-bs", type=int, default=4, help="gradient-accumulation micro batch (0 = none)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=3, help="sft: min epochs (if --max-steps unset); cpt: passes")
    ap.add_argument("--min-steps", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=2048, help="cpt: per-document token cap (>0.5%% truncated = error)")
    ap.add_argument("--init-adapter", type=Path, default=None, help="run directory with adapter/ to continue from")
    ap.add_argument("--replay-file", type=Path, default=None, help="optional base-model self-answer replay JSONL {prompt, response}")
    ap.add_argument("--replay-ratio", type=float, default=0.0)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    alpha = a.alpha or 2 * a.rank
    out = a.out
    if (out / "train_meta.json").exists():
        meta = json.loads((out / "train_meta.json").read_text())
        if meta.get("train_data_sha256") != sha(a.train_data_file):
            raise SystemExit(f"training data changed; refusing to reuse {out}")
        print("SKIP (already trained)", out, flush=True)
        return
    if (out / "train_log.jsonl").exists():
        raise SystemExit(f"an interrupted run exists; inspect and remove {out}/train_log.jsonl first")
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover - GPU environment only
        raise SystemExit(f"training needs torch, transformers and peft ({e}); see requirements.txt") from e
    from ..data.world import build_world
    torch.manual_seed(a.seed); random.seed(a.seed)
    w = build_world(a.type, a.seed)
    tokz = AutoTokenizer.from_pretrained(a.model)
    end = tokz.eos_token or ""
    n_cut = 0
    if a.stage == "sft":
        data = load_sft_rows(a.train_data_file, w, a.K, a.seed, a.type)
        n_replay = 0
        if a.replay_file and a.replay_ratio > 0:
            rep = [json.loads(l) for l in open(a.replay_file)]
            rrng = random.Random(f"replay-{a.seed}")
            n_replay = min(len(rep), round(a.replay_ratio * len(data)))
            for x in rrng.sample(rep, n_replay):
                data.append({"uid": "replay", "kind": "replay", "rule_id": None, "item": None,
                             "messages": [{"role": "user", "content": x["prompt"]}], "label": x["response"]})

        def encode(d):
            p = tokz.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            pi = tokz(p, add_special_tokens=False)["input_ids"]
            ci = tokz(d["label"] + end, add_special_tokens=False)["input_ids"]
            return pi + ci, [-100] * len(pi) + ci
        enc = [encode(d) for d in data]
        steps_per_epoch = math.ceil(len(enc) / a.bs)
        total = a.max_steps or steps_per_epoch * max(int(a.epochs), math.ceil(a.min_steps / steps_per_epoch))
    else:
        data = [json.loads(l) for l in a.train_data_file.read_text().splitlines()]
        n_replay = 0; enc = []
        for d in data:
            ids = tokz(d["text"] + end, add_special_tokens=False)["input_ids"]
            if len(ids) > a.max_len:
                n_cut += 1; ids = ids[:a.max_len]
            enc.append((ids, list(ids)))
        if n_cut > 0.005 * len(enc):
            raise RuntimeError(f"too many over-long CPT documents: {n_cut}/{len(enc)}")
        steps_per_epoch = math.ceil(len(enc) / a.bs)
        total = a.max_steps or max(1, math.ceil(steps_per_epoch * a.epochs))

    config = AutoConfig.from_pretrained(a.model)
    if config.model_type == "qwen3_5":
        from transformers import AutoModelForImageTextToText as loader
    else:
        loader = AutoModelForCausalLM
    model = loader.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    targets = lora_targets(model)
    if not targets:
        raise RuntimeError("no language-model LoRA targets found")
    cfg = LoraConfig(r=a.rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM")
    if a.init_adapter:
        init_dir = Path(a.init_adapter) / "adapter"
        from types import SimpleNamespace
        module_names = [n for n, _ in model.named_modules()]
        stored = json.loads((init_dir / "adapter_config.json").read_text())
        check_adapter_config(SimpleNamespace(**stored), cfg, module_names)
        model = PeftModel.from_pretrained(model, str(init_dir), is_trainable=True)
        check_adapter_config(model.peft_config["default"], cfg, module_names)
    else:
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    adapted = [n for n, p in model.named_parameters() if "lora_" in n]
    if not adapted or any(not p.requires_grad for n, p in model.named_parameters() if "lora_" in n):
        raise RuntimeError("LoRA parameters are not trainable")
    if any("visual" in n or "vision" in n or ".layers." not in n
           or (config.model_type == "qwen3_5" and ".language_model.layers." not in n) for n in adapted):
        raise RuntimeError("LoRA outside the language-model layers")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 20)) * max(0.0, (total - s) / total))
    pad = tokz.pad_token_id if tokz.pad_token_id is not None else tokz.eos_token_id

    out.mkdir(parents=True, exist_ok=True)
    log = out / "train_log.jsonl"
    rng = random.Random(a.seed); step = 0; t0 = time.time(); ntok = 0
    model.train()
    while step < total:
        order = list(range(len(enc))); rng.shuffle(order)
        for b in range(0, len(order), a.bs):
            if step >= total:
                break
            batch = [enc[i] for i in order[b:b + a.bs]]
            mb = a.micro_bs or len(batch)
            if mb >= len(batch):
                L = max(len(x) for x, _ in batch)
                ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in batch], device="cuda")
                lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device="cuda")
                att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device="cuda")
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward(); ntok += int(att.sum())
            else:  # answer-token-weighted accumulation == token mean over the full batch
                n_lab = sum(sum(1 for t in y[1:] if t != -100) for _, y in batch)
                tot = 0.0
                for c in range(0, len(batch), mb):
                    chunk = batch[c:c + mb]; L = max(len(x) for x, _ in chunk)
                    ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in chunk], device="cuda")
                    lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in chunk], device="cuda")
                    att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in chunk], device="cuda")
                    k = int((lab[:, 1:] != -100).sum())
                    part = model(input_ids=ids, attention_mask=att, labels=lab).loss * (k / n_lab)
                    part.backward(); tot += float(part); ntok += int(att.sum())
                loss = torch.tensor(tot)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == total:
                with open(log, "a") as f:
                    f.write(json.dumps({"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                                        "t": round(time.time() - t0, 1)}) + "\n")
    model.save_pretrained(out / "adapter")
    tokz.save_pretrained(out / "adapter")
    meta = {"stage": a.stage, "model": a.model, "rule_type": a.type, "K": a.K, "seed": a.seed, "world_hash": w.hash(),
            "lora_rank": a.rank, "alpha": alpha, "lr": a.lr, "bs": a.bs, "micro_bs": a.micro_bs, "total_steps": total,
            "steps_per_epoch": steps_per_epoch, "targets": targets, "n_train": len(enc), "n_replay": n_replay,
            "n_truncated": n_cut, "train_data_file": str(a.train_data_file), "train_data_sha256": sha(a.train_data_file),
            "init_adapter": str(a.init_adapter) if a.init_adapter else None,
            "init_adapter_weights_sha256": (sha(next((Path(a.init_adapter) / "adapter").glob("adapter_model.*")))
                                            if a.init_adapter else None),
            "train_seconds": time.time() - t0, "train_tokens": ntok, "enable_thinking": False}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print("DONE", out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
