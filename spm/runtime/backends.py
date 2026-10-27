"""Backends with one interface:
  train(rows_path, out_dir, steps, lr, init=None) -> info
  answer(adapter_dir, queries) -> [output text]

``MockBackend`` (CPU) is a cartoon for demos and tests of the data flow ONLY; it is not evidence.
Memory: per task, the policy version that the majority of no-context training rows encode.
Context deference probability p: rises when the training file contains conflict practice rows and
otherwise stays at the plain-SFT level (0.12). Real numbers come only from ``RealBackend``.

``RealBackend`` trains with ``spm.training.lora`` (subprocess) and answers with vLLM (GPU).
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

CONFLICT_SOURCES = ("override", "substitute", "irrelevant_", "closeout_conflict")


class MockBackend:
    P_NAIVE, P_MAX, DECAY_STEPS, GAIN_STEPS = 0.12, 0.96, 60.0, 15.0

    def __init__(self, dom):
        self.dom = dom

    def train(self, rows_path, out_dir, steps, lr=None, init=None):
        rows = [json.loads(l) for l in Path(rows_path).read_text().splitlines()]
        state = json.loads((Path(init) / "mock_adapter.json").read_text()) if init else {"memory": {}, "p": None}
        votes = {}
        for d in rows:
            it = d.get("item")
            if not it or d["kind"] == "statement" or d["messages"][0]["content"].count("## Task Specifications"):
                continue  # only no-context rows write memory
            var = it["fields"].get("variant")
            votes.setdefault(str(it["fields"]["task"]), Counter())[var["vseed"] if var else None] += 1
        mem = dict(state["memory"])
        mem.update({k: c.most_common(1)[0][0] for k, c in votes.items()})
        frac = sum(str(d.get("source") or "").startswith(CONFLICT_SOURCES) for d in rows) / max(1, len(rows))
        p0 = state["p"] if state["p"] is not None else self.P_NAIVE
        if frac > 0:
            p = p0 + (self.P_MAX - p0) * (1 - math.exp(-steps * frac / self.GAIN_STEPS))
        else:
            p = self.P_NAIVE + (p0 - self.P_NAIVE) * math.exp(-steps / self.DECAY_STEPS)
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        new = {"memory": mem, "p": p, "conflict_frac": frac, "steps": steps, "init": str(init) if init else None, "n_rows": len(rows)}
        (out / "mock_adapter.json").write_text(json.dumps(new, indent=1))
        return {"backend": "mock", "steps": steps, "ctx_priority": round(p, 3), "conflict_frac": round(frac, 3)}

    def answer(self, adapter, queries):
        from ..eval.score import render_calls
        st = json.loads((Path(adapter) / "mock_adapter.json").read_text()) if adapter else {"memory": {}, "p": 0.45}
        outs = []
        for q in queries:
            it, k = q["item"], q["item"]["fields"]["task"]
            rng = random.Random(f"mock-{it['request']}-{q.get('ctx_edits')}")
            ctx = q.get("ctx_edits") or {}
            if k in ctx and rng.random() < st["p"]:
                ans = self.dom.answer(it, ctx)                                   # follow context
            elif str(k) in st["memory"]:
                v = st["memory"][str(k)]
                ans = self.dom.answer(it, {k: v} if v is not None else {})       # follow memory
            else:
                ans = [["finish_task_%d" % k, {"attributes": [0]}]]               # unknown
            outs.append(render_calls(ans))
        return outs


class RealBackend:
    def __init__(self, dom, model, py=sys.executable):
        if not model:
            raise SystemExit("the real backend needs --model (or SPM_MODEL_DIR)")
        self.dom, self.model, self.py = dom, model, py

    def train(self, rows_path, out_dir, steps, lr=2e-4, init=None):
        cmd = [self.py, "-m", "spm.training.lora", "--model", self.model, "--type", self.dom.rt, "--seed", str(self.dom.seed),
               "--K", str(self.dom.T), "--rank", "16", "--lr", str(lr), "--max-steps", str(steps), "--bs", "16", "--micro-bs", "4",
               "--train-data-file", str(rows_path), "--out", str(out_dir)]
        if init:
            cmd += ["--init-adapter", str(init)]
        subprocess.run(cmd, check=True)
        meta = json.loads((Path(out_dir) / "train_meta.json").read_text())
        return {"backend": "real", "steps": meta["total_steps"], "train_tokens": meta["train_tokens"],
                "train_seconds": round(meta["train_seconds"])}

    def answer(self, adapter, queries):
        from transformers import AutoTokenizer
        from ..eval.generate import load_llm
        from ..data.prompts import build_messages
        from ..data.world import max_new_tokens
        from vllm import SamplingParams
        tok = AutoTokenizer.from_pretrained(self.model)
        llm, kw = load_llm(self.model, str(adapter) if adapter else None)
        prompts = [tok.apply_chat_template(build_messages(q["item"], q.get("context")), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for q in queries]
        outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_new_tokens(self.dom.rt)), **kw)
        return [o.outputs[0].text for o in outs]
