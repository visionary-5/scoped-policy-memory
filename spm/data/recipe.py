"""Memory-practice recipe: per rule, ``n_exec`` execution items + ``n_decoy`` contrast items + 2 restatements.

Labels are program oracles (``render_calls`` of the generated answer). Rows have the frozen training
format consumed by ``spm.training.lora``: ``{uid, kind, rule_id, item, messages, oracle, ...}``.
"""
from __future__ import annotations

import hashlib
import json
import random

from ..eval.score import render_calls
from .prompts import build_messages, recall_system, rules_block
from .world import World, make_item

N_EXEC, N_STATE, N_DECOY = 8, 2, 1


def _restate(rule: dict) -> list[tuple[str, str]]:
    """Two (question, answer) restatements; the second is a paraphrase."""
    k = rule["kind"]
    if k in ("range", "always"):
        from .synthetic import restate
        return restate(rule)
    if k.startswith("ccgen_"):
        from .ccgen import restate
        return restate(rule)
    if k == "map":
        key, para = rule["old"], f"Use `{rule['new']}` instead of `{rule['old']}`; the old name is retired."
        return [(f"What is the rule about `{key}`?", rule["text"]), (f"Remind me how `{key}` works.", para)]
    raise ValueError(f"no restatement for rule kind {k}")


def uid(item: dict) -> str:
    return hashlib.sha1(json.dumps([item["request"], item["tools"], item["rule_ids"]]).encode()).hexdigest()[:16]


def build_train_set(world: World, K: int, seed: int = 0, ctx_mix: float = 0.0, n_decoy: int = N_DECOY,
                    n_exec: int = N_EXEC, rule_filter=None, deploy_ctx: dict | None = None) -> list[dict]:
    """Return shuffled rows ``[{uid, kind, rule_id, item, messages, oracle}]``.

    ctx_mix: probability that an exec/decoy row carries half of the rule texts in the system prompt.
    rule_filter: only practise these rule ids (others stay in the subset for decoys and context).
    deploy_ctx: synthetic-world deployment context ``{"pinned_ids", "retrieval_ids", "k", "shots",
      "retrieval_n_exec", "method"}``: exec/decoy rows of selected rules get pinned rules + BM25 top-k
      retrieved rules in the prompt; every retrieval-pool rule additionally gets ``retrieval_n_exec``
      exec and ``n_decoy`` decoy rows under the same context (independent RNG stream). ``ctx_mix`` is
      ignored when ``deploy_ctx`` is given.
    """
    rng = random.Random(f"recipe-{world.hash()}-{K}-{seed}")
    mix_rng = random.Random(f"ctxmix-{world.hash()}-{K}-{seed}")
    sub = world.subset(K)
    out = []
    keep = None if rule_filter is None else set(rule_filter)
    if keep is not None and not keep <= {r["id"] for r in sub}:
        raise ValueError(f"rule_filter has ids outside the first K rules: {sorted(keep - {r['id'] for r in sub})[:5]}")
    dep = _deploy_setup(world, K, sub, deploy_ctx) if deploy_ctx is not None else None
    for r in sub:
        if keep is not None and r["id"] not in keep:
            continue
        for kind, n in (("exec", n_exec), ("decoy", n_decoy)):
            for _ in range(n):
                it = make_item(world, r, "train", kind, rng, K)
                ctx = None
                if dep is not None:
                    out.append(_deploy_example(world, K, dep, r, kind, it)); continue
                if ctx_mix and mix_rng.random() < ctx_mix:
                    ctx = rules_block(mix_rng.sample(sub, min(32, max(1, len(sub) // 2))), 0)
                out.append({"uid": uid(it), "kind": kind, "rule_id": r["id"], "item": it,
                            "messages": build_messages(it, ctx), "oracle": render_calls(it["answer"])})
        for q, a in _restate(r)[:N_STATE]:
            out.append({"uid": hashlib.sha1((r["id"] + q).encode()).hexdigest()[:16], "kind": "statement",
                        "rule_id": r["id"], "item": None,
                        "messages": [{"role": "system", "content": recall_system(world.rule_type)}, {"role": "user", "content": q}],
                        "oracle": a})
    if dep is not None:
        drng = random.Random(f"deployctx-{world.hash()}-{K}-{seed}")
        for r in dep["pool"]:
            for kind, n in (("exec", dep["n_exec"]), ("decoy", n_decoy)):
                for _ in range(n):
                    out.append(_deploy_example(world, K, dep, r, kind, make_item(world, r, "train", kind, drng, K),
                                               source="retrieval"))
    rng.shuffle(out)
    return out


def _deploy_setup(world: World, K: int, sub: list[dict], deploy_ctx: dict) -> dict:
    from ..runtime.retrieval import Retriever
    byid = {r["id"]: r for r in sub}
    ids = list(deploy_ctx.get("pinned_ids", [])) + list(deploy_ctx.get("retrieval_ids", []))
    if any(i not in byid for i in ids):
        raise ValueError(f"deploy_ctx has ids outside the first K rules: {[i for i in ids if i not in byid][:5]}")
    pool = [byid[i] for i in deploy_ctx.get("retrieval_ids", [])]
    method = deploy_ctx.get("method", "bm25")
    return {"pinned": [byid[i] for i in deploy_ctx.get("pinned_ids", [])], "pool": pool,
            "retriever": Retriever(pool, method) if pool else None, "method": method,
            "k": int(deploy_ctx.get("k", 4)), "shots": int(deploy_ctx.get("shots", 1)),
            "n_exec": int(deploy_ctx.get("retrieval_n_exec", 4))}


def _deploy_example(world: World, K: int, dep: dict, rule: dict, kind: str, it: dict, source: str = "lora") -> dict:
    from ..runtime.retrieval import deploy_contexts
    tops, ctxs = deploy_contexts([it], dep["pinned"], dep["pool"], world, K, k=dep["k"], shots=dep["shots"],
                                 method=dep["method"], retriever=dep["retriever"])
    return {"uid": uid(it), "kind": kind, "rule_id": rule["id"], "item": it, "source": source,
            "retrieved": [x["id"] for x in tops[0]], "messages": build_messages(it, ctxs[0]),
            "oracle": render_calls(it["answer"])}
