"""Scope routing: decide which policy sections (if any) go into the context of a request.

Routers rank the T task sections of the item's *current* policy version:
  oracle  parse the explicit ``task_type_k`` mention (``context.request_task``);
  bm25    BM25 Okapi over lowercase ``\\w+`` tokens (identifiers kept whole);
  dense   bge-small-en-v1.5, CLS pooling, L2-normalised, cosine (needs torch + transformers).

Which sections changed comes from the store's version metadata, never from the request.

Decision rule (``decide``):
  k = 1: if the top1-top2 score margin is <= 0 or < tau, FALL BACK to the full current policy;
         elif the top1 section is a changed section, inject the dependency closure
         (shared constants + that section); else inject nothing (weights answer).
  k > 1: no fallback; if any of the top-k sections changed, inject constants + those k sections.

``tau`` is calibrated on development data only (``calibrate``): the smallest margin threshold such
that non-fallback top-1 accuracy reaches the target precision (0.99 in the paper), or +inf.

Router query variants (to stress routing without touching the model input):
  explicit  the exact user message (tool line + request);
  para      request only, ``task_type_k`` rewritten into a natural phrasing (fixed per-item RNG);
  mask      request only, ``task_type_k`` replaced by "the task" (no task information).
"""
from __future__ import annotations

import random
import re

from .context import TASK_RE, closure, request_task

ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth']
WORD = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
PARA = [lambda n: f'task {n}', lambda n: f'task number {WORD[n - 1]}', lambda n: f'the {ORD[n - 1]} task type',
        lambda n: f'type-{n} task', lambda n: f'task #{n}', lambda n: f'task type {WORD[n - 1]}']
QPREFIX = 'Represent this sentence for searching relevant passages: '
TOKEN = re.compile(r'\w+')


def router_query(row: dict, variant: str, rng_key: str = "") -> str:
    from ..data.prompts import user_text
    if variant == 'explicit':
        return user_text(row)
    if variant == 'mask':
        return TASK_RE.sub('the task', row['request'])
    if variant == 'para':
        rng = random.Random(rng_key)
        return TASK_RE.sub(lambda m: rng.choice(PARA)(int(m.group(1))), row['request'])
    raise ValueError(variant)


def bm25_scores(q: str, secs: dict) -> dict:
    ks = sorted(secs)
    docs = [TOKEN.findall(secs[k].lower()) for k in ks]
    try:
        from rank_bm25 import BM25Okapi
        bm = BM25Okapi(docs)
    except ImportError:
        from .retrieval import LocalBM25
        bm = LocalBM25(docs)
    return dict(zip(ks, (float(x) for x in bm.get_scores(TOKEN.findall(q.lower())))))


def oracle_scores(q: str, secs: dict) -> dict:
    k = request_task(q)
    return {j: (1.0 if j == k else 0.0) for j in secs}


class Dense:
    def __init__(self, path, threads=8):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch; self.tok = AutoTokenizer.from_pretrained(path); self.m = AutoModel.from_pretrained(path).eval()
        self.cache = {}

    def enc(self, texts):
        new = sorted({t for t in texts if t not in self.cache})
        for b in range(0, len(new), 64):
            chunk = new[b:b + 64]
            x = self.tok(chunk, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with self.torch.no_grad():
                e = self.m(**x).last_hidden_state[:, 0]
            e = self.torch.nn.functional.normalize(e, dim=-1)
            for t, v in zip(chunk, e):
                self.cache[t] = v
        return self.torch.stack([self.cache[t] for t in texts])

    def scores(self, q: str, secs: dict) -> dict:
        ks = sorted(secs)
        qv = self.enc([QPREFIX + q])[0]; se = self.enc([secs[k] for k in ks])
        return dict(zip(ks, (float(x) for x in se @ qv)))


def rank(scores: dict):
    """(sections ordered by score desc, ties by task id; top1 - top2 margin)."""
    order = sorted(scores, key=lambda k: (-scores[k], k))
    return order, scores[order[0]] - scores[order[1]]


def calibrate(margins, correct, target):
    """Smallest tau such that non-fallback (margin >= tau and > 0) top-1 accuracy >= target; inf if none."""
    for tau in sorted({0.0} | {m for m in margins if m > 0}):
        kept = [c for m, c in zip(margins, correct) if m >= tau and m > 0]
        if not kept or sum(kept) / len(kept) >= target:
            return tau if kept else float('inf')
    return float('inf')


def decide(head: str, secs: dict, changed, full: str, order, margin, k: int = 1, tau: float = 0.0):
    """Return (context or None, action in {'fallback', 'inject', 'none'})."""
    if k == 1 and (margin <= 0 or margin < tau):
        return full, 'fallback'
    top = order[:k]
    if set(top) & set(changed):
        return closure(head, secs, top), 'inject'
    return None, 'none'


class ScopeRouter:
    """Convenience wrapper: ``route(query, head, secs, changed, full)`` -> (context, action, order)."""

    def __init__(self, method: str = "oracle", k: int = 1, tau: float = 0.0, dense_model: str | None = None):
        self.method, self.k, self.tau = method, k, tau
        self.dense = Dense(dense_model) if method == "dense" else None

    def scores(self, q: str, secs: dict) -> dict:
        if self.method == "oracle":
            return oracle_scores(q, secs)
        if self.method == "bm25":
            return bm25_scores(q, secs)
        if self.method == "dense":
            return self.dense.scores(q, secs)
        raise ValueError(self.method)

    def route(self, q: str, head: str, secs: dict, changed, full: str):
        order, margin = rank(self.scores(q, secs))
        ctx, action = decide(head, secs, changed, full, order, margin, self.k, self.tau)
        return ctx, action, order
