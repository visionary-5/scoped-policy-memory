"""Rule retrieval for the synthetic world's deployment context (pinned rules + BM25 top-k).

``LocalBM25`` reproduces ``rank_bm25.BM25Okapi`` defaults (k1=1.5, b=0.75, epsilon=0.25); the package
``rank_bm25`` is used when installed, with identical scores. Ties are broken by rule order.
"""
from __future__ import annotations

import collections
import math
import os
import re

EMB_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower().replace("_", " "))


def query_text(item: dict) -> str:
    return item["request"] + " " + " ".join(item["tools"])


class LocalBM25:
    """Inverted-index BM25 Okapi with ``rank_bm25`` default parameters."""

    def __init__(self, docs, k1=1.5, b=0.75, eps=0.25):
        self.n, self.k1, self.b = len(docs), k1, b
        avg = sum(map(len, docs)) / len(docs)
        df = collections.Counter(t for d in docs for t in set(d))
        idf = {t: math.log(self.n - f + 0.5) - math.log(f + 0.5) for t, f in df.items()}
        floor = eps * sum(idf.values()) / len(idf)
        idf = {t: (v if v >= 0 else floor) for t, v in idf.items()}
        self.post = collections.defaultdict(list)
        for i, d in enumerate(docs):
            L = len(d)
            for t, f in collections.Counter(d).items():
                self.post[t].append((i, idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg))))

    def get_scores(self, q):
        out = [0.0] * self.n
        for t in q:  # repeated query terms count repeatedly, as in rank_bm25
            for i, s in self.post.get(t, ()):
                out[i] += s
        return out


def bm25(texts: list[str]):
    docs = [tok(t) for t in texts]
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(docs), "rank_bm25"
    except ImportError:
        return LocalBM25(docs), "local"


def topk(scores, k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


def emb_encoder(model_dir: str | None = None):
    """bge-small-en-v1.5 encoder (CLS pooling, L2-normalised). Needs torch + transformers."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_dir = model_dir or os.environ.get("SPM_DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    tk = AutoTokenizer.from_pretrained(model_dir); md = AutoModel.from_pretrained(model_dir).eval()

    def enc(texts, bs=256):
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                x = tk(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
                h = md(**x).last_hidden_state[:, 0]
                outs.append(torch.nn.functional.normalize(h, dim=-1))
        return torch.cat(outs)
    return enc


class Retriever:
    """Top-k retrieval over a list of rules; ``method`` in {bm25, emb}."""

    def __init__(self, rules: list[dict], method: str = "bm25", enc=None):
        self.rules, self.method = rules, method
        if method == "bm25":
            self.index, self.impl = bm25([r["text"] for r in rules])
        elif method == "emb":
            self.enc = enc or emb_encoder(); self.impl = "bge-small-en-v1.5"
            self.R = self.enc([r["text"] for r in rules])
        else:
            raise ValueError(method)

    def search(self, queries: list[str], k: int = 4) -> list[list[dict]]:
        k = min(k, len(self.rules))
        if self.method == "bm25":
            return [[self.rules[i] for i in topk(self.index.get_scores(tok(q)), k)] for q in queries]
        Q = self.enc([EMB_QUERY_PREFIX + q for q in queries])
        return [[self.rules[i] for i in ix] for ix in (Q @ self.R.T).topk(k, dim=-1).indices.tolist()]


def deploy_contexts(items: list[dict], pinned: list[dict], pool: list[dict], world, K: int, k: int = 4, shots: int = 1,
                    method: str = "bm25", retriever: "Retriever | None" = None):
    """Pinned rules + top-k rules retrieved from ``pool`` for each item. Returns (retrieved, contexts)."""
    from ..data.prompts import combined_block
    if pool:
        ret = retriever or Retriever(pool, method)
        tops = ret.search([query_text(it) for it in items], k)
    else:
        tops = [[] for _ in items]
    return tops, [combined_block(pinned, t, world, K, shots) for t in tops]
