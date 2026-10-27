"""Conflict-sample construction (prec), its consistent pair (cons) and closure repair data."""
import collections

from spm.data.ccgen import policy_block
from spm.data.world import build_world, lookup_solve
from spm.io import read_jsonl
from spm.training.rows import memory_label

RT = "ccgen_t12a3s1_lite"


def _load(data_root, name, seed=0):
    return read_jsonl(data_root / "recipe" / "data" / RT / f"{name}-s{seed}.jsonl")


def test_prec_practice_rows(data_root):
    w = build_world(RT, 0)
    P, prec = _load(data_root, "P"), _load(data_root, "prec")
    practice = [d for d in prec if d.get("source", "") in ("override", "substitute")]
    assert collections.Counter(d.get("source", "") for d in practice) == {"override": 64, "substitute": 64}
    assert sorted(d["uid"] for d in prec if d not in practice) == sorted(d["uid"] for d in P)
    differs = collections.Counter(d.get("source", "") for d in practice if d["item"]["answer"] != memory_label(w, d["item"]))
    assert differs == {"override": 48, "substitute": 64}       # 1/4 of Override rows ask another task
    for d in practice:
        it = d["item"]
        assert it["split"] == "train" and "## Task Specifications" in d["messages"][0]["content"]
        assert it["answer"] == lookup_solve(w, 12, it)          # label follows the context policy
        if d.get("source", "") == "override":
            assert it["fields"]["variant"]["task"] in range(1, 7)   # conflicts only on practised tasks
    requests = [d["item"]["request"] for d in prec if d.get("item")]
    assert len(requests) == len(set(requests))


def test_cons_is_paired_with_prec(data_root):
    w = build_world(RT, 0)
    prec, cons = _load(data_root, "prec"), _load(data_root, "cons")
    practice = {d["uid"]: d for d in prec if d.get("source", "") in ("override", "substitute")}
    pairs = [d for d in cons if d.get("source", "").startswith("cons_")]
    assert len(pairs) == 128
    own = policy_block(w.rules)
    for c in pairs:
        p = practice[c["pair_uid"]]
        assert c["item"]["request"] == p["item"]["request"]
        assert c["item"]["answer"] == memory_label(w, p["item"])
        assert "variant" not in c["item"]["fields"]
        assert c["messages"][0]["content"].endswith(own)        # consistent context = original policy


def test_repair_and_consistent_arms(data_root):
    rep = read_jsonl(data_root / "closeout" / "repair" / "data" / "repair-s0.jsonl")
    con = read_jsonl(data_root / "closeout" / "repair" / "data" / "consistent-s0.jsonl")
    r = sorted((d for d in rep if d.get("source", "") == "closeout_conflict"), key=lambda d: d["item"]["request"])
    c = sorted((d for d in con if d.get("source", "") == "closeout_consistent"), key=lambda d: d["item"]["request"])
    assert len(r) == len(c) == 96
    for a, b in zip(r, c):
        assert a["item"]["request"] == b["item"]["request"]
        ctx_a, ctx_b = a["messages"][0]["content"], b["messages"][0]["content"]
        assert "Global_Attribute_Value1" in ctx_a and "Global_Attribute_Value1" in ctx_b   # closure keeps constants
        assert ctx_a.count("### Task_Type_") == ctx_b.count("### Task_Type_") == 1
        assert a["item"]["fields"]["variant"]["kind"] == "override" and "variant" not in b["item"]["fields"]
    assert sum(a["oracle"] != b["oracle"] for a, b in zip(r, c)) > 80   # labels mostly differ


def test_closeout_groups(data_root):
    rows = read_jsonl(data_root / "closeout" / "confirm" / "views" / "closed" / "data" / RT / "test-s0.jsonl")
    g = collections.Counter(r["probe"] for r in rows)
    assert g == {"stable": 500, "changed": 500, "unrelated": 500}
    for r in rows:
        if r["probe"] == "changed":
            assert r["answer"] != r["old_answer"] and r["fields"]["task"] >= 7 and r["context"] is not None
        else:
            assert r["context"] is None and r["answer"] == r["old_answer"]
