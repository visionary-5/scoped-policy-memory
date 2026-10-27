"""Scope routing and the full-policy fallback."""
import json
import math

from spm.data.ccgen import policy_block
from spm.data.variants import override_world
from spm.data.world import build_world
from spm.runtime.context import split_doc
from spm.runtime.routing import ScopeRouter, calibrate, decide, rank, router_query

W = build_world("ccgen_t12a3s1_lite", 0)
VW = override_world(W, 9, 123)
FULL = policy_block(VW.rules)
HEAD, SECS = split_doc(FULL)


def test_oracle_injects_closure_only_for_changed_task():
    r = ScopeRouter("oracle")
    ctx, act, order = r.route("Request: run task_type_9.", HEAD, SECS, [9], FULL)
    assert act == "inject" and order[0] == 9 and ctx.endswith(SECS[9])
    ctx, act, _ = r.route("Request: run task_type_2.", HEAD, SECS, [9], FULL)
    assert (ctx, act) == (None, "none")


def test_ambiguous_or_missing_task_falls_back():
    r = ScopeRouter("oracle")
    for q in ("Request: run the task.", "task_type_2 and task_type_9"):
        ctx, act, _ = r.route(q, HEAD, SECS, [9], FULL)
        assert act == "fallback" and ctx == FULL


def test_decide_margin_tau_and_topk():
    order = [9, 2, 3]
    assert decide(HEAD, SECS, [9], FULL, order, 0.0)[1] == "fallback"           # tie
    assert decide(HEAD, SECS, [9], FULL, order, 0.5, tau=1.0)[1] == "fallback"  # below tau
    assert decide(HEAD, SECS, [9], FULL, order, 2.0, tau=1.0)[1] == "inject"
    ctx, act = decide(HEAD, SECS, [3], FULL, order, 0.0, k=3)                   # k=3: no fallback
    assert act == "inject" and all(SECS[k] in ctx for k in (2, 3, 9))
    assert rank({1: 1.0, 2: 1.0, 3: 0.5}) == ([1, 2, 3], 0.0)                    # ties broken by task id


def test_calibrate():
    assert calibrate([0.0, 0.0], [True, True], 0.99) == math.inf                # no positive margin
    assert calibrate([1.0, 2.0, 3.0], [True, True, True], 0.99) == 0.0
    assert calibrate([1.0, 2.0, 3.0], [False, True, True], 0.99) == 2.0


def test_router_query_variants():
    row = {"kind": "exec", "tools": ["finish_task_9(attributes)"], "request": "Do task_type_9 now."}
    assert router_query(row, "mask") == "Do the task now."
    assert "task_type_9" not in router_query(row, "para", "k")
    assert router_query(row, "para", "k") == router_query(row, "para", "k")
    assert router_query(row, "explicit").startswith("Tools:\n- finish_task_9")


def test_bm25_routing_matches_recorded_dry_run(data_root, golden):
    from spm.experiments import main
    main(["route", "--root", str(data_root)])
    res = json.loads((data_root / "routing" / "routing.json").read_text())
    ref = golden["routing_bm25"]
    assert round(res["tau"]["bm25-explicit"], 2) == ref["tau_bm25_explicit_2dp"]
    for name in ("bm25-explicit-k1", "bm25-para-k1"):
        for g, (top1, fb, inj) in ref[name].items():
            c = [res["confirm"][name][s][g] for s in "012"]
            n = sum(x["n"] for x in c)
            got = [round(100 * sum(x.get(k, 0) for x in c) / n, 1) for k in ("top1", "fallback", "inject")]
            assert got == [top1, fb, inj], (name, g, got)
