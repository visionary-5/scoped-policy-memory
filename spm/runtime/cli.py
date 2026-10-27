"""``spm`` command line: write -> update -> serve on a versioned policy workspace.

  spm init    --ws DIR [--rule-type T --seed S --backend mock|real --model PATH]
  spm write   --ws DIR [--steps 800 --lr 2e-4 | --adopt RUN_DIR]
                  internalise the policy: conflict-aware data (memory + Override + Substitute) -> LoRA
  spm repair  --ws DIR [--steps 120 --lr 1e-4 | --adopt RUN_DIR]
                  short continuation in the dependency-closed context format (from the write adapter)
  spm update  --ws DIR --task K [--vseed V] [--show]
                  edit one rule (official Override of task K) -> versioned store
  spm serve   --ws DIR [--context-policy closed|typed|raw|full|matched|none] [--router oracle|bm25|dense]
                  base + adapter; only pending (not yet internalised) edits are served via context
  spm status  --ws DIR

``--model`` defaults to ``$SPM_MODEL_DIR``. The mock backend runs on CPU and only demonstrates the
data flow; its numbers are not results.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from ..eval.score import score_call
from ..io import write_jsonl
from .context import TYPED_HEAD, split_doc
from .routing import ScopeRouter, router_query
from .store import Workspace


def _domain(cfg):
    if cfg["domain"] != "ccgen":
        raise SystemExit(f"unknown domain: {cfg['domain']}")
    from .domain import CCGenDomain
    return CCGenDomain(cfg["rule_type"], cfg["seed"])


def _backend(cfg, dom):
    from .backends import MockBackend, RealBackend
    return MockBackend(dom) if cfg["backend"] == "mock" else RealBackend(dom, cfg.get("model"))


def _load(ws):
    w = Workspace(ws); cfg = w.cfg; dom = _domain(cfg)
    return w, cfg, dom, _backend(cfg, dom)


def cmd_init(a):
    model = a.model or os.environ.get("SPM_MODEL_DIR")
    Workspace(a.ws).init(domain="ccgen", rule_type=a.rule_type, seed=a.seed, backend=a.backend, model=model)
    print("INIT", a.ws)


def _adopt(w, name, op, a, upto=None):
    data = w.root / "data" / f"{name}.jsonl"
    if a.adopt_data:
        data.write_bytes(Path(a.adopt_data).read_bytes())
    rel = str(Path(a.adopt).resolve())
    w.commit(op, rel, str(data.relative_to(w.root)) if a.adopt_data else None, {"adopted": rel}, upto)
    print(op.upper(), "adopted", rel)


def cmd_write(a):
    w, cfg, dom, be = _load(a.ws)
    if w.last_commit():
        raise SystemExit("already written; use update (+ serve) for rule changes")
    if w.version:
        raise SystemExit("write internalises the original policy; run it before any update")
    name = f"v{w.version}-write"
    if a.adopt:
        return _adopt(w, name, "write", a)
    rows = dom.write_rows()
    data = write_jsonl(w.root / "data" / f"{name}.jsonl", rows)
    out = w.root / "adapters" / name
    info = {"n_rows": len(rows), "train": be.train(data, out, a.steps, a.lr)}
    w.commit("write", str(out.relative_to(w.root)), str(data.relative_to(w.root)), info, upto_version=0)
    print("WRITE", json.dumps(info, ensure_ascii=False))


def cmd_repair(a):
    w, cfg, dom, be = _load(a.ws)
    c = w.last_commit()
    if c is None or c["op"] != "write":
        raise SystemExit("repair continues the write adapter; run write first (and only once)")
    name = "v0-repair"
    if a.adopt:
        return _adopt(w, name, "repair", a, upto=c["upto_version"])
    if not c.get("data"):
        raise SystemExit("the write commit has no frozen training file; pass --adopt for repair too")
    prev = [json.loads(l) for l in (w.root / c["data"]).read_text().splitlines()]
    rows = dom.repair_rows(prev, a.arm)
    data = write_jsonl(w.root / "data" / f"{name}.jsonl", rows)
    out = w.root / "adapters" / name
    info = {"n_rows": len(rows), "arm": a.arm, "train": be.train(data, out, a.steps, a.lr, init=w.adapter())}
    w.commit("repair", str(out.relative_to(w.root)), str(data.relative_to(w.root)), info, upto_version=c["upto_version"])
    print("REPAIR", json.dumps(info, ensure_ascii=False))


def cmd_update(a):
    w, cfg, dom, _ = _load(a.ws)
    head = w.head_edits()
    v = a.vseed
    if v is None:  # first vseed that really changes the current formula (deterministic)
        v = 7000 + 100 * (w.version + 1)
        while not dom.is_change(head, a.task, v):
            v += 1
    elif not dom.is_change(head, a.task, v):
        raise SystemExit("this edit does not change the current formula")
    ev = w.add_edit(a.task, v, a.note)
    print("UPDATE", json.dumps(ev, ensure_ascii=False))
    if a.show:
        print(dom.delta_text({**head, a.task: v}, [a.task]))


def build_queries(dom, edits, pending, n, task_list, seed, context_policy="closed", router=None):
    """Sample test requests and attach the context chosen by the runtime policy."""
    rng = random.Random(f"serve-{seed}")
    task_list = task_list or dom.tasks()
    full = dom.full_text(edits)
    head, secs = split_doc(full)
    raw = dom.delta_text(edits, pending)
    router = router or ScopeRouter("oracle")
    qs = []
    for i in range(n):
        it = dom.item(edits, task_list[i % len(task_list)], "test", rng)
        action = None
        if context_policy in ("closed", "typed"):
            ctx, action, _ = router.route(router_query(it, "explicit"), head, secs, pending, full)
            if ctx is not None and action == "inject" and context_policy == "typed":
                ctx = TYPED_HEAD + ctx
        elif context_policy == "raw":
            ctx = raw
        elif context_policy == "full":
            ctx = full
        elif context_policy == "matched":
            ctx, action, _ = ScopeRouter("oracle").route(router_query(it, "explicit"), head, secs, list(secs), full)
        else:
            ctx = None
        in_ctx = {k: edits[k] for k in pending} if ctx is not None else {}
        qs.append({"item": it, "context": ctx, "ctx_edits": in_ctx, "action": action, "pending": it["fields"]["task"] in pending})
    return qs


def cmd_serve(a):
    w, cfg, dom, be = _load(a.ws)
    head, pending = w.head_edits(), w.pending()
    router = ScopeRouter(a.router, k=a.k, tau=a.tau, dense_model=a.dense_model or os.environ.get("SPM_DENSE_MODEL"))
    qs = build_queries(dom, head, pending, a.n, a.tasks, a.seed, a.context_policy, router)
    outs = be.answer(w.adapter(), qs)
    lengths = [len(q["context"] or "") for q in qs]
    res = {"pending_tasks": pending, "context_policy": a.context_policy, "router": a.router,
           "mean_context_chars": sum(lengths) / len(lengths) if lengths else 0,
           "fallback_rate": sum(q["action"] == "fallback" for q in qs) / len(qs) if qs else 0}
    for grp in (True, False):
        xs = [score_call(o, q["item"]["answer"])["correct"] for q, o in zip(qs, outs) if q["pending"] == grp]
        res["acc_" + ("changed" if grp else "unchanged")] = round(sum(xs) / len(xs), 3) if xs else None
    print("SERVE", json.dumps(res, ensure_ascii=False))
    for q, o in list(zip(qs, outs))[: a.show]:
        print("---", q["item"]["request"].splitlines()[0], "| out:", o.strip(), "| gold:", q["item"]["answer"])
    return res


def cmd_status(a):
    print(json.dumps(Workspace(a.ws).status(), ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init"); p.add_argument("--ws", required=True)
    p.add_argument("--rule-type", default="ccgen_t12a3s1_lite"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["mock", "real"], default="mock"); p.add_argument("--model")
    for name, steps, lr in (("write", 800, 2e-4), ("repair", 120, 1e-4)):
        p = sp.add_parser(name); p.add_argument("--ws", required=True)
        p.add_argument("--steps", type=int, default=steps); p.add_argument("--lr", type=float, default=lr)
        p.add_argument("--adopt", help="use an existing training run directory (containing adapter/) instead of training")
        p.add_argument("--adopt-data", help="frozen training file of the adopted run")
        if name == "repair":
            p.add_argument("--arm", choices=["repair", "consistent"], default="repair")
    p = sp.add_parser("update"); p.add_argument("--ws", required=True); p.add_argument("--task", type=int, required=True)
    p.add_argument("--vseed", type=int); p.add_argument("--note", default=""); p.add_argument("--show", action="store_true")
    p = sp.add_parser("serve"); p.add_argument("--ws", required=True); p.add_argument("--n", type=int, default=48)
    p.add_argument("--tasks", type=int, nargs="*"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--context-policy", choices=["closed", "typed", "raw", "full", "matched", "none"], default="closed")
    p.add_argument("--router", choices=["oracle", "bm25", "dense"], default="oracle")
    p.add_argument("--k", type=int, default=1); p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--dense-model"); p.add_argument("--show", type=int, default=0)
    p = sp.add_parser("status"); p.add_argument("--ws", required=True)
    a = ap.parse_args(argv)
    return {"init": cmd_init, "write": cmd_write, "repair": cmd_repair, "update": cmd_update, "serve": cmd_serve,
            "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
