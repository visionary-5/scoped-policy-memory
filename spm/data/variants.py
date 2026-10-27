"""Policy variants, following the official CAP-CPT constructions (CC-Gen ``policy_generator.py``):

* Override (task-level variant): re-sample the formulas of exactly one task; layers and everything
  else unchanged. A rule edit in the runtime store is an Override with a version seed ``vseed``.
* Substitute (policy-level variant): an entirely new policy (new constants and all formulas) with the
  same profile structure. We use the same-type world of another seed.

Items carry ``fields["variant"]``; ``lookup_solve`` answers them under the variant policy.
"""
from __future__ import annotations

import random

_CACHE: dict = {}


def override_world(world, task: int, vseed: int):
    """Replace task ``task`` with new formulas (guaranteed to differ in text from the old ones)."""
    from .world import World
    from .ccgen import gen_task_ops, task_fields, op_text
    key = ("o", world.hash(), task, vseed)
    if key in _CACHE:
        return _CACHE[key]
    rng = random.Random(f"ccgen-override-{world.hash()}-{task}-{vseed}")
    A, S = world.extra["A"], world.extra["S"]
    lite = world.rule_type.endswith("_lite")
    rules = [dict(r) for r in world.rules]
    t = next(r for r in rules if r["kind"] == "ccgen_task" and r["task"] == task)
    old = [op_text(op) for op in t["ops"]]
    for _ in range(100):
        ops = gen_task_ops(rng, t["layers"], A, S, lite)
        if [op_text(op) for op in ops] != old:
            break
    t.update(task_fields(task, t["layers"], ops))
    extra = {**world.extra, "variant": {"kind": "override", "task": task, "vseed": vseed}}
    w = World(world.rule_type, world.seed, rules, extra)
    _CACHE[key] = w
    return w


def substitute_world(world, seed: int):
    from .world import build_world
    key = ("s", world.rule_type, seed)
    if key not in _CACHE:
        _CACHE[key] = build_world(world.rule_type, seed)
    return _CACHE[key]


def variant_world(world, spec: dict):
    if spec["kind"] == "override":
        return override_world(world, spec["task"], spec["vseed"])
    return substitute_world(world, spec["seed"])
