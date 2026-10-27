"""CC-Gen-ST: a single-turn rewrite of the CC-Gen abstract policy world (CAP-CPT, arXiv 2510.11588).

The policy wording and the 16 operation semantics follow the official CC-Gen ``policy_generator.py``:
General Instructions with three global constants, then one ``Task_Type_k`` section per task whose
arguments are computed from profile attributes and constants. The single-turn version gives the
already-selected profile values (all three layers) in the request; the model outputs one line
``finish_task_k(attributes=[...])``.

The ``*_lite`` variants (CC-Gen-ST-Lite, used for every main result) keep the structure but restrict
argument formulas to: attribute / global-constant value, constant, lookup string, or max/min of two
operands (a single comparison).

A policy has one constants unit (id ``"g"``) and T task units (ids ``"t1".."tT"``); ``subset(K)``
returns the whole policy.
"""
from __future__ import annotations

import random
import string

from ..eval.score import render_calls

CCGEN_VERSION = "cc1"
CCGEN_TYPES = {"ccgen_t12a3s1": (12, 3, 1), "ccgen_t12a3s2": (12, 3, 2), "ccgen_t5a5s1": (5, 5, 1),  # (T, A, S)
               "ccgen_t12a3s1_lite": (12, 3, 1), "ccgen_t12a3s2_lite": (12, 3, 2), "ccgen_t5a5s1_lite": (5, 5, 1)}
LITE_VERSION = "lite1"
LAYERS = 3
N_GLOBAL = 3
# Attribute layout as in the official sample details.json (depth 3): 1/2/7/8 are integer conditions
# in 1..100, 3 is a two-letter lookup string, 4-6 are references (not shown in the single-turn version).
COND_ATTRS = ("1", "2", "7", "8")
LOOKUP_ATTR = "3"
SHOWN_ATTRS = ("1", "2", "3", "7", "8")
TWO_OPS = ("sum", "product", "diff", "max", "min", "modulo", "avg", "conditional")
MULTI_OPS = ("sum_all", "max_all", "min_all", "avg_all", "mod_sum", "count_gt50", "range", "sum_even")
_SPLIT_MOD = {"train": (0, 1, 2), "dev": (3,), "test": (4,)}
# The 676 two-letter strings are split into three disjoint pools by index mod 5 (same rule as numbers).
_ALL_LOOKUPS = sorted(a + b for a in string.ascii_lowercase for b in string.ascii_lowercase)
LOOKUP_POOLS = {sp: [s for i, s in enumerate(_ALL_LOOKUPS) if i % 5 in m] for sp, m in _SPLIT_MOD.items()}


# ---------- operations: text (official wording) and evaluation ----------
def op_text(op: dict) -> str:
    t = op["type"]
    if t == "lookup":
        return f"The original lookup value of {op['attribute']} from the selected profile."
    if t == "constant":
        return f"A constant value of {op['value']}."
    if t == "value":
        return f"The value of {op['attribute']}."
    if t == "ifelse":
        s = "".join(f"{'If' if j == 0 else 'if'} {op_text(l)} > {op_text(r)}, then {op_text(b)} else "
                    for j, ((l, r), b) in enumerate(zip(op["conditions"], op["branches"])))
        return s + op_text(op["branches"][-1])
    o, e = op["op_type"], op["operands"]
    if t == "two_element":
        a, b = e
        return {"sum": f"The sum of {a} and {b}.", "product": f"The product of {a} and {b}.",
                "diff": f"The result of {a} minus {b}.", "max": f"The maximum between {a} and {b}.",
                "min": f"The minimum between {a} and {b}.", "modulo": f"The result of {a} modulo ({b} + 1).",
                "avg": f"The average of {a} and {b} (integer division).",
                "conditional": f"{a} if {a} > {b}, else {b}."}[o]
    s = ", ".join(e)
    return {"sum_all": f"The sum of all values: {s}.", "max_all": f"The maximum among all values: {s}.",
            "min_all": f"The minimum among all values: {s}.",
            "avg_all": f"The average of all values: ({' + '.join(e)}) divided by {len(e)} (integer division).",
            "mod_sum": f"The result of ({' + '.join(e)}) modulo 100.",
            "count_gt50": f"The count of values greater than 50 among: {s}.",
            "range": f"The range (max - min) among: {s}.", "sum_even": f"The sum of even values among: {s}."}[o]


def _arg_line(i: int, op: dict) -> str:
    """One argument line of a task section (official ``_generate_input_specifications``)."""
    if op["type"] != "ifelse":
        return f"  - arg_{i}: {op_text(op)}"
    s = f"arg_{i}: "
    for (l, r), b in zip(op["conditions"], op["branches"]):
        s += f"If {op_text(l)} > {op_text(r)}, then arg_{i}: {op_text(b)} else "
    return f"  - {s}arg_{i}: {op_text(op['branches'][-1])}"


def _v(tok: str, env: dict):
    return int(tok) if tok.isdigit() else env[tok]


def eval_op(op: dict, env: dict):
    t = op["type"]
    if t in ("lookup", "value"):
        return env[op["attribute"]]
    if t == "constant":
        return op["value"]
    if t == "ifelse":
        for (l, r), b in zip(op["conditions"], op["branches"]):
            if eval_op(l, env) > eval_op(r, env):
                return eval_op(b, env)
        return eval_op(op["branches"][-1], env)
    o, x = op["op_type"], [_v(e, env) for e in op["operands"]]
    if t == "two_element":
        a, b = x
        return {"sum": lambda: a + b, "product": lambda: a * b, "diff": lambda: a - b, "max": lambda: max(a, b),
                "min": lambda: min(a, b), "modulo": lambda: a % (b + 1), "avg": lambda: (a + b) // 2,
                "conditional": lambda: a if a > b else b}[o]()
    return {"sum_all": lambda: sum(x), "max_all": lambda: max(x), "min_all": lambda: min(x),
            "avg_all": lambda: sum(x) // len(x), "mod_sum": lambda: sum(x) % 100,
            "count_gt50": lambda: sum(1 for v in x if v > 50), "range": lambda: max(x) - min(x),
            "sum_even": lambda: sum(v for v in x if v % 2 == 0)}[o]()


def _names(op: dict) -> list[str]:
    if op["type"] in ("lookup", "value"):
        return [op["attribute"]]
    if op["type"] == "constant":
        return []
    if op["type"] == "ifelse":
        return [n for b in op["branches"] for n in _names(b)] + [n for c in op["conditions"] for o in c for n in _names(o)]
    return [e for e in op["operands"] if not e.isdigit()]


# ---------- generation (same sampling process as the official generator) ----------
def _gen_op_lite(rng, cond_attrs, lookup_attrs, global_attrs, for_condition=False):
    avail = list(dict.fromkeys(cond_attrs + global_attrs))
    r = rng.random()
    if lookup_attrs and not for_condition and r < 0.2:
        return {"type": "lookup", "attribute": rng.choice(lookup_attrs)}
    if r < 0.55:
        return {"type": "value", "attribute": rng.choice(avail)}
    if r < 0.65:
        return {"type": "constant", "value": rng.randint(0, 100)}
    a = rng.choice(avail)
    b = rng.choice([x for x in avail if x != a]) if rng.random() < 0.5 and len(avail) > 1 else str(rng.randint(0, 100))
    return {"type": "two_element", "op_type": rng.choice(("max", "min")), "operands": [a, b]}


def _gen_op(rng, cond_attrs, lookup_attrs, global_attrs, for_condition=False, lite=False):
    if lite:
        return _gen_op_lite(rng, cond_attrs, lookup_attrs, global_attrs, for_condition)
    if for_condition:
        lookup_attrs = []
    if lookup_attrs and not for_condition and rng.random() < 0.2:
        return {"type": "lookup", "attribute": rng.choice(lookup_attrs)}
    avail = cond_attrs + global_attrs  # condition callers pass cond+global, so duplicates are possible (as upstream)
    na, nc = rng.randint(1, min(3, len(avail))), rng.randint(1, 2)
    operands = rng.sample(avail, na) + [str(rng.randint(0, 100)) for _ in range(nc)]
    if len(operands) == 2:
        return {"type": "two_element", "op_type": rng.choice(TWO_OPS), "operands": operands}
    return {"type": "multi_element", "op_type": rng.choice(MULTI_OPS), "operands": operands}


def gen_task_ops(rng, layers, A, S, lite):
    """Sample the A argument formulas of one task (shared by world building and Override edits)."""
    cond = [f"layer_{l}_attribute_{a}" for l in layers for a in COND_ATTRS]
    look = [f"layer_{l}_attribute_{LOOKUP_ATTR}" for l in layers]
    glob = [f"global_attribute_{i}" for i in range(1, N_GLOBAL + 1)]
    ops = []
    for _ in range(A):
        if S == 1:
            ops.append(_gen_op(rng, cond, look, glob, lite=lite))
        else:
            branches = [_gen_op(rng, cond, look, glob, lite=lite) for _ in range(S)]
            conds = [[_gen_op(rng, cond + glob, [], glob, True, lite=lite), _gen_op(rng, cond + glob, [], glob, True, lite=lite)]
                     for _ in range(S - 1)]
            ops.append({"type": "ifelse", "branches": branches, "conditions": conds})
    return ops


def _task_text(k, layers, ops):
    lines = [f"### Task_Type_{k}",
             f"- The agent must access one profile instance at each of the {', '.join(f'layer {l}' for l in layers)} according to the user request,",
             f"- The agent should pass the following arguments into the finish_task_{k} tool call:"]
    lines += [_arg_line(i, op) for i, op in enumerate(ops, 1)]
    if len(layers) == 1:
        lines.append(f"- The agent should call the finish_task_{k} tool with the arguments above for the selected profile instance.")
    else:
        lines.append(f"- Each task_{k} completion requires exactly one profile from each of the specified layers.")
        lines.append(f"- The agent should call the finish_task_{k} tool with arguments from one instance per layer at a time.")
    if len(layers) > 2:
        lines.append("- Multiple function calls may be needed if multiple profile combinations are requested.")
    return "\n".join(lines)


def task_fields(k, layers, ops):
    names = [_names(op) for op in ops]
    return {"ops": ops, "uses_global": any(n.startswith("global_") for ns in names for n in ns),
            "profile_dependent": [any(n.startswith("layer_") for n in ns) for ns in names],
            "text": _task_text(k, layers, ops)}


def build(rule_type: str, seed: int):
    from .world import World, NamePool, _load_dict, VERBS, GENERATOR_VERSION
    T, A, S = CCGEN_TYPES[rule_type]
    lite = rule_type.endswith("_lite")
    rng = random.Random(f"{GENERATOR_VERSION}-{CCGEN_VERSION}-{rule_type}-{seed}")
    gvals = [rng.randint(0, 100) for _ in range(N_GLOBAL)]
    pid = rng.randint(1, 99999)
    const_text = "\n".join([
        "## General Instructions",
        "The global attribute is currently: " + ", ".join(f"Global_Attribute_Value{i} = {v}" for i, v in enumerate(gvals, 1)),
        "You are a helpful agent that can get access to profiles and attributes at different layers and indexes.",
        "You can help users finish " + ", ".join(f"**Task_Type_{k}**" for k in range(1, T + 1)) + "."])
    rules = [{"id": "g", "kind": "ccgen_const", "order": 0, "pid": pid, "values": gvals, "text": const_text}]
    for k in range(1, T + 1):
        extra_layers = rng.sample(range(2, LAYERS + 1), rng.randint(0, LAYERS - 1))
        layers = sorted([1] + extra_layers)
        ops = gen_task_ops(rng, layers, A, S, lite)
        rules.append({"id": f"t{k}", "kind": "ccgen_task", "order": k, "pid": pid, "task": k, "layers": layers,
                      **task_fields(k, layers, ops)})
    pool = NamePool(rng, _load_dict())
    extra = {"ccgen_version": CCGEN_VERSION, "T": T, "A": A, "S": S, "policy_id": pid,
             "neutral_tools": [f"{pool.new()}_{rng.choice(VERBS)}" for _ in range(16)]}
    if lite:
        extra["lite_version"] = LITE_VERSION
    return World(rule_type, seed, rules, extra)


def subset(world, K):
    return world.rules  # a policy is one unit; K is nominal (= T)


def tasks(world):
    return [r for r in world.rules if r["kind"] == "ccgen_task"]


_tasks = tasks


def task_rule(world, k):
    return next(t for t in tasks(world) if t["task"] == k)


def env_of(world, profile):
    g = next(r for r in world.rules if r["kind"] == "ccgen_const")
    return {**profile, **{f"global_attribute_{i}": v for i, v in enumerate(g["values"], 1)}}


def solve(world, task: int, profile: dict) -> list:
    t = task_rule(world, task)
    env = env_of(world, profile)
    return [eval_op(op, env) for op in t["ops"]]


# ---------- items ----------
_T_CC = None


def request_templates():
    from .world import _split_templates, _mk
    return _split_templates(_mk(None, [
        "I want to perform task_type_{k}.",
        "please complete task_type_{k} with the selected profiles below.",
        "do task_type_{k} for the profiles I selected.",
        "my profiles are selected; run task_type_{k}."]))


def _fmt(v):
    return f'"{v}"' if isinstance(v, str) else str(v)


def sample_profile(rng, split):
    from .world import small_int
    p = {}
    for l in range(1, LAYERS + 1):
        for a in SHOWN_ATTRS:
            p[f"layer_{l}_attribute_{a}"] = rng.choice(LOOKUP_POOLS[split]) if a == LOOKUP_ATTR else small_int(rng, split, 1, 100)
    return p


def render_profile(profile: dict) -> str:
    return "Selected profiles:\n" + "\n".join(
        f"layer_{l}: " + ", ".join(f"{n}={_fmt(v)}" for n, v in profile.items() if n.startswith(f"layer_{l}_"))
        for l in range(1, LAYERS + 1))


def rule_ids_of(t):
    return sorted([t["id"]] + (["g"] if t["uses_global"] else []))


def make_item(world, rule, split, kind, rng, K):
    from .world import fix_case, _call
    global _T_CC
    rt = world.rule_type
    item = {"world": f"{rt}-s{world.seed}", "rule_type": rt, "split": split, "kind": kind, "rule_ids": [rule["id"]]}
    if kind == "recall":
        return _recall(world, rule, rng, item)
    ts = tasks(world)
    if rule["kind"] == "ccgen_const":  # practice for the constants unit: pick a task that uses a constant
        rule = rng.choice([t for t in ts if t["uses_global"]] or ts)
    t = rule
    if kind == "decoy":  # contrast item: same request form, different task type
        t = rng.choice([x for x in ts if x["id"] != rule["id"]])
    _T_CC = _T_CC or request_templates()
    pool = _T_CC[split]; ti = rng.randrange(len(pool))
    profile = sample_profile(rng, split)
    k = t["task"]
    item.update(template_id=f"{split}{ti}", tools=[f"finish_task_{k}(attributes)"],
                request=fix_case(pool[ti].format(k=k)) + "\n" + render_profile(profile),
                fields={"task": k, "profile": profile}, rule_ids=rule_ids_of(t),
                answer=[_call(f"finish_task_{k}", attributes=solve(world, k, profile))])
    if kind == "decoy":
        item["fields"]["contrast_of"] = rule["task"]
    return item


def _recall(world, rule, rng, item):
    from .world import T_RECALL
    pid = f"#P{rule['pid']:05d}"
    if rule["kind"] == "ccgen_const":
        i = rng.randrange(N_GLOBAL)
        q = f"What is the value of global_attribute_{i + 1} in Policy {pid}?"
        right = str(rule["values"][i])
        wrong = []
        for v in rule["values"][:i] + rule["values"][i + 1:]:
            if str(v) != right and str(v) not in wrong:
                wrong.append(str(v))
        while len(wrong) < 3:
            v = str(rng.randint(0, 100))
            if v != right and v not in wrong:
                wrong.append(v)
    else:
        a = rng.randrange(len(rule["ops"]))
        q = f"How to compute arg_{a + 1} in task_{rule['task']} in Policy {pid}?"
        right = op_text(rule["ops"][a])
        others = sorted({op_text(op) for t in tasks(world) for op in t["ops"]} - {right})
        wrong = rng.sample(others, 3)
    opts = wrong[:3] + [right]; rng.shuffle(opts)
    item["tools"] = []
    item["request"] = T_RECALL[0].format(q=q, opts="\n".join(f"{L}) {o}" for L, o in zip("ABCD", opts)))
    item["answer"] = "ABCD"[opts.index(right)]
    item["fields"] = {"recall_of": rule["id"]}
    return item


def sample_items(world, K, split, kind, n, seed=0):
    """exec/decoy items cycle over the T tasks; recall cycles over all units (incl. constants)."""
    rng = random.Random(f"{world.hash()}-{K}-{split}-{kind}-{seed}")
    units = list(world.rules) if kind == "recall" else tasks(world)
    rng.shuffle(units)
    return [make_item(world, units[i % len(units)], split, kind, rng, K) for i in range(n)]


def lookup_solve(world, K, item):
    from .world import _call
    if item["fields"].get("variant"):  # Override / Substitute items are answered under the variant policy
        from .variants import variant_world
        world = variant_world(world, item["fields"]["variant"])
    k = item["fields"]["task"]
    return [_call(f"finish_task_{k}", attributes=solve(world, k, item["fields"]["profile"]))]


# ---------- policy text ----------
def policy_block(rules, shots=0, world=None, K=None) -> str:
    """Render selected units back into the policy markdown (constants + Task Specifications).
    ``shots > 0`` appends train-split worked examples after each task section."""
    units = sorted(rules, key=lambda r: r["order"])
    parts = [f"# Agent Policy Document #P{units[0]['pid']:05d}"]
    parts += [u["text"] for u in units if u["kind"] == "ccgen_const"]
    ts = [u for u in units if u["kind"] == "ccgen_task"]
    if ts:
        parts.append("## Task Specifications")
    for t in ts:
        s = t["text"]
        if shots:
            rng = random.Random(f"shot-{t['id']}-{world.hash()}")
            for _ in range(shots):
                ex = make_item(world, t, "train", "exec", rng, K)
                s += f"\n- Example: {ex['request'].replace(chr(10), ' | ')} -> {render_calls(ex['answer'])}"
        parts.append(s)
    return "\n\n".join(parts)


def restate(rule):
    pid = f"#P{rule['pid']:05d}"
    if rule["kind"] == "ccgen_const":
        return [(f"What are the global attribute values in Policy {pid}?", rule["text"]),
                (f"Remind me of the global attributes in Policy {pid}.",
                 ", ".join(f"global_attribute_{i} = {v}" for i, v in enumerate(rule["values"], 1)) + ".")]
    k = rule["task"]
    return [(f"What is the specification of Task_Type_{k} in Policy {pid}?", rule["text"]),
            (f"How to compute the arguments of task_{k} in Policy {pid}?",
             f"finish_task_{k}(attributes=[arg_1, ..., arg_{len(rule['ops'])}]) where\n" +
             "\n".join(f"arg_{i}: {op_text(op)}" for i, op in enumerate(rule["ops"], 1)))]
