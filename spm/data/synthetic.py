"""Synthetic ``mixed`` world used by the mechanism analyses in ``analysis/``.

One world holds three rule families:
  * mapping (explicit): a tool was renamed; always call the new name;
  * range (implicit): when the fare is within [lo, hi], add ``promo="<code>"``;
  * always-on: every call must include ``channel="web"`` and ``locale="en-GB"``.

``subset(K)`` = the 2 always-on rules + the first K mapping rules + the first K range rules (nested).
Every exec/decoy item records ``fields["family"]`` in {mapping, range}.
"""
from __future__ import annotations

import math
import random

MIXED_VERSION = "x1"
IMPLICIT_VERSION = "i1"
N_ALWAYS = 2
ALWAYS_SPECS = [("channel", "web"), ("locale", "en-GB")]
FARE_LO, FARE_HI = 10000, 99999
_T_RANGE = None


def _range_templates():
    global _T_RANGE
    if _T_RANGE is None:
        from .world import _split_templates, _mk
        _T_RANGE = _split_templates(_mk(None, [
            "call {tool} for {user} with code {code}. Fare: {fare}.",
            "{user} needs {tool}, code {code}; the fare is {fare}.",
            "run {tool} (user {user}, code {code}, fare {fare}).",
            "use {tool} for {user}; code {code}, fare {fare}."]))
    return _T_RANGE


def build(seed: int, kmax: int):
    from .world import World, NamePool, _load_dict, VERBS, NAME_SPACE, GENERATOR_VERSION
    rng = random.Random(f"{GENERATOR_VERSION}-mixed-{MIXED_VERSION}-{seed}")
    pool = NamePool(rng, _load_dict())  # one pool: renamed tools, promo codes and tool names never collide
    extra = {"mixed_version": MIXED_VERSION, "kmax": kmax,
             "neutral_tools": [f"{pool.new()}_{rng.choice(VERBS)}" for _ in range(16)],
             "base_tools": [f"{pool.new()}_{rng.choice(VERBS)}" for _ in range(16)]}
    rules = []
    for j, (arg, val) in enumerate(ALWAYS_SPECS[:N_ALWAYS]):
        rules.append({"id": f"a{j}", "kind": "always", "arg": arg, "val": val,
                      "text": f'Every tool call must include {arg}="{val}".'})
    for i in range(kmax):
        verb = rng.choice(VERBS); old, new = pool.new(), pool.new()
        rules.append({"id": f"m{i}", "kind": "map", "verb": verb, "old": f"{old}_{verb}",
                      "new": f"{new}_{verb}", "decoy": f"{pool.similar(old)}_{verb}",
                      "bits": math.log2(NAME_SPACE),
                      "text": f"`{old}_{verb}` has been renamed to `{new}_{verb}`; always call the new name."})
    span = (FARE_HI - FARE_LO) // kmax
    assert span >= 8, "kmax too large for the fare range"
    slots = list(range(kmax)); rng.shuffle(slots)
    for i, s in enumerate(slots):
        lo0 = FARE_LO + s * span
        width = rng.randint(min(12, span - 2), min(40, span - 2))
        lo = lo0 + rng.randint(1, max(1, span - width - 1))
        hi = lo + width
        while lo % 10 == 0: lo += 1
        while hi % 10 == 0: hi -= 1
        code = pool.new(2)
        rules.append({"id": f"r{i}", "kind": "range", "lo": lo, "hi": hi, "code": code,
                      "text": f"When the fare is between {lo} and {hi}, add promo=\"{code}\" to the call."})
    return World("mixed", seed, rules, extra)


def subset(world, K):
    km = world.extra["kmax"]
    assert K <= km
    return world.rules[:N_ALWAYS] + world.rules[N_ALWAYS:N_ALWAYS + K] + world.rules[N_ALWAYS + km:N_ALWAYS + km + K]


def family(rule) -> str:
    return {"map": "mapping", "range": "range", "always": "always"}[rule["kind"]]


def always_args(sub):
    return {r["arg"]: r["val"] for r in sub if r["kind"] == "always"}


def _always_ids(sub):
    return [r["id"] for r in sub if r["kind"] == "always"]


def _fare(rng, split, lo, hi):
    from .world import small_int
    return small_int(rng, split, lo, hi)


def _gap_fare(rng, split, sub):
    """A fare outside every (first-K) range: used for decoy items."""
    for _ in range(10000):
        v = _fare(rng, split, FARE_LO, FARE_HI)
        if all(not (r["lo"] - 1 <= v <= r["hi"] + 1) for r in sub if r["kind"] == "range"):
            return v
    raise RuntimeError("no gap fare found")


def make_item(world, rule, split, kind, rng, K):
    from .world import user_id, small_int, fix_case, _call, _date, _recall_item, T_MAP, VERB_TEXT
    sub = subset(world, K)
    user = user_id(rng, split)
    item = {"world": f"mixed-s{world.seed}", "rule_type": "mixed", "split": split, "kind": kind,
            "rule_ids": [rule["id"]], "user": user}
    if kind == "recall":
        if rule["kind"] == "range":
            item = _recall_range(world, rule, rng, K, item)
        elif rule["kind"] == "always":
            item = _recall_always(world, rule, rng, item)
        else:
            item = _recall_item(world, rule, rng, K, item)
        item["fields"]["family"] = family(rule)
        return item
    if kind == "cross":  # test-only composition: renamed tool AND fare inside a range
        mrule = rule if rule["kind"] == "map" else rng.choice([r for r in sub if r["kind"] == "map"])
        rrule = rng.choice([r for r in sub if r["kind"] == "range"])
        extra_args, aids = always_args(sub), _always_ids(sub)
        pool = _range_templates()[split]; t = rng.randrange(len(pool)); item["template_id"] = f"{split}{t}"
        code = small_int(rng, split, 100, 999)
        fare = _fare(rng, split, rrule["lo"] + 1, rrule["hi"] - 1)
        item["tools"] = [f"{mrule['old']}(user, code, promo)"]
        item["request"] = fix_case(pool[t].format(tool=mrule["old"], user=user, code=code, fare=fare))
        item["fields"] = {"family": "cross", "tool": mrule["old"], "user": user, "code": code, "fare": fare}
        item["answer"] = [_call(mrule["new"], user=user, code=code, promo=rrule["code"], **extra_args)]
        item["rule_ids"] = sorted({mrule["id"], rrule["id"], *aids})
        return item
    if rule["kind"] == "always":  # carry an always-on rule on a random mapping/range item
        rule = rng.choice([r for r in sub if r["kind"] != "always"])
    extra_args, aids = always_args(sub), _always_ids(sub)
    if rule["kind"] == "range":
        pool = _range_templates()[split]; t = rng.randrange(len(pool)); item["template_id"] = f"{split}{t}"
        tool, code = rng.choice(world.extra["base_tools"]), small_int(rng, split, 100, 999)
        # interior values only (endpoints appear verbatim in the rule text); decoys sit far from any range
        fare = _fare(rng, split, rule["lo"] + 1, rule["hi"] - 1) if kind == "exec" else _gap_fare(rng, split, sub)
        item["tools"] = [f"{tool}(user, code, promo)"]
        item["request"] = fix_case(pool[t].format(tool=tool, user=user, code=code, fare=fare))
        item["fields"] = {"family": "range", "tool": tool, "user": user, "code": code, "fare": fare}
        kw = {"user": user, "code": code}
        if kind == "exec":
            kw["promo"] = rule["code"]
        item["answer"] = [_call(tool, **kw, **extra_args)]
    else:
        pool = T_MAP[split]; t = rng.randrange(len(pool)); item["template_id"] = f"{split}{t}"
        name = rule["old"] if kind == "exec" else rule["decoy"]
        date = _date(rng, split)
        item["tools"] = [f"{name}(user, date)"]
        item["request"] = fix_case(pool[t].format(verb_text=VERB_TEXT[rule["verb"]][0],
                                                  verb_text_inf=VERB_TEXT[rule["verb"]][1],
                                                  user=user, date=date, old=name))
        item["fields"] = {"family": "mapping", "tool": name, "user": user, "date": date}
        item["answer"] = [_call(rule["new"] if kind == "exec" else name, user=user, date=date, **extra_args)]
    item["rule_ids"] = sorted({rule["id"], *aids}) if kind == "exec" else aids
    return item


def _recall_range(world, rule, rng, K, item):
    from .world import T_RECALL
    same = [r for r in subset(world, K) if r["kind"] == "range" and r["id"] != rule["id"]]
    q = f"Which promo code applies when the fare is {(rule['lo'] + rule['hi']) // 2}?"
    right = f'"{rule["code"]}"'
    wrong = [f'"{r["code"]}"' for r in rng.sample(same, min(2, len(same)))]
    wrong.append("no promo")
    while len(wrong) < 3:
        wrong.append(f"none ({len(wrong)})")
    opts = wrong[:3] + [right]; rng.shuffle(opts)
    item["tools"] = []
    item["request"] = T_RECALL[0].format(q=q, opts="\n".join(f"{L}) {o}" for L, o in zip("ABCD", opts)))
    item["answer"] = "ABCD"[opts.index(right)]
    item["fields"] = {"recall_of": rule["id"]}
    return item


def _recall_always(world, rule, rng, item):
    from .world import T_RECALL
    v = lambda x: f'"{x}"' if isinstance(x, str) else str(x)
    q = f"What value of `{rule['arg']}` must every tool call include?"
    right = v(rule["val"])
    pool = {"channel": ["mobile", "phone", "kiosk"], "locale": ["en-US", "fr-FR", "de-DE"]}.get(rule["arg"], ["False", "None", "0"])
    opts = [v(x) if rule["arg"] in ("channel", "locale") else x for x in pool] + [right]; rng.shuffle(opts)
    item["tools"] = []
    item["request"] = T_RECALL[0].format(q=q, opts="\n".join(f"{L}) {o}" for L, o in zip("ABCD", opts)))
    item["answer"] = "ABCD"[opts.index(right)]
    item["fields"] = {"recall_of": rule["id"]}
    return item


def sample_items(world, K, split, kind, n, seed=0):
    """exec/decoy alternate mapping and range rules; recall cycles over all rules."""
    rng = random.Random(f"{world.hash()}-{K}-{split}-{kind}-{seed}")
    sub = subset(world, K)
    if kind == "recall":
        units = list(sub); rng.shuffle(units)
    elif kind == "cross":
        units = [r for r in sub if r["kind"] == "map"]; rng.shuffle(units)
    else:
        m = [r for r in sub if r["kind"] == "map"]; r_ = [r for r in sub if r["kind"] == "range"]
        rng.shuffle(m); rng.shuffle(r_)
        units = [x for pair in zip(m, r_) for x in pair]
    return [make_item(world, units[i % len(units)], split, kind, rng, K) for i in range(n)]


_FOREIGN: dict = {}


def _foreign(seed, kmax):
    from .world import build_world
    if (seed, kmax) not in _FOREIGN:
        _FOREIGN[seed, kmax] = build_world("mixed", seed, kmax)
    return _FOREIGN[seed, kmax]


def lookup_solve(world, K, item):
    from .world import _call
    if item["fields"].get("foreign_seed") is not None:
        world = _foreign(item["fields"]["foreign_seed"], world.extra["kmax"])
    f, sub = item["fields"], subset(world, K)
    extra = always_args(sub)
    if "fare" in f:
        kw = {"user": f["user"], "code": f["code"]}
        for r in sub:
            if r["kind"] == "range" and r["lo"] <= f["fare"] <= r["hi"]:
                kw["promo"] = r["code"]
        u = f.get("update")  # an in-context policy update overrides the stored range rule
        if u and u["lo"] <= f["fare"] <= u["hi"]:
            if u["kind"] == "change":
                kw["promo"] = u["new_code"]
            else:
                kw.pop("promo", None)
        ren = {r["old"]: r["new"] for r in sub if r["kind"] == "map"}
        return [_call(ren.get(f["tool"], f["tool"]), **kw, **extra)]
    ren = {r["old"]: r["new"] for r in sub if r["kind"] == "map"}
    return [_call(ren.get(f["tool"], f["tool"]), user=f["user"], date=f["date"], **extra)]


def restate(rule):
    if rule["kind"] == "range":
        return [(f"What is the rule about fares from {rule['lo']} to {rule['hi']}?", rule["text"]),
                (f"Remind me which promo applies to fares {rule['lo']}–{rule['hi']}.",
                 f"Any fare from {rule['lo']} up to {rule['hi']} gets promo=\"{rule['code']}\"; other fares get no promo from this rule.")]
    v = f'"{rule["val"]}"' if isinstance(rule["val"], str) else str(rule["val"])
    return [(f"What is the rule about `{rule['arg']}`?", rule["text"]),
            (f"Remind me how `{rule['arg']}` works.", f"Add {rule['arg']}={v} to every single tool call, whatever the request.")]
