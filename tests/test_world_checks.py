"""World self-checks: program oracle, an independent text-only policy parser, and split leakage."""
import re

from spm.data.ccgen import CCGEN_TYPES, LOOKUP_POOLS
from spm.data.prompts import rules_block
from spm.data.world import build_world, lookup_solve, sample_items
from spm.eval.score import render_calls, score_call


def _val(t, env):
    t = t.strip()
    return int(t) if t.isdigit() else env[t]


RULES = [
    (r"The original lookup value of (\S+) from the selected profile\.", lambda a, E: _val(a[0], E)),
    (r"A constant value of (\d+)\.", lambda a, E: int(a[0])),
    (r"The value of (\S+)\.", lambda a, E: _val(a[0], E)),
    (r"The maximum between (\S+) and (\S+)\.", lambda a, E: max(_val(a[0], E), _val(a[1], E))),
    (r"The minimum between (\S+) and (\S+)\.", lambda a, E: min(_val(a[0], E), _val(a[1], E))),
]
IFELSE = re.compile(r"If ([^.]*\.) > ([^.]*\.), then arg_\d+: ([^.]*\.) else arg_\d+: ([^.]*\.)")


def _compile(text):
    m = IFELSE.fullmatch(text)
    if m:
        l, r, b1, b2 = (_compile(x) for x in m.groups())
        return lambda E: b1(E) if l(E) > r(E) else b2(E)
    for rx, fn in RULES:
        mm = re.fullmatch(rx, text)
        if mm:
            return lambda E, fn=fn, g=mm.groups(): fn(g, E)
    raise ValueError(text)


def parse_policy(md):
    G = {f"global_attribute_{k}": int(v) for k, v in re.findall(r"Global_Attribute_Value(\d+) = (\d+)", md)}
    specs = {int(t): [_compile(x) for _, x in re.findall(r"^  - arg_(\d+): (.*)$", body, re.M)]
             for t, body in re.findall(r"### Task_Type_(\d+)\n(.*?)(?=\n\n### |\Z)", md, re.S)}
    return G, specs


def test_lite_worlds_oracle_parser_and_splits():
    for rt in [t for t in CCGEN_TYPES if t.endswith("_lite")]:
        T = CCGEN_TYPES[rt][0]
        for seed in range(2):
            w = build_world(rt, seed)
            G, specs = parse_policy(rules_block(w.subset(T), 0))
            assert len(specs) == T and len(G) == 3
            for split in ("train", "test"):
                for kind in ("exec", "decoy", "neutral"):
                    for it in sample_items(w, T, split, kind, 60):
                        g = it["answer"]
                        assert score_call(render_calls(g), g)["correct"]
                        assert lookup_solve(w, T, it) == g
                        if kind != "neutral":
                            E = {**it["fields"]["profile"], **G}
                            assert [f(E) for f in specs[it["fields"]["task"]]] == g[0][1]["attributes"]
                            for v in it["fields"]["profile"].values():
                                assert (v in LOOKUP_POOLS[split]) if isinstance(v, str) else \
                                    v % 5 in {"train": (0, 1, 2), "test": (4,)}[split]
            tr = sample_items(w, T, "train", "exec", 1000)
            te = sample_items(w, T, "test", "exec", 300)
            assert not {i["request"] for i in tr} & {i["request"] for i in te}
            assert {i["template_id"][5:] for i in tr} <= {str(x) for x in range(12)}
