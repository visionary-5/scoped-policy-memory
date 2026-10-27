"""Item-by-item parity with the original research implementation (read-only reference).

Skipped unless the original code is found (``SPM_REFERENCE_REPO`` or ``../rules2weights``). The golden
SHA tests cover the frozen paper data; these tests extend the comparison to seeds, rule types and
functions that no frozen file covers. Byte-code writing is disabled so the reference tree is never
modified.
"""
import json
import sys
import types
from pathlib import Path

import pytest

from conftest import reference_repo

REF = reference_repo()
pytestmark = pytest.mark.skipif(REF is None, reason="original implementation not available")
RT = "ccgen_t12a3s1_lite"


@pytest.fixture(scope="module")
def ref():
    sys.dont_write_bytecode = True
    for p in (REF, REF / "scripts", REF / "scripts" / "explore_v5" / "capcpt_base_v1", REF / "scripts" / "closeout_v1"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import ccgen_probe
    import r2w.ccgen_variants as variants
    import r2w.prompts as prompts
    import r2w.recipe as recipe
    import r2w.score as score
    import r2w.world as world
    import r2w.world_ccgen as wc
    import system.context_policy as cp
    return types.SimpleNamespace(world=world, wc=wc, variants=variants, recipe=recipe, prompts=prompts, score=score,
                                 cp=cp, probe=ccgen_probe)


def _dump(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=False)


@pytest.mark.parametrize("rt", ["ccgen_t12a3s1_lite", "ccgen_t12a3s2_lite", "ccgen_t5a5s1_lite", "ccgen_t12a3s1", "mixed"])
@pytest.mark.parametrize("seed", [3, 7])
def test_items_and_recipes(ref, rt, seed):
    from spm.data import recipe, world
    from spm.data.prompts import build_messages
    a, b = world.build_world(rt, seed), ref.world.build_world(rt, seed)
    assert a.hash() == b.hash() and _dump(a.rules) == _dump(b.rules)
    K = a.extra.get("T", 16)
    kinds = ("exec", "decoy", "recall", "neutral") + (("cross",) if rt == "mixed" else ())
    for split in ("train", "dev", "test"):
        for kind in kinds:
            x = world.sample_items(a, K, split, kind, 40); y = ref.world.sample_items(b, K, split, kind, 40)
            assert _dump(x) == _dump(y), (split, kind)
            if kind != "recall":
                assert [world.lookup_solve(a, K, i) for i in x] == [ref.world.lookup_solve(b, K, i) for i in y]
            assert [build_messages(i) for i in x] == [ref.prompts.build_messages(i) for i in y]
    if rt != "mixed":
        assert _dump(recipe.build_train_set(a, K, seed, n_exec=4, n_decoy=2, ctx_mix=0.5)) == \
            _dump(ref.recipe.build_train_set(b, K, seed, n_exec=4, n_decoy=2, ctx_mix=0.5))
        from spm.data.ccgen import policy_block
        assert policy_block(a.rules, 1, a, K) == ref.wc.policy_block(b.rules, 1, b, K)


@pytest.mark.parametrize("seed", [3, 4])
def test_variants(ref, seed):
    from spm.data.variants import override_world, substitute_world
    from spm.data.world import build_world
    a, b = build_world(RT, seed), ref.world.build_world(RT, seed)
    for task in (1, 7, 12):
        for vseed in (0, 99, 40001):
            x, y = override_world(a, task, vseed), ref.variants.override_world(b, task, vseed)
            assert x.hash() == y.hash() and _dump(x.rules) == _dump(y.rules)
    assert substitute_world(a, 51).hash() == ref.variants.substitute_world(b, 51).hash()


def test_ccgen_arms_and_probe_set_other_seed(ref, tmp_path):
    """The whole recipe + probe-set construction for an unfrozen seed, against the original script."""
    from spm.io import sha256_file
    from spm.training.datasets import prepare_recipe
    prepare_recipe(tmp_path / "new", seeds=(3,), with_cons=False)
    old = ref.probe.SEEDS
    try:
        ref.probe.SEEDS = (3,)
        ref.probe.prepare(types.SimpleNamespace(root=tmp_path / "old", rt=RT))
    finally:
        ref.probe.SEEDS = old
    for name in ("P", "M", "prec", "ctrl", "test"):
        assert sha256_file(tmp_path / "new" / "recipe" / "data" / RT / f"{name}-s3.jsonl") == \
            sha256_file(tmp_path / "old" / "data" / RT / f"{name}-s3.jsonl"), name


def test_closure_and_scoring_functions(ref):
    from spm.data.variants import override_world
    from spm.data.world import build_world
    from spm.eval import score
    from spm.runtime.context import context_for
    a, b = build_world(RT, 5), ref.world.build_world(RT, 5)
    for k in (2, 9):
        va, vb = override_world(a, k, 77), ref.variants.override_world(b, k, 77)
        for req in (f"do task_type_{k} now", "do task_type_3 now", "task_type_1 and task_type_2"):
            for mode in ("raw", "closed", "typed", "full", "matched"):
                assert context_for(a, va, req, k, mode) == ref.cp.context_for(b, vb, req, k, mode)
    outs = ["finish_task_1(attributes=[1, 2])", "```\nf(a=1)\n```", "f(a=[1,'x'], b=True); g(c=2.0)", "f(1)", "", "x=1",
            "f(a=max(1, 2))", "f(**k)", "  \n f(a=-3)  \nmore"]
    for o in outs:
        assert score.normalize_lenient(o) == ref.score.normalize_lenient(o)
        assert score.parse_calls(o) == ref.score.parse_calls(o)
        assert score.score_call(o, [["f", {"a": 1}]]) == ref.score.score_call(o, [["f", {"a": 1}]])
    sys.path.append(str(REF / "scripts" / "explore_v5" / "rescore_v1"))
    import rescore_expr
    for o in outs + ["f(a=ord('x') + 3 * 2)", "f(a=sum([1, 2]) // 2)"]:
        assert score.parse_calls_eval(score.normalize_lenient(o)) == rescore_expr.parse_calls_eval(score.normalize_lenient(o))


def test_capcpt_data(ref, tmp_path):
    """Spec inventory, QA, LLM prompts, anchor sets and the CPT corpus (fallback templates, fake tokenizer)."""
    from spm.baselines import capcpt as new
    from spm.data.world import build_world
    from spm.io import read_jsonl, write_jsonl, sha256_file
    from spm.training.datasets import build_arms
    from spm.eval.probes import build_probe_rows
    import capcpt as old
    seed = 1; w = build_world(RT, seed); wo = ref.world.build_world(RT, seed)
    specs = new.spec_list(w)
    assert _dump(specs) == _dump(old.spec_list(wo))
    assert _dump(new.qa_list(w, specs)) == _dump(old.qa_list(wo, old.spec_list(wo)))
    prompt = "Analyse.\n" + new.CUT + "\n(ignored tail)"
    pf = tmp_path / "prompt.md"; pf.write_text(prompt)
    assert new.prompts_for(w, specs, new.qa_list(w, specs), prompt) == old.prompts_for(wo, old.spec_list(wo), old.qa_list(wo, old.spec_list(wo)), pf)
    # build the original prepare inputs (ccgen-v1 layout) and run the original prepare + assemble
    src = tmp_path / "src" / "data" / RT; src.mkdir(parents=True)
    arms = build_arms(RT, seed); rows, _ = build_probe_rows(w)
    for name in ("P", "prec"):
        write_jsonl(src / f"{name}-s{seed}.jsonl", arms[name])
    write_jsonl(src / f"test-s{seed}.jsonl", rows)
    man = {str(s): {} for s in (0, 1, 2)}
    man[str(seed)] = {"P": {"sha256": sha256_file(src / f"P-s{seed}.jsonl")}, "prec": {"sha256": sha256_file(src / f"prec-s{seed}.jsonl")},
                      "test": {"sha256": sha256_file(src / f"test-s{seed}.jsonl")}}
    (src / "manifest.json").write_text(json.dumps(man))
    root = tmp_path / "old"; root.mkdir()
    (root / "protocol.json").write_text(json.dumps({"capcpt_data_per_seed": {"n_scenario_per_spec": 40, "n_behavior_per_spec": 10}}))
    saved = old.SEEDS
    try:
        old.SEEDS = (seed,)
        old.prepare(types.SimpleNamespace(root=root, src=tmp_path / "src", prompt_file=pf, tmpl=tmp_path / "none"))
    finally:
        old.SEEDS = saved
    sets = new.anchor_sets(w, arms["P"], arms["prec"])
    for name, data in sets.items():
        assert sha256_file(write_jsonl(tmp_path / f"{name}.jsonl", data)) == sha256_file(root / "data" / RT / f"{name}-s{seed}.jsonl")

    class FakeTok:
        eos_token = "</s>"

        def apply_chat_template(self, m, **kw):
            return "|".join(f"{x['role']}:{x['content']}" for x in m) + "|assistant:"

        def __call__(self, text, **kw):
            return {"input_ids": list(range(len(text) // 4 + 1))}
    fake = types.ModuleType("transformers"); fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: FakeTok())
    real = sys.modules.get("transformers"); sys.modules["transformers"] = fake
    try:
        old.assemble(types.SimpleNamespace(root=root, seed=seed, allow_fallback=True, model="fake", skip_tokens=True))
    finally:
        if real is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = real
    tok = FakeTok()
    full, nosim, _ = new.assemble(w, specs, new.qa_list(w, specs), {}, rows, read_jsonl(root / "data" / RT / f"P_anchor-s{seed}.jsonl"),
                                  lambda m: tok.apply_chat_template(m), n_scen=40, n_beh=10)
    for name, docs in (("cpt_full", full), ("cpt_nosim", nosim)):
        assert sha256_file(write_jsonl(tmp_path / f"{name}.jsonl", docs)) == sha256_file(root / "data" / RT / f"{name}-s{seed}.jsonl"), name
