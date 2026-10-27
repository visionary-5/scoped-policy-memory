"""Row builders shared by all training-data constructions."""
from __future__ import annotations

from ..data.ccgen import task_rule
from ..data.prompts import build_messages
from ..data.recipe import uid
from ..data.world import lookup_solve, make_item
from ..eval.score import render_calls, score_call


def variant_item(w, vw, spec, k, split, rng):
    """An exec item for task ``k`` generated in variant world ``vw`` and labelled with ``spec``.
    The answer is re-checked against the oracle of the base world (which resolves the variant)."""
    it = make_item(vw, task_rule(vw, k), split, 'exec', rng, w.extra['T'])
    it['fields']['variant'] = spec
    it['world'] = f'{w.rule_type}-s{w.seed}'
    assert it['answer'] == lookup_solve(w, w.extra['T'], it)
    return it


def train_row(w, it, ctx, source):
    """One frozen training row; the oracle label is verified before it is written."""
    oracle = render_calls(it['answer'])
    assert score_call(oracle, lookup_solve(w, w.extra['T'], it))['correct']
    return {'uid': uid(it), 'kind': 'exec', 'rule_id': f"t{it['fields']['task']}", 'item': it, 'oracle': oracle,
            'source': source, 'messages': build_messages(it, ctx), 'world_hash': w.hash(), 'K': w.extra['T'],
            'seed': w.seed, 'rule_type': w.rule_type}


def memory_label(w, it):
    """The answer the stored (original) policy gives for a variant item."""
    return lookup_solve(w, w.extra['T'], {**it, 'fields': {**it['fields'], 'variant': None}})


def practiced_tasks(w):
    """Conflict practice is restricted to the first half of the tasks; the rest are held out."""
    T = w.extra['T']
    return list(range(1, T // 2 + 1))


def heldout_tasks(w):
    T = w.extra['T']
    return list(range(T // 2 + 1, T + 1))


def fresh_sampler(seen: set):
    """Draw items whose request was never used before (shared de-duplication across row groups)."""
    def fresh(make):
        for _ in range(10000):
            it = make()
            if it['request'] not in seen:
                seen.add(it['request'])
                return it
        raise RuntimeError('cannot generate a new training request')
    return fresh
