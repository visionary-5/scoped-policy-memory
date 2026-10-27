"""Training-recipe baselines on the same base data as the conflict-aware recipe (``prec``).

* P     plain SFT: memory practice only, no policy in context.
* M     global mixing: each exec/decoy row carries the full original policy with probability 0.5.
* ctrl  P + 128 ordinary rows (equal data amount to prec's 128 practice rows).
* cons  P + 128 consistent-context rows paired 1:1 with prec's practice rows: same request (task,
        profile, template), the original policy in context, label = memory. ``cons`` and ``prec``
        differ only in whether the context conflicts with memory and which source the label follows.
"""
from __future__ import annotations

import json
import random

from ..data.ccgen import policy_block, rule_ids_of, task_rule
from ..data.prompts import build_messages
from ..data.recipe import uid
from ..data.world import lookup_solve, make_item
from ..eval.score import render_calls, score_call
from ..training.rows import train_row

N_ORDINARY = 128
M_RATE = 0.5


def ordinary_rows(w, rng, fresh, n=N_ORDINARY):
    T = w.extra['T']
    return [train_row(w, fresh(lambda: make_item(w, task_rule(w, rng.randint(1, T)), 'train', 'exec', rng, T)), None, 'ordinary')
            for _ in range(n)]


def mixed_exposure_rows(w, base, seed):
    own = policy_block(w.rules)
    mrng = random.Random(f'ccgen-v1-M-{w.rule_type}-{seed}'); out = []
    for d in base:
        d = dict(d)
        if d.get('item') and mrng.random() < M_RATE:
            d['messages'] = build_messages(d['item'], own)
        out.append(d)
    return out


def consistent_row(w, d):
    """prec practice row -> same-request row with the original policy in context and the memory label."""
    it = json.loads(json.dumps(d['item'])); T = w.extra['T']; k = it['fields']['task']
    it['fields'].pop('variant', None)
    it['rule_ids'] = rule_ids_of(task_rule(w, k))
    it['answer'] = lookup_solve(w, T, it)
    assert it['split'] == 'train' and str(it['template_id']).startswith('train')
    oracle = render_calls(it['answer']); assert score_call(oracle, lookup_solve(w, T, it))['correct']
    return {'uid': uid(it), 'kind': 'exec', 'rule_id': f't{k}', 'item': it, 'oracle': oracle, 'source': 'cons_' + d['source'],
            'pair_uid': d['uid'], 'messages': build_messages(it, policy_block(w.rules)), 'world_hash': w.hash(), 'K': T,
            'seed': w.seed, 'rule_type': w.rule_type}


def cons_from(w, P_rows, prec_rows):
    practice = [d for d in prec_rows if d.get('source') in ('override', 'substitute')]
    pairs = [consistent_row(w, d) for d in practice]
    return sorted(P_rows + pairs, key=lambda d: d['uid'])
