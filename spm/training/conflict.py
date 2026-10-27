"""Conflict-aware practice rows: the context states a rule that differs from what is being memorised,
and the label follows the context.

* Override practice (``override_rows``): the full policy with one practised task re-sampled. Three in
  four rows ask about the changed task (label = new formula); one in four asks about another task
  (label = memory), so the model learns to defer only where the context differs.
* Substitute practice (``substitute_rows``): an entirely different policy (foreign seeds 50/51) in the
  context; label = that policy's answer (out-of-scope policy, memory must not leak in).
* Closure repair (``repair_rows``): continuation data in the runtime's dependency-closed format.
  The ``repair`` arm pairs a changed practised task with its new section (label = new answer); the
  ``consistent`` arm uses the identical requests with the unchanged section (label = memory).
"""
from __future__ import annotations

import copy
import random

from ..data.ccgen import policy_block
from ..data.variants import override_world, substitute_world
from ..data.world import lookup_solve
from ..runtime.context import context_for
from .rows import practiced_tasks, train_row, variant_item

FOREIGN = (50, 51)
N_OVERRIDE, N_SUBSTITUTE = 64, 64
N_REPAIR = 96
REPAIR_VSEED_BASE = 70000


def override_rows(w, rng, fresh, n=N_OVERRIDE):
    T = w.extra['T']; P = practiced_tasks(w); out = []
    for i in range(n):
        k = P[i % len(P)]; spec = {'kind': 'override', 'task': k, 'vseed': i}; vw = override_world(w, k, i)
        target = k if i % 4 else rng.choice([x for x in range(1, T + 1) if x != k])
        out.append(train_row(w, fresh(lambda: variant_item(w, vw, spec, target, 'train', rng)), policy_block(vw.rules), 'override'))
    return out


def substitute_rows(w, rng, fresh, n=N_SUBSTITUTE):
    T = w.extra['T']; out = []
    for i in range(n):
        fs = FOREIGN[i % 2]; spec = {'kind': 'substitute', 'seed': fs}; fw = substitute_world(w, fs)
        out.append(train_row(w, fresh(lambda: variant_item(w, fw, spec, rng.randint(1, T), 'train', rng)),
                             policy_block(fw.rules), 'substitute'))
    return out


def repair_rows(w, original_rows, n=N_REPAIR):
    """Closure-format continuation rows for both arms (identical requests, different labels/context)."""
    T = w.extra['T']; P = practiced_tasks(w)
    seen = {d['item']['request'] for d in original_rows if d.get('item')}
    rng = random.Random(f'closeout-v1-repair-{w.seed}')
    extra = {'repair': [], 'consistent': []}
    for i in range(n):
        k = P[i % len(P)]; vs = REPAIR_VSEED_BASE + i; vw = override_world(w, k, vs)
        spec = dict(kind='override', task=k, vseed=vs)
        for _ in range(10000):
            it = variant_item(w, vw, spec, k, 'train', rng)
            if it['request'] not in seen:
                break
        else:
            raise RuntimeError('repair request pool exhausted')
        seen.add(it['request'])
        old = copy.deepcopy(it); old['fields'].pop('variant', None); old['answer'] = lookup_solve(w, T, old)
        extra['repair'].append(train_row(w, it, context_for(w, vw, it['request'], k, 'closed'), 'closeout_conflict'))
        extra['consistent'].append(train_row(w, old, context_for(w, w, old['request'], k, 'closed'), 'closeout_consistent'))
    return extra
