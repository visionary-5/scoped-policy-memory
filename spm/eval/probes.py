"""CC-Gen probe set for the training-recipe comparison (300 items per probe).

Probes (official CAP-CPT constructions plus held-out subsets):
  none               no policy in context: answers must come from memory;
  orig               the original policy in context (consistent);
  override_official  one Override variant (random task, vseed 1000) in context, items over all tasks;
  override_held      Override of a HELD-OUT task (never practised in conflict training), only items
                     whose answer actually changes; ``old_answer`` = memory answer;
  substitute         an entirely new policy (seed 100+s), only items whose answer changes.
"""
from __future__ import annotations

import collections
import random

from ..data.ccgen import policy_block
from ..data.variants import override_world, substitute_world
from ..data.world import lookup_solve, sample_items
from ..training.rows import heldout_tasks, variant_item

N_PROBE = 300
PROBES = ('none', 'orig', 'override_official', 'override_held', 'substitute')


def build_probe_rows(w, n=N_PROBE):
    T = w.extra['T']; own = policy_block(w.rules)
    trng = random.Random(f'ccgen-v1-test-{w.rule_type}-{w.seed}'); rows = []
    std = sample_items(w, T, 'test', 'exec', n)
    rows += [{**it, 'probe': 'none', 'context': None} for it in std]
    rows += [{**it, 'probe': 'orig', 'context': own} for it in std]
    k0 = trng.randint(1, T); spec0 = {'kind': 'override', 'task': k0, 'vseed': 1000}; vw0 = override_world(w, k0, 1000)
    for it in std:
        v = dict(it); v['fields'] = {**it['fields'], 'variant': spec0}; v['answer'] = lookup_solve(w, T, v)
        rows.append({**v, 'probe': 'override_official', 'context': policy_block(vw0.rules), 'old_answer': it['answer']})
    H = heldout_tasks(w); i = 0
    while sum(r['probe'] == 'override_held' for r in rows) < n:
        k = H[i % len(H)]; spec = {'kind': 'override', 'task': k, 'vseed': 2000 + i}; vw = override_world(w, k, 2000 + i); i += 1
        it = variant_item(w, vw, spec, k, 'test', trng); old = lookup_solve(w, T, {**it, 'fields': {**it['fields'], 'variant': None}})
        if old != it['answer']:
            rows.append({**it, 'probe': 'override_held', 'context': policy_block(vw.rules), 'old_answer': old})
    fw = substitute_world(w, 100 + w.seed); spec = {'kind': 'substitute', 'seed': 100 + w.seed}; m = 0
    while m < n:
        it = variant_item(w, fw, spec, trng.randint(1, T), 'test', trng); old = lookup_solve(w, T, {**it, 'fields': {**it['fields'], 'variant': None}})
        if old != it['answer']:
            rows.append({**it, 'probe': 'substitute', 'context': policy_block(fw.rules), 'old_answer': old}); m += 1
    info = {'groups': dict(collections.Counter(r['probe'] for r in rows)), 'override_official_task': k0,
            'override_official_changed': sum(r['probe'] == 'override_official' and r['answer'] != r['old_answer'] for r in rows)}
    return rows, info


def summarize(records):
    """Per-probe success rate, truncation, prompt tokens and stale-answer rate."""
    g = collections.defaultdict(list)
    for r in records:
        g[r['probe']].append(r)
    out = []
    for p, rs in sorted(g.items()):
        row = {'probe': p, 'n': len(rs), 'sr': sum(r['correct'] for r in rs) / len(rs),
               'truncation': sum(r['truncated'] for r in rs) / len(rs),
               'prompt_tokens': sum(r['prompt_tokens'] for r in rs) / len(rs)}
        if any('old' in r for r in rs):
            row['old_answer_rate'] = sum(r.get('old', False) for r in rs) / len(rs)
        if p == 'override_official':
            ch = [r for r in rs if r['answer'] != r['old_answer']]
            row['changed_n'] = len(ch); row['changed_sr'] = sum(r['correct'] for r in ch) / len(ch) if ch else None
        out.append(row)
    return out


def summarize_expr(records):
    g = collections.defaultdict(list)
    for r in records:
        g[r['probe']].append(r)
    return [{'probe': p, 'n': len(rs), 'sr_expr': sum(r['correct_expr'] for r in rs) / len(rs)} for p, rs in sorted(g.items())]
