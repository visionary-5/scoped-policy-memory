"""System confirmation sets (stable / changed / unrelated) and the frozen acceptance gates.

Request groups (equal size, ``n`` each):
  stable     no rule changed; any task;
  changed    a held-out task (never practised in conflict training) was edited with a fresh Override
             and the request asks about it; only requests whose answer actually changes are kept;
  unrelated  the same request as a stable item, but some OTHER task was edited (answer = memory).
Development (``phase='dev'``) uses dev-split requests and edit seeds 30000+/50000+; confirmation uses
fresh test-split requests and unseen edit seeds 40000+/60000+, excluding every request used by the
training data, the probe set or the development set.

Every row is rendered under each context view (``spm.runtime.context.VIEWS``) so that all arms and
views are item-paired.
"""
from __future__ import annotations

import copy
import random

from ..data.ccgen import tasks
from ..data.variants import override_world
from ..data.world import lookup_solve, make_item
from ..eval.score import render_calls, score_call
from ..runtime.context import VIEWS, context_for
from ..training.rows import variant_item

GROUPS = ('stable', 'changed', 'unrelated')
N_DEV, N_CONFIRM = 150, 500
GATE = {"changed_min": 0.90, "vs_prec_full_max_drop": 0.03, "unaffected_max_drop": 0.03,
        "changed_prompt_ratio_vs_full_max": 0.5, "truncation_max_exclusive": 0.02, "token_ratio_vs_matched_max": 0.90}


def build_rows(w, phase: str, excluded: set, n: int | None = None):
    """Return (rows, worlds) where worlds[i] = (current policy world, changed task or None)."""
    dev = phase == 'dev'; n = n or (N_DEV if dev else N_CONFIRM)
    ts = tasks(w); T = w.extra['T']; first_held = T // 2 + 1; n_held = T - T // 2
    seen = set(excluded); rows = []; worlds = []
    rng = random.Random(f'closeout-v1-{phase}-{w.seed}')
    split = 'dev' if dev else 'test'
    for group in ('stable', 'changed'):
        count = 0; attempt = 0
        while count < n:
            attempt += 1
            if attempt > n * 100:
                raise RuntimeError('request pool exhausted')
            if group == 'changed':
                k = first_held + count % n_held; vseed = (30000 if dev else 40000) + attempt
                vw = override_world(w, k, vseed)
                spec = dict(kind='override', task=k, vseed=vseed)
                it = variant_item(w, vw, spec, k, split, rng)
                old = lookup_solve(w, T, {**it, 'fields': {**it['fields'], 'variant': None}})
                if old == it['answer']:
                    continue
            else:
                k = rng.randint(1, T); vw = w
                it = make_item(w, ts[k - 1], split, 'exec', rng, T)
                old = it['answer']
            if it['request'] in seen:
                continue
            seen.add(it['request']); count += 1
            assert score_call(render_calls(it['answer']), it['answer'])['correct']
            rows.append({**it, 'probe': group, 'old_answer': old}); worlds.append((vw, k if group == 'changed' else None))
            if group == 'stable':
                other = rng.choice([x for x in range(1, T + 1) if x != k])
                uv = override_world(w, other, (50000 if dev else 60000) + count)
                ur = copy.deepcopy(it); ur['probe'] = 'unrelated'; ur['old_answer'] = old
                rows.append(ur); worlds.append((uv, other))
    assert all(r['request'] not in excluded for r in rows)
    return rows, worlds


def render_views(w, rows, worlds, views=VIEWS):
    return {view: [{**r, 'context': context_for(w, vw, r['request'], change, view)} for r, (vw, change) in zip(rows, worlds)]
            for view in views}


# ---------- acceptance gates (thresholds frozen before any result) ----------
def dev_gate(cells: dict, ref_full: dict, gate=GATE) -> bool:
    """``cells``/``ref_full`` map group -> {'sr', 'prompt_tokens', 'truncation'} (one seed, dev)."""
    return (cells['changed']['sr'] >= max(gate['changed_min'], ref_full['changed']['sr'] - gate['vs_prec_full_max_drop'])
            and min(cells[g]['sr'] for g in ('stable', 'unrelated')) >= ref_full['stable']['sr'] - gate['unaffected_max_drop']
            and cells['changed']['prompt_tokens'] <= ref_full['changed']['prompt_tokens'] * gate['changed_prompt_ratio_vs_full_max']
            and max(cells[g]['truncation'] for g in cells) < gate['truncation_max_exclusive'])


def confirm_checks(system: dict, prec_full: dict, base_matched: dict, base_full: dict, gate=GATE) -> dict:
    """Per-seed confirmation checks of the selected system against the frozen gates."""
    quality = all(system[g]['sr'] >= max(base_matched[g]['sr'], base_full[g]['sr']) - gate['unaffected_max_drop'] for g in system)
    token_ratio = sum(system[g]['prompt_tokens'] for g in system) / sum(base_matched[g]['prompt_tokens'] for g in system)
    c = dict(changed_ge_90=system['changed']['sr'] >= gate['changed_min'],
             retains_full=system['changed']['sr'] >= prec_full['changed']['sr'] - gate['vs_prec_full_max_drop'],
             unaffected_ge_90=min(system[g]['sr'] for g in ('stable', 'unrelated')) >= gate['changed_min'],
             vs_strong_prompt=quality, token_ratio_vs_matched=token_ratio,
             token_saving=token_ratio <= gate['token_ratio_vs_matched_max'],
             truncation=max(system[g]['truncation'] for g in system))
    c['passes'] = all(c[k] for k in ('changed_ge_90', 'retains_full', 'unaffected_ge_90', 'vs_strong_prompt', 'token_saving')) \
        and c['truncation'] < gate['truncation_max_exclusive']
    return c
