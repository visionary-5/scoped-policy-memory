"""Gate V2: conflict samples vs consistent exposure at equal dose (synthetic world, seed 0).

Base: the R arm of the exposure analysis (480 rows; mapping list in context, range list never).
Practice: 24 rows (4.8% of 504) taken from the update-practice rows on the practised rules r0-r7,
three per rule: r0-r5 change/revoke/irrelevant, r6 change/revoke/change, r7 change/revoke/revoke
(change 9, revoke 9, irrelevant 6).
  conf_k05  context = mapping list + a "Policy updates" section; label = the updated answer
            (irrelevant rows: label = memory).
  cons_u05  the same 24 requests with the update removed (plain range exec items); context = mapping
            list + full range list (consistent with memory); label = memory.
Reading (frozen): conflict_better if cons_u05 range conflict-following <= conf_k05 - 15pp or range memory
<= conf_k05 - 10pp; decisive_context_suffices if cons_u05 also follows >= 90%; otherwise inconclusive.
Evaluate both arms on the exposure test set (``analysis.exposure``).

    python -m analysis.gate_v2 prepare --root OUT
    python -m analysis.gate_v2 audit   --root OUT --evals EVAL_DIR
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from spm.data.prompts import build_messages
from spm.data.recipe import uid
from spm.data.world import build_world, lookup_solve
from spm.eval.score import render_calls, score_call
from spm.io import read_jsonl, save_json, sha256_file, write_jsonl

from .exposure import arm_rows, block, source_metrics
from .synth_data import K, PRACTICED, fresh_data, update_practice_rows, with_update

SEED = 0
ARMS = ('conf_k05', 'cons_u05')
PLAN = {rid: (['change', 'revoke', 'irrelevant'] if i < 6 else ['change', 'revoke', 'change'] if i == 6 else ['change', 'revoke', 'revoke'])
        for i, rid in enumerate(PRACTICED)}
THRESHOLDS = {'follow_min': .90, 'memory_min': .95, 'cons_follow_ge': .90, 'follow_gap': .15, 'memory_gap': .10}


def pick_practice(upd):
    by = collections.defaultdict(list)
    for d in sorted(upd, key=lambda d: d['uid']):
        by[d['rule_id'], d['item']['fields']['update_case']].append(d)
    out = []
    for rid, cases in PLAN.items():
        used = collections.Counter()
        for c in cases:
            out.append(by[rid, c][used[c]]); used[c] += 1
    return out


def to_consistent(w, d):
    it = json.loads(json.dumps(d['item'])); f = it['fields']; u = f.pop('update'); f.pop('update_case'); f.pop('group', None)
    it['kind'] = 'exec'
    if f.get('fare') is not None and not (u['lo'] <= f['fare'] <= u['hi']):
        it['rule_ids'] = sorted(set(it['rule_ids']) - {u['rule_id']})
    it['answer'] = lookup_solve(w, K, it)
    r = next(x for x in w.subset(K) if x['id'] == d['rule_id'])
    assert r['lo'] <= f['fare'] <= r['hi'] and it['answer'][0][1].get('promo') == r['code']
    oracle = render_calls(it['answer']); assert score_call(oracle, lookup_solve(w, K, it))['correct']
    return it, oracle


def build(seed=SEED):
    w = build_world('mixed', seed); sub = w.subset(K)
    ctx_map = block(sub, {'map': 1}); ctx_full = block(sub, {'map': 1, 'range': 1})
    base = fresh_data(seed)
    R = arm_rows(seed, base, 'R')
    prac = pick_practice(update_practice_rows(seed, base))
    assert collections.Counter(d['item']['fields']['update_case'] for d in prac) == {'change': 9, 'revoke': 9, 'irrelevant': 6}
    conf, cons = [], []
    for d in prac:
        it = d['item']
        conf.append({**d, 'source': 'conf_' + it['fields']['update_case'], 'pair_uid': d['uid'],
                     'messages': build_messages(it, with_update(ctx_map, it))})
        ci, co = to_consistent(w, d)
        cons.append({'uid': uid(ci), 'kind': 'exec', 'rule_id': d['rule_id'], 'item': ci, 'oracle': co, 'source': 'cons_' + it['fields']['update_case'],
                     'pair_uid': d['uid'], 'messages': build_messages(ci, ctx_full), 'world_hash': w.hash(), 'K': K, 'seed': seed, 'rule_type': 'mixed'})
    out = {}
    for arm, rows in (('conf_k05', conf), ('cons_u05', cons)):
        data = sorted(R + rows, key=lambda d: d['uid']); assert len({d['uid'] for d in data}) == len(data)
        out[arm] = data
    return R, out


def prepare(root, seed=SEED):
    root = Path(root); R, arms = build(seed)
    man = {'R': sha256_file(write_jsonl(root / 'data' / f'R-s{seed}.jsonl', R))}
    for arm, rows in arms.items():
        man[arm] = sha256_file(write_jsonl(root / 'data' / f'{arm}-s{seed}.jsonl', rows))
    save_json(root / 'data' / 'manifest.json', man)
    return man


def audit(root, evals, seed=SEED):
    res = {'missing': []}
    m = {}
    for arm in ARMS:
        p = Path(evals) / f'{arm}-s{seed}' / 'eval.jsonl'
        if not p.exists():
            res['missing'].append(arm); continue
        m[arm] = source_metrics(read_jsonl(p), seed)['range']
    if not res['missing']:
        c, u = m['conf_k05'], m['cons_u05']; t = THRESHOLDS
        better = u['conf_new'] <= c['conf_new'] - t['follow_gap'] or u['memory_src'] <= c['memory_src'] - t['memory_gap']
        res.update(conf_k05=c, cons_u05=u, G2a_holds=c['conf_new'] >= t['follow_min'] and c['memory_src'] >= t['memory_min'],
                   reading='conflict_better' if better else 'decisive_context_suffices' if u['conf_new'] >= t['cons_follow_ge'] else 'inconclusive')
    save_json(Path(root) / 'audit_v2.json', res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['prepare', 'audit']); ap.add_argument('--root', required=True); ap.add_argument('--evals')
    ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args(argv)
    print(json.dumps(prepare(a.root, a.seed) if a.mode == 'prepare' else audit(a.root, a.evals, a.seed), indent=1))


if __name__ == '__main__':
    main()
