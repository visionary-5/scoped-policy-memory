"""Synthetic-world (``mixed``, K=16) training base and update items for the mechanism analyses.

* ``fresh_data(seed)``: the 480-row base training set used by all exposure arms. Range rules are
  practised (8 exec + 4 decoy each, plus 4 fresh exec + 4 fresh decoy), mapping rules sit in a BM25
  retrieval pool (deployment context = pinned always-on rules + top-4 retrieved rules with one
  example each). For seed 0 the de-duplication set also contains the requests of 64 contrast pairs
  from an earlier curriculum probe (``uniform_contrast_requests``), which is required to reproduce
  the frozen seed-0 file byte-for-byte.
* ``update`` items: an in-context policy update changes or revokes a stored range rule
  (``change`` / ``revoke`` the rule of the item's fare, or ``irrelevant``: another rule).
"""
from __future__ import annotations

import collections
import copy
import random
import re

from spm.data.prompts import build_messages
from spm.data.recipe import build_train_set, uid
from spm.data.world import NamePool, _load_dict, build_world, lookup_solve, make_item, sample_items
from spm.eval.score import render_calls, score_call
from spm.runtime.retrieval import deploy_contexts

K = 16
SEEDS = (0, 1, 2)
DEV = {'exec': 500, 'decoy': 200, 'neutral': 100, 'cross': 200}
STANDARD = DEV
HEADER = "Policy updates (these replace any earlier policy):"
PRACTICED = [f"r{i}" for i in range(8)]   # updates are practised on r0-r7; r8-r15 are held out
TEST_CASES = {'change': 6, 'revoke': 6, 'irrelevant': 4}
TRAIN_CASES = {'change': 3, 'revoke': 3, 'irrelevant': 2}


# ---------------------------------------------------------------- policy updates
def code_pools(world, n=300):
    """Disjoint new promo codes per split that collide with no name in the world."""
    pool = NamePool(random.Random(f"update-v1-codes-{world.seed}"), _load_dict())
    text = str(world.rules) + str(world.extra)
    pool.used |= set(re.findall(r"[a-z]+", text))
    return {split: [pool.new(2) for _ in range(n)] for split in ("train", "dev", "test")}


def update_text(u):
    if u["kind"] == "change":
        return f'When the fare is between {u["lo"]} and {u["hi"]}, add promo="{u["new_code"]}" to the call.'
    return f'Fares between {u["lo"]} and {u["hi"]} no longer receive any promo; do not add promo for them.'


def with_update(context, item):
    return (context or "Rules:") + "\n\n" + HEADER + "\n- " + update_text(item["fields"]["update"])


def make_update_item(world, rule, split, case, rng, K, codes, targets=None):
    """A range exec item for ``rule`` plus one update (``case`` in change / revoke / irrelevant)."""
    from spm.data.synthetic import subset
    item = make_item(world, rule, split, "exec", rng, K)
    ranges = [r for r in subset(world, K) if r["kind"] == "range"]
    target = rule if case in ("change", "revoke") else rng.choice(
        [r for r in ranges if r["id"] != rule["id"] and (targets is None or r["id"] in targets)])
    kind = case if case != "irrelevant" else rng.choice(["change", "revoke"])
    item["kind"] = "update"
    item["fields"]["update_case"] = case
    item["fields"]["update"] = {"kind": kind, "rule_id": target["id"], "lo": target["lo"], "hi": target["hi"],
                                "new_code": rng.choice(codes[split]) if kind == "change" else None}
    item["fields"]["group"] = "practiced" if rule["id"] in PRACTICED else "heldout"
    item["rule_ids"] = sorted(set(item["rule_ids"]) | {target["id"]})
    item["answer"] = lookup_solve(world, K, item)
    return item


# ---------------------------------------------------------------- base training set
def parts(w):
    sub = w.subset(K)
    return ([r for r in sub if r['kind'] == 'always'], [r for r in sub if r['kind'] == 'map'],
            {r['id']: r for r in sub if r['kind'] == 'range'})


def dev_requests(w):
    return {it['request'] for kind, n in DEV.items() for it in sample_items(w, K, 'dev', kind, n)}


def uniform_contrast_requests():
    """Requests of the 64 boundary/far contrast pairs selected uniformly (seed 0 only)."""
    w = build_world('mixed', 0); sub = w.subset(K)
    ranges = [r for r in sub if r['kind'] == 'range']; pinned = [r for r in sub if r['kind'] == 'always']
    original = build_train_set(w, K, 0, n_exec=8, n_decoy=4, rule_filter=[r['id'] for r in ranges])
    rng = random.Random('curriculum-probe-v1-candidates')
    seen = {d['item']['request'] for d in original if d.get('item')}
    pairs = []
    for r in ranges:
        for j in range(16):
            for _ in range(10000):
                positive = make_item(w, r, 'train', 'exec', rng, K)
                if j % 2 == 0:
                    edge = r['lo'] if j % 4 == 0 else r['hi']
                    direction = -1 if j % 4 == 0 else 1
                    options = [edge + direction * delta for delta in range(1, 31) if (edge + direction * delta) % 5 in (0, 1, 2)]
                    fare = rng.choice(options)
                else:
                    fare = rng.randint(10000, 99999)
                    if fare % 5 not in (0, 1, 2):
                        continue
                if any(x['lo'] <= fare <= x['hi'] for x in ranges):
                    continue
                negative = copy.deepcopy(positive)
                negative['kind'] = 'decoy'; negative['fields']['fare'] = fare
                negative['rule_ids'] = [x['id'] for x in pinned]
                negative['request'], count = re.subn(r'\b' + str(positive['fields']['fare']) + r'\b', str(fare), positive['request'])
                assert count == 1
                negative['answer'] = lookup_solve(w, K, negative)
                if positive['request'] not in seen and negative['request'] not in seen:
                    seen.update([positive['request'], negative['request']])
                    break
            else:
                raise RuntimeError('cannot generate unique contrast candidates')
            pairs.append({'pair_id': f"{r['id']}-p{j}", 'rule_id': r['id'], 'items': [positive, negative]})
    trng = random.Random('curriculum-probe-v1-ties'); by_rule = collections.defaultdict(list)
    for p in pairs:
        by_rule[p['rule_id']].append(p)
    selected = []
    for rule in sorted(by_rule):
        pool = list(by_rule[rule]); trng.shuffle(pool)
        selected += pool[:4]
    return {it['request'] for p in selected for it in p['items']}


def fresh_data(seed):
    w = build_world('mixed', seed); sub = w.subset(K)
    ids = {kind: [r['id'] for r in sub if r['kind'] == kind] for kind in ['range', 'map', 'always']}
    byid = {r['id']: r for r in sub}; pinned = [byid[i] for i in ids['always']]; maps = [byid[i] for i in ids['map']]
    dep = {'pinned_ids': ids['always'], 'retrieval_ids': ids['map'], 'k': 4, 'shots': 1, 'retrieval_n_exec': 4, 'method': 'bm25'}
    base = build_train_set(w, K, seed, n_exec=8, n_decoy=4, rule_filter=ids['range'], deploy_ctx=dep)
    seen = {d['item']['request'] for d in base if d.get('item')}
    if seed == 0:
        seen |= uniform_contrast_requests()
    rng = random.Random('reliability-v1-fresh' if seed == 0 else f'reliability-v1-fresh-s{seed}')
    fresh = []
    for r in [byid[i] for i in ids['range']]:
        for kind in ('exec', 'decoy'):
            for _ in range(4):
                for _ in range(10000):
                    it = make_item(w, r, 'train', kind, rng, K)
                    if it['request'] not in seen:
                        break
                else:
                    raise RuntimeError('cannot generate a new training request')
                seen.add(it['request'])
                fresh.append({'uid': uid(it), 'kind': kind, 'rule_id': r['id'], 'item': it,
                              'oracle': render_calls(it['answer']), 'source': 'fresh'})
    data = copy.deepcopy(base + fresh)
    items = [d['item'] for d in data if d.get('item')]
    _, contexts = deploy_contexts(items, pinned, maps, w, K)
    dev = dev_requests(w); index = 0
    for d in data:
        if d.get('item'):
            it = d['item']; assert it['split'] == 'train'
            assert score_call(d['oracle'], lookup_solve(w, K, it))['correct']
            assert it['request'] not in dev
            if 'fare' in it['fields']:
                assert it['fields']['fare'] % 5 in (0, 1, 2)
            d['messages'] = build_messages(it, contexts[index]); index += 1
        d.update(world_hash=w.hash(), K=K, seed=seed, rule_type='mixed')
    data.sort(key=lambda d: d['uid'])
    requests = [d['item']['request'] for d in data if d.get('item')]
    assert len(data) == 480 and len({d['uid'] for d in data}) == 480 and len(set(requests)) == len(requests)
    return data


# ---------------------------------------------------------------- update test items and practice
def update_test_items(w):
    """Standard test items + 16 update items per range rule (contexts are added by the caller)."""
    _, _, ranges = parts(w); codes = code_pools(w)
    items = [it for kind, n in STANDARD.items() for it in sample_items(w, K, 'test', kind, n)]
    rng = random.Random(f'update-v1-test-{w.hash()}')
    for r in ranges.values():
        for case, n in TEST_CASES.items():
            items += [make_update_item(w, r, 'test', case, rng, K, codes) for _ in range(n)]
    return items


def update_practice_rows(seed, base):
    """The 64 update-practice rows (practised rules r0-r7) without messages; labels follow the update."""
    w = build_world('mixed', seed); _, _, ranges = parts(w); codes = code_pools(w)
    test = update_test_items(w)
    held = {it['request'] for it in test} | dev_requests(w)
    seen = {d['item']['request'] for d in base if d.get('item')}
    assert not seen & held
    rng = random.Random(f'update-v1-train-{seed}'); out = []
    for rid in PRACTICED:
        r = ranges[rid]
        for arm, plan in (('plus_fresh', {'exec': 4, 'decoy': 4}), ('plus_update', TRAIN_CASES)):
            for case, n in plan.items():
                for _ in range(n):
                    for _ in range(10000):
                        it = (make_item(w, r, 'train', case, rng, K) if arm == 'plus_fresh'
                              else make_update_item(w, r, 'train', case, rng, K, codes, PRACTICED))
                        if it['request'] not in seen and it['request'] not in held:
                            break
                    else:
                        raise RuntimeError('cannot generate a new training request')
                    seen.add(it['request'])
                    if arm == 'plus_update':
                        oracle = render_calls(it['answer'])
                        assert it['split'] == 'train' and score_call(oracle, lookup_solve(w, K, it))['correct']
                        out.append({'uid': uid(it), 'kind': it['kind'], 'rule_id': rid, 'item': it, 'oracle': oracle,
                                    'source': 'plus_update', 'world_hash': w.hash(), 'K': K, 'seed': seed, 'rule_type': 'mixed'})
    return out
