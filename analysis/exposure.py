"""E7/E8 exposure mechanism (synthetic ``mixed`` world, K=16, seeds 0-2).

Question: does the way a rule family is exposed in the training prompt decide whether the model later
relies on parametric memory or on the context? Same 480 training rows and labels (``fresh_data``);
only the rule lists shown in the training prompt differ:
  W  neither mapping nor range rules in context (only the always-on rules);
  C  both families' full lists in context;
  X  mapping not in context, range in context;
  M  each family shown with probability 0.5 per row;
  R  (E8) mapping list in context, range never.

Test set per seed (same file for every arm and the base model):
  own_none      only the always-on rules in context (memory);
  own_full      both full lists in context (consistent);
  own_conflict  full lists with the tested rule replaced by a new version (answer follows context);
  new_full      a new world (seed 3) with its full lists (transfer; contamination by source names).

    python -m analysis.exposure prepare --root OUT
    python -m analysis.exposure audit   --root OUT --evals EVAL_DIR
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from spm.data.prompts import build_messages
from spm.data.world import build_world, sample_items
from spm.io import read_jsonl, save_json, sha256_file, write_jsonl

from .synth_data import K, SEEDS, STANDARD, code_pools, fresh_data, update_test_items

NEW = 3
ARMS = {'W': ('w', 'w'), 'C': ('ctx', 'ctx'), 'X': ('w', 'ctx'), 'M': ('mix', 'mix')}  # (mapping, range)
TOOL = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(')
PROMO = re.compile(r'promo\s*=\s*"([^"]*)"')


def block(sub, include, override=None):
    """Always-on rules + the selected families' full lists; ``override`` = {rule_id: new text}."""
    override = override or {}
    lines = ['Rules:'] + ['- ' + override.get(r['id'], r['text']) for r in sub if r['kind'] == 'always' or
                          (r['kind'] == 'map' and include.get('map')) or (r['kind'] == 'range' and include.get('range'))]
    return '\n'.join(lines)


def arm_rows(seed, base, arm):
    sub = build_world('mixed', seed).subset(K); data = []
    if arm == 'R':
        for d in base:
            d = dict(d)
            if d.get('item'):
                d['messages'] = build_messages(d['item'], block(sub, {'map': 1})); d['source'] = 'exposure-R'
            data.append(d)
        return data
    em, er = ARMS[arm]
    rng = random.Random(f'exposure-v1-{arm}-{seed}')
    for d in base:
        d = dict(d)
        if d.get('item'):
            inc = {'map': em == 'ctx' or (em == 'mix' and rng.random() < .5), 'range': er == 'ctx' or (er == 'mix' and rng.random() < .5)}
            d['messages'] = build_messages(d['item'], block(sub, inc)); d['source'] = f'exposure-{arm}'
        data.append(d)
    return data


def new_world_rows():
    nw = build_world('mixed', NEW); nsub = nw.subset(K)
    items = [it for kind, n in STANDARD.items() for it in sample_items(nw, K, 'test', kind, n)]
    rows = [{**it, 'condition': 'full', 'context': block(nsub, {'map': 1, 'range': 1})} for it in items]
    for r in rows:
        r['probe'] = 'new_full'
    return rows


def test_rows(seed, new_rows):
    w = build_world('mixed', seed); sub = w.subset(K); byid = {r['id']: r for r in sub}
    std = [it for it in update_test_items(w) if it['kind'] != 'update']
    rows = []
    for it in std:
        rows.append({**it, 'probe': 'own_none', 'context': block(sub, {})})
        rows.append({**it, 'probe': 'own_full', 'context': block(sub, {'map': 1, 'range': 1})})
    codes = code_pools(w); rng = random.Random(f'exposure-v1-conflict-{seed}')
    for fam in ('mapping', 'range'):
        pool = [it for it in std if it['kind'] == 'exec' and it['fields'].get('family') == fam]
        for it in rng.sample(pool, 96):
            it = {k: v for k, v in it.items() if k != 'context'}; ans = json.loads(json.dumps(it['answer']))
            rid = next(i for i in it['rule_ids'] if byid[i]['kind'] == ('map' if fam == 'mapping' else 'range')); r = byid[rid]
            new = rng.choice(codes['test'])
            if fam == 'mapping':
                newname = f"{new}_{r['verb']}"; text = f"`{r['old']}` has been renamed to `{newname}`; always call the new name."; ans[0][0] = newname
            else:
                text = f'When the fare is between {r["lo"]} and {r["hi"]}, add promo="{new}" to the call.'; ans[0][1]['promo'] = new
            rows.append({**it, 'answer': ans, 'probe': 'own_conflict', 'conflict': {'rule_id': rid, 'old': r['new'] if fam == 'mapping' else r['code'],
                         'new': newname if fam == 'mapping' else new}, 'context': block(sub, {'map': 1, 'range': 1}, {rid: text})})
    rows += [{**r, 'source_world': seed} for r in new_rows]
    return rows


def prepare(root, seeds=SEEDS):
    root = Path(root); manifest = {}; new_rows = new_world_rows()
    for seed in seeds:
        base = fresh_data(seed)
        manifest[str(seed)] = {'fresh': {'sha256': sha256_file(write_jsonl(root / 'data' / f'fresh-s{seed}.jsonl', base))}}
        for arm in (*ARMS, 'R'):
            p = write_jsonl(root / 'data' / f'{arm}-s{seed}.jsonl', arm_rows(seed, base, arm))
            manifest[str(seed)][arm] = {'sha256': sha256_file(p), 'n': len(base)}
        rows = test_rows(seed, new_rows)
        p = write_jsonl(root / 'data' / f'test-s{seed}.jsonl', rows)
        manifest[str(seed)]['test'] = {'sha256': sha256_file(p), 'groups': dict(collections.Counter(r['probe'] + '/' + r['kind'] for r in rows))}
    save_json(root / 'data' / 'manifest.json', manifest)
    return manifest


# ---------------------------------------------------------------- scoring helpers
def pick(out, fam):
    """Source-selection metric: which name (mapping) or promo code (range) the output chose."""
    if fam == 'mapping':
        m = TOOL.match(out); return m.group(1) if m else None
    m = PROMO.search(out); return m.group(1) if m else None


def target(ans, fam):
    return ans[0][0] if fam == 'mapping' else ans[0][1].get('promo')


def source_metrics(rows, seed):
    """Format-independent memory / conflict-following rates per family (new / old / none / other)."""
    sub = build_world('mixed', seed).subset(K); out = {}
    own_names = {x['new'] for x in sub if x['kind'] == 'map'}; own_codes = {x['code'] for x in sub if x['kind'] == 'range'}
    for fam in ('mapping', 'range'):
        mem = [r for r in rows if r['probe'] == 'own_none' and r['kind'] == 'exec' and r['fields'].get('family') == fam]
        conf = [r for r in rows if r['probe'] == 'own_conflict' and r['fields'].get('family') == fam]
        cats = collections.Counter()
        for r in conf:
            c = pick(r['output'], fam); n_, o = r['conflict']['new'], r['conflict']['old']
            cats['new' if c == n_ else 'old' if c == o else 'none' if (c is None or (fam == 'mapping' and c == r['fields']['tool']))
                 else 'other_mem' if c in (own_names if fam == 'mapping' else own_codes) else 'other'] += 1
        out[fam] = {'memory_src': sum(pick(r['output'], fam) == target(r['answer'], fam) for r in mem) / len(mem),
                    **{f'conf_{k}': cats[k] / len(conf) for k in ('new', 'old', 'none', 'other_mem', 'other')}}
    return out


def summarize(records):
    groups = collections.defaultdict(list)
    for r in records:
        groups[r['probe'], r['kind'], r['fields'].get('family', 'neutral')].append(r)
    out = []
    for (p, k, f), rs in sorted(groups.items()):
        row = {'probe': p, 'kind': k, 'family': f, 'n': len(rs), 'accuracy': sum(r['correct'] for r in rs) / len(rs)}
        out.append(row)
    return out


def audit(root, evals, seeds=SEEDS):
    """Per-arm memory / conflict-follow means and the frozen H7 readings (exposure decides the source)."""
    res = {'cells': {}, 'missing': []}
    for arm in (*ARMS, 'R'):
        vals = []
        for s in seeds:
            p = Path(evals) / f'{arm}-s{s}' / 'eval.jsonl'
            if not p.exists():
                res['missing'].append(f'{arm}-s{s}'); continue
            vals.append(source_metrics(read_jsonl(p), s))
        if len(vals) == len(seeds):
            res['cells'][arm] = {fam: {k: sum(v[fam][k] for v in vals) / len(vals) for k in vals[0][fam]} for fam in ('mapping', 'range')}
    c = res['cells']
    if all(arm in c for arm in ARMS):
        ex = {fam: {'ctx': [c[a][fam] for a in ARMS if ARMS[a][i] == 'ctx'], 'w': [c[a][fam] for a in ARMS if ARMS[a][i] == 'w']}
              for i, fam in enumerate(('mapping', 'range'))}
        avg = lambda xs, k: sum(x[k] for x in xs) / len(xs)
        res['H7a'] = {fam: {'follow_ctx_minus_w': avg(ex[fam]['ctx'], 'conf_new') - avg(ex[fam]['w'], 'conf_new'),
                            'memory_w_minus_ctx': avg(ex[fam]['w'], 'memory_src') - avg(ex[fam]['ctx'], 'memory_src')} for fam in ex}
    save_json(Path(root) / 'audit.json', res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['prepare', 'audit']); ap.add_argument('--root', required=True)
    ap.add_argument('--evals'); ap.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    a = ap.parse_args(argv)
    if a.mode == 'prepare':
        print(json.dumps(prepare(a.root, a.seeds))[:800])
    else:
        print(json.dumps(audit(a.root, a.evals, a.seeds), indent=1)[:3000])


if __name__ == '__main__':
    main()
