"""``spm-exp``: CPU-side experiment steps (data preparation, routing views, audits, re-scoring).

GPU steps are separate modules: ``python -m spm.training.lora`` (training), ``python -m
spm.eval.generate`` (evaluation), ``python -m spm.eval.ifeval`` (capability guard). The shell scripts in
``scripts/`` chain them. All paths are arguments; nothing machine-specific is hard-coded.

  spm-exp prepare        --root DATA          all frozen data: recipe arms + probe set, cons, closeout
                                              dev/confirm views, repair data, few-shot views
  spm-exp route          --root DATA          routed views (oracle / BM25 / dense) with dev-calibrated tau
  spm-exp capcpt-prepare --root DATA --prompt-file Final_Prompting.md
  spm-exp capcpt-gen     --root DATA --seed S --gen-model LLM            (GPU)
  spm-exp capcpt-assemble --root DATA --seed S --model BASE [--allow-fallback]
  spm-exp audit-recipe   --evals EVAL_DIR --out F         training-recipe table (probe set)
  spm-exp audit-dev      --evals EVAL_DIR --out F         dev-set selection of the runtime view (seed 0)
  spm-exp audit-closeout --evals EVAL_DIR --out F         system table + frozen confirmation gates
  spm-exp rescore        FILES...                           expression-aware re-scoring of eval.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import time
from pathlib import Path

from .data.world import DEFAULT_RULE_TYPE, build_world
from .io import read_jsonl, save_json, sha256_file, write_jsonl

RT = DEFAULT_RULE_TYPE
SEEDS = (0, 1, 2)
GROUPS = ('stable', 'changed', 'unrelated')


def view_path(base, view, seed, rt=RT):
    return Path(base) / view / 'data' / rt / f'test-s{seed}.jsonl'


# ---------------------------------------------------------------- prepare
def cmd_prepare(a):
    from .baselines.prompting import fewshot_views
    from .training.datasets import prepare_closeout, prepare_recipe, prepare_repair
    root = Path(a.root); t0 = time.time()
    info = {'recipe': prepare_recipe(root, a.rule_type, a.seeds)}
    info['closeout_dev'] = prepare_closeout(root, 'dev', a.rule_type, (0,))
    info['closeout_confirm'] = prepare_closeout(root, 'confirm', a.rule_type, a.seeds)
    info['repair'] = prepare_repair(root, a.rule_type, a.seeds)
    excluded = set()
    for ph in ('confirm', 'dev'):
        for p in sorted((root / 'closeout' / ph / 'views').glob(f'*/data/{a.rule_type}/test-s*.jsonl')):
            excluded.update(r['request'] for r in read_jsonl(p))
    base = root / 'closeout' / 'confirm' / 'views'; info['fewshot'] = {}
    for s in a.seeds:
        views, stats = fewshot_views(s, read_jsonl(view_path(base, 'full', s, a.rule_type)),
                                     read_jsonl(view_path(base, 'matched', s, a.rule_type)), excluded, a.rule_type)
        info['fewshot'][str(s)] = stats
        for name, rows in views.items():
            p = write_jsonl(view_path(base, name, s, a.rule_type), rows)
            info['fewshot'][str(s)][name] = sha256_file(p)
    save_json(root / 'prepare_info.json', info)
    print('PREPARED', root, f'{time.time() - t0:.1f}s')


# ---------------------------------------------------------------- routing
CONFIGS = [('bm25', 'explicit', 1), ('dense', 'explicit', 1), ('bm25', 'para', 1), ('dense', 'para', 1),
           ('bm25', 'para', 3), ('dense', 'para', 3), ('bm25', 'mask', 1), ('dense', 'mask', 1)]


def load_routing_items(base, seed, rt=RT):
    """Per item: request row, current policy sections, changed sections (version metadata), oracle closure."""
    from .runtime.context import split_doc, TASK_RE
    full = read_jsonl(view_path(base, 'full', seed, rt)); closed = read_jsonl(view_path(base, 'closed', seed, rt))
    assert len(full) == len(closed)
    head0, sec0 = split_doc(next(r['context'] for r in full if r['probe'] == 'stable'))
    items = []
    for i, (f, c) in enumerate(zip(full, closed)):
        assert f['request'] == c['request'] and f['probe'] == c['probe']
        head, secs = split_doc(f['context'])
        assert head == head0 and set(secs) == set(sec0)
        changed = sorted(k for k in secs if secs[k] != sec0[k])
        ks = {int(k) for k in TASK_RE.findall(f['request'])}
        assert len(ks) == 1 and ks == {f['fields']['task']}
        items.append(dict(i=i, row=c, task=f['fields']['task'], head=head, secs=secs, changed=changed,
                          full=f['context'], oracle=c['context']))
    return items


def cmd_route(a):
    from .runtime.routing import Dense, bm25_scores, calibrate, decide, rank, router_query
    root = Path(a.root); dev_base = root / 'closeout' / 'dev' / 'views'; conf_base = root / 'closeout' / 'confirm' / 'views'
    routers = ['bm25'] + (['dense'] if a.dense_model else [])
    dense = Dense(a.dense_model, a.threads) if a.dense_model else None

    def scores(items, router, variant, tag):
        qs = [router_query(it['row'], variant, f'sysprobe-v1-para-{tag}-{it["i"]}') for it in items]
        if router == 'dense':
            return qs, [dense.scores(q, it['secs']) for q, it in zip(qs, items)]
        return qs, [bm25_scores(q, it['secs']) for q, it in zip(qs, items)]
    dev = load_routing_items(dev_base, 0)
    result = {'tau': {}, 'dev': {}, 'confirm': {}, 'oracle_equal_checks': 0}
    for router in routers:
        for variant in ('explicit', 'para', 'mask'):
            _, sc = scores(dev, router, variant, 'dev-s0')
            rk = [rank(s) for s in sc]
            correct = [o[0] == it['task'] for (o, _), it in zip(rk, dev)]
            tau = calibrate([m for _, m in rk], correct, a.target_precision)
            result['tau'][f'{router}-{variant}'] = tau
            result['dev'][f'{router}-{variant}'] = {'top1_acc': sum(correct) / len(correct),
                                                   'fallback_rate': sum(m <= 0 or m < tau for _, m in rk) / len(rk)}
    for seed in a.seeds:
        items = load_routing_items(conf_base, seed)
        for router in routers:
            for variant in ('explicit', 'para', 'mask'):
                qs, sc = scores(items, router, variant, f'confirm-s{seed}')
                tau = result['tau'][f'{router}-{variant}']
                for (rr, vv, k) in CONFIGS:
                    if (rr, vv) != (router, variant):
                        continue
                    name = f'{rr}-{vv}-k{k}'; rows = []
                    stats = {g: collections.Counter() for g in GROUPS}
                    for it, q, s in zip(items, qs, sc):
                        order, margin = rank(s)
                        ctx, act = decide(it['head'], it['secs'], it['changed'], it['full'], order, margin, k, tau)
                        st = stats[it['row']['probe']]
                        st['n'] += 1; st['top1'] += order[0] == it['task']; st['topk'] += it['task'] in order[:k]; st[act] += 1
                        if k == 1 and act != 'fallback' and order[0] == it['task']:
                            assert ctx == it['oracle']  # correct routing reproduces the oracle closure byte-for-byte
                            result['oracle_equal_checks'] += 1
                        rows.append({**it['row'], 'context': ctx, 'route': {'q': q, 'top': order[:3], 'margin': margin, 'action': act}})
                    write_jsonl(view_path(root / 'routing' / 'views', name, seed), rows)
                    result['confirm'].setdefault(name, {})[str(seed)] = {g: dict(c) for g, c in stats.items()}
    save_json(root / 'routing' / 'routing.json', result)
    print('ROUTED', json.dumps(result['tau']))


# ---------------------------------------------------------------- CAP-CPT
def cmd_capcpt_prepare(a):
    from .baselines import capcpt
    root = Path(a.root); src = root / 'recipe' / 'data' / RT; d = root / 'capcpt' / 'data' / RT
    prompt_text = capcpt.read_prompt_file(a.prompt_file) if a.prompt_file else None
    man = {'seeds': {}}
    for seed in a.seeds:
        w = build_world(RT, seed)
        sets = capcpt.anchor_sets(w, read_jsonl(src / f'P-s{seed}.jsonl'), read_jsonl(src / f'prec-s{seed}.jsonl'))
        for name, rows in sets.items():
            write_jsonl(d / f'{name}-s{seed}.jsonl', rows)
        test = read_jsonl(src / f'test-s{seed}.jsonl')
        write_jsonl(root / 'capcpt' / 'pid' / 'data' / RT / f'test-s{seed}.jsonl',
                    [{**r, 'probe': 'none_pid', 'context': capcpt.anchor(w)} for r in test if r['probe'] == 'none'])
        specs = capcpt.spec_list(w); qa = capcpt.qa_list(w, specs)
        save_json(d / f'specs-s{seed}.json', {'specs': specs, 'qa': qa, 'policy_id': w.extra['policy_id'], 'world_hash': w.hash()})
        if prompt_text:
            write_jsonl(root / 'capcpt' / 'gen' / f'prompts-s{seed}.jsonl', capcpt.prompts_for(w, specs, qa, prompt_text))
        man['seeds'][str(seed)] = {n: {'sha256': sha256_file(d / f'{n}-s{seed}.jsonl')} for n in sets}
    save_json(d / 'manifest.json', man)
    print('CAPCPT_PREPARED', d)


def cmd_capcpt_gen(a):
    from .baselines import capcpt
    out = Path(a.root) / 'capcpt' / 'gen' / f'raw-s{a.seed}.jsonl'
    if out.exists():
        print('exists', out); return
    prompts = read_jsonl(Path(a.root) / 'capcpt' / 'gen' / f'prompts-s{a.seed}.jsonl')
    write_jsonl(out, capcpt.generate_raw(prompts, a.gen_model, a.seed))


def cmd_capcpt_assemble(a):
    from .baselines import capcpt
    root = Path(a.root); d = root / 'capcpt' / 'data' / RT; w = build_world(RT, a.seed)
    spec = json.loads((d / f'specs-s{a.seed}.json').read_text())
    raw_path = root / 'capcpt' / 'gen' / f'raw-s{a.seed}.jsonl'
    if raw_path.exists():
        raw = {r['pid']: r for r in read_jsonl(raw_path)}
    elif a.allow_fallback:
        raw = {}
    else:
        raise SystemExit(f'missing LLM outputs: {raw_path} (run capcpt-gen, or --allow-fallback for a dry run)')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    render = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full, nosim, rep = capcpt.assemble(w, spec['specs'], [tuple(x) for x in spec['qa']], raw,
                                       read_jsonl(root / 'recipe' / 'data' / RT / f'test-s{a.seed}.jsonl'),
                                       read_jsonl(d / f'P_anchor-s{a.seed}.jsonl'), render, a.n_scenario, a.n_behavior)
    for name, docs in (('cpt_full', full), ('cpt_nosim', nosim)):
        p = write_jsonl(d / f'{name}-s{a.seed}.jsonl', docs); rep[name.split('_')[1]]['sha256'] = sha256_file(p)
    rep['raw_sha256'] = sha256_file(raw_path) if raw else None
    save_json(d / f'cpt_report-s{a.seed}.json', rep)
    print('CAPCPT_ASSEMBLED', json.dumps({k: rep[k] for k in ('full', 'nosim')}))


# ---------------------------------------------------------------- audits
def _cell(evals, table, view, arm, seed):
    """Evaluation cell ``<evals>/<table>/<view>/<arm>-s<seed>`` (metrics + re-scored records)."""
    d = Path(evals) / table / view / f'{arm}-s{seed}'
    if not (d / 'metrics.json').exists():
        return None, None
    rows = read_jsonl(d / 'eval.jsonl')
    from .eval.score import check_scores
    check_scores(rows)
    return {r['probe']: r for r in json.loads((d / 'metrics.json').read_text())['rows']}, rows


def cmd_audit_recipe(a):
    """Training-recipe table on the probe set + prec - cons paired bootstrap (seed-stratified)."""
    from .eval.stats import paired_ci
    arms = ('base', 'P', 'ctrl', 'M', 'cons', 'prec') + tuple(a.extra_arms or ())
    res = {'cells': {}, 'missing': [], 'means': {}, 'comparisons': {}}
    recs = {}
    for arm in arms:
        for s in a.seeds:
            m, rows = _cell(a.evals, 'recipe', 'probe', arm, s)
            if m is None:
                res['missing'].append(f'{arm}-s{s}'); continue
            res['cells'][f'{arm}-s{s}'] = m; recs[arm, s] = rows
    for arm in arms:
        if all((arm, s) in recs for s in a.seeds):
            res['means'][arm] = {p: sum(res['cells'][f'{arm}-s{s}'][p]['sr'] for s in a.seeds) / len(a.seeds)
                                 for p in res['cells'][f'{arm}-s{a.seeds[0]}']}
    for x, y in (('prec', 'cons'), ('prec', 'P')):
        if all((arm, s) in recs for arm in (x, y) for s in a.seeds):
            out = {}
            for p in ('override_held', 'substitute', 'none'):
                diffs = []
                for s in a.seeds:
                    A = [r for r in recs[x, s] if r['probe'] == p]; B = [r for r in recs[y, s] if r['probe'] == p]
                    assert [r['request'] for r in A] == [r['request'] for r in B], 'items not paired'
                    diffs.append([int(u['correct']) - int(v['correct']) for u, v in zip(A, B)])
                out[p] = {'diff': sum(sum(d) / len(d) for d in diffs) / len(diffs), 'ci95': paired_ci(diffs),
                          'per_seed': [sum(d) / len(d) for d in diffs]}
            res['comparisons'][f'{x}-{y}'] = out
    save_json(Path(a.out), res)
    print(json.dumps({'missing': res['missing'], 'means': res['means']}, indent=1))


def cmd_audit_closeout(a):
    """System table: selected system vs baselines on the confirmation set, frozen gates, paired diffs."""
    from .eval.closeout import confirm_checks
    from .eval.stats import boot
    cells = [tuple(c.split(':')) for c in a.cells]
    res = {'cells': {}, 'missing': [], 'checks': [], 'comparisons': {}}
    vec = {}
    for arm, view in cells:
        for s in a.seeds:
            m, rows = _cell(a.evals, 'closeout', view, arm, s)
            if m is None:
                res['missing'].append(f'{arm}:{view}:s{s}'); continue
            res['cells'][f'{arm}:{view}:s{s}'] = {g: {k: m[g][k] for k in ('n', 'sr', 'truncation', 'prompt_tokens')} for g in m}
            vec[arm, view, s] = {g: [int(r['correct']) for r in rows if r['probe'] == g] for g in GROUPS}
    sysarm, sysview = a.system.split(':')
    for s in a.seeds:
        need = [f'{sysarm}:{sysview}:s{s}', f'prec:full:s{s}', f'base:matched:s{s}', f'base:full:s{s}']
        if all(n in res['cells'] for n in need):
            c = res['cells']
            res['checks'].append({'seed': s, **confirm_checks(c[need[0]], c[need[1]], c[need[2]], c[need[3]])})
    for arm, view in cells:
        if (arm, view) == (sysarm, sysview) or not all((x, v, s) in vec for x, v in ((sysarm, sysview), (arm, view)) for s in a.seeds):
            continue
        res['comparisons'][f'{a.system} - {arm}:{view}'] = {
            g: {'pooled': boot([(vec[sysarm, sysview, s][g], vec[arm, view, s][g]) for s in a.seeds], seed=99),
                'per_seed': {f's{s}': boot([(vec[sysarm, sysview, s][g], vec[arm, view, s][g])], seed=s) for s in a.seeds}}
            for g in GROUPS}
    res['gate'] = bool(res['checks']) and len(res['checks']) == len(a.seeds) and all(c['passes'] for c in res['checks'])
    save_json(Path(a.out), res)
    print(json.dumps({'missing': res['missing'], 'gate': res['gate'], 'checks': res['checks']}, indent=1))


def cmd_audit_dev(a):
    """Dev-set selection (seed 0): which runtime candidates pass the frozen gates against prec:full."""
    from .eval.closeout import dev_gate
    ref, _ = _cell(a.evals, a.table, 'full', 'prec', 0)
    if ref is None:
        raise SystemExit('missing prec:full dev cell')
    res = {'candidates': {}}
    for cell in a.cells:
        arm, view = cell.split(':')
        m, _ = _cell(a.evals, a.table, view, arm, 0)
        if m is not None:
            res['candidates'][cell] = {'passes': dev_gate(m, ref), 'changed_sr': m['changed']['sr'],
                                       'changed_prompt_tokens': m['changed']['prompt_tokens']}
    passing = [c for c, v in res['candidates'].items() if v['passes']]
    res['selected'] = min(passing, key=lambda c: res['candidates'][c]['changed_prompt_tokens']) if passing else None
    save_json(Path(a.out), res)
    print(json.dumps(res, indent=1))


def cmd_rescore(a):
    from .eval.score import calls_equal, normalize_lenient, parse_calls_eval
    res = {}
    for pat in a.files:
        for p in sorted(glob.glob(pat)):
            by = {}
            for r in read_jsonl(p):
                g = r.get('probe') or 'all'; orig = bool(r.get('correct'))
                new = (not r.get('truncated')) and calls_equal(parse_calls_eval(normalize_lenient(r['output'])), r['answer'])
                d = by.setdefault(g, [0, 0, 0, 0]); d[0] += 1; d[1] += orig; d[2] += new or orig; d[3] += (new and not orig)
            res[p] = {g: {'n': d[0], 'sr': d[1] / d[0], 'sr_expr': d[2] / d[0], 'gained': d[3]} for g, d in by.items()}
    if a.out:
        save_json(Path(a.out), res)
    for k, v in res.items():
        print(k, {g: (round(x['sr'] * 100, 1), round(x['sr_expr'] * 100, 1)) for g, x in v.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(prog='spm-exp', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('prepare'); p.add_argument('--root', required=True); p.add_argument('--rule-type', default=RT)
    p.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    p = sp.add_parser('route'); p.add_argument('--root', required=True); p.add_argument('--dense-model')
    p.add_argument('--threads', type=int, default=8); p.add_argument('--target-precision', type=float, default=0.99)
    p.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    p = sp.add_parser('capcpt-prepare'); p.add_argument('--root', required=True); p.add_argument('--prompt-file')
    p.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    p = sp.add_parser('capcpt-gen'); p.add_argument('--root', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gen-model', required=True)
    p = sp.add_parser('capcpt-assemble'); p.add_argument('--root', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--model', required=True); p.add_argument('--allow-fallback', action='store_true')
    p.add_argument('--n-scenario', type=int, default=5000); p.add_argument('--n-behavior', type=int, default=1000)
    p = sp.add_parser('audit-recipe'); p.add_argument('--evals', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS)); p.add_argument('--extra-arms', nargs='*')
    p = sp.add_parser('audit-closeout'); p.add_argument('--evals', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    p.add_argument('--system', default='repair:closed')
    p.add_argument('--cells', nargs='*', default=['repair:closed', 'P:closed', 'cons:closed', 'prec:full', 'base:matched', 'base:full'])
    p = sp.add_parser('audit-dev'); p.add_argument('--evals', required=True); p.add_argument('--out', required=True)
    p.add_argument('--table', default='closeout-dev')
    p.add_argument('--cells', nargs='*', default=['prec:closed', 'prec:typed', 'repair:closed', 'consistent:closed'])
    p = sp.add_parser('rescore'); p.add_argument('files', nargs='+'); p.add_argument('--out')
    a = ap.parse_args(argv)
    {'prepare': cmd_prepare, 'route': cmd_route, 'capcpt-prepare': cmd_capcpt_prepare, 'capcpt-gen': cmd_capcpt_gen,
     'capcpt-assemble': cmd_capcpt_assemble, 'audit-recipe': cmd_audit_recipe, 'audit-closeout': cmd_audit_closeout,
     'audit-dev': cmd_audit_dev, 'rescore': cmd_rescore}[a.cmd](a)


if __name__ == '__main__':
    main()
