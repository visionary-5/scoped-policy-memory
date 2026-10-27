"""Frozen data construction for every arm of the paper, written in a fixed directory layout.

Layout under ``<root>`` (all paths relative, nothing machine-specific):
  recipe/data/<rt>/{P,M,prec,ctrl,cons}-s<seed>.jsonl   training arms (800-step SFT from the base model)
  recipe/data/<rt>/test-s<seed>.jsonl                   probe set (spm.eval.probes)
  recipe/data/<rt>/manifest.json                        SHA-256 of P/M/prec/ctrl/test per seed
  closeout/{dev,confirm}/views/<view>/data/<rt>/test-s<seed>.jsonl   system evaluation views
  closeout/repair/data/{repair,consistent}-s<seed>.jsonl              120-step continuation data

Generation is deterministic: the same ``(rule_type, seed)`` always yields byte-identical files. All RNG
seed strings are kept from the original experiments (e.g. ``ccgen-v1-train-...``) so that the files
match the SHA-256 recorded in the frozen protocols (``tests/golden/frozen_sha256.json``).
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from ..baselines.sft import cons_from, mixed_exposure_rows, ordinary_rows
from ..data.recipe import build_train_set
from ..data.world import DEFAULT_RULE_TYPE, build_world
from ..eval.closeout import build_rows, render_views
from ..eval.probes import build_probe_rows
from ..io import read_jsonl, sha256_file, write_jsonl
from .conflict import override_rows, repair_rows, substitute_rows
from .rows import fresh_sampler

SEEDS = (0, 1, 2)
N_EXEC, N_DECOY = 32, 4


def build_arms(rt: str, seed: int) -> dict:
    """P / M / prec / ctrl for one seed (rows sorted by uid). prec = P + Override + Substitute practice."""
    w = build_world(rt, seed); T = w.extra['T']
    base = build_train_set(w, T, seed, n_exec=N_EXEC, n_decoy=N_DECOY)
    for d in base:
        d.update(world_hash=w.hash(), K=T, seed=seed, rule_type=rt)
    seen = {d['item']['request'] for d in base if d.get('item')}
    rng = random.Random(f'ccgen-v1-train-{rt}-{seed}')
    fresh = fresh_sampler(seen)
    over = override_rows(w, rng, fresh)        # the three groups share one RNG stream and one
    subs = substitute_rows(w, rng, fresh)      # de-duplication set; their order is part of the
    ordin = ordinary_rows(w, rng, fresh)       # reproducibility contract
    arms = {'P': base, 'M': mixed_exposure_rows(w, base, seed), 'prec': base + over + subs, 'ctrl': base + ordin}
    out = {}
    for arm, data in arms.items():
        data = sorted(data, key=lambda d: d['uid'])
        assert len({d['uid'] for d in data}) == len(data)
        out[arm] = data
    return out


def _dump(path, obj):
    """Manifest format of the original experiments (indent 2, no trailing newline)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)); tmp.replace(path)


def prepare_recipe(root, rt: str = DEFAULT_RULE_TYPE, seeds=SEEDS, with_cons: bool = True) -> dict:
    d = Path(root) / 'recipe' / 'data' / rt; d.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for seed in seeds:
        arms = build_arms(rt, seed)
        manifest[str(seed)] = {}
        for arm, data in arms.items():
            p = write_jsonl(d / f'{arm}-s{seed}.jsonl', data)
            manifest[str(seed)][arm] = {'sha256': sha256_file(p), 'n': len(data)}
        rows, info = build_probe_rows(build_world(rt, seed))
        p = write_jsonl(d / f'test-s{seed}.jsonl', rows)
        manifest[str(seed)]['test'] = {'sha256': sha256_file(p), **info}
        if with_cons:
            cons = cons_from(build_world(rt, seed), arms['P'], arms['prec'])
            write_jsonl(d / f'cons-s{seed}.jsonl', cons)
    _dump(d / 'manifest.json', manifest)
    return manifest


def _recipe_requests(root, rt, seed):
    d = Path(root) / 'recipe' / 'data' / rt
    out = set()
    for name in ('P', 'M', 'prec', 'ctrl', 'test'):
        out.update((r.get('item') or r).get('request') for r in read_jsonl(d / f'{name}-s{seed}.jsonl'))
    return out


def prepare_closeout(root, phase: str, rt: str = DEFAULT_RULE_TYPE, seeds=None) -> dict:
    """Write all context views of the dev (seed 0) or confirmation (seeds 0-2) sets.
    Requires ``prepare_recipe`` (and, for 'confirm', the dev phase) to have run under ``root``."""
    root = Path(root); seeds = seeds or ((0,) if phase == 'dev' else SEEDS)
    out = root / 'closeout' / phase; info = {}
    for seed in seeds:
        w = build_world(rt, seed)
        excluded = _recipe_requests(root, rt, seed)
        if phase != 'dev':
            dp = root / 'closeout' / 'dev' / 'views' / 'full' / 'data' / rt / f'test-s{seed}.jsonl'
            if dp.exists():
                excluded.update(r['request'] for r in read_jsonl(dp))
        rows, worlds = build_rows(w, phase, excluded)
        info[str(seed)] = {}
        for view, vr in render_views(w, rows, worlds).items():
            p = write_jsonl(out / 'views' / view / 'data' / rt / f'test-s{seed}.jsonl', vr)
            info[str(seed)][view] = {'sha256': sha256_file(p), 'n': len(vr),
                                     'groups': dict(collections.Counter(r['probe'] for r in vr))}
    _dump(out / 'data_manifest.json', info)
    return info


def prepare_repair(root, rt: str = DEFAULT_RULE_TYPE, seeds=SEEDS) -> dict:
    root = Path(root); info = {}
    for seed in seeds:
        w = build_world(rt, seed)
        original = read_jsonl(root / 'recipe' / 'data' / rt / f'prec-s{seed}.jsonl')
        extra = repair_rows(w, original)
        info[str(seed)] = {}
        for arm in extra:
            data = sorted(original + extra[arm], key=lambda d: d['uid'])
            assert len({d['uid'] for d in data}) == len(data)
            p = write_jsonl(root / 'closeout' / 'repair' / 'data' / f'{arm}-s{seed}.jsonl', data)
            info[str(seed)][arm] = {'sha256': sha256_file(p), 'rows': len(data)}
    _dump(root / 'closeout' / 'repair' / 'manifest.json', info)
    return info
