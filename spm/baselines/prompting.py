"""Prompting baselines for an untrained model.

* full     the entire current policy in the system prompt (``closeout`` view ``full``);
* matched  oracle-matched section: shared constants + the request's own current task section
           (view ``matched``; same task ID and dependency information as the system's closure);
* few-shot ``full_fs3`` / ``matched_fs3``: the same views plus 3 worked tool-call examples given as
           multi-turn chat (user example, assistant answer), demonstrating the exact output format.

Few-shot leakage guards (all asserted):
  L1 examples come from the train split (templates and value classes disjoint from dev/test);
  L2 example requests never occur in any dev or confirmation request;
  L3 example tasks are neither the tested task nor any task changed in the item's current version;
  L4 example answers are program-computed and the example's section and constants are unchanged in
     the item's current policy (so they are correct under the context shown);
  L5 both views of an item share the same examples; the RNG depends only on (seed, index, request).
"""
from __future__ import annotations

import collections
import random

from ..data.ccgen import tasks
from ..data.prompts import build_messages, user_text
from ..data.world import build_world, lookup_solve, make_item
from ..eval.score import render_calls, score_call
from ..runtime.context import split_doc

K_SHOTS = 3


def fewshot_views(seed, full_rows, matched_rows, excluded, rt="ccgen_t12a3s1_lite", k_shots=K_SHOTS):
    """Return ({'full_fs3': rows, 'matched_fs3': rows}, stats). ``excluded`` = all dev/confirm requests."""
    w = build_world(rt, seed); T = w.extra['T']; ts = tasks(w)
    assert len(full_rows) == len(matched_rows)
    orig = next(r['context'] for r in full_rows if r['probe'] == 'stable')
    head0, sec0 = split_doc(orig)
    out = {'full': [], 'matched': []}; stats = collections.Counter(); task_hist = collections.Counter()
    for i, (f, m) in enumerate(zip(full_rows, matched_rows)):
        assert f['request'] == m['request'] and f['probe'] == m['probe']
        head, secs = split_doc(f['context'])
        assert head == head0 and set(secs) == set(sec0)
        changed = {k for k in secs if secs[k] != sec0[k]}
        k = f['fields']['task']
        rng = random.Random(f"fewshot-v1-{seed}-{i}-{f['request']}")
        pool = [j for j in range(1, T + 1) if j != k and j not in changed]
        chosen = rng.sample(pool, k_shots)
        shots = []; used = set()
        for j in chosen:
            for _ in range(100):
                ex = make_item(w, ts[j - 1], 'train', 'exec', rng, T)
                if ex['request'] not in excluded and ex['request'] not in used:
                    break
                stats['resampled'] += 1
            else:
                raise RuntimeError('example pool exhausted')
            used.add(ex['request'])
            assert ex['split'] == 'train' and ex['template_id'].startswith('train'), 'L1'
            assert ex['request'] not in excluded, 'L2'
            assert ex['fields']['task'] == j and j != k and j not in changed, 'L3'
            assert lookup_solve(w, T, ex) == ex['answer'] and secs[j] == sec0[j], 'L4'
            ans = render_calls(ex['answer'])
            assert score_call(ans, ex['answer'])['correct']
            shots += [{'role': 'user', 'content': user_text(ex)}, {'role': 'assistant', 'content': ans}]
            task_hist[j] += 1
        stats['items'] += 1
        for view, row in (('full', f), ('matched', m)):
            out[view].append({**row, 'shots': shots, 'shot_tasks': chosen})
    return {f'{v}_fs{k_shots}': rows for v, rows in out.items()}, {'stats': dict(stats), 'shot_task_hist': dict(sorted(task_hist.items()))}


def messages(row: dict) -> list[dict]:
    """Chat messages for an evaluation row (system + optional few-shot turns + user)."""
    m = build_messages(row, row.get('context'))
    if row.get('shots'):
        m = [m[0], *row['shots'], m[1]]
    return m
