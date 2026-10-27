"""Paired statistics over item-paired evaluation records."""
from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p, z2 = k / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z / (1 + z2 / n) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def acc_stats(key: str, k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {key: k / n if n else float("nan"), f"{key}_n": n, f"{key}_ci95_lo": lo, f"{key}_ci95_hi": hi}


def paired_ci(diffs, n_boot=10000, seed=0):
    """diffs[s] = per-item (a - b) differences of seed s. Seed-stratified item bootstrap of the
    mean over seeds; returns the 95% percentile interval (pure Python; training-recipe audits)."""
    rng = random.Random(seed); ms = []
    for _ in range(n_boot):
        ms.append(sum(sum(rng.choice(d) for _ in d) / len(d) for d in diffs) / len(diffs))
    ms.sort()
    return [ms[int(.025 * n_boot)], ms[int(.975 * n_boot) - 1]]


def boot(pairs, n_boot=2000, seed=0):
    """pairs: per seed (a_vec, b_vec). Point difference and seed-stratified paired bootstrap 95% CI
    (numpy; system-level audits)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    diffs = [np.asarray(av, float) - np.asarray(bv, float) for av, bv in pairs]
    point = float(np.mean([d.mean() for d in diffs]))
    stats = np.mean([d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1) for d in diffs], 0)
    lo, hi = np.quantile(stats, [.025, .975])
    return dict(diff=point, ci95=[float(lo), float(hi)])
