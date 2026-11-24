#!/usr/bin/env python
"""
kep_dist.py

Distribution of per-transplant true graft survival years for the two
optimisation objectives (max transplants vs max predicted survival), at
S = 0, K = 3, aggregated across all 60 dpi instances.

The point of this figure is to make explicit *what* the max-survival objective
does to the pool of transplants delivered: it shifts the entire distribution
towards higher-quality arcs, not just the mean.  Rows are noise levels α.

Output
------
  kep_dist_hist.png
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

from kep_io import read_kep
from kep_cycle import get_all_cycles, kep_cycle
from ml_eval import read_hist, read_pool, build_pool_X, make_pipeline

DATA_DIR    = 'DATA'
RESULTS_DIR = 'RESULTS'
K           = 3
S           = 0                    # no filter — pure objective comparison

os.makedirs(RESULTS_DIR, exist_ok=True)
pattern = re.compile(r'kep-surv_(\d+\.\d+)_(\w+)_(\d+)\.hist\.gz')


def cycle_arcs(cycle):
    n = len(cycle)
    return [(cycle[i], cycle[(i + 1) % n]) for i in range(n)]


def collect_arcs(base):
    """Return (per_arc_true_max_tx, per_arc_true_max_surv) for one instance."""
    X_hist, y_hist = read_hist(base + '.hist.gz')
    pool_df        = read_pool(base + '.pool.gz')
    adj, _         = read_kep(base + '.kep.gz')
    _, hidd        = read_kep(base + '.hidd.gz')

    X_pool, pairs = build_pool_X(pool_df, adj)
    pipe = make_pipeline(HistGradientBoostingRegressor(random_state=0))
    pipe.fit(X_hist, y_hist)
    y_pred = np.maximum(pipe.predict(X_pool), 0.0)  # clip: negative predicted survival is nonsensical
    pred_surv = {(d, p): float(yp) for (d, p), yp in zip(pairs, y_pred)}

    cycles = list(get_all_cycles(adj, K))
    if not cycles:
        return [], []

    cycle_pred_sv = [
        sum(pred_surv.get(a, 0.0) for a in cycle_arcs(c)) for c in cycles
    ]

    def arcs_of(sol):
        return [a for c in sol for a in cycle_arcs(c)]

    _, sol_a = kep_cycle(adj, cycles)                          # max transplants
    _, sol_b = kep_cycle(adj, cycles, weights=cycle_pred_sv)   # max predicted survival

    true_a = [hidd.get(a, 0.0) for a in arcs_of(sol_a)]
    true_b = [hidd.get(a, 0.0) for a in arcs_of(sol_b)]
    return true_a, true_b


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if (m := pattern.match(f)) and m.group(2) == 'dpi')

    per_alpha = {}    # alpha -> dict(tx=[..], surv=[..])

    for file_idx, fname in enumerate(files, 1):
        m       = pattern.match(fname)
        alpha   = float(m.group(1))
        inst    = int(m.group(3))
        base    = os.path.join(DATA_DIR, fname[:-len('.hist.gz')])

        print(f"[{file_idx:3d}/{len(files)}]  alpha={alpha}  inst={inst:02d}",
              flush=True)
        true_a, true_b = collect_arcs(base)

        bucket = per_alpha.setdefault(alpha, dict(tx=[], surv=[]))
        bucket['tx'].extend(true_a)
        bucket['surv'].extend(true_b)

    # ── plot: rows = alpha, one histogram per row overlaying the two solutions ─
    alphas = sorted(per_alpha)
    fig, axes = plt.subplots(len(alphas), 1, figsize=(7, 1.8 * len(alphas)),
                             sharex=True)
    if len(alphas) == 1:
        axes = [axes]

    bins = np.linspace(0, 25, 26)

    for ax, a in zip(axes, alphas):
        tx   = np.asarray(per_alpha[a]['tx'])
        surv = np.asarray(per_alpha[a]['surv'])
        ax.hist(tx,   bins=bins, color='steelblue', alpha=0.55,
                label=('max transplants\n'
                       f'  avg. per-arc surv: {tx.mean():.1f} yr\n'
                       f'  # transplants: {len(tx)}'))
        ax.hist(surv, bins=bins, color='darkorange', alpha=0.55,
                label=('max pred. survival\n'
                       f'  avg. per-arc surv: {surv.mean():.1f} yr\n'
                       f'  # transplants: {len(surv)}'))
        ax.set_ylabel(fr'$\alpha={a}$', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)

    axes[-1].set_xlabel('True graft survival per transplant (years)')
    fig.suptitle('Distribution of per-transplant true survival, S = 0, K = 3\n'
                 '(pooled across 10 instances per α)', y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'kep_dist_hist.png'),
                dpi=300, bbox_inches='tight')
    print(f"Plot saved to {RESULTS_DIR}/kep_dist_hist.png")
    plt.close(fig)
