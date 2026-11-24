#!/usr/bin/env python
"""
kep_solve.py

Full KEP optimisation pipeline with MST (minimum survival time) filter, for
both K = 2 and K = 3 maximum cycle lengths.

For every instance in DATA/:
  1. Trains an HGB model on .hist.gz historical data.
  2. Predicts survival for all compatible pool pairs.
  3. For each K in {2, 3} and each MST threshold S in {0, 1, ..., 18}:
       - Retains only arcs (d → p) with predicted_survival(d, p) >= S.
       - Enumerates all cycles up to length K in the filtered graph.
       - Solves objective (a): maximise number of transplants.
       - Solves objective (b): maximise total predicted survival.
       - Solves objective (c): maximise total *true* survival (oracle upper
         bound; uses the hidden ground-truth labels as cycle weights, so it is
         unattainable in practice and the oracle-vs-(b) gap is pure prediction
         error).
       - Evaluates all three solutions against the hidden noiseless ground truth.

Output
------
  kep_solve_results.csv   – one row per (instance, K, S, objective)
  kep_solve_summary.png   – true survival and transplant count vs S,
                            for K = 3, broken down by alpha and objective
  kep_solve_K_compare.png – K = 2 vs K = 3 comparison at S = 0
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

# ── configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = 'DATA'
RESULTS_DIR = 'RESULTS'
K_VALUES    = [2, 3]              # maximum cycle length
MST_VALUES  = list(range(19))     # S ∈ {0, 1, …, 18} (S≥19 leaves empty solutions)

os.makedirs(RESULTS_DIR, exist_ok=True)

pattern = re.compile(r'kep-surv_(\d+\.\d+)_(\w+)_(\d+)\.hist\.gz')


# ── helpers ───────────────────────────────────────────────────────────────────

def cycle_arcs(cycle):
    """Directed arcs (d, p) for a cycle tuple (v0, v1, …, vk-1)."""
    n = len(cycle)
    return [(cycle[i], cycle[(i + 1) % n]) for i in range(n)]


def solve_instance(alpha, surv_fn, inst, base):
    """Run the full K × S loop for one instance; return list of record dicts.

    Optimisation trick: enumerate cycles once per K on the full graph and
    cache the min-arc-survival of each cycle.  Filtering by S then reduces
    to a linear scan over cached cycles, avoiding expensive re-enumeration.
    """
    X_hist, y_hist = read_hist(base + '.hist.gz')
    pool_df         = read_pool(base + '.pool.gz')
    adj_full, _     = read_kep(base + '.kep.gz')   # weights = 1 (compatibility)
    _, hidd         = read_kep(base + '.hidd.gz')  # noiseless true survival (years)

    X_pool, pairs = build_pool_X(pool_df, adj_full)
    pipe = make_pipeline(HistGradientBoostingRegressor(random_state=0))
    pipe.fit(X_hist, y_hist)
    y_pred = np.maximum(pipe.predict(X_pool), 0.0)  # clip: negative predicted survival is nonsensical
    pred_surv = {(d, p): float(yp) for (d, p), yp in zip(pairs, y_pred)}

    records = []

    for K in K_VALUES:
        all_cycles = list(get_all_cycles(adj_full, K))
        # Cache per-cycle summaries once.
        cycle_min_pred = []
        cycle_pred_sv  = []
        cycle_true_sv  = []
        for c in all_cycles:
            arcs = cycle_arcs(c)
            preds = [pred_surv.get(a, -1e9) for a in arcs]
            cycle_min_pred.append(min(preds))
            cycle_pred_sv.append(sum(preds))
            cycle_true_sv.append(sum(hidd.get(a, 0.0) for a in arcs))

        for S in MST_VALUES:
            # Keep only cycles whose weakest arc's predicted survival is >= S.
            keep_idx = [i for i, m in enumerate(cycle_min_pred) if m >= S]
            cycles     = [all_cycles[i]     for i in keep_idx]
            cyc_pred   = [cycle_pred_sv[i]  for i in keep_idx]
            cyc_true   = [cycle_true_sv[i]  for i in keep_idx]

            empty = dict(
                alpha=alpha, surv_fn=surv_fn, inst=inst, K=K, S=S,
                n_cycles_avail=0, n_cycles_sel=0,
                n_transplants=0, true_survival=0.0, pred_survival=0.0,
            )

            if not cycles:
                records.append({**empty, 'obj': 'transplants'})
                records.append({**empty, 'obj': 'survival'})
                records.append({**empty, 'obj': 'oracle'})
                continue

            # Adjacency restricted to arcs used by the retained cycles.
            adj_filt = {}
            for c in cycles:
                for (u, v) in cycle_arcs(c):
                    adj_filt.setdefault(u, [])
                    if v not in adj_filt[u]:
                        adj_filt[u].append(v)

            cycle_idx = {c: i for i, c in enumerate(cycles)}

            def evaluate(sol_cycles):
                n_tx = sum(len(c) for c in sol_cycles)
                idx  = [cycle_idx[c] for c in sol_cycles]
                tsv  = sum(cyc_true[i] for i in idx)
                psv  = sum(cyc_pred[i] for i in idx)
                return n_tx, tsv, psv

            _, sol_a = kep_cycle(adj_filt, cycles)                       # max transplants
            ntx_a, tsv_a, psv_a = evaluate(sol_a)

            _, sol_b = kep_cycle(adj_filt, cycles, weights=cyc_pred)     # max predicted survival
            ntx_b, tsv_b, psv_b = evaluate(sol_b)

            # Oracle: maximise the HIDDEN true survival directly.  Unattainable
            # in practice (the true labels are never known at decision time); it
            # is the exact-optimisation upper bound, so the oracle-vs-survival
            # gap is pure prediction error.  Solved on the same filtered graph
            # as the other two objectives.
            _, sol_c = kep_cycle(adj_filt, cycles, weights=cyc_true)     # max true survival (oracle)
            ntx_c, tsv_c, psv_c = evaluate(sol_c)

            base_rec = dict(alpha=alpha, surv_fn=surv_fn, inst=inst,
                            K=K, S=S, n_cycles_avail=len(cycles))
            records.append({**base_rec, 'obj': 'transplants',
                            'n_cycles_sel': len(sol_a),
                            'n_transplants': ntx_a,
                            'true_survival': tsv_a,
                            'pred_survival': psv_a})
            records.append({**base_rec, 'obj': 'survival',
                            'n_cycles_sel': len(sol_b),
                            'n_transplants': ntx_b,
                            'true_survival': tsv_b,
                            'pred_survival': psv_b})
            records.append({**base_rec, 'obj': 'oracle',
                            'n_cycles_sel': len(sol_c),
                            'n_transplants': ntx_c,
                            'true_survival': tsv_c,
                            'pred_survival': psv_c})

    return records


# ── main experiment ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = os.path.join(RESULTS_DIR, 'kep_solve_results.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

    files   = sorted(f for f in os.listdir(DATA_DIR)
                     if (m := pattern.match(f)) and m.group(2) == 'dpi')
    n_total = len(files)

    for file_idx, fname in enumerate(files, 1):
        m = pattern.match(fname)
        alpha   = float(m.group(1))
        surv_fn = m.group(2)
        inst    = int(m.group(3))
        base    = os.path.join(DATA_DIR, fname[: -len('.hist.gz')])

        print(f"[{file_idx:3d}/{n_total}]  alpha={alpha}  surv={surv_fn}  inst={inst:02d}",
              flush=True)

        recs = solve_instance(alpha, surv_fn, inst, base)

        _cols = ['alpha','surv_fn','inst','K','S','n_cycles_avail','obj',
                 'n_cycles_sel','n_transplants','true_survival','pred_survival']
        pd.DataFrame(recs)[_cols].to_csv(csv_path, mode='a',
                                         header=not os.path.exists(csv_path),
                                         index=False)

    df = pd.read_csv(csv_path)
    print(f"\nResults written to {csv_path}")

    # ── summary table (K = 3 only, the headline) ─────────────────────────────
    print("\n── Mean true survival and transplants by (alpha, obj, S) at K=3 ──")
    summary = (df[df['K'] == 3]
                 .groupby(['alpha', 'obj', 'S'])[['true_survival', 'n_transplants']]
                 .mean().round(2))
    print(summary.to_string())

    # ── plot 1: S sensitivity at K = 3 (rows = alpha, cols = metric) ────────
    df3 = df[df['K'] == 3].copy()
    # per-instance average per-transplant survival (NaN if n_tx = 0)
    df3['avg_survival'] = df3['true_survival'] / df3['n_transplants'].replace(0, float('nan'))
    alphas  = sorted(df3['alpha'].unique())
    n_alpha = len(alphas)

    fig, axes = plt.subplots(n_alpha, 3, figsize=(11, 2.2 * n_alpha),
                             sharex='col', sharey='col')

    for row, a in enumerate(alphas):
        sub = df3[df3['alpha'] == a]

        for obj, color, ls in [('transplants', 'steelblue',  '-'),
                                ('survival',   'darkorange', '--'),
                                ('oracle',     'seagreen',   ':')]:
            grp = sub[sub['obj'] == obj].groupby('S')

            mean_sv = grp['true_survival'].mean()
            mean_tx = grp['n_transplants'].mean()
            mean_av = grp['avg_survival'].mean()
            std_sv  = grp['true_survival'].std()
            std_tx  = grp['n_transplants'].std()
            std_av  = grp['avg_survival'].std()

            label = {'transplants': 'max transplants',
                     'survival':    'max pred. survival',
                     'oracle':      'oracle (max true surv.)'}[obj]

            axes[row, 0].plot(mean_sv.index, mean_sv.values,
                              color=color, ls=ls, marker='o', ms=3, label=label)
            axes[row, 0].fill_between(mean_sv.index,
                                       mean_sv - std_sv, mean_sv + std_sv,
                                       color=color, alpha=0.12)

            axes[row, 1].plot(mean_tx.index, mean_tx.values,
                              color=color, ls=ls, marker='o', ms=3, label=label)
            axes[row, 1].fill_between(mean_tx.index,
                                       mean_tx - std_tx, mean_tx + std_tx,
                                       color=color, alpha=0.12)

            axes[row, 2].plot(mean_av.index, mean_av.values,
                              color=color, ls=ls, marker='o', ms=3, label=label)
            axes[row, 2].fill_between(mean_av.index,
                                       mean_av - std_av, mean_av + std_av,
                                       color=color, alpha=0.12)

        axes[row, 0].set_ylabel(fr'$\alpha={a}$', fontsize=11)
        axes[row, 0].grid(True, linestyle='--', alpha=0.4)
        axes[row, 1].grid(True, linestyle='--', alpha=0.4)
        axes[row, 2].grid(True, linestyle='--', alpha=0.4)

    axes[0, 0].set_title('True total survival (years)')
    axes[0, 1].set_title('Number of transplants')
    axes[0, 2].set_title('Avg. true survival per transplant (yr)')
    axes[-1, 0].set_xlabel('Min. survival threshold S (years)')
    axes[-1, 1].set_xlabel('Min. survival threshold S (years)')
    axes[-1, 2].set_xlabel('Min. survival threshold S (years)')
    axes[0, 2].legend(loc='lower right', fontsize=8)

    fig.suptitle('KEP optimisation, K = 3: three objectives × survival-threshold filter\n'
                 '(mean ± 1 SD over 10 instances; per-transplant average NaN-filtered when n_tx = 0)', y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'kep_solve_summary.png'),
                dpi=300, bbox_inches='tight')
    print(f"Plot saved to {RESULTS_DIR}/kep_solve_summary.png")
    plt.close(fig)

    # ── plot 2: K = 2 vs K = 3 at S = 0 ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    df_s0 = df[df['S'] == 0]

    for ax, metric, ylabel in zip(axes2,
                                    ['true_survival', 'n_transplants'],
                                    ['True total survival (years)',
                                     'Number of transplants']):
        obj_label = {'transplants': 'max tx', 'survival': 'max surv',
                     'oracle': 'oracle'}
        for K, marker, ls in [(2, 's', '--'), (3, 'o', '-')]:
            for obj, color in [('transplants', 'steelblue'),
                                ('survival',   'darkorange'),
                                ('oracle',     'seagreen')]:
                grp = (df_s0[(df_s0['K'] == K) & (df_s0['obj'] == obj)]
                          .groupby('alpha')[metric])
                means = grp.mean()
                stds  = grp.std()
                ax.errorbar(means.index, means.values, yerr=stds.values,
                            marker=marker, ls=ls, capsize=3,
                            color=color, ms=6,
                            label=f'K={K}, {obj_label[obj]}')
        ax.set_xlabel(r'$\alpha$ (training-label noise)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.4)
    # Legend only on the right subplot, at the bottom-right corner
    axes2[1].legend(fontsize=8, loc='lower right')

    fig2.suptitle('K = 2 vs K = 3, S = 0: no filter (mean ± 1 SD over 10 instances)',
                  y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, 'kep_solve_K_compare.png'),
                 dpi=300, bbox_inches='tight')
    print(f"Plot saved to {RESULTS_DIR}/kep_solve_K_compare.png")
    plt.close(fig2)
