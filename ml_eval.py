#!/usr/bin/env python
"""
ml_eval.py

Trains survival-prediction models on KEP historical data and evaluates their
quality on the pool by comparing predictions with the hidden ground truth.

Best-practice choices
---------------------
* sklearn Pipeline + ColumnTransformer: preprocessing is fitted only on
  training data, eliminating any risk of data leakage.
* OneHotEncoder for all categorical / binary string features.
* StandardScaler for numeric features (needed by Ridge; harmless for trees).
* Models compared
    Ridge  – fast linear baseline
    HGB    – HistGradientBoostingRegressor, generally the strongest sklearn
             model on tabular regression without feature engineering
* 5-fold cross-validation on historical data (10 000 pairs) to verify that
  the model learns a useful signal before trusting pool predictions.
* Pool predictions are compared with the noiseless ground truth stored in the
  .hidd.gz file to measure out-of-domain generalisation.
"""

import os
import re
import gzip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kep_io import read_kep

# ── column layout (mirrors kep_io._output_donor / _output_patient) ─────────────
DONOR_COLS   = ['d_age', 'd_sex', 'd_egfr', 'd_sbp', 'd_bmi',
                'd_smoker', 'd_relat', 'd_abo', 'd_weight', 'd_hlaA', 'd_hlaB', 'd_hlaDR']
PATIENT_COLS = ['p_age', 'p_sex', 'p_smoker', 'p_abo', 'p_weight', 'p_hlaA', 'p_hlaB', 'p_hlaDR', 'p_egfr']
ALL_COLS     = DONOR_COLS + PATIENT_COLS

NUMERIC      = ['d_age', 'd_egfr', 'd_sbp', 'd_bmi', 'd_weight', 'p_age', 'p_weight', 'p_egfr']
CATEGORICAL  = ['d_sex', 'd_abo', 'd_hlaA', 'd_hlaB', 'd_hlaDR',
                'p_sex', 'p_abo', 'p_hlaA', 'p_hlaB', 'p_hlaDR']
PASSTHROUGH  = ['d_smoker', 'd_relat', 'p_smoker']   # already 0/1 integers


# ── I/O helpers ─────────────────────────────────────────────────────────────────

def read_hist(path):
    """Return (X, y) from a .hist.gz file."""
    with gzip.open(path) as f:
        lines = f.read().decode().splitlines()
    nrecs = int(lines[0])
    rows = [l.split(',') for l in lines[1:nrecs + 1]]
    df = pd.DataFrame(rows, columns=ALL_COLS + ['Longevity'])
    df[NUMERIC + PASSTHROUGH + ['Longevity']] = (
        df[NUMERIC + PASSTHROUGH + ['Longevity']].apply(pd.to_numeric)
    )
    return df[ALL_COLS], df['Longevity']


def read_pool(path):
    """Return DataFrame indexed by vertex id from a .pool.gz file."""
    with gzip.open(path) as f:
        lines = f.read().decode().splitlines()
    nvert = int(lines[0])
    rows = {}
    for l in lines[1:nvert + 1]:
        vals = l.split(',')
        rows[int(vals[0])] = vals[1:]
    df = pd.DataFrame.from_dict(rows, orient='index', columns=ALL_COLS)
    df[NUMERIC + PASSTHROUGH] = df[NUMERIC + PASSTHROUGH].apply(pd.to_numeric)
    return df


def build_pool_X(pool_df, adj):
    """
    For every compatible (donor d → patient p) pair in adj, assemble a feature
    row by combining donor d's donor-side columns with patient p's patient-side
    columns.  Returns X (DataFrame) and the list of (d, p) pairs.
    """
    rows, pairs = [], []
    for d, patients in adj.items():
        for p in patients:
            row = pd.concat([pool_df.loc[d, DONOR_COLS],
                             pool_df.loc[p, PATIENT_COLS]])
            rows.append(row)
            pairs.append((d, p))
    return pd.DataFrame(rows, columns=ALL_COLS), pairs


# ── model factory ────────────────────────────────────────────────────────────────

def make_pipeline(model):
    """
    Build a leakage-free Pipeline:
      numeric  → StandardScaler
      categorical/binary strings → OneHotEncoder (unknown categories ignored)
      binary integers → pass through unchanged
    """
    pre = ColumnTransformer([
        ('num', StandardScaler(),                                              NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),  CATEGORICAL),
        ('bin', 'passthrough',                                                 PASSTHROUGH),
    ])
    return Pipeline([('pre', pre), ('model', model)])


MODELS = {
    'Ridge': Ridge(),
    'RF':    RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
    'HGB':   HistGradientBoostingRegressor(random_state=0),
}

CV = KFold(n_splits=5, shuffle=True, random_state=0)

# ── main experiment ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR    = 'DATA'
    RESULTS_DIR = 'RESULTS'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pattern  = re.compile(r'kep-surv_(\d+\.\d+)_(\w+)_(\d+)\.hist\.gz')

    csv_path = os.path.join(RESULTS_DIR, 'ml_results.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

    files = sorted(f for f in os.listdir(DATA_DIR) if pattern.match(f))
    n_total = len(files)
    for idx, fname in enumerate(files, 1):
        m = pattern.match(fname)
        alpha, surv_fn, inst = float(m.group(1)), m.group(2), int(m.group(3))
        base = os.path.join(DATA_DIR, fname[:-len('.hist.gz')])

        X_hist, y_hist = read_hist(base + '.hist.gz')
        pool_df        = read_pool(base + '.pool.gz')
        adj, w_true    = read_kep(base + '.hidd.gz')
        X_pool, pairs  = build_pool_X(pool_df, adj)
        y_true         = np.array([w_true[d, p] for d, p in pairs])

        print(f"[{idx:3d}/{n_total}] alpha={alpha}  surv={surv_fn}  inst={inst:02d}"
              f"  pool_pairs={len(pairs)}", flush=True)

        for name, model in MODELS.items():
            pipe = make_pipeline(model)

            # ── 5-fold CV on historical data ─────────────────────────────────────
            cv_res = cross_validate(
                pipe, X_hist, y_hist, cv=CV,
                scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error'],
                n_jobs=-1,
            )
            cv_mae  = -cv_res['test_neg_mean_absolute_error'].mean()
            cv_rmse = -cv_res['test_neg_root_mean_squared_error'].mean()

            # ── train on full historical data, predict pool pairs ─────────────────
            pipe.fit(X_hist, y_hist)
            y_pred = pipe.predict(X_pool)

            # ── compare predictions with hidden ground truth ──────────────────────
            pool_mae  = mean_absolute_error(y_true, y_pred)
            pool_rmse = root_mean_squared_error(y_true, y_pred)

            print(f"         {name:5s}  cv_mae={cv_mae:.2f}  cv_rmse={cv_rmse:.2f}"
                  f"  pool_mae={pool_mae:.2f}  pool_rmse={pool_rmse:.2f}", flush=True)

            rec = dict(
                alpha=alpha, surv_fn=surv_fn, inst=inst, model=name,
                cv_mae=cv_mae,   cv_rmse=cv_rmse,
                pool_mae=pool_mae, pool_rmse=pool_rmse,
            )
            pd.DataFrame([rec]).to_csv(csv_path, mode='a',
                                       header=not os.path.exists(csv_path),
                                       index=False)

    df = pd.read_csv(csv_path)
    print(f"\nResults written to {csv_path}")

    # ── summary table: mean errors by alpha and model ────────────────────────────────
    print("\n── Pool MAE (mean over instances) ──")
    print(df.groupby(['model', 'alpha'])[['pool_mae', 'pool_rmse']]
            .mean().round(3).to_string())

    # ── plots: prediction error vs alpha ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in zip(axes,
                                   ['pool_mae',  'pool_rmse'],
                                   ['MAE (years)', 'RMSE (years)']):
        for name in MODELS:
            grp = df[df.model == name].groupby('alpha')[metric]
            means = grp.mean()
            stds  = grp.std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker='o', capsize=3, label=name)
        ax.set_xlabel('alpha  (noise level in training data)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Pool prediction {ylabel} vs noise level')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ml_error_vs_alpha.png'), dpi=300)
    print(f"Plot saved to {RESULTS_DIR}/ml_error_vs_alpha.png")
    plt.close(fig)
