"""
Decomposition of the prediction loss (Proposition 1 and Table 3 of the paper).

For every instance at K=3, S=0: train HGB exactly as kep_solve.py does,
solve the max-predicted-survival programme (x_hat) and the oracle (x_star)
on the same graph, and decompose the oracle gap (the regret R)

    R = F(y, x_star) - F(y, x_hat) = E_or + E_surv - D_hat

where E_or = sum over sol(x_star) minus sol(x_hat) of (y - y_hat),
      E_surv = sum over sol(x_hat) minus sol(x_star) of (y_hat - y),
      D_hat = F(y_hat, x_hat) - F(y_hat, x_star) >= 0 (optimisation slack);
arcs used by both solutions cancel, so only the symmetric difference of the
two solutions' arc sets enters.  The bound B1 = E_or + E_surv is in turn
bounded by L1_sym, the l1 prediction error on that symmetric difference,
and by L1_all, the l1 error on the whole filtered arc set A(S).

Writes RESULTS/regret_bound.csv and prints the per-alpha summary used in
the paper.  Run time: ~2 minutes (60 HGB fits, 120 solves).
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from kep_io import read_kep
from kep_cycle import get_all_cycles, kep_cycle
from kep_solve import cycle_arcs
from ml_eval import read_hist, read_pool, build_pool_X, make_pipeline

DATA_DIR = "DATA"
RESULTS_DIR = "RESULTS"
K = 3
S = 0
pattern = re.compile(r"kep-surv_(\d+\.\d+)_(dpi)_(\d+)\.hist\.gz")


def regret_instance(base):
    X_hist, y_hist = read_hist(base + ".hist.gz")
    pool_df = read_pool(base + ".pool.gz")
    adj, _ = read_kep(base + ".kep.gz")
    _, hidd = read_kep(base + ".hidd.gz")

    X_pool, pairs = build_pool_X(pool_df, adj)
    pipe = make_pipeline(HistGradientBoostingRegressor(random_state=0))
    pipe.fit(X_hist, y_hist)
    y_pred = np.maximum(pipe.predict(X_pool), 0.0)
    yhat = {a: float(v) for a, v in zip(pairs, y_pred)}
    y = {a: hidd.get(a, 0.0) for a in pairs}

    # Same cycle-level filter as kep_solve.py (weakest arc >= S).
    cycles = [c for c in get_all_cycles(adj, K)
              if min(yhat[a] for a in cycle_arcs(c)) >= S]
    adj_filt = {}
    for c in cycles:
        for (u, v) in cycle_arcs(c):
            adj_filt.setdefault(u, [])
            if v not in adj_filt[u]:
                adj_filt[u].append(v)
    w_pred = [sum(yhat[a] for a in cycle_arcs(c)) for c in cycles]
    w_true = [sum(y[a] for a in cycle_arcs(c)) for c in cycles]

    _, sol_hat = kep_cycle(adj_filt, cycles, weights=w_pred)
    _, sol_star = kep_cycle(adj_filt, cycles, weights=w_true)
    arcs_hat = [a for c in sol_hat for a in cycle_arcs(c)]
    arcs_star = [a for c in sol_star for a in cycle_arcs(c)]

    def F(u, arcs):
        return sum(u[a] for a in arcs)

    filt = [a for a in pairs if yhat[a] >= S]
    only_star = set(arcs_star) - set(arcs_hat)
    only_hat = set(arcs_hat) - set(arcs_star)
    sym = only_star | only_hat
    return dict(
        T_surv=F(y, arcs_hat),
        T_or=F(y, arcs_star),
        E_or=F(y, only_star) - F(yhat, only_star),
        E_surv=F(yhat, only_hat) - F(y, only_hat),
        D_hat=F(yhat, arcs_hat) - F(yhat, arcs_star),
        L1_sym=sum(abs(y[a] - yhat[a]) for a in sym),
        L1_all=sum(abs(y[a] - yhat[a]) for a in filt),
        n_sym=len(sym),
        n_arcs=len(filt),
    )


def main():
    rows = []
    files = sorted(f for f in os.listdir(DATA_DIR) if pattern.match(f))
    for i, fname in enumerate(files, 1):
        m = pattern.match(fname)
        alpha, inst = float(m.group(1)), int(m.group(3))
        print(f"[{i:2d}/{len(files)}] alpha={alpha} inst={inst:02d}", flush=True)
        r = regret_instance(os.path.join(DATA_DIR, fname[: -len(".hist.gz")]))
        rows.append(dict(alpha=alpha, inst=inst, **r))

    df = pd.DataFrame(rows)
    df["R"] = df["T_or"] - df["T_surv"]
    df["B1"] = df["E_or"] + df["E_surv"]
    assert np.allclose(df["R"], df["B1"] - df["D_hat"], atol=1e-6)
    assert (df["R"] <= df["B1"] + 1e-6).all() and (df["B1"] <= df["L1_sym"] + 1e-6).all()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "regret_bound.csv"), index=False)

    cols = ["T_or", "T_surv", "R", "E_or", "E_surv", "D_hat", "B1", "L1_sym",
            "L1_all", "n_sym", "n_arcs"]
    print("\nMeans over 10 seeds (K=3, S=0):")
    print(df.groupby("alpha")[cols].mean().round(1).to_string())
    print("\nRatios of means:")
    g = df.groupby("alpha")[["R", "B1", "L1_sym", "L1_all"]].mean()
    print(pd.DataFrame({"R/B1": g.R / g.B1, "R/L1_sym": g.R / g.L1_sym,
                        "R/L1_all": g.R / g.L1_all}).round(3).to_string())


if __name__ == "__main__":
    main()
