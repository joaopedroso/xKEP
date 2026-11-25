"""
Fairness analysis of the minimum-survival-time filter (Section 7.6 of the paper).

For each alpha=0 instance: train the HGB regressor exactly as kep_solve.py
does, predict survival for every compatible arc, and for each threshold S
compute which patients have *donor access* (at least one incoming arc with
predicted survival >= S).  Donor-access rates are reported by patient
sensitisation group (PRA 5/45/90, read from the .pra.gz side files) and by
the number of homozygous HLA-B/DR loci of the patient (homozygous patients
present fewer antigens, so donor antigens are more often foreign, raising
the LKDPI mismatch term and hence lowering predicted survival).

Writes RESULTS/fairness_pra.png and prints the donor-access table used in the
paper.  Run time: ~1 minute (10 HGB fits).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

from kep_io import read_kep, read_pra
from ml_eval import read_hist, read_pool, build_pool_X, make_pipeline

ALPHA = 0.0
SURV_FN = "dpi"
INSTANCES = range(1, 11)
MST_VALUES = list(range(19))
PRA_GROUPS = [5, 45, 90]


def patient_best_pred(base):
    """Return (best_pred, pra, homoz) dicts keyed by patient vertex id."""
    X_hist, y_hist = read_hist(base + ".hist.gz")
    pool_df = read_pool(base + ".pool.gz")
    adj, _ = read_kep(base + ".kep.gz")
    pra = read_pra(base + ".pra.gz")

    X_pool, pairs = build_pool_X(pool_df, adj)
    pipe = make_pipeline(HistGradientBoostingRegressor(random_state=0))
    pipe.fit(X_hist, y_hist)
    y_pred = np.maximum(pipe.predict(X_pool), 0.0)

    # Best incoming predicted survival; -1 marks patients with no compatible
    # donor at all (excluded by the crossmatch, not by the filter), so that
    # the S=0 row reports baseline reachability.
    best = {p: -1.0 for p in pool_df.index}
    reachable = {p: False for p in pool_df.index}
    for (d, p), yp in zip(pairs, y_pred):
        reachable[p] = True
        best[p] = max(best[p], float(yp))

    homoz = {}
    for p in pool_df.index:
        hb, hdr = pool_df.loc[p, "p_hlaB"], pool_df.loc[p, "p_hlaDR"]
        homoz[p] = int(hb[0] == hb[1]) + int(hdr[0] == hdr[1])
    return best, reachable, pra, homoz


def main():
    rows = []
    for inst in INSTANCES:
        base = f"DATA/kep-surv_{ALPHA:.1f}_{SURV_FN}_{inst:03}"
        best, reachable, pra, homoz = patient_best_pred(base)
        for p in best:
            rows.append(dict(inst=inst, patient=p, best_pred=best[p],
                             reachable=reachable[p], pra=pra[p], homoz=homoz[p]))
    df = pd.DataFrame(rows)
    df.to_csv("RESULTS/fairness_patients.csv", index=False)

    print(f"{len(df)} patients across {len(INSTANCES)} instances (alpha={ALPHA})")
    print("\nPatients per PRA group and baseline reachability (S=0):")
    for g in PRA_GROUPS:
        sub = df[df.pra == g]
        print(f"  PRA {g:2d}: n={len(sub):4d}  reachable={sub.reachable.mean():.1%}")
    print("\nPatients per homozygous-loci group (HLA-B/DR) and reachability:")
    for h in [0, 1, 2]:
        sub = df[df.homoz == h]
        print(f"  homoz {h}: n={len(sub):4d}  reachable={sub.reachable.mean():.1%}")

    print("\nDonor access (fraction of patients with >=1 incoming arc with "
          "predicted survival >= S):")
    header = "S    " + "".join(f"PRA{g:<5}" for g in PRA_GROUPS) \
             + "".join(f"hom{h:<5}" for h in [0, 1, 2]) + "all"
    print(header)
    ret_pra = {g: [] for g in PRA_GROUPS}
    ret_hom = {h: [] for h in [0, 1, 2]}
    ret_all = []
    for S in MST_VALUES:
        vals = []
        for g in PRA_GROUPS:
            sub = df[df.pra == g]
            r = (sub.best_pred >= S).mean()
            ret_pra[g].append(r)
            vals.append(f"{r:7.1%}")
        for h in [0, 1, 2]:
            sub = df[df.homoz == h]
            r = (sub.best_pred >= S).mean()
            ret_hom[h].append(r)
            vals.append(f"{r:7.1%}")
        r_all = (df.best_pred >= S).mean()
        ret_all.append(r_all)
        print(f"{S:<4d}" + "".join(vals) + f"{r_all:7.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {5: "tab:green", 45: "tab:orange", 90: "tab:red"}
    for g in PRA_GROUPS:
        axes[0].plot(MST_VALUES, ret_pra[g], marker="o", ms=3,
                     color=colors[g], label=f"PRA = {g}")
    axes[0].plot(MST_VALUES, ret_all, color="k", ls="--", lw=1, label="all patients")
    axes[0].set_xlabel("minimum survival threshold $S$ (years)")
    axes[0].set_ylabel("fraction of patients with donor access")
    axes[0].set_title("by sensitisation (PRA)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    styles = {0: "tab:blue", 1: "tab:purple", 2: "tab:brown"}
    labels = {0: "0 homozygous loci", 1: "1 homozygous locus", 2: "2 homozygous loci"}
    for h in [0, 1, 2]:
        axes[1].plot(MST_VALUES, ret_hom[h], marker="s", ms=3,
                     color=styles[h], label=labels[h])
    axes[1].plot(MST_VALUES, ret_all, color="k", ls="--", lw=1, label="all patients")
    axes[1].set_xlabel("minimum survival threshold $S$ (years)")
    axes[1].set_title("by patient HLA-B/DR homozygosity")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("RESULTS/fairness_pra.png", dpi=200)
    print("\nwrote RESULTS/fairness_pra.png and RESULTS/fairness_patients.csv")


if __name__ == "__main__":
    main()
