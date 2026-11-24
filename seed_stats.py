"""
Per-seed comparison of the survival and max-transplants objectives
(Table A1 and the Wilcoxon p-values of Table 2 of the paper).

Reads RESULTS/kep_solve_results.csv (K=3, S=0) and, for every alpha,
pairs the ten seeds to report: the per-seed true-survival totals of the two
objectives, their difference and relative gain, the correlation of the two
totals across seeds, and the two-sided Wilcoxon signed-rank p-value of the
difference (the p column of Table 2 of the paper).

Writes RESULTS/seed_stats.csv (one row per alpha and seed) and prints the
per-alpha summary and the alpha=0 per-seed table used in the paper.
"""
import os
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = "RESULTS"
K = 3
S = 0


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "kep_solve_results.csv"))
    d = (df[(df.K == K) & (df.S == S)]
         .pivot_table(index=["alpha", "inst"], columns="obj",
                      values="true_survival")
         .reset_index())
    d["diff"] = d["survival"] - d["transplants"]
    d["gain_pct"] = 100 * d["diff"] / d["transplants"]
    d.to_csv(os.path.join(RESULTS_DIR, "seed_stats.csv"), index=False)

    rows = []
    for alpha, g in d.groupby("alpha"):
        rows.append(dict(
            alpha=alpha,
            diff_mean=g["diff"].mean(), diff_sd=g["diff"].std(),
            sd_surv=g["survival"].std(), sd_tx=g["transplants"].std(),
            corr=g["survival"].corr(g["transplants"]),
            gain_min=g["gain_pct"].min(), gain_max=g["gain_pct"].max(),
            n_improved=int((g["diff"] > 0).sum()),
            p_wilcoxon=wilcoxon(g["survival"], g["transplants"]).pvalue,
        ))
    print("Paired comparison, survival vs transplants (K=3, S=0):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nPer-seed totals at alpha=0:")
    cols = ["inst", "transplants", "survival", "diff", "gain_pct"]
    print(d[d.alpha == 0.0][cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
