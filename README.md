# xKEP — Maximising Patient Lifetime in Kidney Exchange Programmes

Reproducibility code for the paper *Data-Driven Kidney Exchange Programmes*
(J. P. Pedroso, D. Líbano, A. Viana; submitted to MDPI *Mathematics*).

The pipeline simulates kidney exchange programme (KEP) instances, trains
machine-learning models to predict graft survival, and solves an
optimisation problem that selects transplant cycles maximising total
predicted graft survival, rather than the number of transplants.

> **Synthetic data only.** No real patients or registry records are used.
> The pipeline is a reproducibility artifact, not a deployed clinical
> system, and the paper does not prescribe allocation policy.

## Background

In a KEP, incompatible donor–patient pairs are pooled and matched in
cycles: each donor gives to the next pair's patient, and the last donor
gives to the first pair's patient. Standard KEP solvers maximise the
*number of transplants*. This project asks what happens if we instead
maximise *total predicted graft survival*.

With noise-free training labels, the survival objective delivers
**+27.5 % more true graft-survival years** than the max-transplants
baseline, at an average cost of 1.3 fewer transplants, and realises
**93.2 %** of the gain attainable by an oracle with perfect knowledge of
survival. Because the optimiser is exact for every objective, the gap to
the oracle measures the cost of prediction error alone.

## Repository structure

```
kep_gen.py        synthetic KEP instance generator
kep_io.py         read/write compressed .gz data files
kep_cycle.py      DFS cycle enumeration + AMPL/Gurobi solver
kep.mod           AMPL integer-programming model (per-cycle weights)
ml_eval.py        ML pipeline (Ridge / RF / HGB), 5-fold CV, error analysis
kep_solve.py      main experiment: survival threshold × three objectives × K∈{2,3} × 60 instances
kep_dist.py       per-transplant true-survival distribution for the two objectives
fairness_eval.py  patients retained by the survival threshold, by PRA group and HLA homozygosity
seed_stats.py     per-seed paired comparison and Wilcoxon signed-rank tests
regret_eval.py    decomposition of the prediction loss (Proposition 1 of the paper)
gen_data.sh       generates all 60 instances
DATA/             generated instances (.gz) — not committed; created by gen_data.sh
RESULTS/          experiment outputs (CSV, PNG) as reported in the paper
```

## Data generation

Synthetic instances follow the Portuguese blood-type distribution and use
clinical parameters (age, sex, eGFR, SBP, BMI, smoking, weight,
HLA-A/B/DR, PRA) to generate donor–patient pairs.

HLA is modelled with three loci (A, B, DR) typed as diploid strings over
the three-allele pool `{a, b, c}` with uniform allele frequencies. PRA
follows the Saidman et al. (2006) point-mass distribution — 5 %, 45 %,
90 % with weights 0.7019, 0.20, 0.0981 — and each patient's
forbidden-antigen inclusion probability is set in closed form so that the
expected crossmatch-negative rate matches 1 − PRA/100.

The ground-truth survival function maps the raw Living Kidney Donor
Profile Index (Massie et al. 2016, *AJT*) to graft survival in years via
a logistic:

```
survival_dpi(LKDPI) = max(0,  25 / (1 + exp(0.040 × (LKDPI + 2))))
```

| LKDPI | Survival (yr) | Clinical meaning                |
|-------|---------------|---------------------------------|
| < 0   | > 12          | Better than median living donor |
| 0     | 12            | —                               |
| 13    | ~9            | Median living donor             |
| 33    | 5             | 75th percentile (high risk)     |
| > 47  | ~0            | Very high risk                  |

The parameter `alpha` controls training-label noise: the label is
`(1−α)·survival + α·Uniform(0, 30)`, so `0` is the noise-free ground
truth and `1` is pure noise. The experiments use
α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, ten seeds each (60 instances). Only
the historical training labels are noised; the pool ground truth is
stored noiselessly and never seen by the model.

Compatibility requires (1) ABO blood-type compatibility and (2) a
negative virtual crossmatch: the donor carries none of the patient's
forbidden HLA antigens.

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv);
Python ≥ 3.12.

```bash
uv sync
```

The optimisation stage uses AMPL with Gurobi, which require a licence
(a free [AMPL Community Edition](https://ampl.com/ce) or an academic
licence). Activate it once in the project environment:

```bash
uv run python -c "from ampltools import modules; modules.activate('<your-licence-uuid>')"
```

## Running the experiments

```bash
# 1. Generate all 60 instances (6 alpha × 10 seeds) into DATA/
bash gen_data.sh

# 2. ML evaluation: Ridge / RF / HGB, 5-fold CV, comparison with the hidden truth
uv run python ml_eval.py        # → RESULTS/ml_results.csv, ml_error_vs_alpha.png

# 3. KEP optimisation: threshold S ∈ {0,…,18}, three objectives, K ∈ {2,3}
#    (about 1 h on an Apple M4 Mac mini, Gurobi on one thread)
uv run python kep_solve.py      # → RESULTS/kep_solve_results.csv,
                                #   kep_solve_summary.png, kep_solve_K_compare.png

# 4. Per-transplant true-survival distribution at S=0, K=3
uv run python kep_dist.py       # → RESULTS/kep_dist_hist.png

# 5. Fairness of the survival threshold
uv run python fairness_eval.py  # → RESULTS/fairness_pra.png, fairness_patients.csv

# 6. Per-seed paired comparison (reads kep_solve_results.csv; no solver calls)
uv run python seed_stats.py     # → RESULTS/seed_stats.csv

# 7. Decomposition of the prediction loss, K=3, S=0 (about 2 min)
uv run python regret_eval.py    # → RESULTS/regret_bound.csv

# Standalone AMPL/Gurobi test on a built-in example graph
uv run python kep_cycle.py
```

## Key results

### Three objectives at S = 0 (no threshold), K = 3

Means over the 10 instances per α. *Gain* is the survival objective's
improvement over max-transplants; *captured* is the fraction of the
oracle-attainable gain that it realises. The oracle (max *true* survival)
delivers 815.1 true graft-years at 87.5 transplants, independently of α.

| α   | Max survival (yr) | #tx  | Max transplants (yr) | #tx  | Gain        | Captured   |
|-----|-------------------|------|----------------------|------|-------------|------------|
| 0.0 | **802.5**         | 88.2 | 629.2                | 89.5 | **+27.5 %** | **93.2 %** |
| 0.2 | 792.2             | 89.3 | 629.2                | 89.5 | +25.9 %     | 87.7 %     |
| 0.4 | 743.1             | 89.5 | 629.2                | 89.5 | +18.1 %     | 61.3 %     |
| 0.6 | 691.5             | 89.5 | 629.2                | 89.5 | +9.9 %      | 33.5 %     |
| 0.8 | 656.5             | 89.5 | 629.2                | 89.5 | +4.3 %      | 14.7 %     |
| 1.0 | 629.3             | 89.5 | 629.2                | 89.5 | +0.0 %      | 0.1 %      |

### Survival threshold

Removing arcs with predicted survival below S trades transplant volume for
per-transplant quality. At α = 0, raising S from 0 to 10 years lifts mean
per-transplant true survival from about 9.1 to 13.4 years while reducing
transplants from 88.2 to 30.6 (total 802.5 → 409.3 years). At high noise
the threshold cannot single out low-quality arcs.

### Why total, not average, survival?

The ratio total/transplants is trivially maximised by performing fewer
transplants, so it is not a meaningful objective on its own. The
clinically meaningful dial is the survival threshold S, which decides
which arcs are available to be matched.

## Licence

MIT — see [LICENSE](LICENSE).

## Authors

- João Pedro Pedroso — Faculty of Sciences, University of Porto; CMUP
- Danielle Líbano — Faculty of Sciences, University of Porto
- Ana Viana — INESC TEC; ISEP, Polytechnic of Porto
