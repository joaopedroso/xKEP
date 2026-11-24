import argparse
import math
import random
import os
from typing import NamedTuple
from kep_io import write_kep, write_pool, write_hist, write_hidden, write_pra
from collections import defaultdict

compat = {
    "O": ["O"],
    "A": ["O", "A"],
    "B": ["O", "B"],
    "AB": ["O", "A", "B", "AB"],
}

blood_type_frequencies = {"O": 0.406, "A": 0.318, "B": 0.220, "AB": 0.056}   # portuguese population
blood_type_keys = list(blood_type_frequencies.keys())
blood_type_values = list(blood_type_frequencies.values())
blood_type_gen = lambda: random.choices(blood_type_keys, blood_type_values)[0]

# PRA point-mass levels and their bucket frequencies, per Saidman et al.
# (2006), Transplantation 81(5):773-782.  Body text (p. 775) defines the
# per-bucket positive-crossmatch probabilities used in the simulator as
# low=5% (bucket PRA<10%), medium=45% (10-80%), high=90% (>80%); the
# "PRA distribution" table on p. 780 gives the bucket frequencies as
# low=0.7019, medium=0.20, high=0.0981.  Note: some reference
# implementations use 10% (not 45%) for the medium-bucket crossmatch
# probability; we follow the paper text.
_PRA_LEVELS      = [5, 45, 90]
_PRA_FREQUENCIES = [0.7019, 0.20, 0.0981]
pra_gen          = lambda: random.choices(_PRA_LEVELS, _PRA_FREQUENCIES)[0]


class Agent(NamedTuple):
    """Donor or patient in a KEP instance."""
    age:   int   # 18–75
    sex:   str   # "M" / "F"
    egfr:  int   # donors: 60–125 mL/min/1.73m²; patients: 5–20 (ESRD)
    sbp:   int   # systolic blood pressure, 80–200 mmHg, step 5
    bmi:   int   # 15–40 kg/m²
    aa:    bool  # African-American (unused, kept for LKDPI compatibility)
    smoker: bool # cigarette history
    relat: bool  # related donor
    abo:   str   # blood type: "O","A","B","AB"
    weight: int  # kg
    hlaA:  str   # HLA-A diploid genotype: "aa","ab","ac","bb","bc","cc"
    hlaB:  str   # HLA-B diploid genotype: "aa","ab","ac","bb","bc","cc"
    hlaDR: str   # HLA-DR diploid genotype: "aa","ab","ac","bb","bc","cc"
    pra:   int   # Panel Reactive Antibody % (0–99); used only for patients


# ── HLA allele encoding note ──────────────────────────────────────────────────
# Each locus field is a diploid genotype over the three-allele pool {a, b, c},
# stored as a two-character string in canonical (sorted) form.  The two allele
# slots are drawn independently and uniformly at random, so the six genotypes
# carry weights (aa:1, ab:2, ac:2, bb:1, bc:2, cc:1), summing to 9.
# Every agent carries two alleles per locus (no empty state), and the
# UNOS-style asymmetric mismatch — the number of donor antigens absent from
# the recipient's set — is computed by
#   sum(1 for a in "abc" if a in donor.hlaX and a not in patient.hlaX)
# which takes values 0, 1, or 2.  This is the mismatch count LKDPI (Massie
# et al., 2016) was fitted against.  The three-allele pool is the minimum
# that admits all three mismatch outcomes: with only two alleles, every
# recipient carries at least one of the two, capping mismatch at 1.
# ─────────────────────────────────────────────────────────────────────────────

_HLA_ALLELES   = "abc"
_HLA_GENOTYPES = ["aa", "ab", "ac", "bb", "bc", "cc"]
_HLA_WEIGHTS   = [   1,    2,    2,    1,    2,    1]  # two-allele-slot i.i.d. uniform


def rnd_donor():
    sex = "M" if random.random() < .45 else "F"
    return Agent(
        age    = random.randint(18, 75),
        sex    = sex,
        egfr   = random.randint(60, 125),
        sbp    = random.choice(range(80, 201, 5)),
        bmi    = random.randint(15, 40),
        aa     = 0,
        smoker = 0 if random.random() < .75 else 1,
        relat  = 1 if random.random() < .15 else 0,
        abo    = blood_type_gen(),
        weight = random.randint(50, 100) if sex == "M" else random.randint(40, 85),
        hlaA   = random.choices(_HLA_GENOTYPES, weights=_HLA_WEIGHTS)[0],
        hlaB   = random.choices(_HLA_GENOTYPES, weights=_HLA_WEIGHTS)[0],
        hlaDR  = random.choices(_HLA_GENOTYPES, weights=_HLA_WEIGHTS)[0],
        pra    = pra_gen(),
    )


def rnd_patient():
    sex = "M" if random.random() < .45 else "F"
    return Agent(
        age    = random.randint(18, 75),
        sex    = sex,
        egfr   = random.randint(5, 20),   # ESRD range
        sbp    = random.choice(range(80, 201, 5)),
        bmi    = random.randint(15, 40),
        aa     = 0,
        smoker = 0 if random.random() < .75 else 1,
        relat  = 1 if random.random() < .15 else 0,
        abo    = blood_type_gen(),
        weight = random.randint(50, 100) if sex == "M" else random.randint(40, 85),
        hlaA   = random.choices(_HLA_GENOTYPES, weights=_HLA_WEIGHTS)[0],
        hlaB   = random.choices(_HLA_GENOTYPES, weights=_HLA_WEIGHTS)[0],
        hlaDR  = random.choices(_HLA_GENOTYPES, weights=_HLA_WEIGHTS)[0],
        pra    = pra_gen(),
    )


# ── HLA-based virtual crossmatch ──────────────────────────────────────────────
#
# Antigens are identified by locus+allele: "A:a", "A:b", "A:c", "B:a", … (9
# total under the three-allele pool).  A patient's forbidden set is drawn at
# generation time; each antigen enters the set independently with probability
# q calibrated so that
#   E[P(crossmatch negative | random donor)] = 1 − pra/100.
#
# Derivation (diploid, three-allele pool, allele slots i.i.d. uniform):
# each locus contributes 1 unique antigen (the three homozygotes, prob 1/3)
# or 2 unique antigens (the three heterozygotes, prob 2/3).  Hence, with
# u = 1 − q,
#   E[u^|donor antigens at locus|] = u/3 + 2u²/3 = u(1 + 2u)/3.
# Over three independent loci this gives [u(1+2u)/3]³.  Setting this equal to
# (1 − pra/100) reduces to 2u² + u − 3·(1 − pra/100)^(1/3) = 0, with positive
# root u = (−1 + √(1 + 24·(1 − pra/100)^(1/3))) / 4, so
#   q = (5 − √(1 + 24·(1 − pra/100)^(1/3))) / 4.
# ─────────────────────────────────────────────────────────────────────────────

_ALL_ANTIGENS = [f"{loc}:{a}" for loc in ("A", "B", "DR") for a in _HLA_ALLELES]


def _forbidden_q(pra: int) -> float:
    """Per-antigen inclusion probability for the forbidden set, calibrated to pra."""
    if pra <= 0:
        return 0.0
    p_compat = 1.0 - pra / 100.0
    if p_compat <= 0.0:
        return 1.0
    return max(0.0, min(1.0,
        (5.0 - math.sqrt(1.0 + 24.0 * p_compat ** (1.0 / 3.0))) / 4.0
    ))


def generate_forbidden_antigens(pra: int) -> frozenset:
    """Draw patient's forbidden antigen set, calibrated to their PRA level."""
    q = _forbidden_q(pra)
    return frozenset(a for a in _ALL_ANTIGENS if random.random() < q)


def _donor_antigens(don) -> frozenset:
    """Antigens carried by the donor across all three loci."""
    return frozenset(
        f"{locus}:{allele}"
        for locus, field in (("A", don.hlaA), ("B", don.hlaB), ("DR", don.hlaDR))
        for allele in set(field)     # "aa" → {"a"}; "ab" → {"a","b"}; …
    )


def hla_compatible(don, forbidden: frozenset) -> bool:
    """True when the donor carries none of the patient's forbidden antigens."""
    return _donor_antigens(don).isdisjoint(forbidden)


# ── Survival functions ────────────────────────────────────────────────────────

def lkdpi(don, pat):
    """
    Living Kidney Donor Profile Index (Massie et al., 2016).
    Uses HLA-B and HLA-DR mismatches only, matching the original formula.
    Returns a score typically in [−77, 110]; median ≈ 13.
    """
    return (
        -11.30
        + 1.85  * max(0, don.age - 50)
        - 0.381 * don.egfr
        + 1.17  * don.bmi
        + (22.34 if don.aa else 0)
        + (14.33 if don.smoker else 0)
        + 0.44  * don.sbp
        - (21.68 if don.sex == "M" and pat.sex == "M" else 0)
        + (27.30 if don.abo not in compat[pat.abo] else 0)
        + 8.57  * sum(1 for a in _HLA_ALLELES if a in don.hlaB  and a not in pat.hlaB)
        + 8.26  * sum(1 for a in _HLA_ALLELES if a in don.hlaDR and a not in pat.hlaDR)
        - 50.87 * min(don.weight / pat.weight, 0.9)
    )


def survival_dpi(don, pat):
    """
    Map raw LKDPI score to graft survival years via a logistic.
    Anchor points: LKDPI=0 → 12 yr, LKDPI=13 → ~9 yr, LKDPI=33 → 5 yr.
    Asymptotes: 25 yr for very good donors, 0 yr for very poor donors.
    """
    L = lkdpi(don, pat)
    return max(0.0, 25.0 / (1.0 + math.exp(0.040 * (L + 2.0))))


def survival_add_noise(surv, alpha, maximum):
    """
    Blend deterministic survival with uniform noise.
    alpha=0 → pure deterministic; alpha=1 → pure uniform noise in [0, maximum].
    """
    r = random.uniform(0, maximum)
    return (1 - alpha) * surv + alpha * r


# ── Instance generation ───────────────────────────────────────────────────────

def draw_kep_intance(seed, surv_fn, alpha, maximum, hist_size, pool_size):
    random.seed(seed)

    # ── historical data: ABO + HLA-compatible pairs only ─────────────────────
    hist = []
    n = 0
    while n < hist_size:
        pat = rnd_patient()
        don = rnd_donor()
        if don.abo in compat[pat.abo]:
            forbidden = generate_forbidden_antigens(pat.pra)
            if hla_compatible(don, forbidden):
                surv = surv_fn(don, pat)
                hist.append((don, pat, survival_add_noise(surv, alpha, maximum)))
                n += 1

    # ── pool: generate pairs ──────────────────────────────────────────────────
    pat = {}
    don = {}
    pool_keys = range(1, pool_size + 1)
    for i in pool_keys:
        pat[i] = rnd_patient()
        don[i] = rnd_donor()

    # Pre-generate each patient's forbidden antigen set once (their sensitisation
    # is fixed; only the donor identity varies for each crossmatch).
    pat_forbidden = {i: generate_forbidden_antigens(pat[i].pra) for i in pool_keys}

    adj  = defaultdict(list)
    surv = defaultdict(int)
    w    = defaultdict(lambda: 1)
    for d in pool_keys:
        for p in pool_keys:
            if don[d].abo in compat[pat[p].abo]:
                if hla_compatible(don[d], pat_forbidden[p]):
                    adj[d].append(p)
                    surv[d, p] = surv_fn(don[d], pat[p])
    pool = (don, pat, surv)

    return hist, pool, adj, w


def mk_kep_intance(filename, hist, pool, adj, w):
    don, pat, surv = pool
    write_hist(filename + ".hist.gz", hist)
    write_pool(filename + ".pool.gz", don, pat)
    write_hidden(filename + ".hidd.gz", adj, surv)
    write_kep(filename + ".kep.gz", adj, w)
    # Side file with each pool patient's PRA; never used as an ML feature,
    # only for post-hoc fairness analysis of the survival-threshold filter.
    write_pra(filename + ".pra.gz", {i: pat[i].pra for i in pat})


if __name__ == "__main__":
    surv_fns = {"dpi": survival_dpi}

    parser = argparse.ArgumentParser(description="Generate a KEP instance.")
    parser.add_argument("--seed",      type=int,   required=True)
    parser.add_argument("--dir",       type=str,   default="DATA")
    parser.add_argument("--hist-size", type=int,   default=10000)
    parser.add_argument("--pool-size", type=int,   default=100)
    parser.add_argument("--maximum",   type=float, default=30.0)
    parser.add_argument("--alpha",     type=float, required=True)
    parser.add_argument("--surv",      type=str,   required=True, choices=surv_fns)
    parser.add_argument("--n",         type=int,   required=True)
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    surv_fn  = surv_fns[args.surv]
    filename = f"{args.dir}/kep-surv_{args.alpha:.1f}_{args.surv}_{args.n:03}"
    print(f"creating instance {filename}")
    hist, pool, adj, w = draw_kep_intance(
        args.seed, surv_fn, args.alpha, args.maximum, args.hist_size, args.pool_size
    )
    mk_kep_intance(filename, hist, pool, adj, w)
