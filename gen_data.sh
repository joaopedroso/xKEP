#!/usr/bin/env bash
set -euo pipefail

DIR="DATA"
HIST_SIZE=10000
POOL_SIZE=100
MAXIMUM=30
N_INSTANCES=10
ALPHAS=(0.0 0.2 0.4 0.6 0.8 1.0)
SURVS=(dpi)

mkdir -p "$DIR"

for SURV in "${SURVS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        for N in $(seq 1 $N_INSTANCES); do
            uv run python kep_gen.py \
                --seed "$N" \
                --dir "$DIR" \
                --hist-size "$HIST_SIZE" \
                --pool-size "$POOL_SIZE" \
                --maximum "$MAXIMUM" \
                --alpha "$ALPHA" \
                --surv "$SURV" \
                --n "$N"
        done
    done
done
