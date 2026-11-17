#!/usr/bin/env bash
# 7 variants x 3 seeds on Protocol A (21 runs) + 3 models x 3 seeds on Protocol B_G2L (9 runs) = 30 runs.
set -e
DATA=data/processed; SPL=data/splits.csv
for seed in 0 1 2; do
  for v in post_only early_fusion siam_full siam_no_diff siam_no_prior siam_focal_only siam_ce_only; do
    python train.py --data $DATA --splits $SPL --protocol A --variant $v --seed $seed --out runs
  done
  for v in post_only early_fusion siam_full; do
    python train.py --data $DATA --splits $SPL --protocol B_G2L --variant $v --seed $seed --out runs
  done
done
python aggregate.py runs | tee results_summary.txt
python make_figures.py --data $DATA --splits $SPL --run runs/A/siam_full/seed0 --out figs/qualitative.pdf
