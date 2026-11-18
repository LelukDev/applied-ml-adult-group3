#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
rm -f runs/run_log.csv
for script in 01_eda 02_compare_models 03_tune_optuna 04_explain 05_calibration 06_final_test 07_fairness; do
  echo "== $script"
  python $script.py
done
