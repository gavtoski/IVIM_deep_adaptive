#!/bin/bash

# === Define seeds you want to process ===
SEEDS=(24 69 97)

# === Loop over each seed and submit a job ===
for SEED in "${SEEDS[@]}"; do
  echo "[SUBMIT] compare_and_plot_results_synthetic.py for seed $SEED"

  sbatch --job-name=IVIMcompare_$SEED \
         --output=./out_logs/IVIM_synthplot_seed${SEED}_%j.out \
         --time=01:00:00 \
         --mem=16gb \
         --ntasks=1 \
         --cpus-per-task=1 \
         --mail-user=nhat_hoang@urmc.rochester.edu \
         --mail-type=END,FAIL \
         --wrap="python compare_and_plot_results_synthetic.py $SEED"
done

echo "[DONE] All comparison jobs submitted."
