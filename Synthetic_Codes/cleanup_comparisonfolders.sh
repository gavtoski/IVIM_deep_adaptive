#!/bin/bash

SEEDS=(24 69 97)
NOISE_MODES=("nonoise" "lownoise" "highnoise")

for seed in "${SEEDS[@]}"; do
  for noise in "${NOISE_MODES[@]}"; do
    target_path="/scratch/nhoang2/IVIM_NeuroCovid/Result/Synth_Result_May2025_seed${seed}_${noise}"
    
    echo "[INFO] Checking $target_path"

    # Remove all folders starting with "comparison_results"
    for folder in "$target_path"/comparison_results*; do
      if [ -d "$folder" ]; then
        echo "  [DELETING] $folder"
        rm -rf "$folder"
      fi
    done
  done
done

echo "[DONE] Cleaned all comparison_results folders for seeds 24, 69, 97."
