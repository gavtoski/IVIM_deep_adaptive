#!/bin/bash

# === Create output log folder if missing ===
mkdir -p out_logs

# === Define seeds and noise modes ===
seeds=(24 69 97)
noises=("nonoise" "lownoise" "highnoise")
models=("2C" "3C")

# === Submission Counter ===
counter=0

# === Loop through all combinations ===
for seed in "${seeds[@]}"; do
  for noise in "${noises[@]}"; do
    for model in "${models[@]}"; do
      echo "[SUBMIT] Seed: $seed | Noise: $noise | Model: $model"
      sbatch Generate_Synthetic_Data_single.sh "$seed" "$noise" "$model"
      
      ((counter++))
      if (( counter % 5 == 0 )); then
        echo "[WAIT] Sleeping for 5 minutes after $counter submissions..."
        sleep 300
      fi
    done
  done
done
