#!/bin/bash

# === Parameters ===
seeds=(24 69 97)
noise_modes=("nonoise" "lownoise" "highnoise")
wait_hours=6

# === Countdown Timer Function ===
countdown() {
  local secs=$(( $1 * 3600 ))
  while [ $secs -gt 0 ]; do
    hrs=$((secs / 3600))
    mins=$(( (secs % 3600) / 60 ))
    secs_rem=$((secs % 60))
    printf "\r⏳ Waiting: %02dh:%02dm:%02ds remaining..." $hrs $mins $secs_rem
    sleep 1
    ((secs--))
  done
  echo ""
}

# === Main Submission Loop ===
count=0
total_jobs=$(( ${#seeds[@]} * ${#noise_modes[@]} ))

for seed in "${seeds[@]}"; do
  for noise_mode in "${noise_modes[@]}"; do
    echo "[${count}/${total_jobs}] Submitting: Seed=${seed}, Noise=${noise_mode}"
    bash IVIM_NNsynthetic_alloptions.sh "$seed" "$noise_mode"

    # Wait between jobs
    if [ $count -lt $((total_jobs - 1)) ]; then
      echo "Sleeping for ${wait_hours} hours before next submission..."
      countdown "$wait_hours"
    fi

    ((count++))
  done
done

echo "All jobs for seeds 24, 69 and 97 submitted."
