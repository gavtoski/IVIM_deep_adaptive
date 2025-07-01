#!/bin/bash

# === Config ===
SEEDS=(24 69 97)
NOISE_MODES=("nonoise" "lownoise" "highnoise")

# Subject IR status
declare -A SUBJECTS_IR=(
  ["S1_signal"]=0
  ["WMH_signal"]=0
  ["NAWM_signal"]=0
  ["S1_signal_IR"]=1
  ["WMH_signal_IR"]=1
  ["NAWM_signal_IR"]=1
)

echo "=== CHECKING CLEANUP INTEGRITY ==="

# Loop over seeds and noise modes
for SEED in "${SEEDS[@]}"; do
  for NOISE in "${NOISE_MODES[@]}"; do
    echo ">> Seed $SEED | Noise: $NOISE"
    result_base="/scratch/nhoang2/IVIM_NeuroCovid/Result/Synth_Result_May2025_seed${SEED}_${NOISE}"
    cleanup_needed=0

    for subject in "${!SUBJECTS_IR[@]}"; do
      is_IR="${SUBJECTS_IR[$subject]}"
      subject_dir="${result_base}/${subject}"

      if [ ! -d "$subject_dir" ]; then
        echo "  [MISSING] $subject_dir"
        continue
      fi

      for config_dir in "$subject_dir"/*; do
        [ -d "$config_dir" ] || continue
        config_name=$(basename "$config_dir")

        if [[ "$is_IR" == "0" && "$config_name" == *"IR1"* ]]; then
          echo "  [ERROR] IR1 config found for non-IR subject: $subject → $config_name"
          cleanup_needed=1
        elif [[ "$is_IR" == "1" && "$config_name" == *"IR0"* ]]; then
          echo "  [ERROR] IR0 config found for IR subject: $subject → $config_name"
          cleanup_needed=1
        fi
      done
    done

    if [[ "$cleanup_needed" -eq 1 ]]; then
      echo "  [ACTION] Running cleanup_IRmodes.sh $SEED $NOISE"
      bash cleanup_IRmodes.sh "$SEED" "$NOISE"
    else
      echo "  [OK] No invalid configs found."
    fi

    echo ""
  done
done

echo "=== CHECK COMPLETE ==="
