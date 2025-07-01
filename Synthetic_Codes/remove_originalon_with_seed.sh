#!/bin/bash

# Set seed and noise modes
SEED=$1
NOISE_MODES=("nonoise" "lownoise" "highnoise")

# Subject folders
SUBJECTS=("S1_signal" "S1_signal_IR" "NAWM_signal" "NAWM_signal_IR" "WMH_signal" "WMH_signal_IR")

# Root result path
BASE_DIR="/scratch/nhoang2/IVIM_NeuroCovid/Result"

for noise_mode in "${NOISE_MODES[@]}"; do
  result_base="${BASE_DIR}/Synth_Result_May2025_seed${SEED}_${noise_mode}"
  echo "[INFO] Checking: $result_base"

  for subject in "${SUBJECTS[@]}"; do
    rm -rf "${result_base}/${subject}/originalon"
    rm -rf "${result_base}/${subject}/originalon_ir"
    echo "  [CLEANED] ${subject} - originalon + originalon_ir"
  done
done

echo "[DONE] All originalon and originalon_ir folders removed for seed ${SEED}."
