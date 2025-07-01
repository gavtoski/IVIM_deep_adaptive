#!/bin/bash

# === Config ===
SEEDS=(24 69 97)
NOISE_MODES=("nonoise" "lownoise" "highnoise")
SUBJECTS=("S1_signal" "S1_signal_IR" "NAWM_signal" "NAWM_signal_IR" "WMH_signal" "WMH_signal_IR")
BVAL_PATH="/scratch/nhoang2/IVIM_NeuroCovid/Data/bvals.txt"

for seed in "${SEEDS[@]}"; do
  for noise in "${NOISE_MODES[@]}"; do
    echo "[INFO] Processing seed=$seed noise=$noise"
    # Input base
    data_base="/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_seed${seed}_${noise}"
    # Output base
    result_base="/scratch/nhoang2/IVIM_NeuroCovid/Result/Synth_Result_May2025_seed${seed}_${noise}"

    for subj in "${SUBJECTS[@]}"; do
      input_file="${data_base}/${subj}.npy"

      # === OriginalON (IR off) ===
      dest_on="${result_base}/${subj}/originalon"
      mkdir -p "$dest_on"
      echo "  [SUBMIT] ${subj} | originalon"
      sbatch IVIM_NNsynthetic_single.sh "$input_file" "$BVAL_PATH" "$dest_on" True 0 0 0 0 none array

      # === OriginalON_IR (IR on) ===
      dest_ir="${result_base}/${subj}/originalon_ir"
      mkdir -p "$dest_ir"
      echo "  [SUBMIT] ${subj} | originalon_ir"
      sbatch IVIM_NNsynthetic_single.sh "$input_file" "$BVAL_PATH" "$dest_ir" True 0 1 0 0 none array

    done
  done
done

echo "[DONE] All originalon + originalon_ir jobs submitted."
