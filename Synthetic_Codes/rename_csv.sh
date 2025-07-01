#!/bin/bash

base_dir="/scratch/nhoang2/IVIM_NeuroCovid/Result"
seeds=(24 69 97)
noise_modes=("nonoise" "lownoise" "highnoise")
subjects=("S1_signal" "S1_signal_IR" "NAWM_signal" "NAWM_signal_IR" "WMH_signal" "WMH_signal_IR")

for seed in "${seeds[@]}"; do
  for noise in "${noise_modes[@]}"; do
    for subject in "${subjects[@]}"; do
      folder="${base_dir}/Synth_Result_May2025_seed${seed}_${noise}/comparison_results_seed${seed}_${noise}/${subject}"
      old_file="${folder}/ivim_model_comparison_results_${subject}_seed${seed}.csv"
      new_file="${folder}/ivim_model_comparison_results_${subject}_${noise}_seed${seed}.csv"

      if [[ -f "$old_file" ]]; then
        if [[ -f "$new_file" ]]; then
          echo "[SKIP] $new_file already exists"
        else
          echo "[RENAME] $old_file → $new_file"
          mv "$old_file" "$new_file"
        fi
      fi
    done
  done
done

echo "[DONE] All filenames updated."
