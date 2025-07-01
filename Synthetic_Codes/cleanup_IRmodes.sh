#!/bin/bash

# === Inputs ===
SEED="$1"
NOISE_MODE="$2"

if [ -z "$SEED" ] || [ -z "$NOISE_MODE" ]; then
    echo "[USAGE] $0 <SEED> <NOISE_MODE>"
    echo "Example: $0 42 lownoise"
    exit 1
fi

# === Base Path ===
result_base="/scratch/nhoang2/IVIM_NeuroCovid/Result/Synth_Result_May2025_seed${SEED}_${NOISE_MODE}"

# === Subjects and IR status ===
# 0 = non-IR subject → delete IR1 configs
# 1 = IR subject     → delete IR0 configs
declare -A SUBJECTS_IR=(
  ["S1_signal"]=0
  ["WMH_signal"]=0
  ["NAWM_signal"]=0
  ["S1_signal_IR"]=1
  ["WMH_signal_IR"]=1
  ["NAWM_signal_IR"]=1
)

# === Loop over all subjects ===
for subject in "${!SUBJECTS_IR[@]}"; do
  is_IR="${SUBJECTS_IR[$subject]}"
  subject_dir="${result_base}/${subject}"

  if [ ! -d "$subject_dir" ]; then
    echo "[SKIP] $subject_dir does not exist"
    continue
  fi

  echo "[PROCESSING] $subject (IR=${is_IR})"

  # Go through subdirectories inside subject folder
  for config_dir in "$subject_dir"/*; do
    [ -d "$config_dir" ] || continue  # skip non-folders

    config_name=$(basename "$config_dir")

    # Rule 1: For IR subjects, delete anything with IR0
    if [[ "$is_IR" == "1" && "$config_name" == *"IR0"* ]]; then
      echo "[DELETE] $config_dir → IR0 on IR subject"
      rm -rf "$config_dir"
    fi

    # Rule 2: For non-IR subjects, delete anything with IR1
    if [[ "$is_IR" == "0" && "$config_name" == *"IR1"* ]]; then
      echo "[DELETE] $config_dir → IR1 on non-IR subject"
      rm -rf "$config_dir"
    fi
  done

done
