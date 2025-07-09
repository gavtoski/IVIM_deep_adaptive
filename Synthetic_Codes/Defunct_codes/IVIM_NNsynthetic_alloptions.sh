#!/bin/bash

# === Inputs ===
SEED="$1"
NOISE_MODE="$2"

if [ -z "$SEED" ] || [ -z "$NOISE_MODE" ]; then
  echo "[USAGE] $0 <SEED> <NOISE_MODE>"
  echo "Example: $0 42 lownoise"
  exit 1
fi

# === Global Settings ===
bval_path="/scratch/nhoang2/IVIM_NeuroCovid/Data/bvals.txt"
declare -a MODEL_TYPES=("2C" "3C")
declare -a SUBJECTS=("S1_signal" "S1_signal_IR" "WMH_signal" "WMH_signal_IR" "NAWM_signal" "NAWM_signal_IR")
declare -a ABLATIONS=("none")

submission_count=0

# === Function to generate model tag ===
generate_model_tag() {
  local model_type="$1"
  local original="$2"
  local weight_tuning="$3"
  local IR="$4"
  local freeze_param="$5"
  local boost_toggle="$6"
  local ablate_option="$7"
  local tissue_type="$8"
  local bval_count="15"

  local tag="${model_type}_"

  if [[ "$original" == "True" ]]; then
    tag+="OriginalON"
  else
    tag+="OriginalOFF"
    if [[ "$weight_tuning" == "1" ]]; then
      tag+="_Tune"
      tag+="_FreezeON"
    fi
  fi

  if [[ "$IR" == "1" ]]; then
    tag+="_IR"
  fi

  if [[ "$ablate_option" != "none" ]]; then
    tag+="_${ablate_option}"
  fi

  if [[ "$boost_toggle" == "1" ]]; then
    tag+="_Boost"
  else
    tag+="_NoBoost"
  fi

  tag+="_b${bval_count}"
  tag+="_${tissue_type}"

  echo "$tag"
}

# === Main loop ===
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
  out_base="/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_${MODEL_TYPE}_seed${SEED}_${NOISE_MODE}"
  result_base="/scratch/nhoang2/IVIM_NeuroCovid/Result/Synth_Result_May2025_${MODEL_TYPE}_seed${SEED}_${NOISE_MODE}"

  for subject_file in "${SUBJECTS[@]}"; do
    preproc_loc="${out_base}/${subject_file}.npy"

    # === Assign tissue_type ===
    if [[ "$subject_file" == WMH_signal* ]]; then
      tissue_type="WMH"
    elif [[ "$subject_file" == NAWM_signal* ]]; then
      tissue_type="NAWM"
    else
      tissue_type="mixed"
    fi

    # === Skip IR for 2C ===
    if [[ "$MODEL_TYPE" == "2C" && "$subject_file" == *_IR ]]; then
      echo "[SKIP] Skipping IR variant for 2C: $subject_file"
      continue
    fi

    ### === ORIGINAL MODE ===
    for originalIR in 0 1; do
      if [[ "$MODEL_TYPE" == "2C" && "$originalIR" == "1" ]]; then continue; fi
      if [[ "$originalIR" == "1" && "$subject_file" != *_IR ]]; then continue; fi
      if [[ "$originalIR" == "0" && "$subject_file" == *_IR ]]; then continue; fi

      model_tag=$(generate_model_tag "$MODEL_TYPE" "True" 0 "$originalIR" 0 0 "none" "$tissue_type")
      dest_dir="${result_base}/${subject_file}/${model_tag}"
      log_file="${dest_dir}/log_submission.txt"

      if [ ! -d "$dest_dir" ] || [ -z "$(ls -A "$dest_dir" 2>/dev/null)" ]; then
        mkdir -p "$dest_dir"
        echo "[INFO] Submitting $model_tag for $subject_file [$MODEL_TYPE]" | tee -a "$log_file"
        sbatch IVIM_NNsynthetic_single.sh "$preproc_loc" "$bval_path" "$dest_dir" \
          True 0 "$originalIR" 0 0 "none" array "$tissue_type"
        ((submission_count++))
        if (( submission_count % 5 == 0 )); then
          echo "[WAIT] Reached 5 submissions. Sleeping for 3 minutes..."
          sleep 180
        fi
      else
        echo "[SKIP] $model_tag already processed for $subject_file [$MODEL_TYPE]" | tee -a "$log_file"
      fi
    done

    ### === ADAPTIVE MODE ===
    for IR in 0 1; do
      for boost_toggle in 0; do
        for ablate_option in "${ABLATIONS[@]}"; do
          if [[ "$MODEL_TYPE" == "2C" && "$IR" == "1" ]]; then continue; fi
          if [[ "$IR" == "1" && "$subject_file" != *_IR ]]; then continue; fi
          if [[ "$IR" == "0" && "$subject_file" == *_IR ]]; then continue; fi

          model_tag=$(generate_model_tag "$MODEL_TYPE" "False" 1 "$IR" 1 "$boost_toggle" "$ablate_option" "$tissue_type")
          dest_dir="${result_base}/${subject_file}/${model_tag}"
          log_file="${dest_dir}/log_submission.txt"

          if [ ! -d "$dest_dir" ] || [ -z "$(ls -A "$dest_dir" 2>/dev/null)" ]; then
            mkdir -p "$dest_dir"
            echo "[INFO] Submitting $model_tag for $subject_file [$MODEL_TYPE]" | tee -a "$log_file"
            sbatch IVIM_NNsynthetic_single.sh "$preproc_loc" "$bval_path" "$dest_dir" \
              False 1 "$IR" 1 "$boost_toggle" "$ablate_option" array "$tissue_type"
            ((submission_count++))
            if (( submission_count % 5 == 0 )); then
              echo "[WAIT] Reached 5 submissions. Sleeping for 3 minutes..."
              sleep 180
            fi
          else
            echo "[SKIP] $model_tag already processed for $subject_file [$MODEL_TYPE]" | tee -a "$log_file"
          fi
        done
      done
    done
  done
done
