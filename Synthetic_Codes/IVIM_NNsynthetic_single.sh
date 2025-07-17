#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --job-name=IVIM_triexp
#SBATCH --output=./out_logs/IVIM_triexp_%j.out
#SBATCH --mail-user=nhat_hoang@urmc.rochester.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# Load micromamba
export MAMBA_ROOT_PREFIX=/scratch/nhoang2/mamba_root
export PATH="/scratch/nhoang2/bin:$PATH"
eval "$(/scratch/nhoang2/bin/micromamba shell hook --shell=bash)"
micromamba activate pytorch2

# Usage instructions
if [ "$#" -lt 10 ]; then
    echo "[USAGE] $0 <train_loc> <val_loc> <GT_loc> <bval_path> <dest_dir> <original_mode> <weight_tuning> <IR> <freeze_param> <boost_toggle> [ablate_option] [use_three_compartment] [input_type] [tissue_type] [custom_dict]"
    exit 1
fi


# Parse inputs 
train_loc="$1"
val_loc="$2"
GT_loc="$3"
bval_path="$4"
dest_dir="$5"
original_mode="$6"
weight_tuning="$7"
IR="$8"
freeze_param="$9"
boost_toggle="${10:-none}"
ablate_option="${11:-none}"
use_three_compartment="${12:-True}"
input_type="${13:-array}"
tissue_type="${14:-mixed}"
custom_dict="${15:-None}"

# Ensure output dirs 
mkdir -p "$dest_dir"
mkdir -p ./out_logs

# Skip if already done 
DONE_FLAG="$dest_dir/.done"
if [ -f "$DONE_FLAG" ]; then
    echo "[SKIP] Already finished this config. Found $DONE_FLAG. Exiting."
    exit 0
fi

# === Log run info ===
echo "[RUN START] $(date) | TRAIN: $train_loc | VAL: $val_loc | DEST: $dest_dir | IR=$IR" >> "$dest_dir/run_trace.log"

# Launch IVIM model generation
python IVIM_mapgenerator_synthetic_BH.py \
    --train_loc "$train_loc" \
    --val_loc "$val_loc" \
    --GT_loc "$GT_loc" \
    --bval_path "$bval_path" \
    --dest_dir "$dest_dir" \
    --original_mode "$original_mode" \
    --weight_tuning "$weight_tuning" \
    --IR "$IR" \
    --freeze_param "$freeze_param" \
    --boost_toggle "$boost_toggle" \
    --ablate_option "$ablate_option" \
    --use_three_compartment "$use_three_compartment" \
    --input_type "$input_type" \
    --tissue_type "$tissue_type" \
    --custom_dict "$custom_dict"

touch "$DONE_FLAG"
echo "[DONE] Completed successfully at $(date)" >> "$dest_dir/run_trace.log"
