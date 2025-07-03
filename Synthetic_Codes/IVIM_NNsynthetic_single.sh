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

# === Usage instructions ===
if [ "$#" -lt 8 ]; then
    echo "[USAGE] $0 <preproc_loc> <bval_path> <dest_dir> <original_mode> <weight_tuning> <IR> <freeze_param> <boost_toggle> [ablate_option] [use_three_compartment] [input_type] [tissue_type] [custom_dict]"
    exit 1
fi

# === Parse inputs ===
preproc_loc="$1"
bval_path="$2"
dest_dir="$3"
original_mode="$4"
weight_tuning="$5"
IR="$6"
freeze_param="$7"
boost_toggle="$8"
ablate_option="${9:-none}"
use_three_compartment="${10:-True}"
input_type="${11:-array}"
tissue_type="${12:-mixed}"
custom_dict="${13:-None}"
dummy_arg="${14:-safe}"

# === Make sure output dir exists ===
mkdir -p "$dest_dir"
mkdir -p ./out_logs

# === Add a guard to skip reruns ===
DONE_FLAG="$dest_dir/.done"
if [ -f "$DONE_FLAG" ]; then
    echo "[SKIP] Already finished this config. Found $DONE_FLAG. Exiting."
    exit 0
fi

# === Log run info ===
echo "[RUN START] $(date) | SUBJECT: $preproc_loc | IR=$IR | DEST=$dest_dir" >> "$dest_dir/run_trace.log"

# === Launch main Python script ===
python IVIM_mapgenerator_synthetic_BH.py \
    --preproc_loc "$preproc_loc" \
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

# === Mark job as done ===
touch "$DONE_FLAG"
echo "[DONE] Completed successfully at $(date)" >> "$dest_dir/run_trace.log"
