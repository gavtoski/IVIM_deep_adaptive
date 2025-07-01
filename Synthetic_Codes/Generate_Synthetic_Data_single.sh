#!/bin/bash
#SBATCH --job-name=IVIM_gen_synthetic
#SBATCH --output=./out_logs/IVIM_triexp_%j.out
#SBATCH --error=./out_logs/IVIM_triexp_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=nhat_hoang@urmc.rochester.edu
#SBATCH --mail-type=END,FAIL

# === Load micromamba and activate env ===
export MAMBA_ROOT_PREFIX=/scratch/nhoang2/mamba_root
export PATH="/scratch/nhoang2/bin:$PATH"
eval "$(/scratch/nhoang2/bin/micromamba shell hook --shell=bash)"
micromamba activate pytorch2

# === Safety: Output logs ===
mkdir -p out_logs

# === Input params ===
SEED=$1
NOISE_MODE=$2
MODEL_TYPE=$3

echo "[START] IVIM synthetic generation - seed=$SEED, noise=$NOISE_MODE, model=$MODEL_TYPE"
python Generate_Synthetic_Data.py --seed "$SEED" --noise_mode "$NOISE_MODE" --model_type "$MODEL_TYPE"
echo "[DONE] Finished IVIM synthetic generation"
