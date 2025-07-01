#!/bin/bash
#SBATCH --job-name=IVIM_compare
#SBATCH --output=./out_logs/IVIM_compare_seed%j.out
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=nhat_hoang@urmc.rochester.edu
#SBATCH --mail-type=END,FAIL

# === Micromamba setup
export MAMBA_ROOT_PREFIX=/scratch/nhoang2/mamba_root
export PATH="/scratch/nhoang2/bin:$PATH"
eval "$(/scratch/nhoang2/bin/micromamba shell hook --shell=bash)"
micromamba activate pytorch2

# === Input: Seed
if [ -z "$1" ]; then
  echo "[ERROR] Must provide a seed (e.g., 24, 69, or 97)"
  exit 1
fi

SEED=$1
mkdir -p out_logs

echo "[INFO] Running comparisons for seed=${SEED}"
python compare_and_plot_results_synthetic.py "$SEED"
echo "[DONE] Seed $SEED complete"