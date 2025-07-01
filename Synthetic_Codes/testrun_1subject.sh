#!/bin/bash

sbatch IVIM_NNsynthetic_single.sh \
  /scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_seed24_nonoise/S1_signal.npy \
  /scratch/nhoang2/IVIM_NeuroCovid/Data/bvals.txt \
  /scratch/nhoang2/IVIM_NeuroCovid/Result/Synth_Result_May2025_seed24_nonoise/S1_signal/originaloff_WT1_IR1_FP1_BT1_ABLnone \
  False \
  1 \
  1 \
  1 \
  1 \
  none \
  array