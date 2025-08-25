#!/bin/bash

#Bin Hoang, University of Rochester
# Usage: ./rsync_to_volbrain.sh NC001

SUB_ID=$1

# Paths to original NIFTI files
FLAIR_path="/Volumes/SchifittoLab/Project_NeuroCOVID/NIFTI_NeuroCOVID/sub-${SUB_ID}/ses-bsl/anat/sub-${SUB_ID}_ses-bsl_flair_anonymized.nii.gz"
T1_path="/Volumes/SchifittoLab/Project_NeuroCOVID/NIFTI_NeuroCOVID/sub-${SUB_ID}/ses-bsl/anat/sub-${SUB_ID}_ses-bsl_T1w_anonymized.nii.gz"

# Destination paths for VolBrain
FLAIR_Volbrain_path="/Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/input/sub-${SUB_ID}_ses-bsl/${SUB_ID}_bsl_flair.nii.gz"
T1_Volbrain_path="/Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/input/sub-${SUB_ID}_ses-bsl/${SUB_ID}_bsl_T1w.nii.gz"

# Create target directory if it doesn't exist
mkdir -p "/Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/input/sub-${SUB_ID}_ses-bsl/"

# Rsync FLAIR and T1
echo "Copying FLAIR: ${FLAIR_path} to ${FLAIR_Volbrain_path}"
rsync -av "${FLAIR_path}" "${FLAIR_Volbrain_path}"

echo "Copying T1w: ${T1_path} to ${T1_Volbrain_path}"
rsync -av "${T1_path}" "${T1_Volbrain_path}"
