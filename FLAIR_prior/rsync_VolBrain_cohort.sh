#Bin Hoang, University of Rochester

#!/bin/bash

# This script loops through all baseline subjects in NIFTI_NeuroCOVID and calls single subject rsync script

ROOT_DIR="/Volumes/SchifittoLab/Project_NeuroCOVID/NIFTI_NeuroCOVID"

for subj_dir in ${ROOT_DIR}/sub-*/ses-bsl; do
	# Extract subject ID (e.g., sub-NC001 → NC001)
	SUB_ID=$(basename $(dirname "$subj_dir") | sed 's/sub-//')

	# Call single subject rsync script
	echo "[INFO] Running rsync for subject: ${SUB_ID}"
	./rsync_VolBrain_singlesubject.sh ${SUB_ID}
done
