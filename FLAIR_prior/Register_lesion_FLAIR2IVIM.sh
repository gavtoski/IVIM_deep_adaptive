#!/bin/bash

# Bin Hoang, University of Rochester
# Register_lesion_FLAIR2IVIM.sh

SUBJECT_ID="$1"
lesion_parent_folder=$(ls -d /Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/output/${SUBJECT_ID}_bsl_T1w_job* 2>/dev/null | head -n 1)
mkdir -p "result_folder/${SUBJECT_ID}"

if [[ -z "$lesion_parent_folder" ]]; then
	echo "[ERROR] Could not find VolBrain folder for subject ${SUBJECT_ID}"
	exit 1
fi

##################################################### file path labeling begins ###################################################
tissue_map=$(ls ${lesion_parent_folder}/native_tissues_job*.nii.gz 2>/dev/null | head -n 1) # CSF=1, GM=2, WM=3
lesion_map=$(ls ${lesion_parent_folder}/native_lesions_job*.nii.gz 2>/dev/null | head -n 1) # PV=1, DW=2, JC=3, INF=4

if [[ ! -f "$tissue_map" ]]; then
	echo "[ERROR] Tissue map not found for ${SUBJECT_ID}"
	exit 1
fi
if [[ ! -f "$lesion_map" ]]; then
	echo "[ERROR] Lesion map not found for ${SUBJECT_ID}"
	exit 1
fi

T1_path="/Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/input/sub-${SUBJECT_ID}_ses-bsl/${SUBJECT_ID}_bsl_T1w.nii.gz"
FLAIR_path="/Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/input/sub-${SUBJECT_ID}_ses-bsl/${SUBJECT_ID}_bsl_flair.nii.gz"

# Outputs
FLAIR_STRIPPED="result_folder/${SUBJECT_ID}/flair_stripped.nii.gz"
T1_STRIPPED="result_folder/${SUBJECT_ID}/t1_stripped.nii.gz"

# IVIM data + brain mask (subject-specific)
parent_path="/Users/nhoang2/Dropbox/Classes/IVIM_project/test_subjects"
ivim_path="${parent_path}/sub-${SUBJECT_ID}/ses-bsl/dwi/preproc_ivim/preproc_ivim_dwidata.nii.gz"
ivim_brain_mask="${parent_path}/sub-${SUBJECT_ID}/ses-bsl/dwi/preproc_ivim/hifi_nodif_brain_mask.nii.gz"
dest_path="result_folder/${SUBJECT_ID}/preproc_ivim_dwidata.nii.gz"

echo "[IVIM path] Using PREPROCESSED IVIM file:"
echo "  ${ivim_path}"
echo "[Mask path] Using IVIM brain mask:"
echo "  ${ivim_brain_mask}"


# Ensure destination folder exists
mkdir -p "$(dirname "${dest_path}")"
# Verify source exists & is readable
if [[ ! -r "$ivim_path" ]]; then
  echo "[ERROR] Source file missing or unreadable."; ls -l "$(dirname "$ivim_path")"; exit 1
fi
# Copy and overwrite if file exists
cp -f "${ivim_path}" "${dest_path}"
##################################################### file path labeling ends ######################################################


#################################################### Prepping necessary files begins ##############################################
# 1) Skull-strip FLAIR (this also produces 'flair_stripped_mask.nii.gz')
if [[ -f "$FLAIR_STRIPPED" ]]; then
	echo "[INFO] Skull-stripped FLAIR already exists — skipping BET."
else
	echo "[INFO] Skull-stripping FLAIR..."
	bet "${FLAIR_path}" "$FLAIR_STRIPPED" -f 0.5 -g 0 -m
fi

# 2) Use the FLAIR skullstrip mask to strip T1 (since T1 and FLAIR share native space)
FLAIR_MASK_NATIVE="result_folder/${SUBJECT_ID}/flair_stripped_mask.nii.gz"
if [[ ! -f "${FLAIR_MASK_NATIVE}" ]]; then
	echo "[ERROR] Expected FLAIR mask not found: ${FLAIR_MASK_NATIVE}"
	exit 1
fi

if [[ -f "$T1_STRIPPED" ]]; then
	echo "[INFO] T1 already masked with FLAIR mask — skipping."
else
	echo "[INFO] Applying FLAIR skullstrip mask to T1..."
	fslmaths "${T1_path}" -mas "${FLAIR_MASK_NATIVE}" "${T1_STRIPPED}"
fi

# Clean lesion map using non-CSF mask
lesion_map_clean="result_folder/${SUBJECT_ID}/native_lesions_WMH_cleaned.nii.gz"
if [[ -f "$lesion_map_clean" ]]; then
	echo "[INFO] Cleaned lesion map already exists — skipping cleanup."
else
	echo "[INFO] Cleaning lesion map using white matter mask..."
	CSF_MASK="result_folder/${SUBJECT_ID}/CSFmask.nii.gz"
	INVERT_CSF_MASK="result_folder/${SUBJECT_ID}/CSFmask_inverted.nii.gz"
	fslmaths "$tissue_map" -thr 1 -uthr 1 -bin "$CSF_MASK"           # CSF=1
	fslmaths "$CSF_MASK" -binv "$INVERT_CSF_MASK"                    # not-CSF
	fslmaths "$lesion_map" -mas "$INVERT_CSF_MASK" "$lesion_map_clean"
	rm -f "$CSF_MASK" "$INVERT_CSF_MASK"
fi

# Remove WMH voxels from tissue map
tissue_map_clean="result_folder/${SUBJECT_ID}/native_tissues_cleaned.nii.gz"
if [[ -f "$tissue_map_clean" ]]; then
	echo "[INFO] Cleaned tissue map already exists — skipping cleanup."
else
	echo "[INFO] Cleaning tissue map by removing lesion voxels..."
	lesion_bin="result_folder/${SUBJECT_ID}/lesion_mask_binary.nii.gz"
	lesion_inv="result_folder/${SUBJECT_ID}/lesion_mask_inverted.nii.gz"
	fslmaths "$lesion_map_clean" -thr 0.5 -bin "$lesion_bin"
	fslmaths "$lesion_bin" -binv "$lesion_inv"
	fslmaths "$tissue_map" -mas "$lesion_inv" "$tissue_map_clean"
	rm -f "$lesion_bin" "$lesion_inv"
fi

# Step 2: Create masked b0 reference from first volume
IVIM_b0="result_folder/${SUBJECT_ID}/b0_avg.nii"  # final kept output
rm -f result_folder/${SUBJECT_ID}/b0_raw.nii 2>/dev/null

if [[ -f "$IVIM_b0" ]]; then
	echo "[INFO] b0 file already exists for ${SUBJECT_ID}"
else
	if [[ ! -f "${ivim_brain_mask}" ]]; then
		echo "[ERROR] IVIM brain mask not found for ${SUBJECT_ID}: ${ivim_brain_mask}"
		exit 1
	fi
	tmp_b0_1vol="result_folder/${SUBJECT_ID}/b0_ivim.nii"
	tmp_b0_masked="result_folder/${SUBJECT_ID}/b0_ivim_masked.nii.gz"
	fslroi "${ivim_path}" "${tmp_b0_1vol}" 0 1
	fslmaths "${tmp_b0_1vol}" -mas "${ivim_brain_mask}" "${tmp_b0_masked}"
	fslmaths "${tmp_b0_masked}" -Tmean "$IVIM_b0"
	rm -f "${tmp_b0_1vol}" "${tmp_b0_masked}"
	echo "[INFO] Masked average IVIM b0 file created: $IVIM_b0"
fi
#################################################### Prepping necessary files ends ##################################################


#################################################### Registration begins ###########################################################
FLAIR2B0_MAT="result_folder/${SUBJECT_ID}/flair2b0.mat"
FLAIR2B0_AFF="result_folder/${SUBJECT_ID}/flair2b0_affine.nii.gz"   # keep this

# Ensure we have FLAIR→b0 affine once (keep preview image + .mat)
if [[ ! -f "${FLAIR2B0_MAT}" || ! -f "${FLAIR2B0_AFF}" ]]; then
	flirt -in "$FLAIR_STRIPPED" -ref "$IVIM_b0" -dof 12 \
	      -omat "${FLAIR2B0_MAT}" \
	      -out  "${FLAIR2B0_AFF}"
fi

# Apply same FLAIR→b0 transform to lesion & tissue labels (NN interp)
LESION2BRAIN="result_folder/${SUBJECT_ID}/lesion2ivim.nii.gz"
if [[ ! -f "${LESION2BRAIN}" ]]; then
	flirt -in "$lesion_map_clean" -ref "$IVIM_b0" \
	      -applyxfm -init "${FLAIR2B0_MAT}" \
	      -interp nearestneighbour \
	      -out "$LESION2BRAIN"
fi

TISSUE2BRAIN="result_folder/${SUBJECT_ID}/tissue2ivim.nii.gz"
if [[ ! -f "${TISSUE2BRAIN}" ]]; then
	flirt -in "$tissue_map_clean" -ref "$IVIM_b0" \
	      -applyxfm -init "${FLAIR2B0_MAT}" \
	      -interp nearestneighbour \
	      -out "${TISSUE2BRAIN}"
fi

# Apply the SAME FLAIR→b0 transform to T1 (trilinear interp)
T1w2b0="result_folder/${SUBJECT_ID}/T1w2b0_affine.nii.gz"           # keep this
if [[ ! -f "${T1w2b0}" ]]; then
	flirt -in "${T1_STRIPPED}" -ref "$IVIM_b0" \
	      -applyxfm -init "${FLAIR2B0_MAT}" \
	      -interp trilinear \
	      -out "$T1w2b0"
fi

# ================= HIFI-mask registered outputs (in-place, final) =================
# Inputs created above:
#   FLAIR2B0_AFF="result_folder/${SUBJECT_ID}/flair2b0_affine.nii.gz"
#   T1w2b0="result_folder/${SUBJECT_ID}/T1w2b0_affine.nii.gz"
#   TISSUE2BRAIN="result_folder/${SUBJECT_ID}/tissue2ivim.nii.gz"
#   LESION2BRAIN="result_folder/${SUBJECT_ID}/lesion2ivim.nii.gz"
#   ivim_brain_mask="${parent_path}/sub-${SUBJECT_ID}/ses-bsl/dwi/preproc_ivim/hifi_nodif_brain_mask.nii.gz"

if [[ ! -f "${ivim_brain_mask}" ]]; then
  echo "[ERROR] IVIM brain mask not found for ${SUBJECT_ID}: ${ivim_brain_mask}"
  exit 1
fi

# Write to tmp first, then atomically replace originals
FLAIR2B0_AFF_TMP="result_folder/${SUBJECT_ID}/flair2b0_affine_tmp.nii.gz"
T1w2b0_TMP="result_folder/${SUBJECT_ID}/T1w2b0_affine_tmp.nii.gz"
TISSUE2BRAIN_TMP="result_folder/${SUBJECT_ID}/tissue2ivim_tmp.nii.gz"
LESION2BRAIN_TMP="result_folder/${SUBJECT_ID}/lesion2ivim_tmp.nii.gz"

# Anatomicals
fslmaths "${FLAIR2B0_AFF}" -mas "${ivim_brain_mask}" "${FLAIR2B0_AFF_TMP}" && mv -f "${FLAIR2B0_AFF_TMP}" "${FLAIR2B0_AFF}"
fslmaths "${T1w2b0}"       -mas "${ivim_brain_mask}" "${T1w2b0_TMP}"       && mv -f "${T1w2b0_TMP}"       "${T1w2b0}"

# Labels (masking preserves integers; outside brain becomes 0)
fslmaths "${TISSUE2BRAIN}" -mas "${ivim_brain_mask}" "${TISSUE2BRAIN_TMP}" && mv -f "${TISSUE2BRAIN_TMP}" "${TISSUE2BRAIN}"
fslmaths "${LESION2BRAIN}" -mas "${ivim_brain_mask}" "${LESION2BRAIN_TMP}" && mv -f "${LESION2BRAIN_TMP}" "${LESION2BRAIN}"

echo "[INFO] HiFi-masked b0-space outputs finalized:"
echo "      ${FLAIR2B0_AFF}"
echo "      ${T1w2b0}"
echo "      ${TISSUE2BRAIN}"
echo "      ${LESION2BRAIN}"
# ================================================================================


# ================= CLEAN UP =================
# Required finals
KEEP_B0="result_folder/${SUBJECT_ID}/b0_avg.nii"
KEEP_FLAIR="result_folder/${SUBJECT_ID}/flair2b0_affine.nii.gz"   # HiFi-masked in-place
KEEP_T1="result_folder/${SUBJECT_ID}/T1w2b0_affine.nii.gz"        # HiFi-masked in-place
KEEP_TISSUE="result_folder/${SUBJECT_ID}/tissue2ivim.nii.gz"      # HiFi-masked in-place
KEEP_LESION="result_folder/${SUBJECT_ID}/lesion2ivim.nii.gz"      # HiFi-masked in-place

# Remove intermediates safely (ignore if missing)
rm -f "result_folder/${SUBJECT_ID}/flair2b0.mat" \
      "result_folder/${SUBJECT_ID}/mask_label"*.nii.gz \
      "result_folder/${SUBJECT_ID}/mask_tissue"*.nii.gz \
      "result_folder/${SUBJECT_ID}/native_lesions_WMH_cleaned.nii.gz" \
      "result_folder/${SUBJECT_ID}/native_tissues_cleaned.nii.gz" \
      "result_folder/${SUBJECT_ID}/flair_stripped.nii.gz" \
      "result_folder/${SUBJECT_ID}/t1_stripped.nii.gz" \
      "result_folder/${SUBJECT_ID}/flair_stripped_mask.nii.gz" \
      "result_folder/${SUBJECT_ID}/t1_stripped_mask.nii.gz" \
      "result_folder/${SUBJECT_ID}/flair2b0_affine_tmp.nii.gz" \
      "result_folder/${SUBJECT_ID}/T1w2b0_affine_tmp.nii.gz" \
      "result_folder/${SUBJECT_ID}/tissue2ivim_tmp.nii.gz" \
      "result_folder/${SUBJECT_ID}/lesion2ivim_tmp.nii.gz" \
      2>/dev/null || true

echo "[INFO] Kept finals:"
echo "      ${KEEP_B0}"
echo "      ${KEEP_FLAIR}"
echo "      ${KEEP_T1}"
echo "      ${KEEP_TISSUE}"
echo "      ${KEEP_LESION}"
# ================================================================================

#################################################### Registration ends #############################################################

#################################################### Extract tissue and lesion mean #################################################
echo "[INFO] Extracting mean/std b0 values for each lesion label..."

MASTER_CSV="result_folder/b0val_by_tissue_types.csv"
if [[ ! -f "$MASTER_CSV" ]]; then
	echo "SubjectID,TissueType,Mean_b0,Std_b0,VoxelCount" > "$MASTER_CSV"
fi

declare -A label_names
label_names[1]="Periventricular"
label_names[2]="DeepWhite"
label_names[3]="Juxtacortical"
label_names[4]="Infratentorial"

declare -A label_tissues
label_tissues[1]="CSF"
label_tissues[2]="GM"
label_tissues[3]="WM"

# Lesion types (1–4)
for label in 1 2 3 4; do
	MASK_OUT="result_folder/${SUBJECT_ID}/mask_label${label}.nii.gz"
	fslmaths "${LESION2BRAIN}" -thr $label -uthr $label -bin "$MASK_OUT"

	vox=$(fslstats "$MASK_OUT" -V | awk '{print $1}')
	if [[ "$vox" -gt 0 ]]; then
		read mean std <<< $(fslstats "$IVIM_b0" -k "$MASK_OUT" -M -S)
	else
		mean="NA"; std="NA"
	fi
	echo "${SUBJECT_ID},${label_names[$label]},${mean},${std},${vox}" >> "$MASTER_CSV"
	rm -f "$MASK_OUT"
done

# Tissue types (1–3)
for label in 1 2 3; do
	MASK_TISSUE="result_folder/${SUBJECT_ID}/mask_tissue${label}.nii.gz"
	fslmaths "${TISSUE2BRAIN}" -thr $label -uthr $label -bin "$MASK_TISSUE"

	vox=$(fslstats "$MASK_TISSUE" -V | awk '{print $1}')
	if [[ "$vox" -gt 0 ]]; then
		read mean std <<< $(fslstats "$IVIM_b0" -k "$MASK_TISSUE" -M -S)
	else
		mean="NA"; std="NA"
	fi
	echo "${SUBJECT_ID},${label_tissues[$label]},${mean},${std},${vox}" >> "$MASTER_CSV"
	rm -f "$MASK_TISSUE"
done

echo "[INFO] Output saved to: $MASTER_CSV"
#################################################### Extract tissue and lesion mean #################################################
