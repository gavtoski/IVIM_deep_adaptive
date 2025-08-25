#!/bin/bash
set -euo pipefail

# Bin Hoang, University of Rochester
# Register_lesion_FLAIR2IVIM.sh

if [[ $# -lt 1 ]]; then
	echo "[USAGE] $0 <SUBJECT_ID>"
	exit 1
fi

SUBJECT_ID="$1"
lesion_parent_folder=$(ls -d /Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/output/${SUBJECT_ID}_bsl_T1w_job* 2>/dev/null | head -n 1)
mkdir -p "result_folder/${SUBJECT_ID}"

if [[ -z "${lesion_parent_folder}" ]]; then
	echo "[ERROR] Could not find VolBrain folder for subject ${SUBJECT_ID}"
	exit 1
fi

############################################ Paths ############################################
tissue_map=$(ls ${lesion_parent_folder}/native_tissues_job*.nii.gz 2>/dev/null | head -n 1) # CSF=1, GM=2, WM=3
lesion_map=$(ls ${lesion_parent_folder}/native_lesions_job*.nii.gz 2>/dev/null | head -n 1) # PV=1, DW=2, JC=3, INF=4

T1_path="/Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/input/sub-${SUBJECT_ID}_ses-bsl/${SUBJECT_ID}_bsl_T1w.nii.gz"
FLAIR_path="/Volumes/SchifittoLab/Project_NeuroCOVID/VolBrain/input/sub-${SUBJECT_ID}_ses-bsl/${SUBJECT_ID}_bsl_flair.nii.gz"

# DTI inputs (optional)
DTI_b0="/Users/nhoang2/Dropbox/Classes/IVIM_project/test_subjects/sub-${SUBJECT_ID}/ses-bsl/dwi/preproc/dti_S0.nii.gz" # skull-stripped
FA="/Users/nhoang2/Dropbox/Classes/IVIM_project/test_subjects/sub-${SUBJECT_ID}/ses-bsl/dwi/preproc/dti_FA.nii.gz"
MD="/Users/nhoang2/Dropbox/Classes/IVIM_project/test_subjects/sub-${SUBJECT_ID}/ses-bsl/dwi/preproc/dti_MD.nii.gz"

# IVIM
parent_path="/Users/nhoang2/Dropbox/Classes/IVIM_project/test_subjects"
ivim_path="${parent_path}/sub-${SUBJECT_ID}/ses-bsl/dwi/preproc_ivim/preproc_ivim_dwidata.nii.gz"
ivim_brain_mask="${parent_path}/sub-${SUBJECT_ID}/ses-bsl/dwi/preproc_ivim/hifi_nodif_brain_mask.nii.gz"
dest_path="result_folder/${SUBJECT_ID}/preproc_ivim_dwidata.nii.gz"

# Outputs
FLAIR_STRIPPED="result_folder/${SUBJECT_ID}/flair_stripped.nii.gz"
T1_STRIPPED="result_folder/${SUBJECT_ID}/t1_stripped.nii.gz"
IVIM_b0="result_folder/${SUBJECT_ID}/b0_avg.nii"

# Required inputs (fail if missing)
for f in "$tissue_map" "$lesion_map" "$T1_path" "$FLAIR_path" "$ivim_path" "$ivim_brain_mask"; do
	if [[ ! -f "$f" ]]; then echo "[ERROR] Missing required input: $f"; exit 1; fi
done

# Optional DTI flags
HAS_DTI_B0=0; [[ -f "$DTI_b0" ]] && HAS_DTI_B0=1
HAS_FA=0;     [[ -f "$FA"     ]] && HAS_FA=1
HAS_MD=0;     [[ -f "$MD"     ]] && HAS_MD=1

echo "[INFO] DTI_b0 present: $HAS_DTI_B0 | FA present: $HAS_FA | MD present: $HAS_MD"

#################################### Prepare IVIM b0 ##########################################
mkdir -p "$(dirname "${dest_path}")"
cp -f "${ivim_path}" "${dest_path}"
ivim_src="${dest_path}"

if [[ -f "$IVIM_b0" ]]; then
	echo "[INFO] b0 exists — skipping."
else
	tmp_b0_1vol="result_folder/${SUBJECT_ID}/b0_ivim.nii"
	tmp_b0_masked="result_folder/${SUBJECT_ID}/b0_ivim_masked.nii.gz"
	fslroi "${ivim_src}" "${tmp_b0_1vol}" 0 1
	fslmaths "${tmp_b0_1vol}" -mas "${ivim_brain_mask}" "${tmp_b0_masked}"
	fslmaths "${tmp_b0_masked}" -Tmean "$IVIM_b0"
	rm -f "${tmp_b0_1vol}" "${tmp_b0_masked}"
fi

################################### Skull-strip + Cleaning ####################################
# FLAIR BET (produces flair_stripped_mask)
if [[ -f "$FLAIR_STRIPPED" ]]; then
	echo "[INFO] Skull-stripped FLAIR exists — skipping BET."
else
	bet "${FLAIR_path}" "$FLAIR_STRIPPED" -f 0.5 -g 0 -m
fi
FLAIR_MASK_NATIVE="result_folder/${SUBJECT_ID}/flair_stripped_mask.nii.gz"
[[ -f "${FLAIR_MASK_NATIVE}" ]] || { echo "[ERROR] Missing ${FLAIR_MASK_NATIVE}"; exit 1; }

# T1 mask with FLAIR mask
if [[ -f "$T1_STRIPPED" ]]; then
	echo "[INFO] T1 already masked — skipping."
else
	fslmaths "${T1_path}" -mas "${FLAIR_MASK_NATIVE}" "${T1_STRIPPED}"
fi

# Clean lesion map: remove CSF
lesion_map_clean="result_folder/${SUBJECT_ID}/native_lesions_WMH_cleaned.nii.gz"
if [[ ! -f "$lesion_map_clean" ]]; then
	CSF_MASK="result_folder/${SUBJECT_ID}/CSFmask.nii.gz"
	INVERT_CSF_MASK="result_folder/${SUBJECT_ID}/CSFmask_inverted.nii.gz"
	fslmaths "$tissue_map" -thr 1 -uthr 1 -bin "$CSF_MASK"
	fslmaths "$CSF_MASK" -binv "$INVERT_CSF_MASK"
	fslmaths "$lesion_map" -mas "$INVERT_CSF_MASK" "$lesion_map_clean"
	rm -f "$CSF_MASK" "$INVERT_CSF_MASK"
fi

# Remove WMH from tissue map
tissue_map_clean="result_folder/${SUBJECT_ID}/native_tissues_cleaned.nii.gz"
if [[ ! -f "$tissue_map_clean" ]]; then
	lesion_bin="result_folder/${SUBJECT_ID}/lesion_mask_binary.nii.gz"
	lesion_inv="result_folder/${SUBJECT_ID}/lesion_mask_inverted.nii.gz"
	fslmaths "$lesion_map_clean" -thr 0.5 -bin "$lesion_bin"
	fslmaths "$lesion_bin" -binv "$lesion_inv"
	fslmaths "$tissue_map" -mas "$lesion_inv" "$tissue_map_clean"
	rm -f "$lesion_bin" "$lesion_inv"
fi

####################################### Registration ##########################################
FLAIR2B0_MAT="result_folder/${SUBJECT_ID}/flair2b0.mat"
FLAIR2B0_AFF="result_folder/${SUBJECT_ID}/flair2b0_affine.nii.gz"

# FLAIR → IVIM b0 (affine)
if [[ ! -f "${FLAIR2B0_MAT}" || ! -f "${FLAIR2B0_AFF}" ]]; then
	flirt -in "$FLAIR_STRIPPED" -ref "$IVIM_b0" -dof 12 \
	      -omat "${FLAIR2B0_MAT}" \
	      -out  "${FLAIR2B0_AFF}"
fi

# Apply to labels (NN) and T1 (trilinear)
LESION2BRAIN="result_folder/${SUBJECT_ID}/lesion2ivim.nii.gz"
[[ -f "${LESION2BRAIN}" ]] || flirt -in "$lesion_map_clean" -ref "$IVIM_b0" -applyxfm -init "${FLAIR2B0_MAT}" -interp nearestneighbour -out "$LESION2BRAIN"

TISSUE2BRAIN="result_folder/${SUBJECT_ID}/tissue2ivim.nii.gz"
[[ -f "${TISSUE2BRAIN}" ]] || flirt -in "$tissue_map_clean" -ref "$IVIM_b0" -applyxfm -init "${FLAIR2B0_MAT}" -interp nearestneighbour -out "${TISSUE2BRAIN}"

T1w2b0="result_folder/${SUBJECT_ID}/T1w2b0_affine.nii.gz"
[[ -f "${T1w2b0}" ]] || flirt -in "${T1_STRIPPED}" -ref "$IVIM_b0" -applyxfm -init "${FLAIR2B0_MAT}" -interp trilinear -out "${T1w2b0}"

# -------- Optional DTI branch --------
DTI2B0_MAT="result_folder/${SUBJECT_ID}/dti2b0.mat"
DTI2B0_AFF="result_folder/${SUBJECT_ID}/dti2b0_affine.nii.gz"
FA2b0="result_folder/${SUBJECT_ID}/FA2ivim.nii.gz"
MD2b0="result_folder/${SUBJECT_ID}/MD2ivim.nii.gz"

if [[ $HAS_DTI_B0 -eq 1 ]]; then
	if [[ ! -f "${DTI2B0_MAT}" || ! -f "${DTI2B0_AFF}" ]]; then
		flirt -in "$DTI_b0" -ref "$IVIM_b0" -dof 12 -omat "${DTI2B0_MAT}" -out  "${DTI2B0_AFF}"
	fi

	# FA if present
	if [[ $HAS_FA -eq 1 && ! -f "${FA2b0}" ]]; then
		flirt -in "${FA}" -ref "${IVIM_b0}" -applyxfm -init "${DTI2B0_MAT}" -interp trilinear -out "${FA2b0}"
	else
		[[ $HAS_FA -eq 1 ]] || echo "[WARN] FA not found — skipping FA registration."
	fi

	# MD if present
	if [[ $HAS_MD -eq 1 && ! -f "${MD2b0}" ]]; then
		flirt -in "${MD}" -ref "${IVIM_b0}" -applyxfm -init "${DTI2B0_MAT}" -interp trilinear -out "${MD2b0}"
	else
		[[ $HAS_MD -eq 1 ]] || echo "[WARN] MD not found — skipping MD registration."
	fi
else
	echo "[WARN] DTI_b0 not found — skipping all DTI/FA/MD registrations."
fi

# ----------------- HiFi mask (conditional for FA/MD) -----------------
[[ -f "${ivim_brain_mask}" ]] || { echo "[ERROR] Missing IVIM brain mask ${ivim_brain_mask}"; exit 1; }

FLAIR2B0_AFF_TMP="result_folder/${SUBJECT_ID}/flair2b0_affine_tmp.nii.gz"
T1w2b0_TMP="result_folder/${SUBJECT_ID}/T1w2b0_affine_tmp.nii.gz"
TISSUE2BRAIN_TMP="result_folder/${SUBJECT_ID}/tissue2ivim_tmp.nii.gz"
LESION2BRAIN_TMP="result_folder/${SUBJECT_ID}/lesion2ivim_tmp.nii.gz"
FA_TMP="result_folder/${SUBJECT_ID}/fa2ivim_tmp.nii.gz"
MD_TMP="result_folder/${SUBJECT_ID}/md2ivim_tmp.nii.gz"

fslmaths "${FLAIR2B0_AFF}" -mas "${ivim_brain_mask}" "${FLAIR2B0_AFF_TMP}" && mv -f "${FLAIR2B0_AFF_TMP}" "${FLAIR2B0_AFF}"
fslmaths "${T1w2b0}"       -mas "${ivim_brain_mask}" "${T1w2b0_TMP}"       && mv -f "${T1w2b0_TMP}"       "${T1w2b0}"
fslmaths "${TISSUE2BRAIN}" -mas "${ivim_brain_mask}" "${TISSUE2BRAIN_TMP}" && mv -f "${TISSUE2BRAIN_TMP}" "${TISSUE2BRAIN}"
fslmaths "${LESION2BRAIN}" -mas "${ivim_brain_mask}" "${LESION2BRAIN_TMP}" && mv -f "${LESION2BRAIN_TMP}" "${LESION2BRAIN}"

# Only mask FA/MD if they exist
if [[ -f "${FA2b0}" ]]; then
	fslmaths "${FA2b0}" -mas "${ivim_brain_mask}" "${FA_TMP}" && mv -f "${FA_TMP}" "${FA2b0}"
fi
if [[ -f "${MD2b0}" ]]; then
	fslmaths "${MD2b0}" -mas "${ivim_brain_mask}" "${MD_TMP}" && mv -f "${MD_TMP}" "${MD2b0}"
fi

echo "[INFO] Finalized (existing) outputs:"
echo "      ${IVIM_b0}"
echo "      ${FLAIR2B0_AFF}"
echo "      ${T1w2b0}"
echo "      ${TISSUE2BRAIN}"
echo "      ${LESION2BRAIN}"
[[ -f "${FA2b0}" ]] && echo "      ${FA2b0}" || true
[[ -f "${MD2b0}" ]] && echo "      ${MD2b0}" || true

# ----------------- CLEAN UP -----------------
# Finals to keep
KEEP_B0="result_folder/${SUBJECT_ID}/b0_avg.nii"
KEEP_FLAIR="result_folder/${SUBJECT_ID}/flair2b0_affine.nii.gz"
KEEP_T1="result_folder/${SUBJECT_ID}/T1w2b0_affine.nii.gz"
KEEP_TISSUE="result_folder/${SUBJECT_ID}/tissue2ivim.nii.gz"
KEEP_LESION="result_folder/${SUBJECT_ID}/lesion2ivim.nii.gz"
KEEP_FA="result_folder/${SUBJECT_ID}/FA2ivim.nii.gz"
KEEP_MD="result_folder/${SUBJECT_ID}/MD2ivim.nii.gz"

# Remove everything else in result_folder for this subject except finals
rm -f \
  "result_folder/${SUBJECT_ID}/native_"*.nii.gz \
  "result_folder/${SUBJECT_ID}/flair_stripped"*.nii.gz \
  "result_folder/${SUBJECT_ID}/t1_stripped"*.nii.gz \
  "result_folder/${SUBJECT_ID}/mask_"*.nii.gz \
  "result_folder/${SUBJECT_ID}/flair2b0.mat" \
  "result_folder/${SUBJECT_ID}/flair2b0_affine_tmp.nii.gz" \
  "result_folder/${SUBJECT_ID}/T1w2b0_affine_tmp.nii.gz" \
  "result_folder/${SUBJECT_ID}/tissue2ivim_tmp.nii.gz" \
  "result_folder/${SUBJECT_ID}/lesion2ivim_tmp.nii.gz" \
  "result_folder/${SUBJECT_ID}/fa2ivim_tmp.nii.gz" \
  "result_folder/${SUBJECT_ID}/md2ivim_tmp.nii.gz" \
  "result_folder/${SUBJECT_ID}/dti2b0.mat" \
  "result_folder/${SUBJECT_ID}/dtib0_affine.nii.gz" \
  2>/dev/null || true

echo "[INFO] Kept finals:"
echo "      ${KEEP_B0}"
echo "      ${KEEP_FLAIR}"
echo "      ${KEEP_T1}"
echo "      ${KEEP_TISSUE}"
echo "      ${KEEP_LESION}"
[[ -f "${KEEP_FA}" ]] && echo "      ${KEEP_FA}"
[[ -f "${KEEP_MD}" ]] && echo "      ${KEEP_MD}"
# ======================================================================


######################################## Extraction ###########################################
echo "[INFO] Extracting mean/std b0 by lesion and tissue labels..."

MASTER_CSV="result_folder/b0val_by_tissue_types.csv"
[[ -f "$MASTER_CSV" ]] || echo "SubjectID,TissueType,Mean_b0,Std_b0,VoxelCount" > "$MASTER_CSV"

declare -A label_names=([1]="Periventricular" [2]="DeepWhite" [3]="Juxtacortical" [4]="Infratentorial")
declare -A label_tissues=([1]="CSF" [2]="GM" [3]="WM")

for label in 1 2 3 4; do
	MASK_OUT="result_folder/${SUBJECT_ID}/mask_label${label}.nii.gz"
	fslmaths "${LESION2BRAIN}" -thr $label -uthr $label -bin "$MASK_OUT"
	vox=$(fslstats "$MASK_OUT" -V | awk '{print $1}')
	if [[ "$vox" -gt 0 ]]; then read mean std <<< $(fslstats "$IVIM_b0" -k "$MASK_OUT" -M -S); else mean="NA"; std="NA"; fi
	echo "${SUBJECT_ID},${label_names[$label]},${mean},${std},${vox}" >> "$MASTER_CSV"
	rm -f "$MASK_OUT"
done

for label in 1 2 3; do
	MASK_TISSUE="result_folder/${SUBJECT_ID}/mask_tissue${label}.nii.gz"
	fslmaths "${TISSUE2BRAIN}" -thr $label -uthr $label -bin "$MASK_TISSUE"
	vox=$(fslstats "$MASK_TISSUE" -V | awk '{print $1}')
	if [[ "$vox" -gt 0 ]]; then read mean std <<< $(fslstats "$IVIM_b0" -k "$MASK_TISSUE" -M -S); else mean="NA"; std="NA"; fi
	echo "${SUBJECT_ID},${label_tissues[$label]},${mean},${std},${vox}" >> "$MASTER_CSV"
	rm -f "$MASK_TISSUE"
done

echo "[INFO] Output saved to: $MASTER_CSV"
