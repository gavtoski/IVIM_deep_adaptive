#!/bin/bash

# Inputs
MODEL_TYPE="$1"
NOISE="${2:-nonoise}"	# Default to 'nonoise' if not provided
TISSUE_FILTER="$3"		# Optional third argument to filter a specific tissue

if [[ -z "$TISSUE_FILTER" ]]; then
	echo "[INFO] No tissue filter specified — running all tissue types."
fi

if [[ "$MODEL_TYPE" != "2C" && "$MODEL_TYPE" != "3C" ]]; then
	echo "[USAGE] $0 <MODEL_TYPE> [NOISE_MODE] [TISSUE_TYPE]"
	echo "Example: $0 2C lownoise WMH"
	exit 1
fi

if [[ -z "$NOISE" ]]; then
	echo "[ERROR] Missing NOISE_MODE argument."
	echo "Example: $0 2C lownoise"
	exit 1
fi

# Global Params
SEED="69"
BVALS="/scratch/nhoang2/IVIM_NeuroCovid/Data/bvals.txt"
DEST_ROOT="/scratch/nhoang2/IVIM_NeuroCovid/Result/Test_SingleRun_${MODEL_TYPE}_${NOISE}"
INPUT_TYPE="array"
USE_3C=$([[ "$MODEL_TYPE" == "3C" ]] && echo "True" || echo "False")

# Tissue types to process
declare -a SUBJECTS=("S1_signal" "NAWM_signal" "WMH_signal")

# Loop over tissue types
for SUBJECT in "${SUBJECTS[@]}"; do
	# Determine tissue_type from SUBJECT name
	if [[ "$SUBJECT" == "WMH_signal" ]]; then
		tissue_type="WMH"
	elif [[ "$SUBJECT" == "NAWM_signal" ]]; then
		tissue_type="NAWM"
	else
		tissue_type="mixed"
	fi

	# Skip if tissue filter is specified and doesn't match
	if [[ -n "$TISSUE_FILTER" && "$tissue_type" != "$TISSUE_FILTER" ]]; then
		continue
	fi

	PREPROC="/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_${MODEL_TYPE}_seed${SEED}_${NOISE}_train/${SUBJECT}.npy"
	VAL_LOC="/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_${MODEL_TYPE}_seed${SEED}_${NOISE}_val/${SUBJECT}.npy"
	GT_LOC="/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_${MODEL_TYPE}_seed${SEED}_nonoise_val/${SUBJECT}.npy"

	# Run Adaptive Mode (OriginalOFF, TuneON, FreezeON, BoostOFF)
	if [[ "$MODEL_TYPE" == "3C" ]]; then
		for IR in False True; do
			dest_dir="${DEST_ROOT}/${SUBJECT}/${MODEL_TYPE}_OriginalOFF_TuneON_FreezeON_BoostOFF_none_IR${IR}_${tissue_type}"
			echo "[RUN] ${MODEL_TYPE} | Adaptive | $SUBJECT | Noise=${NOISE} | IR=${IR} | Tissue=${tissue_type} → $dest_dir"
			sbatch IVIM_NNsynthetic_single.sh "$PREPROC" "$VAL_LOC" "$GT_LOC" "$BVALS" "$dest_dir" \
				False True "$IR" True False none "$USE_3C" "$INPUT_TYPE" "$tissue_type" None
		done
	else
		dest_dir="${DEST_ROOT}/${SUBJECT}/${MODEL_TYPE}_OriginalOFF_TuneON_FreezeON_BoostOFF_none_IRFalse_${tissue_type}"
		echo "[RUN] ${MODEL_TYPE} | Adaptive | $SUBJECT | Noise=${NOISE} | IR=False | Tissue=${tissue_type} → $dest_dir"
		sbatch IVIM_NNsynthetic_single.sh "$PREPROC" "$VAL_LOC" "$GT_LOC" "$BVALS" "$dest_dir" \
			False True False True False none "$USE_3C" "$INPUT_TYPE" "$tissue_type" None
	fi

	# Run Original Mode (OriginalON, TuneOFF, FreezeON, BoostOFF)
	if [[ "$MODEL_TYPE" == "3C" ]]; then
		for IR in False True; do
			dest_dir="${DEST_ROOT}/${SUBJECT}/${MODEL_TYPE}_OriginalON_IR${IR}_${tissue_type}"
			echo "[RUN] ${MODEL_TYPE} | OriginalON | $SUBJECT | Noise=${NOISE} | IR=${IR} | Tissue=${tissue_type} → $dest_dir"
			sbatch IVIM_NNsynthetic_single.sh "$PREPROC" "$VAL_LOC" "$GT_LOC" "$BVALS" "$dest_dir" \
				True False "$IR" False False none "$USE_3C" "$INPUT_TYPE" "$tissue_type" None
		done
	else
		dest_dir="${DEST_ROOT}/${SUBJECT}/${MODEL_TYPE}_OriginalON_IRFalse_${tissue_type}"
		echo "[RUN] ${MODEL_TYPE} | OriginalON | $SUBJECT | Noise=${NOISE} | IR=False | Tissue=${tissue_type} → $dest_dir"
		sbatch IVIM_NNsynthetic_single.sh "$PREPROC" "$VAL_LOC" "$GT_LOC" "$BVALS" "$dest_dir" \
			True False False False False none "$USE_3C" "$INPUT_TYPE" "$tissue_type" None
	fi
done
