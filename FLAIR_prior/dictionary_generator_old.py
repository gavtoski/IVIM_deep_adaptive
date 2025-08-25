"""
Bin Hoang - University of Rochester

dictionary_generator.py

Generate a dictionary that includes b0 and FLAIR signal at the current voxel as well of the mean and max of the current b0 and FLAIR,
along with its tissue
"""

#import libraries
import numpy as np
import nibabel as nib
import os
import argparse
from tqdm import tqdm


def load_nifti(path):
	return nib.load(path).get_fdata()

def process_subject(subject_id, base_path):
	subj_dir = os.path.join(base_path, "result_folder", subject_id)
	tissue_map_path = os.path.join(subj_dir, "tissue2ivim.nii.gz")
	lesion_map_path = os.path.join(subj_dir, "lesion2ivim.nii.gz")
	b0_map_path = os.path.join(subj_dir, "b0_avg.nii.gz")
	flair_map_path = os.path.join(subj_dir, "flair2b0_affine.nii.gz")

	# Check existence
	for path in [tissue_map_path, lesion_map_path, b0_map_path, flair_map_path]:
		if not os.path.exists(path):
			print(f"[SKIP] Missing file for {subject_id}: {path}")
			return None

	tissue_map = load_nifti(tissue_map_path)
	lesion_map = load_nifti(lesion_map_path)
	b0_map = load_nifti(b0_map_path)
	flair_map = load_nifti(flair_map_path)

	b0_mean, b0_max = np.mean(b0_map), np.max(b0_map)
	flair_mean, flair_max = np.mean(flair_map), np.max(flair_map)

	# Flatten
	flat_b0 = b0_map.flatten()
	flat_flair = flair_map.flatten()
	flat_tissue = tissue_map.flatten().astype(int)
	flat_lesion = lesion_map.flatten().astype(int)
	flat_lesion_offset = np.where(flat_lesion > 0, flat_lesion + 10, 0)

	#Check valid voxel and capture indices
	valid_mask = (flat_tissue > 0) | (flat_lesion > 0)
	indices = np.where(valid_mask)[0]
	if len(indices) == 0:
		print(f"[WARN] No valid voxels for {subject_id}")
		return None

	final_labels = np.where(flat_tissue > 0, flat_tissue, flat_lesion_offset)

	#Grab info for each voxel using numpy indices (vectorization to replace for loop)
	return {
		'b0': flat_b0[indices],
		'b0_mean': np.full(len(indices), b0_mean, dtype=np.float32),
		'b0_max': np.full(len(indices), b0_max, dtype=np.float32),
		'FLAIR': flat_flair[indices],
		'FLAIR_mean': np.full(len(indices), flair_mean, dtype=np.float32),
		'FLAIR_max': np.full(len(indices), flair_max, dtype=np.float32),
		'tissue_label': final_labels[indices],
		'subject_id': np.full(len(indices), subject_id)
	}

def merge_and_save(dictionaries, output_path):
	merged = {}
	keys = dictionaries[0].keys()
	for key in keys:
		merged[key] = np.concatenate([d[key] for d in dictionaries if d is not None])
	np.savez_compressed(output_path, **merged)
	print(f"[FINAL] Merged dictionary saved → {output_path} ({len(merged['b0'])} voxels)")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject_list", required=True, help="Text file with subject IDs (one per line)")
	parser.add_argument("--base_path", default="/Users/nhoang2/Dropbox/Classes/IVIM_project/IVIM_deep_adaptive/FLAIR_prior", help="Base path to result_folder/")
	parser.add_argument("--output_path", default="/Users/nhoang2/Dropbox/Classes/IVIM_project/IVIM_deep_adaptive/FLAIR_prior/training_dictionary.npz", help="Output npz file")

	args = parser.parse_args()
	with open(args.subject_list, 'r') as f:
		subject_ids = [line.strip() for line in f.readlines() if line.strip()]

	print(f"[INFO] Processing {len(subject_ids)} subjects...")
	all_dicts = []
	for sid in tqdm(subject_ids):
		d = process_subject(sid, args.base_path)
		if d: all_dicts.append(d)

	if all_dicts:
		merge_and_save(all_dicts, args.output_path)
	else:
		print("[ERROR] No valid subjects processed.")
