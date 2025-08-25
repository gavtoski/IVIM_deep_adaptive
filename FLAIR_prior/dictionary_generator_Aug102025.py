import os
import numpy as np
import nibabel as nib
import pandas as pd
from typing import List, Dict, Tuple

class IVIMProcessor:
    """Simplified IVIM dictionary generator with configurable paths and streamlined processing."""

    def __init__(self):
        # Path templates with {SUBJECTID} placeholder
        self.path_templates = {
            "ivim": "/content/drive/MyDrive/IVIM_NeuroCovid/subject_folder/{SUBJECTID}/preproc_ivim_dwidata.nii.gz",
            "bvals": "/content/drive/MyDrive/IVIM_NeuroCovid/subject_folder/bvals.txt",
            "b0avg": "/content/drive/MyDrive/IVIM_NeuroCovid/subject_folder/{SUBJECTID}/b0_avg.nii.gz",
            "flair": "/content/drive/MyDrive/IVIM_NeuroCovid/subject_folder/{SUBJECTID}/flair2b0_affine.nii.gz",
            "tissue": "/content/drive/MyDrive/IVIM_NeuroCovid/subject_folder/{SUBJECTID}/tissue2ivim.nii.gz",
            "lesion": "/content/drive/MyDrive/IVIM_NeuroCovid/subject_folder/{SUBJECTID}/lesion2ivim.nii.gz",
            "T1": "/content/drive/MyDrive/IVIM_NeuroCovid/subject_folder/{SUBJECTID}/T1w2b0_affine.nii.gz",
            "FA": "/path/to/FA/",
            "MD": "/path/to/MD"
        }

        # Label mappings
        self.tissue_labels = {1: "CSF", 2: "GM", 3: "WM"}
        self.lesion_labels = {1: "WMH_PV", 2: "WMH_Deep", 3: "WMH_Juxtacortical", 4: "WMH_Infratentorial"}
        self.lesion_offset = 10

    def get_subject_paths(self, subject_id: str) -> Dict[str, str]:
        """Generate file paths for a subject using template substitution."""
        return {key: template.format(SUBJECTID=subject_id)
                for key, template in self.path_templates.items()}

    def load_and_validate_data(self, paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Load all data and validate spatial alignment."""
        # Load b-values from file (always 46 values: 15 unique × 3 + 1 padding)
        bvals_file = np.genfromtxt(paths["bvals"], dtype=np.float32)
        print(f"B-values loaded from file: {len(bvals_file)} values")
        print(f"First 10 b-values: {bvals_file[:10]}")

        # Extract unique b-values for reference
        bvals_unique = np.unique(bvals_file)
        print(f"Unique b-values: {bvals_unique}")
        print(f"Number of unique b-values: {len(bvals_unique)}")

        # Load IVIM to check timepoints
        ivim_img = nib.load(paths["ivim"])
        ivim_data = ivim_img.get_fdata().astype(np.float32)

        print(f"IVIM shape: {ivim_data.shape}")
        T = ivim_data.shape[3]

        # Handle different acquisition protocols based on IVIM timepoints
        if T == 46:
            # Case 1: 46 timepoints - use b-values as-is
            print("46 timepoints detected: 15 unique b-values × 3 repetitions + 1 padding")
            bvals_final = bvals_file

        elif T == 67:
            # Case 2: 67 timepoints - truncate IVIM to 46 and use standard b-values
            print("67 timepoints detected: truncating to 46 timepoints")
            ivim_data = ivim_data[:, :, :, :46]
            bvals_final = bvals_file
            T = 46  # Update T after truncation

        elif T == 91:
            # Case 3: 91 timepoints - expand b-values to match (15 unique × 6 + 1)
            print("91 timepoints detected: 15 unique b-values × 6 repetitions + 1 padding")
            # Remove the first padding from the 46 b-values to get the 45 actual measurements
            bvals_no_padding = bvals_file[1:]  # Remove first 0
            # Get unique values (should be 15)
            unique_vals = []
            seen = set()
            for val in bvals_no_padding:
                if val not in seen:
                    unique_vals.append(val)
                    seen.add(val)
            # Repeat each unique value 6 times
            bvals_expanded = np.repeat(unique_vals, 6)
            # Add padding at the beginning
            bvals_final = np.concatenate([[0], bvals_expanded])

        else:
            raise ValueError(f"Unsupported IVIM timepoints: {T}. Expected 46, 67, or 91.")

        print(f"Final IVIM shape after processing: {ivim_data.shape}")
        print(f"Final b-values length: {len(bvals_final)}")

        data = {
            "ivim": ivim_data,
            "b0avg": nib.load(paths["b0avg"]).get_fdata().astype(np.float32),
            "flair": nib.load(paths["flair"]).get_fdata().astype(np.float32),
            "T1": nib.load(paths["T1"]).get_fdata().astype(np.float32),
            "tissue": nib.load(paths["tissue"]).get_fdata().astype(np.int16),
            "lesion": nib.load(paths["lesion"]).get_fdata().astype(np.int16),
            "bvals": bvals_final
        }

        # Validate shapes (simplified - just check first 3 dimensions)
        ref_shape = data["ivim"].shape[:3]
        for key in ["b0avg", "flair", "tissue", "lesion"]:
            if data[key].shape[:3] != ref_shape:
                raise ValueError(f"Shape mismatch: {key} {data[key].shape[:3]} vs IVIM {ref_shape}")

        if data["ivim"].ndim != 4:
            raise ValueError(f"IVIM must be 4D, got {data['ivim'].ndim}D with shape {data['ivim'].shape}")

        if data["ivim"].shape[3] != len(data["bvals"]):
            raise ValueError(f"IVIM time dimension ({data['ivim'].shape[3]}) doesn't match b-values length ({len(data['bvals'])})")

        return data

    def create_labels(self, tissue: np.ndarray, lesion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create final labels with lesion override. Lesion labels get +10 offset to avoid conflicts."""
        # Start with tissue labels (1=CSF, 2=GM, 3=WM)
        labels_int = tissue.copy()

        # Override with lesion labels where present (1=PV, 2=Deep, 3=Juxtacortical, 4=Infratentorial)
        # Add offset to avoid conflicts: lesion labels become 11, 12, 13, 14
        lesion_mask = lesion > 0
        labels_int[lesion_mask] = lesion[lesion_mask] + self.lesion_offset

        # Create string labels - lesion overrides tissue
        labels_str = np.empty_like(labels_int, dtype=object)
        for val in np.unique(labels_int):
            mask = labels_int == val
            if val >= 11:  # Lesion labels (with offset)
                lesion_type = val - self.lesion_offset
                labels_str[mask] = self.lesion_labels.get(lesion_type, "UNKNOWN_LESION")
            else:  # Tissue labels
                labels_str[mask] = self.tissue_labels.get(val, "UNKNOWN_TISSUE")

        return labels_int, labels_str

    def process_ivim_signals(self, ivim_data: np.ndarray, bvals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process IVIM signals: sort by mean intensity and normalize."""
        X, Y, Z, T = ivim_data.shape
        ivim_flat = ivim_data.reshape(-1, T)

        # Sort volumes by mean intensity (descending)
        mean_intensities = np.nanmean(ivim_flat, axis=0)
        sort_order = np.argsort(mean_intensities)[::-1]
        bvals_sorted = bvals[sort_order]
        ivim_sorted = ivim_flat[:, sort_order]

        # Compute S0 (mean of b=0 volumes) and normalize
        b0_mask = bvals_sorted == 0
        S0 = np.nanmean(ivim_sorted[:, b0_mask], axis=1)
        S0[S0 <= 0] = np.nan

        # Valid voxels: S0 > 0.5 * median(S0)
        valid_S0 = S0[S0 > 0]
        valid_mask = S0 > (0.5 * np.nanmedian(valid_S0)) if len(valid_S0) > 0 else np.zeros_like(S0, dtype=bool)

        # Normalize signals
        ivim_normalized = ivim_sorted / S0[:, np.newaxis]

        return ivim_normalized, bvals_sorted, valid_mask

    def sample_voxels(self, labels: np.ndarray, valid_mask: np.ndarray,
                     n_per_class: int = 500, seed: int = 69) -> np.ndarray:
        """Sample up to n_per_class voxels per label, return flat indices."""
        rng = np.random.default_rng(seed)
        selected_flat_indices = []

        # Get valid voxel indices and their labels
        valid_flat_indices = np.where(valid_mask)[0]
        valid_labels = labels[valid_mask]

        for label in np.unique(valid_labels):
            if label == 0:
                continue

            # Find flat indices for this label among valid voxels
            label_mask_in_valid = valid_labels == label
            label_flat_indices = valid_flat_indices[label_mask_in_valid]

            if len(label_flat_indices) > n_per_class:
                # Randomly sample n_per_class voxels
                label_flat_indices = rng.choice(label_flat_indices, size=n_per_class, replace=False)

            selected_flat_indices.extend(label_flat_indices)

        return np.array(selected_flat_indices, dtype=np.int64)

    def process_subject(self, subject_id: str, n_per_class: int = 500, seed: int = 69) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Process a single subject:
        1. Load 4D IVIM volume and corresponding maps
        2. Sample voxels from valid regions
        3. For each selected voxel, extract its full time series (all b-values) AND corresponding values from other maps
        """
        print(f"\nProcessing subject: {subject_id}")
        print("="*50)

        # Load and validate data
        paths = self.get_subject_paths(subject_id)
        data = self.load_and_validate_data(paths)

        # Get volume dimensions
        X, Y, Z, T = data["ivim"].shape

        # Flatten all volumes for consistent indexing
        ivim_flat = data["ivim"].reshape(-1, T)  # (N_voxels, T_timepoints)
        b0avg_flat = data["b0avg"].flatten()     # (N_voxels,)
        flair_flat = data["flair"].flatten()     # (N_voxels,)
        tissue_flat = data["tissue"].flatten()   # (N_voxels,)
        lesion_flat = data["lesion"].flatten()   # (N_voxels,)
        T1_flat = data["T1"].flatten()           # (N_voxels,)

        # Sort volumes by mean intensity and get sorted b-values
        mean_intensities = np.nanmean(ivim_flat, axis=0)
        sort_order = np.argsort(mean_intensities)[::-1]
        bvals_sorted = data["bvals"][sort_order]
        ivim_sorted = ivim_flat[:, sort_order]  # Apply same sorting to all voxel time series

        # Compute S0 (mean of b=0 volumes) and normalize each voxel's time series
        b0_mask = bvals_sorted == 0
        n_b0 = np.sum(b0_mask)
        print(f"Found {n_b0} b=0 volumes for S0 calculation")

        S0 = np.nanmean(ivim_sorted[:, b0_mask], axis=1) if n_b0 > 0 else np.ones(ivim_flat.shape[0])
        S0[S0 <= 0] = np.nan

        # Define valid voxels
        valid_S0 = S0[~np.isnan(S0) & (S0 > 0)]
        if len(valid_S0) > 0:
            threshold = 0.5 * np.nanmedian(valid_S0)
            valid_mask = S0 > threshold
            print(f"Valid voxels: {np.sum(valid_mask)} out of {len(S0)} (threshold: {threshold:.2f})")
        else:
            valid_mask = np.zeros_like(S0, dtype=bool)
            print("Warning: No valid S0 values found")

        # Normalize IVIM signals (ensure float division)
        with np.errstate(divide='ignore', invalid='ignore'):
            ivim_normalized = ivim_sorted / S0[:, np.newaxis]
            ivim_normalized = np.nan_to_num(ivim_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # Create final labels (lesion overrides tissue)
        labels_int, labels_str = self.create_labels(tissue_flat, lesion_flat)

        # Report label distribution
        unique_labels, counts = np.unique(labels_int[valid_mask], return_counts=True)
        print("\nLabel distribution in valid voxels:")
        for label, count in zip(unique_labels, counts):
            if label > 0:
                if label >= 11:
                    label_name = self.lesion_labels.get(label - self.lesion_offset, "UNKNOWN")
                else:
                    label_name = self.tissue_labels.get(label, "UNKNOWN")
                print(f"  {label_name}: {count} voxels")

        # Sample voxels (only from valid ones)
        selected_flat_indices = self.sample_voxels(labels_int, valid_mask, n_per_class, seed)
        print(f"\nSampled {len(selected_flat_indices)} voxels total")

        if len(selected_flat_indices) == 0:
            return pd.DataFrame(), np.array([]), bvals_sorted

        # Extract normalized IVIM time series for selected voxels
        selected_signals = ivim_normalized[selected_flat_indices]     # (N_selected, T_timepoints)

        # Average signals by unique b-values to avoid redundant columns
        unique_bvals = np.unique(bvals_sorted)
        averaged_signals = np.zeros((len(selected_flat_indices), len(unique_bvals)))

        for i, b_val in enumerate(unique_bvals):
            # Find all timepoints with this b-value
            b_mask = bvals_sorted == b_val
            # Average across repetitions for this b-value
            averaged_signals[:, i] = np.mean(selected_signals[:, b_mask], axis=1)

        print(f"Averaged signals from {T} timepoints to {len(unique_bvals)} unique b-values")

        df = pd.DataFrame({
            "SubjectID": subject_id,
            "tissue_lesion_label": labels_str[selected_flat_indices],  # Final label name (e.g., "WM", "WMH_PV")
            "b0_avg": b0avg_flat[selected_flat_indices],               # Single b0 value per voxel
            "flair_signal": flair_flat[selected_flat_indices],          # Single FLAIR value per voxel
            "T1_signal": T1_flat[selected_flat_indices]                # Single T1 value per voxel
        })

        # Add averaged IVIM signals as columns (one per unique b-value)
        for i, b_val in enumerate(unique_bvals):
            df[f"ivim_b{int(b_val):04d}"] = averaged_signals[:, i]

        print(f"Created DataFrame with shape: {df.shape}")

        return df, averaged_signals, unique_bvals

    def process_multiple_subjects(self, subject_ids: List[str], output_dir: str,
                                n_per_class: int = 500, seed: int = 69) -> Tuple[str, str]:
        """Process multiple subjects and save combined results."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        all_dfs = []
        all_signals = []
        bvals_ref = None

        for subject_id in subject_ids:
            try:
                df, signals, bvals = self.process_subject(subject_id, n_per_class, seed)

                if df.empty:
                    print(f"Warning: No valid data for subject {subject_id}")
                    continue

                # Ensure consistent b-values across subjects
                if bvals_ref is None:
                    bvals_ref = bvals
                elif not np.array_equal(bvals, bvals_ref):
                    raise ValueError(f"Unique b-values mismatch for subject {subject_id}")

                all_dfs.append(df)
                all_signals.append(signals)

            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
                import traceback
                traceback.print_exc()  # Print full error traceback for debugging
                continue

        if not all_dfs:
            raise RuntimeError("No subjects processed successfully")

        # Combine results
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_signals = np.vstack(all_signals)

        # Save files
        csv_path = os.path.join(output_dir, "ivim_dictionary.csv")
        npz_path = os.path.join(output_dir, "ivim_dictionary.npz")

        # Save CSV (already has signal columns from process_subject)
        combined_df.to_csv(csv_path, index=False)

        # Compact NPZ with metadata and signals
        np.savez_compressed(
            npz_path,
            SubjectID=combined_df["SubjectID"].values,
            tissue_lesion_label=combined_df["tissue_lesion_label"].values,
            b0_avg=combined_df["b0_avg"].values,
            flair_signal=combined_df["flair_signal"].values,
            T1_signal=combined_df["T1_signal"].values,
            signals=combined_signals,
            bvals=bvals_ref
        )

        print(f"\n{'='*50}")
        print(f"SUMMARY:")
        print(f"Processed {len(combined_df)} voxels from {len(all_dfs)} subjects")
        print(f"Saved: {csv_path} and {npz_path}")
        print(f"Final DataFrame shape: {combined_df.shape}")
        print(f"Signal array shape: {combined_signals.shape}")

        # Report final label distribution
        print("\nFinal label distribution:")
        label_counts = combined_df["tissue_lesion_label"].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} voxels")

        return npz_path, csv_path


# Usage example
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from typing import Dict, List, Tuple

    processor = IVIMProcessor()
    subject_ids = ["NC145", "NC144", "NC142", "NC135", "NC130" ]  # Test with just one subject first
    processor.process_multiple_subjects(
        subject_ids,
        output_dir="/content/drive/MyDrive/IVIM_NeuroCovid/tissue_dictionary",
        n_per_class=500,
        seed=69
    )