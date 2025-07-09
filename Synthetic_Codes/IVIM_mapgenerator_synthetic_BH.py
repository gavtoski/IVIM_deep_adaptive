# IVIM code to train a NN and fit given a DWI image, adapted from Oliver Champion and Paulien Voorter
# Original code source: https://github.com/paulienvoorter/IVIM3brain-NET
# Original adaptive code: https://github.com/BinHoang/IVIM-3C-adaptive
# For pretrained net, please modify code according using the option in the NN class
# By Bin Hoang, University of Rochester, Department of Physics
# Email: nhat_hoang@urmc.rochester.edu


import os
import time
import nibabel as nib
import numpy as np
import torch
import sys
import tqdm
import json

# Load IVIM neural net dependencies from main folder
sys.path.append('/scratch/nhoang2/IVIM_NeuroCovid/Synthetic_Codes/IVIMNET/')
sys.path.append('/scratch/nhoang2/IVIM_NeuroCovid/Synthetic_Codes/')

import IVIMNET.deep2_adaptive_hybrid_ablation_expanded as deep2
import IVIMNET.deep as deep
#from IVIMNET.fitting_algorithms import fit_dats
from hyperparams import hyperparams as hp
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Create IVIM class object
class map_generator_NN():

	def __init__(self, train_input_path, bvalues_path, dest_dir, arg, pre_trained_net=None, original_mode=False, weight_tuning= False, IR=False, freeze_param= False,
				 boost_toggle=True, ablate_option=None, use_three_compartment=False, input_type="image",tissue_type="mixed", custom_dict=None, val_input_path=None):
		self.train_input_path = train_input_path
		self.bvalues_path = bvalues_path
		self.arg = arg
		self.dest_dir = dest_dir
		self.pre_trained_net = pre_trained_net
		self.original_mode = original_mode
		self.IR=IR
		self.freeze_param = freeze_param
		self.weight_tuning = weight_tuning
		self.boost_toggle= boost_toggle
		self.ablate_option= ablate_option
		self.input_type = input_type
		self.use_three_compartment = use_three_compartment
		self.tissue_type = tissue_type
		self.custom_dict = custom_dict
		
		if val_input_path is None:
		    self.val_input_path = train_input_path
		    print("[WARNING]: validation set missing — defaulting to training set for validation")
		else:
		    self.val_input_path = val_input_path
		    print("[INFO]: Using customized validation set")


		if not hasattr(arg, 'net_pars') or arg.net_pars is None:
			print("[INFO] net_pars not found in arg — defaulting to brain3 config.")
			if self.use_three_compartment:
				arg.net_pars = net_pars("brain3")
			else:
				arg.net_pars = net_pars("brain2")


		# Set subject tag for saving logs
		# Get base tag like "NAWM_signal.npy" → "NAWM"
		basename = os.path.basename(self.train_input_path).replace('.npy', '').replace('.nii.gz', '')
		label_guess = basename.split("_")[0]  # "NAWM", "S1", "WMH", etc.
		self.arg.subject_tag = label_guess
		print(self.arg.subject_tag)

	def load_and_preproc_data(self):
	    # Load and reorder b-values
	    bval_txt = np.genfromtxt(self.bvalues_path)
	    self.bvalues_raw = np.array(bval_txt)
	    self.n_b_values = len(self.bvalues_raw)

	    # Determine sorting order using TRAIN input (assume train and val has the same format, they should)
	    if self.input_type == "image":
	        self.data = nib.load(self.train_input_path)
	        train_vol = self.data.get_fdata()  # Shape: [X, Y, Z, B]
	        self.sx, self.sy, self.sz, _ = train_vol.shape
	        mean_signals = np.mean(train_vol.reshape(-1, self.n_b_values), axis=0)
	    elif self.input_type == "array":
	        train_vol = np.load(self.train_input_path)  # Shape: [N, B]
	        mean_signals = np.mean(train_vol, axis=0)
	    else:
	        raise ValueError(f"[ERROR] Unsupported input_type: {self.input_type}")

	    self.sorted_indices = np.argsort(mean_signals)[::-1]
	    self.bvalues = self.bvalues_raw[self.sorted_indices]
	    self.selsb = self.bvalues == 0

	    # Subfunction to clean data
	    def clean_and_normalize(data, label):
	        if self.input_type == "image":
	            data = data[..., self.sorted_indices]
	            flat = data.reshape(-1, self.n_b_values)
	        else:
	            flat = data[:, self.sorted_indices]

	        # Filter: voxels with strong S0
	        S0 = np.nanmean(flat[:, self.selsb], axis=1)
	        S0[S0 == 0] = np.nan
	        valid = S0 > 0.5 * np.nanmedian(S0[S0 > 0])
	        filtered = flat[valid]
	        S0_filtered = np.nanmean(filtered[:, self.selsb], axis=1).astype('<f')
	        norm = filtered / S0_filtered[:, None]

	        print(f"[INFO] {label}: {norm.shape[0]} voxels retained")
	        return norm, valid

	    # Clean training data 
	    print(f"[INFO] Preprocessing training data from {self.train_input_path}")
	    if self.input_type == "image":
	        train_data = self.data.get_fdata()
	    else:
	        train_data = np.load(self.train_input_path)
	    self.datatot_train, self.valid_id_train = clean_and_normalize(train_data, "Train")

	    # Clean validation data
	    print(f"[INFO] Preprocessing validation data from {self.val_input_path}")
	    if self.input_type == "image":
	        val_data = nib.load(self.val_input_path).get_fdata()
	    else:
	        val_data = np.load(self.val_input_path)
	    
	    self.datatot_val, self.valid_id_val = clean_and_normalize(val_data, "Val")

	    # Post-filter summary
	    print(f"[INFO] Voxel summary — Train: {self.datatot_train.shape[0]} | Val: {self.datatot_val.shape[0]}")
	    if self.datatot_train.shape[0] == 0:
	        raise ValueError("[ERROR] No valid voxels in training set after S0 filtering.")
	    if self.datatot_val.shape[0] == 0:
	        raise ValueError("[ERROR] No valid voxels in validation set after S0 filtering.")




	def train_NN(self):
	    # Load training and validation data
	    self.load_and_preproc_data()

	    # Remove rows with NaNs in training data
	    res = ~np.isnan(self.datatot_train).any(axis=1)  # Boolean mask

	    # Assign destination directory
	    self.arg.train_pars.dest_dir = self.dest_dir

	    # Disable tuning if in original mode
	    if self.original_mode:
	        self.weight_tuning = False
	        self.freeze_param = False
	        self.boost_toggle = False
	        self.ablate_option = "none"

	    # Train neural network using training data
	    start_time = time.time()
	    self.net = deep2.learn_IVIM(
	        self.datatot_train[res], self.bvalues, self.arg,
	        original_mode=self.original_mode,
	        weight_tuning=self.weight_tuning,
	        IR=self.IR,
	        freeze_param=self.freeze_param,
	        boost_toggle=self.boost_toggle,
	        ablate_option=self.ablate_option,
	        use_three_compartment=self.use_three_compartment,
	        tissue_type=self.tissue_type,
	        custom_dict=self.custom_dict
	    )
	    elapsed_time = time.time() - start_time
	    print(f"\n[INFO] Time elapsed for Net training: {elapsed_time:.2f} seconds\n")


	def reconstruct_IVIM(self, IVIM_maps, norm=True):
	    """
	    Reconstruct IVIM signal using model parameters.
	    Supports:
	    - 3C: [Dpar, fmv, Dmv, Dint, fint, S0]
	    - 2C: [Dpar, fmv, Dmv, S0]
	    IR modeling is only applied for 3C.
	    """
	    # Detect model type by length
	    if len(IVIM_maps) == 6:
	        model_type = "3C"
	        Dpar, fmv, Dmv, Dint, fint, S0 = IVIM_maps
	    elif len(IVIM_maps) == 4:
	        model_type = "2C"
	        Dpar, fmv, Dmv, S0 = IVIM_maps
	    else:
	        raise ValueError(f"[ERROR] Invalid IVIM_maps length: {len(IVIM_maps)}")

	    # Reshape b-values
	    if self.input_type == "image":
	        b_values = self.bvalues.reshape((1, 1, 1, -1))
	    else:
	        b_values = self.bvalues.reshape((1, -1))
	    b_values = np.sort(b_values)

	    # Clean NaNs
	    Dpar = np.nan_to_num(Dpar, nan=0)
	    Dmv  = np.nan_to_num(Dmv,  nan=0)
	    fmv  = np.nan_to_num(fmv,  nan=0)
	    S0   = np.nan_to_num(S0,   nan=1)

	    use_IR = self.IR if model_type == "3C" else False
	    if self.IR and model_type == "2C":
	        print("[WARNING] IR mode is not supported for 2C — ignoring IR flag.")

	    if model_type == "2C":
	        scale = 1 if norm else S0[..., None]
	        signal = scale * (
	            fmv[..., None] * np.exp(-b_values * Dmv[..., None]) +
	            (1 - fmv[..., None]) * np.exp(-b_values * Dpar[..., None])
	        )

	    elif model_type == "3C":
	        Dint = np.nan_to_num(Dint, nan=0)
	        fint = np.nan_to_num(fint, nan=0)

	        if not use_IR:
	            scale = 1 if norm else S0[..., None]
	            signal = scale * (
	                fmv[..., None] * np.exp(-b_values * Dmv[..., None]) +
	                fint[..., None] * np.exp(-b_values * Dint[..., None]) +
	                (1 - fmv[..., None] - fint[..., None]) * np.exp(-b_values * Dpar[..., None])
	            )
	        else:
	            # IR constants
	            TE, TR, TI = 84, 6800, 2230
	            T1_tissue, T2_tissue = 1081, 95
	            T1_isf, T2_isf = 1250, 503
	            T1_blood, T2_blood = 1624, 275

	            fpar = 1 - fmv - fint

	            num = (
	                fpar[..., None] * (1 - 2 * np.exp(-TI / T1_tissue) + np.exp(-TR / T1_tissue)) *
	                np.exp(-TE / T2_tissue - b_values * Dpar[..., None]) +
	                fint[..., None] * (1 - 2 * np.exp(-TI / T1_isf) + np.exp(-TR / T1_isf)) *
	                np.exp(-TE / T2_isf - b_values * Dint[..., None]) +
	                fmv[..., None] * (1 - np.exp(-TR / T1_blood)) *
	                np.exp(-TE / T2_blood - b_values * Dmv[..., None])
	            )
	            denom = (
	                fpar[..., None] * (1 - 2 * np.exp(-TI / T1_tissue) + np.exp(-TR / T1_tissue)) * np.exp(-TE / T2_tissue) +
	                fint[..., None] * (1 - 2 * np.exp(-TI / T1_isf) + np.exp(-TR / T1_isf)) * np.exp(-TE / T2_isf) +
	                fmv[..., None] * (1 - np.exp(-TR / T1_blood)) * np.exp(-TE / T2_blood)
	            )

	            signal = (1 if norm else S0[..., None]) * (num / denom)

	    return signal



	def calculate_nrmse_and_plot(self, IVIM_reconstructed, norm=True):
	    today = datetime.today().strftime("%Y-%m-%d")

	    # Use preprocessed, normalized val data
	    S_original = self.datatot_val.copy()
	    S_reconstructed = np.nan_to_num(IVIM_reconstructed, nan=0)

	    # Align masks
	    brain_mask = self.valid_id_val if self.input_type == "array" else np.ones_like(S_reconstructed[..., 0], dtype=bool)
	    S_reconstructed[~brain_mask] = 0

	    # Compute NRMSE
	    squared_error = (S_original - S_reconstructed) ** 2
	    mse_map = np.mean(squared_error, axis=-1)
	    rmse_map = np.sqrt(mse_map)
	    norm_map = np.linalg.norm(S_original, axis=-1)
	    avg_nrmse_map = np.divide(rmse_map, norm_map, out=np.zeros_like(rmse_map), where=norm_map != 0)

	    # Save NRMSE map if image input
	    if self.input_type == "image":
	        nrmse_nifti = nib.Nifti1Image(avg_nrmse_map, affine=self.data.affine, header=self.data.header)
	        nib.save(nrmse_nifti, os.path.join(self.arg.train_pars.dest_dir, "nrmse_map.nii.gz"))

	    # Save visualizations in structured folder
	    dest_dir = self.arg.train_pars.dest_dir
	    result_base = os.path.dirname(os.path.dirname(dest_dir))
	    mode_tag = os.path.basename(dest_dir)
	    save_dir = os.path.join(result_base, f"loss_log_allpenalty_{today}", mode_tag)
	    os.makedirs(save_dir, exist_ok=True)

	    # Save global NRMSE
	    global_nrmse = np.mean(avg_nrmse_map)
	    print(f"Global NRMSE: {global_nrmse}")
	    model_tag = "IVIM3C" if self.use_three_compartment else "IVIM2C"
	    with open(os.path.join(save_dir, f"global_nrmse_{model_tag}.txt"), "w") as f:
	        f.write(f"{global_nrmse}\n")
	    # Also save a copy in main output folder
		#with open(os.path.join(self.arg.train_pars.dest_dir, f"global_nrmse_{model_tag}.txt"), "w") as f:
		#    f.write(f"{global_nrmse}\n")
    

	    # Plot NRMSE histogram
	    if self.input_type == "image":
	        plt.figure(figsize=(10, 6))
	        plt.hist(avg_nrmse_map.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
	        plt.xlabel("NRMSE")
	        plt.ylabel("Frequency")
	        plt.title("Distribution of NRMSE Across Voxels")
	        plt.grid(axis='y', linestyle='--', alpha=0.7)
	        plt.savefig(os.path.join(save_dir, "NRMSE_Dist_plot.png"))
	        print("NRMSE map and distribution plot saved successfully.")

	    # Plot signal fit from center voxel
	    if self.input_type == "image":
	        x, y, z = avg_nrmse_map.shape[0] // 2, avg_nrmse_map.shape[1] // 2, avg_nrmse_map.shape[2] // 2
	        idx_voxel = np.ravel_multi_index((x, y, z), dims=avg_nrmse_map.shape)
	    else:
	        idx_voxel = S_original.shape[0] // 2

	    true_signal_voxel = S_original[idx_voxel]
	    reconstructed_signal_voxel = S_reconstructed[idx_voxel]

	    b_values = np.sort(self.bvalues.flatten())
	    plt.figure(figsize=(10, 8))
	    plt.scatter(b_values, true_signal_voxel, color='red', label="True Signal", marker="o", s=50)
	    plt.plot(b_values, reconstructed_signal_voxel, label='Reconstructed IVIM Signal', linestyle="--", linewidth=2)
	    plt.xlabel("b-value (s/mm²)")
	    plt.ylabel("Signal Intensity")
	    plt.title("IVIM Reconstructed Signal vs b-values")
	    plt.legend()
	    plt.grid(True)
	    plt.savefig(os.path.join(save_dir, f"IVIM_{model_tag}_fit_curve.png"))
	    plt.close()

	    print("Reconstructed IVIM curve fit saved successfully.")
	    return global_nrmse



	def predict_IVIM_maps(self, return_maps=False):
	    print(f"[INFO] Running predict_IVIM() with input_type: {self.input_type}")
	    assert self.input_type in ("image", "array"), f"[ERROR] Invalid input_type: {self.input_type}"

	    # [DATA TRAINING] Train if there is no pre-trained net 
	    if self.pre_trained_net is None:
	        print("[WARNING] No pre-trained net provided — training using training data")
	        self.train_NN()
	    else:
	        print("[INFO] Using pre-trained net for inference only.")
	        self.net = self.pre_trained_net.to(self.arg.train_pars.device)

	    # [INFERENCE] Run inference on VAL data 
	    start_time = time.time()
	    paramsNN = deep2.predict_IVIM(self.datatot_val, self.bvalues, self.net, self.arg)
	    elapsed_time = time.time() - start_time
	    print(f"\nTime elapsed for Net inference: {elapsed_time:.2f} seconds\n")

	    if self.arg.train_pars.use_cuda:
	        torch.cuda.empty_cache()

	    # [MODEL SAVING PARAMS] Define model type and parameter names
	    model_type = "3C" if self.use_three_compartment else "2C"

	    names = {
	        "3C": ['Dpar_NN_triexp', 'fmv_NN_triexp', 'Dmv_NN_triexp', 'Dint_NN_triexp', 'fint_NN_triexp', 'S0_NN_triexp'],
	        "2C": ['Dpar_NN_biexp', 'Dmv_NN_biexp', 'fmv_NN_biexp', 'S0_NN_biexp']
	    }[model_type]

	    # Determine shape and initialize outputs 
	    if self.input_type == "image":
	        n_voxels = self.sx * self.sy * self.sz
	        output_shape = (self.sx, self.sy, self.sz)
	    else:
	        n_voxels = self.datatot_val.shape[0]
	        output_shape = (n_voxels,)

	    saved_files = []
	    ivim_maps = []

	    # [SAVE MAPS] Save each parameter map
	    for k, name in enumerate(names):
	        img = np.zeros(n_voxels)
	        img[self.valid_id_val] = paramsNN[k][:np.sum(self.valid_id_val)]
	        img = np.where(np.isnan(img) | (img < 0), 0, img)

	        if self.input_type == "image":
	            img_reshaped = img.reshape(output_shape)
	            path = os.path.join(self.dest_dir, f"{name}.nii.gz")
	            nib.save(nib.Nifti1Image(img_reshaped, self.data.affine, self.data.header), path)
	        else:
	            path = os.path.join(self.dest_dir, f"{name}.npy")
	            np.save(path, img.astype(np.float32))

	        saved_files.append(path)
	        ivim_maps.append(img)

	    if not saved_files:
	        raise RuntimeError("[ERROR] No IVIM parameter files were saved!")

	    print("[INFO] IVIM parameter files saved:")
	    for f in saved_files:
	        print(f" - {f}")

	    # [ERROR COMPUTATION] Reconstruct signal + compute NRMSE
	    IVIM_reconstructed = self.reconstruct_IVIM(ivim_maps, norm=True)
	    global_nrmse = self.calculate_nrmse_and_plot(IVIM_reconstructed)

	    if return_maps:
	        return global_nrmse, ivim_maps
	    else:
	        print("Maps saved as NIfTI files or .npy arrays.")
	        return global_nrmse



def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Generate IVIM maps.")

    # Add command-line arguments
    parser.add_argument('--train_loc', type=str, required=True, help='Path to the preprocess/train data')
    parser.add_argument('--bval_path', type=str, default='/scratch/nhoang2/IVIM_NeuroCovid/Data/bvals.txt',
                        help='Path to the appropriate bval file (default: shared 15 bvals)')
    parser.add_argument('--dest_dir', type=str, required=True, help='Destination folder')

    # Add training options
    parser.add_argument('--original_mode', type=str2bool, required=True, help='Enable Original Mode')
    parser.add_argument('--weight_tuning', type=str2bool, required=True, help='Enable weight tuning by bvals')
    parser.add_argument('--IR', type=str2bool, required=True, help='Enable Inversion Recovery modeling')
    parser.add_argument('--freeze_param', type=str2bool, required=True, help='Freeze Dpar during tuning')
    parser.add_argument('--boost_toggle', type=str2bool, required=True, help='Enable low-b or mid-b loss boosting')
    parser.add_argument('--ablate_option', type=str, required=True, help='Constraint ablation mode')
    parser.add_argument('--use_three_compartment', type=str2bool, required=True, help='Use 3C or 2C model')
    parser.add_argument('--input_type', type=str, required=True, help='image or array input')
    parser.add_argument('--tissue_type', type=str, required=True, help='tissue type: mixed, NAWM, or WMH')
    parser.add_argument('--val_loc', type=str, required=True, help='Path to the validation data')

    # Handle custom constraint bounds
    def parse_custom_dict(val):
        if val == "None" or val == 'none':
            return None
        return json.loads(val)
    parser.add_argument('--custom_dict', type=parse_custom_dict, default=None)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create destination directory if it doesn't exist
    os.makedirs(args.dest_dir, exist_ok=True)

    # Determine model and tissue config
    model_type = "3C" if args.use_three_compartment else "2C"
    tissue_type = "mixed" if args.original_mode else args.tissue_type

    print('\n[INFO] Loading hyperparams with model/tissue...\n')

    # Load full hyperparameter object
    arg = hp(model_type=model_type, tissue_type=tissue_type, IR=args.IR)

    # Set any training-specific flags
    arg.use_three_compartment = args.use_three_compartment

    # Fill missing or derived fields
    arg = deep2.checkarg(arg)

    # Assign destination directory for training outputs
    arg.train_pars.dest_dir = args.dest_dir

    print(vars(arg.train_pars))
    print(f"[CONFIG] Using model_type: {model_type}, tissue_type: {tissue_type}, IR: {args.IR}")

    # Run IVIM prediction
    IVIM_object = map_generator_NN(
        train_input_path=args.train_loc,
        bvalues_path=args.bval_path,
        dest_dir=args.dest_dir,
        arg=arg,
        original_mode=args.original_mode,
        weight_tuning=args.weight_tuning,
        IR=args.IR,
        freeze_param=args.freeze_param,
        boost_toggle=args.boost_toggle,
        ablate_option=args.ablate_option,
        use_three_compartment=args.use_three_compartment,
        input_type=args.input_type,
        tissue_type=args.tissue_type,
        custom_dict=args.custom_dict,
        val_input_path=args.val_loc
    )

    IVIM_object.predict_IVIM_maps()
    plt.close('all')








