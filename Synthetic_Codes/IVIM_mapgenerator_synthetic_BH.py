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
sys.path.append('/scratch/gschifit_lab/NeuroCovid/IVIM/')
sys.path.append('/scratch/gschifit_lab/NeuroCovid/IVIM/IVIM3brain-NET-main/')

import IVIMNET.deep2_adaptive_hybrid_ablation_expanded as deep2
import IVIMNET.deep as deep
#from IVIMNET.fitting_algorithms import fit_dats
from hyperparams import hyperparams as hp
import matplotlib.pyplot as plt

# Load preset hyper-parameters from arg and check if GPU is available (done in main)
#arg = hp()

# import argparse for arguments input
import argparse

# Create IVIM class object
class map_generator_NN():

	def __init__(self, input_path, bvalues_path, dest_dir, arg, pre_trained_net=None, original_mode=False, weight_tuning= False, IR=False, freeze_param= False,
				 boost_toggle=True, ablate_option=None, use_three_compartment=False, input_type="image",tissue_type="mixed", custom_dict=None):
		self.input_path = input_path
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

		if not hasattr(arg, 'net_pars') or arg.net_pars is None:
			print("[INFO] net_pars not found in arg — defaulting to brain3 config.")
			if self.use_three_compartment:
				arg.net_pars = net_pars("brain3")
			else:
				arg.net_pars = net_pars("brain2")


		# === Set subject tag for saving logs ===
		self.arg.subject_tag = os.path.basename(input_path).replace('.nii.gz', '').replace('.npy', '')
		print(self.arg.subject_tag)

	def load_and_preproc_data(self):
		# Load DWI NIfTI
		# Load data depending on input type
		if self.input_type == "image":
			self.data = nib.load(self.input_path)  # Store the data as a class attribute
			datas = self.data.get_fdata()
			self.sx, self.sy, self.sz, self.n_b_values = datas.shape
		elif self.input_type == "array":
			datas = np.load(self.input_path)
			self.n_b_values = datas.shape[1]  # Expect shape (n_voxels, n_b_values)
		else:
			raise ValueError(f"Unsupported input_type: {self.input_type}. Expected 'image' or 'array'.")

		print(f"Data dimensions: {datas.shape}")


		# Handling different volume cases, uncomment if real data
		#if self.n_b_values == 67:
		#    print("Detected 67 volumes: Truncating to the first 46 volumes.")
		#    datas = datas[..., :46]
		#    self.n_b_values = 46
		#elif self.n_b_values == 91:
		#    print("Detected 91 volumes: Using the whole dataset.")
		#elif self.n_b_values == 46:
		#    print("Detected 46 volumes: Using the dataset as is.")
		#else:
		#    raise ValueError(f"Unexpected number of volumes: {self.n_b_values}. Expected 46, 67, or 91.")

		# Compute mean signal intensity per volume
		if self.input_type == "image":
			mean_signals = np.array([np.mean(datas[..., i]) for i in range(self.n_b_values)])
		else:  # array
			mean_signals = np.mean(datas, axis=0)  # shape: (n_bvals,)

		# Get sorting indices (descending order of signal intensity)
		self.sorted_indices = np.argsort(mean_signals)[::-1]  # Descending order

		# Reorder data and b-values based on signal intensity
		datas = datas[..., self.sorted_indices]

		bval_txt = np.genfromtxt(self.bvalues_path)
		self.bvalues = np.array(bval_txt)[self.sorted_indices]  # Reorder b-values too
		print(f"b values being fitted:{self.bvalues}")

		# Identify b=0 images after reordering
		self.selsb = self.bvalues == 0

		# Reshape DWI data
		if self.input_type == "image":
			X_dw = np.reshape(datas, (self.sx * self.sy * self.sz, self.n_b_values))
		else:  # already flat
			X_dw = datas

		# Extract S0 from DWI by averaging S0 signal over b=0 slices with np.nanmean
		S0 = np.nanmean(X_dw[:, self.selsb], axis=1)
		S0[np.isnan(S0)] = 0

		# Apply filter and normalize datatot, only for the voxels above threshold
		self.valid_id = (S0 > (0.5 * np.median(S0[S0 > 0])))
		datatot = X_dw[self.valid_id, :]
		S0 = np.nanmean(datatot[:, self.selsb], axis=1).astype('<f')
		self.datatot = datatot / S0[:, None]


	def train_NN(self):
		# Call load_and_preproc_data() to initialize self.valid_id and self.datatot
		self.load_and_preproc_data()

		# Remove rows with any NaN values
		res = ~np.isnan(self.datatot).any(axis=1)  # True for rows with no NaN values

		# Neural Network fitting
		start_time = time.time()
		self.arg.train_pars.dest_dir = self.dest_dir  # set dest_dir

		# Turn off phase tuning hyperparams if original mode is on
		if self.original_mode:
		    self.weight_tuning = False
		    self.freeze_param = False
		    self.boost_toggle = False
		    self.ablate_option = "none"

		self.net = deep2.learn_IVIM(self.datatot[res], self.bvalues, self.arg,
							original_mode=self.original_mode,
							weight_tuning=self.weight_tuning,
							IR=self.IR,
							freeze_param=self.freeze_param,
							boost_toggle=self.boost_toggle,
							ablate_option=self.ablate_option,
							use_three_compartment=self.use_three_compartment,
							tissue_type=self.tissue_type,
							custom_dict=self.custom_dict)


		elapsed_time1net = time.time() - start_time
		print('\nTime elapsed for Net: {}\n'.format(elapsed_time1net))

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
		# === Step 1: Load true signal ===
		if norm:
			if self.input_type == "image":
				S_data = self.data.get_fdata()[..., self.sorted_indices]
				S0 = np.nanmean(S_data[..., self.selsb], axis=-1)
				brain_mask = S0 > (0.5 * np.median(S0[S0 > 0]))
				S0[~brain_mask] = np.nan
				S_data_norm = np.zeros_like(S_data)
				S_data_norm[brain_mask] = S_data[brain_mask] / S0[brain_mask, None]
				S_original = np.nan_to_num(S_data_norm, nan=0)
			else:
				S_original = np.load(self.input_path)[:, self.sorted_indices]
				S0 = np.nanmean(S_original[:, self.selsb], axis=1)
				valid_vox = S0 > (0.5 * np.median(S0[S0 > 0]))
				S_original = S_original[valid_vox] / S0[valid_vox, None]
				S_original = np.nan_to_num(S_original, nan=0)
				brain_mask = valid_vox
		else:
			S_data = self.data.get_fdata()[..., :self.n_b_values]
			S_data = S_data[..., self.sorted_indices]
			S_original = np.nan_to_num(S_data, nan=0)
			brain_mask = np.ones(S_original.shape[:3], dtype=bool)

		# Step 2: Prepare prediction
		S_reconstructed = np.nan_to_num(IVIM_reconstructed, nan=0)
		S_reconstructed[~brain_mask] = 0

		# Step 3: Compute NRMSE per voxel 
		epsilon = np.percentile(S_original[S_original > 0], 1) if np.any(S_original > 0) else 1e-8
		squared_error = (S_original - S_reconstructed) ** 2
		mse_map = np.mean(squared_error, axis=-1)
		rmse_map = np.sqrt(mse_map)
		norm_map = np.linalg.norm(S_original, axis=-1)
		avg_nrmse_map = np.divide(rmse_map, norm_map, out=np.zeros_like(rmse_map), where=norm_map != 0)

		# Step 4: Save map + histogram for image input
		if self.input_type == "image":
		  nrmse_nifti = nib.Nifti1Image(avg_nrmse_map, affine=self.data.affine, header=self.data.header)
		  nib.save(nrmse_nifti, os.path.join(self.dest_dir, "nrmse_map.nii.gz"))

		global_nrmse = np.mean(avg_nrmse_map)
		print(f"Global NRMSE: {global_nrmse}")
		model_tag = "IVIM3C" if self.use_three_compartment else "IVIM2C"
		with open(os.path.join(self.dest_dir, f"global_nrmse_{model_tag}.txt"), "w") as f:
		    f.write(f"{global_nrmse}\n")


		if self.input_type == "image":
			flattened_nrmse = avg_nrmse_map.flatten()
			plt.figure(figsize=(10, 6))
			plt.hist(flattened_nrmse, bins=50, color='blue', alpha=0.7, edgecolor='black')
			plt.xlabel("NRMSE")
			plt.ylabel("Frequency")
			plt.title("Distribution of NRMSE Across Voxels")
			plt.grid(axis='y', linestyle='--', alpha=0.7)
			plt.savefig(os.path.join(self.dest_dir, "NRMSE_Dist_plot.png"))
			print("NRMSE map and distribution plot saved successfully.")

		# Step 5: Plot example voxel signal
		if self.input_type == "image":
			x, y, z = S_original.shape[0] // 2, S_original.shape[1] // 2, S_original.shape[2] // 2
			true_signal_voxel = S_original[x, y, z, :]
			reconstructed_signal_voxel = S_reconstructed[x, y, z, :]
		else:
			idx = S_original.shape[0] // 2
			true_signal_voxel = S_original[idx, :]
			reconstructed_signal_voxel = S_reconstructed[idx, :]

		b_values = np.sort(self.bvalues.flatten())
		plt.figure(figsize=(10, 8))
		plt.scatter(b_values, true_signal_voxel, color='red', label="True Signal", marker="o", s=50)
		plt.plot(b_values, reconstructed_signal_voxel, label='Reconstructed IVIM Signal', linestyle="--", linewidth=2)
		plt.xlabel("b-value (s/mm²)")
		plt.ylabel("Signal Intensity")
		plt.title("IVIM Reconstructed Signal vs b-values")
		plt.legend()
		plt.grid(True)
		plt.savefig(os.path.join(self.dest_dir, f"IVIM_{model_tag}_fit_curve.png"))
		plt.close()

		print("Reconstructed IVIM curve fit saved successfully.")

		return global_nrmse

	def predict_IVIM(self, return_maps=False):

		print(f"[INFO] Running predict_IVIM() with input_type: {self.input_type}")
		assert self.input_type in ("image", "array"), f"[ERROR] Invalid input_type: {self.input_type}"

		if self.pre_trained_net is None:
			self.train_NN()
		else:
			self.net = self.pre_trained_net.to(self.arg.train_pars.device)

		# Run inference
		start_time = time.time()

		# Always use deep2.predict_IVIM now
		paramsNN = deep2.predict_IVIM(self.datatot, self.bvalues, self.net, self.arg)

		elapsed_time1netinf = time.time() - start_time
		print('\nTime elapsed for Net inference: {}\n'.format(elapsed_time1netinf))


		if self.arg.train_pars.use_cuda:
			torch.cuda.empty_cache()

		# === Determine model type and param names ===
		model_type = "3C" if self.use_three_compartment else "2C"

		if model_type == "3C":
			names = [
				'Dpar_NN_triexp',
				'fmv_NN_triexp',
				'Dmv_NN_triexp',
				'Dint_NN_triexp',
				'fint_NN_triexp',
				'S0_NN_triexp'
			]
			
		elif model_type == "2C":
		    names = [
		        'Dpar_NN_biexp',
		        'Dmv_NN_biexp',     # previously Dstar_NN_biexp
		        'fmv_NN_biexp',     # previously f_NN_biexp
		        'S0_NN_biexp'
    		]

		else:
			raise ValueError(f"[ERROR] Unsupported model_type: {model_type}")

		# === Determine output shape ===
		if self.input_type == "image":
			n_voxels = self.sx * self.sy * self.sz
			output_shape = (self.sx, self.sy, self.sz)
		else:
			n_voxels = self.datatot.shape[0]
			output_shape = (n_voxels,)

		saved_files = []
		ivim_maps = []

		# === Save all predicted parameter maps ===
		for k in range(len(names)):
			img = np.zeros(n_voxels)
			img[self.valid_id] = paramsNN[k][0:np.sum(self.valid_id)]
			img = np.where(np.isnan(img) | (img < 0), 0, img)

			if self.input_type == "image":
				img_reshaped = np.reshape(img, output_shape)
				path = os.path.join(self.dest_dir, f"{names[k]}.nii.gz")
				nib.save(nib.Nifti1Image(img_reshaped, self.data.affine, self.data.header), path)
			else:
				path = os.path.join(self.dest_dir, f"{names[k]}.npy")
				np.save(path, img.astype(np.float32))

			saved_files.append(path)
			ivim_maps.append(img)

		if len(saved_files) == 0:
			raise RuntimeError("[ERROR] No IVIM parameter files were saved!")

		print("[INFO] IVIM parameter files saved:")
		for f in saved_files:
			print(f" - {f}")

		# === Reconstruct IVIM signal for validation ===
		IVIM_reconstructed = self.reconstruct_IVIM(ivim_maps, norm=True)
		global_nrmse_error = self.calculate_nrmse_and_plot(IVIM_reconstructed)

		if return_maps:
			return global_nrmse_error, ivim_maps
		else:
			print("Maps saved as NIfTI files or .npy arrays.")
			return global_nrmse_error


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
    parser.add_argument('--preproc_loc', type=str, required=True, help='Path to the preprocess data')
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

    print('\n Loading net_pars...\n')

    # Load hyperparameters using updated net_pars logic
    arg = hp(model_type=model_type, tissue_type=tissue_type, IR=args.IR)

    # Set additional flags before validation
    arg.use_three_compartment = args.use_three_compartment
    arg.tissue_type = args.tissue_type
    arg.model_type = model_type

    arg = deep2.checkarg(arg)  # Auto-fill missing args

    # Set training directory
    arg.train_pars.dest_dir = args.dest_dir

    print(vars(arg.train_pars))
    print(f"[CONFIG] Using model_type: {model_type}, tissue_type: {tissue_type}, IR: {args.IR}")

    # Run the map generator
    IVIM_object = map_generator_NN(
        input_path=args.preproc_loc,
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
        custom_dict=args.custom_dict
    )

    IVIM_object.predict_IVIM()
    plt.close('all')







