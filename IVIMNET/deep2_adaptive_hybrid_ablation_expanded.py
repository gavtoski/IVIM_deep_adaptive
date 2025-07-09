"""
Created September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion
Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim

IVIM-3C with inversion pulse implemented by Paulien Voorter (2023)
p.voorter@maastrichtuniversity.nl 
https://www.github.com/paulienvoorter

Adaptive constraints + multi-phase tune implemented by Bin Hoang (2025)
nhat_hoang@urmc.rochester.edu

Code is uploaded as part of the publication: Voorter et al. Physics-informed neural networks improve three-component model fitting of intravoxel incoherent motion MR imaging in cerebrovascular disease (2022)

requirements:
numpy
torch
tqdm
matplotlib
"""
# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torch.nn.functional as F
from torch import isnan
from tqdm import tqdm
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import copy
import warnings
import pandas as pd
from collections import defaultdict
from scipy.stats import linregress
import sys
sys.path.append('/scratch/nhoang2/IVIM_NeuroCovid/Synthetic_Codes/') #change this to your hyperparam loc
from hyperparams import net_pars as net_pars_class
# seed 
torch.manual_seed(0)
np.random.seed(0)


# Classify tissue type, used by Net's forward fucntion
def classify_tissue_by_signal(signal_val, model_type="3C", IR=False, custom_signal_dict=None):
	"""
	Classifies a voxel as 'NAWM', 'WMH', or 'S1' based on its *mean b=1000 signal* value.

	Args:
		signal_val (float): The voxel's normalized signal intensity at b=1000.
		model_type (str): '3C' or '2C' IVIM model.
		IR (bool): Whether IR signal model is used (only applies to '3C').
		custom_signal_dict (dict, optional): Overrides default priors. Must be a dict of form:
			{"NAWM": val, "WMH": val, "S1": val}

	Returns:
		str: One of 'NAWM', 'WMH', or 'S1' depending on signal thresholding.
	"""

	# Default signal priors at b=1000
	default_signals = {
		"3C": {
			False: {"NAWM": 0.5019, "WMH": 0.3664, "S1": 0.1602},
			True:  {"NAWM": 0.1564, "WMH": 0.1162, "S1": 0.0527}
		},
		"2C": {
			False: {"NAWM": 0.3157, "WMH": 0.1673, "S1": 0.2248}
		}
	}

	signal_dict = custom_signal_dict if custom_signal_dict is not None else default_signals[model_type][IR]

	# Sort tissue types by signal intensity (descending)
	ordered = sorted(signal_dict.items(), key=lambda x: x[1], reverse=True)

	# Compute dynamic thresholds
	t1 = (ordered[0][1] + ordered[1][1]) / 2
	t2 = (ordered[1][1] + ordered[2][1]) / 2

	# Classify
	if signal_val > t1:
		return ordered[0][0]
	elif signal_val > t2:
		return ordered[1][0]
	else:
		return ordered[2][0]


# Define the neural network.
class Net(nn.Module):
	def __init__(self, bvalues, net_pars, rel_times, scaling_factor=None, original_mode=False, weight_tuning=False, IR=False, freeze_param=False, use_three_compartment=True):
		"""
		this defines the Net class which is the network we want to train.
		:param bvalues: a 1D array with the b-values
		:param net_pars: an object with network design options, as explained in the publication Kaandorp et al., with attributes:
		fitS0 --> Boolean determining whether S0 is fixed to 1 (False) or fitted (True)
		times len(bvalues), with data sorted per voxel. This option was not explored in the publication
		dropout --> Number between 0 and 1 indicating the amount of dropout regularisation
		batch_norm --> Boolean determining whether to use batch normalisation
		parallel --> Boolean determining whether to use separate networks for estimating the different IVIM parameters
		(True), or have them all estimated by a single network (False)
		con --> string which determines what type of constraint is used for the parameters. Options are:
		'sigmoid' allowing a sigmoid constraint
		'sigmoidabs' allowing a sigmoid constraint for the diffusivities Dpar, Dint and Dmv, while constraining the corresponding fraction to be positive
		'abs' having the absolute of the estimated values to constrain parameters to be positive
		'none' giving no constraints
		cons_min --> 1D array, if sigmoid is the constraint, these values give [Dpar_min, Fint_min, Dint_min, Fmv_min, Dmv_min, S0_min]
		cons_max --> 1D array, if sigmoid is the constraint, these values give [Dpar_max, Fint_max, Dint_max, Fmv_max, Dmv_min, S0_max]
		depth --> integer giving the network depth (number of layers)
		:params rel_times: an object with relaxation times of compartments and acquisition parameters, which is needed to correct for inversion recovery
		bloodT2 --> T2 of blood
		tissueT2 --> T2 of parenchymal tissue
		isfT2 --> T2 of interstitial fluid
		bloodT1 --> T1 of blood
		tissueT1 --> T1 of parenchymal tissue
		isfT1--> T1 of interstitial fluid
		echotime 
		repetitiontime
		inversiontime
		"""
		super(Net, self).__init__()

		# --------------------------------------------------------------------------------------------------------------
		# Declare necessary variables for training/fine-tuning/validation
		# --------------------------------------------------------------------------------------------------------------

		self.bvalues = bvalues
		self.rel_times = rel_times
		self.original_mode = original_mode  # Toggle tuning/adaptive constraints on and off
		self.use_three_compartment = use_three_compartment
		model_type = "3C" if self.use_three_compartment else "2C"

		if isinstance(net_pars, type):
			self.net_pars = net_pars(model_type=model_type, tissue_type=tissue_type, IR=IR, pad_fraction=pad_fraction)

		else:
			self.net_pars = net_pars  

		# Ensure constraint type is set
		if not hasattr(self.net_pars, 'con') or not self.net_pars.con:
			self.net_pars.con = 'sigmoid'

		self.net_pars.IR = IR
		self.IR=IR # This is to overwrite net_pars with the customized IR flag for safety


		if self.net_pars.width == 0:
			self.net_pars.width = len(bvalues)
		# define number of parameters being estimated
		self.est_pars = 5
		if self.net_pars.fitS0:
			self.est_pars += 1

		if not self.use_three_compartment:
			self.net_pars.depth += 0 # can try switching to 1 if you prefer but I dont think you need
			print(f"Increased network depth for 2C model: depth = {self.net_pars.depth}") #2C tends to underfit
		
		self.freeze_param = False if original_mode else freeze_param
		self.bval_mask = torch.ones_like(bvalues).bool()  # mask for fine tuning
		self.weight_tuning = weight_tuning
		self.bval_weights = torch.ones_like(bvalues)  # Placeholder, updated later per phase


		# Weight tuning for constraint scaling in custom loss fucntion, only if not original 
		if not self.original_mode:
			self.weight_tuning = True
			if self.use_three_compartment:
				default_scaling = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)  # 3C: order, fmv, fint, ftotal, magnitude
			else:
				default_scaling = torch.tensor([1, 1, 1], dtype=torch.float32)  # 2C: fmv, ftotal, magnitude

			if scaling_factor is not None:
				self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor, dtype=torch.float32, requires_grad=True))
			else:
				self.scaling_factor = nn.Parameter(default_scaling.clone().detach().requires_grad_(True))
		else:
			self.weight_tuning = False
			if self.use_three_compartment:
				self.scaling_factor = nn.Parameter(torch.zeros(5), requires_grad=False)
			else:
				self.scaling_factor = nn.Parameter(torch.zeros(3), requires_grad=False)

		# --------------------------------------------------------------------------------------------------------------
		# Creating neural network layers (still in init)
		# --------------------------------------------------------------------------------------------------------------
		self.fc_layers0 = nn.ModuleList() 
		if self.net_pars.parallel:
			self.fc_layers1 = nn.ModuleList()
			self.fc_layers2 = nn.ModuleList()
			self.fc_layers5 = nn.ModuleList() # this is for S0
			if self.use_three_compartment:
				self.fc_layers3 = nn.ModuleList() #Dint
				self.fc_layers4 = nn.ModuleList() #fint

		# loop over the layers
		width = len(bvalues)
		for i in range(self.net_pars.depth):
			# extend with a fully-connected linear layer
			self.fc_layers0.extend([nn.Linear(width, self.net_pars.width)])
			if self.net_pars.parallel:
				self.fc_layers1.extend([nn.Linear(width, self.net_pars.width)])
				self.fc_layers2.extend([nn.Linear(width, self.net_pars.width)])
				self.fc_layers5.extend([nn.Linear(width, self.net_pars.width)])
				if self.use_three_compartment:
					self.fc_layers3.extend([nn.Linear(width, self.net_pars.width)])
					self.fc_layers4.extend([nn.Linear(width, self.net_pars.width)])

			width = self.net_pars.width
			# if desired, add batch normalisation
			if self.net_pars.batch_norm:
				self.fc_layers0.extend([nn.BatchNorm1d(self.net_pars.width)])
				if self.net_pars.parallel:
					self.fc_layers1.extend([nn.BatchNorm1d(self.net_pars.width)])
					self.fc_layers2.extend([nn.BatchNorm1d(self.net_pars.width)])
					self.fc_layers5.extend([nn.BatchNorm1d(self.net_pars.width)])
					if self.use_three_compartment:
						self.fc_layers3.extend([nn.BatchNorm1d(self.net_pars.width)])
						self.fc_layers4.extend([nn.BatchNorm1d(self.net_pars.width)])

			# add ELU units for non-linearity
			self.fc_layers0.extend([nn.ELU()])
			if self.net_pars.parallel:
				self.fc_layers1.extend([nn.ELU()])
				self.fc_layers2.extend([nn.ELU()])
				self.fc_layers5.extend([nn.ELU()])
				if self.use_three_compartment:
					self.fc_layers3.extend([nn.ELU()])
					self.fc_layers4.extend([nn.ELU()])

			# if dropout is desired, add dropout regularisation
			if self.net_pars.dropout != 0 and i != (self.net_pars.depth - 1):
				self.fc_layers0.extend([nn.Dropout(self.net_pars.dropout)])
				if self.net_pars.parallel:
					self.fc_layers1.extend([nn.Dropout(self.net_pars.dropout)])
					self.fc_layers2.extend([nn.Dropout(self.net_pars.dropout)])
					self.fc_layers5.extend([nn.Dropout(self.net_pars.dropout)])
					if self.use_three_compartment:
						self.fc_layers3.extend([nn.Dropout(self.net_pars.dropout)])
						self.fc_layers4.extend([nn.Dropout(self.net_pars.dropout)])

		# Final layer yielding output, with either 5 (fix S0) or 6 outputs of a single network, or 1 output
		# per network in case of parallel networks.
		if self.net_pars.parallel:
			self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, 1))
			self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(self.net_pars.width, 1))
			self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(self.net_pars.width, 1))
			if self.use_three_compartment:
				self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(self.net_pars.width, 1))
				self.encoder4 = nn.Sequential(*self.fc_layers4, nn.Linear(self.net_pars.width, 1))
			if self.net_pars.fitS0:
				self.encoder5 = nn.Sequential(*self.fc_layers5, nn.Linear(self.net_pars.width, 1))
		else:
			self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, self.est_pars))

	# --------------------------------------------------------------------------------------------------------------
	# Necessary function for fine_tuning
	# --------------------------------------------------------------------------------------------------------------
	def compute_bval_weights(self, bvalues, phase): 
		weights = torch.ones_like(bvalues)

		if phase == 1:
			# Phase 1: No weighting — global fit
			weights[:] = 1.0

		elif phase == 2:
			# Phase 2: Emphasize low-b for Dmv/Fmv
			weights[bvalues <= 100] = 1.1
			weights[bvalues > 100] = 1.0

		elif phase == 3:
			# Phase 3: Emphasize mid-to-high b for Dpar
			weights[bvalues > 100] = 1.1
			weights[bvalues <= 100] = 1.0

		elif phase == 4:
			# Phase 4: Emphasize mid-range for Dint/Fint
			weights[(bvalues > 50) & (bvalues < 700)] = 1.1
			weights[(bvalues <= 50) | (bvalues >= 700)] = 1.0

		return weights


	def update_clipping_constraints(self, tissue_type=None, model_type=None, pad_fraction=None, IR=None):
		"""
		Dynamically rebuild constraint clipping bounds using a fresh net_pars instance.
		Allows phase-dependent updates and tissue/model-specific shifts.

		Args:
			tissue_type (str): Override tissue type (e.g., 'WMH', 'NAWM', 'mixed')
			model_type (str): Override model type ('2C' or '3C')
			pad_fraction (float): Optional override of constraint padding
			IR (bool): Optional override of IR flag
		"""
		tissue_type = tissue_type or self.net_pars.tissue_type
		model_type = model_type or self.net_pars.model_type
		IR = IR if IR is not None else self.net_pars.IR

		updated_pars = net_pars_class(
			model_type=model_type,
			tissue_type=tissue_type,
			pad_fraction=pad_fraction,
			IR=IR
		)

		self.net_pars.cons_min = torch.tensor(updated_pars.cons_min, dtype=torch.float32, device=self.bvalues.device)
		self.net_pars.cons_max = torch.tensor(updated_pars.cons_max, dtype=torch.float32, device=self.bvalues.device)
		self.net_pars.tissue_type = updated_pars.tissue_type
		self.net_pars.model_type = updated_pars.model_type
		self.net_pars.IR = updated_pars.IR

		print(f"[MODEL] Clipping constraints updated → model: {model_type}, tissue: {tissue_type}, pad={pad_fraction}")

	# --------------------------------------------------------------------------------------------------------------
	# Forward function to propagate neural net/inference:
	# --------------------------------------------------------------------------------------------------------------  
	def forward(self, X):
		# Mask only used for signal reconstruction
		bvals = self.bvalues
		if bvals.dim() == 1:
			bvals = bvals.view(1, -1)
		if self.training and hasattr(self, 'bval_mask'):
			bvals = bvals[:, self.bval_mask]

		#  Run encoders
		if self.net_pars.parallel:
			Dmv_raw  = self.encoder0(X).view(-1, 1)
			Dpar_raw = self.encoder1(X).view(-1, 1)
			Fmv_raw  = self.encoder2(X).view(-1, 1)
			if self.use_three_compartment:
				Dint_raw = self.encoder3(X).view(-1, 1)
				Fint_raw = self.encoder4(X).view(-1, 1)
			S0_raw = self.encoder5(X).view(-1, 1) if self.net_pars.fitS0 else torch.ones_like(Dpar_raw)
		else:
			out = self.encoder0(X)
			if self.use_three_compartment:
				Dmv_raw, Dpar_raw, Fmv_raw, Dint_raw, Fint_raw = [out[:, i].unsqueeze(1) for i in range(5)]
			else:
				Dmv_raw, Dpar_raw, Fmv_raw = [out[:, i].unsqueeze(1) for i in range(3)]
			S0_raw = out[:, 5].unsqueeze(1) if self.net_pars.fitS0 else torch.ones_like(Dpar_raw)

		#--------------------
		#  Apply constraints 
		#--------------------
		# Get constraint function and bounds
		if self.original_mode and (not hasattr(self.net_pars, 'con') or not self.net_pars.con):
			self.net_pars.con = 'sigmoid'

		# Determine net_pars constraints based on tissue types using b1000 signal
		if not self.original_mode:
			X_b1000 = X[:, -1]  # last b-value assumed to be b=1000
			b1000_mean = X_b1000.mean().item()

			tissue_guess = classify_tissue_by_signal(
				signal_val=b1000_mean,
				model_type=self.net_pars.model_type,
				IR=self.net_pars.IR
			)

			self.update_clipping_constraints(tissue_type=tissue_guess)

		# Get constraints for the updated net_pars
		con = self.net_pars.con
		cmin = self.net_pars.cons_min
		cmax = self.net_pars.cons_max

		# Constraint function
		def constrain(param, cmin_val, cmax_val):
			if con == 'sigmoid':
				val = cmin_val + torch.sigmoid(param) * (cmax_val - cmin_val)
			elif con == 'abs':
				val = torch.abs(param)
			elif con == 'sigmoidabs':
				val = torch.abs(cmin_val + torch.sigmoid(param) * (cmax_val - cmin_val))
			elif con == 'none':
				val = param
			else:
				raise ValueError(f"[ERROR] Unknown constraint function: {con}")
			return torch.clamp(val, min=0.0)


		# Assign constrained outputs
		raw_outputs = {}

		# Constraint dictionary to call hyperparam's net_pars
		if self.use_three_compartment:
			raw_outputs = {
				'Dpar': Dpar_raw,
				'Fint': Fint_raw,
				'Dint': Dint_raw,
				'Fmv': Fmv_raw,
				'Dmv': Dmv_raw,
			}
		else:
			raw_outputs = {
				'Dpar': Dpar_raw,
				'Fmv': Fmv_raw,
				'Dmv': Dmv_raw,
			}

		if self.net_pars.fitS0:
			raw_outputs['S0'] = S0_raw
		else:
			raw_outputs['S0'] = torch.ones_like(Dpar_raw)

		# Apply constraints dynamically
		constrained_outputs = {}
		for i, pname in enumerate(self.net_pars.param_names):
			constrained_outputs[pname] = constrain(raw_outputs[pname], cmin[i], cmax[i])


		#--------------------------------
		# Reconstruct predicted signal
		#--------------------------------

		# Extract constrained parameters for convenience
		Dpar = constrained_outputs['Dpar']
		Fmv  = constrained_outputs['Fmv']
		Dmv  = constrained_outputs['Dmv']
		S0   = constrained_outputs['S0']

		if self.use_three_compartment:
			Dint = constrained_outputs['Dint']
			Fint = constrained_outputs['Fint']
		else:
			Dint = Fint = None

		# Convert all constants to torch tensors on the correct device
		if self.use_three_compartment: 
			if self.IR:
				
				rt = self.rel_times
				device = X.device  # Ensure all constants are on the same device as inputs
				TE = torch.tensor(rt.echotime, dtype=torch.float32, device=device)
				TR = torch.tensor(rt.repetitiontime, dtype=torch.float32, device=device)
				TI = torch.tensor(rt.inversiontime, dtype=torch.float32, device=device)
				T1_tissue = torch.tensor(rt.tissueT1, dtype=torch.float32, device=device)
				T2_tissue = torch.tensor(rt.tissueT2, dtype=torch.float32, device=device)
				T1_isf = torch.tensor(rt.isfT1, dtype=torch.float32, device=device)
				T2_isf = torch.tensor(rt.isfT2, dtype=torch.float32, device=device)
				T1_blood = torch.tensor(rt.bloodT1, dtype=torch.float32, device=device)
				T2_blood = torch.tensor(rt.bloodT2, dtype=torch.float32, device=device)

				num = (
					(1 - Fmv - Fint) * (1 - 2 * torch.exp(-TI / T1_tissue) + torch.exp(-TR / T1_tissue)) * torch.exp(-TE / T2_tissue - bvals * Dpar) +
					Fint * (1 - 2 * torch.exp(-TI / T1_isf) + torch.exp(-TR / T1_isf)) * torch.exp(-TE / T2_isf - bvals * Dint) +
					Fmv  * (1 - torch.exp(-TR / T1_blood)) * torch.exp(-TE / T2_blood - bvals * Dmv)
				)

				denom = (
					(1 - Fmv - Fint) * (1 - 2 * torch.exp(-TI / T1_tissue) + torch.exp(-TR / T1_tissue)) * torch.exp(-TE / T2_tissue) +
					Fint * (1 - 2 * torch.exp(-TI / T1_isf) + torch.exp(-TR / T1_isf)) * torch.exp(-TE / T2_isf) +
					Fmv  * (1 - torch.exp(-TR / T1_blood)) * torch.exp(-TE / T2_blood)
				)

				X_pred = S0 * (num / denom)

			else:
				X_pred = S0 * (
					(1 - Fmv - Fint) * torch.exp(-bvals * Dpar) +
					Fint * torch.exp(-bvals * Dint) +
					Fmv  * torch.exp(-bvals * Dmv)
				)

			return X_pred, Dpar, Fmv, Dmv, Dint, Fint, S0
		else:
			Dint = Fint = None  # For clarity/debugging
			X_pred = S0 * (
					(1 - Fmv ) * torch.exp(-bvals * Dpar) +
					Fmv  * torch.exp(-bvals * Dmv)
				)

			return X_pred, Dpar, Fmv, Dmv, S0

#################################################################### End of Net ##############################################################################
##############################################################################################################################################################

# Constraint for custom loss functions
def expand_range(rng, buffer=0.2):
	low, high = rng
	width = high - low
	return (
		max(low - buffer * width, 1e-8),
		high + buffer * width
	)

def constraint_prior_func(tissue_type="mixed", model_type="3C", custom_dict=None):
	if custom_dict is not None:
		return custom_dict

	prior = {}

	if tissue_type == "mixed":
		if model_type == "2C":
			return {
				"Dpar": expand_range((0.0004, 0.0020)),
				"Dmv":  expand_range((0.003, 0.020)),
				"Fmv":  expand_range((0.01, 0.65))
			}

		elif model_type == "3C":
			return {
				"Dpar":  expand_range((0.0008, 0.0018)),
				"Dint":  expand_range((0.0022, 0.0048)),
				"Dmv":   expand_range((0.032, 0.24)),
				"Fmv":   expand_range((0.08, 0.24)),
				"Fint":  expand_range((0.16, 0.48)),
				"Ftotal": (None, 0.75)
			}

	elif tissue_type == "NAWM":
		if model_type == "2C":
			return {
				"Dpar": expand_range((0.00065, 0.0008)),
				"Dmv":  expand_range((0.004, 0.015)),
				"Fmv":  expand_range((0.2, 0.5))
			}
		elif model_type == "3C":
			prior = {
				"Dpar":  expand_range((0.00050, 0.00074)),
				"Dint":  expand_range((0.00212, 0.00318)),
				"Dmv":   expand_range((0.0736, 0.1104)),
				"Fint":  expand_range((0.0584, 0.0876)),
				"Fmv":   expand_range((0.0048, 0.0072)),
				"Ftotal": (None, 0.75)
			}

	elif tissue_type == "WMH":
		if model_type == "2C":
			return {
				"Dpar": expand_range((0.0010, 0.0014)),
				"Dmv":  expand_range((0.004, 0.015)),
				"Fmv":  expand_range((0.3, 0.6))
			}
		elif model_type == "3C":
			prior = {
				"Dpar":  expand_range((0.00067, 0.00101)),
				"Dint":  expand_range((0.00219, 0.00329)),
				"Dmv":   expand_range((0.0608, 0.0912)),
				"Fint":  expand_range((0.14, 0.21)),
				"Fmv":   expand_range((0.006, 0.009)),
				"Ftotal": (None, 0.75)
			}

	else:
		raise ValueError(f"[ERROR] Unsupported tissue type: {tissue_type}")

	POS_EPS = 1e-8
	safe_prior = {}
	for key, (low, high) in prior.items():
		safe_low = POS_EPS if (low is None or low <= 0) else low
		safe_prior[key] = (safe_low, high)

	return safe_prior



#################################################################### Custom Loss Functions ###################################################################
##############################################################################################################################################################

def custom_loss_function_2C(X_pred, X_batch, Dpar, Dmv, Fmv, model, 
	freeze_param=False,
	debug=0,
	phase=1,
	boost_mse_by_phase=False,
	ablate_option=None,
	tissue_type="mixed",
	custom_dict=None):

	#---------------------
	# Calculating mse loss
	#---------------------
	if model.original_mode or phase == 1:
		mse_loss = nn.MSELoss(reduction='mean')(X_pred, X_batch)
		if debug == 1:
			return mse_loss, {'mse_loss': mse_loss} 
		return mse_loss

	if not model.original_mode and phase > 1:

		# Apply bval mask always (train only)
		if  model.training and hasattr(model, 'bval_mask') and model.bval_mask is not None:
			X_batch = X_batch[:, model.bval_mask]
			if X_pred.shape[1] != X_batch.shape[1]:
				X_pred = X_pred[:, model.bval_mask]

		# Weighted MSE loss
		if model.training and getattr(model, 'weight_tuning', False):
			weights = model.bval_weights.to(X_batch.device)
			if model.training and hasattr(model, 'bval_mask') and model.bval_mask is not None:
				weights = weights[model.bval_mask]
			weights = weights.view(1, -1)
			weights = weights / weights.mean()
			mse_loss = ((X_pred - X_batch) ** 2 * weights).mean()
		else:
			mse_loss = nn.MSELoss(reduction='mean')(X_pred, X_batch)

		# Optional phase-based MSE boost
		if model.training and boost_mse_by_phase:
			boost_factor = {1: 1.0, 2: 1.0, 3: 1.05, 4: 1.05}.get(phase, 1.0)
			mse_loss *= boost_factor

		# ----------------------------------------------------
		# Return only mse_loss from signal for validation (no constraint penalty)
		# ----------------------------------------------------
		if not model.training:
			if debug == 1:
				return mse_loss, {'mse_loss': mse_loss}
			return mse_loss

		# ----------------------------
		# Constraint penalties (2C)
		# ----------------------------
		X_b1000 = X_batch[:,-1] # extract the b-1000 signal to determine tissue prior
		if not model.original_mode:
			tissue_type = classify_tissue_by_signal(
				signal_val=X_b1000.mean().item(),
				model_type='2C',
				IR=model.IR
				)
		
		bounds = constraint_prior_func(tissue_type=tissue_type, model_type="2C", custom_dict=custom_dict)

		penalty_order = torch.mean(F.softplus(Dpar - Dmv)) 
		penalty_dpar  = torch.mean(F.softplus(bounds["Dpar"][0] - Dpar) + F.softplus(Dpar - bounds["Dpar"][1]))
		penalty_dmv   = torch.mean(F.softplus(bounds["Dmv"][0]  - Dmv)  + F.softplus(Dmv  - bounds["Dmv"][1]))
		penalty_fmv   = torch.mean(F.softplus(bounds["Fmv"][0]  - Fmv)  + F.softplus(Fmv  - bounds["Fmv"][1]))

		viol_mask = {
			"Dpar_low":  Dpar < 0.1e-3,
			"Dpar_high": Dpar > 4.0e-3,
			"Dmv_low":   Dmv < 1e-3,
			"Dmv_high":  Dmv > 50e-3,
			"fmv_low" :  Fmv < 1e-4,
			"fmv_high":  Fmv > 0.2,
			"order_viol": (Dpar > Dmv),
		}
		viol_rates = {f"viol_{k}": v.float().mean().item() for k, v in viol_mask.items()}

		# Constraint weights (learnable)
		penalty_order_weight, penalty_fmv_weight, penalty_magnitude_weight = model.scaling_factor.clone()

		# Apply ablations if specified to remove certain constraints
		if ablate_option == 'remove_fmv':
			penalty_fmv_weight = 0.0
		elif ablate_option == 'remove_order':
			penalty_order_weight = 0.0
		elif ablate_option == 'remove_magnitude':
			penalty_magnitude_weight = 0.0

		#### Normalize and apply weights to constraint terms
		constraint_terms = [penalty_order, penalty_fmv, penalty_dpar + penalty_dmv]
		total_penalty = sum(constraint_terms).detach() + 1e-8

		order_term = (penalty_order / total_penalty) * penalty_order_weight
		fmv_term   = (penalty_fmv   / total_penalty) * penalty_fmv_weight
		mag_term   = ((penalty_dpar + penalty_dmv) / total_penalty) * penalty_magnitude_weight

		# ------------------------------------------------------
		# Compute total loss
		# ------------------------------------------------------
		penalty_scale = 0.2 * mse_loss.detach()
		raw_penalty = order_term + fmv_term + mag_term
		scaled_constraint_loss = torch.min(raw_penalty, penalty_scale)
		total_loss = mse_loss + scaled_constraint_loss

		if debug == 1:
			return total_loss, {
				'mse_loss': mse_loss,
				'order_penalty': order_term * penalty_scale,
				'fmv_penalty': fmv_term * penalty_scale,
				'mag_penalty': mag_term * penalty_scale,
				'total_loss': total_loss,
				**viol_rates
			}

		return total_loss


def custom_loss_function(X_pred, X_batch, Dpar, Dmv, Dint, Fmv, Fint, model,
	freeze_param=False,
	debug=0,
	phase=1,
	boost_mse_by_phase=False,
	ablate_option=None, 
	tissue_type="mixed",
	custom_dict=None
	):
	
	#---------------------
	# Calculating mse loss
	#---------------------
	if model.original_mode or phase == 1:
		mse_loss = nn.MSELoss(reduction='mean')(X_pred, X_batch)
		if debug == 1:
			return mse_loss, {'mse_loss': mse_loss} 
		return mse_loss

	if not model.original_mode and phase > 1:

		# Apply bval mask  (train only)
		if model.training and hasattr(model, 'bval_mask') and model.bval_mask is not None:
			X_batch = X_batch[:, model.bval_mask]
			if X_pred.shape[1] != X_batch.shape[1]:
				X_pred = X_pred[:, model.bval_mask]

		# Weighted MSE loss
		if model.training and getattr(model, 'weight_tuning', False):
			weights = model.bval_weights.to(X_batch.device)
			if model.training and hasattr(model, 'bval_mask') and model.bval_mask is not None:
				weights = weights[model.bval_mask]
			weights = weights.view(1, -1)
			weights = weights / weights.mean()
			mse_loss = ((X_pred - X_batch) ** 2 * weights).mean()
		else:
			mse_loss = nn.MSELoss(reduction='mean')(X_pred, X_batch)

		# Optional MSE boost by phase
		if model.training and boost_mse_by_phase:
			boost_factor = {1: 1.0, 2: 1.0, 3: 1.05, 4: 1.1}.get(phase, 1.0)
			mse_loss *= boost_factor

		# -------------------------------------------------------------------------
		# Return only mse_loss from signal for validation (no constraint penalty)
		# -------------------------------------------------------------------------
		if not model.training:
			if debug == 1:
				return mse_loss, {'mse_loss': mse_loss}
			return mse_loss

		#------------------------
		# Constraint penalties
		#------------------------
		X_b1000 = X_batch[:,-1] # extract the b-1000 signal to determine tissue prior
		if not model.original_mode:
			tissue_type = classify_tissue_by_signal(
				signal_val=X_b1000.mean().item(),
				model_type='3C',
				IR=model.IR
				)
		bounds = constraint_prior_func(tissue_type=tissue_type, model_type="3C", custom_dict=custom_dict)

		penalty_order  = torch.mean(F.softplus(Dpar - Dint) + F.softplus(Dint - Dmv)) 
		penalty_dpar   = torch.mean(F.softplus(bounds["Dpar"][0] - Dpar) + F.softplus(Dpar - bounds["Dpar"][1]))
		penalty_dint   = torch.mean(F.softplus(bounds["Dint"][0] - Dint) + F.softplus(Dint - bounds["Dint"][1]))
		penalty_dmv    = torch.mean(F.softplus(bounds["Dmv"][0]  - Dmv)  + F.softplus(Dmv  - bounds["Dmv"][1]))
		penalty_fmv    = torch.mean(F.softplus(bounds["Fmv"][0]  - Fmv)  + F.softplus(Fmv  - bounds["Fmv"][1]))
		penalty_fint   = torch.mean(F.softplus(bounds["Fint"][0] - Fint) + F.softplus(Fint - bounds["Fint"][1]))
		penalty_ftotal = torch.mean(F.softplus(Fmv + Fint - bounds["Ftotal"][1]))

		# Violation mask
		viol_mask = {
			"Dpar_low":  Dpar < 0.1e-3,
			"Dpar_high": Dpar > 2.0e-3,
			"Dint_low":  Dint < 1.5e-3,
			"Dint_high": Dint > 3.0e-3,
			"Dmv_low":   Dmv < 10e-3,
			"Dmv_high":  Dmv > 140e-3,
			"fint_low":  Fint < 0.02,
			"fint_high": Fint > 0.6,
			"fmv_high":  Fmv > 0.1,
			"ftotal_high": (Fmv + Fint) > 0.75,
			"order_viol": (Dpar > Dint) | (Dint > Dmv),
		}
		viol_rates = {f"viol_{k}": v.float().mean().item() for k, v in viol_mask.items()}

		# Constraint weights
		penalty_order_weight, penalty_fmv_weight, penalty_fint_weight, \
		penalty_ftotal_weight, penalty_magnitude_weight = model.scaling_factor

		# Apply ablation logic
		if ablate_option == 'remove_fmv':
			penalty_fmv_weight = 0.0
		elif ablate_option == 'remove_order':
			penalty_order_weight = 0.0
		elif ablate_option == 'remove_ftotal':
			penalty_ftotal_weight = 0.0
		elif ablate_option == 'remove_fint':
			penalty_fint_weight = 0.0
		elif ablate_option == 'remove_magnitude':
			penalty_magnitude_weight = 0.0

		# Normalize scaling
		constraint_terms = [
			penalty_order,
			penalty_fmv,
			penalty_fint,
			penalty_ftotal,
			penalty_dpar + penalty_dmv + penalty_dint
		]
		total_penalty = sum(constraint_terms).detach() + 1e-8

		order_term   = (penalty_order / total_penalty) * penalty_order_weight
		fmv_term     = (penalty_fmv   / total_penalty) * penalty_fmv_weight
		fint_term    = (penalty_fint  / total_penalty) * penalty_fint_weight
		ftotal_term  = (penalty_ftotal/ total_penalty) * penalty_ftotal_weight
		mag_term     = ((penalty_dpar + penalty_dmv + penalty_dint) / total_penalty) * penalty_magnitude_weight

		#-----------------------------
		# Return loss and breakdown
		#-----------------------------
		penalty_scale = 0.2 * mse_loss.detach()
		raw_penalty = (
			order_term +
			fmv_term +
			fint_term +
			mag_term +
			ftotal_term
		)

		scaled_constraint_loss = torch.min(raw_penalty, penalty_scale)
		total_loss = mse_loss + scaled_constraint_loss


		if debug == 1:
			return total_loss, {
				'mse_loss': mse_loss,
				'order_penalty': order_term * penalty_scale,
				'fmv_penalty': fmv_term * penalty_scale,
				'fint_penalty': fint_term * penalty_scale,
				'ftotal_penalty': ftotal_term * penalty_scale,
				'mag_penalty': mag_term * penalty_scale,
				'total_loss': total_loss,
				**viol_rates
			}

		return total_loss

######################################################### End of Custom Loss Functions ######################################################
#############################################################################################################################################


def to_numpy_dict(tensor_dict):
	"""
	Convert a dict of scalar tensors to a dict of Python floats.
	"""
	return {k: v.detach().cpu().item() if torch.is_tensor(v) else v for k, v in tensor_dict.items()}


def encoder_grad_norm(encoder):
	"""
	Computes the total gradient norm across all Linear.weight parameters
	in the given encoder module. Ignores bias terms and non-Linear layers.
	"""
	total = 0.0
	for module in encoder.modules():
		if isinstance(module, nn.Linear) and module.weight.grad is not None:
			total += module.weight.grad.norm().item()
	return total


def learn_IVIM(X_train, bvalues, arg, net=None, original_mode=False, weight_tuning=False, IR=False, freeze_param=False, boost_toggle=False, ablate_option=None, use_three_compartment=False,
				tissue_type="mixed", custom_dict=None):
	"""
	This program builds a IVIM-NET network and trains it.
	:param X_train: 2D array of IVIM data we use for training. First axis are the voxels and second axis are the b-values
	:param bvalues: a 1D array with the b-values
	:param arg: an object with network design options, as explained in the publication Kaandorp et al. --> check hyperparameters.py for
	options
	:param net: an optional input pre-trained network with initialized weights for e.g. transfer learning or warm start
	:return net: returns a trained network
	"""
	########################################################################################################################################
	
	#------------------------------------------
	# TRAIN AND VAL LOADER PREP
	#------------------------------------------

	torch.backends.cudnn.benchmark = True
	arg = checkarg(arg)

	## normalise the signal to b=0 and remove data with nans
	S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype('<f')
	X_train = X_train / S0[:, None]
	np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)
	# removing non-IVIM-like data; this often gets through when background data is not correctly masked
	# Estimating IVIM parameters in these data is meaningless anyways.
	X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
	X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
	X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
	X_train[X_train > 1.5] = 1.5

	# initialising the network of choice using the input argument arg
	if net is None:
		bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
		net = Net(bvalues, arg.net_pars, arg.rel_times,
                  original_mode=original_mode,
                  weight_tuning=weight_tuning,
                  IR=IR,
                  freeze_param=freeze_param,
                  use_three_compartment=use_three_compartment).to(arg.train_pars.device)

		print(f"[INIT] Using {'3C' if use_three_compartment else '2C'} IVIM model architecture")

	# Inject noise into weights (only if not original mode)
	if not original_mode:
		def perturb_model_initialization(model, std=0.03):
			with torch.no_grad():
				for param in model.parameters():
					if param.requires_grad:
						param.add_(torch.randn_like(param) * std)

		print("[NOISE] Injecting noise into model weights")
		perturb_model_initialization(net, std=0.03)


	else:
		# if a network was used as input parameter, work with that network instead (transfer learning/warm start).
		net.to(arg.train_pars.device)

	# splitting data into learning and validation set; subsequently initialising the Dataloaders
	split = int(np.floor(len(X_train) * arg.train_pars.split))
	train_set, val_set = torch.utils.data.random_split(torch.from_numpy(X_train.astype(np.float32)),
                                                       [split, len(X_train) - split])
	# train loader loads the training data. We want to shuffle to make sure data order is modified each epoch and different data is selected each epoch.
	trainloader = utils.DataLoader(train_set,
                                   batch_size=arg.train_pars.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=2
									)
	# validation data is loaded here. 
	val_cap = 1000  # cap number of validation batches
	inferloader = utils.DataLoader(val_set,
                               batch_size=32 * arg.train_pars.batch_size,
                               shuffle=False,
                               drop_last=True,
                               num_workers=2
                               )


	# defining the number of training and validation batches for normalisation later
	totalit = np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)])
	totalit = min(totalit, len(trainloader)) # need this for shuffle

	batch_norm2 = np.floor(len(val_set) // (32 * arg.train_pars.batch_size))


	#------------------------------------------
	# PHASE TUNING PARAMETERES
	#------------------------------------------
	# defining optimiser
	optimizer = load_optimizer(net, arg)
	scheduler = init_scheduler(optimizer, arg.train_pars.patience) if arg.train_pars.scheduler else None

	# Initialising parameters
	best = 1e16
	num_bad_epochs = 0
	loss_train = []
	loss_val = []
	penalty_log_list = []

	prev_lr = 0
	#final_model = copy.deepcopy(net.state_dict())
	final_model = {k: v.clone() for k, v in net.state_dict().items() if "scaling_factor" not in k}

	torch.set_num_threads(1)
	debug=1
	batch_size=arg.train_pars.batch_size 

	# Setup fine-tuning control flags
	model_type = "2C" if not use_three_compartment else "3C"
	if original_mode:
		weight_tuning = False
		IR = False
		freeze_param = False
		boost_toggle = False
		ablate_option = "none"

	if original_mode or not weight_tuning:
		fine_tune_master_enable = False
	else:
		fine_tune_master_enable = True

	fine_tune_phase = 1 # if tune, there start with phase 1
	dpar_frozen = False
	dmv_frozen = False
	enable_phase_3 = not original_mode
	boost_toggle = boost_toggle
	ablate_option = ablate_option
	grad_log_list = []

	# Pad scaling:
	# Define schedule: (phase, pad_fraction)
	if original_mode:
		padding_schedule = {1: 0.3, 2: 0.3, 3: 0.3}
	elif use_three_compartment:
		padding_schedule = {1: 0.5, 2: 0.3, 3: 0.25}
	else:  # 2C adaptive
		padding_schedule = {1: 1.0, 2: 0.5, 3: 0.3}

	if use_three_compartment:
		param_tags = {
		'encoder0': 'Dmv',
		'encoder1': 'Dpar',
		'encoder2': 'Fmv',
		'encoder3': 'Dint',
		'encoder4': 'Fint',
		'encoder5': 'S0'  # Include only if fitS0 is True
		}
	else:
		param_tags = {
		'encoder0': 'Dmv',
		'encoder1': 'Dpar',
		'encoder2': 'Fmv',
		'encoder5': 'S0'  # Include only if fitS0 is True
		}

	#------------------------------------------
	# FUNCTIONS FOR PHASE AND WEIGHT TUNING
	#------------------------------------------

	def freeze_encoder_by_phase(net, phase):
		"""
		Applies phase-specific soft freezing strategy to the IVIM model.
		- Phase 1: All parameters trainable
		- Phase 2: Dpar (encoder1) focus; others softly scaled
		- Phase 3: Dmv (encoder0) focus; soft updates to Dpar, Fmv, Fint if applicable
		- Phase 4: All parameters trainable
		"""

		# Initialize or clear soft freeze dict
		net.encoder_soft_weights = {}

		if not getattr(net, 'freeze_param', False):
			for name, param in net.named_parameters():
				param.requires_grad = True
			print(f"[PHASE {phase}] freeze_param=False → All parameters trainable.")
			return

		# Default: all trainable
		for name, param in net.named_parameters():
			param.requires_grad = True

		if phase == 1:
			print("[PHASE 1] All parameters trainable.")

		elif phase == 2:
			for name, param in net.named_parameters():
				if 'encoder0' in name or 'encoder2' in name:
					net.encoder_soft_weights[name] = 0.3  # Dmv, Fmv
				elif 'encoder1' in name:
					net.encoder_soft_weights[name] = 0.1  # Dpar
				elif net.use_three_compartment and 'encoder3' in name:
					net.encoder_soft_weights[name] = 0.1  # Dint
				elif net.use_three_compartment and 'encoder4' in name:
					net.encoder_soft_weights[name] = 0.1  # Fint
			print("[PHASE 2] Focusing on Dmv, Fmv, Dint, Fint — soft update")

		elif phase == 3:
			for name, param in net.named_parameters():
				if 'encoder0' in name or 'encoder2' in name:
					net.encoder_soft_weights[name] = 0.03  # Dmv, Fmv
				elif 'encoder1' in name:
					net.encoder_soft_weights[name] = 0.3  # Dpar
				elif net.use_three_compartment and 'encoder3' in name:
					net.encoder_soft_weights[name] = 0.15  # Dint
				elif net.use_three_compartment and 'encoder4' in name:
					net.encoder_soft_weights[name] = 0.15  # Fint

			print("[PHASE 3] Focusing on Dpar and finally signal tuning")


		elif phase == 4:
			print("[PHASE 4] Final tuning — all parameters trainable.")

		else:
			print(f"[WARNING] Unknown phase {phase}. No changes made.")


	def set_fine_tune_phase(net, bvalues, phase, bval_range, device='cpu'):
		"""
		Sets the model state for a specific fine-tuning phase.

		Args:
			net: your IVIM model
			bvalues: original b-values (torch.Tensor or np.array)
			phase: int (2 = low-b Dmv fit, 3 = mid-b Fint/Dint)
			bval_range: tuple of (b_min, b_max)
			device: 'cuda' or 'cpu'
		"""

		# Step 1: Update freezing (if freeze_param is enabled)
		if getattr(net, 'freeze_param', True):
			freeze_encoder_by_phase(net, phase)
		else:
			print(f"[PHASE {phase}] freeze_param=False → Skipping encoder freezing.")

		# Step 2: Update bval mask and weights
		if phase >= 2:
			bval_array = bvalues.cpu().numpy() if torch.is_tensor(bvalues) else np.array(bvalues)
			b_min, b_max = bval_range
			mask_indices = [i for i, b in enumerate(bval_array) if b_min <= b <= b_max]

			mask = torch.zeros_like(bvalues, dtype=torch.bool)
			mask[mask_indices] = True
			net.bval_mask = mask.to(device)

			print(f"[PHASE {phase}] Activated bval mask with {mask.sum().item()} b-values in range {b_min}-{b_max}")

			if getattr(net, 'weight_tuning', False):
				net.bval_weights = net.compute_bval_weights(bvalues, phase).to(device)
		else:
			# Phase 1 — no masking effect, use all b-values
			net.bval_mask = torch.ones_like(bvalues, dtype=torch.bool).to(device)
			net.bval_weights = torch.ones_like(bvalues).to(device)
			print(f"[PHASE {phase}] Full bval mask (no filtering), uniform weights")

	#------------------------------------------------------
	# NETWORKS TRAINING LOOPS: MULTIPLE PARALLEL NETWORKS
	#------------------------------------------------------
	max_epochs_by_phase = {1: 10, 2: 6, 3: 6}
	fine_tune_phase_epoch_counter = 0
	loss_log = []
	net.update_clipping_constraints(tissue_type="mixed", pad_fraction=padding_schedule.get(1, 0.3))


	if arg.sim.jobs > 1: #when training multiple network instances in parallel processes
		## Train
		for epoch in range(100):  
			max_epochs_per_phase = max_epochs_by_phase.get(fine_tune_phase, 10)

			# Fine-tuning phase
			if freeze_param:
				freeze_encoder_by_phase(net, fine_tune_phase)

			net.train()
			running_loss_train = 0.
			running_loss_val = 0.

			for i, X_batch in enumerate(trainloader, 0):
				if i >= totalit:
					break
				optimizer.zero_grad()
				X_batch = X_batch.to(arg.train_pars.device)
				outputs = net(X_batch)
				if use_three_compartment:
					X_pred, Dpar_pred, Fmv_pred, Dmv_pred, Dint_pred, Fint_pred, S0_pred = outputs
				else:
					X_pred, Dpar_pred, Fmv_pred, Dmv_pred, S0_pred = outputs



				X_pred[isnan(X_pred)] = 0
				X_pred[X_pred < 0] = 0
				X_pred[X_pred > 3] = 3

				#loss = custom_loss_function(X_pred, X_batch, Dpar_pred, Dmv_pred, Dint_pred, Fmv_pred, Fint_pred, net)
				if use_three_compartment:
					loss_output = custom_loss_function(
						X_pred, X_batch, Dpar_pred, Dmv_pred, Dint_pred, Fmv_pred, Fint_pred, net,
						debug=debug,
						phase=fine_tune_phase,
						boost_mse_by_phase=boost_toggle,
						ablate_option=ablate_option,
						tissue_type=tissue_type, 
						custom_dict=custom_dict
					)
				else:
					loss_output = custom_loss_function_2C(
						X_pred, X_batch, Dpar_pred, Dmv_pred, Fmv_pred, net,
						freeze_param=freeze_param,
						debug=debug,
						phase=fine_tune_phase,
						boost_mse_by_phase=boost_toggle,
						ablate_option=ablate_option,
						tissue_type=tissue_type, 
						custom_dict=custom_dict
					)

				# Unpack if dictionary is returned (debug=1)
				if isinstance(loss_output, tuple):
					loss, loss_components = loss_output
				else:
					loss = loss_output
					loss_components = None  

				loss.backward()
				# Soft-freeze gradient scaling
				if hasattr(net, 'encoder_soft_weights'):
					for name, param in net.named_parameters():
						if name in net.encoder_soft_weights and param.grad is not None:
							param.grad.mul_(net.encoder_soft_weights[name])

				optimizer.step()

				with torch.no_grad():
					net.scaling_factor.clamp_(0.001, 10)
				running_loss_train += loss.item()

			# Loop to save gradients
			epoch_grad_accum = defaultdict(float)
			grad_counts = defaultdict(int)

			for name, param in net.named_parameters():
				for key, label in param_tags.items():
					if key in name and param.grad is not None:
						grad_key = f"{label}_grad"
						epoch_grad_accum[grad_key] += param.grad.norm().item()
						grad_counts[grad_key] += 1

			epoch_grad_log = {'epoch': epoch + 1, 'phase': fine_tune_phase}
			for key in epoch_grad_accum:
				epoch_grad_log[key] = epoch_grad_accum[key] / grad_counts[key]

			grad_log_list.append(epoch_grad_log)

			net.eval()
			with torch.no_grad():
				for i, X_batch in enumerate(inferloader, 0):
					if i >= val_cap:
						print(f"[DEBUG] Hit val_cap at batch {i}")
						break
					optimizer.zero_grad()
					X_batch = X_batch.to(arg.train_pars.device)
					outputs = net(X_batch)

					if use_three_compartment:
						X_pred, Dpar_pred, Fmv_pred, Dmv_pred, Dint_pred, Fint_pred, S0_pred = outputs
					else:
						X_pred, Dpar_pred, Fmv_pred, Dmv_pred, S0_pred = outputs


					X_pred[isnan(X_pred)] = 0
					X_pred[X_pred < 0] = 0
					X_pred[X_pred > 3] = 3

					if use_three_compartment:
						loss = custom_loss_function(X_pred, X_batch, Dpar_pred, Dmv_pred, Dint_pred, Fmv_pred, Fint_pred, net, 
							debug=0,
							phase=fine_tune_phase,
							boost_mse_by_phase=boost_toggle,
							ablate_option=ablate_option,
							tissue_type=tissue_type, 
							custom_dict=custom_dict
							)
					else:
						loss = custom_loss_function_2C(X_pred, X_batch, Dpar_pred, Dmv_pred, Fmv_pred, net,
							debug=0,
							phase=fine_tune_phase,
							boost_mse_by_phase=boost_toggle,
							ablate_option=ablate_option,
							tissue_type=tissue_type, 
							custom_dict=custom_dict
							)

					running_loss_val += loss.item()

			running_loss_train = running_loss_train / totalit
			running_loss_val = running_loss_val / batch_norm2
			# save loss history for plot
			loss_train.append(running_loss_train)
			loss_val.append(running_loss_val)
			if arg.train_pars.scheduler:
				scheduler.step(running_loss_val)
				if optimizer.param_groups[0]['lr'] < prev_lr:
					net.load_state_dict(final_model, strict=False)
				prev_lr = optimizer.param_groups[0]['lr']

			if running_loss_val < best:
				final_model = copy.deepcopy(net.state_dict())
				best = running_loss_val
				num_bad_epochs = 0
			else:
				num_bad_epochs += 1
			fine_tune_phase_epoch_counter += 1

			#---------------------------
			# Phase tuning start here
			#---------------------------
			switch_phase = False

			if not original_mode:
				if fine_tune_phase_epoch_counter >= max_epochs_per_phase:
					print(f"[PHASE {fine_tune_phase}] Reached max_epochs_per_phase={max_epochs_per_phase}. Forcing phase switch.")
					num_bad_epochs = 0  # reset so next phase starts clean
					switch_phase = True


			# Handle the switch
			if switch_phase:
				if fine_tune_master_enable:
					if fine_tune_master_enable:

						if fine_tune_phase == 1:
							print("[PHASE 2] Tuning Dmv and Dint...")

							net.load_state_dict(final_model, strict=False)
							best = running_loss_val
							fine_tune_phase = 2
							fine_tune_phase_epoch_counter = 0 # only tune phase 2 for a short time
							num_bad_epochs = 0
							arg.train_pars.patience = 5
							set_fine_tune_phase(net, bvalues, 2, (0, 1000), arg.train_pars.device)

							#Update padfrac
							pad_frac = padding_schedule.get(fine_tune_phase, 0.3)
							print(f"Constraint padding updated to {pad_frac}")
							

						elif fine_tune_phase == 2:
							#if not use_three_compartment:
							#    print("[SKIP] Phase 3 skipped for 2C model.")
							#    break

							print("[PHASE 3] Tuning Dpar and signal...")

							net.load_state_dict(final_model, strict=False)
							best = running_loss_val
							fine_tune_phase = 3
							fine_tune_phase_epoch_counter = 0
							num_bad_epochs = 0
							arg.train_pars.patience = 5
							set_fine_tune_phase(net, bvalues, 3, (0, 1000), arg.train_pars.device)

							#Update padfrac
							pad_frac = padding_schedule.get(fine_tune_phase, 0.3)
							print(f"Constraint padding updated to {pad_frac}")
							
						elif fine_tune_phase == 3:
							print("[PHASE 3 COMPLETE] Max epochs reached. Ending training.")
							break

					else:
						print("[FINE-TUNE OFF] Finetune is turned off. Skip phase training.")
						break

				else:
					print("[ORIGINAL MODE] No fine-tuning allowed. Exiting training.")
					break
			
			#---------------------
			# Phase tuning ends 
			#---------------------


			# plot loss and plot 4 fitted curves
			if epoch > 0:
				# plot progress and intermediate results (if enabled)
				plot_progress(X_batch.cpu(), X_pred.cpu(), bvalues.cpu(), loss_train, loss_val, arg)


			# log and save loss_components
			if loss_components is not None:
				penalty_log_list.append({
					'epoch': epoch + 1,
					'phase': fine_tune_phase,
					**to_numpy_dict(loss_components)
				})

			# Plot training progress
			loss_log.append({
				"epoch": epoch + 1,
				"phase": getattr(net, 'fine_tune_phase', 1),
				"train_loss": running_loss_train,
				"val_loss": running_loss_val,
				"phase": net.fine_tune_phase  

			}) # Collect plots over training

	#------------------------------------------------------
	# NETWORKS TRAINING LOOPS: SINGLE NETWORK
	#------------------------------------------------------
	else:     
		for epoch in range(100):

			max_epochs_per_phase = max_epochs_by_phase.get(fine_tune_phase, 10)

			# Fine-tuning phase
			if freeze_param:
				freeze_encoder_by_phase(net, fine_tune_phase)

			# initialising and resetting parameters
			net.train()
			running_loss_train = 0.
			running_loss_val = 0.
			for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
				if i >= totalit:
					# have a maximum number of batches per epoch to ensure regular updates of whether we are improving
					break
				# zero the parameter gradients
				optimizer.zero_grad()
				# put batch on GPU if pressent
				X_batch = X_batch.to(arg.train_pars.device)

				# Forward inference
				outputs = net(X_batch)
				if use_three_compartment:
					X_pred, Dpar_pred, Fmv_pred, Dmv_pred, Dint_pred, Fint_pred, S0_pred = outputs
				else:
					X_pred, Dpar_pred, Fmv_pred, Dmv_pred, S0_pred = outputs

				# removing nans and too high/low predictions to prevent overshooting
				X_pred[isnan(X_pred)] = 0
				X_pred[X_pred < 0] = 0
				X_pred[X_pred > 1.5] = 1.5

				# determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal + constraints
				#loss = criterion(X_pred, X_batch)
				if use_three_compartment:
					loss_output = custom_loss_function(
						X_pred, X_batch, Dpar_pred, Dmv_pred, Dint_pred, Fmv_pred, Fint_pred, net,
						debug=debug,
						phase=fine_tune_phase,
						boost_mse_by_phase=boost_toggle,
						ablate_option=ablate_option,
						tissue_type=tissue_type, 
						custom_dict=custom_dict 
					)
				else:
					loss_output = custom_loss_function_2C(
						X_pred, X_batch, Dpar_pred, Dmv_pred, Fmv_pred, net,
						freeze_param=freeze_param,
						debug=debug,
						phase=fine_tune_phase,
						boost_mse_by_phase=boost_toggle,
						ablate_option=ablate_option,
						tissue_type=tissue_type, 
						custom_dict=custom_dict
					)

				# Unpack if dictionary is returned (debug=1)
				if isinstance(loss_output, tuple):
					loss, loss_components = loss_output
				else:
					loss = loss_output
					loss_components = None 

				# updating network
				loss.backward()
				# Soft-freeze gradient scaling
				if hasattr(net, 'encoder_soft_weights'):
					for name, param in net.named_parameters():
						if name in net.encoder_soft_weights and param.grad is not None:
							param.grad.mul_(net.encoder_soft_weights[name])
				optimizer.step()

				# Clip scaling_factor values to prevent instability
				with torch.no_grad():
					net.scaling_factor.clamp_(0.001, 10)

				# record total loss and determine max loss over all batches
				running_loss_train += loss.item()


			# Loop to save gradients
			epoch_grad_accum = defaultdict(float)
			grad_counts = defaultdict(int)

			for name, param in net.named_parameters():
				for key, label in param_tags.items():
					if key in name and param.grad is not None:
						grad_key = f"{label}_grad"
						epoch_grad_accum[grad_key] += param.grad.norm().item()
						grad_counts[grad_key] += 1

			epoch_grad_log = {'epoch': epoch + 1, 'phase': fine_tune_phase}
			for key in epoch_grad_accum:
				epoch_grad_log[key] = epoch_grad_accum[key] / grad_counts[key]

			grad_log_list.append(epoch_grad_log)

			# after training, do validation in unseen data without updating gradients
			#print('\n validation \n')
			net.eval()
			with torch.no_grad():

				# validation is always done for a large number randomly chosen batched determined by val_cap
				for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
					if i >= val_cap:
						print(f"[DEBUG] Hit val_cap at batch {i}")
						break
					optimizer.zero_grad()
					X_batch = X_batch.to(arg.train_pars.device)

					# do prediction, only look at predicted IVIM signal
					outputs = net(X_batch)
					if use_three_compartment:
						X_pred, Dpar_pred, Fmv_pred, Dmv_pred, Dint_pred, Fint_pred, S0_pred = outputs
					else:
						X_pred, Dpar_pred, Fmv_pred, Dmv_pred, S0_pred = outputs

					# Clip the result to prevent instability
					X_pred[isnan(X_pred)] = 0
					X_pred[X_pred < 0] = 0
					X_pred[X_pred > 1.5] = 1.5
					

					# validation loss
					if use_three_compartment:
						loss = custom_loss_function(
							X_pred, X_batch, Dpar_pred, Dmv_pred, Dint_pred, Fmv_pred, Fint_pred, net,
							debug=0,
							phase=fine_tune_phase,
							boost_mse_by_phase=boost_toggle,
							ablate_option=ablate_option,
							tissue_type=tissue_type, 
							custom_dict=custom_dict 
						)
					else:
						loss = custom_loss_function_2C(
							X_pred, X_batch, Dpar_pred, Dmv_pred, Fmv_pred, net,
							freeze_param=freeze_param,
							debug=0,
							phase=fine_tune_phase,
							boost_mse_by_phase=boost_toggle,
							ablate_option=ablate_option,
							tissue_type=tissue_type, 
							custom_dict=custom_dict
						)
					running_loss_val += loss.item()

			# scale losses
			running_loss_train = running_loss_train / totalit
			running_loss_val = running_loss_val / batch_norm2
			# save loss history for plot
			loss_train.append(running_loss_train)
			loss_val.append(running_loss_val)

			# set learning rate
			if arg.train_pars.scheduler:
				scheduler.step(running_loss_val)
				if optimizer.param_groups[0]['lr'] < prev_lr:
					net.load_state_dict(final_model, strict=False)
				prev_lr = optimizer.param_groups[0]['lr']

			# early stopping criteria
			if running_loss_val < best:
				final_model = copy.deepcopy(net.state_dict())
				best = running_loss_val
				num_bad_epochs = 0
			else:
				num_bad_epochs += 1
			fine_tune_phase_epoch_counter += 1

			#---------------------------
			# Phase tuning start here
			#---------------------------
			# Check if it's time to switch phase: patience OR max_epochs_per_phase
			switch_phase = False

			if not original_mode:
				if fine_tune_phase_epoch_counter >= max_epochs_per_phase:
					print(f"[PHASE {fine_tune_phase}] Reached max_epochs_per_phase={max_epochs_per_phase}. Forcing phase switch.")
					num_bad_epochs = 0  # reset so next phase starts clean
					switch_phase = True

			# Handle the switch
			if switch_phase:
				if fine_tune_master_enable:
					if fine_tune_master_enable:

						if fine_tune_phase == 1:
							print("[PHASE 2] Tuning Dmv and Dint(3C only)...")

							net.load_state_dict(final_model, strict=False)
							best = running_loss_val
							fine_tune_phase = 2
							fine_tune_phase_epoch_counter = 0
							num_bad_epochs = 0
							arg.train_pars.patience = 3
							set_fine_tune_phase(net, bvalues, 2, (0, 1000), arg.train_pars.device)

							#Update padfrac
							pad_frac = padding_schedule.get(fine_tune_phase, 0.3)
							print(f"Constraint padding updated to {pad_frac}")
							
						elif fine_tune_phase == 2:
							#if not use_three_compartment:
							#    print("[SKIP] Phase 3 skipped for 2C model.")
							#    break

							print("[PHASE 3] Tuning Dpar...")
							net.load_state_dict(final_model, strict=False)
							best = running_loss_val
							fine_tune_phase = 3
							fine_tune_phase_epoch_counter = 0
							num_bad_epochs = 0
							arg.train_pars.patience = 3
							set_fine_tune_phase(net, bvalues, 3, (0, 1000), arg.train_pars.device)

							#Update padfrac
							pad_frac = padding_schedule.get(fine_tune_phase, 0.3)

							print(f"Constraint padding updated to {pad_frac}")


						elif fine_tune_phase == 3:
							print("[PHASE 3 COMPLETE] Max epochs reached. Ending training.")
							break

					else:
						print("[FINE-TUNE OFF] Finetune is turned off. Skip phase training.")
						break
				else:
					print("[ORIGINAL MODE] No fine-tuning allowed. Exiting training.")
					break

			#---------------------
			# Phase tuning ends
			#---------------------

			if loss_components is not None:
				penalty_log_list.append({
					'epoch': epoch + 1,
					'phase': fine_tune_phase,
					**to_numpy_dict(loss_components)
				})

			# Plot training progress
			loss_log.append({
				"epoch": epoch + 1,
				"phase": fine_tune_phase,
				"train_loss": running_loss_train,
				"val_loss": running_loss_val
			})


	#------------------------------------------------------
	# VISUALIZATION: PLOT TRAINING PROGRESS
	#------------------------------------------------------

	# Detect model type
	model_type = "2C" if getattr(arg, 'model_type', '3C') == '2C' else "3C"  # fallback to 3C
	mode_tag = f"{model_type}_{'OriginalON' if original_mode else 'OriginalOFF'}"

	if str(IR).lower() in ("true", "1", "yes"):
		mode_tag += "_IR1"
	else:
		mode_tag += "_IR0"

	if weight_tuning:
		mode_tag += "_Tune"
		mode_tag += "_FreezeON" if freeze_param else "_FreezeOFF"

	# Encode ablation setting
	if str(ablate_option).lower() in ['remove_fmv', 'remove_order', 'remove_ftotal', 'remove_fint', 'remove_magnitude', 'none']:
		mode_tag += f"_{str(ablate_option).lower()}"


	# Encode boost status
	mode_tag += "_Boost" if boost_toggle else "_NoBoost"

	# Add b-value count
	bval_len = len(bvalues)
	mode_tag += f"_b{bval_len}"

	# Encode tissue type
	true_tissue_type = getattr(arg, 'tissue_type', 'mixed')
	if true_tissue_type in ['NAWM', 'WMH', 'mixed']:
		mode_tag += f"_{true_tissue_type}"


	# Directory setup 
	from datetime import datetime
	today = datetime.today().strftime("%Y-%m-%d")
	subject_tag = getattr(arg, 'subject_tag', 'unknown')

	dest_dir = arg.train_pars.dest_dir
	assert dest_dir is not None, "arg.train_pars.dest_dir is not set!"
	result_base = os.path.dirname(os.path.dirname(dest_dir))
	subject_id = os.path.basename(dest_dir)

	# Final path now includes model type
	save_dir = os.path.join(result_base, f"loss_log_allpenalty_{today}", f"{mode_tag}")
	os.makedirs(save_dir, exist_ok=True)
	print(f"[INFO] Saving loss logs and plots to: {save_dir}")

	#  Helper for saving logs 
	def save_dataframe_csv(df, path, name):
		try:
			df.to_csv(path, index=False)
			print(f"[SAVED] {name} CSV: {path}")
		except Exception as e:
			print(f"[WARNING] Failed to save {name} CSV: {e}")

	# Save gradient logs if present
	if len(grad_log_list) > 0:
		grad_df = pd.DataFrame(grad_log_list)
		save_dataframe_csv(grad_df, os.path.join(save_dir, f"gradnorm_log_{mode_tag}.csv"), "Gradient log")
	else:
		print("[WARNING] No gradient logs found to save.")

	# Save penalty logs
	penalty_df = pd.DataFrame(penalty_log_list)

	if penalty_df.empty:
		print(f"[WARNING] Penalty log is empty for: {mode_tag}")
		
		if original_mode:
			print(f"[INFO] Original mode active — skipping penalty plots but saving MSE-only log.")
			# You can optionally skip saving anything:
			# return or continue
			# OR create a dummy df just for mse_loss tracking:
			if len(loss_train) > 0:
				penalty_df = pd.DataFrame({
					'epoch': list(range(1, len(loss_train) + 1)),
					'mse_loss': loss_train,
					'total_loss': loss_train  # identical in this mode
				})
				save_path = os.path.join(save_dir, f"penalty_log_{mode_tag}.csv")
				save_dataframe_csv(penalty_df, save_path, "Penalty log (MSE only)")
			else:
				print(f"[SKIP] No training loss to save either.")
		else:
			raise RuntimeError(f"[ERROR] No penalty log data found — skipping save and plots for {mode_tag}")

	else:
		save_path = os.path.join(save_dir, f"penalty_log_{mode_tag}.csv")
		save_dataframe_csv(penalty_df, save_path, "Penalty log")



	# Determine model type 
	model_type = "2C" if getattr(arg, 'model_type', '3C') == '2C' else "3C"


	#  Plot 1: violation-vs-penalty for each group
	#  Set up violation keys based on model
	if model_type == "3C":
		violation_groups = {
			'viol_order_viol': ['viol_order_viol'],
			'viol_fint': ['viol_fint_low', 'viol_fint_high'],
			'viol_fmv': ['viol_fmv_high'],
			'viol_ftotal': ['viol_ftotal_high'],
			'viol_Dpar': ['viol_Dpar_low', 'viol_Dpar_high'],
			'viol_Dint': ['viol_Dint_low', 'viol_Dint_high'],
			'viol_Dmv': ['viol_Dmv_low', 'viol_Dmv_high'],
		}

		violation_to_penalty = {
			'viol_order_viol': 'order_penalty',
			'viol_fint': 'fint_penalty',
			'viol_fmv': 'fmv_penalty',
			'viol_ftotal': 'ftotal_penalty',
			'viol_Dpar': 'mag_penalty',
			'viol_Dint': 'mag_penalty',
			'viol_Dmv': 'mag_penalty',
		}

	elif model_type == "2C":
		violation_groups = {
			'viol_Dpar': ['viol_Dpar_low', 'viol_Dpar_high'],
			'viol_fmv': ['viol_fmv_low', 'viol_fmv_high'],
			'viol_order': ['viol_order_viol'],
			'viol_Dmv': ['viol_Dmv_low', 'viol_Dmv_high'],
		}

		violation_to_penalty = {
			'viol_Dpar': 'mag_penalty',
			'viol_fmv': 'fmv_penalty',
			'viol_order': 'order_penalty',
			'viol_Dmv': 'mag_penalty',
		}

	else:
		raise ValueError(f"[ERROR] Unknown model type: {model_type}")

	# Plot each group
	for group_key, viol_keys in violation_groups.items():
		penalty_key = violation_to_penalty.get(group_key)
		
		# Check for penalty column
		if not penalty_key or penalty_key not in penalty_df.columns:
			print(f"[SKIP] {group_key}: Missing penalty key {penalty_key}")
			continue

		# Check for missing violation keys
		missing_keys = [key for key in viol_keys if key not in penalty_df.columns]
		if missing_keys:
			print(f"[SKIP] {group_key}: Missing violation columns {missing_keys}")
			continue

		# Compute combined violation rate
		combined_viol = penalty_df[viol_keys].sum(axis=1) if len(viol_keys) > 1 else penalty_df[viol_keys[0]]

		fig, ax1 = plt.subplots(figsize=(8, 4))
		color1, color2 = 'tab:red', 'tab:blue'

		ax1.plot(penalty_df['epoch'], combined_viol, label=f"{group_key} (rate)", color=color1)
		ax1.set_ylabel("Violation Rate", color=color1)
		ax1.tick_params(axis='y', labelcolor=color1)

		ax2 = ax1.twinx()
		ax2.plot(penalty_df['epoch'], penalty_df[penalty_key], label=f"{penalty_key} (loss)", color=color2)
		ax2.set_ylabel("Penalty Loss", color=color2)
		ax2.set_yscale("log")
		ax2.tick_params(axis='y', labelcolor=color2)

		ax1.set_title(f"{model_type}: {group_key} vs {penalty_key}")
		ax1.set_xlabel("Epoch")
		fig.tight_layout(pad=1.0)
		fig.subplots_adjust(hspace=0.4)
		filename = f"violation_vs_penalty_{group_key}_{mode_tag}.png"
		plt.savefig(os.path.join(save_dir, filename), dpi=150)
		print(f"[SAVED] Violation-vs-Penalty plot: {group_key}")
		plt.close()


	# Plot 2: Penalty breakdown over time 
	try:
		fig, ax = plt.subplots(figsize=(12, 8))

		# Determine which penalty terms to plot
		if original_mode:
			penalty_cols = ['mse_loss', 'total_loss']
		else:
			if model_type == "3C":
				penalty_cols = ['order_penalty', 'fmv_penalty', 'fint_penalty',
								'ftotal_penalty', 'mag_penalty', 'total_loss', 'mse_loss']
			elif model_type == "2C":
				penalty_cols = ['fmv_penalty', 'mag_penalty', 'total_loss', 'mse_loss']
			else:
				raise ValueError(f"[ERROR] Unknown model type: {model_type}")

		# Pretty names for legend
		pretty_names = {
			'order_penalty': 'Order Penalty',
			'fmv_penalty': 'Fmv Penalty',
			'fint_penalty': 'Fint Penalty',
			'ftotal_penalty': 'Ftotal Penalty',
			'mag_penalty': 'Magnitude Penalty',
			'total_loss': 'Total Loss',
			'mse_loss': 'MSE Loss'
		}

		# Identify phase transitions
		if 'phase' in penalty_df.columns:
			transition_mask = penalty_df['phase'].diff().fillna(0) != 0
			phase_transitions = penalty_df[transition_mask].copy()

			# shift line to match visual change
			vlines = (phase_transitions['epoch'] + 1).tolist()
			vlabels = [f"Phase {int(p)}" for p in phase_transitions['phase']]
		else:
			vlines, vlabels = [], []


		# Plot each component
		for col in penalty_cols:
			if col not in penalty_df.columns or penalty_df[col].fillna(0).abs().sum() == 0:
				print(f"[SKIP] Missing or empty column: {col}")
				continue

			linestyle = '-' if col == 'total_loss' else '--'
			color = 'blue' if col == 'mse_loss' else None
			label = pretty_names.get(col, col)

			ax.plot(penalty_df['epoch'], penalty_df[col],
					linestyle=linestyle, label=label, linewidth=2, color=color)

		# Phase lines
		for x, label in zip(vlines, vlabels):
			ax.axvline(x=x, color='gray', linestyle=':', linewidth=1)
			ax.text(x, ax.get_ylim()[1] * 0.95, label, rotation=90, va='top', fontsize=9, color='gray')

		ax.set_xlabel("Epoch")
		ax.set_ylabel("Loss Component")
		ax.set_yscale("log")
		ax.set_title(f"Loss Breakdown by Component — {mode_tag}")
		ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='x-small')
		ax.grid(True)
		fig.tight_layout(pad=1.0)
		fig.subplots_adjust(right=0.8)
		filename = os.path.join(save_dir, f"penalty_plot_{mode_tag}.png")
		plt.savefig(filename, dpi=150, bbox_inches='tight')
		print(f"[SAVED] Penalty plot: {filename}")

	except Exception as e:
		print(f"[WARNING] Failed to generate penalty plot: {e}")


	# Plot 3: Gradient vs Penalty (scatter + linear fit)
	try:
		merged_df = pd.merge(grad_df, penalty_df, on='epoch', suffixes=('_grad', '_penalty'))

		if merged_df.empty:
			print("[WARNING] No overlapping epochs between gradient and penalty logs. Cannot plot Gradient vs Penalty scatter.")
		else:
			# Determine keys based on model type
			if model_type == "3C":
				encoder_keys = ['Dpar', 'Dmv', 'Dint', 'Fmv', 'Fint']
				scatter_keys = ['mse_loss', 'order_penalty', 'fmv_penalty', 'fint_penalty',
								'ftotal_penalty', 'mag_penalty']
			elif model_type == "2C":
				encoder_keys = ['Dpar', 'Dmv', 'Fmv']
				scatter_keys = ['mse_loss', 'fmv_penalty', 'mag_penalty']
			else:
				raise ValueError(f"[ERROR] Unknown model type: {model_type}")

			fig, axes = plt.subplots(len(encoder_keys), 1, figsize=(12, 8 * len(encoder_keys)), sharex=False)

			# Ensure axes is iterable
			if len(encoder_keys) == 1:
				axes = [axes]

			for i, key in enumerate(encoder_keys):
				ax = axes[i]
				grad_key = f"{key}_grad"

				for skey in scatter_keys:
					if grad_key in merged_df.columns and skey in merged_df.columns:
						x = merged_df[skey]
						y = merged_df[grad_key]
						mask = (x > 0) & (y > 0)
						if mask.sum() < 5:
							continue

						#  Scatter points
						ax.scatter(x[mask], y[mask], label=skey, alpha=0.2, s=8)

						#  Linear fit in log space
						try:
							slope, intercept, r, p, stderr = linregress(np.log10(x[mask]), np.log10(y[mask]))
							label_fit = f"{skey} fit: slope={slope:.3f}"
							x_fit = np.logspace(np.log10(x[mask].min()), np.log10(x[mask].max()), 100)
							y_fit = 10**(intercept + slope * np.log10(x_fit))
							ax.plot(x_fit, y_fit, linestyle='--', linewidth=1, label=label_fit)
						except Exception as fit_err:
							print(f"[SKIP] Linear fit error for {key} vs {skey}: {fit_err}")

				ax.set_xscale("log")
				ax.set_yscale("log")
				ax.set_xlabel("Penalty Value")
				ax.set_ylabel(f"{key}_grad")
				ax.set_title(f"{key} — Gradient vs Penalty")
				ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='x-small')
				ax.grid(True)

			fig.tight_layout(pad=1.0)
			fig.subplots_adjust(hspace=0.8)
			fig.subplots_adjust(right=0.8)
			save_path = os.path.join(save_dir, f"grad_vs_penalty_scatter_{mode_tag}.png")
			plt.savefig(save_path, dpi=150, bbox_inches='tight')
			print(f"[SAVED] Gradient vs Penalty scatter plot: {save_path}")

	except Exception as e:
		print(f"[WARNING] Failed to generate Gradient vs Penalty scatter plot: {e}")


	# Plot 4: Final Training and Validation Loss with Phase Transitions
	if debug == 1 and 'loss_log' in locals():
		df_loss = pd.DataFrame(loss_log)

		# Identify phase transitions
		if 'phase' in df_loss.columns:
			phase_transitions = df_loss[df_loss["phase"].diff().fillna(0) != 0]
			vlines = (phase_transitions["epoch"]+1).tolist()
			vlabels = [f"Phase {int(p)}" for p in phase_transitions["phase"]]
		else:
			vlines, vlabels = [], []

		# Start plot
		plt.figure(figsize=(10, 6))
		plt.plot(df_loss["epoch"], df_loss["train_loss"], label="Train Loss", linestyle='-', marker='o', markersize=3)
		plt.plot(df_loss["epoch"], df_loss["val_loss"], label="Val Loss", linestyle='-', marker='s', markersize=3)

		# Add vertical phase lines with labels
		ymax = max(df_loss["train_loss"].max(), df_loss["val_loss"].max())
		for x, label in zip(vlines, vlabels):
			plt.axvline(x=x, linestyle='--', color='gray', alpha=0.6)
			plt.text(x + 0.2, ymax * 0.95, label, color='gray', fontsize=9, rotation=90, va='top')

		# Formatting
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.title(f"Training vs Validation Loss Over Epochs\nMode: {mode_tag}, Tissue: {tissue_type}")
		plt.legend(loc='center right', bbox_to_anchor=(1.02, 0.5))
		plt.grid(True)
		plt.tight_layout()

		# Save plot
		save_path = os.path.join(save_dir, f"{mode_tag}_loss_curve.png")
		plt.savefig(save_path, dpi=150)
		print(f"[SAVED] Loss curve with phase lines: {save_path}")
		plt.close()


	# Restore best model
	if arg.train_pars.select_best:
		net.load_state_dict(final_model, strict=False)
	del trainloader
	del inferloader
	if arg.train_pars.use_cuda:
		torch.cuda.empty_cache()
	return net


def load_optimizer(net, arg):
	par_list = []

	if arg.net_pars.parallel:
		par_list.append({'params': net.encoder0.parameters(), 'lr': arg.train_pars.lr})
		par_list.append({'params': net.encoder1.parameters()})
		par_list.append({'params': net.encoder2.parameters()})
		if net.use_three_compartment:
			par_list.append({'params': net.encoder3.parameters()})
			par_list.append({'params': net.encoder4.parameters()})
		if arg.net_pars.fitS0:
			par_list.append({'params': net.encoder5.parameters()})
	else:
		par_list.append({'params': net.encoder0.parameters()})

	# Add constraint scaling factor if applicable
	if hasattr(net, 'scaling_factor'):
		par_list.append({'params': net.scaling_factor, 'lr': 0.5e-3})

	# Select optimizer
	if arg.train_pars.optim == 'adam':
		optimizer = optim.Adam(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
	elif arg.train_pars.optim == 'sgd':
		optimizer = optim.SGD(par_list, lr=arg.train_pars.lr, momentum=0.9, weight_decay=1e-4)
	elif arg.train_pars.optim == 'adagrad':
		optimizer = torch.optim.Adagrad(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)

	# Learning rate scheduler
	#if getattr(arg.train_pars, 'scheduler', False):
	#    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
	#                                                     patience=round(arg.train_pars.patience / 2))
	#    return optimizer, scheduler
	#else:
	return optimizer

def init_scheduler(optimizer, patience, verbose=True):
	return torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='min',
		factor=0.5,
		patience=max(1, round(patience / 2)),
		min_lr=1e-6,
		verbose=verbose
	)


def predict_IVIM(data, bvalues, net, arg):
	"""
	This program takes a trained network and predicts the IVIM parameters from it.
	Predict IVIM parameters from data using a trained network. Supports 2C or 3C depending on `arg.model_type`.
	Falls back to net.use_three_compartent if not set in arg.
	
	:param data: 2D array of IVIM data we want to predict the IVIM parameters from. First axis are the voxels and second axis are the b-values
	:param bvalues: a 1D array with the b-values
	:param net: the trained IVIM-NET network
	:param arg: an object with network design options, as explained in the publication Kaandorp et al. check hyperparameters.py for
	options
	:return param: returns the predicted parameters
	"""
	arg = checkarg(arg)

	# Failsafe: determine model type
	model_type = getattr(arg, 'model_type', None)
	if model_type is None:
		model_type = "3C" if getattr(net, 'use_three_compartent', True) else "2C"

	device = arg.train_pars.device

	# Normalize signal to b=0
	S0 = np.mean(data[:, bvalues == 0], axis=1).astype('<f')
	valid = (~np.isnan(S0)) & (S0 > 0)
	data = data[valid]
	S0 = S0[valid]
	data = data / S0[:, None]

	# Remove non-IVIM-like data
	sels = (
		(np.percentile(data[:, bvalues < 50], 0.95, axis=1) < 1.3) &
		(np.percentile(data[:, bvalues > 50], 0.95, axis=1) < 1.2) &
		(np.percentile(data[:, bvalues > 150], 0.95, axis=1) < 1.0)
	)
	lend = len(data)
	data = data[sels]
	S0 = S0[sels]

	net.eval()

	# Initialize outputs
	Dpar = np.array([])
	S0_out = np.array([])

	if model_type == "3C":
		Dmv, Fmv, Dint, Fint = np.array([]), np.array([]), np.array([]), np.array([])

	elif model_type == "2C":
		Dstar, f = np.array([]), np.array([])

	# Inference loop
	inferloader = utils.DataLoader(torch.from_numpy(data.astype(np.float32)),
                                   batch_size=1024, shuffle=False, drop_last=False)

	with torch.no_grad():
		for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
			X_batch = X_batch.to(device)
			out = net(X_batch)

			if model_type == "3C":
				X_pred, Dpart, Fmvt, Dmvt, Dintt, Fintt, S0t = out
				Dpar = np.append(Dpar, Dpart.cpu().numpy())
				Dmv = np.append(Dmv, Dmvt.cpu().numpy())
				Fmv = np.append(Fmv, Fmvt.cpu().numpy())
				Dint = np.append(Dint, Dintt.cpu().numpy())
				Fint = np.append(Fint, Fintt.cpu().numpy())

			elif model_type == "2C":
				X_pred, Dpart, Dstart, ft, S0t = out
				Dpar = np.append(Dpar, Dpart.cpu().numpy())
				Dstar = np.append(Dstar, Dstart.cpu().numpy())
				f = np.append(f, ft.cpu().numpy())

			try:
				S0_out = np.append(S0_out, S0t.cpu().numpy())
			except:
				S0_out = np.append(S0_out, S0t)

	# 3C only: reorder Dpar/Dint/Dmv if needed
	if model_type == "3C":
		if np.mean(Dmv) < np.mean(Dint):
			Dmv, Dint = Dint.copy(), Dmv.copy()
			Fmv, Fint = Fint.copy(), Fmv.copy()
		if np.mean(Dint) < np.mean(Dpar):
			Dpar, Dint = Dint.copy(), Dpar.copy()
			Fint = 1 - Fint - Fmv
		if np.mean(Dmv) < np.mean(Dint):
			Dmv, Dint = Dint.copy(), Dmv.copy()
			Fmv, Fint = Fint.copy(), Fmv.copy()

	# Zero-fill final outputs
	if model_type == "3C":
		Dpartrue = np.zeros(lend)
		Dmvtrue = np.zeros(lend)
		Fmvtrue = np.zeros(lend)
		Dinttrue = np.zeros(lend)
		Finttrue = np.zeros(lend)
		S0true = np.zeros(lend)

		Dpartrue[sels] = Dpar
		Dmvtrue[sels] = Dmv
		Fmvtrue[sels] = Fmv
		Dinttrue[sels] = Dint
		Finttrue[sels] = Fint
		S0true[sels] = S0_out

		if arg.train_pars.use_cuda:
			torch.cuda.empty_cache()
		return [Dpartrue, Fmvtrue, Dmvtrue, Dinttrue, Finttrue, S0true]

	elif model_type == "2C":
		Dpartrue = np.zeros(lend)
		Dstartrue = np.zeros(lend)
		ftrue = np.zeros(lend)
		S0true = np.zeros(lend)

		Dpartrue[sels] = Dpar
		Dstartrue[sels] = Dstar
		ftrue[sels] = f
		S0true[sels] = S0_out

		if arg.train_pars.use_cuda:
			torch.cuda.empty_cache()
		return [Dpartrue, Dstartrue, ftrue, S0true]

	else:
		raise ValueError(f"Unknown model type: {model_type}")



def isnan(x):
	""" this program indicates what are NaNs  """
	return x != x


def plot_progress(X_batch, X_pred, bvalues, loss_train, loss_val, arg):
	""" this program plots the progress of the training. It will plot the loss and validatin loss, as well as 4 IVIM curve
	fits to 4 data points from the input"""
	inds1 = np.argsort(bvalues)
	X_batch = X_batch[:, inds1]
	X_pred = X_pred[:, inds1]
	bvalues = bvalues[inds1]
	if arg.fig:
		#matplotlib.use('TkAgg')
		plt.close('all')
		fig, axs = plt.subplots(2, 2)
		axs[0, 0].plot(bvalues, X_batch.data[0], 'o')
		axs[0, 0].plot(bvalues, X_pred.data[0])
		axs[0, 0].set_ylim(min(X_batch.data[0]) - 0.3, 1.2 * max(X_batch.data[0]))
		axs[1, 0].plot(bvalues, X_batch.data[1], 'o')
		axs[1, 0].plot(bvalues, X_pred.data[1])
		axs[1, 0].set_ylim(min(X_batch.data[1]) - 0.3, 1.2 * max(X_batch.data[1]))
		axs[0, 1].plot(bvalues, X_batch.data[2], 'o')
		axs[0, 1].plot(bvalues, X_pred.data[2])
		axs[0, 1].set_ylim(min(X_batch.data[2]) - 0.3, 1.2 * max(X_batch.data[2]))
		axs[1, 1].plot(bvalues, X_batch.data[3], 'o')
		axs[1, 1].plot(bvalues, X_pred.data[3])
		axs[1, 1].set_ylim(min(X_batch.data[3]) - 0.3, 1.2 * max(X_batch.data[3]))
		plt.legend(('data', 'estimate from network'))
		for ax in axs.flat:
			ax.set(xlabel='b-value (s/mm2)', ylabel='normalised signal')
		for ax in axs.flat:
			ax.label_outer()
		plt.ion()
		plt.show()
		plt.pause(0.001)
		plt.figure(2)
		plt.clf()
		plt.plot(loss_train)
		plt.plot(loss_val)
		plt.yscale("log")
		plt.xlabel('epoch #')
		plt.ylabel('loss')
		plt.legend(('training loss', 'validation loss (after training epoch)'))
		plt.ion()
		plt.show()
		plt.pause(0.001)


def checkarg_train_pars(arg):
	if not hasattr(arg,'optim'):
		warnings.warn('arg.train.optim not defined. Using default ''adam''')
		arg.optim = 'adam'  # these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
	if not hasattr(arg,'lr'):
		warnings.warn('arg.train.lr not defined. Using default value 0.0001')
		arg.lr = 0.0001  # this is the learning rate. adam needs order of 0.001; 
	if not hasattr(arg, 'patience'):
		warnings.warn('arg.train.patience not defined. Using default value 10')
		arg.patience = 10  # this is the number of epochs without improvement that the network waits untill determining it found its optimum
	if not hasattr(arg,'batch_size'):
		warnings.warn('arg.train.batch_size not defined. Using default value 128')
		arg.batch_size = 128  # number of datasets taken along per iteration
	if not hasattr(arg,'maxit'):
		warnings.warn('arg.train.maxit not defined. Using default value 500')
		arg.maxit = 500  # max iterations per epoch
	if not hasattr(arg,'split'):
		warnings.warn('arg.train.split not defined. Using default value 0.9')
		arg.split = 0.9  # split of test and validation data
	if not hasattr(arg,'load_nn'):
		warnings.warn('arg.train.load_nn not defined. Using default of False')
		arg.load_nn = False
	if not hasattr(arg,'loss_fun'):
		warnings.warn('arg.train.loss_fun not defined. Using default of ''rms''')
		arg.loss_fun = 'rms'  # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
	if not hasattr(arg,'skip_net'):
		warnings.warn('arg.train.skip_net not defined. Using default of False')
		arg.skip_net = False
	if not hasattr(arg,'use_cuda'):
		arg.use_cuda = torch.cuda.is_available()
	if not hasattr(arg, 'device'):
		arg.device = torch.device("cuda:0" if arg.use_cuda else "cpu")
	return arg


def checkarg_net_pars(arg):

	# Basic fields
	if not hasattr(arg, 'dropout'):
		warnings.warn('arg.net_pars.dropout not defined. Using default value of 0.1')
		arg.dropout = 0.1
	if not hasattr(arg, 'batch_norm'):
		warnings.warn('arg.net_pars.batch_norm not defined. Using default of True')
		arg.batch_norm = True
	if not hasattr(arg, 'parallel'):
		warnings.warn('arg.net_pars.parallel not defined. Using default of True')
		arg.parallel = True
	if not hasattr(arg, 'con'):
		warnings.warn('arg.net_pars.con not defined. Using default of "sigmoid"')
		arg.con = 'sigmoid'
	if not hasattr(arg, 'fitS0'):
		warnings.warn('arg.net_pars.fitS0 not defined. Using default of True')
		arg.fitS0 = True
	if not hasattr(arg, 'IR'):
		warnings.warn('arg.net_pars.IR not defined. Using default of False')
		arg.IR = False
	if not hasattr(arg, 'depth'):
		warnings.warn('arg.net_pars.depth not defined. Using default value of 2')
		arg.depth = 2
	if not hasattr(arg, 'width'):
		warnings.warn('arg.net_pars.width not defined. Using default of 0 (auto width)')
		arg.width = 0
	if not hasattr(arg, 'profile'):
		warnings.warn('arg.net_pars.profile not defined. Using fallback "brain3_mixed"')
		arg.profile = "brain3_mixed"

	# Precise constraint setup
	if not hasattr(arg, 'cons_min') or not hasattr(arg, 'cons_max'):
		if not hasattr(arg, 'use_three_compartment') or not hasattr(arg, 'tissue_type'):
			raise ValueError("[checkarg_net_pars] Missing required inputs: use_three_compartment and tissue_type")
		
		precise_net_pars = net_pars_backup(
			model_type=arg.model_type,
			tissue_type=arg.tissue_type
			)


		arg.cons_min = precise_net_pars.cons_min
		arg.cons_max = precise_net_pars.cons_max
		arg.param_names = precise_net_pars.param_names

	return arg


          
def checkarg_sim(arg):
	if not hasattr(arg, 'bvalues'):
		warnings.warn('arg.sim.bvalues not defined. Using default value of [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]')
		arg.bvalues = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
	if not hasattr(arg, 'repeats'):
		warnings.warn('arg.sim.repeats not defined. Using default value of 1')
		arg.repeats = 1  # this is the number of repeats for simulations
	if not hasattr(arg, 'rician'):
		warnings.warn('arg.sim.rician not defined. Using default of False')
		arg.rician = False
	if not hasattr(arg, 'SNR'):
		warnings.warn('arg.sim.SNR not defined. Using default of [20]')
		arg.SNR = [20]
	if not hasattr(arg, 'sims'):
		warnings.warn('arg.sim.sims not defined. Using default of 100000')
		arg.sims = 100000
	if not hasattr(arg, 'num_samples_eval'):
		warnings.warn('arg.sim.num_samples_eval not defined. Using default of 100000')
		arg.num_samples_eval = 100000
	if not hasattr(arg, 'range'):
		warnings.warn('arg.sim.range not defined. Using default of ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.40, 0.004, 0.2, 0.2])')
		arg.range =  ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.40, 0.004, 0.2, 0.2]) # Dpar, Fint, Dint, Fmv, Dmv
	return arg


def checkarg(arg):
	if not hasattr(arg, 'fig'):
		warnings.warn('arg.fig not defined. Using default of False')
		arg.fig = False

	if not hasattr(arg, 'save_name'):
		warnings.warn('arg.save_name not defined. Defaulting to "brain3_mixed"')
		arg.save_name = 'brain3_mixed'


	# Sync use_three_compartment and tissue_type from net_pars
	if not hasattr(arg, 'use_three_compartment'):
		arg.use_three_compartment = arg.net_pars.use_three_compartment

	# Build net_pars first, which sets profile, use_three_compartment, and tissue_type
	if not hasattr(arg, 'net_pars'):
		warnings.warn(f'arg.net_pars not defined. Using net_pars(profile="{arg.save_name}")')
		arg.net_pars = net_pars_backup(use_three_compartment=arg.use_three_compartment, tissue_type="mixed")

	if not hasattr(arg, 'tissue_type'):
		arg.tissue_type = arg.net_pars.tissue_type if hasattr(arg.net_pars, 'tissue_type') else 'mixed'

	# Training Parameters
	if not hasattr(arg, 'train_pars'):
		warnings.warn(f'arg.train_pars not defined. Using train_pars(profile="{arg.save_name}")')
		arg.train_pars = train_pars_backup(arg.save_name)

	# Fit Method 
	if not hasattr(arg, 'fit'):
		warnings.warn('arg.fit not defined. Using default initialization')
		arg.fit = lsqfit()

	# Simulation Settings 
	if not hasattr(arg, 'sim'):
		warnings.warn('arg.sim not defined. Using default initialization')
		arg.sim = sim()

	# Inversion Times (IR Mode) 
	if not hasattr(arg, 'rel_times'):
		warnings.warn('arg.rel_times not defined. Using default initialization')
		arg.rel_times = rel_times()

	# Final sanity checks
	arg.net_pars = checkarg_net_pars(arg.net_pars)
	arg.train_pars = checkarg_train_pars(arg.train_pars)
	arg.sim = checkarg_sim(arg.sim)

	# Debug Output
	print(f"[CHECKARG] Using profile: {arg.net_pars.profile} | 3C: {arg.use_three_compartment}")
	print(f"[CHECKARG] cons_min: {np.round(arg.net_pars.cons_min, 6)}")
	print(f"[CHECKARG] cons_max: {np.round(arg.net_pars.cons_max, 6)}")
	print(f"[CHECKARG] param_names: {arg.net_pars.param_names}")

	return arg




class train_pars_backup:
	def __init__(self,nets):
		self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
		self.lr = 0.00005 # this is the learning rate.
		self.patience= 30 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
		self.batch_size= 128 # number of datasets taken along per iteration
		self.maxit = 500 # max iterations per epoch
		self.split = 0.9 # split of test and validation data
		self.load_nn= False # load the neural network instead of retraining
		self.loss_fun = 'rms' # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
		self.skip_net = False # skip the network training and evaluation
		self.scheduler = True # as discussed in the article, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
		# use GPU if available
		self.use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
		self.select_best = False


class net_pars_backup:
	def __init__(self, use_three_compartment=True, tissue_type="mixed"):
		self.tissue_type = tissue_type
		self.use_three_compartment = use_three_compartment
		self.model_type = "3C" if use_three_compartment else "2C"

		# Architecture settings
		self.dropout = 0.1
		self.batch_norm = True
		self.parallel = True
		self.con = 'sigmoid'
		self.fitS0 = True
		self.depth = 2
		self.width = 0  # use auto-width based on b-values

		if model_type == "3C":
			self.param_names = ['Dpar', 'Fint', 'Dint', 'Fmv', 'Dmv', 'S0']
			if tissue_type == "mixed":
				self.cons_min = [0.0008, 0.16, 0.0022, 0.08, 0.032, 0.9]
				self.cons_max = [0.0018, 0.48, 0.0048, 0.24, 0.24, 1.1]
			elif tissue_type == "NAWM":
				self.cons_min = [0.00050, 0.0584, 0.00212, 0.0048, 0.0736, 0.9]
				self.cons_max = [0.00074, 0.0876, 0.00318, 0.0072, 0.1104, 1.1]
			elif tissue_type == "WMH":
				self.cons_min = [0.00067, 0.14, 0.00219, 0.006, 0.0608, 0.9]
				self.cons_max = [0.00101, 0.21, 0.00329, 0.009, 0.0912, 1.1]
			elif tissue_type == "original":
				self.cons_min = [0.0001, 0.0,   0.000, 0.0,   0.004, 0.9]
				self.cons_max = [0.0015, 0.40,  0.004, 0.2,   0.2,   1.1]
			else:
				raise ValueError(f"[net_pars] Unknown 3C tissue type: {tissue_type}")

		elif model_type == "2C":
			self.param_names = ['Dpar', 'Fmv', 'Dmv', 'S0']
			if tissue_type == "NAWM":
				self.cons_min = [0.00065, 0.20, 0.004, 0.9]
				self.cons_max = [0.00080, 0.50, 0.015, 1.1]
			elif tissue_type == "WMH":
				self.cons_min = [0.0010, 0.30, 0.004, 0.9]
				self.cons_max = [0.0014, 0.60, 0.015, 1.1]
			elif tissue_type == "mixed":
				self.cons_min = [0.0004, 0.01, 0.003, 0.9]
				self.cons_max = [0.0020, 0.65, 0.020, 1.1]
			else:
				raise ValueError(f"[net_pars] Unknown 2C tissue type: {tissue_type}")



		# Pad constraint range for sigmoid scaling
		pad_fraction = 0.5 if tissue_type in ["mixed", "original"] else 0.3
		range_pad = pad_fraction * (np.array(self.cons_max) - np.array(self.cons_min))
		self.cons_min = np.clip(np.array(self.cons_min) - range_pad, a_min=0, a_max=None)
		self.cons_max = np.array(self.cons_max) + range_pad



class lsqfit:
	def __init__(self):
		self.do_fit = True # skip lsq fitting
		self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
		self.jobs = 4 # number of parallel jobs. If set to 1, no parallel computing is used
		self.bounds = ([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2]) # S0, Dpar, Fint, Dint, Fmv, Dmv

class sim:
	def __init__(self):
		self.bvalues = np.array([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]) # array of b-values
		self.SNR = 35 # the SNR to simulate at
		self.sims = 11500000 # number of simulations to run
		self.num_samples_eval = 10000 # number of simualtiosn te evaluate. This can be lower than the number run. Particularly to save time when fitting. More simulations help with generating sufficient data for the neural network
		self.distribution = 'normal' #Define distribution from which IVIM parameters are sampled. Try 'uniform', 'normal' or 'normal-wide'
		self.repeats = 1 # this is the number of repeats for simulations to assess the stability
		self.n_ensemble = 20 # this is the number of instances in the network ensemble
		self.jobs = 4 # number of cores used to train the network instances of the ensemble in parallel 
		self.IR = False #True for IR-IVIM, False for IVIM without inversion recovery
		self.rician = False # add rician noise to simulations; if false, gaussian noise is added instead
		self.range = ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.40, 0.004, 0.2, 0.2]) # Dpar, Fint, Dint, Fmv, Dmv
 
class rel_times:
	def __init__(self):
		self.bloodT2 = 275 #ms [Wong et al. JMRI (2020)]
		self.tissueT2 = 95 #ms [Wong et al. JMRI (2020)]
		self.isfT2 = 503 # ms [Rydhog et al Magn.Res.Im. (2014)]
		self.bloodT1 =  1624 #ms [Wong et al. JMRI (2020)]
		self.tissueT1 =  1081 #ms [Wong et al. JMRI (2020)]
		self.isfT1 =  1250 # ms [Wong et al. JMRI (2020)]
		self.echotime = 84 # ms
		self.repetitiontime = 6800 # ms
		self.inversiontime = 2230 # ms
