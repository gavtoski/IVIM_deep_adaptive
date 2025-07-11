import argparse
import os
import numpy as np

# ==== Parameter Ranges ====
threec_ranges = {
	"NAWM": {"Dpar": (0.00050, 0.00074), "Dint": (0.00212, 0.00318), "Dmv": (0.0736, 0.1104), "fint": (0.0584, 0.0876), "fmv": (0.0048, 0.0072)},
	"WMH":  {"Dpar": (0.00067, 0.00101), "Dint": (0.00219, 0.00329), "Dmv": (0.0608, 0.0912), "fint": (0.14, 0.21),    "fmv": (0.006, 0.009)},
	"S1":   {"Dpar": (0.0008, 0.0018),   "Dint": (0.0022, 0.0048),   "Dmv": (0.032, 0.24),     "fint": (0.16, 0.48),    "fmv": (0.08, 0.24)}
}

twoc_ranges = {
	"NAWM": {"Dpar": (0.00065, 0.0008),  "Dmv": (0.004, 0.015), "fmv": (0.2, 0.5)},
	"WMH":  {"Dpar": (0.00100, 0.0014),   "Dmv": (0.004, 0.015), "fmv": (0.3, 0.6)},
	"S1":   {"Dpar": (0.0004, 0.0020),   "Dmv": (0.003, 0.020), "fmv": (0.01, 0.65)}
}

# ==== Signal Generation ====
def sample_param_array(n, param_range_dict):
	return {param: np.random.uniform(low, high, n) for param, (low, high) in param_range_dict.items()}

def generate_signal_3c(params, bvals, IR=False, noise_mode="nonoise"):
	TI, TR, TE = 2230, 6800, 84
	T1_tissue, T1_isf, T2_tissue, T2_isf = 1081, 1250, 95, 503

	Dpar, Dint, Dmv = params["Dpar"][:, None], params["Dint"][:, None], params["Dmv"][:, None]
	fint, fmv = params["fint"][:, None], params["fmv"][:, None]
	fpar = 1.0 - fint - fmv
	bvals = bvals[None, :]

	if not IR:
		S = fpar * np.exp(-bvals * Dpar) + fint * np.exp(-bvals * Dint) + fmv * np.exp(-bvals * Dmv)
		S = S / S[:, [0]]
	else:
		S = (
			fpar * (1 - 2 * np.exp(-TI / T1_tissue) + np.exp(-TR / T1_tissue)) * np.exp(-TE / T2_tissue - bvals * Dpar) +
			fint * (1 - 2 * np.exp(-TI / T1_isf) + np.exp(-TR / T1_isf)) * np.exp(-TE / T2_isf - bvals * Dint) +
			fmv * np.exp(-bvals * Dmv)
		)
		S = S / S[:, [0]]

	return add_rician_noise(S, bvals, noise_mode)

def generate_signal_2c(params, bvals, noise_mode="nonoise"):
	Dpar, Dmv, fmv = params["Dpar"][:, None], params["Dmv"][:, None], params["fmv"][:, None]
	fpar = 1.0 - fmv
	bvals = bvals[None, :]
	S = fpar * np.exp(-bvals * Dpar) + fmv * np.exp(-bvals * Dmv)
	S = S / S[:, [0]]
	return add_rician_noise(S, bvals, noise_mode)

def add_rician_noise(S, bvals, noise_mode):
	if noise_mode == "nonoise":
		return S / S[:, [0]]
	bvals_flat = bvals.flatten()
	snr_vals = (35 - (35 - 14) * (bvals_flat / 1000)) if noise_mode == "highnoise" else \
               (70 - (70 - 28) * (bvals_flat / 1000))
	sigma = 1 / snr_vals
	noise_real = np.random.normal(0, sigma[None, :], size=S.shape)
	noise_imag = np.random.normal(0, sigma[None, :], size=S.shape)
	S_noisy = np.sqrt((S + noise_real) ** 2 + noise_imag ** 2)
	return S_noisy / S_noisy[:, [0]]

# ==== Save Params + Signal ====
def save_param_and_signal(params, label, out_dir, bvals, model_type, noise_mode):
	os.makedirs(out_dir, exist_ok=True)
	for param, arr in params.items():
		np.save(os.path.join(out_dir, f"{label}_{param}_synthetic.npy"), arr.astype(np.float32))
	if model_type == "3C":
		sig_nonir = generate_signal_3c(params, bvals, IR=False, noise_mode=noise_mode)
		sig_ir    = generate_signal_3c(params, bvals, IR=True,  noise_mode=noise_mode)
		np.save(os.path.join(out_dir, f"{label}_signal.npy"), sig_nonir.astype(np.float32))
		np.save(os.path.join(out_dir, f"{label}_signal_IR.npy"), sig_ir.astype(np.float32))
	else:
		sig = generate_signal_2c(params, bvals, noise_mode=noise_mode)
		np.save(os.path.join(out_dir, f"{label}_signal.npy"), sig.astype(np.float32))

# ==== Main ====
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--noise_mode", type=str, required=True, choices=["nonoise", "lownoise", "highnoise"])
parser.add_argument("--model_type", type=str, required=True, choices=["2C", "3C"])
args = parser.parse_args()

np.random.seed(args.seed)
ranges = threec_ranges if args.model_type == "3C" else twoc_ranges
bvals = np.loadtxt("/scratch/nhoang2/IVIM_NeuroCovid/Data/bvals.txt", dtype=float)

# === Validation set: 500k per tissue ===
val_counts = {"NAWM": 5 * 10**5, "WMH": 5 * 10**5, "S1": 5 * 10**5}
out_val = f"/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_{args.model_type}_seed{args.seed}_{args.noise_mode}_val"
for label, count in val_counts.items():
	params = sample_param_array(count, ranges[label])
	save_param_and_signal(params, label, out_val, bvals, args.model_type, args.noise_mode)

# === Training set: 5M per mixture ===
train_mix = {
	"NAWM": {"NAWM": 0.7, "S1": 0.2, "WMH": 0.1},
	"WMH":  {"WMH": 0.7, "S1": 0.2, "NAWM": 0.1},
	"S1":   {"S1":  0.7, "NAWM": 0.2, "WMH": 0.1}
}

n_train_per_mixture = 5 * 10**6
out_train = f"/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_{args.model_type}_seed{args.seed}_{args.noise_mode}_train"
for mix_label, mix_ratios in train_mix.items():
	all_params = {k: [] for k in next(iter(ranges.values())).keys()}
	for tissue, frac in mix_ratios.items():
		n = int(n_train_per_mixture * frac)
		part = sample_param_array(n, ranges[tissue])
		for p in all_params:
			all_params[p].append(part[p])
	final_params = {p: np.concatenate(all_params[p]) for p in all_params}
	save_param_and_signal(final_params, mix_label, out_train, bvals, args.model_type, args.noise_mode)
