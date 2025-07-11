import numpy as np
import matplotlib.pyplot as plt

# b-values
bvals = np.array([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000])

# === 3C parameter ranges ===
threec_ranges = {
	"NAWM": {"Dpar": (0.00050, 0.00074), "Dint": (0.00212, 0.00318), "Dmv": (0.0736, 0.1104), "fint": (0.0584, 0.0876), "fmv": (0.0048, 0.0072)},
	"WMH":  {"Dpar": (0.00067, 0.00101), "Dint": (0.00219, 0.00329), "Dmv": (0.0608, 0.0912), "fint": (0.14, 0.21),    "fmv": (0.006, 0.009)},
	"S1":   {"Dpar": (0.0008, 0.0018),   "Dint": (0.0022, 0.0048),   "Dmv": (0.032, 0.24),     "fint": (0.16, 0.48),    "fmv": (0.08, 0.24)}
}

# === 2C parameter ranges === (updated S1)
twoc_ranges = {
	"NAWM": {
		"Dpar": (0.00065, 0.0008),
		"Dmv":  (0.004, 0.015),
		"fmv":  (0.2, 0.5)
	},
	"WMH": {
		"Dpar": (0.0010, 0.0014),
		"Dmv":  (0.004, 0.015),
		"fmv":  (0.3, 0.6)
	},
	"S1": {
		"Dpar": (0.0004, 0.0020),     # wider and overlapping both ends
		"Dmv":  (0.003, 0.020),       # lower than both WMH/NAWM to allow flatter decay
		"fmv":  (0.01, 0.65)          # overlaps with both, extended upper range
	}
}

# === Signal generation functions ===
def add_rician_noise(S, bvals, noise_mode="nonoise"):
	if noise_mode == "nonoise":
		return S
	return S  # Placeholder for future noise

def generate_signal_3c(params, bvals, IR=False, noise_mode="nonoise"):
	TI, TR, TE = 2230, 6800, 84
	T1_tissue, T1_isf, T2_tissue, T2_isf = 1081, 1250, 95, 503

	Dpar = params["Dpar"][:, None]
	Dint = params["Dint"][:, None]
	Dmv  = params["Dmv"][:, None]
	fint = params["fint"][:, None]
	fmv  = params["fmv"][:, None]
	fpar = 1.0 - fint - fmv
	bvals = np.array(bvals)[None, :]

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
	Dpar = params["Dpar"][:, None]
	Dmv  = params["Dmv"][:, None]
	fmv  = params["fmv"][:, None]
	fpar = 1.0 - fmv
	bvals = np.array(bvals)[None, :]

	S = fpar * np.exp(-bvals * Dpar) + fmv * np.exp(-bvals * Dmv)
	S = S / S[:, [0]]
	return add_rician_noise(S, bvals, noise_mode)

def sample_param_tensor(range_dict, N):
	return {key: np.random.uniform(low, high, size=(N,)) for key, (low, high) in range_dict.items()}

# === Simulation ===
def generate_signals_3c(N=1000):
	out = {}
	for tissue, ranges in threec_ranges.items():
		params = sample_param_tensor(ranges, N)
		out[tissue] = {
			"std": generate_signal_3c(params, bvals, IR=False),
			"ir":  generate_signal_3c(params, bvals, IR=True)
		}
	return out

def generate_signals_2c(N=1000):
	out = {}
	for tissue, ranges in twoc_ranges.items():
		params = sample_param_tensor(ranges, N)
		out[tissue] = generate_signal_2c(params, bvals)
	return out

# Output avg signals
def compute_avg_signals(signal_dict, mode=None):
	avg_signals = {}
	b1000_signals = {}

	for tissue, signals in signal_dict.items():
		sig = signals[mode] if isinstance(signals, dict) else signals
		mean_curve = np.mean(sig, axis=0)
		avg_signals[tissue] = np.mean(mean_curve)
		b1000_signals[tissue] = mean_curve[-1]  # value at b=1000

	return avg_signals, b1000_signals


# Plotting 
def plot_mean_sd_curves(signal_dict, mode, title_prefix="3C", use_color=None):
	plt.figure(figsize=(8, 6))
	for tissue, signals in signal_dict.items():
		sig = signals[mode] if isinstance(signals, dict) else signals
		mean_sig = np.mean(sig, axis=0)
		std_sig = np.std(sig, axis=0)
		color = use_color[tissue] if use_color else None
		plt.plot(bvals, mean_sig, label=f"{tissue}", color=color, linewidth=2)
		plt.fill_between(bvals, mean_sig - std_sig, mean_sig + std_sig, alpha=0.2, color=color)

	plt.title(f"{title_prefix} IVIM Signal: {'Standard' if mode == 'std' else 'Inversion Recovery'} (n=1000)")
	plt.xlabel("b-value [s/mm²]")
	plt.ylabel("Signal (a.u.)")
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()

def compute_avg_signals_with_std(signal_dict, mode=None):
	avg_signals = {}
	b1000_signals = {}
	b1000_stds = {}

	for tissue, signals in signal_dict.items():
		sig = signals[mode] if isinstance(signals, dict) else signals
		mean_curve = np.mean(sig, axis=0)
		std_curve = np.std(sig, axis=0)

		avg_signals[tissue] = np.mean(mean_curve)
		b1000_signals[tissue] = mean_curve[-1]
		b1000_stds[tissue] = std_curve[-1]

	return avg_signals, b1000_signals, b1000_stds

# === Main ===
if __name__ == "__main__":
	colors = {'NAWM': '#E69F00', 'WMH': '#009E73', 'S1': '#0072B2'}

	sigs_3c = generate_signals_3c(N=100000)
	plot_mean_sd_curves(sigs_3c, mode="std", title_prefix="3C", use_color=colors)
	plot_mean_sd_curves(sigs_3c, mode="ir", title_prefix="3C", use_color=colors)

	sigs_2c = generate_signals_2c(N=100000)
	plot_mean_sd_curves(sigs_2c, mode=None, title_prefix="2C", use_color=colors)

	avg_signals_3c_std, b1000_3c_std, b1000_std_3c_std = compute_avg_signals_with_std(sigs_3c, mode="std")
	avg_signals_3c_ir, b1000_3c_ir, b1000_std_3c_ir = compute_avg_signals_with_std(sigs_3c, mode="ir")
	avg_signals_2c, b1000_2c, b1000_std_2c = compute_avg_signals_with_std(sigs_2c, mode=None)

	print("avg_signals_3c_std", avg_signals_3c_std)
	print("avg_signals_3c_ir", avg_signals_3c_ir)
	print("avg_signals_2c", avg_signals_2c)

	print("avg_signals_3c_std at b1000 (mean ± std):")
	for tissue in b1000_3c_std:
		print(f"{tissue}: {b1000_3c_std[tissue]:.6f} ± {b1000_std_3c_std[tissue]:.6f}")

	print("avg_signals_3c_ir at b1000 (mean ± std):")
	for tissue in b1000_3c_ir:
		print(f"{tissue}: {b1000_3c_ir[tissue]:.6f} ± {b1000_std_3c_ir[tissue]:.6f}")

	print("avg_signals_2c at b1000 (mean ± std):")
	for tissue in b1000_2c:
		print(f"{tissue}: {b1000_2c[tissue]:.6f} ± {b1000_std_2c[tissue]:.6f}")
