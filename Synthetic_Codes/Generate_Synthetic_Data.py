# Generate_Synthetic_Data.py (separated 2C vs 3C logic for clarity)
import argparse
import os
import numpy as np

threec_ranges = {
    "NAWM": {
        "Dpar": (0.00050, 0.00074),
        "Dint": (0.00212, 0.00318),
        "Dmv":  (0.0736, 0.1104),
        "fint": (0.0584, 0.0876),
        "fmv":  (0.0048, 0.0072)
    },
    "WMH": {
        "Dpar": (0.00067, 0.00101),
        "Dint": (0.00219, 0.00329),
        "Dmv":  (0.0608, 0.0912),
        "fint": (0.14, 0.21),
        "fmv":  (0.006, 0.009)
    },
    "S1": {
        "Dpar": (0.0008, 0.0018),
        "Dint": (0.0022, 0.0048),
        "Dmv":  (0.032, 0.24),
        "fint": (0.16, 0.48),
        "fmv":  (0.08, 0.24)
    }
}


# ==== 2C parameter ranges ====
twoc_ranges = {
    "NAWM": {
        "Dpar": (0.00065, 0.00075), "Dmv": (0.010, 0.025), "fmv": (0.03, 0.06)
    },
    "WMH": {
        "Dpar": (0.00055, 0.00065), "Dmv": (0.007, 0.015), "fmv": (0.06, 0.10)
    }
}

twoc_ranges["S1"] = {
    "Dpar": (0.00055, 0.00075),   
    "Dmv":  (0.007,  0.025),      
    "fmv":  (0.03,   0.10)        
}

# Combine NAWM and WMH ranges for S1 in 2C mode
#def combine_ranges_2c(r1, r2):
#    return {
#        key: (min(r1[key][0], r2[key][0]), max(r1[key][1], r2[key][1]))
#        for key in r1
#    }



#twoc_ranges["S1"] = combine_ranges_2c(twoc_ranges["NAWM"], twoc_ranges["WMH"])

# ==== Signal generation logic ====
def sample_param_array(n, param_range_dict):
    out = {}
    for param, (low, high) in param_range_dict.items():
        out[param] = np.random.uniform(low, high, n)
    return out

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
        S = (fpar * np.exp(-bvals * Dpar) +
             fint * np.exp(-bvals * Dint) +
             fmv  * np.exp(-bvals * Dmv))
    else:
        S = (
            fpar * (1 - 2 * np.exp(-TI / T1_tissue) + np.exp(-TR / T1_tissue)) *
            np.exp(-TE / T2_tissue - bvals * Dpar) +
            fint * (1 - 2 * np.exp(-TI / T1_isf) + np.exp(-TR / T1_isf)) *
            np.exp(-TE / T2_isf - bvals * Dint) +
            fmv * np.exp(-bvals * Dmv)
        )

    return add_rician_noise(S, bvals, noise_mode)

def generate_signal_2c(params, bvals, noise_mode="nonoise"):
    Dpar = params["Dpar"][:, None]
    Dmv  = params["Dmv"][:, None]
    fmv  = params["fmv"][:, None]
    fpar = 1.0 - fmv
    bvals = np.array(bvals)[None, :]

    S = fpar * np.exp(-bvals * Dpar) + fmv * np.exp(-bvals * Dmv)
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

# ==== Parse Arguments ====
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--noise_mode", type=str, required=True, choices=["nonoise", "lownoise", "highnoise"])
parser.add_argument("--model_type", type=str, required=True, choices=["2C", "3C"])
args = parser.parse_args()

# ==== Setup ====
np.random.seed(args.seed)
n_voxels = int(11.5e6)
bvals = np.loadtxt("/scratch/nhoang2/IVIM_NeuroCovid/Data/bvals.txt", dtype=float)

out_base = f"/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_{args.model_type}_seed{args.seed}_{args.noise_mode}"
os.makedirs(out_base, exist_ok=True)

ranges = threec_ranges if args.model_type == "3C" else twoc_ranges

for label in ["S1", "NAWM", "WMH"]:
    param_dict = sample_param_array(n_voxels, ranges[label])
    for param, arr in param_dict.items():
        np.save(os.path.join(out_base, f"{label}_{param}_synthetic.npy"), arr.astype(np.float32))

    if args.model_type == "3C":
        sig_nonir = generate_signal_3c(param_dict, bvals, IR=False, noise_mode=args.noise_mode)
        sig_ir    = generate_signal_3c(param_dict, bvals, IR=True,  noise_mode=args.noise_mode)
        np.save(os.path.join(out_base, f"{label}_signal.npy"), sig_nonir.astype(np.float32))
        np.save(os.path.join(out_base, f"{label}_signal_IR.npy"), sig_ir.astype(np.float32))
    else:
        sig_nonir = generate_signal_2c(param_dict, bvals, noise_mode=args.noise_mode)
        np.save(os.path.join(out_base, f"{label}_signal.npy"), sig_nonir.astype(np.float32))
