import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
today_str = datetime.datetime.today().strftime("%Y-%m-%d")

# --- CONFIGURATION ---
seed = 69
data_root = f"/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025"
result_root = "/scratch/nhoang2/IVIM_NeuroCovid/Result/Test_SingleRun_"
output_dir = "/scratch/nhoang2/IVIM_NeuroCovid/Result"
os.makedirs(output_dir, exist_ok=True)

# --- MODEL + SUBJECT CONFIGS ---
model_types = ["2C", "3C"]
noise_modes = ["nonoise", "lownoise", "highnoise"]
subjects = ["S1_signal", "NAWM_signal", "WMH_signal"]
ivim_params = {
	"2C": ["Dpar", "Dmv", "fmv"],
	"3C": ["Dpar", "Dmv", "fmv", "Dint", "fint"]
}
all_params = ["Dpar", "Dmv", "Dint", "fmv", "fint"]

# --- DATA EXTRACTION ---
records = []
for model in model_types:
	for noise in noise_modes:
		model_result_root = os.path.join(f"{result_root}{model}_{noise}")
		if not os.path.isdir(model_result_root):
			print(f"[SKIP] {model_result_root} does not exist. Skipping.")
			continue
		for subject in subjects:
			tissue = subject.split("_")[0]
			subject_dir = os.path.join(model_result_root, subject)
			if not os.path.isdir(subject_dir): continue
			for mode_folder in os.listdir(subject_dir):
				mode_path = os.path.join(subject_dir, mode_folder)
				if not os.path.isdir(mode_path): continue
				mse_file = os.path.join(mode_path, f"global_nrmse_IVIM{model}.txt")
				if not os.path.isfile(mse_file): continue
				with open(mse_file, 'r') as f:
					try: nrmse_val = float(f.read().strip())
					except ValueError: nrmse_val = None
				gt_path = os.path.join(f"{data_root}_{model}_seed{seed}_nonoise_val")
				param_list = ivim_params[model]
				param_errors = {f"{p}_Err": np.nan for p in all_params}
				param_errors.update({f"{p}_{s}": np.nan for p in all_params for s in ["GT_min", "GT_max", "Pred_min", "Pred_max"]})
				for param in param_list:
					gt_file = os.path.join(gt_path, f"{tissue}_{param}_synthetic.npy")
					pred_suffix = "biexp" if model == "2C" else "triexp"
					pred_file = os.path.join(mode_path, f"{param}_NN_{pred_suffix}.npy")
					if not os.path.isfile(gt_file) or not os.path.isfile(pred_file): continue
					gt = np.load(gt_file)
					pred = np.load(pred_file)
					if gt.shape != pred.shape: continue
					nrmse = np.sqrt(np.mean((pred - gt) ** 2)) / max(np.mean(gt), 1e-6)
					param_errors[f"{param}_Err"] = nrmse
					param_errors[f"{param}_GT_min"] = np.min(gt)
					param_errors[f"{param}_GT_max"] = np.max(gt)
					param_errors[f"{param}_Pred_min"] = np.min(pred)
					param_errors[f"{param}_Pred_max"] = np.max(pred)
				avg_ivim_err = np.nanmean([v for k, v in param_errors.items() if k.endswith("_Err")])
				mode_type = "Original" if "OriginalON" in mode_folder else "Adaptive"
				IR = "IRTrue" in mode_folder
				records.append({
					"ModelType": model, "NoiseMode": noise, "ModeType": mode_type, "IR": IR,
					"Subject": subject, "Tissue": tissue, "Mode": mode_folder,
					"NRMSE": nrmse_val, **param_errors,
					"Avg_IVIM_Err": avg_ivim_err, "Path": mode_path
				})


# --- SAVE TO EXCEL ---
df_final = pd.DataFrame(records)
excel_path = os.path.join(output_dir, f"IVIM_GTparamErrors_Detail_{today_str}.xlsx")
df_final.to_excel(excel_path, index=False)

if df_final.empty:
	print("[WARNING] No valid model data found. Skipping all plots.")
	exit()

# --- PER NOISE MODE PLOTTING ---
for noise in noise_modes:
	df_noise = df_final[df_final["NoiseMode"] == noise]
	if df_noise.empty:
		continue
	print(f"[INFO] Plotting for Noise Mode: {noise}")
	
	df_plot_ivim = df_noise.groupby(["ModelType", "Tissue", "ModeType", "IR"])["Avg_IVIM_Err"].mean().reset_index()
	df_plot_ivim["Group"] = df_plot_ivim["ModeType"] + "_IR" + df_plot_ivim["IR"].astype(int).astype(str)
	df_pivot_ivim = df_plot_ivim.pivot(index=["ModelType", "Tissue"], columns="Group", values="Avg_IVIM_Err").reset_index()

	df_plot_signal = df_noise.groupby(["ModelType", "Tissue", "ModeType", "IR"])["NRMSE"].mean().reset_index()
	df_plot_signal["Group"] = df_plot_signal["ModeType"] + "_IR" + df_plot_signal["IR"].astype(int).astype(str)
	df_pivot_signal = df_plot_signal.pivot(index=["ModelType", "Tissue"], columns="Group", values="NRMSE").reset_index()
	df_pivot_signal = df_pivot_signal.replace(0, 1e-6)

	fig, axes = plt.subplots(nrows=2, figsize=(12, 12))
	bar_width = 0.2
	group_colors = {"Original_IR0": 'darkorange', "Original_IR1": 'sandybrown', "Adaptive_IR0": 'dodgerblue', "Adaptive_IR1": 'skyblue'}

	ax1 = axes[0]
	index_ivim = np.arange(len(df_pivot_ivim))
	for i, group in enumerate(group_colors):
		if group in df_pivot_ivim.columns:
			ax1.barh(index_ivim + (i - 1.5) * bar_width, df_pivot_ivim[group], bar_width, label=group, color=group_colors[group])
	ax1.set_yticks(index_ivim)
	ax1.set_yticklabels(df_pivot_ivim["ModelType"] + "_" + df_pivot_ivim["Tissue"])
	ax1.set_xlabel("Avg IVIM Error (NRMSE)")
	ax1.set_title(f"Avg IVIM Parameter Error - {noise}")
	ax1.grid(True, axis='x', linestyle='--', alpha=0.6)
	ax1.legend()

	ax2 = axes[1]
	index_signal = np.arange(len(df_pivot_signal))
	ax2.set_xscale("log")
	for i, group in enumerate(group_colors):
		if group in df_pivot_signal.columns:
			ax2.barh(index_signal + (i - 1.5) * bar_width, df_pivot_signal[group], bar_width, label=group, color=group_colors[group])
	ax2.set_yticks(index_signal)
	ax2.set_yticklabels(df_pivot_signal["ModelType"] + "_" + df_pivot_signal["Tissue"])
	ax2.set_xlabel("Reconstructed Signal Error (NRMSE, log scale)")
	ax2.set_title(f"Avg Reconstructed Signal Error - {noise}")
	ax2.grid(True, axis='x', linestyle='--', alpha=0.6, which='both')
	ax2.legend()

	plt.tight_layout()
	barplot_path = os.path.join(output_dir, f"Avg_IVIM_Error_Barplot_{today_str}_{noise}.png")
	plt.savefig(barplot_path)
	plt.close()
	print(f"[SAVED] Barplot to: {barplot_path}")

	def plot_range_widths_allparams(df, param_groups, model_type, output_dir, today_str, noise):
		n_params = len(param_groups)
		fig, axes = plt.subplots(nrows=n_params, figsize=(14, 4 * n_params), sharex=False)
		if n_params == 1: axes = [axes]
		for idx, (param, (gt_min_col, gt_max_col, pred_min_col, pred_max_col)) in enumerate(param_groups.items()):
			ax = axes[idx]
			labels = df["Tissue"] + "_" + df["ModeType"] + "_IR" + df["IR"].astype(int).astype(str)
			y_pos = np.arange(len(df))
			gt_min = df[gt_min_col]
			gt_max = df[gt_max_col]
			pred_min = df[pred_min_col]
			pred_max = df[pred_max_col]
			ax.barh(y_pos, gt_max - gt_min, left=gt_min, height=0.4, label="GT", color='orange')
			ax.barh(y_pos + 0.4, pred_max - pred_min, left=pred_min, height=0.4, label="Pred", color='dodgerblue')
			ax.set_yticks(y_pos + 0.2)
			ax.set_yticklabels(labels)
			ax.set_title(f"{model_type} - {param} Range")
			ax.set_xlabel("Parameter Range")
			ax.grid(True, axis='x', linestyle='--', alpha=0.6)
			ax.legend(loc='upper right')
		plt.tight_layout()
		save_path = os.path.join(output_dir, f"{model_type}_AllParam_RangeWidths_{today_str}_{noise}.png")
		plt.savefig(save_path)
		plt.close()
		print(f"[SAVED] {model_type} parameter range plot to: {save_path}")

	df_2c = df_noise[df_noise["ModelType"] == "2C"].copy()
	param_groups_2c = {"Dpar": ("Dpar_GT_min", "Dpar_GT_max", "Dpar_Pred_min", "Dpar_Pred_max"), "Dmv": ("Dmv_GT_min", "Dmv_GT_max", "Dmv_Pred_min", "Dmv_Pred_max"), "fmv": ("fmv_GT_min", "fmv_GT_max", "fmv_Pred_min", "fmv_Pred_max")}
	if not df_2c.empty:
		plot_range_widths_allparams(df_2c, param_groups_2c, "2C", output_dir, today_str, noise)

	df_3c = df_noise[df_noise["ModelType"] == "3C"].copy()
	param_groups_3c = {"Dpar": ("Dpar_GT_min", "Dpar_GT_max", "Dpar_Pred_min", "Dpar_Pred_max"), "Dmv": ("Dmv_GT_min", "Dmv_GT_max", "Dmv_Pred_min", "Dmv_Pred_max"), "fmv": ("fmv_GT_min", "fmv_GT_max", "fmv_Pred_min", "fmv_Pred_max"), "Dint": ("Dint_GT_min", "Dint_GT_max", "Dint_Pred_min", "Dint_Pred_max"), "fint": ("fint_GT_min", "fint_GT_max", "fint_Pred_min", "fint_Pred_max")}
	if not df_3c.empty:
		plot_range_widths_allparams(df_3c, param_groups_3c, "3C", output_dir, today_str, noise)

