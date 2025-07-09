import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import sys
from pathlib import Path
import shutil
import math


# === Parse seed from CLI
if len(sys.argv) < 2:
	print("[USAGE] python compare_and_plot_results_synthetic.py <SEED>")
	sys.exit(1)

seed = sys.argv[1]  # e.g., "24", "69", "97"
print(f"\n[INFO] Running comparison for seed: {seed}")

# === Define noise modes and subjects
noise_modes = ["nonoise", "lownoise", "highnoise"]

subjects = [
	"S1_signal", "S1_signal_IR",
	"NAWM_signal", "NAWM_signal_IR",
	"WMH_signal", "WMH_signal_IR"
]


def compare_ivim_errors_and_nrmse(baseline_path, comparison_paths, subject_id, input_type="image"):
	"""
	Compare NRMSE, Fractional Error, and Frac/NMRSE ratio of IVIM models for a subject.
	Returns:
		df_ivim: IVIM parameter errors (per parameter, per model)
		df_signal: one row per model with SignalMSE
		best_models: dictionary with best model per metric
	"""

	ivim_params = ["Dpar", "Dint", "Dmv", "fint", "fmv"]
	all_results = []
	signal_summary = []

	for model_name, model_path in comparison_paths.items():
		if not os.path.exists(model_path):
			print(f"[WARNING] Missing: {model_name} → {model_path}")
			continue

		# === Load signal MSE from .txt file ===
		mse_path = os.path.join(model_path, "global_nrmse_IVIM3C.txt")
		signal_mse = None

		if os.path.exists(mse_path):
			try:
				with open(mse_path, "r") as f:
					raw_val = f.readline().strip()
					signal_mse = float(raw_val)
					if not math.isfinite(signal_mse):
						print(f"[WARNING] Signal MSE for {model_name} is not finite: '{raw_val}'")
						signal_mse = None
					else:
						signal_summary.append({
							"Comparison": model_name,
							"SignalMSE": signal_mse
						})
			except Exception as e:
				print(f"[WARNING] Could not read signal MSE from: {mse_path} ({e})")


		row_entries = []
		ext = ".nii.gz" if input_type == "image" else ".npy"

		for param in ivim_params:
			true_path = os.path.join(baseline_path, f"{param}_synthetic{ext}")
			pred_path = os.path.join(model_path, f"{param}_NN_triexp{ext}")

			if not os.path.exists(true_path):
				print(f"[WARNING] Missing GT: {true_path}")
				continue
			if not os.path.exists(pred_path):
				print(f"[WARNING] Missing pred map: {pred_path}")
				continue

			true = nib.load(true_path).get_fdata().flatten() if input_type == "image" else np.load(true_path).flatten()
			pred = nib.load(pred_path).get_fdata().flatten() if input_type == "image" else np.load(pred_path).flatten()

			folder_check = os.path.basename(os.path.normpath(baseline_path))
			assert subject_id.startswith(folder_check), f"GT folder mismatch: {folder_check} vs {subject_id}"

			valid_mask = np.isfinite(pred) & np.isfinite(true) & (true != 0)
			if valid_mask.sum() < 10:
				print(f"[SKIP] Too few valid voxels for {param} in {model_name}")
				continue

			pred = pred[valid_mask]
			true = true[valid_mask]

			mse = np.mean((pred - true) ** 2)
			rmse = np.sqrt(mse)
			mean_true = np.mean(true)
			nrmse = rmse / (mean_true + 1e-8)
			frac_error = np.mean(np.abs(pred - true) / np.abs(true))
			frac_nrmse_ratio = frac_error / (nrmse + 1e-8)

			row_entries.append({
			"Comparison": model_name,
			"Parameter": param,
			"NRMSE": nrmse,
			"FracError": frac_error,
			"Frac_NRMSE_Ratio": frac_nrmse_ratio,
			"SignalMSE": signal_mse,  # 
			"Subject": subject_id     
		})


		if row_entries:
			all_results.extend(row_entries)
		else:
			print(f"[INFO] Skipping model {model_name} due to missing/invalid parameter maps.")

	df_results = pd.DataFrame(all_results)
	df_signal = pd.DataFrame(signal_summary)

	if df_results.empty or df_signal.empty:
		return df_results, df_signal, {"best_fracerr": ("None", np.inf)}

	# === Model averages from IVIM metrics
	df_avg = df_results.groupby("Comparison", as_index=False)[["NRMSE", "FracError", "Frac_NRMSE_Ratio"]].mean()
	df_avg = df_avg.merge(df_signal, on="Comparison", how="left")
	df_avg["Parameter"] = "Avg IVIM Error"
	# Add Avg IVIM Error row to df_results
	df_results = pd.concat([df_results, df_avg], ignore_index=True)


	# === Determine best models
	best_models = {
		"best_fracerr": df_avg.sort_values("FracError").iloc[0][["Comparison", "FracError"]],
		"best_nrmse":   df_avg.sort_values("NRMSE").iloc[0][["Comparison", "NRMSE"]],
		"best_signal":  df_avg.sort_values("SignalMSE").iloc[0][["Comparison", "SignalMSE"]]
	}

	return df_results, df_signal, best_models



def reshape_df_for_plotting(df_results):
	"""
	Transform df_results from wide to long format to feed into the plotting function.
	"""
	metric_cols = ["NRMSE", "FracError", "Frac_NRMSE_Ratio"]

	# Drop any existing 'Value' to avoid melt conflict
	df_results = df_results.drop(columns=["Value"], errors="ignore")

	df_long = df_results.melt(
		id_vars=["Comparison", "Parameter", "Subject"],
		value_vars=metric_cols,
		var_name="Metric",
		value_name="Value"
	)
	df_long = df_long.rename(columns={"Comparison": "Model"})
	return df_long



def plot_ivim_metric_summary(df, subject_id, noise_mode, metric, out_dir, seed, model_color_dict=None):
	ivim_params = ["Dpar", "Dint", "Dmv", "fint", "fmv", "Avg_IVIM_Err"]
	metric_titles = {
		"NRMSE": "NRMSE",
		"FracError": "Fractional Error",
		"Frac_NRMSE_Ratio": "FracError ÷ NRMSE",
		"SignalMSE": "Signal MSE"
	}

	if metric == "SignalMSE":
		df_signal = df.copy()
		df_signal = df_signal.rename(columns={metric: "Value"}) if metric in df_signal.columns else df_signal
		df_signal["Model"] = df_signal["Comparison"] if "Comparison" in df_signal.columns else df_signal["Model"]
		df_signal = df_signal[["Model", "Value"]].drop_duplicates()
		df_signal = df_signal.sort_values(by="Value")

		df_signal = df_signal.sort_values(by="Value")
		df_signal["Label"] = df_signal["Model"] + " (" + df_signal["Value"].map("{:.3f}".format) + ")"

		plt.figure(figsize=(10, 6))
		sns.barplot(data=df_signal, x="Value", y="Label",
					palette=[model_color_dict.get(m, "gray") for m in df_signal["Model"]] if model_color_dict else "Purples")
		plt.xlabel(metric_titles[metric])
		plt.ylabel("")
		plt.title(f"{subject_id} | Signal MSE – {noise_mode}")
		filename = f"{out_dir}/{subject_id}_SignalMSE_Top_by_model_seed{seed}_{noise_mode}.png"
		plt.tight_layout()
		plt.savefig(filename, dpi=150)
		plt.close()
		print(f"[SAVED] {filename}")
		return

	# Filter and keep only entries matching current metric and IVIM parameters
	df = df[(df["Metric"] == metric) & (df["Parameter"].isin(ivim_params))].copy()

	if df.empty:
		print(f"[WARNING] No data to plot for {metric} in {subject_id}")
		return

	fig, axes = plt.subplots(len(ivim_params), 1, figsize=(10, 18), sharex=False)

	for i, param in enumerate(ivim_params):
		ax = axes[i]
		df_sub = df[df["Parameter"] == param].copy()
		if df_sub.empty:
			continue
		df_sub = df_sub.sort_values(by="Value")
		df_sub["Label"] = df_sub["Model"] + " (" + df_sub["Value"].map("{:.3f}".format) + ")"

		sns.barplot(
			data=df_sub,
			x="Value",
			y="Label",
			ax=ax,
			palette=[model_color_dict.get(m, "gray") for m in df_sub["Model"]] if model_color_dict else "Set2"
		)
		ax.set_title(f"Top Models by {param}")
		ax.set_xlabel(metric_titles[metric])
		ax.set_ylabel("")

	plt.suptitle(f"{subject_id} | {metric_titles[metric]} – {noise_mode}", fontsize=16)
	plt.tight_layout(rect=[0, 0.03, 1, 0.97])
	filename = f"{out_dir}/{subject_id}_{metric}_Top5_by_param_seed{seed}_{noise_mode}.png"
	plt.savefig(filename, dpi=150)
	plt.close()
	print(f"[SAVED] {filename}")



def generate_and_plot_subject_comparison(baseline_path, subject_id, compare_base, save_dir, noise_mode, seed, input_type="array"):
	subject_dir = os.path.join(compare_base, subject_id)

	# Gather valid model folders
	model_folders = [
		name for name in os.listdir(subject_dir)
		if os.path.isdir(os.path.join(subject_dir, name))
	]
	comparison_paths = {
		name: os.path.join(subject_dir, name)
		for name in model_folders
	}

	if not comparison_paths:
		print(f"[SKIP] No model folders for {subject_id}")
		return

	# Run comparison func
	df_results, df_signal, best_models = compare_ivim_errors_and_nrmse(
		baseline_path, comparison_paths, subject_id, input_type
	)
	df_results["Subject"] = subject_id

	# Save per-subject CSV
	subject_out = os.path.join(save_dir, subject_id)
	if os.path.exists(subject_out):
		shutil.rmtree(subject_out)
	Path(subject_out).mkdir(parents=True, exist_ok=True)

	subject_csv = os.path.join(
		subject_out,
		f"ivim_model_comparison_results_{subject_id}_{noise_mode}_seed{seed}.csv"
	)
	df_results.to_csv(subject_csv, index=False, float_format="%.3f")

	# === Add avg row to df_results (needed for non-SignalMSE plotting)
	df_avg = df_results.groupby("Comparison", as_index=False)[["NRMSE", "FracError", "Frac_NRMSE_Ratio"]].mean()
	df_avg = df_avg.merge(df_signal, on="Comparison", how="left")
	df_avg["Parameter"] = "Avg_IVIM_Err"
	df_avg["Subject"] = subject_id
	df_full = pd.concat([df_results, df_avg], ignore_index=True)

	available_models = df_avg["Comparison"].str.lower().unique()
	subject_is_IR = "IR" in subject_id.upper()

	# Find original models that match the subject's IR condition
	originals = []
	# Improved original model matcher
	for model in df_avg["Comparison"].unique():
		name = model.lower()
		if "originalon" in name or model.startswith("2C_OriginalON") or model.startswith("3C_OriginalON"):
			is_model_IR = "_ir" in name
			if is_model_IR == subject_is_IR:
				originals.append(model)


	non_originals_df = df_avg[~df_avg["Comparison"].isin(originals)].copy()

	# Rank models separately for each metric
	metrics_to_plot = ["NRMSE", "FracError", "Frac_NRMSE_Ratio", "SignalMSE"]
	top_models_by_metric = {}
	# Also track unmatched original models
	originals_complement = []
	for model in df_avg["Comparison"].unique():
		name = model.lower()
		if "originalon" in name or model.startswith("2C_OriginalON") or model.startswith("3C_OriginalON"):
			is_model_IR = "_ir" in name
			if is_model_IR != subject_is_IR:
				originals_complement.append(model)

	for metric in metrics_to_plot:
		df_target = df_avg.copy()

		if metric != "SignalMSE":
			# Remove unmatched originals for IVIM metrics
			df_target = df_target[~df_target["Comparison"].isin(originals_complement)]

		df_target[metric] = pd.to_numeric(df_target[metric], errors="coerce")
		ranked = df_target.sort_values(metric)
		top5 = ranked["Comparison"].head(5).tolist()
		combined = list(dict.fromkeys(originals + top5))  # prepend matching originals only
		top_models_by_metric[metric] = combined

         #DEBUG LOGGING
		print(f"[DEBUG] Metric: {metric} → Top5: {top5}")
		print(f"[DEBUG] Originals added: {originals}")
		print(f"[DEBUG] Final combined list ({len(combined)} models): {combined}\n")

	# === Assign consistent color to ALL unique models appearing in any plot
	all_models = set()
	for model_list in top_models_by_metric.values():
		all_models.update(model_list)
	all_models = sorted(all_models)

	palette = sns.color_palette("tab20", n_colors=len(all_models))
	model_color_dict = {model: palette[i] for i, model in enumerate(all_models)}


	# === Plotting
	for metric, models in top_models_by_metric.items():
		if metric == "SignalMSE":
			df_signal_plot = df_signal[df_signal["Comparison"].isin(models)].copy()
			if df_signal_plot.empty:
				print(f"[SKIP] SignalMSE plot: No matching models found in df_signal for {subject_id}")
				continue
			df_signal_plot = df_signal_plot.rename(columns={"SignalMSE": "Value"})
			df_signal_plot["Model"] = df_signal_plot["Comparison"]
			plot_ivim_metric_summary(
				df_signal_plot,
				subject_id=subject_id,
				noise_mode=noise_mode,
				metric=metric,
				out_dir=subject_out,
				seed=seed,
				model_color_dict=model_color_dict
			)
		else:
			df_subset = df_full[df_full["Comparison"].isin(models)].copy()
			df_subset["Metric"] = metric
			df_subset["Value"] = df_subset[metric]
			df_subset["Model"] = df_subset["Comparison"]
			df_subset["Subject"] = subject_id

			df_long_filtered = reshape_df_for_plotting(df_subset)
			if df_long_filtered.empty:
				print(f"[SKIP] Empty plot frame for {metric} in {subject_id}")
				continue

			plot_ivim_metric_summary(
				df_long_filtered,
				subject_id=subject_id,
				noise_mode=noise_mode,
				metric=metric,
				out_dir=subject_out,
				seed=seed,
				model_color_dict=model_color_dict
			)

	print(f"[DONE] {subject_id} | {noise_mode}")



if __name__ == "__main__":
	for noise_mode in noise_modes:
		print(f"\n=== Processing Noise Mode: {noise_mode} ===")

		baseline_base = f"/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025_seed{seed}_{noise_mode}"
		compare_base  = f"/scratch/nhoang2/IVIM_NeuroCovid/Result/Synth_Result_May2025_seed{seed}_{noise_mode}"
		save_dir      = f"{compare_base}/comparison_results_seed{seed}_{noise_mode}"
		os.makedirs(save_dir, exist_ok=True)

		for subject_id in subjects:
			baseline_path = os.path.join(baseline_base, subject_id)
			generate_and_plot_subject_comparison(
				baseline_path=baseline_path,
				subject_id=subject_id,
				compare_base=compare_base,
				save_dir=save_dir,
				noise_mode=noise_mode,
				seed=seed,
				input_type="array"
			)
