import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# === CONFIGURATION ===
seed = 69
noise_mode = "nonoise"
data_root = f"/scratch/nhoang2/IVIM_NeuroCovid/Data/Synth_Data_May2025"
result_root = "/scratch/nhoang2/IVIM_NeuroCovid/Result/Test_SingleRun_"
output_dir = "/scratch/nhoang2/IVIM_NeuroCovid/Result"
os.makedirs(output_dir, exist_ok=True)

# === MODEL + SUBJECT CONFIGS ===
model_types = ["2C", "3C"]
subjects = ["S1_signal", "NAWM_signal", "WMH_signal"]

# === PARAMS per model ===
ivim_params = {
    "2C": ["Dpar", "Dmv", "fmv"],
    "3C": ["Dpar", "Dmv", "fmv", "Dint", "fint"]
}
all_params = ["Dpar", "Dmv", "Dint", "fmv", "fint"]

records = []

for model in model_types:
    model_result_root = os.path.join(result_root + model)

    for subject in subjects:
        tissue = subject.split("_")[0]
        subject_dir = os.path.join(model_result_root, subject)
        if not os.path.isdir(subject_dir):
            continue

        for mode_folder in os.listdir(subject_dir):
            mode_path = os.path.join(subject_dir, mode_folder)
            if not os.path.isdir(mode_path):
                continue

            mse_file = os.path.join(mode_path, f"global_nrmse_IVIM{model}.txt")
            if not os.path.isfile(mse_file):
                continue
            with open(mse_file, 'r') as f:
                try:
                    nrmse_val = float(f.read().strip())
                except ValueError:
                    nrmse_val = None

            gt_path = os.path.join(f"{data_root}_{model}_seed{seed}_{noise_mode}")
            param_list = ivim_params[model]
            param_errors = {f"{p}_Err": np.nan for p in all_params}
            param_errors.update({f"{p}_{suffix}": np.nan for p in all_params for suffix in ["GT_min", "GT_max", "Pred_min", "Pred_max"]})

            for param in param_list:
                gt_file = os.path.join(gt_path, f"{tissue}_{param}_synthetic.npy")
                pred_suffix = "biexp" if model == "2C" else "triexp"
                pred_file = os.path.join(mode_path, f"{param}_NN_{pred_suffix}.npy")

                print(f"[CHECK] GT: {os.path.basename(gt_file)} ⟷ Pred: {os.path.basename(pred_file)}")

                if not os.path.isfile(gt_file) or not os.path.isfile(pred_file):
                    continue

                gt = np.load(gt_file)
                pred = np.load(pred_file)
                if gt.shape != pred.shape:
                    continue

                nrmse = np.sqrt(np.mean((pred - gt) ** 2)) / max(np.mean(gt), 1e-6)
                param_errors[f"{param}_Err"] = nrmse
                param_errors[f"{param}_GT_min"] = np.min(gt)
                param_errors[f"{param}_GT_max"] = np.max(gt)
                param_errors[f"{param}_Pred_min"] = np.min(pred)
                param_errors[f"{param}_Pred_max"] = np.max(pred)

            param_nrmse_values = [v for k, v in param_errors.items() if k.endswith("_Err") and not np.isnan(v)]
            avg_ivim_err = np.mean(param_nrmse_values) if param_nrmse_values else np.nan

            mode_type = "Original" if "OriginalON" in mode_folder else "Adaptive"
            IR = "IRTrue" in mode_folder

            records.append({
                "ModelType": model,
                "ModeType": mode_type,
                "IR": IR,
                "Subject": subject,
                "Tissue": tissue,
                "Mode": mode_folder,
                "NRMSE": nrmse_val,
                **param_errors,
                "Avg_IVIM_Err": avg_ivim_err,
                "Path": mode_path
            })

# === SAVE TO EXCEL ===
df_final = pd.DataFrame(records)
output_path = os.path.join(output_dir, "IVIM_GTparamErrors_Detail.xlsx")
df_final.to_excel(output_path, index=False)

# === PLOT AVERAGE ERROR (IVIM + Signal Error) ===
df_plot_ivim = df_final.groupby(["ModelType", "Tissue", "ModeType", "IR"])["Avg_IVIM_Err"].mean().reset_index()
df_plot_ivim["Group"] = df_plot_ivim["ModeType"] + "_IR" + df_plot_ivim["IR"].astype(int).astype(str)
df_pivot_ivim = df_plot_ivim.pivot(index=["ModelType", "Tissue"], columns="Group", values="Avg_IVIM_Err").reset_index()
df_pivot_ivim = df_pivot_ivim.sort_values(by=["ModelType", "Tissue"])

df_plot_signal = df_final.groupby(["ModelType", "Tissue", "ModeType", "IR"])["NRMSE"].mean().reset_index()
df_plot_signal["Group"] = df_plot_signal["ModeType"] + "_IR" + df_plot_signal["IR"].astype(int).astype(str)
df_pivot_signal = df_plot_signal.pivot(index=["ModelType", "Tissue"], columns="Group", values="NRMSE").reset_index()
df_pivot_signal = df_pivot_signal.sort_values(by=["ModelType", "Tissue"])
df_pivot_signal = df_pivot_signal.replace(0, 1e-6)

fig, axes = plt.subplots(nrows=2, figsize=(12, 12), sharex=False)
bar_width = 0.2

group_colors = {
    "Original_IR0": 'darkorange',
    "Original_IR1": 'sandybrown',
    "Adaptive_IR0": 'dodgerblue',
    "Adaptive_IR1": 'skyblue'
}

index_ivim = np.arange(len(df_pivot_ivim))
ax1 = axes[0]
ax1.set_xscale("linear")
for i, group in enumerate(group_colors):
    if group in df_pivot_ivim.columns:
        ax1.barh(index_ivim + (i - 1.5) * bar_width,
                 df_pivot_ivim[group], bar_width,
                 label=group, color=group_colors[group])
ax1.set_yticks(index_ivim)
ax1.set_yticklabels(df_pivot_ivim["ModelType"] + "_" + df_pivot_ivim["Tissue"])
ax1.set_xlabel("Avg IVIM Error (NRMSE)")
ax1.set_title("Avg IVIM Parameter Error by Model, Tissue, ModeType, and IR")
ax1.grid(True, axis='x', linestyle='--', alpha=0.6)
ax1.legend()

index_signal = np.arange(len(df_pivot_signal))
ax2 = axes[1]
df_pivot_signal.replace(0, 1e-6, inplace=True)
ax2.set_xscale("log")
for i, group in enumerate(group_colors):
    if group in df_pivot_signal.columns:
        ax2.barh(index_signal + (i - 1.5) * bar_width,
                 df_pivot_signal[group], bar_width,
                 label=group, color=group_colors[group])
ax2.set_yticks(index_signal)
ax2.set_yticklabels(df_pivot_signal["ModelType"] + "_" + df_pivot_signal["Tissue"])
ax2.set_xlabel("Reconstructed Signal Error (NRMSE, log scale)")
ax2.set_title("Avg Reconstructed Signal Error by Model, Tissue, ModeType, and IR")
ax2.grid(True, axis='x', linestyle='--', alpha=0.6, which='both')
ax2.legend()

plot_path = os.path.join(output_dir, "Avg_IVIM_Error_Barplot.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"[SAVED] Plot to: {plot_path}")

# === BARPLOT GT vs PREDICTED RANGE WIDTH PER PARAM ===
for model in model_types:
    param_list = ivim_params[model]
    df_model = df_final[df_final["ModelType"] == model].copy()
    df_model["Group"] = df_model["ModeType"] + "_IR" + df_model["IR"].astype(int).astype(str)

    fig, axs = plt.subplots(nrows=len(param_list), ncols=1, figsize=(12, 4 * len(param_list)), sharex=False)
    if len(param_list) == 1:
        axs = [axs]

    for i, param in enumerate(param_list):
        ax = axs[i]
        df_model[f"{param}_GT_range"] = df_model[f"{param}_GT_max"] - df_model[f"{param}_GT_min"]
        df_model[f"{param}_Pred_range"] = df_model[f"{param}_Pred_max"] - df_model[f"{param}_Pred_min"]

        pivot = df_model.pivot_table(index=["Tissue", "Group"], 
                                     values=[f"{param}_GT_range", f"{param}_Pred_range"],
                                     aggfunc='first').reset_index()
        x_labels = pivot["Tissue"] + "_" + pivot["Group"]
        y = np.arange(len(x_labels))

        ax.barh(y - 0.2, pivot[f"{param}_GT_range"], height=0.4, label="GT", color="darkorange")
        ax.barh(y + 0.2, pivot[f"{param}_Pred_range"], height=0.4, label="Pred", color="dodgerblue")

        ax.set_title(f"{model} - {param} Range Width")
        ax.set_yticks(y)
        ax.set_yticklabels(x_labels)
        ax.set_xlabel("Range (max - min)")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.legend()

        # Add text annotations
        for j, row in pivot.iterrows():
            gt_min = df_model.loc[
                (df_model["Tissue"] == row["Tissue"]) &
                (df_model["Group"] == row["Group"]),
                f"{param}_GT_min"
            ].values[0]
            gt_max = df_model.loc[
                (df_model["Tissue"] == row["Tissue"]) &
                (df_model["Group"] == row["Group"]),
                f"{param}_GT_max"
            ].values[0]
            pred_min = df_model.loc[
                (df_model["Tissue"] == row["Tissue"]) &
                (df_model["Group"] == row["Group"]),
                f"{param}_Pred_min"
            ].values[0]
            pred_max = df_model.loc[
                (df_model["Tissue"] == row["Tissue"]) &
                (df_model["Group"] == row["Group"]),
                f"{param}_Pred_max"
            ].values[0]

            ax.text(row[f"{param}_GT_range"] + 1e-5, j - 0.2,
                    f"[{gt_min:.3g}, {gt_max:.3g}]", va='center', ha='left', fontsize=8, color='darkorange')
            ax.text(row[f"{param}_Pred_range"] + 1e-5, j + 0.2,
                    f"[{pred_min:.3g}, {pred_max:.3g}]", va='center', ha='left', fontsize=8, color='dodgerblue')


    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"GT_vs_Pred_RangeWidth_{model}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] Range width plot to: {plot_path}")
