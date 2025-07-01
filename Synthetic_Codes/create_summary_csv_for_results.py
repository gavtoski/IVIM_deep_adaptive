import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import MaxNLocator

def extract_best_models_detailed(base_dir, seeds, noise_modes, subjects, output_csv):
    results = []

    for seed in seeds:
        for noise in noise_modes:
            for subject in subjects:
                folder = os.path.join(
                    base_dir,
                    f"Synth_Result_May2025_seed{seed}_{noise}",
                    f"comparison_results_seed{seed}_{noise}",
                    subject
                )
                csv_name = f"ivim_model_comparison_results_{subject}_{noise}_seed{seed}.csv"
                csv_path = os.path.join(folder, csv_name)

                if not os.path.exists(csv_path):
                    print(f"[SKIP] Missing: {csv_path}")
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    ivim_params = ["Dpar", "Dint", "Dmv", "fint", "fmv"]
                    df_ivim = df[df["Parameter"].isin(ivim_params)]
                    grouped = df_ivim.groupby("Comparison")[["FracError", "NRMSE", "Frac_NRMSE_Ratio", "SignalMSE"]].mean()

                    subject_is_IR = "_IR" in subject.upper()

                    # === STEP 1: Get best model — only matching IR condition
                    grouped_filtered = grouped.copy()
                    for model in grouped.index:
                        name = model.lower()
                        is_model_IR = "_ir" in name or "ir1" in name
                        if is_model_IR != subject_is_IR:
                            grouped_filtered = grouped_filtered.drop(index=model)

                    if grouped_filtered.empty:
                        print(f"[WARNING] No matching IR models for {subject} — skipping best model selection")
                        continue

                    best_model_fracerr = grouped_filtered.sort_values("FracError").index[0]
                    best_model_nrmse   = grouped_filtered.sort_values("NRMSE").index[0]
                    best_model_ratio   = grouped_filtered.sort_values("Frac_NRMSE_Ratio").index[0]
                    best_model_signal  = grouped_filtered.sort_values("SignalMSE").index[0]

                    avg_row_fracerr = grouped.loc[best_model_fracerr]
                    avg_row_nrmse   = grouped.loc[best_model_nrmse]
                    avg_row_ratio   = grouped.loc[best_model_ratio]
                    avg_row_signal  = grouped.loc[best_model_signal]

                    # === STEP 2: Write one row for EACH model (even IR-mismatched)
                    for model in grouped.index:
                        df_model_params = df_ivim[df_ivim["Comparison"] == model]
                        avg_row = grouped.loc[model]

                        output_row = {
                            "ID": f"{subject}_{model}_{noise}_seed{seed}",
                            "Subject": subject,
                            "Noise": noise,
                            "Seed": seed,
                            "Model": model,
                            "FracError_IVIM": avg_row["FracError"],
                            "NRMSE_IVIM": avg_row["NRMSE"],
                            "Frac_NRMSE_Ratio_IVIM": avg_row["Frac_NRMSE_Ratio"],
                            "SignalMSE_IVIM": avg_row["SignalMSE"],
                            "Best_Model_FracErr": best_model_fracerr,
                            "Best_Model_NRMSE": best_model_nrmse,
                            "Best_Model_Ratio": best_model_ratio,
                            "Best_Model_SignalMSE": best_model_signal
                        }

                        for _, row in df_model_params.iterrows():
                            prefix = row["Parameter"]
                            output_row[f"{prefix}_FracError"] = row["FracError"]
                            output_row[f"{prefix}_NRMSE"] = row["NRMSE"]
                            output_row[f"{prefix}_Frac_NRMSE_Ratio"] = row["Frac_NRMSE_Ratio"]

                        results.append(output_row)

                except Exception as e:
                    import traceback
                    print(f"[ERROR] Exception in file: {csv_path}")
                    traceback.print_exc()

    df_best = pd.DataFrame(results)
    if df_best.empty:
        print("[WARNING] No results to write.")
    else:
        try:
            df_best.to_csv(output_csv, index=False)
            print(f"[SAVED] {output_csv}")
        except Exception as e:
            print(f"[ERROR] Failed to write CSV: {e}")


def summarize_and_plot_best_models(detailed_csv, combined_csv, out_dir, date_str):
    if not os.path.exists(detailed_csv):
        print(f"[ERROR] Missing input CSV: {detailed_csv}")
        return

    df = pd.read_csv(detailed_csv)
    metrics = ["Best_Model_FracErr", "Best_Model_NRMSE", "Best_Model_Ratio", "Best_Model_SignalMSE"]

    # Add IR flag
    df["IR_Flag"] = df["Subject"].str.contains("_IR", case=False).map({True: "IR", False: "nonIR"})

    # Save grouped summary
    grouped_summary = df[["Subject", "Noise", "IR_Flag"] + metrics].copy()
    grouped_summary.to_csv(combined_csv, index=False)
    print(f"[SAVED] {combined_csv}")

    # === Build win counts split by IR_Flag
    all_counts = []
    for ir_flag, df_sub in df.groupby("IR_Flag"):
        wins = pd.concat([
            df_sub[[metric]].rename(columns={metric: "Model"}).assign(Metric=metric)
            for metric in metrics
        ])
        wins["IR_Flag"] = ir_flag
        all_counts.append(wins)

    df_counts = pd.concat(all_counts, ignore_index=True)

    # === Plot (2,1) IR/nonIR global comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    for idx, ir_flag in enumerate(["nonIR", "IR"]):
        df_ir = df_counts[df_counts["IR_Flag"] == ir_flag]
        win_counts = df_ir["Model"].value_counts().reset_index()
        win_counts.columns = ["Model", "Win_Count"]

        sns.barplot(data=win_counts, x="Win_Count", y="Model", order=win_counts["Model"], ax=axes[idx])
        axes[idx].set_title(f"Model Win Count — {ir_flag}")
        axes[idx].set_xlabel("Number of Wins")
        axes[idx].set_ylabel("Model")

    plt.tight_layout()
    plot_filename = f"Modelwincounts_splitIR_{date_str}.png"
    save_path = os.path.join(out_dir, plot_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] {save_path}")

    # === 3x1 plots per noise mode per IR split
    noise_modes = df["Noise"].unique()
    for ir_flag in ["nonIR", "IR"]:
        df_ir = df_counts[df_counts["IR_Flag"] == ir_flag]
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

        for i, noise in enumerate(sorted(noise_modes)):
            df_sub = df[df["IR_Flag"] == ir_flag]
            df_sub = df_sub[df_sub["Noise"] == noise]

            wins = pd.concat([
                df_sub[[metric]].rename(columns={metric: "Model"}).assign(Metric=metric)
                for metric in metrics
            ])
            win_counts = wins["Model"].value_counts().reset_index()
            win_counts.columns = ["Model", "Win_Count"]

            sns.barplot(data=win_counts, x="Win_Count", y="Model", order=win_counts["Model"], ax=axes[i])
            axes[i].set_title(f"{ir_flag} — Noise: {noise}")
            axes[i].set_xlabel("Number of Wins")
            axes[i].set_ylabel("Model")

        plt.tight_layout()
        plot_filename = f"Modelwincounts_{ir_flag}_byNoise_{date_str}.png"
        save_path = os.path.join(out_dir, plot_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[SAVED] {save_path}")


def plot_wins_grouped_by_noise(detailed_csv, output_csv, out_dir, date_str):
    df = pd.read_csv(detailed_csv)
    metrics = ["Best_Model_FracErr", "Best_Model_NRMSE", "Best_Model_Ratio", "Best_Model_SignalMSE"]
    metric_labels = ["FracError", "NRMSE", "Ratio", "SignalMSE"]

    df["Base_Subject"] = df["Subject"].str.replace("_IR", "", case=False)
    df["IR_Flag"] = df["Subject"].str.contains("_IR", case=False).map({True: "IR", False: "nonIR"})

    subject_palette = {
        "S1_signal": "#1f77b4",
        "NAWM_signal": "#2ca02c",
        "WMH_signal": "#ff7f0e"
    }
    subject_order = ["S1_signal", "NAWM_signal", "WMH_signal"]
    noise_levels = ["nonoise", "lownoise", "highnoise"]

    def shorten_label(model_name):
        for prefix in ["IR0_", "IR1_"]:
            if prefix in model_name:
                return "\n".join(model_name.split(prefix, 1))
        return model_name

    # === Aggregate win counts
    all_counts = []
    for noise in noise_levels:
        df_noise = df[df["Noise"] == noise]
        for ir_flag in ["nonIR", "IR"]:
            df_subset = df_noise[df_noise["IR_Flag"] == ir_flag]
            for metric, label in zip(metrics, metric_labels):
                model_order_full = df_subset[metric].dropna().astype(str).unique().tolist()
                label_map = {m: shorten_label(m) for m in model_order_full}
                for subj in subject_order:
                    df_subj = df_subset[df_subset["Base_Subject"] == subj]
                    grouped = df_subj.groupby(metric).size().reset_index(name="Win_Count")
                    grouped = grouped.rename(columns={metric: "Model"})
                    grouped["Subject"] = subj
                    grouped["Model_Display"] = grouped["Model"].map(label_map)
                    grouped["IR_Flag"] = ir_flag
                    grouped["Noise"] = noise
                    grouped["Metric"] = label
                    grouped = grouped[grouped["Win_Count"] > 0]
                    all_counts.append(grouped)

    df_summary = pd.concat(all_counts, ignore_index=True)
    df_summary.to_csv(output_csv, index=False)
    print(f"[SAVED] Win summary CSV: {output_csv}")

    # === Horizontal plots per metric × IR flag ===
    for label in metric_labels:
        for ir_flag in ["nonIR", "IR"]:
            df_plot = df_summary[(df_summary["Metric"] == label) & (df_summary["IR_Flag"] == ir_flag)].copy()
            if df_plot.empty:
                print(f"[SKIP] No data for {label} {ir_flag}")
                continue

            # Define fixed model order (Y-axis category space)
            model_order = (
                df_plot.groupby("Model_Display")["Win_Count"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

            # Get all model-tissue combinations to guarantee fixed height
            all_model_subject_pairs = (
                df_plot[["Model_Display", "Subject"]]
                .drop_duplicates()
                .sort_values(["Model_Display", "Subject"])
            )

            fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharey=True)

            for i, noise in enumerate(noise_levels):
                ax = axes[i]
                df_sub = df_plot[df_plot["Noise"] == noise]

                # Expand grid to fixed bar count
                full_df = all_model_subject_pairs.copy()
                full_df["Noise"] = noise
                full_df["IR_Flag"] = ir_flag
                full_df["Metric"] = label
                full_df = full_df.merge(df_sub, on=["Model_Display", "Subject", "Noise", "IR_Flag", "Metric"], how="left")
                full_df["Win_Count"] = full_df["Win_Count"].fillna(0)

                sns.barplot(
                    data=full_df,
                    y="Model_Display",
                    x="Win_Count",
                    hue="Subject",
                    hue_order=subject_order,
                    palette=subject_palette,
                    ax=ax,
                    order=model_order
                )

                ax.set_title(f"{label} Win Counts — {noise}", fontsize=12)
                ax.set_xlabel("Win Count")
                ax.set_ylabel("Model")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(True, axis='x', linestyle='--', alpha=0.5)

                if i == 2:
                    ax.legend(loc="lower right", fontsize=9)
                else:
                    ax.get_legend().remove()

            plt.suptitle(f"{label} Win Counts — IR: {ir_flag}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = f"Grouped_WinSummary_{label}_{ir_flag}_{date_str}.png"
            plt.savefig(os.path.join(out_dir, filename), dpi=300)
            plt.close()
            print(f"[SAVED] {filename}")




if __name__ == "__main__":
    date_str = datetime.now().strftime("%m_%d_%Y")
    out_dir = "/scratch/nhoang2/IVIM_NeuroCovid/Result"
    best_model_csv = f"{out_dir}/best_models_{date_str}.csv"
    summary_csv = f"{out_dir}/summary_combined_{date_str}.csv"

    extract_best_models_detailed(
        base_dir=out_dir,
        seeds=[24, 69, 97],
        noise_modes=["nonoise", "lownoise", "highnoise"],
        subjects=[
            "S1_signal", "S1_signal_IR",
            "NAWM_signal", "NAWM_signal_IR",
            "WMH_signal", "WMH_signal_IR"
        ],
        output_csv=best_model_csv
    )

    summarize_and_plot_best_models(
        detailed_csv=best_model_csv,
        combined_csv=summary_csv,
        out_dir=out_dir,
        date_str=date_str
    )

    plot_wins_grouped_by_noise(
        detailed_csv=best_model_csv,
        output_csv=summary_csv,
        out_dir=out_dir,
        date_str=date_str
    )

