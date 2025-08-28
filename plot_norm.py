import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import glob
import re
import traceback

# === 設定 ===
base_dir = "."  # スクリプトと同じディレクトリ
norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]

def compute_norm_ratios(df):
    generations = df["Generation"].values
    agent_data = df.drop(columns=["Generation"])
    norm_ratio_per_gen = []
    for _, row in agent_data.iterrows():
        counts = Counter(row)
        total = len(row)
        ratios = {norm: counts.get(norm, 0) / total for norm in norm_patterns}
        norm_ratio_per_gen.append(ratios)
    ratio_df = pd.DataFrame(norm_ratio_per_gen)
    ratio_df["Generation"] = generations
    return ratio_df

def plot_norm_and_coop(ratio_df, coop_df, sim_col, title, output_path):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    for norm in norm_patterns:
        ax1.plot(ratio_df["Generation"], ratio_df[norm], label=norm)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Norm Proportion")
    ax1.set_ylim(0, 1)
    ax1.set_title(title)

    ax2 = ax1.twinx()
    if sim_col in coop_df.columns:
        ax2.plot(coop_df["Generation"], coop_df[sim_col],
                 color="black", linestyle="dotted", linewidth=2, label="Cooperation Rate")
        ax2.set_ylabel("Cooperation Rate")
        ax2.set_ylim(0, 1)

    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Saved] {output_path}")

# === メイン ===
for subdir in os.listdir(base_dir):
    path = os.path.join(base_dir, subdir)
    if not os.path.isdir(path) or not subdir.startswith("pubprob_"):
        continue

    csv_files = glob.glob(os.path.join(path, "norm_distribution*.csv"))

    for csv_file in csv_files:
        try:
            filename = os.path.basename(csv_file)
            df = pd.read_csv(csv_file)

            valid_cols = ["Generation"] + [col for col in df.columns if col.startswith("Agent_")]
            df = df[valid_cols]
            ratio_df = compute_norm_ratios(df)

            base_name = filename.replace("norm_distribution", "").replace(".csv", "")
            match = re.match(r"(.+?)_(\d+)$", base_name)
            if not match:
                print(f"[Skip] Could not extract trial number from {filename}")
                continue

            file_core, trial_suffix = match.groups()
            trial_num = int(trial_suffix) - 1  # Sim列に対応（Sim0〜Sim9）

            coop_name = f"cooperation_rates_{file_core}.csv"
            coop_path = os.path.join(path, coop_name)
            if not os.path.exists(coop_path):
                print(f"[Warning] No cooperation_rates file for {filename}")
                continue

            coop_df = pd.read_csv(coop_path)
            coop_df.columns = [col.strip() for col in coop_df.columns]
            if "Generation" not in coop_df.columns or f"Sim{trial_num}" not in coop_df.columns:
                print(f"[Warning] Sim{trial_num} not found in {coop_name}")
                continue

            coop_df = coop_df[["Generation", f"Sim{trial_num}"]]
            output_png = f"{os.path.splitext(filename)[0]}.png"
            plot_norm_and_coop(ratio_df, coop_df, f"Sim{trial_num}", filename[:-4], output_png)

        except Exception as e:
            print(f"[ERROR] Skipping {csv_file}: {e}")
            traceback.print_exc()
