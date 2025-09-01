import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
from concurrent.futures import ProcessPoolExecutor

# === 設定 ===
base_dir = "."  # pubprob_xxxx フォルダ群のルート
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}
benefits = [round(x * 0.5, 1) for x in range(2, 11)]  # 1.0 ~ 5.0
probabilities = [round(x * 0.05, 2) for x in range(21)]  # 0.0 ~ 1.0

file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)_(\d+)\.csv"
)

def simpson_index(counts):
    total = sum(counts)
    if total <= 1:
        return 0
    return 1 - sum((n / total) ** 2 for n in counts)

def process_public_norm_simpson(public_norm):
    grid = np.full((len(benefits), len(probabilities)), np.nan)

    for subdir in os.listdir(base_dir):
        subpath = os.path.join(base_dir, subdir)
        if not os.path.isdir(subpath) or not subdir.startswith("pubprob_"):
            continue

        for csv_file in glob.glob(os.path.join(subpath, "norm_distribution*.csv")):
            filename = os.path.basename(csv_file)
            match = file_pattern.match(filename)
            if not match:
                continue

            pub_norm, prob, benefit, trial = match.groups()
            if pub_norm != public_norm:
                continue

            prob = float(prob)
            benefit = float(benefit)
            if prob not in probabilities or benefit not in benefits:
                continue

            try:
                df = pd.read_csv(csv_file)
                df = df[df["Generation"] >= 951]
                agent_cols = [col for col in df.columns if col.startswith("Agent_")]
                if df.empty or not agent_cols:
                    continue

                norm_counts = Counter()
                for _, row in df[agent_cols].iterrows():
                    norm_counts.update(row)
                final_counts = [norm_counts.get(norm, 0) + 1 for norm in norm_patterns]
                simpson_val = simpson_index(final_counts)

                # 協力率判定
                coop_file = filename.replace("norm_distribution", "cooperation_rates")
                coop_path = os.path.join(subpath, coop_file.rsplit("_", 1)[0] + ".csv")
                coop_df = pd.read_csv(coop_path)
                sim_col = f"Sim{int(trial)}"
                if sim_col not in coop_df.columns or "Generation" not in coop_df.columns:
                    continue
                coop_df = coop_df[coop_df["Generation"] >= 951]
                coop_mean = coop_df[sim_col].mean()

                bi = benefits.index(benefit)
                pi = probabilities.index(prob)
                if coop_mean >= 0.8:
                    grid[bi, pi] = simpson_val
                else:
                    grid[bi, pi] = np.nan
            except Exception:
                continue

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        grid, cmap="viridis", vmin=0, vmax=1,
        xticklabels=probabilities, yticklabels=benefits,
        mask=np.isnan(grid), cbar=True,
        linewidths=0.5, linecolor="gray"
    )
    ax.set_title(f"Simpson Index (Public Norm: {public_norm})")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Benefit")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"simpson_index_{public_norm}.png"), dpi=300)
    plt.close()

# 実行
with ProcessPoolExecutor() as executor:
    executor.map(process_public_norm_simpson, public_norm_targets)
