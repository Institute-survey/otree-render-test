# 共通インポート
import os
import re
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# 共通設定
base_dir = "."  # スクリプト直下にpubprob_ディレクトリ群
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

benefits = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
probabilities = [round(x * 0.05, 2) for x in range(21)]  # 0.00 ~ 1.00
norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
public_norm_targets = ["GBBB", "GBBG", "GBGB", "GBGG"]
file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)_(\d+)\.csv"
)
def simpson_index(counts):
    total = sum(counts)
    if total == 0:
        return np.nan
    p = [c / total for c in counts]
    return 1 - sum([v**2 for v in p])

def process_public_norm_simpson(public_norm):
    grid = np.full((len(benefits), len(probabilities)), np.nan)

    for subdir in os.listdir(base_dir):
        path = os.path.join(base_dir, subdir)
        if not os.path.isdir(path) or not subdir.startswith("pubprob_"):
            continue

        for file in glob.glob(os.path.join(path, "norm_distribution*.csv")):
            m = file_pattern.match(os.path.basename(file))
            if not m:
                continue
            pn, prob, benefit, trial = m.groups()
            if pn != public_norm:
                continue
            prob = float(prob)
            benefit = float(benefit)
            if prob not in probabilities or benefit not in benefits:
                continue

            df = pd.read_csv(file)
            df = df[df["Generation"] >= 951]
            agent_cols = [c for c in df.columns if c.startswith("Agent_")]
            if df.empty: continue

            all_counts = Counter()
            for _, row in df[agent_cols].iterrows():
                all_counts.update(row)
            # +1スムージング
            counts = [all_counts.get(norm, 0) + 1 for norm in norm_patterns]
            diversity = simpson_index(counts)

            # cooperation_rateの平均チェック
            coop_file = file.replace("norm_distribution", "cooperation_rates")
            coop_file = coop_file.rsplit("_", 1)[0] + ".csv"
            coop_path = os.path.join(path, coop_file)
            coop_df = pd.read_csv(coop_path)
            coop_df.columns = coop_df.columns.str.strip()
            sim_col = f"Sim{int(trial)}"
            if sim_col not in coop_df.columns:
                continue
            coop_avg = coop_df[coop_df["Generation"] >= 951][sim_col].mean()

            bi = benefits.index(benefit)
            pi = probabilities.index(prob)
            if coop_avg >= 0.8:
                grid[bi, pi] = diversity
            else:
                grid[bi, pi] = np.nan  # グレー表示

    # プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        grid, cmap="YlGnBu", vmin=0, vmax=1,
        xticklabels=probabilities,
        yticklabels=list(reversed(benefits)),
        mask=np.isnan(grid), linewidths=0.5, linecolor='gray'
    )
    ax.set_title(f"Simpson's Diversity Index – {public_norm}")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Benefit")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"simpson_index_{public_norm}.png"), dpi=300)
    plt.close()

with ProcessPoolExecutor() as executor:
    executor.map(process_public_norm_simpson, public_norm_targets)
