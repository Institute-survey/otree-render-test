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
def morishita_index(counts, total_agents):
    A = len(counts)
    N = total_agents
    if N == 0 or A == 0:
        return np.nan
    S1 = sum(n * (n - 1) for n in counts)
    return (A / (N * (N - 1))) * S1

def process_public_norm_morishita(public_norm):
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
            counts = [all_counts.get(norm, 0) + 1 for norm in norm_patterns]
            total_agents = len(agent_cols) * len(df)
            m_index = morishita_index(counts, total_agents)

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
                grid[bi, pi] = m_index
            else:
                grid[bi, pi] = np.nan  # グレー

    # プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        grid, cmap="viridis", vmin=0, vmax=np.nanmax(grid),
        xticklabels=probabilities,
        yticklabels=list(reversed(benefits)),
        mask=np.isnan(grid), linewidths=0.5, linecolor='gray'
    )
    ax.set_title(f"Morishita Index Cλ – {public_norm}")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Benefit")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"morishita_index_{public_norm}.png"), dpi=300)
    plt.close()

with ProcessPoolExecutor() as executor:
    executor.map(process_public_norm_morishita, public_norm_targets)
