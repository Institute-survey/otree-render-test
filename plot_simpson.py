# simpson_heatmap_filtered.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter, defaultdict

base_dir = "."
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}

benefits = [round(x, 1) for x in np.arange(1.0, 5.5, 0.5)]
probabilities = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]

file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)_(\d+)\.csv"
)

# 平均協力率データを取得
cooperation_rate = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("pubprob"):
        continue

    for coop_file in glob.glob(os.path.join(subpath, "cooperation_rates_*.csv")):
        match = re.match(
            r"cooperation_rates_400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)\.csv",
            os.path.basename(coop_file)
        )
        if not match:
            continue
        norm, prob, benefit = match.groups()
        prob = float(prob)
        benefit = float(benefit)
        if norm not in public_norm_targets:
            continue

        df = pd.read_csv(coop_file)
        if "Generation" not in df.columns:
            continue

        df = df[df["Generation"] >= 951]
        for col in [c for c in df.columns if c.startswith("Sim")]:
            cooperation_rate[norm][benefit][prob].append(df[col].mean())

# Simpson index 計算とヒートマップ生成
for public_norm in sorted(public_norm_targets):
    heatmap_data = np.full((len(benefits), len(probabilities)), np.nan)

    for filepath in glob.glob(f"{base_dir}/**/norm_distribution400_{public_norm}_*.csv", recursive=True):
        match = file_pattern.match(os.path.basename(filepath))
        if not match:
            continue

        _, prob_str, benefit_str, _ = match.groups()
        prob = float(prob_str)
        benefit = float(benefit_str)
        if benefit not in benefits or prob not in probabilities:
            continue

        coop_vals = cooperation_rate[public_norm][benefit][prob]
        if not coop_vals or np.mean(coop_vals) < 0.8:
            simpson_index = 0  # 条件を満たさない → 色なし
        else:
            df = pd.read_csv(filepath)
            df = df[df["Generation"] >= 951]
            agent_cols = [c for c in df.columns if c.startswith("Agent_")]
            norm_vals = df[agent_cols].values.flatten()
            counts = Counter(norm_vals)
            norm_counts = np.array([counts.get(norm, 0) + 1 for norm in norm_patterns])
            proportions = norm_counts / norm_counts.sum()
            simpson_index = 1 - np.sum(proportions ** 2)

        i = len(benefits) - 1 - benefits.index(benefit)  # 下から上へ
        j = probabilities.index(prob)
        heatmap_data[i, j] = simpson_index

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=[f"{p:.2f}" for p in probabilities],
        yticklabels=[f"{b:.1f}" for b in reversed(benefits)],
        cmap="Greens",
        vmin=0, vmax=1,
        linewidths=0.5, linecolor='gray',
        cbar_kws={"label": "Simpson Diversity Index"}
    )
    plt.xlabel("Probability")
    plt.ylabel("Benefit")
    plt.title(f"Simpson Diversity Index – Public Norm: {public_norm}")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"simpson_heatmap_filtered_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")
