import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter, defaultdict

# === 設定 ===
base_dir = "."  # スクリプトと同じ階層に pubprob_**** ディレクトリ
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}

# 軸設定
benefits = [round(x, 1) for x in np.arange(1, 5.5, 0.5)]  # 1.0〜5.0
probabilities = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]  # 0.00〜1.00

# 出力用データ
data = defaultdict(lambda: defaultdict(dict))  # data[public_norm][benefit][probability] = simpson_index

# 正規表現パターン
file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)_(\d+)\.csv"
)

# === ファイル走査 ===
for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("pubprob_"):
        continue

    for filepath in glob.glob(os.path.join(subpath, "norm_distribution*.csv")):
        filename = os.path.basename(filepath)
        match = file_pattern.match(filename)
        if not match:
            continue

        public_norm, prob_str, benefit_str, _ = match.groups()
        if public_norm not in public_norm_targets:
            continue

        prob = float(prob_str)
        benefit = float(benefit_str)

        df = pd.read_csv(filepath)
        df = df[df["Generation"] >= 951]  # 最終50世代
        agent_cols = [col for col in df.columns if col.startswith("Agent_")]
        agent_values = df[agent_cols].values.flatten()

        # 規範数に+1するため、すべてのnormに1を加算
        counts = Counter(agent_values)
        norm_counts = np.array([counts.get(norm, 0) + 1 for norm in norm_patterns])
        total = norm_counts.sum()
        proportions = norm_counts / total
        simpson_index = 1 - np.sum(proportions ** 2)

        data[public_norm][benefit][prob] = simpson_index

# === ヒートマップ作成 ===
for public_norm in sorted(public_norm_targets):
    heatmap_data = np.full((len(benefits), len(probabilities)), np.nan)

    for i, benefit in enumerate(benefits):
        for j, prob in enumerate(probabilities):
            val = data[public_norm].get(benefit, {}).get(prob, np.nan)
            heatmap_data[i, j] = val

    plt.figure(figsize=(len(probabilities) * 0.5 + 2, len(benefits) * 0.5 + 2))
    sns.heatmap(
        heatmap_data,
        xticklabels=[f"{p:.2f}" for p in probabilities],
        yticklabels=[f"{b:.1f}" for b in benefits],
        cmap="viridis",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"label": "Simpson Diversity Index"}
    )
    plt.xlabel("Probability")
    plt.ylabel("Benefit")
    plt.title(f"Simpson Diversity Index – Public Norm: {public_norm}")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"simpson_heatmap_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")
