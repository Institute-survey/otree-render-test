import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter, defaultdict

# === 設定 ===
base_dir = "."  # スクリプトと同じ階層に pubprob_ フォルダ群がある前提
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

# 規範のパターン
norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}

# === パラメータ定義 ===
benefits = [round(x * 0.5, 1) for x in range(2, 11)]  # 1.0 ~ 5.0
probabilities = [round(x * 0.05, 2) for x in range(21)]  # 0.00 ~ 1.00

# === 正規表現 ===
file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)_(\d+)\.csv"
)

# === データ読み込みと分布構築 ===
# distributions[public_norm][benefit][probability] = Counter(norms)
distributions = defaultdict(lambda: defaultdict(dict))

for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("pubprob_"):
        continue

    for csv_path in glob.glob(os.path.join(subpath, "norm_distribution*.csv")):
        filename = os.path.basename(csv_path)
        match = file_pattern.match(filename)
        if not match:
            continue

        public_norm, prob, benefit, _ = match.groups()
        prob = float(prob)
        benefit = float(benefit)

        if public_norm not in public_norm_targets:
            continue

        df = pd.read_csv(csv_path)
        df = df[df["Generation"] >= 951]
        agent_cols = [col for col in df.columns if col.startswith("Agent_")]
        norm_counts = Counter(df[agent_cols].values.flatten())
        norm_counts = {k: v + 1 for k, v in norm_counts.items()}  # add 1 to each for smoothing
        total = sum(norm_counts.values())
        norm_freq = {k: norm_counts.get(k, 1) / total for k in norm_patterns}
        distributions[public_norm][benefit][prob] = norm_freq

# === Morishita類似度 Cλ の計算 ===
def morishita_index(p, q):
    numer = sum(p[n] * q[n] for n in norm_patterns)
    denom = (sum(p[n]**2 for n in norm_patterns) ** 0.5) * (sum(q[n]**2 for n in norm_patterns) ** 0.5)
    return numer / denom if denom != 0 else np.nan

# === 類似度行列の作成と描画 ===
for public_norm in public_norm_targets:
    fig, ax = plt.subplots(figsize=(12, 6))
    similarity_matrix = np.full((len(benefits), len(probabilities)), np.nan)

    for i, benefit in enumerate(benefits):
        if 0.0 not in distributions[public_norm][benefit]:
            continue
        base_dist = distributions[public_norm][benefit][0.0]
        for j, prob in enumerate(probabilities):
            if prob not in distributions[public_norm][benefit]:
                continue
            current_dist = distributions[public_norm][benefit][prob]
            similarity = morishita_index(base_dist, current_dist)
            similarity_matrix[i, j] = similarity

    sns.heatmap(similarity_matrix, xticklabels=probabilities, yticklabels=benefits,
                cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "Cλ Similarity"})
    ax.set_title(f"Morishita Cλ Similarity (vs prob=0) - Public Norm: {public_norm}")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Benefit")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"morishita_similarity_{public_norm}.png"), dpi=300)
    plt.close()
    print(f"[Saved] morishita_similarity_{public_norm}.png")
