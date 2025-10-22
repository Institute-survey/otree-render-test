import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import numpy as np
from multiprocessing import Pool, cpu_count

# === 設定 ===
base_dir = "."
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

# 修正された規範の順番
norm_patterns = [
    "GGGG", "GGGB", "GBGG", "GBGB",
    "GBBG", "GBBB", "GGBG", "GGBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]

# GGGB を含めた強調色規範セット
color_norms = {"GGGG", "GGGB", "GBGG", "GBGB", "GBBG", "GBBB"}
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}

# データ格納
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# ファイル名の正規表現
file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit5_(\d+)\.csv"
)

# === データ読み込み ===
for subdir in os.listdir(base_dir):
    subpath = os.path.join(subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("pubprob"):
        continue

    for csv_path in glob.glob(os.path.join(subpath, "norm_distribution*.csv")):
        filename = os.path.basename(csv_path)
        match = file_pattern.match(filename)
        if not match:
            continue

        public_norm, prob, sim = match.groups()
        prob = float(prob)
        if public_norm not in public_norm_targets:
            continue

        df = pd.read_csv(csv_path)
        df = df[df["Generation"] >= 951]

        agent_cols = [col for col in df.columns if col.startswith("Agent_")]
        for _, row in df[agent_cols].iterrows():
            counts = Counter(row)
            total = len(row)
            for norm in norm_patterns:
                ratio = counts.get(norm, 0) / total
                data[public_norm][prob][norm].append(ratio)

# === 平均値を計算 ===
avg_data = defaultdict(dict)
for public_norm in public_norm_targets:
    for prob in sorted(data[public_norm].keys()):
        avg_data[public_norm][prob] = {
            norm: np.mean(data[public_norm][prob][norm])
            for norm in norm_patterns
        }

# === プロット関数（並列化） ===
def plot_mosaic(public_norm):
    probs = sorted(avg_data[public_norm].keys())
    bar_width = 1.0
    fig, ax = plt.subplots(figsize=(len(probs) * 0.5 + 2, 6))

    for i, prob in enumerate(probs):
        norm_vals = avg_data[public_norm][prob]
        bottom = 0
        for idx, norm in enumerate(norm_patterns):
            height = norm_vals[norm]
            if norm in color_norms:
                color = {
                    "GGGG": "#00429d",
                    "GGGB": "#225ea8",
                    "GBGG": "#4671c6",
                    "GBGB": "#7ea4d6",
                    "GBBG": "#bcd5e6",
                    "GBBB": "#e1edf3"
                }[norm]
                edgecolor = 'black'
            else:
                color = "white"
                edgecolor = 'none'

            ax.bar(i, height, width=bar_width, bottom=bottom,
                   color=color, edgecolor=edgecolor, linewidth=0.5, align='edge')
            bottom += height

            # --- GBBBとの境界線（GBGB → GBBGの間） ---
            if norm == "GBGB":
                ax.plot([i, i + bar_width], [bottom, bottom],
                        linestyle='--', linewidth=2, color='black')

    ax.set_xticks(np.arange(len(probs)) + bar_width / 2)
    ax.set_xticklabels([f"{p:.2f}" for p in probs], rotation=90)
    ax.set_xlim(0, len(probs))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Norm Ratio")
    ax.set_xlabel("Probability")
    ax.set_title(f"Mosaic Plot of Norms (Public Norm: {public_norm})")

    # 外枠の非表示
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"mosaic_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

# === 並列処理実行 ===
if __name__ == "__main__":
    with Pool(processes=min(len(public_norm_targets), cpu_count())) as pool:
        pool.map(plot_mosaic, list(public_norm_targets))
