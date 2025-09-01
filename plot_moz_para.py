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

norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
color_norms = {"GGGG", "GBGG", "GBGB", "GBBG", "GBBB"}
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}

# === データ格納 ===
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# === 正規表現パターン ===
file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit5_(\d+)\.csv"
)

# === CSV読み込み ===
for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("pubprob_"):
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

# === プロット関数（並列化対象） ===
def plot_mosaic(public_norm):
    probs = sorted(avg_data[public_norm].keys())
    bar_height = 0.8
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(probs) + 1))

    for i, prob in enumerate(probs):
        norm_vals = avg_data[public_norm][prob]
        left = 0
        for norm in norm_patterns:
            width = norm_vals[norm]
            color = "white"
            edgecolor = "black"
            if norm in color_norms:
                color = {
                    "GGGG": "#00429d",
                    "GBGG": "#4671c6",
                    "GBGB": "#7ea4d6",
                    "GBBG": "#bcd5e6",
                    "GBBB": "#e1edf3"
                }[norm]
            ax.barh(i, width, left=left, height=bar_height, color=color, edgecolor='black')
            left += width

    ax.set_yticks(range(len(probs)))
    ax.set_yticklabels([f"{p:.2f}" for p in probs])
    ax.set_xlabel("Norm Ratio")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, 1)
    ax.set_title(f"Mosaic Plot of Norms (Public Norm: {public_norm})")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"mosaic_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

# === 並列実行 ===
if __name__ == "__main__":
    with Pool(processes=min(len(public_norm_targets), cpu_count())) as pool:
        pool.map(plot_mosaic, list(public_norm_targets))
