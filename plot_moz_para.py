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
    "GGGG", "GGGB", "GBGG", "GBGB",
    "GBBG", "GBBB", "GGBG", "GGBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
color_norms = {"GGGG", "GGGB", "GBGG", "GBGB", "GBBG", "GBBB"}
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}

# 色設定（淡色追加）
norm_colors = {
    "GGGG": "#00429d",
    "GGGB": "#2e60b2",
    "GBGG": "#4671c6",
    "GBGB": "#7ea4d6",
    "GBBG": "#bcd5e6",
    "GBBB": "#e1edf3",
}
# 残りは淡いグレー系
for norm in norm_patterns:
    if norm not in norm_colors:
        norm_colors[norm] = "#f0f0f0"

# データ格納
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
coop_rate = defaultdict(dict)

# 正規表現
file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit5_(\d+)\.csv"
)

# === CSV読み込み ===
for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
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

        # cooperation_rates ファイル名を生成
        coop_file = f"cooperation_rates_400_{public_norm}_probability{prob}_action_error0.001_evaluate_error0.001_public_error0.001_benefit5.csv"
        coop_path = os.path.join(subpath, coop_file)
        if os.path.exists(coop_path):
            try:
                coop_df = pd.read_csv(coop_path)
                sim_cols = [col for col in coop_df.columns if col.startswith("Sim")]
                mean = coop_df[coop_df["Generation"] >= 951][sim_cols].mean().mean()
                coop_rate[public_norm][prob] = mean
            except Exception:
                coop_rate[public_norm][prob] = 0
        else:
            coop_rate[public_norm][prob] = 0

# 平均値を計算
avg_data = defaultdict(dict)
for public_norm in public_norm_targets:
    for prob in sorted(data[public_norm].keys()):
        avg_data[public_norm][prob] = {
            norm: np.mean(data[public_norm][prob][norm])
            for norm in norm_patterns
        }

# === プロット関数 ===
def plot_mosaic(public_norm):
    probs = sorted(avg_data[public_norm].keys())
    fig, ax = plt.subplots(figsize=(len(probs) * 0.4 + 2, 6))

    for j, prob in enumerate(probs):
        norm_vals = avg_data[public_norm][prob]
        bottom = 0
        alpha = 0.4 if coop_rate[public_norm].get(prob, 0) < 0.9 else 1.0
        for idx, norm in enumerate(norm_patterns):
            height = norm_vals[norm]
            color = norm_colors[norm]
            bar = ax.bar(j, height, bottom=bottom, width=1.0, color=color, edgecolor='none', alpha=alpha)
            bottom += height

        # 破線：GBGB → GBBG の間
        if "GBGB" in norm_vals and "GBBG" in norm_vals:
            gbgb_height = sum([norm_vals[n] for n in norm_patterns[:4]])
            ax.plot([j - 0.5, j + 0.5], [gbgb_height, gbgb_height],
                    color="black", linewidth=2.5, linestyle="--", zorder=10)

    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f"{p:.2f}" for p in probs], rotation=90)
    ax.set_xlim(-0.5, len(probs) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Norm Ratio")
    ax.set_title(f"Mosaic Plot of Norms (Public Norm: {public_norm})")

    # 凡例
    legend_patches = [plt.Line2D([0], [0], color=norm_colors[n], lw=8, label=n) for n in color_norms]
    ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.15, 1.0), title="Highlighted Norms")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"mosaic_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

# === 並列実行 ===
if __name__ == "__main__":
    with Pool(processes=min(len(public_norm_targets), cpu_count())) as pool:
        pool.map(plot_mosaic, list(public_norm_targets))
