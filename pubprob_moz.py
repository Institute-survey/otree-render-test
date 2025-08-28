import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import traceback

# === カラー設定 ===
highlight_norms = ["GGGG", "GBGG", "GBGB", "GBBG", "GBBB"]
norm_colors = {
    "GGGG": "#1f77b4",
    "GBGG": "#2ca02c",
    "GBGB": "#ff7f0e",
    "GBBG": "#d62728",
    "GBBB": "#9467bd"
}
default_colors = ["#e0e0e0", "#f5f5f5"]

# === ディレクトリとファイル探索 ===
base_dir = "."  # スクリプトと同階層
pattern = re.compile(r"norm_distribution.*_benefit5_(\d+)\.csv")  # benefit=5かつ末尾に _数字.csv

# === 集計用データ構造 ===
aggregated_counts = defaultdict(lambda: Counter())
norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]

# === 出力先ディレクトリ ===
os.makedirs("plot", exist_ok=True)

# === ファイル処理 ===
for subdir in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, subdir)
    if not os.path.isdir(dir_path) or not subdir.startswith("pubprob_"):
        continue

    for file in glob.glob(os.path.join(dir_path, "norm_distribution*_benefit5_*.csv")):
        try:
            filename = os.path.basename(file)
            match = pattern.match(filename)
            if not match:
                continue

            df = pd.read_csv(file)
            if "Generation" not in df.columns:
                continue

            # 最終50世代を抽出
            df = df[df["Generation"] >= 951]
            agent_cols = [col for col in df.columns if col.startswith("Agent_")]
            norm_data = df[agent_cols]

            # 規範を集計
            all_norms = norm_data.values.flatten()
            counts = Counter(all_norms)

            # 条件キー（試行番号を除いたベース）
            key = re.sub(r"_\d+\.csv$", "", filename.replace("norm_distribution", ""))
            aggregated_counts[key].update(counts)

        except Exception as e:
            print(f"[ERROR] Skipped: {file}")
            traceback.print_exc()

# === モザイクプロット描画 ===
for key, count_dict in aggregated_counts.items():
    total = sum(count_dict.values())
    proportions = {norm: count_dict.get(norm, 0) / total for norm in norm_patterns}

    # モザイクプロット（横方向に積み上げ）
    fig, ax = plt.subplots(figsize=(12, 2))
    current_x = 0.0

    for norm in norm_patterns:
        width = proportions[norm]
        color = norm_colors.get(norm, default_colors[norm_patterns.index(norm) % 2])
        ax.barh(0, width, left=current_x, color=color, edgecolor='black')
        if width > 0.02:
            ax.text(current_x + width / 2, 0, norm, va='center', ha='center', fontsize=8)
        current_x += width

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title(f"Norm Distribution (Avg. of final 50 generations)\nCondition: {key}")

    plt.tight_layout()
    output_path = os.path.join("plot", f"mosaic_{key}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[Saved] {output_path}")
