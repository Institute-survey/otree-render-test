import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import glob
import re
from concurrent.futures import ProcessPoolExecutor

# === 設定 ===
base_dir = "."  # スクリプト設置ディレクトリから探索
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)
target_benefit = "5"
target_norms = {"GGGG", "GBGG", "GBGB", "GBBG", "GBBB"}

norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]

# === 規範割合の集計 ===
def compute_avg_norm_distribution(file_list):
    norm_counts = []
    for f in file_list:
        df = pd.read_csv(f)
        df = df[df["Generation"] > df["Generation"].max() - 50]
        agent_cols = [col for col in df.columns if col.startswith("Agent_")]
        for _, row in df[agent_cols].iterrows():
            counts = Counter(row)
            norm_counts.append(counts)
    total_counts = Counter()
    for c in norm_counts:
        total_counts.update(c)
    total = sum(total_counts.values())
    if total == 0:
        return None
    norm_ratios = {norm: total_counts.get(norm, 0) / total for norm in norm_patterns}
    return norm_ratios

# === モザイク描画 ===
def draw_mosaic(norm_ratios, key):
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 2))
    x_start = 0
    for norm in norm_patterns:
        width = norm_ratios.get(norm, 0)
        color = "white"
        if norm in target_norms:
            color_map = {
                "GGGG": "skyblue",
                "GBGG": "lightgreen",
                "GBGB": "salmon",
                "GBBG": "orange",
                "GBBB": "plum"
            }
            color = color_map.get(norm, "gray")
        rect = Rectangle((x_start, 0), width, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        x_start += width
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(key, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mosaic_{key}.png"))
    plt.close()

# === ファイルグループ化 ===
file_dict = defaultdict(list)
for dirpath, _, filenames in os.walk(base_dir):
    for file in filenames:
        if file.startswith("norm_distribution") and file.endswith(".csv"):
            if f"benefit{target_benefit}_" not in file:
                continue
            match = re.match(r"norm_distribution(.+)_\d+\.csv", file)
            if match:
                key = match.group(1)
                full_path = os.path.join(dirpath, file)
                file_dict[key].append(full_path)

# === 並列処理 ===
def process_group(args):
    key, files = args
    avg_ratios = compute_avg_norm_distribution(files)
    if avg_ratios:
        draw_mosaic(avg_ratios, key)
    return key

with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_group, file_dict.items()))

print(f"✅ 完了: {len(results)} 件のモザイクプロットを plot/ に保存しました。")
