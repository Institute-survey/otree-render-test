import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# --- 設定 --- #
base_dir = "."
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)
target_benefit = "5"
norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
highlight_norms = {
    "GGGG": "skyblue",
    "GBGG": "lightgreen",
    "GBGB": "salmon",
    "GBBG": "orange",
    "GBBB": "plum"
}

# --- ファイル名の正規表現 --- #
file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([\d.]+)_action_error[\d.]+_evaluate_error[\d.]+_public_error[\d.]+_benefit5_(\d+)\.csv"
)

# --- 規範割合を計算（最終50世代の平均） --- #
def extract_average_ratios(filepath):
    df = pd.read_csv(filepath)
    agent_cols = [col for col in df.columns if col.startswith("Agent_")]
    df = df[["Generation"] + agent_cols]
    df = df.tail(50)

    all_counts = Counter()
    for _, row in df.iterrows():
        counts = Counter(row[1:])
        all_counts.update(counts)

    total = sum(all_counts.values())
    if total == 0:
        return {norm: 0 for norm in norm_patterns}
    return {norm: all_counts.get(norm, 0) / total for norm in norm_patterns}

# --- ファイル探索とグループ化 --- #
grouped_data = defaultdict(lambda: defaultdict(list))

for root, _, files in os.walk(base_dir):
    for filename in files:
        if not filename.startswith("norm_distribution400_") or "benefit5" not in filename:
            continue

        match = file_pattern.match(filename)
        if not match:
            continue

        public_norm, prob, _ = match.groups()
        prob = float(prob)
        filepath = os.path.join(root, filename)
        grouped_data[public_norm][prob].append(filepath)

# --- public_normごとにプロット作成 --- #
for pubnorm, prob_dict in grouped_data.items():
    sorted_probs = sorted(prob_dict.keys())
    fig, axes = plt.subplots(nrows=len(norm_patterns), ncols=len(sorted_probs),
                             figsize=(len(sorted_probs) * 1.5, len(norm_patterns) * 0.5 + 2),
                             sharex=True, sharey=True)

    if len(sorted_probs) == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, prob in enumerate(sorted_probs):
        file_list = prob_dict[prob]
        df_list = [extract_average_ratios(f) for f in file_list]
        avg_df = pd.DataFrame(df_list).mean().to_dict()

        total_width = 1.0
        cum_width = 0.0
        for row_idx, norm in enumerate(norm_patterns):
            width = avg_df.get(norm, 0)
            color = highlight_norms.get(norm, "lightgrey")
            ax = axes[row_idx][col_idx]
            ax.barh(0, width, left=cum_width, height=1.0, color=color, edgecolor='black')
            cum_width += width

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(norm, rotation=0, labelpad=30, fontsize=8, va='center')

        axes[0][col_idx].set_title(f"p={prob:.2f}", fontsize=9)

    plt.suptitle(f"Mosaic Plot – Public Norm: {pubnorm}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_dir, f"mosaic_summary_{pubnorm}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[Saved] {output_path}")
