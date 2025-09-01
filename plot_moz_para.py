import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool
from matplotlib.patches import Rectangle

# --- 設定 --- #
base_dir = "."  # データフォルダの親
target_benefit = 5
public_norms = ["GBBB", "GBBG", "GBGB", "GBGG"]
probabilities = [round(p, 2) for p in np.arange(0, 1.01, 0.05)]
highlight_norms = ["GGGG", "GBGG", "GBGB", "GBBG", "GBBB"]
norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
colors = {
    "GGGG": "#1f77b4", "GBGG": "#2ca02c", "GBGB": "#ff7f0e",
    "GBBG": "#d62728", "GBBB": "#9467bd"
}
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

# --- 条件抽出ヘルパー --- #
def extract_condition(filename):
    parts = filename.replace(".csv", "").split("_")
    if len(parts) < 8:
        return None
    norm = parts[1]
    prob = float(parts[2].replace("probability", ""))
    benefit = int(parts[-1].replace("benefit", ""))
    return norm, prob, benefit

# --- 対象ファイルの収集と正確な割り当て --- #
def collect_files_by_norm():
    all_dirs = [d for d in sorted(os.listdir(base_dir)) if d.startswith("pubprob_")]
    grouped = {pn: [] for pn in public_norms}

    for d in all_dirs:
        full_path = os.path.join(base_dir, d)
        files = glob.glob(os.path.join(full_path, "norm_distribution*.csv"))
        for f in sorted(files):
            cond = extract_condition(os.path.basename(f))
            if cond is None:
                continue
            norm, prob, benefit = cond
            if norm in grouped and benefit == target_benefit:
                grouped[norm].append(f)

    return grouped

# --- モザイクプロット作成（1 public_norm ごと） --- #
def process_public_norm(norm_and_files):
    norm, files = norm_and_files
    print(f"[INFO] Processing: {norm}")
    aggregated = {}

    for file in files:
        df = pd.read_csv(file)
        if "Generation" not in df.columns:
            continue
        # GenerationとAgent列に限定
        agent_cols = ["Generation"] + [col for col in df.columns if col.startswith("Agent_")]
        df = df[agent_cols]

        # 最終50世代だけ抽出
        df_tail = df[df["Generation"] >= df["Generation"].max() - 49]

        # 行ごとにカウント
        for _, row in df_tail.iterrows():
            prob = extract_condition(os.path.basename(file))[1]
            row = row.drop("Generation")
            total = len(row)
            counts = Counter(row)
            ratios = {n: counts.get(n, 0) / total for n in norm_patterns}
            if prob not in aggregated:
                aggregated[prob] = []
            aggregated[prob].append(ratios)

    # --- 平均を取る --- #
    average_ratios = {}
    for prob in probabilities:
        entries = aggregated.get(prob, [])
        if entries:
            df = pd.DataFrame(entries)
            avg = df.mean().to_dict()
        else:
            avg = {n: 0 for n in norm_patterns}
        average_ratios[prob] = avg

    # --- プロット作成 --- #
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, prob in enumerate(probabilities):
        x_start = 0
        for norm in norm_patterns:
            width = average_ratios[prob].get(norm, 0)
            height = 1 / len(probabilities)
            y = i * height
            rect_color = colors[norm] if norm in colors else "white"
            edge_color = "black"
            ax.add_patch(Rectangle(
                (x_start, y), width, height,
                facecolor=rect_color,
                edgecolor=edge_color
            ))
            x_start += width

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Norm Proportion")
    ax.set_ylabel("Probability")
    ax.set_yticks([(i + 0.5) / len(probabilities) for i in range(len(probabilities))])
    ax.set_yticklabels([str(p) for p in probabilities])
    ax.set_title(f"Mosaic Plot – Public Norm: {norm}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"mosaic_plot_{norm}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")

# --- 並列実行 --- #
if __name__ == "__main__":
    grouped_files = collect_files_by_norm()
    tasks = [(norm, grouped_files[norm]) for norm in public_norms]

    with Pool(processes=4) as pool:
        pool.map(process_public_norm, tasks)
