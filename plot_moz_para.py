import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from multiprocessing import Pool
import glob
import re

# === 設定 ===
base_dir = "."  # pubprob_yyyymmdd フォルダがある場所
plot_dir = "./plot"
os.makedirs(plot_dir, exist_ok=True)

target_norms = ["GGGG", "GBGG", "GBGB", "GBBG", "GBBB"]
norm_order = target_norms + ["OTHER"]
norm_colors = {
    "GGGG": "#1f77b4",
    "GBGG": "#ff7f0e",
    "GBGB": "#2ca02c",
    "GBBG": "#d62728",
    "GBBB": "#9467bd",
    "OTHER": "#cccccc"
}

benefit_filter = "5"
generation_tail = 50  # 最終50世代

# === ファイル探索とグルーピング ===
def group_files_by_condition():
    condition_groups = defaultdict(list)
    for subdir in os.listdir(base_dir):
        if not subdir.startswith("pubprob_"):
            continue
        folder = os.path.join(base_dir, subdir)
        files = glob.glob(os.path.join(folder, "norm_distribution*.csv"))
        for f in files:
            basename = os.path.basename(f)
            match = re.match(r"norm_distribution(\d+_[A-Z]{4}_probability[0-9.]+_action_error[0-9.]+_evaluate_error[0-9.]+_public_error[0-9.]+_benefit5)_\d+\.csv", basename)
            if match:
                key = match.group(1)
                condition_groups[key].append(f)
    return condition_groups

# === 規範カウント（最終50世代平均）===
def compute_avg_norm_ratio(file_list):
    all_counts = []
    for file in file_list:
        df = pd.read_csv(file)
        df = df.tail(generation_tail)
        agent_data = df.drop(columns=["Generation"], errors="ignore")
        for _, row in agent_data.iterrows():
            counts = Counter(row)
            all_counts.append(counts)
    # 合算
    total_counter = sum(all_counts, Counter())
    total = sum(total_counter.values())
    if total == 0:
        return {norm: 0 for norm in norm_order}
    result = {}
    for norm in target_norms:
        result[norm] = total_counter.get(norm, 0) / total
    # 残り全部まとめてOTHER
    result["OTHER"] = 1 - sum(result.values())
    return result

# === public_norm 単位でのプロット処理 ===
def plot_mosaic_for_public_norm(public_norm):
    grouped = group_files_by_condition()
    filtered = {k: v for k, v in grouped.items() if f"_{public_norm}_" in k and k.endswith(f"_benefit{benefit_filter}")}

    # probabilityごとに並べる
    data_by_prob = {}
    for key, files in filtered.items():
        match = re.search(r"probability([0-9.]+)", key)
        if not match:
            continue
        prob = float(match.group(1))
        avg_ratios = compute_avg_norm_ratio(files)
        data_by_prob[prob] = avg_ratios

    # ソート
    sorted_probs = sorted(data_by_prob.keys())
    values_matrix = []
    for norm in norm_order:
        row = [data_by_prob[prob].get(norm, 0) for prob in sorted_probs]
        values_matrix.append(row)

    # 描画
    fig, ax = plt.subplots(figsize=(max(10, len(sorted_probs) * 0.6), 6))
    bottoms = [0] * len(sorted_probs)
    for norm, values in zip(norm_order, values_matrix):
        ax.bar(sorted_probs, values, bottom=bottoms, label=norm, color=norm_colors[norm])
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xlabel("Probability")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_title(f"Mosaic Plot (Benefit=5) – Public Norm: {public_norm}")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    output_path = os.path.join(plot_dir, f"cooperation_mosaic_{public_norm}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[Saved] {output_path}")

# === 並列実行 ===
if __name__ == "__main__":
    public_norms = ["GBGG", "GBGB", "GBBG", "GBBB"]
    with Pool() as pool:
        pool.map(plot_mosaic_for_public_norm, public_norms)
