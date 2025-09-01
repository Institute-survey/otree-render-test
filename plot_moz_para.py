import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import glob
from multiprocessing import Pool

# --- 設定 --- #
base_dir = "."  # 実行場所
target_benefit = "5"
target_norms = ["GBBB", "GBBG", "GBGB", "GBGG"]
colored_norms = ["GGGG", "GBGG", "GBGB", "GBBG", "GBBB"]
norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
probability_values = [round(i * 0.05, 2) for i in range(21)]  # 0.00〜1.00

# --- norm_distributionのパターン --- #
norm_pattern = re.compile(r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit5_(\d+)\.csv")

# --- ディレクトリ下の全CSV探索 --- #
def collect_all_csv():
    all_csv = []
    for subdir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(full_path) or not subdir.startswith("pubprob_"):
            continue
        all_csv.extend(glob.glob(os.path.join(full_path, "norm_distribution*.csv")))
    return all_csv

# --- 規範割合を計算（最終50世代） --- #
def compute_average_ratios(df):
    latest_df = df.tail(50)
    norms_only = latest_df.drop(columns=["Generation"])
    counter = Counter(norm for row in norms_only.itertuples(index=False) for norm in row)
    total = sum(counter.values())
    return {k: counter.get(k, 0) / total for k in norm_patterns}

# --- 各public_normに対して処理 --- #
def process_public_norm(public_norm):
    print(f"[INFO] Processing: {public_norm}")
    files = collect_all_csv()
    records = []

    for file in files:
        fname = os.path.basename(file)
        match = norm_pattern.match(fname)
        if not match:
            continue

        norm_label, prob, sim_id = match.groups()
        if norm_label != public_norm:
            continue
        prob = round(float(prob), 2)

        try:
            df = pd.read_csv(file)
            if "Generation" not in df.columns:
                continue
            avg_ratios = compute_average_ratios(df)
            avg_ratios["probability"] = prob
            records.append(avg_ratios)
        except Exception as e:
            print(f"[ERROR] Failed to read {file}: {e}")
            continue

    if not records:
        print(f"[WARNING] No valid data for {public_norm}")
        return

    # --- DataFrameにまとめて整形 --- #
    df_all = pd.DataFrame(records)
    df_all = df_all.groupby("probability").mean().reset_index()
    df_all = df_all.sort_values(by="probability")

    # --- モザイクプロット作成 --- #
    fig, ax = plt.subplots(figsize=(10, 6))

    bottoms = [0] * len(df_all)
    for norm in norm_patterns:
        heights = df_all[norm].values
        color = "black" if norm not in colored_norms else None  # white以外にしたいなら個別指定可
        barcolor = "white" if norm not in colored_norms else None
        ax.barh(df_all["probability"], heights, left=bottoms, label=norm if norm in colored_norms else "",
                color=barcolor, edgecolor="black", linewidth=0.5)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Probability")
    ax.set_title(f"Norm Distribution (Benefit=5)\nPublic Norm: {public_norm}")
    ax.invert_yaxis()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    output_name = f"mosaic_{public_norm}.png"
    plt.savefig(output_name)
    plt.close()
    print(f"[Saved] {output_name}")

# --- 並列実行 --- #
if __name__ == "__main__":
    with Pool(processes=4) as pool:
        pool.map(process_public_norm, target_norms)
