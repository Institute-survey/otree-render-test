import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import glob
import re
from multiprocessing import Pool

# === 設定 ===
BASE_DIR = "."
PLOT_DIR = os.path.join(BASE_DIR, "plot")
os.makedirs(PLOT_DIR, exist_ok=True)

NORM_PATTERNS = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
COLORED = ["GGGG", "GBGG", "GBGB", "GBBG", "GBBB"]
NORM_ORDER = list(reversed(COLORED + [n for n in NORM_PATTERNS if n not in COLORED]))

# === グループ化処理 ===
def find_simulation_groups():
    pattern = re.compile(r"norm_distribution400_([A-Z]{4})_probability([0-9.]+).*benefit5_(\d+)\.csv")
    groups = defaultdict(list)
    for subdir in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(path) or not subdir.startswith("pubprob_"):
            continue
        for file in glob.glob(os.path.join(path, "norm_distribution400_*_benefit5_*.csv")):
            fname = os.path.basename(file)
            m = pattern.match(fname)
            if m:
                public_norm, prob, _ = m.groups()
                key = (public_norm, float(prob))
                groups[key].append(file)
    return groups

# === 平均規範分布の計算 ===
def compute_average_norm_distribution(file_list):
    all_counts = []
    for file in file_list:
        try:
            df = pd.read_csv(file)
            df = df[df["Generation"] >= df["Generation"].max() - 49]
            norm_cols = [col for col in df.columns if col.startswith("Agent_")]
            counts = []
            for _, row in df[norm_cols].iterrows():
                counter = Counter(row)
                counts.append(counter)
            avg = Counter()
            for c in counts:
                avg.update(c)
            for key in avg:
                avg[key] /= len(counts)
            all_counts.append(avg)
        except Exception as e:
            print(f"[ERROR] Skipped: {file} → {e}")
    final = Counter()
    for c in all_counts:
        final.update(c)
    for key in final:
        final[key] /= len(all_counts)
    return final

# === モザイクプロット描画 ===
def draw_mosaic(public_norm, prob_to_dist):
    probs = sorted(prob_to_dist.keys())
    data = []
    for norm in NORM_ORDER:
        row = [prob_to_dist[prob].get(norm, 0) for prob in probs]
        data.append(row)
    df = pd.DataFrame(data, index=NORM_ORDER, columns=probs)

    fig, ax = plt.subplots(figsize=(1.5 * len(probs), 8))
    bottom = [0] * len(probs)

    for norm in NORM_ORDER:
        values = df.loc[norm].values
        color = "gray"
        if norm in COLORED:
            if norm == "GGGG": color = "#1f77b4"
            elif norm == "GBGG": color = "#2ca02c"
            elif norm == "GBGB": color = "#ff7f0e"
            elif norm == "GBBG": color = "#d62728"
            elif norm == "GBBB": color = "#9467bd"
        ax.bar(probs, values, bottom=bottom, label=norm, color=color, edgecolor='black')
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("probability")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Norm Distribution Summary (public_norm={public_norm})")
    ax.set_ylim(0, 1)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"mosaic_{public_norm}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")

# === public_normごとの処理 ===
def process_public_norm(public_norm):
    all_groups = find_simulation_groups()
    subset = {k[1]: v for k, v in all_groups.items() if k[0] == public_norm}
    if not subset:
        print(f"[Skip] No files found for {public_norm}")
        return
    prob_to_dist = {}
    for prob, files in subset.items():
        avg = compute_average_norm_distribution(files)
        prob_to_dist[prob] = avg
    draw_mosaic(public_norm, prob_to_dist)

# === 並列実行 ===
if __name__ == "__main__":
    target_norms = ["GBBB", "GBBG", "GBGB", "GBGG"]
    with Pool(processes=len(target_norms)) as pool:
        pool.map(process_public_norm, target_norms)
