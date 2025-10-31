import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from glob import glob
import traceback
from collections import defaultdict

# 設定
base_dir = "."
output_dir = "."
dpi = 300
draw_grid = True

norms_target = ["GBBB", "GBBG", "GBGB", "GBGG"]
benefits = [round(b, 1) for b in np.arange(1, 5.1, 0.5)]
probs = [round(p, 2) for p in np.arange(0, 1.01, 0.05)]

pattern = re.compile(
    r"cooperation_rates_400_([A-Z]{4})_probability([0-9.]+)_action_error0\.001_evaluate_error0\.001_public_error0\.001_benefit([0-9.]+)\.csv"
)

# データ格納: data[norm][benefit][prob] = avg
data = defaultdict(lambda: defaultdict(dict))

# 探索・集計
for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("pubprob_"):
        continue

    files = glob(os.path.join(subpath, "cooperation_rates_400_*.csv"))
    for file in files:
        try:
            filename = os.path.basename(file)
            match = pattern.match(filename)
            if not match:
                continue

            norm, prob, benefit = match.groups()
            if norm not in norms_target:
                continue

            prob = float(prob)
            benefit = float(benefit)

            df = pd.read_csv(file)
            df.columns = [c.strip() for c in df.columns]

            # 後半100世代平均（Sim0〜Sim9）
            df_last = df[(df["Generation"] >= 902) & (df["Generation"] <= 1001)]
            sim_cols = [col for col in df_last.columns if col.startswith("Sim")]
            mean_val = df_last[sim_cols].mean(axis=0).mean()

            data[norm][benefit][prob] = float(mean_val)

        except Exception as e:
            print(f"[ERROR] Skipping file {file}: {e}")
            traceback.print_exc()

# ---- ここが追加：prob=0.0 列の統合（全 norm 横断の平均） ----
combined_prob0 = {}  # combined_prob0[benefit] = 統合平均
for benefit in benefits:
    vals = []
    for norm in norms_target:
        v = data[norm].get(benefit, {}).get(0.0, np.nan)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) > 0:
        combined_prob0[benefit] = float(np.mean(vals))
    else:
        combined_prob0[benefit] = np.nan  # 該当データが全く無い場合は NaN のまま

# デバッグ表示（任意）
print("[INFO] Combined prob=0.0 column (benefit -> mean across norms):")
for b in benefits:
    print(f"  benefit={b}: {combined_prob0[b]}")

# ヒートマップ描画
def plot_heatmap(norm):
    grid = np.full((len(benefits), len(probs)), np.nan)

    for bi, benefit in enumerate(benefits):
        for pi, prob in enumerate(probs):
            if prob == 0.0:
                # 左端列だけ統合値を使用
                val = combined_prob0.get(benefit, np.nan)
            else:
                val = data[norm].get(benefit, {}).get(prob, np.nan)
            # y 軸は上が大きな benefit になるよう反転格納
            grid[len(benefits) - 1 - bi, pi] = val

    plt.figure(figsize=(16, 6), dpi=dpi)
    sns.heatmap(
        grid,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=probs,
        yticklabels=list(reversed(benefits)),
        linewidths=0.5 if draw_grid else 0,
        linecolor='gray' if draw_grid else None
    )
    plt.title(f"Average Cooperation Rate (Norm: {norm})\n(Leftmost column uses prob=0.0 combined across norms)", fontsize=14)
    plt.xlabel("Probability of Public Evaluation Use")
    plt.ylabel("Benefit")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cooperation_heatmap_{norm}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[Saved] {output_path}")

# 実行
for norm in norms_target:
    plot_heatmap(norm)
