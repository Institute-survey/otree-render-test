import pandas as pd
import numpy as np
import os
import re
from glob import glob
import traceback
from collections import defaultdict

# 設定
base_dir = "."
output_dir = "."

norms_target = ["GBBB", "GBBG", "GBGB", "GBGG"]
benefits = [round(b, 1) for b in np.arange(1, 5.1, 0.5)]
probs = [round(p, 2) for p in np.arange(0, 1.01, 0.05)]

# キーを安定化（"0.05" のような文字列キーにする）
def key_prob(p_float) -> str:
    return f"{float(p_float):.2f}"

def key_benefit(b_float) -> str:
    return f"{float(b_float):.1f}"

pattern = re.compile(
    r"cooperation_rates_400_([A-Z]{4})_probability([0-9.]+)_action_error0\.001_evaluate_error0\.001_public_error0\.001_benefit([0-9.]+)\.csv"
)

# data_vals[norm][benefit_str][prob_str] = [avg_trial1, avg_trial2, ...]
data_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# 探索・集計（julia* ディレクトリをすべて対象）
for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("julia"):
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

            prob_str = key_prob(prob)
            benefit_str = key_benefit(benefit)

            df = pd.read_csv(file)
            df.columns = [c.strip() for c in df.columns]

            # 後半100世代平均（Sim0〜Sim9）
            df_last = df[(df["Generation"] >= 902) & (df["Generation"] <= 1001)]
            sim_cols = [col for col in df_last.columns if col.startswith("Sim")]
            mean_val = df_last[sim_cols].mean(axis=0).mean()

            data_vals[norm][benefit_str][prob_str].append(float(mean_val))

        except Exception as e:
            print(f"[ERROR] Skipping file {file}: {e}")
            traceback.print_exc()

# 試行間平均: data_mean[norm][benefit_str][prob_str] = mean
data_mean = defaultdict(lambda: defaultdict(dict))
for norm, by_benefit in data_vals.items():
    for benefit_str, by_prob in by_benefit.items():
        for prob_str, vals in by_prob.items():
            data_mean[norm][benefit_str][prob_str] = float(np.mean(vals)) if vals else np.nan

# ---- 左端列（prob=0.00）の統合（全 norm 横断平均） ----
combined_prob0 = {}  # combined_prob0[benefit_str] = 統合平均
for b in benefits:
    bkey = key_benefit(b)
    vals = []
    for norm in norms_target:
        v = data_mean[norm].get(bkey, {}).get("0.00", np.nan)
        if not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    combined_prob0[bkey] = float(np.mean(vals)) if vals else np.nan

# デバッグ表示（任意）
print("[INFO] Combined prob=0.00 column (benefit -> mean across norms):")
for b in benefits:
    bkey = key_benefit(b)
    print(f"  benefit={bkey}: {combined_prob0[bkey]}")

def export_cell_values(norm):
    """
    行=benefit（大→小）、列=prob（小→大）
    左端列(prob=0.00)のみ、norm横断の統合平均(combined_prob0)を使用。
    """
    grid = np.full((len(benefits), len(probs)), np.nan)

    for bi, b in enumerate(benefits):
        bkey = key_benefit(b)
        for pi, p in enumerate(probs):
            pkey = key_prob(p)

            if pi == 0:
                val = combined_prob0.get(bkey, np.nan)  # 左端だけ統合値
            else:
                val = data_mean[norm].get(bkey, {}).get(pkey, np.nan)

            grid[len(benefits) - 1 - bi, pi] = val  # 上ほど benefit が大

    df_out = pd.DataFrame(
        grid,
        index=[f"{b:.1f}" for b in reversed(benefits)],
        columns=[f"{p:.2f}" for p in probs],
    )
    df_out.index.name = "Benefit"
    df_out.columns.name = "Probability"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cooperation_cells_{norm}_prob0combined.csv")
    df_out.to_csv(out_path, float_format="%.6f")
    print(f"[Saved] {out_path}")

# 実行
for norm in norms_target:
    export_cell_values(norm)
