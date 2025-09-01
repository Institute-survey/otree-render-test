import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from glob import glob
from collections import defaultdict

# === 設定 ===
base_dir = "."  # pubprob_ディレクトリがある場所
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

benefits = [round(x * 0.5, 1) for x in range(2, 11)]  # 1.0 ~ 5.0
probabilities = [round(x * 0.05, 2) for x in range(21)]  # 0.00 ~ 1.00
public_norms = ["GBBB", "GBBG", "GBGB", "GBGG"]

# === パルスカウント関数 ===
def count_pulses(series):
    high_threshold = 0.9
    low_threshold = 0.5
    above = series > high_threshold
    below = series < low_threshold
    pulse_count = 0
    state = 0  # 0: neutral, 1: above, 2: waiting for below

    for val in series:
        if state == 0 and val > high_threshold:
            state = 1
        elif state == 1 and val < low_threshold:
            pulse_count += 1
            state = 0
        elif val < high_threshold:
            state = 0
    return pulse_count

# === データ格納 ===
# pulses[public_norm][benefit][probability] = 合計パルス回数（全Simの平均）
pulses = defaultdict(lambda: defaultdict(dict))

# === 正規表現 ===
pattern = re.compile(
    r"cooperation_rates_400_([A-Z]{4})_probability([0-9.]+)_action_error[0-9.]+_evaluate_error[0-9.]+_public_error[0-9.]+_benefit([0-9.]+)\.csv"
)

# === 各ファイルを読み込んでパルス数を数える ===
for subdir in os.listdir(base_dir):
    full_path = os.path.join(base_dir, subdir)
    if not os.path.isdir(full_path) or not subdir.startswith("pubprob_"):
        continue

    for file in glob(os.path.join(full_path, "cooperation_rates_400_*.csv")):
        match = pattern.search(file)
        if not match:
            continue
        norm, prob, benefit = match.groups()
        prob = round(float(prob), 2)
        benefit = round(float(benefit), 1)

        if norm not in public_norms:
            continue

        df = pd.read_csv(file)
        if "Generation" not in df.columns:
            continue

        df = df[df["Generation"] >= 501]

        sim_cols = [col for col in df.columns if col.startswith("Sim")]
        pulse_list = []
        for col in sim_cols:
            pulse_list.append(count_pulses(df[col]))

        avg_pulse = np.mean(pulse_list)
        pulses[norm][benefit][prob] = avg_pulse

# === ヒートマップ作成関数 ===
def plot_pulse_heatmap(norm):
    grid = np.full((len(benefits), len(probabilities)), np.nan)

    for i, b in enumerate(benefits):
        for j, p in enumerate(probabilities):
            val = pulses[norm].get(b, {}).get(p, np.nan)
            grid[i, j] = val

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        grid,
        cmap="Reds",
        xticklabels=probabilities,
        yticklabels=benefits,
        linewidths=0.5,
        linecolor='gray',
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Pulse Count"},
        ax=ax
    )
    ax.set_xlabel("Probability")
    ax.set_ylabel("Benefit")
    ax.set_title(f"Pulse Count Heatmap – Public Norm: {norm}")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"pulse_heatmap_{norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

# === 描画 ===
for norm in public_norms:
    plot_pulse_heatmap(norm)
