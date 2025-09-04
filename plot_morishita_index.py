# morishita_heatmap_filtered.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter, defaultdict

base_dir = "."
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

norm_patterns = [
    "GGGG", "GGGB", "GGBG", "GGBB",
    "GBGG", "GBGB", "GBBG", "GBBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}

benefits = [round(x, 1) for x in np.arange(1.0, 5.5, 0.5)]
probabilities = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]

file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)_(\d+)\.csv"
)

# === 協力率 80%以上の判定用 ===
cooperation_rate = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("pubprob"):
        continue

    for coop_file in glob.glob(os.path.join(subpath, "cooperation_rates_*.csv")):
        match = re.match(
            r"cooperation_rates_400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)\.csv",
            os.path.basename(coop_file)
        )
        if not match:
            continue
        norm, prob, benefit = match.groups()
        prob = float(prob)
        benefit = float(benefit)
        if norm not in public_norm_targets:
            continue

        df = pd.read_csv(coop_file)
        if "Generation" not in df.columns:
            continue

        df = df[df["Generation"] >= 951]
        for col in [c for c in df.columns if c.startswith("Sim")]:
            cooperation_rate[norm][benefit][prob].append(df[col].mean())

# === probability=0 のデータ格納 ===
baseline_counts = defaultdict(dict)  # [norm][benefit] = Counter

for filepath in glob.glob(f"{base_dir}/**/norm_distribution400_*_probability0.0_*.csv", recursive=True):
    match = file_pattern.match(os.path.basename(filepath))
    if not match:
        continue
    norm, prob_str, benefit_str, _ = match.groups()
    benefit = float(benefit_str)
    if norm not in public_norm_targets:
        continue

    df = pd.read_csv(filepath)
    df = df[df["Generation"] >= 951]
    agent_cols = [c for c in df.columns if c.startswith("Agent_")]
    norm_vals = df[agent_cols].values.flatten()
    counts = Counter(norm_vals)
    baseline_counts[norm][benefit] = counts

# === Morishita Index 計算と描画 ===
for public_norm in sorted(public_norm_targets):
    heatmap_data = np.full((len(benefits), len(probabilities)), np.nan)

    for filepath in glob.glob(f"{base_dir}/**/norm_distribution400_{public_norm}_*.csv", recursive=True):
        match = file_pattern.match(os.path.basename(filepath))
        if not match:
            continue
        _, prob_str, benefit_str, _ = match.groups()
        prob = float(prob_str)
        benefit = float(benefit_str)

        if prob == 0.0 or benefit not in benefits or prob not in probabilities:
            continue

        coop_vals = cooperation_rate[public_norm][benefit][prob]
        if not coop_vals or np.mean(coop_vals) < 0.8:
            morishita_index = 0
        else:
            df = pd.read_csv(filepath)
            df = df[df["Generation"] >= 951]
            agent_cols = [c for c in df.columns if c.startswith("Agent_")]
            norm_vals = df[agent_cols].values.flatten()
            counts = Counter(norm_vals)

            # 比較対象: probability=0.0 のnorm分布
            base_counts = baseline_counts[public_norm].get(benefit, Counter())
            all_counts = {norm: (counts.get(norm, 0), base_counts.get(norm, 0))
                          for norm in norm_patterns}

            X = np.array([a + 1 for a, _ in all_counts.values()])
            Y = np.array([b + 1 for _, b in all_counts.values()])
            N = np.sum(X)
            M = np.sum(Y)

            morishita_index = np.sum((X / N) * (Y / M))

        i = len(benefits) - 1 - benefits.index(benefit)
        j = probabilities.index(prob)
        heatmap_data[i, j] = morishita_index

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=[f"{p:.2f}" for p in probabilities],
        yticklabels=[f"{b:.1f}" for b in reversed(benefits)],
        cmap="Greens",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"label": "Morishita Index (Cλ)"}
    )
    plt.xlabel("Probability")
    plt.ylabel("Benefit")
    plt.title(f"Morishita Index (Cλ) – Public Norm: {public_norm}")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"morishita_heatmap_filtered_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")
