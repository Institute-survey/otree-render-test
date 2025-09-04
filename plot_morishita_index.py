import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter, defaultdict

# === 設定 ===
base_dir = "."  # pubprob_**** ディレクトリがある場所
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

# === 出力用データ ===
similarity_data = defaultdict(lambda: defaultdict(dict))  # [public_norm][benefit][prob] = C_Δ
coop_check = defaultdict(lambda: defaultdict(list))        # 協力率チェック用

# === ファイル名抽出正規表現 ===
file_pattern = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)_(\d+)\.csv"
)

# === 分布計算関数 ===
def get_distribution_vector(values):
    counts = Counter(values)
    counts = np.array([counts.get(norm, 0) + 1 for norm in norm_patterns])  # +1 to each count
    return counts / counts.sum()

# === 基準分布収集（prob = 0） ===
baseline_distributions = defaultdict(dict)  # [public_norm][benefit] = ベクトル

for subdir in os.listdir(base_dir):
    path = os.path.join(base_dir, subdir)
    if not os.path.isdir(path) or not subdir.startswith("pubprob"):
        continue

    for file in glob.glob(os.path.join(path, "norm_distribution400_*_probability0.*.csv")):
        match = file_pattern.match(os.path.basename(file))
        if not match:
            continue

        public_norm, prob, benefit, _ = match.groups()
        prob = float(prob)
        benefit = float(benefit)
        if public_norm not in public_norm_targets:
            continue

        df = pd.read_csv(file)
        df = df[df["Generation"] >= 951]
        agent_cols = [c for c in df.columns if c.startswith("Agent_")]
        values = df[agent_cols].values.flatten()
        vec = get_distribution_vector(values)

        # 最初の1個のみ採用（平均でなく代表）
        key = (public_norm, benefit)
        if key not in baseline_distributions:
            baseline_distributions[key] = vec

# === 他のprobとの比較 ===
for subdir in os.listdir(base_dir):
    path = os.path.join(base_dir, subdir)
    if not os.path.isdir(path) or not subdir.startswith("pubprob"):
        continue

    for file in glob.glob(os.path.join(path, "norm_distribution*.csv")):
        match = file_pattern.match(os.path.basename(file))
        if not match:
            continue

        public_norm, prob_str, benefit_str, sim_str = match.groups()
        prob = float(prob_str)
        benefit = float(benefit_str)
        sim = int(sim_str)

        if public_norm not in public_norm_targets or prob == 0.0:
            continue

        df = pd.read_csv(file)
        df = df[df["Generation"] >= 951]

        # 類似度の基準分布がなければスキップ
        base_key = (public_norm, benefit)
        if base_key not in baseline_distributions:
            continue
        p_vec = baseline_distributions[base_key]

        agent_cols = [col for col in df.columns if col.startswith("Agent_")]
        values = df[agent_cols].values.flatten()
        q_vec = get_distribution_vector(values)

        # --- Morishita's Similarity Index (C_Δ)
        c_delta = 2 * np.sum(np.minimum(p_vec, q_vec)) / (np.sum(p_vec) + np.sum(q_vec))
        similarity_data[public_norm][benefit].setdefault(prob, []).append(c_delta)

# === 協力率チェック（Sim0〜Sim9 すべてが平均80%以上か）
coop_pattern = re.compile(
    r"cooperation_rates_400_([A-Z]{4})_probability([0-9.]+)_.*_benefit([0-9.]+)\.csv"
)

for subdir in os.listdir(base_dir):
    path = os.path.join(base_dir, subdir)
    if not os.path.isdir(path) or not subdir.startswith("pubprob"):
        continue

    for file in glob.glob(os.path.join(path, "cooperation_rates*.csv")):
        match = coop_pattern.match(os.path.basename(file))
        if not match:
            continue
        public_norm, prob_str, benefit_str = match.groups()
        prob = float(prob_str)
        benefit = float(benefit_str)

        if public_norm not in public_norm_targets or prob == 0.0:
            continue

        df = pd.read_csv(file)
        if "Generation" not in df.columns:
            continue
        df = df[df["Generation"] >= 951]
        sim_cols = [col for col in df.columns if col.startswith("Sim")]
        means = df[sim_cols].mean()
        pass_check = all(means >= 0.8)
        coop_check[public_norm][benefit].append((prob, pass_check))

# === 描画 ===
for public_norm in sorted(public_norm_targets):
    heatmap_vals = np.full((len(benefits), len(probabilities)), np.nan)

    for i, benefit in enumerate(benefits):
        for j, prob in enumerate(probabilities):
            if prob == 0.0:
                continue
            sim_vals = similarity_data[public_norm].get(benefit, {}).get(prob, [])
            if len(sim_vals) == 0:
                continue
            avg_similarity = np.mean(sim_vals)

            # 協力率チェック
            coop_passed = False
            for p, flag in coop_check[public_norm].get(benefit, []):
                if abs(p - prob) < 1e-5 and flag:
                    coop_passed = True
                    break
            heatmap_vals[len(benefits) - 1 - i, j] = avg_similarity if coop_passed else 0  # 下が benefit=1

    plt.figure(figsize=(len(probabilities)*0.5+2, len(benefits)*0.5+2))
    cmap = sns.light_palette("green", as_cmap=True)
    sns.heatmap(
        heatmap_vals,
        xticklabels=[f"{p:.2f}" for p in probabilities],
        yticklabels=[f"{b:.1f}" for b in reversed(benefits)],
        cmap=cmap,
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"label": "Morishita CΔ Similarity"}
    )
    plt.xlabel("Probability")
    plt.ylabel("Benefit")
    plt.title(f"Morishita Similarity (to prob=0) – Public Norm: {public_norm}")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"morishita_similarity_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")
