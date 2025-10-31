import os
import glob
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# === 設定 ===
base_dir = "."
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

# 規範の順番（指示どおり）
norm_patterns = [
    "GGGG", "GGGB", "GBGG", "GBGB",
    "GBBG", "GBBB", "GGBG", "GGBB",
    "BGGG", "BGGB", "BGBG", "BGBB",
    "BBGG", "BBGB", "BBBG", "BBBB"
]
# ハイライト規範（凡例対象）
color_norms = {"GGGG", "GGGB", "GBGG", "GBGB", "GBBG", "GBBB"}
public_norm_targets = {"GBBB", "GBBG", "GBGB", "GBGG"}

# カラー設定：color_norms は指定色、その他は淡色
norm_colors = {
    "GGGG": "#00429d",
    "GGGB": "#2e60b2",
    "GBGG": "#4671c6",
    "GBGB": "#7ea4d6",
    "GBBG": "#bcd5e6",
    "GBBB": "#e1edf3",
}
for n in norm_patterns:
    if n not in norm_colors:
        norm_colors[n] = "#f0f0f0"  # 淡いグレー

# ファイル名マッチ
norm_file_pat = re.compile(
    r"norm_distribution400_([A-Z]{4})_probability([0-9.]+)_.*_benefit5_(\d+)\.csv"
)
coop_file_pat = re.compile(
    r"cooperation_rates_400_([A-Z]{4})_probability([0-9.]+)_.*_benefit5\.csv"
)

# 収集用
# data[public_norm][prob][norm] = list of ratios (全pubprob*と全試行×最終50世代を蓄積)
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# coop_all[public_norm][prob] = list of 平均協力率（Sim全列・最終50世代の平均）を各ファイルごとに蓄積
coop_all = defaultdict(lambda: defaultdict(list))

# === データ読み込み（pubprob* を総なめ） ===
for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath) or not subdir.startswith("pubprob"):
        continue

    # 規範分布
    for csv_path in glob.glob(os.path.join(subpath, "norm_distribution*.csv")):
        filename = os.path.basename(csv_path)
        m = norm_file_pat.match(filename)
        if not m:
            continue
        public_norm, prob_str, sim_str = m.groups()
        if public_norm not in public_norm_targets:
            continue
        prob = float(prob_str)

        df = pd.read_csv(csv_path)
        if "Generation" not in df.columns:
            continue
        df = df[df["Generation"] >= 951]  # 最終50世代
        agent_cols = [c for c in df.columns if c.startswith("Agent_")]
        if not agent_cols or df.empty:
            continue

        # 各世代ごとにカウント→比率
        for _, row in df[agent_cols].iterrows():
            counts = Counter(row)
            total = len(row)
            for norm in norm_patterns:
                ratio = counts.get(norm, 0) / total
                data[public_norm][prob][norm].append(ratio)

    # 協力率
    for coop_path in glob.glob(os.path.join(subpath, "cooperation_rates_400_*.csv")):
        fname = os.path.basename(coop_path)
        mc = coop_file_pat.match(fname)
        if not mc:
            continue
        pn, prob_str = mc.group(1), mc.group(2)
        if pn not in public_norm_targets:
            continue
        prob = float(prob_str)
        try:
            cdf = pd.read_csv(coop_path)
            if "Generation" not in cdf.columns:
                continue
            cdf = cdf[cdf["Generation"] >= 951]
            sim_cols = [c for c in cdf.columns if c.startswith("Sim")]
            if not sim_cols or cdf.empty:
                continue
            # 各ファイル（＝ディレクトリ×条件）で: Sim全列×最終50世代の平均を1値として追加
            coop_all[pn][prob].append(cdf[sim_cols].mean().mean())
        except Exception:
            continue

# === 平均比率を用意（積み上げ面グラフ用） ===
# avg_data[public_norm][prob][norm] = 平均比率
avg_data = defaultdict(dict)
for public_norm in public_norm_targets:
    for prob, per_norm in data[public_norm].items():
        avg_data[public_norm][prob] = {
            n: (np.mean(per_norm[n]) if len(per_norm[n]) > 0 else 0.0)
            for n in norm_patterns
        }
    # 足し合わせて1にならない場合の微調整（安全策）
    for prob in avg_data[public_norm]:
        s = sum(avg_data[public_norm][prob].values())
        if s > 0:
            for n in norm_patterns:
                avg_data[public_norm][prob][n] /= s

# === 協力率の平均系列（赤破線用） ===
# coop_mean[public_norm][prob] = 全pubprob*統合の平均協力率（Sim全列×最終50世代の平均をさらに平均）
coop_mean = defaultdict(dict)
for public_norm in public_norm_targets:
    for prob, vals in coop_all[public_norm].items():
        coop_mean[public_norm][prob] = float(np.mean(vals)) if len(vals) else np.nan

# === プロット（public_norm ごと） ===
def plot_area(public_norm):
    # この public_norm でデータがある probability を昇順取得
    probs = sorted(avg_data[public_norm].keys())
    if not probs:
        return

    # stackplot 用に series を norm順に用意
    series = []
    colors = []
    for n in norm_patterns:
        series.append([avg_data[public_norm][p][n] for p in probs])
        colors.append(norm_colors[n])

    # 図作成
    fig, ax = plt.subplots(figsize=(max(8, len(probs)*0.45 + 2), 6))

    # 積み上げ面グラフ（ミルフィーユ）— 境界線は描かない
    ax.stackplot(probs, series, colors=colors, edgecolor='none')

    # 規範境界：GBGB と GBBG の間（= 上位4層の上面）を黒の太い破線で
    cum4 = np.zeros(len(probs))
    for k in range(4):  # 0:GGGG,1:GGGB,2:GBGG,3:GBGB
        cum4 += np.array(series[k])
    ax.plot(probs, cum4, linestyle='--', linewidth=2.5, color='black', zorder=10, label="_nolegend_")

    # 協力率の赤い破線（同じYスケール 0〜1）
    coop_y = [coop_mean[public_norm].get(p, np.nan) for p in probs]
    ax.plot(probs, coop_y, linestyle='--', linewidth=2.0, color='red', zorder=11, label='Cooperation rate')

    # 軸・範囲
    ax.set_xlim(min(probs), max(probs))
    ax.set_ylim(0, 1)

    ax.set_xlabel("Probability")
    ax.set_ylabel("Norm Ratio")
    ax.set_title(f"Stacked Area of Norms (Public Norm: {public_norm})")

    # 凡例：color_norms のみを表示 + 協力率の赤い破線
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=norm_colors[n], lw=8, label=n) for n in color_norms]
    legend_handles.append(Line2D([0], [0], color='red', lw=2, linestyle='--', label='Cooperation rate'))

    # 凡例は外側右上へ
    leg = ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.25, 1.0), title="Highlighted Norms")
    fig.subplots_adjust(right=0.80)  # 右に余白

    # 目盛・枠
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='y', which='both', pad=6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"area_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

# 並列実行
if __name__ == "__main__":
    with Pool(processes=min(len(public_norm_targets), cpu_count())) as pool:
        pool.map(plot_area, list(public_norm_targets))
