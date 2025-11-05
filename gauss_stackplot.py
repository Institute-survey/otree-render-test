import os
import glob
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ============ 設定 ============
base_dir = "."
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)

# 規範の順番（指定どおり）
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

# 平滑化パラメータ（必要に応じて調整）
UPSAMPLE_FACTOR = 5   # 補間で点を何倍に増やすか（大きいほど滑らか）
GAUSS_SIGMA     = 2.0 # ガウシアン平滑の標準偏差（補間後のポイント単位）

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

# ============ ユーティリティ（平滑化） ============
def gaussian_kernel(sigma: float):
    """ガウシアンカーネル（1次元）を生成。半径は ~3σ で切り落とし。"""
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x**2) / (2 * sigma * sigma))
    k /= k.sum()
    return k

def smooth_1d(y: np.ndarray, sigma: float):
    """1次元配列にガウシアン平滑を適用（端は反射）。"""
    if sigma <= 0:
        return y.copy()
    k = gaussian_kernel(sigma)
    return np.convolve(np.pad(y, (len(k)//2, len(k)//2), mode='reflect'), k, mode='valid')

def make_fine_grid(x: np.ndarray, upsample: int):
    """x（昇順）を等間隔の細かいグリッドに拡張。端点は一致させる。"""
    if len(x) < 2 or upsample <= 1:
        return x.copy()
    n_fine = (len(x) - 1) * upsample + 1
    return np.linspace(x[0], x[-1], n_fine)

def interp_and_smooth(x, y, x_fine, sigma):
    """(x,y) を x_fine に線形補間し、ガウシアン平滑して返す。"""
    y_interp = np.interp(x_fine, x, y)
    y_smooth = smooth_1d(y_interp, sigma)
    return y_smooth

# ============ データ読み込み（pubprob* を総なめ） ============
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

# ============ 平均比率（積み上げ面グラフ用） ============
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

# ============ 協力率の平均系列 ============
# coop_mean[public_norm][prob] = 全pubprob*統合の平均協力率（Sim全列×最終50世代の平均をさらに平均）
coop_mean = defaultdict(dict)
for public_norm in public_norm_targets:
    for prob, vals in coop_all[public_norm].items():
        coop_mean[public_norm][prob] = float(np.mean(vals)) if len(vals) else np.nan

# ============ プロット（public_norm ごと） ============
def plot_area_smoothed(public_norm):
    # この public_norm でデータがある probability を昇順取得
    probs = sorted(avg_data[public_norm].keys())
    if not probs:
        return

    x = np.array(probs, dtype=float)
    x_fine = make_fine_grid(x, UPSAMPLE_FACTOR)

    # stackplot 用に series を norm順に用意し、補間＋平滑
    series_smooth = []
    colors = []
    for n in norm_patterns:
        y = np.array([avg_data[public_norm][p][n] for p in probs], dtype=float)
        y_s = interp_and_smooth(x, y, x_fine, GAUSS_SIGMA)
        series_smooth.append(y_s)
        colors.append(norm_colors[n])
    series_smooth = np.vstack(series_smooth)  # shape: (num_norms, len(x_fine))

    # 列方向に正規化（各確率で合計=1）
    col_sum = series_smooth.sum(axis=0)
    nonzero = col_sum > 0
    series_smooth[:, nonzero] /= col_sum[nonzero]

    # 規範境界：上位4層(0..3)の累積を算出し、あとで黒破線で重ねる
    cum4 = np.cumsum(series_smooth[:4, :], axis=0)[-1, :]  # shape: (len(x_fine),)

    # 協力率（赤破線）も補間＋平滑
    coop_y = np.array([coop_mean[public_norm].get(p, np.nan) for p in probs], dtype=float)
    # 欠損がある場合は線を切らさないように最近傍補間
    # まず欠損位置を埋める
    if np.isnan(coop_y).all():
        coop_s = np.full_like(x_fine, np.nan)
    else:
        # 線形補間の前に NaN を補完（最近傍）
        mask = ~np.isnan(coop_y)
        coop_y_filled = np.interp(x, x[mask], coop_y[mask])
        coop_s = interp_and_smooth(x, coop_y_filled, x_fine, GAUSS_SIGMA)

    # 図作成
    fig, ax = plt.subplots(figsize=(max(8, len(x_fine)*0.08), 6))

    # 積み上げ面グラフ（ミルフィーユ）— 境界線は描かない
    ax.stackplot(x_fine, series_smooth, colors=colors, edgecolor='none')

    # 規範境界（GBGBとGBBGの間）：黒の太い破線
    ax.plot(x_fine, cum4, linestyle='--', linewidth=2.5, color='black', zorder=10, label="_nolegend_")

    # 協力率の赤い破線（同じYスケール 0〜1）
    if not np.isnan(coop_s).all():
        ax.plot(x_fine, coop_s, linestyle='--', linewidth=2.0, color='red', zorder=11, label='Cooperation rate')

    # 軸・範囲
    ax.set_xlim(x_fine.min(), x_fine.max())
    ax.set_ylim(0, 1)

    ax.set_xlabel("Probability")
    ax.set_ylabel("Norm Ratio")
    ax.set_title(f"Smoothed Stacked Area of Norms (Public Norm: {public_norm})")

    # 凡例：color_norms のみ表示 + 協力率の赤破線
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
    save_path = os.path.join(output_dir, f"area_smoothed_{public_norm}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

# 並列実行
if __name__ == "__main__":
    with Pool(processes=min(len(public_norm_targets), cpu_count())) as pool:
        pool.map(plot_area_smoothed, list(public_norm_targets))
