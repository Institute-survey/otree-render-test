#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 固定設定
# =========================
INPUT_CSV = "all_conditions_group_means.csv"

# 全条件をまとめた図
OUTPUT_ALL_PNG = "condition_cooperation_rates_all.png"

# 条件ごとの個別図の保存先
OUTPUT_DIR_INDIVIDUAL = "condition_plots"

FIGSIZE_ALL = (14, 8)
FIGSIZE_SINGLE = (8, 5)
DPI = 300

LABEL_BY = "condition_id"   # "condition_id" または "filename"
SHOW_LEGEND_ALL = False     # 条件数が多いので通常は False 推奨


def find_round_columns(df: pd.DataFrame) -> List[str]:
    round_cols = [col for col in df.columns if col.startswith("round_")]
    if not round_cols:
        raise ValueError("round_ で始まる列が見つかりません。入力CSVを確認してください。")
    return sorted(round_cols, key=lambda x: int(x.split("_")[1]))


def validate_columns(df: pd.DataFrame, label_by: str) -> None:
    if label_by not in df.columns:
        raise ValueError(f"必要な列がありません: {label_by}")


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    sanitized = str(name)
    for ch in invalid_chars:
        sanitized = sanitized.replace(ch, "_")
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def load_grouped_means(input_csv: str, label_by: str) -> tuple[pd.DataFrame, List[str]]:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df, label_by)
    round_cols = find_round_columns(df)

    # 同じ条件の group_id を平均して、1条件=1本の折れ線にする
    grouped = df.groupby(label_by, sort=True)[round_cols].mean()
    return grouped, round_cols


def plot_all_conditions(grouped: pd.DataFrame, round_cols: List[str]) -> None:
    rounds = list(range(1, len(round_cols) + 1))

    plt.figure(figsize=FIGSIZE_ALL)

    for condition_label, row in grouped.iterrows():
        plt.plot(rounds, row.values, label=str(condition_label), linewidth=1.0)

    plt.xlabel("Round")
    plt.ylabel("Cooperation Rate")
    plt.title("Cooperation Rate by Condition")
    plt.xlim(1, len(round_cols))
    plt.grid(True, alpha=0.3)

    if SHOW_LEGEND_ALL:
        plt.legend(
            fontsize=6,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_ALL_PNG, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved combined plot to: {OUTPUT_ALL_PNG}")


def plot_individual_conditions(grouped: pd.DataFrame, round_cols: List[str], output_dir: str) -> None:
    rounds = list(range(1, len(round_cols) + 1))
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for condition_label, row in grouped.iterrows():
        plt.figure(figsize=FIGSIZE_SINGLE)
        plt.plot(rounds, row.values, linewidth=1.8)

        plt.xlabel("Round")
        plt.ylabel("Cooperation Rate")
        plt.title(str(condition_label))
        plt.xlim(1, len(round_cols))
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = sanitize_filename(str(condition_label)) + ".png"
        output_path = outdir / filename
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close()

    print(f"Saved individual condition plots to: {outdir}")


def main() -> None:
    grouped, round_cols = load_grouped_means(INPUT_CSV, LABEL_BY)

    plot_all_conditions(grouped, round_cols)
    plot_individual_conditions(grouped, round_cols, OUTPUT_DIR_INDIVIDUAL)

    print(f"Number of lines plotted in combined figure: {len(grouped)}")
    print(f"Number of individual plots saved: {len(grouped)}")


if __name__ == "__main__":
    main()