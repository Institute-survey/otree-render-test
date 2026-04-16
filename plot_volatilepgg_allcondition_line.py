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
OUTPUT_PNG = "condition_cooperation_rates.png"
FIGSIZE = (14, 8)
DPI = 300
LABEL_BY = "condition_id"   # "condition_id" または "filename"
SHOW_LEGEND = False         # 条件数が多いので通常は False 推奨


def find_round_columns(df: pd.DataFrame) -> List[str]:
    round_cols = [col for col in df.columns if col.startswith("round_")]
    if not round_cols:
        raise ValueError("round_ で始まる列が見つかりません。入力CSVを確認してください。")
    return sorted(round_cols, key=lambda x: int(x.split("_")[1]))


def validate_columns(df: pd.DataFrame, label_by: str) -> None:
    if label_by not in df.columns:
        raise ValueError(f"必要な列がありません: {label_by}")


def main() -> None:
    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_PNG)

    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df, LABEL_BY)
    round_cols = find_round_columns(df)

    # 同じ条件の group_id を平均して、1条件=1本の折れ線にする
    grouped = df.groupby(LABEL_BY, sort=True)[round_cols].mean()

    rounds = list(range(1, len(round_cols) + 1))

    plt.figure(figsize=FIGSIZE)

    for condition_label, row in grouped.iterrows():
        plt.plot(rounds, row.values, label=str(condition_label), linewidth=1.2)

    plt.xlabel("Round")
    plt.ylabel("Cooperation Rate")
    plt.title("Cooperation Rate by Condition")
    plt.xlim(1, len(round_cols))
    plt.grid(True, alpha=0.3)

    if SHOW_LEGEND:
        plt.legend(
            fontsize=6,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {output_path}")
    print(f"Number of lines plotted: {len(grouped)}")


if __name__ == "__main__":
    main()