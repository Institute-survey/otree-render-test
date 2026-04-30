#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


META_COLS = [
    "condition_id",
    "condition_code",
    "filename",
    "condition_type",
    "rho",
    "sigma_e",
    "base_error_prob",
    "error_sensitivity",
    "error_mode",
    "earned",
    "cost_weight_increase",
    "gratitude_steps",
    "group_id",
]


def get_round_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("round_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mean_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    round_cols = get_round_cols(df)
    group_cols = [
        "condition_id",
        "condition_code",
        "condition_type",
        "rho",
        "sigma_e",
        "base_error_prob",
        "error_sensitivity",
        "error_mode",
        "earned",
        "cost_weight_increase",
        "gratitude_steps",
    ]
    return df.groupby(group_cols, dropna=False)[round_cols].mean().reset_index()


def plot_all_conditions(coop_df: pd.DataFrame, out_path: Path) -> None:
    round_cols = get_round_cols(coop_df)
    rounds = np.arange(1, len(round_cols) + 1)
    cond_mean = mean_by_condition(coop_df)

    plt.figure(figsize=(14, 8))

    for _, row in cond_mean.iterrows():
        values = row[round_cols].to_numpy(dtype=float)
        label = row["condition_code"] if "condition_code" in row else row["condition_id"]
        alpha = 0.95 if row["condition_type"] == "baseline" else 0.10
        linewidth = 3.0 if row["condition_type"] == "baseline" else 0.8
        plt.plot(rounds, values, alpha=alpha, linewidth=linewidth, label=label if row["condition_type"] == "baseline" else None)

    plt.xlabel("Round")
    plt.ylabel("Cooperation rate")
    plt.title("Cooperation rate across all conditions")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_condition_pngs(coop_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    round_cols = get_round_cols(coop_df)
    rounds = np.arange(1, len(round_cols) + 1)

    for condition_id, sub in coop_df.groupby("condition_id", sort=False):
        code = str(sub["condition_code"].iloc[0]) if "condition_code" in sub.columns else str(condition_id)
        values = sub[round_cols].to_numpy(dtype=float)
        mean = values.mean(axis=0)
        sd = values.std(axis=0)
        se = sd / math.sqrt(values.shape[0])

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, mean, linewidth=2, label="Mean")
        plt.fill_between(rounds, mean - se, mean + se, alpha=0.25, label="± SE")
        plt.xlabel("Round")
        plt.ylabel("Cooperation rate")
        plt.title(f"Cooperation rate: {code}")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"{code}.png", dpi=300)
        plt.close()


def plot_endowment_overview(endowment_df: pd.DataFrame, out_path: Path) -> None:
    round_cols = get_round_cols(endowment_df)
    rounds = np.arange(1, len(round_cols) + 1)
    cond_mean = mean_by_condition(endowment_df)

    plt.figure(figsize=(14, 8))

    for _, row in cond_mean.iterrows():
        values = row[round_cols].to_numpy(dtype=float)
        alpha = 0.95 if row["condition_type"] == "baseline" else 0.12
        linewidth = 3.0 if row["condition_type"] == "baseline" else 0.8
        label = row["condition_code"] if row["condition_type"] == "baseline" else None
        plt.plot(rounds, values, alpha=alpha, linewidth=linewidth, label=label)

    plt.xlabel("Round")
    plt.ylabel("Endowment")
    plt.title("Endowment trajectories across all conditions")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_endowment_by_rho_sigma(endowment_df: pd.DataFrame, out_path: Path) -> None:
    round_cols = get_round_cols(endowment_df)
    rounds = np.arange(1, len(round_cols) + 1)

    experimental = endowment_df[endowment_df["condition_type"] != "baseline"].copy()
    group_cols = ["rho", "sigma_e"]
    mean_df = experimental.groupby(group_cols, dropna=False)[round_cols].mean().reset_index()

    plt.figure(figsize=(12, 7))

    for _, row in mean_df.iterrows():
        label = f"rho={row['rho']}, sigma={row['sigma_e']}"
        values = row[round_cols].to_numpy(dtype=float)
        plt.plot(rounds, values, linewidth=1.5, label=label)

    plt.axhline(100, linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Mean Endowment")
    plt.title("Mean Endowment by rho and sigma")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_endowment_volatility_summary(endowment_df: pd.DataFrame, out_path: Path) -> None:
    round_cols = get_round_cols(endowment_df)

    tmp = endowment_df.copy()
    values = tmp[round_cols].to_numpy(dtype=float)
    tmp["endowment_sd_over_rounds"] = values.std(axis=1)
    tmp["mean_abs_round_change"] = np.abs(np.diff(values, axis=1)).mean(axis=1)

    experimental = tmp[tmp["condition_type"] != "baseline"].copy()

    summary = (
        experimental
        .groupby(["rho", "sigma_e"], dropna=False)[["endowment_sd_over_rounds", "mean_abs_round_change"]]
        .mean()
        .reset_index()
    )

    labels = [f"r={r}, s={s}" for r, s in zip(summary["rho"], summary["sigma_e"])]
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(13, 7))
    plt.bar(x - width / 2, summary["endowment_sd_over_rounds"], width, label="SD over rounds")
    plt.bar(x + width / 2, summary["mean_abs_round_change"], width, label="Mean abs round change")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Endowment volatility")
    plt.title("Endowment volatility summary by rho and sigma")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PGG simulation outputs.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing all_conditions_group_means.csv and all_conditions_endowments.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save PNG files. Default: <input-dir>/figures",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    ensure_dir(output_dir)

    condition_plot_dir = output_dir / "condition_cooperation"
    ensure_dir(condition_plot_dir)

    coop_path = input_dir / "all_conditions_group_means.csv"
    endowment_path = input_dir / "all_conditions_endowments.csv"

    if not coop_path.exists():
        raise FileNotFoundError(f"Not found: {coop_path}")
    if not endowment_path.exists():
        raise FileNotFoundError(f"Not found: {endowment_path}")

    coop_df = pd.read_csv(coop_path)
    endowment_df = pd.read_csv(endowment_path)

    plot_all_conditions(coop_df, output_dir / "cooperation_all_conditions.png")
    plot_condition_pngs(coop_df, condition_plot_dir)
    plot_endowment_overview(endowment_df, output_dir / "endowment_all_conditions.png")
    plot_endowment_by_rho_sigma(endowment_df, output_dir / "endowment_by_rho_sigma.png")
    plot_endowment_volatility_summary(endowment_df, output_dir / "endowment_volatility_summary.png")

    print(f"Saved figures to: {output_dir}")
    print(f"Condition-specific cooperation figures: {condition_plot_dir}")


if __name__ == "__main__":
    main()
