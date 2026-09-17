# make_analysis_data.py
# -*- coding: utf-8 -*-

"""
Create analysis-friendly CSV files from tolerant norm simulation outputs.

Place this script directly under the simulation output directory, e.g.:

    outputs_tolerant_norms_L50_b0to5_step0p5_error01/
        make_analysis_data.py
        summary_by_run.csv
        summary_aggregated.csv
        best_conditions.csv
        model_parameters.json
        all_generations_timeseries_by_worker/
            worker_001/
            worker_002/
            ...

Run:

    python make_analysis_data.py

Main outputs:

    analysis_data/summary_by_run_compact.csv
    analysis_data/condition_summary_compact.csv
    analysis_data/heatmap_ready_last20.csv
    analysis_data/best_conditions_compact.csv
    analysis_data/final_generation_by_run.csv
    analysis_data/timeseries_checkpoints_by_run.csv
    analysis_data/timeseries_compact_parts/*.csv

Notes:
- The script reads large time-series CSVs in chunks.
- It does not concatenate the original worker CSVs.
- It creates narrower, easier-to-analyze CSVs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ============================================================
# User settings
# ============================================================

# Number of rows read at once from time-series CSVs.
# If memory is tight, reduce this to 100_000 or 200_000.
CHUNK_SIZE = 500_000

# If True, create compact time-series part files with only key columns and major metrics.
# This still keeps every generation and every run, so it can be very large.
CREATE_TIMESERIES_COMPACT_PARTS = True

# If True, create:
#   final_generation_by_run.csv
#   timeseries_checkpoints_by_run.csv
PROCESS_TIMESERIES_FOR_FINAL_AND_CHECKPOINTS = True

# Checkpoint generations for easier trajectory plotting.
# Generations larger than the model's GENERATIONS value are ignored automatically.
CHECKPOINT_GENERATIONS = [
    1, 2, 5, 10, 20, 50,
    100, 150, 200, 250,
    300, 350, 400, 450, 500,
    750, 1000
]

# Rows per compact part file.
COMPACT_PART_ROWS_PER_FILE = 5_000_000

# Columns kept from the full time series.
# These are usually enough for dynamics, strategy frequencies, and heatmap-style summaries.
TIMESERIES_KEEP_COLUMNS = [
    "norm_type",
    "norm_id",
    "L",
    "theta",
    "theta_ratio",
    "benefit",
    "cost",
    "b_over_c",
    "error_condition",
    "action_error",
    "assessment_error",
    "run_id",
    "generation",
    "cooperation_rate",
    "frequency_ALLC",
    "frequency_ALLD",
    "frequency_DISC",
    "mean_payoff",
    "fraction_good_reputation",
    "good_reputation_ALLC",
    "good_reputation_ALLD",
    "good_reputation_DISC",
]

# Columns kept from summary_by_run.csv.
SUMMARY_BY_RUN_KEEP_COLUMNS = [
    "norm_type",
    "norm_id",
    "L",
    "theta",
    "theta_ratio",
    "benefit",
    "cost",
    "b_over_c",
    "error_condition",
    "action_error",
    "assessment_error",
    "run_id",
    "last20_start_generation",
    "last20_end_generation",
    "last20_cooperation_rate",
    "last20_frequency_ALLC",
    "last20_frequency_ALLD",
    "last20_frequency_DISC",
    "last20_mean_payoff",
    "last20_fraction_good_reputation",
    "last20_good_reputation_ALLC",
    "last20_good_reputation_ALLD",
    "last20_good_reputation_DISC",
    "last10_cooperation_rate",
    "last50_cooperation_rate",
    "final_generation_cooperation_rate",
]

# Columns kept from summary_aggregated.csv.
SUMMARY_AGG_KEEP_COLUMNS = [
    "norm_type",
    "norm_id",
    "L",
    "theta",
    "theta_ratio",
    "benefit",
    "cost",
    "b_over_c",
    "error_condition",
    "action_error",
    "assessment_error",
    "n_runs",
    "mean_last20_cooperation_rate",
    "sd_last20_cooperation_rate",
    "se_last20_cooperation_rate",
    "mean_last20_frequency_ALLC",
    "mean_last20_frequency_ALLD",
    "mean_last20_frequency_DISC",
    "mean_last20_mean_payoff",
    "mean_last20_fraction_good_reputation",
    "mean_last20_good_reputation_ALLC",
    "mean_last20_good_reputation_ALLD",
    "mean_last20_good_reputation_DISC",
    "mean_last10_cooperation_rate",
    "mean_last50_cooperation_rate",
    "mean_final_generation_cooperation_rate",
]


# ============================================================
# Path helpers
# ============================================================

def get_root_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


ROOT_DIR = get_root_dir()
ANALYSIS_DIR = ROOT_DIR / "analysis_data"
COMPACT_TS_DIR = ANALYSIS_DIR / "timeseries_compact_parts"

SUMMARY_BY_RUN_FILE = ROOT_DIR / "summary_by_run.csv"
SUMMARY_AGG_FILE = ROOT_DIR / "summary_aggregated.csv"
BEST_CONDITIONS_FILE = ROOT_DIR / "best_conditions.csv"
MODEL_PARAMETERS_FILE = ROOT_DIR / "model_parameters.json"
TIMESERIES_WORKER_DIR = ROOT_DIR / "all_generations_timeseries_by_worker"


# ============================================================
# Utility functions
# ============================================================

def ensure_dirs() -> None:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    COMPACT_TS_DIR.mkdir(exist_ok=True)


def read_model_generations() -> Optional[int]:
    if not MODEL_PARAMETERS_FILE.exists():
        return None

    try:
        with MODEL_PARAMETERS_FILE.open("r", encoding="utf-8") as f:
            params = json.load(f)
        generations = params.get("generations", None)
        if generations is None:
            return None
        return int(generations)
    except Exception as e:
        print(f"[WARN] Could not read generations from model_parameters.json: {e}")
        return None


def read_csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def existing_columns(path: Path, desired_columns: list[str]) -> list[str]:
    header = read_csv_header(path)
    return [c for c in desired_columns if c in header]


def add_analysis_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add convenient labels for plotting and filtering.
    """
    if "error_condition" in df.columns:
        df["error_label"] = df["error_condition"].map({
            0: "error_absent",
            1: "error_present",
            "0": "error_absent",
            "1": "error_present",
        }).fillna(df["error_condition"].astype(str))

    if "L" in df.columns and "theta" in df.columns:
        df["L_theta_label"] = (
            "L" + df["L"].astype(str).str.zfill(2)
            + "_theta" + df["theta"].astype(str).str.zfill(2)
        )

    if "theta_ratio" in df.columns:
        df["theta_ratio_rounded"] = pd.to_numeric(
            df["theta_ratio"],
            errors="coerce"
        ).round(4)

    if "benefit" in df.columns:
        df["benefit_label"] = (
            "b" + pd.to_numeric(df["benefit"], errors="coerce")
            .round(3)
            .astype(str)
        )

    if "b_over_c" in df.columns:
        df["b_over_c_label"] = (
            "bc" + pd.to_numeric(df["b_over_c"], errors="coerce")
            .round(3)
            .astype(str)
        )

    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {path.relative_to(ROOT_DIR)}  rows={len(df):,}")


def find_timeseries_files() -> list[Path]:
    if not TIMESERIES_WORKER_DIR.exists():
        return []

    files = sorted(TIMESERIES_WORKER_DIR.glob("worker_*/all_generations_timeseries_worker_*_part_*.csv"))

    if not files:
        files = sorted(TIMESERIES_WORKER_DIR.glob("**/*.csv"))

    return files


def compact_part_path(part_index: int) -> Path:
    return COMPACT_TS_DIR / f"timeseries_compact_part_{part_index:06d}.csv"


# ============================================================
# Summary processing
# ============================================================

def create_summary_by_run_compact() -> None:
    if not SUMMARY_BY_RUN_FILE.exists():
        print("[WARN] summary_by_run.csv not found. Skipping.")
        return

    usecols = existing_columns(SUMMARY_BY_RUN_FILE, SUMMARY_BY_RUN_KEEP_COLUMNS)

    if not usecols:
        print("[WARN] No matching columns found in summary_by_run.csv. Skipping.")
        return

    df = pd.read_csv(SUMMARY_BY_RUN_FILE, usecols=usecols)
    df = add_analysis_labels(df)

    # More analysis-friendly aliases.
    rename_map = {
        "last20_cooperation_rate": "coop_last20",
        "last20_frequency_ALLC": "ALLC_last20",
        "last20_frequency_ALLD": "ALLD_last20",
        "last20_frequency_DISC": "DISC_last20",
        "last20_mean_payoff": "payoff_last20",
        "last20_fraction_good_reputation": "good_rep_last20",
        "last10_cooperation_rate": "coop_last10",
        "last50_cooperation_rate": "coop_last50",
        "final_generation_cooperation_rate": "coop_final_generation",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    out = ANALYSIS_DIR / "summary_by_run_compact.csv"
    write_csv(df, out)


def create_condition_summary_compact() -> None:
    if not SUMMARY_AGG_FILE.exists():
        print("[WARN] summary_aggregated.csv not found. Skipping.")
        return

    usecols = existing_columns(SUMMARY_AGG_FILE, SUMMARY_AGG_KEEP_COLUMNS)

    if not usecols:
        print("[WARN] No matching columns found in summary_aggregated.csv. Skipping.")
        return

    df = pd.read_csv(SUMMARY_AGG_FILE, usecols=usecols)
    df = add_analysis_labels(df)

    rename_map = {
        "mean_last20_cooperation_rate": "coop_mean_last20",
        "sd_last20_cooperation_rate": "coop_sd_last20",
        "se_last20_cooperation_rate": "coop_se_last20",
        "mean_last20_frequency_ALLC": "ALLC_mean_last20",
        "mean_last20_frequency_ALLD": "ALLD_mean_last20",
        "mean_last20_frequency_DISC": "DISC_mean_last20",
        "mean_last20_mean_payoff": "payoff_mean_last20",
        "mean_last20_fraction_good_reputation": "good_rep_mean_last20",
        "mean_last10_cooperation_rate": "coop_mean_last10",
        "mean_last50_cooperation_rate": "coop_mean_last50",
        "mean_final_generation_cooperation_rate": "coop_mean_final_generation",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    out = ANALYSIS_DIR / "condition_summary_compact.csv"
    write_csv(df, out)


def create_heatmap_ready_last20() -> None:
    """
    Create a compact condition-level file for heatmaps.

    Typical use:
        x-axis: theta_ratio or theta
        y-axis: b_over_c
        facet: norm_type × error_condition
        value: coop_mean_last20
    """
    condition_summary = ANALYSIS_DIR / "condition_summary_compact.csv"

    if not condition_summary.exists():
        print("[WARN] condition_summary_compact.csv not found. Creating it first.")
        create_condition_summary_compact()

    if not condition_summary.exists():
        print("[WARN] Cannot create heatmap_ready_last20.csv.")
        return

    df = pd.read_csv(condition_summary)

    keep = [
        "norm_type",
        "norm_id",
        "error_condition",
        "error_label",
        "benefit",
        "cost",
        "b_over_c",
        "L",
        "theta",
        "theta_ratio",
        "theta_ratio_rounded",
        "coop_mean_last20",
        "coop_sd_last20",
        "coop_se_last20",
        "ALLC_mean_last20",
        "ALLD_mean_last20",
        "DISC_mean_last20",
        "payoff_mean_last20",
        "good_rep_mean_last20",
        "n_runs",
    ]

    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Stable sorting helps plotting scripts.
    sort_cols = [
        c for c in [
            "norm_type",
            "error_condition",
            "benefit",
            "L",
            "theta"
        ]
        if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols)

    out = ANALYSIS_DIR / "heatmap_ready_last20.csv"
    write_csv(df, out)


def create_best_conditions_compact() -> None:
    if not BEST_CONDITIONS_FILE.exists():
        print("[WARN] best_conditions.csv not found. Skipping.")
        return

    df = pd.read_csv(BEST_CONDITIONS_FILE)
    df = add_analysis_labels(df)

    rename_map = {
        "mean_last20_cooperation_rate": "coop_mean_last20",
        "sd_last20_cooperation_rate": "coop_sd_last20",
        "se_last20_cooperation_rate": "coop_se_last20",
        "mean_last20_frequency_ALLC": "ALLC_mean_last20",
        "mean_last20_frequency_ALLD": "ALLD_mean_last20",
        "mean_last20_frequency_DISC": "DISC_mean_last20",
        "mean_final_generation_cooperation_rate": "coop_mean_final_generation",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    out = ANALYSIS_DIR / "best_conditions_compact.csv"
    write_csv(df, out)


# ============================================================
# Time-series processing
# ============================================================

def process_timeseries_files() -> None:
    files = find_timeseries_files()

    if not files:
        print("[WARN] No time-series files found. Skipping time-series processing.")
        return

    generations = read_model_generations()
    if generations is None:
        print("[WARN] Could not determine final generation from model_parameters.json.")
        print("[WARN] final_generation_by_run.csv will be skipped.")
    else:
        print(f"[INFO] Final generation detected: {generations}")

    checkpoints = set(CHECKPOINT_GENERATIONS)
    if generations is not None:
        checkpoints = {g for g in checkpoints if g <= generations}

    final_out = ANALYSIS_DIR / "final_generation_by_run.csv"
    checkpoint_out = ANALYSIS_DIR / "timeseries_checkpoints_by_run.csv"

    # Reset outputs if they already exist.
    if final_out.exists():
        final_out.unlink()
    if checkpoint_out.exists():
        checkpoint_out.unlink()

    for old in COMPACT_TS_DIR.glob("timeseries_compact_part_*.csv"):
        old.unlink()

    compact_part_index = 1
    compact_rows_in_part = 0
    compact_header_written = False

    final_header_written = False
    checkpoint_header_written = False

    total_input_rows = 0
    total_compact_rows = 0
    total_final_rows = 0
    total_checkpoint_rows = 0

    print(f"[INFO] Time-series files found: {len(files):,}")

    for file_index, path in enumerate(files, start=1):
        print(f"[INFO] Reading {file_index:,}/{len(files):,}: {path.relative_to(ROOT_DIR)}")

        try:
            header = read_csv_header(path)
        except Exception as e:
            print(f"[WARN] Could not read header: {path}  error={e}")
            continue

        usecols = [c for c in TIMESERIES_KEEP_COLUMNS if c in header]

        if not usecols:
            print(f"[WARN] No matching time-series columns in {path}. Skipping.")
            continue

        try:
            reader = pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE)
        except Exception as e:
            print(f"[WARN] Could not open {path}: {e}")
            continue

        for chunk_index, chunk in enumerate(reader, start=1):
            total_input_rows += len(chunk)

            chunk = add_analysis_labels(chunk)

            # ------------------------------------------------
            # 1. Compact time-series parts
            # ------------------------------------------------
            if CREATE_TIMESERIES_COMPACT_PARTS:
                start = 0
                n = len(chunk)

                while start < n:
                    remaining_capacity = COMPACT_PART_ROWS_PER_FILE - compact_rows_in_part
                    end = min(start + remaining_capacity, n)

                    sub = chunk.iloc[start:end]
                    out_path = compact_part_path(compact_part_index)

                    sub.to_csv(
                        out_path,
                        mode="a",
                        index=False,
                        header=not compact_header_written or compact_rows_in_part == 0,
                        encoding="utf-8-sig"
                    )

                    rows_written = len(sub)
                    compact_rows_in_part += rows_written
                    total_compact_rows += rows_written
                    start = end

                    if compact_rows_in_part >= COMPACT_PART_ROWS_PER_FILE:
                        compact_part_index += 1
                        compact_rows_in_part = 0
                        compact_header_written = True

                compact_header_written = True

            # ------------------------------------------------
            # 2. Final generation by run
            # ------------------------------------------------
            if PROCESS_TIMESERIES_FOR_FINAL_AND_CHECKPOINTS and generations is not None:
                if "generation" in chunk.columns:
                    final_df = chunk[chunk["generation"] == generations].copy()

                    if not final_df.empty:
                        final_df.to_csv(
                            final_out,
                            mode="a",
                            index=False,
                            header=not final_header_written,
                            encoding="utf-8-sig"
                        )
                        final_header_written = True
                        total_final_rows += len(final_df)

            # ------------------------------------------------
            # 3. Checkpoint generations by run
            # ------------------------------------------------
            if PROCESS_TIMESERIES_FOR_FINAL_AND_CHECKPOINTS and checkpoints:
                if "generation" in chunk.columns:
                    checkpoint_df = chunk[chunk["generation"].isin(checkpoints)].copy()

                    if not checkpoint_df.empty:
                        checkpoint_df.to_csv(
                            checkpoint_out,
                            mode="a",
                            index=False,
                            header=not checkpoint_header_written,
                            encoding="utf-8-sig"
                        )
                        checkpoint_header_written = True
                        total_checkpoint_rows += len(checkpoint_df)

            if chunk_index % 20 == 0:
                print(
                    f"    processed chunks={chunk_index:,}, "
                    f"total_input_rows={total_input_rows:,}"
                )

    print("[OK] Time-series processing finished.")
    print(f"     input rows processed: {total_input_rows:,}")

    if CREATE_TIMESERIES_COMPACT_PARTS:
        print(f"     compact rows written: {total_compact_rows:,}")
        print(f"     compact directory: {COMPACT_TS_DIR.relative_to(ROOT_DIR)}")

    if PROCESS_TIMESERIES_FOR_FINAL_AND_CHECKPOINTS:
        print(f"     final rows written: {total_final_rows:,}")
        print(f"     checkpoint rows written: {total_checkpoint_rows:,}")


# ============================================================
# Optional: create a small README for analysis outputs
# ============================================================

def write_readme() -> None:
    path = ANALYSIS_DIR / "README_analysis_data.txt"

    text = f"""Analysis data generated from simulation outputs.

Root directory:
{ROOT_DIR}

Files:

1. summary_by_run_compact.csv
   Run-level compact summary.
   Useful for checking run-to-run variation.

2. condition_summary_compact.csv
   Condition-level aggregated summary.
   One row per condition after averaging across runs.

3. heatmap_ready_last20.csv
   Compact condition-level file for heatmaps.
   Suggested mapping:
       x = theta_ratio or theta
       y = b_over_c
       facet = norm_type × error_condition
       value = coop_mean_last20

4. best_conditions_compact.csv
   Best L-theta condition for each norm × benefit × error_condition group,
   based on mean_last20_cooperation_rate.

5. final_generation_by_run.csv
   Final generation state for each run, extracted from time series.
   Useful for final strategy frequencies.

6. timeseries_checkpoints_by_run.csv
   Selected generations only.
   Useful for trajectory plots without loading the full time series.

7. timeseries_compact_parts/
   Compact full time series.
   Same number of rows as the original time series, but fewer columns.
   Still very large when the full simulation is large.

Important columns:

- norm_type:
    image_scoring, shunning, stern_judging, simple_standing

- error_condition:
    0 = no action or assessment error
    1 = both action and assessment error are present

- b_over_c:
    benefit / cost

- L, theta, theta_ratio:
    Tolerance parameters

- coop_mean_last20:
    Mean cooperation rate in the last 20% of generations,
    where each generation's cooperation_rate was calculated from the latter 20%
    of interactions within that generation.
"""

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Saved: {path.relative_to(ROOT_DIR)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("[INFO] Root directory:", ROOT_DIR)

    ensure_dirs()

    create_summary_by_run_compact()
    create_condition_summary_compact()
    create_heatmap_ready_last20()
    create_best_conditions_compact()

    if CREATE_TIMESERIES_COMPACT_PARTS or PROCESS_TIMESERIES_FOR_FINAL_AND_CHECKPOINTS:
        process_timeseries_files()

    write_readme()

    print("[DONE] Analysis data creation completed.")
    print("[DONE] Output directory:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()