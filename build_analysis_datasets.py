#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build analysis-ready datasets from the raw CSV files produced by pubprob_test(4).jl.

Place this script in the same directory as:
  cooperation_rates_*.csv
  norm_distribution*.csv

Outputs six partitioned Parquet datasets under ./analysis_data/: 
  1) run_summary/
  2) condition_summary/
  3) condition_timeseries/
  4) run_trajectory_downsampled/
  5) event_catalog/
  6) event_aligned_profiles/

The script is resumable. Each parameter condition is written to its own atomic
Parquet part. Existing complete parts are skipped unless --overwrite is used.

Required packages:
  numpy, pandas, pyarrow

Example:
  python build_analysis_datasets.py
  python build_analysis_datasets.py --workers 12 --bin-width 10
  python build_analysis_datasets.py --overwrite

Notes on simulation IDs:
  cooperation CSV column Sim0 corresponds to norm_distribution..._1.csv,
  Sim1 corresponds to ..._2.csv, and so on.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "pyarrow is required for fast, compact Parquet output. Install it with:\n"
        "  python -m pip install pyarrow\n"
    ) from exc


SCRIPT_VERSION = "1.0.1"
COOP_PREFIX = "cooperation_rates_"
NORM_PREFIX = "norm_distribution"

NORM_LABELS = ["".join(x) for x in itertools.product("GB", repeat=4)]
NORM_TO_INDEX = {label: i for i, label in enumerate(NORM_LABELS)}
NORM_G_BITS = np.asarray(
    [[1.0 if char == "G" else 0.0 for char in label] for label in NORM_LABELS],
    dtype=np.float32,
)
LOG_N_NORMS = math.log(len(NORM_LABELS))

FLOAT_TOKEN = r"[-+0-9.eE]+"
CONDITION_RE = re.compile(
    rf"^(?P<num_agents>\d+)_"
    rf"(?P<public_norm>[GB]{{4}})_"
    rf"probability(?P<probability>{FLOAT_TOKEN})_"
    rf"action_error(?P<action_error>{FLOAT_TOKEN})_"
    rf"evaluate_error(?P<evaluation_error>{FLOAT_TOKEN})_"
    rf"public_error(?P<public_error>{FLOAT_TOKEN})_"
    rf"benefit(?P<benefit>{FLOAT_TOKEN})$"
)
SIM_COLUMN_RE = re.compile(r"^Sim(?P<sim>\d+)$")

DATASET_NAMES = (
    "run_summary",
    "condition_summary",
    "condition_timeseries",
    "run_trajectory_downsampled",
    "event_catalog",
    "event_aligned_profiles",
)

EVENT_CATALOG_COLUMNS = [
    "condition_key", "num_agents", "public_norm", "probability",
    "action_error", "evaluation_error", "public_error", "benefit",
    "simulation_id", "norm_file_simulation_id", "trajectory_type",
    "event_type", "event_index", "event_end_index", "event_duration",
    "event_order", "event_norm", "from_norm", "to_norm",
    "event_generation", "event_end_generation",
    "cooperation_pre_mean", "cooperation_post_mean", "cooperation_change",
    "entropy_pre_mean", "entropy_post_mean", "entropy_change",
    "dominance_pre_mean", "dominance_post_mean", "dominance_change",
    "alignment_pre_mean", "alignment_post_mean", "alignment_change",
    "q1_pre_mean", "q1_post_mean", "q1_change",
    "q2_pre_mean", "q2_post_mean", "q2_change",
    "q3_pre_mean", "q3_post_mean", "q3_change",
    "q4_pre_mean", "q4_post_mean", "q4_change",
    "event_norm_pre_frequency", "event_norm_post_frequency",
    "event_norm_frequency_change", "invasion_peak_generation",
    "invasion_pre_frequency", "invasion_peak_frequency",
    "invasion_success", "recovery_within_success_window",
]

EVENT_PROFILE_COLUMNS = [
    "condition_key", "num_agents", "public_norm", "probability",
    "action_error", "evaluation_error", "public_error", "benefit",
    "event_type", "event_norm", "trajectory_group",
    "relative_generation", "n_events_total", "n_events_available",
    "cooperation_mean", "cooperation_sd", "cooperation_q25",
    "cooperation_q75", "entropy_mean", "dominance_mean",
    "alignment_mean", "q1_mean", "q2_mean", "q3_mean", "q4_mean",
    *[f"norm_mean_{label}" for label in NORM_LABELS],
]


@dataclass(frozen=True)
class Config:
    final_fraction: float = 0.20
    low_threshold: float = 0.10
    high_threshold: float = 0.80
    min_state_duration: int = 20
    burnin_generations: int = 20
    delayed_takeoff_generation: int = 100
    bin_width: int = 10
    event_window: int = 100
    event_summary_window: int = 20
    dominant_min_frequency: float = 0.25
    dominant_min_duration: int = 10
    invasion_baseline_threshold: float = 0.025
    invasion_threshold: float = 0.10
    invasion_min_duration: int = 10
    invasion_pre_window: int = 20
    invasion_success_threshold: float = 0.25
    invasion_success_window: int = 100
    parquet_compression: str = "zstd"
    include_norm_invasion_profiles: bool = True
    min_valid_norm_fraction: float = 0.95


@dataclass(frozen=True)
class ConditionMeta:
    condition_key: str
    num_agents: int
    public_norm: str
    probability: float
    action_error: float
    evaluation_error: float
    public_error: float
    benefit: float

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------
# Generic utilities
# -----------------------------

def parse_args() -> argparse.Namespace:
    cpu = os.cpu_count() or 1
    default_workers = max(1, min(8, cpu // 2 if cpu > 2 else 1))

    parser = argparse.ArgumentParser(
        description="Build six analysis-ready Parquet datasets from simulation CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the raw CSV files. Default: script directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input-dir>/analysis_data.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel condition workers. Default: {default_workers}.",
    )
    parser.add_argument("--bin-width", type=int, default=10)
    parser.add_argument("--event-window", type=int, default=100)
    parser.add_argument("--low-threshold", type=float, default=0.10)
    parser.add_argument("--high-threshold", type=float, default=0.80)
    parser.add_argument("--min-state-duration", type=int, default=20)
    parser.add_argument("--burnin-generations", type=int, default=20)
    parser.add_argument("--final-fraction", type=float, default=0.20)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N cooperation files; useful for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute conditions even when all six output parts already exist.",
    )
    parser.add_argument(
        "--no-invasion-profiles",
        action="store_true",
        help="Do not create event-aligned profiles for norm-invasion events.",
    )
    return parser.parse_args()


def validate_config(config: Config) -> None:
    if not 0 < config.final_fraction <= 1:
        raise ValueError("final_fraction must be in (0, 1].")
    if not 0 <= config.low_threshold < config.high_threshold <= 1:
        raise ValueError("Require 0 <= low_threshold < high_threshold <= 1.")
    integer_fields = (
        "min_state_duration",
        "burnin_generations",
        "delayed_takeoff_generation",
        "bin_width",
        "event_window",
        "event_summary_window",
        "dominant_min_duration",
        "invasion_min_duration",
        "invasion_pre_window",
        "invasion_success_window",
    )
    for field in integer_fields:
        if getattr(config, field) < 1:
            raise ValueError(f"{field} must be >= 1.")


def parse_condition_key(condition_key: str) -> ConditionMeta:
    match = CONDITION_RE.fullmatch(condition_key)
    if match is None:
        raise ValueError(f"Unrecognized condition key: {condition_key}")
    gd = match.groupdict()
    return ConditionMeta(
        condition_key=condition_key,
        num_agents=int(gd["num_agents"]),
        public_norm=gd["public_norm"],
        probability=float(gd["probability"]),
        action_error=float(gd["action_error"]),
        evaluation_error=float(gd["evaluation_error"]),
        public_error=float(gd["public_error"]),
        benefit=float(gd["benefit"]),
    )


def condition_key_from_coop_path(path: Path) -> str:
    name = path.name
    if not name.startswith(COOP_PREFIX) or not name.endswith(".csv"):
        raise ValueError(f"Not a cooperation-rate CSV: {path}")
    return name[len(COOP_PREFIX) : -4]


def condition_part_id(condition_key: str) -> str:
    return hashlib.sha1(condition_key.encode("utf-8")).hexdigest()[:20]


def part_paths(output_dir: Path, part_id: str) -> dict[str, Path]:
    return {
        name: output_dir / name / f"part-{part_id}.parquet"
        for name in DATASET_NAMES
    }


def all_parts_exist(paths: dict[str, Path]) -> bool:
    return all(path.is_file() for path in paths.values())


def atomic_write_parquet(
    df: pd.DataFrame,
    destination: Path,
    compression: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    try:
        df.to_parquet(
            tmp,
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def safe_nanmean(values: np.ndarray, axis: Optional[int] = None) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def safe_nanstd(values: np.ndarray, axis: Optional[int] = None, ddof: int = 0) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(values, axis=axis, ddof=ddof)


def safe_nanquantile(values: np.ndarray, q: float, axis: Optional[int] = None) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanquantile(values, q, axis=axis)


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x2 = x[mask].astype(np.float64)
    y2 = y[mask].astype(np.float64)
    x_centered = x2 - x2.mean()
    denominator = np.dot(x_centered, x_centered)
    if denominator == 0:
        return float("nan")
    return float(np.dot(x_centered, y2 - y2.mean()) / denominator)


def maximum_drawdown(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    x = values.copy()
    # Fill internal NaNs forward only for the drawdown calculation.
    last = np.nan
    for i in range(len(x)):
        if np.isfinite(x[i]):
            last = x[i]
        elif np.isfinite(last):
            x[i] = last
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(x)
    return float(np.max(running_max - x))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        return float("nan")
    ps = p.sum()
    qs = q.sum()
    if ps <= 0 or qs <= 0:
        return float("nan")
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_divergence_series(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence for two equally shaped arrays."""
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("left and right must be equally shaped 2D arrays")
    valid = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    out = np.full(p.shape[0], np.nan, dtype=np.float64)
    if not valid.any():
        return out
    pv = p[valid]
    qv = q[valid]
    ps = pv.sum(axis=1, keepdims=True)
    qs = qv.sum(axis=1, keepdims=True)
    positive = (ps[:, 0] > 0) & (qs[:, 0] > 0)
    if not positive.any():
        return out
    pv2 = pv[positive] / ps[positive]
    qv2 = qv[positive] / qs[positive]
    m = 0.5 * (pv2 + qv2)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.sum(np.where(pv2 > 0, pv2 * np.log(pv2 / m), 0.0), axis=1)
        kl_qm = np.sum(np.where(qv2 > 0, qv2 * np.log(qv2 / m), 0.0), axis=1)
    values = 0.5 * (kl_pm + kl_qm)
    valid_indices = np.flatnonzero(valid)[positive]
    out[valid_indices] = values
    return out


def find_true_runs(mask: np.ndarray, min_duration: int, start_index: int = 0) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).copy()
    start_index = max(0, min(int(start_index), len(mask)))
    mask[:start_index] = False
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends_exclusive = np.flatnonzero(changes == -1)
    runs: list[tuple[int, int]] = []
    for start, end_exclusive in zip(starts, ends_exclusive):
        if end_exclusive - start >= min_duration:
            runs.append((int(start), int(end_exclusive - 1)))
    return runs


def bounded_window_mean(values: np.ndarray, start: int, end_exclusive: int) -> float:
    start = max(0, int(start))
    end_exclusive = min(len(values), int(end_exclusive))
    if end_exclusive <= start:
        return float("nan")
    return float(safe_nanmean(np.asarray(values)[start:end_exclusive]))


def aligned_vector(values: np.ndarray, center: int, radius: int) -> np.ndarray:
    out = np.full(2 * radius + 1, np.nan, dtype=np.float32)
    source_start = max(0, center - radius)
    source_end = min(len(values), center + radius + 1)
    target_start = source_start - (center - radius)
    target_end = target_start + (source_end - source_start)
    out[target_start:target_end] = np.asarray(values[source_start:source_end], dtype=np.float32)
    return out


def canonicalize_event_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=EVENT_CATALOG_COLUMNS)
    string_columns = [
        "condition_key", "public_norm", "trajectory_type", "event_type",
        "event_norm", "from_norm", "to_norm",
    ]
    integer_columns = [
        "num_agents", "simulation_id", "norm_file_simulation_id",
        "event_index", "event_end_index", "event_duration", "event_order",
    ]
    boolean_columns = ["invasion_success", "recovery_within_success_window"]
    for column in string_columns:
        df[column] = df[column].astype("string")
    for column in integer_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in boolean_columns:
        df[column] = df[column].astype("boolean")
    for column in EVENT_CATALOG_COLUMNS:
        if column not in string_columns + integer_columns + boolean_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return df


def canonicalize_event_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=EVENT_PROFILE_COLUMNS)
    string_columns = [
        "condition_key", "public_norm", "event_type", "event_norm",
        "trajectory_group",
    ]
    integer_columns = [
        "num_agents", "relative_generation", "n_events_total",
        "n_events_available",
    ]
    for column in string_columns:
        df[column] = df[column].astype("string")
    for column in integer_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in EVENT_PROFILE_COLUMNS:
        if column not in string_columns + integer_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return df


# -----------------------------
# Raw-data readers
# -----------------------------

def read_cooperation_csv(path: Path) -> tuple[np.ndarray, list[int], np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    if "Generation" not in df.columns:
        raise ValueError(f"Missing Generation column: {path}")

    generations = pd.to_numeric(df["Generation"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(generations)):
        raise ValueError(f"Non-numeric Generation values: {path}")

    sim_pairs: list[tuple[int, str]] = []
    for column in df.columns:
        match = SIM_COLUMN_RE.fullmatch(str(column))
        if match:
            sim_pairs.append((int(match.group("sim")), str(column)))
    sim_pairs.sort(key=lambda x: x[0])
    if not sim_pairs:
        raise ValueError(f"No Sim<number> columns found: {path}")

    sim_ids = [sim_id for sim_id, _ in sim_pairs]
    values = df[[column for _, column in sim_pairs]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32).T
    return generations, sim_ids, values


def read_norm_frequencies(
    path: Path,
    expected_generations: int,
) -> tuple[np.ndarray, dict[str, int]]:
    frequencies = np.full(
        (expected_generations, len(NORM_LABELS)), np.nan, dtype=np.float32
    )
    invalid_tokens = 0
    rows_read = 0
    valid_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty norm file: {path}") from exc
        if len(header) < 2 or header[0] != "Generation":
            raise ValueError(f"Unexpected norm header: {path}")

        for row_index, row in enumerate(reader):
            if row_index >= expected_generations:
                break
            rows_read += 1
            counts = np.zeros(len(NORM_LABELS), dtype=np.int32)
            valid = 0
            for token in row[1:]:
                idx = NORM_TO_INDEX.get(token)
                if idx is None:
                    invalid_tokens += 1
                    continue
                counts[idx] += 1
                valid += 1
            if valid > 0:
                frequencies[row_index] = counts / float(valid)
                valid_rows += 1

    diagnostics = {
        "rows_read": rows_read,
        "valid_rows": valid_rows,
        "invalid_tokens": invalid_tokens,
    }
    return frequencies, diagnostics


# -----------------------------
# Norm and trajectory metrics
# -----------------------------

def public_alignment_weights(public_norm: str) -> np.ndarray:
    return np.asarray(
        [1.0 - sum(a != b for a, b in zip(label, public_norm)) / 4.0 for label in NORM_LABELS],
        dtype=np.float32,
    )


def compute_norm_metrics(
    frequencies: np.ndarray,
    public_norm: str,
) -> dict[str, np.ndarray]:
    frequencies = np.asarray(frequencies, dtype=np.float32)
    row_sums = np.nansum(frequencies, axis=1)
    valid = np.isfinite(frequencies).all(axis=1) & (row_sums > 0)

    entropy = np.full(len(frequencies), np.nan, dtype=np.float32)
    dominance = np.full(len(frequencies), np.nan, dtype=np.float32)
    alignment = np.full(len(frequencies), np.nan, dtype=np.float32)
    q = np.full((len(frequencies), 4), np.nan, dtype=np.float32)
    dominant_index = np.full(len(frequencies), -1, dtype=np.int16)

    if valid.any():
        f = frequencies[valid].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_values = -np.sum(np.where(f > 0, f * np.log(f), 0.0), axis=1) / LOG_N_NORMS
        entropy[valid] = entropy_values.astype(np.float32)
        dominance[valid] = np.max(f, axis=1).astype(np.float32)
        dominant_index[valid] = np.argmax(f, axis=1).astype(np.int16)
        alignment[valid] = (f @ public_alignment_weights(public_norm)).astype(np.float32)
        q[valid] = (f @ NORM_G_BITS).astype(np.float32)

    return {
        "entropy": entropy,
        "dominance": dominance,
        "alignment": alignment,
        "q1": q[:, 0],
        "q2": q[:, 1],
        "q3": q[:, 2],
        "q4": q[:, 3],
        "dominant_index": dominant_index,
    }


def detect_state_events(
    cooperation: np.ndarray,
    generations: np.ndarray,
    config: Config,
) -> dict[str, Any]:
    cooperation = np.asarray(cooperation, dtype=np.float32)
    low_runs = find_true_runs(
        cooperation <= config.low_threshold,
        config.min_state_duration,
        config.burnin_generations,
    )
    high_runs = find_true_runs(
        cooperation >= config.high_threshold,
        config.min_state_duration,
        config.burnin_generations,
    )

    recovery_runs: list[tuple[int, int]] = []
    for high_start, high_end in high_runs:
        preceding = [run for run in low_runs if run[1] < high_start]
        if preceding:
            latest_low = preceding[-1]
            # The high run is a recovery only if no earlier high run occurred after
            # the latest low run and before this one.
            intervening_high = [
                run for run in high_runs
                if latest_low[1] < run[0] < high_start
            ]
            if not intervening_high:
                recovery_runs.append((high_start, high_end))

    first_collapse = low_runs[0][0] if low_runs else None
    first_recovery = recovery_runs[0][0] if recovery_runs else None
    first_takeoff = high_runs[0][0] if high_runs else None

    final_count = max(1, int(math.ceil(len(cooperation) * config.final_fraction)))
    final_mean = float(safe_nanmean(cooperation[-final_count:]))
    final_high = np.isfinite(final_mean) and final_mean >= config.high_threshold
    final_low = np.isfinite(final_mean) and final_mean <= config.low_threshold

    had_collapse = bool(low_runs)
    had_recovery = bool(recovery_runs)
    post_recovery_collapse = False
    if first_recovery is not None:
        post_recovery_collapse = any(start > first_recovery for start, _ in low_runs)

    final_window_start = max(0, len(cooperation) - final_count)
    high_state_reaches_final_window = any(end >= final_window_start for _, end in high_runs)

    if had_collapse and had_recovery:
        if post_recovery_collapse or not final_high:
            trajectory_type = "fluctuating"
        else:
            trajectory_type = "collapse_and_recovery"
    elif had_collapse and not had_recovery:
        trajectory_type = "permanent_collapse" if final_low else "collapsed_unresolved"
    elif not had_collapse and len(high_runs) >= 2:
        trajectory_type = "fluctuating"
    elif (
        not had_collapse
        and first_takeoff is not None
        and final_high
        and high_state_reaches_final_window
    ):
        takeoff_generation = generations[first_takeoff]
        if takeoff_generation <= config.delayed_takeoff_generation:
            trajectory_type = "persistent_cooperation"
        else:
            trajectory_type = "delayed_takeoff"
    else:
        trajectory_type = "intermediate"

    low_durations = [end - start + 1 for start, end in low_runs]
    high_durations = [end - start + 1 for start, end in high_runs]

    return {
        "low_runs": low_runs,
        "high_runs": high_runs,
        "recovery_runs": recovery_runs,
        "first_collapse_index": first_collapse,
        "first_recovery_index": first_recovery,
        "first_takeoff_index": first_takeoff,
        "had_collapse": had_collapse,
        "had_recovery": had_recovery,
        "trajectory_type": trajectory_type,
        "n_collapse_events": len(low_runs),
        "n_recovery_events": len(recovery_runs),
        "n_high_runs": len(high_runs),
        "longest_low_duration": max(low_durations, default=0),
        "longest_high_duration": max(high_durations, default=0),
    }


def detect_dominant_switches(
    norm_frequencies: np.ndarray,
    norm_metrics: dict[str, np.ndarray],
    config: Config,
) -> list[dict[str, Any]]:
    dominant = norm_metrics["dominant_index"]
    dominance = norm_metrics["dominance"]
    events: list[dict[str, Any]] = []
    if len(dominant) == 0:
        return events

    start = config.burnin_generations
    i = start
    previous_stable_norm: Optional[int] = None
    while i < len(dominant):
        current = int(dominant[i])
        if current < 0:
            i += 1
            continue
        j = i + 1
        while j < len(dominant) and int(dominant[j]) == current:
            j += 1
        duration = j - i
        mean_dom = float(safe_nanmean(dominance[i:j]))
        if duration >= config.dominant_min_duration and mean_dom >= config.dominant_min_frequency:
            if previous_stable_norm is not None and current != previous_stable_norm:
                events.append(
                    {
                        "event_type": "dominant_norm_switch",
                        "event_index": i,
                        "event_end_index": j - 1,
                        "event_duration": duration,
                        "from_norm": NORM_LABELS[previous_stable_norm],
                        "to_norm": NORM_LABELS[current],
                        "event_norm": NORM_LABELS[current],
                    }
                )
            previous_stable_norm = current
        i = j
    return events


def detect_norm_invasions(
    norm_frequencies: np.ndarray,
    cooperation: np.ndarray,
    state_info: dict[str, Any],
    config: Config,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    n_generations = norm_frequencies.shape[0]

    for norm_index, norm_label in enumerate(NORM_LABELS):
        freq = norm_frequencies[:, norm_index]
        high_runs = find_true_runs(
            np.isfinite(freq) & (freq >= config.invasion_threshold),
            config.invasion_min_duration,
            config.burnin_generations,
        )
        for start, end in high_runs:
            pre_start = start - config.invasion_pre_window
            if pre_start < 0:
                continue
            pre_mean = bounded_window_mean(freq, pre_start, start)
            if not np.isfinite(pre_mean) or pre_mean > config.invasion_baseline_threshold:
                continue

            success_end = min(n_generations, start + config.invasion_success_window + 1)
            future_freq = freq[start:success_end]
            if np.isfinite(future_freq).any():
                peak_local = int(np.nanargmax(future_freq))
                peak_frequency = float(future_freq[peak_local])
                peak_index = start + peak_local
            else:
                peak_frequency = float("nan")
                peak_index = start

            recovery_within_window = any(
                start <= recovery_start < success_end
                for recovery_start, _ in state_info["recovery_runs"]
            )
            events.append(
                {
                    "event_type": "norm_invasion",
                    "event_index": start,
                    "event_end_index": end,
                    "event_duration": end - start + 1,
                    "event_norm": norm_label,
                    "from_norm": "",
                    "to_norm": norm_label,
                    "invasion_pre_frequency": pre_mean,
                    "invasion_peak_frequency": peak_frequency,
                    "invasion_peak_index": peak_index,
                    "invasion_success": bool(
                        np.isfinite(peak_frequency)
                        and peak_frequency >= config.invasion_success_threshold
                    ),
                    "recovery_within_success_window": recovery_within_window,
                }
            )
    return events


def build_event_rows(
    meta: ConditionMeta,
    sim_id: int,
    norm_file_sim_id: int,
    generations: np.ndarray,
    cooperation: np.ndarray,
    norm_frequencies: np.ndarray,
    norm_metrics: dict[str, np.ndarray],
    state_info: dict[str, Any],
    config: Config,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for order, (start, end) in enumerate(state_info["low_runs"], start=1):
        events.append(
            {
                "event_type": "collapse",
                "event_index": start,
                "event_end_index": end,
                "event_duration": end - start + 1,
                "event_order": order,
                "event_norm": "",
                "from_norm": "",
                "to_norm": "",
            }
        )

    for order, (start, end) in enumerate(state_info["recovery_runs"], start=1):
        events.append(
            {
                "event_type": "recovery",
                "event_index": start,
                "event_end_index": end,
                "event_duration": end - start + 1,
                "event_order": order,
                "event_norm": "",
                "from_norm": "",
                "to_norm": "",
            }
        )

    if state_info["high_runs"]:
        start, end = state_info["high_runs"][0]
        events.append(
            {
                "event_type": "takeoff",
                "event_index": start,
                "event_end_index": end,
                "event_duration": end - start + 1,
                "event_order": 1,
                "event_norm": "",
                "from_norm": "",
                "to_norm": "",
            }
        )

    switch_events = detect_dominant_switches(norm_frequencies, norm_metrics, config)
    for order, event in enumerate(switch_events, start=1):
        event["event_order"] = order
        events.append(event)

    invasion_events = detect_norm_invasions(
        norm_frequencies, cooperation, state_info, config
    )
    for order, event in enumerate(invasion_events, start=1):
        event["event_order"] = order
        events.append(event)

    entropy = norm_metrics["entropy"]
    entropy_search = entropy[config.burnin_generations :]
    if np.isfinite(entropy_search).any():
        peak_index = config.burnin_generations + int(np.nanargmax(entropy_search))
        events.append(
            {
                "event_type": "entropy_peak",
                "event_index": peak_index,
                "event_end_index": peak_index,
                "event_duration": 1,
                "event_order": 1,
                "event_norm": "",
                "from_norm": "",
                "to_norm": "",
            }
        )

    base = meta.as_row()
    base.update(
        {
            "simulation_id": sim_id,
            "norm_file_simulation_id": norm_file_sim_id,
            "trajectory_type": state_info["trajectory_type"],
        }
    )

    event_rows: list[dict[str, Any]] = []
    w = config.event_summary_window
    for event in events:
        idx = int(event["event_index"])
        end_idx = int(event.get("event_end_index", idx))
        row = dict(base)
        row.update(event)
        row["event_generation"] = float(generations[idx])
        row["event_end_generation"] = float(generations[end_idx])

        for name, series in (
            ("cooperation", cooperation),
            ("entropy", norm_metrics["entropy"]),
            ("dominance", norm_metrics["dominance"]),
            ("alignment", norm_metrics["alignment"]),
            ("q1", norm_metrics["q1"]),
            ("q2", norm_metrics["q2"]),
            ("q3", norm_metrics["q3"]),
            ("q4", norm_metrics["q4"]),
        ):
            pre = bounded_window_mean(series, idx - w, idx)
            post = bounded_window_mean(series, idx, idx + w)
            row[f"{name}_pre_mean"] = pre
            row[f"{name}_post_mean"] = post
            row[f"{name}_change"] = post - pre if np.isfinite(pre) and np.isfinite(post) else float("nan")

        event_norm = str(row.get("event_norm", ""))
        if event_norm in NORM_TO_INDEX:
            norm_series = norm_frequencies[:, NORM_TO_INDEX[event_norm]]
            pre = bounded_window_mean(norm_series, idx - w, idx)
            post = bounded_window_mean(norm_series, idx, idx + w)
            row["event_norm_pre_frequency"] = pre
            row["event_norm_post_frequency"] = post
            row["event_norm_frequency_change"] = (
                post - pre if np.isfinite(pre) and np.isfinite(post) else float("nan")
            )
        else:
            row["event_norm_pre_frequency"] = float("nan")
            row["event_norm_post_frequency"] = float("nan")
            row["event_norm_frequency_change"] = float("nan")

        peak_index = row.get("invasion_peak_index")
        if peak_index is not None and np.isfinite(peak_index):
            row["invasion_peak_generation"] = float(generations[int(peak_index)])
        else:
            row["invasion_peak_generation"] = float("nan")
        row.pop("invasion_peak_index", None)
        event_rows.append(row)

    return event_rows


def summarize_run(
    meta: ConditionMeta,
    sim_id: int,
    norm_file_sim_id: int,
    norm_path: Path,
    generations: np.ndarray,
    cooperation: np.ndarray,
    norm_frequencies: np.ndarray,
    norm_metrics: dict[str, np.ndarray],
    norm_diagnostics: dict[str, int],
    state_info: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    n = len(cooperation)
    final_n = max(1, int(math.ceil(n * config.final_fraction)))
    final_slice = slice(n - final_n, n)
    final_generations = generations[final_slice]
    final_coop = cooperation[final_slice]
    final_norm_mean = safe_nanmean(norm_frequencies[final_slice], axis=0)

    row = meta.as_row()
    row.update(
        {
            "simulation_id": sim_id,
            "norm_file_simulation_id": norm_file_sim_id,
            "norm_file": norm_path.name,
            "n_generations": n,
            "final_window_generations": final_n,
            "final_window_start_generation": float(final_generations[0]),
            "valid_norm_generations": norm_diagnostics["valid_rows"],
            "valid_norm_fraction": norm_diagnostics["valid_rows"] / n if n else float("nan"),
            "invalid_norm_tokens": norm_diagnostics["invalid_tokens"],
            "coop_final_mean": float(safe_nanmean(final_coop)),
            "coop_final_sd": float(safe_nanstd(final_coop, ddof=1)),
            "coop_final_slope": linear_slope(final_generations, final_coop),
            "coop_last": float(cooperation[-1]) if np.isfinite(cooperation[-1]) else float("nan"),
            "coop_all_mean": float(safe_nanmean(cooperation)),
            "coop_min": float(np.nanmin(cooperation)) if np.isfinite(cooperation).any() else float("nan"),
            "coop_max": float(np.nanmax(cooperation)) if np.isfinite(cooperation).any() else float("nan"),
            "coop_auc_mean": float(safe_nanmean(cooperation)),
            "coop_prop_high": float(np.mean(cooperation[np.isfinite(cooperation)] >= config.high_threshold)) if np.isfinite(cooperation).any() else float("nan"),
            "coop_prop_low": float(np.mean(cooperation[np.isfinite(cooperation)] <= config.low_threshold)) if np.isfinite(cooperation).any() else float("nan"),
            "coop_max_drawdown": maximum_drawdown(cooperation),
            "trajectory_type": state_info["trajectory_type"],
            "had_collapse": state_info["had_collapse"],
            "had_recovery": state_info["had_recovery"],
            "n_collapse_events": state_info["n_collapse_events"],
            "n_recovery_events": state_info["n_recovery_events"],
            "n_high_runs": state_info["n_high_runs"],
            "longest_low_duration": state_info["longest_low_duration"],
            "longest_high_duration": state_info["longest_high_duration"],
        }
    )

    for key, event_index_key in (
        ("first_collapse_generation", "first_collapse_index"),
        ("first_recovery_generation", "first_recovery_index"),
        ("first_takeoff_generation", "first_takeoff_index"),
    ):
        idx = state_info[event_index_key]
        row[key] = float(generations[idx]) if idx is not None else float("nan")

    for norm_label, value in zip(NORM_LABELS, final_norm_mean):
        row[f"final_norm_freq_{norm_label}"] = float(value)

    final_entropy = float(safe_nanmean(norm_metrics["entropy"][final_slice]))
    final_dominance = float(safe_nanmean(norm_metrics["dominance"][final_slice]))
    final_alignment = float(safe_nanmean(norm_metrics["alignment"][final_slice]))
    row["final_norm_entropy_mean"] = final_entropy
    row["final_norm_dominance_mean"] = final_dominance
    row["final_public_alignment_mean"] = final_alignment
    for i in range(1, 5):
        row[f"final_q{i}_mean"] = float(safe_nanmean(norm_metrics[f"q{i}"][final_slice]))

    if np.isfinite(final_norm_mean).any():
        final_dom_index = int(np.nanargmax(final_norm_mean))
        row["final_dominant_norm"] = NORM_LABELS[final_dom_index]
        row["final_dominant_norm_frequency"] = float(final_norm_mean[final_dom_index])
    else:
        row["final_dominant_norm"] = ""
        row["final_dominant_norm_frequency"] = float("nan")

    initial_norm_mean = safe_nanmean(norm_frequencies[:final_n], axis=0)
    row["norm_jsd_initial_vs_final"] = js_divergence(initial_norm_mean, final_norm_mean)

    turnover = js_divergence_series(norm_frequencies[:-1], norm_frequencies[1:])
    finite_turnover = turnover[np.isfinite(turnover)]
    row["norm_turnover_jsd_mean"] = float(np.mean(finite_turnover)) if finite_turnover.size else float("nan")
    row["norm_turnover_jsd_max"] = float(np.max(finite_turnover)) if finite_turnover.size else float("nan")

    dominant = norm_metrics["dominant_index"]
    valid_dom = dominant[dominant >= 0]
    row["dominant_norm_raw_switch_count"] = int(np.sum(valid_dom[1:] != valid_dom[:-1])) if len(valid_dom) > 1 else 0
    row["dominant_norm_stable_switch_count"] = len(
        detect_dominant_switches(norm_frequencies, norm_metrics, config)
    )
    return row


def build_downsampled_rows(
    meta: ConditionMeta,
    sim_id: int,
    norm_file_sim_id: int,
    generations: np.ndarray,
    cooperation: np.ndarray,
    norm_frequencies: np.ndarray,
    norm_metrics: dict[str, np.ndarray],
    trajectory_type: str,
    config: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(generations)
    for bin_id, start in enumerate(range(0, n, config.bin_width)):
        end = min(n, start + config.bin_width)
        mean_freq = safe_nanmean(norm_frequencies[start:end], axis=0)
        row = meta.as_row()
        row.update(
            {
                "simulation_id": sim_id,
                "norm_file_simulation_id": norm_file_sim_id,
                "trajectory_type": trajectory_type,
                "time_bin": bin_id,
                "bin_start_generation": float(generations[start]),
                "bin_end_generation": float(generations[end - 1]),
                "bin_mid_generation": float(0.5 * (generations[start] + generations[end - 1])),
                "bin_size": end - start,
                "cooperation_mean": float(safe_nanmean(cooperation[start:end])),
                "cooperation_sd": float(safe_nanstd(cooperation[start:end], ddof=1)),
                "entropy_mean": float(safe_nanmean(norm_metrics["entropy"][start:end])),
                "dominance_mean": float(safe_nanmean(norm_metrics["dominance"][start:end])),
                "alignment_mean": float(safe_nanmean(norm_metrics["alignment"][start:end])),
            }
        )
        for i in range(1, 5):
            row[f"q{i}_mean"] = float(safe_nanmean(norm_metrics[f"q{i}"][start:end]))
        for norm_label, value in zip(NORM_LABELS, mean_freq):
            row[f"norm_freq_{norm_label}"] = float(value)
        if np.isfinite(mean_freq).any():
            dom_idx = int(np.nanargmax(mean_freq))
            row["dominant_norm"] = NORM_LABELS[dom_idx]
            row["dominant_norm_frequency"] = float(mean_freq[dom_idx])
        else:
            row["dominant_norm"] = ""
            row["dominant_norm_frequency"] = float("nan")
        rows.append(row)
    return rows


def aggregate_condition_timeseries(
    meta: ConditionMeta,
    generations: np.ndarray,
    coop_stack: np.ndarray,
    norm_stack: np.ndarray,
    metric_stacks: dict[str, np.ndarray],
) -> pd.DataFrame:
    n_runs, n_generations = coop_stack.shape
    data: dict[str, Any] = {**meta.as_row()}
    frame = pd.DataFrame(
        {
            "condition_key": meta.condition_key,
            "num_agents": meta.num_agents,
            "public_norm": meta.public_norm,
            "probability": meta.probability,
            "action_error": meta.action_error,
            "evaluation_error": meta.evaluation_error,
            "public_error": meta.public_error,
            "benefit": meta.benefit,
            "generation": generations,
            "n_runs": n_runs,
            "coop_mean": safe_nanmean(coop_stack, axis=0),
            "coop_median": safe_nanquantile(coop_stack, 0.50, axis=0),
            "coop_sd": safe_nanstd(coop_stack, axis=0, ddof=1),
            "coop_q05": safe_nanquantile(coop_stack, 0.05, axis=0),
            "coop_q25": safe_nanquantile(coop_stack, 0.25, axis=0),
            "coop_q75": safe_nanquantile(coop_stack, 0.75, axis=0),
            "coop_q95": safe_nanquantile(coop_stack, 0.95, axis=0),
        }
    )

    norm_mean = safe_nanmean(norm_stack, axis=0)
    for norm_index, norm_label in enumerate(NORM_LABELS):
        frame[f"norm_mean_{norm_label}"] = norm_mean[:, norm_index]

    for metric in ("entropy", "dominance", "alignment", "q1", "q2", "q3", "q4"):
        values = metric_stacks[metric]
        frame[f"{metric}_mean"] = safe_nanmean(values, axis=0)
        frame[f"{metric}_q25"] = safe_nanquantile(values, 0.25, axis=0)
        frame[f"{metric}_q75"] = safe_nanquantile(values, 0.75, axis=0)

    dominant_indices = metric_stacks["dominant_index"].astype(np.int16)
    mode_labels: list[str] = []
    mode_shares: list[float] = []
    for generation_index in range(n_generations):
        vals = dominant_indices[:, generation_index]
        vals = vals[vals >= 0]
        if vals.size == 0:
            mode_labels.append("")
            mode_shares.append(float("nan"))
        else:
            counts = np.bincount(vals, minlength=len(NORM_LABELS))
            mode = int(np.argmax(counts))
            mode_labels.append(NORM_LABELS[mode])
            mode_shares.append(float(counts[mode] / vals.size))
    frame["dominant_norm_mode"] = mode_labels
    frame["dominant_norm_mode_share"] = mode_shares
    return frame


def aggregate_condition_summary(
    meta: ConditionMeta,
    run_df: pd.DataFrame,
    n_sim_columns: int,
    n_missing_norm_files: int,
    config: Config,
) -> pd.DataFrame:
    row = meta.as_row()
    final = run_df["coop_final_mean"].to_numpy(dtype=float)
    finite_final = final[np.isfinite(final)]
    n = len(run_df)

    row.update(
        {
            "n_sim_columns": n_sim_columns,
            "n_complete_runs": n,
            "n_missing_norm_files": n_missing_norm_files,
            "coop_final_mean": float(np.mean(finite_final)) if finite_final.size else float("nan"),
            "coop_final_median": float(np.median(finite_final)) if finite_final.size else float("nan"),
            "coop_final_sd": float(np.std(finite_final, ddof=1)) if finite_final.size > 1 else float("nan"),
            "coop_final_sem": float(np.std(finite_final, ddof=1) / math.sqrt(finite_final.size)) if finite_final.size > 1 else float("nan"),
            "coop_final_q05": float(np.quantile(finite_final, 0.05)) if finite_final.size else float("nan"),
            "coop_final_q25": float(np.quantile(finite_final, 0.25)) if finite_final.size else float("nan"),
            "coop_final_q75": float(np.quantile(finite_final, 0.75)) if finite_final.size else float("nan"),
            "coop_final_q95": float(np.quantile(finite_final, 0.95)) if finite_final.size else float("nan"),
            "coop_final_min": float(np.min(finite_final)) if finite_final.size else float("nan"),
            "coop_final_max": float(np.max(finite_final)) if finite_final.size else float("nan"),
            "prob_final_high": float(np.mean(finite_final >= config.high_threshold)) if finite_final.size else float("nan"),
            "prob_final_low": float(np.mean(finite_final <= config.low_threshold)) if finite_final.size else float("nan"),
            "prob_collapse": float(run_df["had_collapse"].mean()) if n else float("nan"),
            "prob_recovery": float(run_df["had_recovery"].mean()) if n else float("nan"),
        }
    )

    collapsed = run_df["had_collapse"].astype(bool)
    row["prob_recovery_given_collapse"] = (
        float(run_df.loc[collapsed, "had_recovery"].mean()) if collapsed.any() else float("nan")
    )

    for event_name in ("first_collapse_generation", "first_recovery_generation", "first_takeoff_generation"):
        values = pd.to_numeric(run_df[event_name], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        row[f"{event_name}_mean"] = float(np.mean(values)) if values.size else float("nan")
        row[f"{event_name}_median"] = float(np.median(values)) if values.size else float("nan")

    trajectory_counts = run_df["trajectory_type"].value_counts(dropna=False)
    known_types = (
        "persistent_cooperation",
        "collapse_and_recovery",
        "permanent_collapse",
        "collapsed_unresolved",
        "delayed_takeoff",
        "fluctuating",
        "intermediate",
    )
    for trajectory_type in known_types:
        row[f"prob_type_{trajectory_type}"] = float(
            trajectory_counts.get(trajectory_type, 0) / n
        ) if n else float("nan")

    # Average final norm composition across runs.
    norm_columns = [f"final_norm_freq_{label}" for label in NORM_LABELS]
    mean_norm = run_df[norm_columns].mean(axis=0, skipna=True).to_numpy(dtype=float)
    for label, value in zip(NORM_LABELS, mean_norm):
        row[f"final_norm_mean_{label}"] = float(value)

    if np.isfinite(mean_norm).any() and np.nansum(mean_norm) > 0:
        normalized = np.nan_to_num(mean_norm, nan=0.0)
        normalized = normalized / normalized.sum()
        positive = normalized > 0
        row["final_mean_distribution_entropy"] = float(
            -np.sum(normalized[positive] * np.log(normalized[positive])) / LOG_N_NORMS
        )
        dom = int(np.argmax(normalized))
        row["final_mean_distribution_dominant_norm"] = NORM_LABELS[dom]
        row["final_mean_distribution_dominant_frequency"] = float(normalized[dom])
    else:
        row["final_mean_distribution_entropy"] = float("nan")
        row["final_mean_distribution_dominant_norm"] = ""
        row["final_mean_distribution_dominant_frequency"] = float("nan")

    for source in (
        "final_norm_entropy_mean",
        "final_norm_dominance_mean",
        "final_public_alignment_mean",
        "final_q1_mean",
        "final_q2_mean",
        "final_q3_mean",
        "final_q4_mean",
        "norm_turnover_jsd_mean",
        "norm_turnover_jsd_max",
        "coop_max_drawdown",
    ):
        values = pd.to_numeric(run_df[source], errors="coerce").to_numpy(dtype=float)
        row[f"{source}_across_runs_mean"] = float(safe_nanmean(values))
        row[f"{source}_across_runs_sd"] = float(safe_nanstd(values, ddof=1))

    return pd.DataFrame([row])


def aggregate_event_profiles(
    meta: ConditionMeta,
    generations: np.ndarray,
    run_payloads: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    config: Config,
) -> pd.DataFrame:
    if not event_rows:
        return pd.DataFrame()

    run_lookup = {payload["simulation_id"]: payload for payload in run_payloads}
    allowed = {"collapse", "recovery", "takeoff"}
    if config.include_norm_invasion_profiles:
        allowed.add("norm_invasion")

    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for event in event_rows:
        event_type = str(event["event_type"])
        if event_type not in allowed:
            continue
        simulation_id = int(event["simulation_id"])
        event_index = int(event["event_index"])
        event_norm = str(event.get("event_norm", "")) if event_type == "norm_invasion" else ""
        trajectory_type = str(event["trajectory_type"])
        grouped[(event_type, event_norm, "ALL")].append((simulation_id, event_index))
        grouped[(event_type, event_norm, trajectory_type)].append((simulation_id, event_index))

    radius = config.event_window
    relative = np.arange(-radius, radius + 1, dtype=int)
    profile_rows: list[dict[str, Any]] = []

    variable_names = (
        "cooperation",
        "entropy",
        "dominance",
        "alignment",
        "q1",
        "q2",
        "q3",
        "q4",
    )

    for (event_type, event_norm, trajectory_group), references in grouped.items():
        stacks: dict[str, list[np.ndarray]] = {name: [] for name in variable_names}
        norm_stacks: list[np.ndarray] = []

        for simulation_id, center in references:
            payload = run_lookup.get(simulation_id)
            if payload is None:
                continue
            stacks["cooperation"].append(aligned_vector(payload["cooperation"], center, radius))
            for name in variable_names[1:]:
                stacks[name].append(aligned_vector(payload["norm_metrics"][name], center, radius))

            aligned_norm = np.full(
                (2 * radius + 1, len(NORM_LABELS)), np.nan, dtype=np.float32
            )
            source_start = max(0, center - radius)
            source_end = min(len(generations), center + radius + 1)
            target_start = source_start - (center - radius)
            target_end = target_start + (source_end - source_start)
            aligned_norm[target_start:target_end] = payload["norm_frequencies"][source_start:source_end]
            norm_stacks.append(aligned_norm)

        if not stacks["cooperation"]:
            continue

        stacked = {name: np.vstack(values) for name, values in stacks.items()}
        norm_array = np.stack(norm_stacks, axis=0)
        n_available = np.sum(np.isfinite(stacked["cooperation"]), axis=0)

        means = {name: safe_nanmean(values, axis=0) for name, values in stacked.items()}
        coop_sd = safe_nanstd(stacked["cooperation"], axis=0, ddof=1)
        coop_q25 = safe_nanquantile(stacked["cooperation"], 0.25, axis=0)
        coop_q75 = safe_nanquantile(stacked["cooperation"], 0.75, axis=0)
        norm_mean = safe_nanmean(norm_array, axis=0)

        for rel_index, rel_generation in enumerate(relative):
            row = meta.as_row()
            row.update(
                {
                    "event_type": event_type,
                    "event_norm": event_norm,
                    "trajectory_group": trajectory_group,
                    "relative_generation": int(rel_generation),
                    "n_events_total": len(references),
                    "n_events_available": int(n_available[rel_index]),
                    "cooperation_mean": float(means["cooperation"][rel_index]),
                    "cooperation_sd": float(coop_sd[rel_index]),
                    "cooperation_q25": float(coop_q25[rel_index]),
                    "cooperation_q75": float(coop_q75[rel_index]),
                }
            )
            for name in variable_names[1:]:
                row[f"{name}_mean"] = float(means[name][rel_index])
            for norm_index, norm_label in enumerate(NORM_LABELS):
                row[f"norm_mean_{norm_label}"] = float(norm_mean[rel_index, norm_index])
            profile_rows.append(row)

    return pd.DataFrame(profile_rows)


# -----------------------------
# One-condition processing
# -----------------------------

def process_condition(
    coop_path_str: str,
    output_dir_str: str,
    config_dict: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    started = time.time()
    coop_path = Path(coop_path_str)
    output_dir = Path(output_dir_str)
    config = Config(**config_dict)
    condition_key = condition_key_from_coop_path(coop_path)
    meta = parse_condition_key(condition_key)
    part_id = condition_part_id(condition_key)
    outputs = part_paths(output_dir, part_id)

    if not overwrite and all_parts_exist(outputs):
        return {
            "status": "skipped",
            "condition_key": condition_key,
            "seconds": time.time() - started,
            "n_complete_runs": None,
            "n_missing_norm_files": None,
        }

    generations, sim_ids, coop_matrix = read_cooperation_csv(coop_path)
    n_generations = len(generations)

    run_rows: list[dict[str, Any]] = []
    downsampled_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []
    missing_norm_files: list[str] = []
    invalid_norm_runs = 0

    for row_index, sim_id in enumerate(sim_ids):
        norm_file_sim_id = sim_id + 1
        norm_path = coop_path.parent / f"{NORM_PREFIX}{condition_key}_{norm_file_sim_id}.csv"
        if not norm_path.is_file():
            missing_norm_files.append(norm_path.name)
            continue

        cooperation = coop_matrix[row_index]
        norm_frequencies, norm_diagnostics = read_norm_frequencies(
            norm_path, expected_generations=n_generations
        )
        valid_fraction = norm_diagnostics["valid_rows"] / n_generations if n_generations else 0.0
        if valid_fraction < config.min_valid_norm_fraction:
            invalid_norm_runs += 1
            continue

        norm_metrics = compute_norm_metrics(norm_frequencies, meta.public_norm)
        state_info = detect_state_events(cooperation, generations, config)

        run_row = summarize_run(
            meta=meta,
            sim_id=sim_id,
            norm_file_sim_id=norm_file_sim_id,
            norm_path=norm_path,
            generations=generations,
            cooperation=cooperation,
            norm_frequencies=norm_frequencies,
            norm_metrics=norm_metrics,
            norm_diagnostics=norm_diagnostics,
            state_info=state_info,
            config=config,
        )
        run_rows.append(run_row)

        downsampled_rows.extend(
            build_downsampled_rows(
                meta=meta,
                sim_id=sim_id,
                norm_file_sim_id=norm_file_sim_id,
                generations=generations,
                cooperation=cooperation,
                norm_frequencies=norm_frequencies,
                norm_metrics=norm_metrics,
                trajectory_type=state_info["trajectory_type"],
                config=config,
            )
        )

        event_rows = build_event_rows(
            meta=meta,
            sim_id=sim_id,
            norm_file_sim_id=norm_file_sim_id,
            generations=generations,
            cooperation=cooperation,
            norm_frequencies=norm_frequencies,
            norm_metrics=norm_metrics,
            state_info=state_info,
            config=config,
        )
        all_event_rows.extend(event_rows)

        run_payloads.append(
            {
                "simulation_id": sim_id,
                "cooperation": cooperation,
                "norm_frequencies": norm_frequencies,
                "norm_metrics": norm_metrics,
            }
        )

    if not run_rows:
        raise RuntimeError(
            f"No complete valid runs for {condition_key}. "
            f"Missing norm files: {len(missing_norm_files)}; invalid runs: {invalid_norm_runs}."
        )

    run_df = pd.DataFrame(run_rows)
    downsampled_df = pd.DataFrame(downsampled_rows)
    event_df = canonicalize_event_catalog(pd.DataFrame(all_event_rows))

    coop_stack = np.vstack([payload["cooperation"] for payload in run_payloads])
    norm_stack = np.stack([payload["norm_frequencies"] for payload in run_payloads], axis=0)
    metric_stacks = {
        name: np.stack([payload["norm_metrics"][name] for payload in run_payloads], axis=0)
        for name in ("entropy", "dominance", "alignment", "q1", "q2", "q3", "q4", "dominant_index")
    }

    condition_summary_df = aggregate_condition_summary(
        meta=meta,
        run_df=run_df,
        n_sim_columns=len(sim_ids),
        n_missing_norm_files=len(missing_norm_files) + invalid_norm_runs,
        config=config,
    )
    condition_timeseries_df = aggregate_condition_timeseries(
        meta=meta,
        generations=generations,
        coop_stack=coop_stack,
        norm_stack=norm_stack,
        metric_stacks=metric_stacks,
    )
    event_profiles_df = canonicalize_event_profiles(
        aggregate_event_profiles(
            meta=meta,
            generations=generations,
            run_payloads=run_payloads,
            event_rows=all_event_rows,
            config=config,
        )
    )

    dataframes = {
        "run_summary": run_df,
        "condition_summary": condition_summary_df,
        "condition_timeseries": condition_timeseries_df,
        "run_trajectory_downsampled": downsampled_df,
        "event_catalog": event_df,
        "event_aligned_profiles": event_profiles_df,
    }

    # Write all temporary files first; atomic replacement happens per dataset.
    for dataset_name, dataframe in dataframes.items():
        atomic_write_parquet(
            dataframe,
            outputs[dataset_name],
            compression=config.parquet_compression,
        )

    return {
        "status": "processed",
        "condition_key": condition_key,
        "seconds": time.time() - started,
        "n_complete_runs": len(run_df),
        "n_missing_norm_files": len(missing_norm_files),
        "n_invalid_norm_runs": invalid_norm_runs,
        "n_events": len(event_df),
        "n_profile_rows": len(event_profiles_df),
    }


# -----------------------------
# Main orchestration
# -----------------------------

def prepare_output_directory(output_dir: Path, config: Config, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in DATASET_NAMES:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "_analysis_config.json"
    current = {
        "script_version": SCRIPT_VERSION,
        "config": asdict(config),
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != current:
            raise RuntimeError(
                f"Existing output configuration differs: {config_path}\n"
                "Use a new --output-dir. Mixing different analysis definitions "
                "in one dataset is intentionally prohibited."
            )
    config_path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_processing_log(log_path: Path, result: dict[str, Any]) -> None:
    write_header = not log_path.exists()
    fieldnames = [
        "timestamp",
        "status",
        "condition_key",
        "seconds",
        "n_complete_runs",
        "n_missing_norm_files",
        "n_invalid_norm_runs",
        "n_events",
        "n_profile_rows",
        "error",
    ]
    row = {field: result.get(field, "") for field in fieldnames}
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "analysis_data")).resolve()

    config = Config(
        final_fraction=args.final_fraction,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        min_state_duration=args.min_state_duration,
        burnin_generations=args.burnin_generations,
        bin_width=args.bin_width,
        event_window=args.event_window,
        include_norm_invasion_profiles=not args.no_invasion_profiles,
    )
    validate_config(config)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    prepare_output_directory(output_dir, config, args.overwrite)
    success_marker = output_dir / "_SUCCESS"
    if success_marker.exists():
        success_marker.unlink()

    coop_files = sorted(input_dir.glob(f"{COOP_PREFIX}*.csv"))
    if args.limit is not None:
        coop_files = coop_files[: args.limit]
    if not coop_files:
        raise SystemExit(
            f"No {COOP_PREFIX}*.csv files were found in {input_dir}"
        )

    # Validate all names before launching workers, so malformed names fail early.
    valid_files: list[Path] = []
    malformed: list[str] = []
    for path in coop_files:
        try:
            parse_condition_key(condition_key_from_coop_path(path))
            valid_files.append(path)
        except Exception:
            malformed.append(path.name)
    if malformed:
        print(
            f"Warning: skipped {len(malformed)} files with unrecognized names. "
            f"See {output_dir / '_malformed_cooperation_files.txt'}",
            file=sys.stderr,
        )
        (output_dir / "_malformed_cooperation_files.txt").write_text(
            "\n".join(malformed), encoding="utf-8"
        )

    config_dict = asdict(config)
    log_path = output_dir / "processing_log.csv"
    error_path = output_dir / "processing_errors.log"
    total = len(valid_files)
    processed = skipped = failed = 0
    started = time.time()

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Conditions      : {total}")
    print(f"Workers         : {args.workers}")
    print("Parquet compression: " + config.parquet_compression)

    if args.workers == 1:
        for index, path in enumerate(valid_files, start=1):
            try:
                result = process_condition(
                    str(path), str(output_dir), config_dict, args.overwrite
                )
                if result["status"] == "processed":
                    processed += 1
                else:
                    skipped += 1
                append_processing_log(log_path, result)
            except Exception as exc:
                failed += 1
                tb = traceback.format_exc()
                error_path.open("a", encoding="utf-8").write(
                    f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {path}\n{tb}\n"
                )
                append_processing_log(
                    log_path,
                    {
                        "status": "failed",
                        "condition_key": path.name,
                        "error": repr(exc),
                    },
                )
            if index == 1 or index % 10 == 0 or index == total:
                elapsed = time.time() - started
                print(
                    f"[{index}/{total}] processed={processed}, skipped={skipped}, "
                    f"failed={failed}, elapsed={elapsed/60:.1f} min",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_condition,
                    str(path),
                    str(output_dir),
                    config_dict,
                    args.overwrite,
                ): path
                for path in valid_files
            }
            for index, future in enumerate(as_completed(futures), start=1):
                path = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "processed":
                        processed += 1
                    else:
                        skipped += 1
                    append_processing_log(log_path, result)
                except Exception as exc:
                    failed += 1
                    tb = traceback.format_exc()
                    with error_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {path}\n{tb}\n"
                        )
                    append_processing_log(
                        log_path,
                        {
                            "status": "failed",
                            "condition_key": path.name,
                            "error": repr(exc),
                        },
                    )
                if index == 1 or index % 10 == 0 or index == total:
                    elapsed = time.time() - started
                    print(
                        f"[{index}/{total}] processed={processed}, skipped={skipped}, "
                        f"failed={failed}, elapsed={elapsed/60:.1f} min",
                        flush=True,
                    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "conditions_found": total,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
        "datasets": list(DATASET_NAMES),
    }
    (output_dir / "_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failed == 0:
        (output_dir / "_SUCCESS").write_text("", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Required for safe multiprocessing on Windows.
    raise SystemExit(main())
