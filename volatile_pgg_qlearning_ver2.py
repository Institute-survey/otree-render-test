#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

GROUP_SIZE = 4
BASE_ENDOWMENT = 100.0
INITIAL_ENDOWMENT = 100.0
ENDOWMENT_FLOOR = 1.0
MULTIPLIER = 1.6
MPCR = MULTIPLIER / GROUP_SIZE
ACTION_LEVELS = 11
ALPHA = 0.2

RHO_GRID = [0.2, 0.5, 0.8]
SIGMA_E_GRID = [5.0, 15.0, 30.0]
BASE_ERROR_PROB_GRID = [0.0, 0.1]
ERROR_SENSITIVITY_GRID = [0.0, 1.5, 3.0]
ERROR_MODE_GRID = ["flip", "random"]
EARNED_GRID = [False, True]
COST_WEIGHT_INCREASE_GRID = [0.1, 0.3]
GRATITUDE_STEPS_GRID = [0, 1, 5]


@dataclass(frozen=True)
class Condition:
    condition_type: str
    rho: float
    sigma_e: float
    base_error_prob: float
    error_sensitivity: float
    error_mode: str
    earned: bool
    cost_weight_increase: float
    gratitude_steps: int


@dataclass(frozen=True)
class TaskSpec:
    condition: Condition
    condition_id: str
    group_id: int
    rounds: int
    seed: int


def condition_id(cond: Condition) -> str:
    if cond.condition_type == "baseline":
        return "baseline"
    earned_label = "earned" if cond.earned else "not_earned"
    cwi = f"{cond.cost_weight_increase:.1f}" if cond.earned else "na"
    grat = f"{cond.gratitude_steps}" if cond.earned else "na"
    return (
        f"rho_{cond.rho:.1f}"
        f"__sigma_{int(cond.sigma_e)}"
        f"__p0_{cond.base_error_prob:.1f}"
        f"__alpha_{cond.error_sensitivity:.1f}"
        f"__err_{cond.error_mode}"
        f"__{earned_label}"
        f"__cost_{cwi}"
        f"__grat_{grat}"
    )


def _code_from_decimal(value: float, scale: int = 10, width: int = 2) -> str:
    return f"{int(round(value * scale)):0{width}d}"


def condition_code(cond: Condition) -> str:
    if cond.condition_type == "baseline":
        return "base"

    rho_code = _code_from_decimal(cond.rho, scale=10, width=2)
    sigma_code = f"{int(round(cond.sigma_e)):02d}"
    p0_code = _code_from_decimal(cond.base_error_prob, scale=10, width=2)
    alpha_code = _code_from_decimal(cond.error_sensitivity, scale=10, width=2)
    err_code_map = {"none": "n", "flip": "f", "random": "r"}
    err_code = err_code_map[cond.error_mode]

    if not cond.earned:
        return f"r{rho_code}_s{sigma_code}_p{p0_code}_a{alpha_code}_{err_code}_n"

    cost_code = _code_from_decimal(cond.cost_weight_increase, scale=10, width=2)
    return f"r{rho_code}_s{sigma_code}_p{p0_code}_a{alpha_code}_{err_code}_e_c{cost_code}_g{int(cond.gratitude_steps)}"


def condition_filename(cond: Condition) -> str:
    return f"{condition_code(cond)}.csv"


def build_conditions(include_baseline: bool = True) -> List[Condition]:
    conditions: List[Condition] = []
    if include_baseline:
        conditions.append(
            Condition("baseline", 0.0, 0.0, 0.0, 0.0, "none", False, 0.0, 0)
        )

    for rho, sigma_e, p0, alpha, err_mode, earned in itertools.product(
        RHO_GRID,
        SIGMA_E_GRID,
        BASE_ERROR_PROB_GRID,
        ERROR_SENSITIVITY_GRID,
        ERROR_MODE_GRID,
        EARNED_GRID,
    ):
        if earned:
            for cwi, grat in itertools.product(COST_WEIGHT_INCREASE_GRID, GRATITUDE_STEPS_GRID):
                conditions.append(
                    Condition("experimental", rho, sigma_e, p0, alpha, err_mode, True, cwi, grat)
                )
        else:
            conditions.append(
                Condition("experimental", rho, sigma_e, p0, alpha, err_mode, False, 0.0, 0)
            )
    return conditions


def action_grid_for_endowment(endowment: float) -> np.ndarray:
    return np.linspace(0.0, endowment, ACTION_LEVELS)


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def clip_index(index: int) -> int:
    return max(0, min(ACTION_LEVELS - 1, index))


def generate_group_endowments(rounds: int, rho: float, sigma_e: float, rng: np.random.Generator) -> np.ndarray:
    endowments = np.empty(rounds, dtype=float)
    endowments[0] = max(ENDOWMENT_FLOOR, INITIAL_ENDOWMENT)
    for t in range(1, rounds):
        shock = rng.normal(0.0, sigma_e)
        endowments[t] = rho * endowments[t - 1] + (1.0 - rho) * BASE_ENDOWMENT + shock
        if endowments[t] < ENDOWMENT_FLOOR:
            endowments[t] = ENDOWMENT_FLOOR
    return endowments


def subjective_cost_multiplier(cond: Condition) -> float:
    return 1.0 + cond.cost_weight_increase if cond.earned else 1.0


def subjective_marginal_effect(cond: Condition) -> float:
    return MPCR - subjective_cost_multiplier(cond)


def simulate_group(condition: Condition, rounds: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    endowments = generate_group_endowments(rounds, condition.rho, condition.sigma_e, rng)

    marginal_estimates = np.zeros(GROUP_SIZE, dtype=float)
    target_marginal = subjective_marginal_effect(condition)

    initial_grid = action_grid_for_endowment(endowments[0])
    current_indices = rng.integers(0, ACTION_LEVELS, size=GROUP_SIZE)
    current_contributions = initial_grid[current_indices]

    mean_rates = np.empty(rounds, dtype=float)

    for t in range(rounds):
        current_endowment = endowments[t]
        total_contribution = float(np.sum(current_contributions))
        share_from_others = MPCR * (total_contribution - current_contributions)

        mean_rates[t] = float(np.mean(current_contributions / current_endowment))
        marginal_estimates += ALPHA * (target_marginal - marginal_estimates)

        if t == rounds - 1:
            break

        next_endowment = endowments[t + 1]
        next_grid = action_grid_for_endowment(next_endowment)
        next_indices = np.empty(GROUP_SIZE, dtype=int)
        next_contributions = np.empty(GROUP_SIZE, dtype=float)

        volatility = abs(endowments[t] - endowments[t - 1]) / 100.0 if t >= 1 else 0.0
        p_error = min(1.0, condition.base_error_prob + condition.error_sensitivity * volatility)

        for i in range(GROUP_SIZE):
            if marginal_estimates[i] > 0:
                direction = 1
            elif marginal_estimates[i] < 0:
                direction = -1
            else:
                direction = 0

            if condition.error_mode != "none" and rng.random() < p_error:
                if condition.error_mode == "flip":
                    direction = -direction if direction != 0 else int(rng.choice(np.array([-1, 1], dtype=int)))
                elif condition.error_mode == "random":
                    direction = int(rng.choice(np.array([-1, 0, 1], dtype=int)))
                else:
                    raise ValueError(f"Unknown error_mode: {condition.error_mode}")

            gratitude_extra_steps = 0
            if condition.earned and condition.gratitude_steps > 0 and share_from_others[i] > 0:
                gratitude_extra_steps = condition.gratitude_steps

            reference_amount = current_contributions[i]
            base_index = nearest_index(next_grid, reference_amount)
            new_index = clip_index(base_index + direction + gratitude_extra_steps)

            next_indices[i] = new_index
            next_contributions[i] = next_grid[new_index]

        current_indices = next_indices
        current_contributions = next_contributions

    return mean_rates, endowments


def run_task(task: TaskSpec) -> Tuple[str, int, np.ndarray, np.ndarray]:
    mean_rates, endowments = simulate_group(task.condition, task.rounds, task.seed)
    return task.condition_id, task.group_id, mean_rates, endowments


def build_tasks(conditions: List[Condition], groups: int, rounds: int, base_seed: int) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    for condition_index, cond in enumerate(conditions):
        cid = condition_id(cond)
        for group_id in range(1, groups + 1):
            seed = base_seed + condition_index * 1_000_000 + group_id
            tasks.append(TaskSpec(cond, cid, group_id, rounds, seed))
    return tasks


def wide_rows(series_list: List[np.ndarray], rounds: int) -> pd.DataFrame:
    rows = []
    for group_index, series in enumerate(series_list, start=1):
        row = {"group_id": group_index}
        for r in range(rounds):
            row[f"round_{r + 1}"] = float(series[r])
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[["group_id"] + [f"round_{i + 1}" for i in range(rounds)]]


def add_metadata(df: pd.DataFrame, meta: dict, rounds: int) -> pd.DataFrame:
    out = df.copy()
    for k, v in meta.items():
        if not k.endswith("_csv_path"):
            out[k] = v
    ordered_cols = [
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
    ] + [f"round_{i + 1}" for i in range(rounds)]
    return out[ordered_cols]


def write_outputs(
    output_dir: Path,
    conditions: List[Condition],
    cooperation_by_condition: Dict[str, List[np.ndarray]],
    endowment_by_condition: Dict[str, List[np.ndarray]],
    rounds: int,
) -> None:
    condition_dir = output_dir / "condition_csvs"
    endowment_dir = output_dir / "endowment_csvs"
    condition_dir.mkdir(parents=True, exist_ok=True)
    endowment_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    combined_frames: List[pd.DataFrame] = []
    endowment_combined_frames: List[pd.DataFrame] = []

    for cond in conditions:
        cid = condition_id(cond)
        ccode = condition_code(cond)
        filename = condition_filename(cond)

        coop_df = wide_rows(cooperation_by_condition[cid], rounds)
        endowment_df = wide_rows(endowment_by_condition[cid], rounds)

        coop_csv_path = condition_dir / filename
        endowment_csv_path = endowment_dir / filename
        coop_df.to_csv(coop_csv_path, index=False)
        endowment_df.to_csv(endowment_csv_path, index=False)

        meta = asdict(cond)
        meta["condition_id"] = cid
        meta["condition_code"] = ccode
        meta["filename"] = filename
        meta["cooperation_csv_path"] = str(coop_csv_path)
        meta["endowment_csv_path"] = str(endowment_csv_path)
        manifest_rows.append(meta)

        combined_frames.append(add_metadata(coop_df, meta, rounds))
        endowment_combined_frames.append(add_metadata(endowment_df, meta, rounds))

    pd.DataFrame(manifest_rows).sort_values(
        ["condition_type", "condition_code", "condition_id"]
    ).to_csv(output_dir / "manifest_conditions.csv", index=False)

    pd.concat(combined_frames, ignore_index=True).sort_values(
        ["condition_type", "condition_code", "group_id"]
    ).to_csv(output_dir / "all_conditions_group_means.csv", index=False)

    pd.concat(endowment_combined_frames, ignore_index=True).sort_values(
        ["condition_type", "condition_code", "group_id"]
    ).to_csv(output_dir / "all_conditions_endowments.csv", index=False)


def run_all_conditions(
    output_dir: Path,
    groups: int,
    rounds: int,
    workers: int | None,
    seed: int,
    include_baseline: bool,
    max_conditions: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(include_baseline=include_baseline)
    if max_conditions is not None:
        conditions = conditions[:max_conditions]

    tasks = build_tasks(conditions, groups, rounds, seed)

    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    cooperation_by_condition: Dict[str, List[np.ndarray]] = {}
    endowment_by_condition: Dict[str, List[np.ndarray]] = {}
    for cond in conditions:
        cid = condition_id(cond)
        cooperation_by_condition[cid] = [None] * groups
        endowment_by_condition[cid] = [None] * groups

    if not tasks:
        raise ValueError("No tasks were generated. Check condition settings.")

    chunksize = max(1, len(tasks) // max(1, workers * 16))

    completed = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for cid, group_id, mean_rates, endowments in executor.map(run_task, tasks, chunksize=chunksize):
            cooperation_by_condition[cid][group_id - 1] = mean_rates
            endowment_by_condition[cid][group_id - 1] = endowments
            completed += 1
            if completed % 1000 == 0 or completed == total:
                print(f"Completed {completed}/{total} simulations")

    write_outputs(output_dir, conditions, cooperation_by_condition, endowment_by_condition, rounds)

    print(f"Saved manifest to: {output_dir / 'manifest_conditions.csv'}")
    print(f"Saved cooperation results to: {output_dir / 'all_conditions_group_means.csv'}")
    print(f"Saved Endowment results to: {output_dir / 'all_conditions_endowments.csv'}")
    print(f"Per-condition cooperation CSVs are in: {output_dir / 'condition_csvs'}")
    print(f"Per-condition Endowment CSVs are in: {output_dir / 'endowment_csvs'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded PGG simulations with subjective marginal-effect learning.")
    parser.add_argument("--output-dir", type=str, default="expanded_pgg_outputs_marginal_learning")
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--include-baseline", action="store_true", default=True)
    parser.add_argument("--exclude-baseline", action="store_true")
    parser.add_argument("--max-conditions", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    include_baseline = args.include_baseline and not args.exclude_baseline
    run_all_conditions(
        output_dir=Path(args.output_dir),
        groups=args.groups,
        rounds=args.rounds,
        workers=args.workers,
        seed=args.seed,
        include_baseline=include_baseline,
        max_conditions=args.max_conditions,
    )
