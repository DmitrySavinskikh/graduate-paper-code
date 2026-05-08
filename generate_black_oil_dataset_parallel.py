"""
Parallel dataset generation for the simplified quarter-five-spot Black-Oil simulator.

This script is meant for creating ML datasets where only final scalar targets are needed.
It reuses the simulator from generate_black_oil_dataset.py and runs many independent
simulations in parallel.

Main idea:
    one simulation = one independent task = perfect for multiprocessing.

Recommended usage examples:

1) Fast screening dataset, one row per simulation:

    python generate_black_oil_dataset_parallel.py \
        --n-simulations 100000 \
        --preset fast \
        --workers 12 \
        --output black_oil_100k_fast.csv

2) Higher-fidelity dataset, still parallel:

    python generate_black_oil_dataset_parallel.py \
        --n-simulations 10000 \
        --preset high_fidelity \
        --workers 12 \
        --output black_oil_10k_high_fidelity.csv

3) Save final fields too, only recommended for smaller datasets:

    python generate_black_oil_dataset_parallel.py \
        --n-simulations 1000 \
        --preset fast \
        --workers 12 \
        --output black_oil_1k_fast.csv \
        --save-fields black_oil_1k_fast_fields.npz

Notes:
- For 100,000 records, keep final fields disabled unless you really need field targets.
- The CSV is written incrementally in batches, so memory usage stays modest.
- Gas production is zero because this simulator is a dead-oil oil-water model.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from generate_black_oil_dataset import (
    SimConfig,
    QuarterFiveSpotSimulator,
    sample_parameters,
)


def make_config(preset: str) -> SimConfig:
    """Numerical presets for speed/fidelity tradeoff."""
    if preset == "high_fidelity":
        # Same quality as the refined previous notebook.
        return SimConfig(Nx=31, Ny=31, total_years=20.0, dt_days=3.0)

    if preset == "fast":
        # Good default for large scalar-output ML datasets.
        # Much fewer cells and time steps than high_fidelity.
        return SimConfig(Nx=15, Ny=15, total_years=20.0, dt_days=10.0)

    if preset == "ultra_fast":
        # Useful for quick tests or very large preliminary datasets.
        return SimConfig(Nx=11, Ny=11, total_years=20.0, dt_days=15.0)

    raise ValueError(
        f"Unknown preset={preset!r}. Choose from: high_fidelity, fast, ultra_fast."
    )


def run_one_simulation(
    sim_id: int,
    seed: int,
    config: SimConfig,
    keep_final_fields: bool = False,
) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Run one independent simulation and return one ML row."""
    rng = np.random.default_rng(seed + sim_id)
    params = sample_parameters(rng)

    simulator = QuarterFiveSpotSimulator(config, params)
    out = simulator.run(keep_final_fields=keep_final_fields)

    row: Dict[str, float] = {"sim_id": sim_id}
    row.update(asdict(config))
    row.update(asdict(params))

    for key, value in out.items():
        if key not in {"p_final", "Sw_final"}:
            row[key] = value

    p_final = out.get("p_final") if keep_final_fields else None
    Sw_final = out.get("Sw_final") if keep_final_fields else None

    return row, p_final, Sw_final


def write_batch_csv(path: str, rows: List[Dict[str, float]], write_header: bool) -> None:
    """Append a batch of rows to CSV."""
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", index=False, header=write_header)


def generate_parallel_dataset(
    n_simulations: int,
    output_csv: str,
    preset: str = "fast",
    seed: int = 42,
    workers: Optional[int] = None,
    batch_size: int = 500,
    save_fields_npz: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a one-row-per-simulation dataset in parallel.

    Parameters
    ----------
    n_simulations:
        Number of independent simulations / dataset rows.
    output_csv:
        Output CSV path.
    preset:
        high_fidelity, fast, or ultra_fast.
    seed:
        Base random seed. Simulation i uses seed + i.
    workers:
        Number of processes. Default: max(1, cpu_count - 1).
    batch_size:
        Number of completed simulations accumulated before appending to CSV.
    save_fields_npz:
        Optional path for final pressure and Sw fields. Not recommended for 100k unless
        storage is planned carefully.
    """
    config = make_config(preset)
    keep_final_fields = save_fields_npz is not None

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    # Start from a clean CSV.
    if os.path.exists(output_csv):
        os.remove(output_csv)

    started = time.time()
    completed = 0
    write_header = True
    rows_buffer: List[Dict[str, float]] = []

    # Only for optional field saving.
    field_sim_ids: List[int] = []
    p_fields: List[np.ndarray] = []
    Sw_fields: List[np.ndarray] = []

    print("Dataset generation settings")
    print(f"  n_simulations = {n_simulations}")
    print(f"  preset        = {preset}")
    print(f"  grid          = {config.Nx} x {config.Ny}")
    print(f"  dt_days       = {config.dt_days}")
    print(f"  total_years   = {config.total_years}")
    print(f"  workers       = {workers}")
    print(f"  output_csv    = {output_csv}")
    print(f"  save_fields   = {keep_final_fields}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_one_simulation,
                sim_id,
                seed,
                config,
                keep_final_fields,
            )
            for sim_id in range(n_simulations)
        ]

        for future in as_completed(futures):
            row, p_final, Sw_final = future.result()
            rows_buffer.append(row)

            if keep_final_fields:
                field_sim_ids.append(int(row["sim_id"]))
                p_fields.append(p_final)
                Sw_fields.append(Sw_final)

            completed += 1

            if len(rows_buffer) >= batch_size:
                rows_buffer.sort(key=lambda r: int(r["sim_id"]))
                write_batch_csv(output_csv, rows_buffer, write_header=write_header)
                write_header = False
                rows_buffer.clear()

            if completed == 1 or completed % max(1, n_simulations // 100) == 0:
                elapsed = time.time() - started
                sims_per_sec = completed / max(elapsed, 1e-12)
                remaining = (n_simulations - completed) / max(sims_per_sec, 1e-12)
                print(
                    f"completed {completed}/{n_simulations} | "
                    f"{sims_per_sec:.3f} sims/s | "
                    f"elapsed {elapsed/60:.1f} min | "
                    f"estimated remaining {remaining/3600:.2f} h"
                )

    if rows_buffer:
        rows_buffer.sort(key=lambda r: int(r["sim_id"]))
        write_batch_csv(output_csv, rows_buffer, write_header=write_header)
        rows_buffer.clear()

    # Sort CSV by sim_id after all asynchronous results have been appended.
    df = pd.read_csv(output_csv).sort_values("sim_id").reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    if keep_final_fields:
        order = np.argsort(np.asarray(field_sim_ids))
        np.savez_compressed(
            save_fields_npz,
            sim_id=np.asarray(field_sim_ids, dtype=int)[order],
            p_final_MPa=np.asarray(p_fields)[order] / 1.0e6,
            Sw_final=np.asarray(Sw_fields)[order],
        )
        print(f"Saved final fields to: {save_fields_npz}")

    elapsed = time.time() - started
    print(f"Saved summary dataset to: {output_csv}")
    print(f"Total elapsed: {elapsed/60:.2f} min")
    print(f"Average speed: {n_simulations / max(elapsed, 1e-12):.3f} simulations/s")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--output", type=str, default="black_oil_parallel_dataset.csv")
    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        choices=["high_fidelity", "fast", "ultra_fast"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--save-fields", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_parallel_dataset(
        n_simulations=args.n_simulations,
        output_csv=args.output,
        preset=args.preset,
        seed=args.seed,
        workers=args.workers,
        batch_size=args.batch_size,
        save_fields_npz=args.save_fields,
    )
