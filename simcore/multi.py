"""Batch and parallel simulation helpers."""
from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, Tuple

from .single import SimulationResult, run_sim


def _result_to_dict(result: SimulationResult) -> Dict[str, object]:
    # Normalize SimulationResult into the legacy dict structure callers expect
    return {
        "t": result.time_to_finish,  # backward-compat alias
        "time": result.time_to_finish,
        "log": result.physics_log,
        "killed": result.killed,
        "kill_y": result.kill_y,
        "contact_episodes": result.contact_episodes,
        "time_on_ramp": result.time_on_ramp,
    }


def run_multi(ramp_points, x_locations, y_height, display=False, kill_offset=250):
    """Run a simulation for each starting X-location (sequential)."""
    results = {}
    for x in x_locations:
        # Sequentially run each start position
        print(f"\n=== Sim from (x={x}, y={y_height}) ===")
        sim_result = run_sim(
            ramp_points,
            start_x=x,
            start_y=y_height,
            show_buttons=False,
            allow_restart=False,
            display=display,
            kill_y=None,
            kill_offset=kill_offset,
        )
        results[x] = _result_to_dict(sim_result)
    return results


def _sim_worker(args: Tuple[list, float, float, float]):
    # Child-process worker for multiprocessing Pool
    ramp_points, x, y_height, kill_offset = args
    sim_result = run_sim(
        ramp_points,
        start_x=x,
        start_y=y_height,
        show_buttons=False,
        allow_restart=False,
        display=False,
        kill_y=None,
        kill_offset=kill_offset,
    )
    return x, _result_to_dict(sim_result)


def run_multi_parallel(ramp_points, x_locations: Iterable[float], y_height, kill_offset=250):
    """Run multiple simulations in parallel (headless)."""
    args = [(ramp_points, x, y_height, kill_offset) for x in x_locations]
    # Use all available cores for independent simulations
    n_cpus = cpu_count()
    print(f"Running {len(args)} simulations on {n_cpus} cores (headless)...")

    results = {}
    with Pool(processes=n_cpus) as pool:
        for x, data in pool.map(_sim_worker, args):
            results[x] = data
    return results
