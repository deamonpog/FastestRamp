"""Public-facing API that re-exports the simulation helpers.

This file used to contain all logic. It now delegates to smaller modules while
preserving the same top-level functions for callers (e.g., `multi_run.py`).
"""

from simcore import constants
from simcore.single import SimulationResult, run_sim
from simcore.multi import run_multi, run_multi_parallel
from simcore.visualize import visualize_results_grid

__all__ = [
    "SimulationResult",
    "run_sim",
    "run_multi",
    "run_multi_parallel",
    "visualize_results_grid",
    "constants",
]