# src/ns_solver/__init__.py
from .grid import Grid
from .simulation import SimulationClass, CavitySimulation, CylinderSimulation
from .solver import build_up_b, pressure_poisson, update_velocity
from .finite_differences import (
    backward_diff_x, backward_diff_y,
    forward_diff_x, forward_diff_y,
    centered_diff_x, centered_diff_y,
    laplacian_2d
)

__all__ = [
    "Grid",
    "SimulationClass",
    "CavitySimulation",
    "CylinderSimulation",
    "build_up_b",
    "pressure_poisson",
    "update_velocity",
    "backward_diff_x",
    "backward_diff_y",
    "forward_diff_x",
    "forward_diff_y",
    "centered_diff_x",
    "centered_diff_y",
    "laplacian_2d",
]