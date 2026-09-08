#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .__version__ import __version__
from .grid import Grid
from .simulation import SimulationClass, CavitySimulation, CylinderSimulation, load_fields
from .solver import build_up_b, pressure_poisson, update_velocity
from .finite_differences import (
    backward_diff_x, backward_diff_y,
    forward_diff_x, forward_diff_y,
    centered_diff_x, centered_diff_y,
    laplacian_2d
)

__author__ = ['Filippo Di Ludovico']
__email__ = ['filippo.diludovico@studio.unibo.it']

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "Grid",
    "SimulationClass",
    "CavitySimulation",
    "CylinderSimulation",
    "load_fields",
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