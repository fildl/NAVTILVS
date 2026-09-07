#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Filippo Di Ludovico']
__email__ = ['filippo.diludovico@studio.unibo.it']

import numpy as np
from dataclasses import dataclass

@dataclass
class Grid:
    """
    This class represents the spatial 2D domain of the simulation.

    It handles spatial discretization and coordinate mesh generation
    for Cartesian grids.

    Parameters
    ----------
    lx : float
        Length of the physical domain in the x-direction.
    ly : float
        Length of the physical domain in the y-direction.
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    """

    lx : float
    ly : float
    nx : int
    ny : int

    def __post_init__(self):
        """
        Validate that grid lengths and number of grid points are valid.
        """

        if not isinstance(self.nx, int) or not isinstance(self.ny, int):
            raise TypeError("Number of grid points must be an integer.")

        if self.lx <= 0 or self.ly <= 0:
            raise ValueError("Grid lengths must be positive (> 0).")

        if self.nx < 2 or self.ny < 2:
            raise ValueError("Number of grid points must be >= 2.")            

    @property
    def dx(self) -> float:
        """
        Compute the grid spacing in the x-direction.

        Returns
        -------
        float
            Grid spacing :math:`\\Delta x = l_x / (n_x - 1)`.
        """

        return self.lx / (self.nx - 1)
    
    @property
    def dy(self) -> float:
        """
        Compute the grid spacing in the y-direction.

        Returns
        -------
        float
            Grid spacing :math:`\\Delta y = l_y / (n_y - 1)`.
        """

        return self.ly / (self.ny - 1)
    
    @property
    def X(self) -> np.ndarray:
        """
        2D meshgrid array of x-coordinates.

        Returns
        -------
        np.ndarray
            Coordinate array :math:`X` with shape ``(ny, nx)``.
        """

        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        X, _ = np.meshgrid(x, y)
        return X
    
    @property
    def Y(self) -> np.ndarray:
        """
        2D meshgrid array of y-coordinates.

        Returns
        -------
        np.ndarray
            Coordinate array :math:`Y` with shape ``(ny, nx)``.
        """

        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        _, Y = np.meshgrid(x, y)
        return Y