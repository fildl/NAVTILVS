import numpy as np
from dataclasses import dataclass

@dataclass
class Grid:
    """
    This class contains the spatial 2D domian of the simulation.

    It handles spatial discretization and initialization of velocity and 
    pressure fields.

    Parameters
    ----------
    lx : float
        Length of the domain in the x-direction.
    ly : float
        Length of the domain in the y-direction.
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
        return self.lx / (self.nx - 1)
    
    @property
    def dy(self) -> float:
        return self.ly / (self.ny - 1)