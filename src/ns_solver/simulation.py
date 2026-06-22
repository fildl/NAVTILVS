from dataclasses import dataclass
import numpy as np
from .grid import Grid
from .solver import build_up_b, pressure_poisson, update_velocity

@dataclass
class SimulationClass:
    """
    This class handles the simulation of the Navier-Stokes equations.
    
    Parameters
    ----------
    grid : Grid
        Spatial grid of the simulation.
    rho: float
        Fluid density.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.
    """

    grid : Grid
    rho  : float
    nu   : float
    dt   : float

    def __post_init__(self):
        """
        Initialize velocity and pressure fields and validate inputs.
        """

        # Check that parameters are valid
        if self.rho <= 0:
            raise ValueError("Density must be strictly positive.")
        if self.nu <= 0:
            raise ValueError("Viscosity must be strictly positive.")
        if self.dt <= 0:
            raise ValueError("Time step (dt) must be strictly positive.")
        
        # Store the input time step for compute dynamic time stepping
        self.dt_max = self.dt

        # Initialize velocity and pressure fields
        self.u = np.zeros((self.grid.ny, self.grid.nx))
        self.v = np.zeros((self.grid.ny, self.grid.nx))
        self.p = np.zeros((self.grid.ny, self.grid.nx))

    def compute_dynamic_dt(self) -> float:
        """
        Compute the time step :math:`dt` based on CFL and viscous limits.

        For the derivation and context, see :ref:`numerical_stability`.

        Returns
        -------
        float
            The dynamically computed stable time step size.
        """

        dx = self.grid.dx
        dy = self.grid.dy

        # 1. Convective Limit (CFL condition)
        # Define maximum velocity in the domain
        max_u = np.max(np.abs(self.u))
        max_v = np.max(np.abs(self.v))

        # Avoid division by zero if the fluid is completely at rest
        if max_u + max_v == 0.0:
            dt_cfl = float('inf')
        else:
            dt_cfl = 0.5 / (max_u / dx + max_v / dy)

        # 2. Viscous Limit (Diffusive limit)
        # C_visc è il fattore di sicurezza (tipicamente 0.9)
        denom_visc = 2.0 * self.nu * (dx**2 + dy**2)
        dt_visc = 0.9 * (dx**2 * dy**2) / denom_visc

        # Define time step as the minimum of the two limits
        stable_dt = min(dt_cfl, dt_visc)

        return min(stable_dt, self.dt_max)

    def pressure_bc(self,
                    p : np.ndarray
                    ) -> np.ndarray:
        """
        Apply pressure boundary conditions.

        Parameters
        ----------
        p : np.ndarray
            Pressure field matrix.

        Returns
        -------
        np.ndarray
            Pressure field with boundary conditions applied.
        """
        return p

    def velocity_bc(self,
                    u : np.ndarray,
                    v : np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply velocity boundary conditions.

        Parameters
        ----------
        u : np.ndarray
            Velocity field in the x-direction.
        v : np.ndarray
            Velocity field in the y-direction.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple containing velocity fields with boundary conditions applied.
        """
        return u, v

    def step(self):
        """
        Solve NS equations for one time step using dynamic time-stepping.
        """

        # Compute dynamic time step
        self.dt = self.compute_dynamic_dt()

        dx = self.grid.dx
        dy = self.grid.dy

        un = self.u.copy()
        vn = self.v.copy()

        # Compute source term :math:`b`
        b = build_up_b(dx, dy,
                       self.u, self.v,
                       self.rho,
                       self.dt
                       )
        
        # Solve Poisson equation for pressure
        self.p = pressure_poisson(self.p,
                                  dx, dy,
                                  b,
                                  self.pressure_bc)
        
        # Update velocity
        self.u, self.v = update_velocity(self.u, self.v,
                                         un, vn,
                                         self.dt,
                                         dx, dy,
                                         self.p,
                                         self.rho,
                                         self.nu)
        
        # Boundary conditions for velocity fileds
        self.u, self.v = self.velocity_bc(self.u, self.v)
        
    def solve(self,
              nt : int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the NS equations for :math:`nt` time steps.

        Parameters
        ----------
        nt : int
            Number of time steps to simulate.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Final fields for u, v, and p.
        """

        for _ in range(nt):
            self.step()

        return self.u, self.v, self.p
    
@dataclass
class CavitySimulation(SimulationClass):
    """
    This class handles the simulation of the Lid-Driven Cavity problem.
    """

    def pressure_bc(self,
                    p : np.ndarray
                    ) -> np.ndarray:
        """
        Apply pressure boundary conditions for cavity flow.

        It imposes zero gradient conditions on left, right and bottom walls
        and a :math:`p = 0` condition on the top wall.

        Parameters
        ----------
        p : np.ndarray
            Pressure field matrix.

        Returns
        -------
        np.ndarray
            Pressure field with boundary conditions applied.
        """
        
        p[:, -1] = p[:, -2]  # Right wall
        p[0, :] = p[1, :]    # Bottom wall
        p[:, 0] = p[:, 1]    # Left wall
        p[-1, :] = 0.0       # Top lid
        
        return p
    
    def velocity_bc(self,
                    u : np.ndarray,
                    v : np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply velocity boundary conditions.

        It imposes no-slip conditions :math:`u = 0` and :math:`v = 0` on left, right and bottom walls
        and a costant velocity :math:`u = 1` on the top wall.


        Parameters
        ----------
        u : np.ndarray
            Velocity field in the x-direction.
        v : np.ndarray
            Velocity field in the y-direction.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple containing velocity fields with boundary conditions applied.
        """

        # Moving top lid
        u[-1, :] = 1
        
        # No-slip conditions on other walls
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0

        return u, v