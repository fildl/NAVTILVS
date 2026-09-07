#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Filippo Di Ludovico']
__email__ = ['filippo.diludovico@studio.unibo.it']

from dataclasses import dataclass
import numpy as np
from .grid import Grid
from .solver import build_up_b, pressure_poisson, update_velocity

@dataclass
class SimulationClass:
    """
    Base class managing the time integration of the 2D incompressible Navier-Stokes equations.
    
    Parameters
    ----------
    grid : Grid
        Spatial grid of the simulation.
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity.
    dt : float
        Initial time step size.

    Attributes
    ----------
    dt_max : float
        Maximum allowable time step size set from initial dt.
    t : float
        Elapsed physical simulation time in seconds.
    u : np.ndarray
        Velocity field in the x-direction with shape ``(ny, nx)``.
    v : np.ndarray
        Velocity field in the y-direction with shape ``(ny, nx)``.
    p : np.ndarray
        Pressure field with shape ``(ny, nx)``.
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

        # Initialize simulation time
        self.t = 0.0 

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
        # Viscous safety factor (typically 0.9)
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

    def step(self,
             dt_override : float = None):
        """
        Perform a single time step of the Navier-Stokes solver.

        This function updates the velocity and pressure fields by solving the momentum
        and pressure Poisson equations.
        It dynamically calculates the time step :math:`dt`.
        If the time step exceeds the target end time, it is cut to match the target time.

        Parameters
        ----------
        dt_override : float, optional
            Time step used instead of the dynamically computed one,
            for matching simulation end-time.
        """

        if dt_override is None:
            self.dt = self.compute_dynamic_dt()
        else:
            self.dt = dt_override

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
        
        # Boundary conditions for velocity fields
        self.u, self.v = self.velocity_bc(self.u, self.v)

        # Update simulation time
        self.t += self.dt
        
    def solve(self,
              t_end : float = None,
              nt : int = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the Navier-Stokes equations.

        This function runs the simulation for a target simulated time t_end
        or for a specific number of time steps nt.
        The time step size is dynamically adapted during the simulation.

        Parameters
        ----------
        t_end : float, optional
            Target time to reach (in seconds).
        nt : int, optional
            Number of time steps to simulate.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the final fields (u, v, p).

        Raises
        ------
        ValueError
            If neither or both parameters are specified.
        """

        if t_end is None and nt is None:
            raise ValueError("You must specify at least one of 't_end' or 'nt'.")
        if t_end is not None and nt is not None:
            raise ValueError("You cannot specify both 't_end' and 'nt'.")
        
        if nt is not None:
            for _ in range(nt):
                self.step()
        else:
            # Simulate until reaching t_end
            # We include a 1e-8 tolerance to avoid too small dt
            while self.t < t_end - 1e-8:
                dt = self.compute_dynamic_dt()
                
                # If the next time step exceeds the target time, cut it
                if self.t + dt > t_end:
                    dt = t_end - self.t
                
                self.step(dt_override=dt)

        return self.u, self.v, self.p
    
@dataclass
class CavitySimulation(SimulationClass):
    """
    Simulation solver for the classical 2D Lid-Driven Cavity benchmark problem.

    Solves recirculating viscous flow in a closed cavity driven by a
    moving top lid, with Dirichlet zero-velocity boundaries on walls and
    Neumann zero-gradient pressure conditions.

    Parameters
    ----------
    grid : Grid
        Spatial grid discretization of the cavity domain.
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity.
    dt : float
        Initial time step size.
    u_lid : float, default=1.0
        Tangential velocity of the moving top lid.
    """

    # Define the velocity of the moving lid
    u_lid: float = 1.0

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
        and a constant velocity :math:`u = u_{\\text{lid}}` on the top wall.

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
        u[-1, :] = self.u_lid
        
        # No-slip conditions on other walls
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0

        return u, v

    @property
    def reynolds_number(self) -> float:
        """
        Compute the Reynolds number for the Lid-Driven Cavity flow.

        Returns
        -------
        float
            Reynolds number :math:`Re = u_{\\text{lid}} l_x / \\nu`.
        """

        characteristic_length = self.grid.lx
        return (self.u_lid * characteristic_length) / self.nu

@dataclass
class CylinderSimulation(SimulationClass):
    """
    Simulation solver for 2D channel flow past an immersed circular cylinder.

    Extends the base Navier-Stokes solver to model flow separation, wake recirculation,
    and vortex shedding (Kármán vortex street) past a solid circular obstacle.

    Parameters
    ----------
    grid : Grid
        Spatial grid discretization of the channel domain.
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity.
    dt : float
        Initial time step size.
    cylinder_center : tuple[float, float]
        Physical coordinates ``(xc, yc)`` of the cylinder's center.
    cylinder_radius : float
        Physical radius of the circular cylinder.
    u_inlet : float, default=1.0
        Uniform horizontal velocity imposed at the channel inlet.

    Attributes
    ----------
    obstacle_mask : np.ndarray
        Boolean mask array of shape ``(ny, nx)`` where True represents solid obstacle cells.
    """

    cylinder_center: tuple[float, float]
    cylinder_radius: float
    u_inlet: float = 1.0

    def __post_init__(self):
        r"""
        Initialize the cylinder simulation by creating the obstacle mask and 
        pre-computing the nearest fluid nodes for pressure boundary conditions.

        To apply the pressure boundary condition :math:`\frac{\partial p}{\partial n} = 0`
        on the surface of the cylinder, each obstacle node is mapped to its closest fluid neighbor.
        During the pressure Poisson iterations, the pressure on the obstacle cells is updated
        by copying the value of their nearest fluid neighbors. 
        
        If a node is equidistant from multiple fluid cells (e.g., at the center of symmetry),
        the algorithm selects one of them.
        This approximation introduces negligible local errors that vanish as the grid resolution increases.
        """

        # Initialize velocity and pressure fields
        super().__post_init__()

        # Define a mask for the cylinder obstacle
        xc, yc = self.cylinder_center
        r = self.cylinder_radius
        # We include a 1e-9 tolerance to be sure that boundaries node are included in the mask
        self.obstacle_mask = (self.grid.X - xc)**2 + (self.grid.Y - yc)**2 <= r**2 + 1e-9

        # Define coordinates for obstacle and fluid
        self.obs_y, self.obs_x = np.where(self.obstacle_mask)
        self.fluid_y, self.fluid_x = np.where(~self.obstacle_mask)

        # Convert fluid indexes into spatial coordinates in order to find minimum distance 
        fluid_coords = np.column_stack((self.fluid_x * self.grid.dx,
                                        self.fluid_y * self.grid.dy))
        
        nearest_fluid_y = []
        nearest_fluid_x = []
        
        # For each obstacle node, compute the distance with fluid nodes to find the closest one
        for oy, ox in zip(self.obs_y, self.obs_x):
            op_phys = np.array([ox * self.grid.dx, oy * self.grid.dy])
            dists = np.sum((fluid_coords - op_phys)**2, axis=1)
            min_idx = np.argmin(dists)
            nearest_fluid_y.append(self.fluid_y[min_idx])
            nearest_fluid_x.append(self.fluid_x[min_idx])

        self.nearest_fluid_y = np.array(nearest_fluid_y)
        self.nearest_fluid_x = np.array(nearest_fluid_x)

    def pressure_bc(self,
                    p: np.ndarray) -> np.ndarray:
        """
        Apply pressure boundary conditions for channel flow around a cylinder.

        It imposes zero gradient on top, bottom, and left (inlet) walls, 
        constant reference pressure p = 0 at the right boundary (outlet),
        and zero normal gradient on the cylinder surface.

        Parameters
        ----------
        p : np.ndarray
            Pressure field matrix.

        Returns
        -------
        np.ndarray
            Pressure field with boundary conditions applied.
        """

        # Top and bottom walls: zero normal gradient
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        
        # Inlet (left): zero normal gradient
        p[:, 0] = p[:, 1]
        
        # Outlet (right): constant pressure
        p[:, -1] = 0.0
        
        # Cylinder obstacle: copy pressure from the pre-computed nearest fluid cells
        if len(self.obs_y) > 0:
            p[self.obs_y, self.obs_x] = p[self.nearest_fluid_y, self.nearest_fluid_x]
            
        return p
    
    def velocity_bc(self,
                    u: np.ndarray,
                    v: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply velocity boundary conditions for channel flow around a cylinder.

        It imposes no-slip conditions (u = 0, v = 0) on the top/bottom walls 
        and the cylinder surface, a uniform velocity profile at the inlet,
        and convective/zero-gradient outlet conditions.

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

        # Inlet (left): uniform flow
        u[:, 0] = self.u_inlet
        v[:, 0] = 0.0

        # Outlet (right): zero gradient (flow continues out of the channel)
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]

        # Top and bottom walls: no-slip
        u[0, :] = 0.0
        u[-1, :] = 0.0
        v[0, :] = 0.0
        v[-1, :] = 0.0
        
        # Cylinder obstacle: no-slip
        u[self.obstacle_mask] = 0.0
        v[self.obstacle_mask] = 0.0
        
        return u, v

    @property
    def reynolds_number(self) -> float:
        """
        Compute the Reynolds number for the flow around the cylinder.

        Returns
        -------
        float
            Reynolds number :math:`Re = u_{\\text{inlet}} (2 r) / \\nu`.
        """
        
        characteristic_length = 2.0 * self.cylinder_radius
        return (self.u_inlet * characteristic_length) / self.nu