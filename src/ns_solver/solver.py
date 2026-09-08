#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Filippo Di Ludovico']
__email__ = ['filippo.diludovico@studio.unibo.it']

import numpy as np
from .finite_differences import (
    backward_diff_x, backward_diff_y,
    forward_diff_x, forward_diff_y,
    centered_diff_x, centered_diff_y,
    laplacian_2d
)

def build_up_b(dx : float,
               dy : float,
               u : np.ndarray,
               v : np.ndarray,
               rho : float,
               dt : float
               ) -> np.ndarray:
    """
    Compute the source term :math:`b` for the pressure Poisson equation.

    For the physical derivation and context, see :ref:`incompressible_fluid`.

    Parameters
    ----------
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    u : np.ndarray
        Velocity field in the x-direction with shape ``(ny, nx)``.
    v : np.ndarray
        Velocity field in the y-direction with shape ``(ny, nx)``.
    rho : float
        Fluid density.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        2D array for the source term :math:`b` with shape ``(ny, nx)``.
    """
    
    b = np.zeros_like(u)

    du_dx = centered_diff_x(u, dx)
    dv_dy = centered_diff_y(v, dy)
    du_dy = centered_diff_y(u, dy)
    dv_dx = centered_diff_x(v, dx)

    b[1:-1, 1:-1] = (rho *
                     (1 / dt * (du_dx + dv_dy) -
                      du_dx**2 -
                      2 * (du_dy * dv_dx) - 
                      dv_dy**2))

    return b

def pressure_poisson(p : np.ndarray,
                     dx : float,
                     dy : float,
                     b : np.ndarray,
                     boundary_conditions,
                     max_iter : int = 500
                     ) -> np.ndarray:
    """
    Solve the Poisson equation for pressure :math:`p`.

    Parameters
    ----------
    p : np.ndarray
        Initial pressure field with shape ``(ny, nx)``.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    b : np.ndarray
        Source term array with shape ``(ny, nx)``.
    boundary_conditions : callable
        Function ``bc(p) -> np.ndarray`` applying boundary conditions to the pressure field.
    max_iter : int, default=500
        Maximum number of iterations.

    Returns
    -------
    np.ndarray
        2D array for the solved pressure field with shape ``(ny, nx)``.
    """

    # Precompute constants for efficiency
    denom = 2 * (dx**2 + dy**2)
    const_x = dy**2 / denom
    const_y = dx**2 / denom
    const_b = dx**2 * dy**2 / denom

    # Pre-allocate array for pressure field and precompute source term
    pn = np.empty_like(p)
    b_term = b[1:-1, 1:-1] * const_b

    for _ in range(max_iter):

        # Copy current pressure field
        np.copyto(pn, p)

        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * const_x +
                         (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * const_y -
                         b_term)

        # Apply boundary conditions
        p = boundary_conditions(p)

    return p

def update_velocity(u : np.ndarray,
                    v : np.ndarray,
                    un : np.ndarray,
                    vn : np.ndarray,
                    dt : float,
                    dx : float,
                    dy : float,
                    p : np.ndarray,
                    rho : float,
                    nu : float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve momentum equation for both components.
    
    Discretizes the Navier-Stokes momentum equations in time.
    To ensure numerical stability, convective terms are discretized using a first-order Upwind scheme:
    backward differences where local velocity is positive, forward differences where local velocity is negative.
    Pressure gradient and diffusion terms are discretized using second-order central differences.
    
    Parameters
    ----------
    u : np.ndarray
        Velocity field in the x-direction.
    v : np.ndarray
        Velocity field in the y-direction.
    un : np.ndarray
        Velocity field in the x-direction at the previous time step.
    vn : np.ndarray
        Velocity field in the y-direction at the previous time step.
    dt : float
        Time step size.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    p : np.ndarray
        Pressure field.
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated velocity fields (u, v).

    """

    # Extract velocity at internal nodes
    u_mid = un[1:-1, 1:-1]
    v_mid = vn[1:-1, 1:-1]

    # Upwind derivatives for u
    du_dx = np.where(u_mid > 0, backward_diff_x(un, dx), forward_diff_x(un, dx))
    du_dy = np.where(v_mid > 0, backward_diff_y(un, dy), forward_diff_y(un, dy))

    # Upwind derivatives for v
    dv_dx = np.where(u_mid > 0, backward_diff_x(vn, dx), forward_diff_x(vn, dx))
    dv_dy = np.where(v_mid > 0, backward_diff_y(vn, dy), forward_diff_y(vn, dy))

    # Update u component
    u[1:-1, 1:-1] = (u_mid -
                     u_mid * dt * du_dx -
                     v_mid * dt * du_dy -
                     dt / rho * centered_diff_x(p, dx) +
                     nu * dt * laplacian_2d(un, dx, dy))
    # Update v component
    v[1:-1, 1:-1] = (v_mid -
                     u_mid * dt * dv_dx -
                     v_mid * dt * dv_dy -
                     dt / rho * centered_diff_y(p, dy) +
                     nu * dt * laplacian_2d(vn, dx, dy))

    return u, v