import numpy as np

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
        Velocity field in the x-direction.
    v : np.ndarray
        Velocity field in the y-direction.
    rho : float
        Fluid density.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        2D array for the source term b.
    """
    b = np.zeros_like(u)

    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

def pressure_poisson(p : np.ndarray,
                     dx : float,
                     dy : float,
                     b : np.ndarray,
                     boundary_conditions,
                     max_iter : int = 500
                     ) -> np.ndarray:
    """
    Solves the Poisson equation for pressure :math:`p`.

    Parameters
    ----------
    p : np.ndarray
        pressure field.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    b : np.ndarray
        Source term.
    boundary_conditions : callable
        Function to apply boundary conditions to the pressure field.
    max_iter : int, optional
        Maximum number of iterations (default is 500).

    Returns
    -------
    np.ndarray
        2D array for the solved pressure field.
    """

    # Precompute constants for efficiency
    denom = 2 * (dx**2 + dy**2)
    const_x = dy**2 / denom
    const_y = dx**2 / denom
    const_b = dx**2 * dy**2 / denom

    # Pre-allocate array for pressure field
    pn = np.empty_like(p)

    for _ in range(max_iter):

        # Copy current pressure field
        np.copyto(pn, p)

        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * const_x +
                         (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * const_y -
                         b[1:-1,1:-1] * const_b)

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

    Discretizes the Navier-Stokes momentum equations using backward differences for convection
    and central differences for diffusion.

    
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

    # u component
    u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    # v component
    v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    return u, v