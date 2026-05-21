import numpy as np

def build_up_b(dx, dy,
               u, v,
               rho,
               dt
               ):
    """
    Compute the source term :math:`b` for the pressure Poisson equation.

    For the physical derivation and context, see :ref:`incompressible_fluid`.

    Parameters
    ----------
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    rho : float
        Fluid density.
    dt : float
        Time step size.

    Returns
    -------

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

def pressure_poisson(p, dx, dy, b, boundary_conditions, max_iter = 500):
    """
    Solves the Poisson equation for pressure :math:`p`.

    Parameters
    ----------

    Returns
    -------
    """

    for _ in range(max_iter):

        pn = p.copy()

        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        # boundary conditions for cavity flow
        p = boundary_conditions(p)

    return p

def cavity_flow_bc(p):
    p[:, -1] = p[:, -2]
    p[0, :] = p[1, :]
    p[:, 0] = p[:, 1]
    p[-1, :] = 0

    return p

def update_velocity(u, v,
                    un, vn,
                    dt,
                    dx, dy,
                    p,
                    rho,
                    nu):
    """
    Solve momentum equation for both components.

    Parameters
    ----------

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

def step(dx, dy, u, v, rho, nu, dt):

    un = u.copy()
    vn = v.copy()

    p = np.zeros_like(u)

    b = build_up_b(dx, dy, u, v, rho, dt)
    
    p = pressure_poisson(p, dx, dy, b,
                         boundary_conditions=cavity_flow_bc)

    u, v = update_velocity(u, v,
                           un, vn,
                           dt,
                           dx, dy,
                           p,
                           rho,
                           nu)

    # boundary conditions
    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = 1  # set velocity on cavity lid equal to 1
    v[0, :]  = 0
    v[-1, :] = 0
    v[:, 0]  = 0
    v[:, -1] = 0

    return u, v, p

def NS_solver(u, v, p,
              dx, dy,
              nx, ny,
              rho,
              nu,
              nt, dt
              ):

    for n in range(nt):

        u, v, p = step(dx, dy, u, v, rho, nu, dt)

    return u, v, p