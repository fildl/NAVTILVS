import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ns_solver import Grid
from ns_solver import centered_diff_x, centered_diff_y

def plot_stream(u : np.ndarray,
                v : np.ndarray,
                p : np.ndarray,
                grid : Grid,
                rho : float,
                nu : float
                ) -> None:
    """
    Plot the streamlines of the velocity field and the pressure contours.

    This function generates a 2D visualization of the simulation results,
    displaying the pressure field as a contour plot and
    the velocity field as streamlines.

    Parameters
    ----------
    u : np.ndarray
        Velocity field in the x-direction.
    v : np.ndarray
        Velocity field in the y-direction.
    p : np.ndarray
        Pressure field.
    grid : Grid
        Spatial grid of the simulation.
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity.
    """
    
    X, Y = grid.X, grid.Y

    plt.figure(figsize=(11, 7), dpi=100)

    # Plot pressure contours
    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar(label='Pressure')
    plt.contour(X, Y, p, cmap=cm.viridis)

    # Plot velocity streamlines
    plt.streamplot(X, Y, u, v)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Final State\n$\\rho = {rho}$, $\\nu = {nu}$")

    # Show plot
    plt.show()

def plot_cylinder_flow(u: np.ndarray,
                       v: np.ndarray,
                       p: np.ndarray,
                       grid: Grid,
                       obstacle_mask: np.ndarray,
                       mode: str = 'vorticity') -> None:
    """
    Plot the flow field around a cylinder.

    This function generates a visualization of the pressure, velocity,
    or vorticity field around the cylinder.

    Parameters
    ----------
    u : np.ndarray
        Velocity field in the x-direction.
    v : np.ndarray
        Velocity field in the y-direction.
    p : np.ndarray
        Pressure field.
    grid : Grid
        Spatial grid of the simulation.
    obstacle_mask : np.ndarray
        Boolean mask where True represents the cylinder obstacle.
    mode : str, default='vorticity'
        The field to plot. Options are:
        - 'vorticity': Plots the vorticity field :math:`(\frac{dv}{dx} - \frac{du}{dy})`.
        - 'velocity': Plots the velocity magnitude field (sqrt(u^2 + v^2)).
        - 'pressure': Plots the pressure field.
    """

    X, Y = grid.X, grid.Y

    plt.figure(figsize=(12, 6), dpi=100)

    if mode == 'vorticity':
        # Compute vorticity using central differences
        dv_dx = centered_diff_x(v, grid.dx)
        du_dy = centered_diff_y(u, grid.dy)
        
        vorticity = np.zeros_like(u)
        vorticity[1:-1, 1:-1] = dv_dx - du_dy
        
        # Mask the obstacle region to avoid plotting it
        vorticity[obstacle_mask] = np.nan
        
        # Set a symmetric colormap
        limit = max(abs(np.nanmin(vorticity)), abs(np.nanmax(vorticity)))
        # Set a minimum limit to avoid error if the flow is at rest
        if limit == 0.0:
            limit = 1.0
            
        plt.contourf(X, Y, vorticity, levels=100, cmap='coolwarm', vmin=-limit, vmax=limit)
        plt.colorbar(label='Vorticity (rad/s)')
        plt.title('Vorticity Field')

    elif mode == 'velocity':
        # Compute velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)

        # Mask the obstacle region to avoid plotting it
        vel_mag[obstacle_mask] = np.nan
        
        plt.contourf(X, Y, vel_mag, levels=100, cmap='viridis')
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.streamplot(X, Y, u, v, color='black', linewidth=0.8, density=1.5)
        plt.title('Velocity Magnitude Field')

    elif mode == 'pressure':
        # Compute pressure field
        p_masked = p.copy()

        # Mask the obstacle region to avoid plotting it
        p_masked[obstacle_mask] = np.nan
        
        plt.contourf(X, Y, p_masked, levels=100, cmap='coolwarm')
        plt.colorbar(label='Pressure (Pa)')
        plt.title('Pressure Field')

    # Fill the obstacle region
    plt.contourf(X, Y, obstacle_mask.astype(float), levels=[0.5, 1.5], colors=['#333333'])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()