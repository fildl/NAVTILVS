import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from ns_solver import Grid
from ns_solver import centered_diff_x, centered_diff_y

def plot_stream(u : np.ndarray,
                v : np.ndarray,
                p : np.ndarray,
                grid : Grid,
                rho : float,
                nu : float,
                t : float = None,
                reynolds : float = None,
                save_path : str = None
                ) -> None:
    """
    Plot the streamlines of the velocity field and the pressure contours for Cavity flow.

    This function generates a 2D visualization of the cavity simulation results,
    displaying the pressure field as a contour plot and the velocity field as streamlines.

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
    t : float, optional
        Elapsed simulation time in seconds.
    reynolds : float, optional
        Reynolds number of the simulation. If None, it is calculated as (1.0 * grid.lx) / nu.
    save_path : str, optional
        If specified, saves the figure to this path.
    """
    
    X, Y = grid.X, grid.Y

    if reynolds is None:
        reynolds = (1.0 * grid.lx) / nu

    title_str = f"Lid-Driven Cavity Flow ($Re = {reynolds:.0f}$)"
    if t is not None:
        title_str += f", $t = {t:.2f}\\text{{ s}}$"

    plt.figure(figsize=(7, 6), dpi=100)
    plt.gca().set_aspect('equal')

    # Plot pressure contours
    cf = plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar(cf, label='Pressure (Pa)', fraction=0.046, pad=0.04)
    plt.contour(X, Y, p, cmap=cm.viridis)

    # Plot velocity streamlines
    plt.streamplot(X, Y, u, v, density=1.2)
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(0, grid.lx)
    plt.ylim(0, grid.ly)
    plt.title(title_str)
    plt.tight_layout()

    # Save figure if requested
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')

def plot_cylinder_flow(u: np.ndarray,
                       v: np.ndarray,
                       p: np.ndarray,
                       grid: Grid,
                       obstacle_mask: np.ndarray,
                       mode: str = 'vorticity',
                       t: float = None,
                       reynolds: float = None,
                       save_path: str = None) -> None:
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
    t : float, optional
        Elapsed simulation time in seconds.
    reynolds : float, optional
        Reynolds number of the simulation.
    save_path : str, optional
        If specified, saves the figure to this path.

    Raises
    ------
    ValueError
        If an invalid mode is provided.
    """

    X, Y = grid.X, grid.Y

    # Base title with optional Reynolds and time
    sub_title = ""
    if reynolds is not None:
        sub_title += f" ($Re = {reynolds:.1f}$"
        if t is not None:
            sub_title += f", $t = {t:.2f}\\text{{ s}}$)"
        else:
            sub_title += ")"
    elif t is not None:
        sub_title += f" ($t = {t:.2f}\\text{{ s}}$)"

    plt.figure(figsize=(12, 4.5), dpi=100)
    plt.gca().set_aspect('equal')

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
        if limit == 0.0:
            limit = 1.0
            
        plt.contourf(X, Y, vorticity, levels=100, cmap='coolwarm', vmin=-limit, vmax=limit)
        plt.colorbar(label='Vorticity (rad/s)')
        plt.title(f'Vorticity Field{sub_title}')

    elif mode == 'velocity':
        # Compute velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)

        # Mask the obstacle region to avoid plotting it
        vel_mag[obstacle_mask] = np.nan
        
        plt.contourf(X, Y, vel_mag, levels=100, cmap='viridis')
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.streamplot(X, Y, u, v, color='black', linewidth=0.8, density=1.5)
        plt.title(f'Velocity Magnitude Field{sub_title}')

    elif mode == 'pressure':
        # Compute pressure field
        p_masked = p.copy()

        # Mask the obstacle region to avoid plotting it
        p_masked[obstacle_mask] = np.nan
        
        plt.contourf(X, Y, p_masked, levels=100, cmap='coolwarm')
        plt.colorbar(label='Pressure (Pa)')
        plt.title(f'Pressure Field{sub_title}')

    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'vorticity', 'velocity', or 'pressure'.")

    # Fill the obstacle region
    plt.contourf(X, Y, obstacle_mask.astype(float), levels=[0.5, 1.5], colors=['#333333'])

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(0, grid.lx)
    plt.ylim(0, grid.ly)
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    # Save figure if requested
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')