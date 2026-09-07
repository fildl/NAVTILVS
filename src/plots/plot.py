import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from ns_solver import Grid
from ns_solver import centered_diff_x, centered_diff_y

def plot_cavity_flow(u: np.ndarray,
                     v: np.ndarray,
                     p: np.ndarray,
                     grid: Grid,
                     mode: str = 'velocity',
                     rho: float = None,
                     nu: float = None,
                     t: float = None,
                     reynolds: float = None,
                     quiver_density: int = 20,
                     quiver_scale: float = None,
                     show_axes: bool = True,
                     save_path: str = None) -> None:
    """
    Plot the flow field inside a lid-driven cavity.

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
    mode : str, default='velocity'
        Visualization mode:
        - 'velocity': Velocity magnitude field with 'turbo' colormap and white quiver vectors.
        - 'stream': Pressure contours overlaid with velocity streamlines.
        - 'vorticity': Vorticity field with symmetric 'coolwarm' colormap.
        - 'pressure': Pressure field with 'coolwarm' colormap.
    rho : float, optional
        Fluid density.
    nu : float, optional
        Kinematic viscosity.
    t : float, optional
        Elapsed simulation time in seconds.
    reynolds : float, optional
        Reynolds number. If None and nu is provided, calculated as (1.0 * grid.lx) / nu.
    quiver_density : int, default=20
        Approximate number of arrows per dimension for the quiver plot in 'velocity' mode.
    quiver_scale : float, optional
        Scaling factor for quiver arrow lengths. If None, matplotlib auto-scales.
    show_axes : bool, default=True
        Whether to show axis ticks, labels, and title. If False, displays a clean plot frame.
    save_path : str, optional
        If specified, saves the figure to this path at 300 DPI.

    Raises
    ------
    ValueError
        If an invalid mode is provided.
    """

    X, Y = grid.X, grid.Y

    if reynolds is None and nu is not None:
        reynolds = (1.0 * grid.lx) / nu

    title_str = "Lid-Driven Cavity Flow"
    if reynolds is not None:
        title_str += f" ($Re = {reynolds:.0f}$"
        if t is not None:
            title_str += f", $t = {t:.2f}\\text{{ s}}$)"
        else:
            title_str += ")"
    elif t is not None:
        title_str += f" ($t = {t:.2f}\\text{{ s}}$)"

    plt.figure(figsize=(7, 6), dpi=100)
    plt.gca().set_aspect('equal')

    if mode == 'velocity':
        vel_mag = np.sqrt(u**2 + v**2)
        cf = plt.contourf(X, Y, vel_mag, levels=100, cmap='turbo')
        plt.colorbar(cf, label='Velocity Magnitude (m/s)', fraction=0.046, pad=0.04)

        # Quiver arrows subsampled
        step_x = max(1, grid.nx // quiver_density)
        step_y = max(1, grid.ny // quiver_density)
        plt.quiver(X[::step_y, ::step_x], Y[::step_y, ::step_x],
                   u[::step_y, ::step_x], v[::step_y, ::step_x],
                   color='white', pivot='mid', scale=quiver_scale)

    elif mode == 'stream':
        cf = plt.contourf(X, Y, p, alpha=0.5, cmap='turbo')
        plt.colorbar(cf, label='Pressure (Pa)', fraction=0.046, pad=0.04)
        plt.contour(X, Y, p, cmap='turbo')
        plt.streamplot(X, Y, u, v, density=1.2, color='white')

    elif mode == 'vorticity':
        dv_dx = centered_diff_x(v, grid.dx)
        du_dy = centered_diff_y(u, grid.dy)
        vorticity = np.zeros_like(u)
        vorticity[1:-1, 1:-1] = dv_dx - du_dy
        limit = max(abs(np.nanmin(vorticity)), abs(np.nanmax(vorticity)))
        if limit == 0.0:
            limit = 1.0
        cf = plt.contourf(X, Y, vorticity, levels=100, cmap='coolwarm', vmin=-limit, vmax=limit)
        plt.colorbar(cf, label='Vorticity (rad/s)', fraction=0.046, pad=0.04)

    elif mode == 'pressure':
        cf = plt.contourf(X, Y, p, levels=100, cmap='coolwarm')
        plt.colorbar(cf, label='Pressure (Pa)', fraction=0.046, pad=0.04)

    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'velocity', 'stream', 'vorticity', or 'pressure'.")

    plt.xlim(0, grid.lx)
    plt.ylim(0, grid.ly)

    if show_axes:
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(title_str)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

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
        
        plt.contourf(X, Y, vel_mag, levels=100, cmap='turbo')
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.streamplot(X, Y, u, v, color='white', linewidth=0.8, density=1.5)
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

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')