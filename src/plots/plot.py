#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Filippo Di Ludovico']
__email__ = ['filippo.diludovico@studio.unibo.it']

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ns_solver import Grid
from ns_solver import centered_diff_x, centered_diff_y

def _compute_vorticity_limits(vorticity: np.ndarray,
                              vlim: float | tuple[float, float] | None,
                              clip_percentile: float | None) -> tuple[float, str]:
    """
    Compute symmetric bounds [-limit, limit] and colorbar extension for vorticity.

    Parameters
    ----------
    vorticity : np.ndarray
        2D array of vorticity values.
    vlim : float or tuple of (float, float) or None
        Explicit scalar limit for symmetric scaling [-vlim, vlim].
        If a tuple (vmin, vmax) is provided, max(|vmin|, |vmax|) is used.
    clip_percentile : float or None
        Percentile threshold for colormap scaling. If None or >= 100, uses max magnitude.

    Returns
    -------
    limit : float
        Positive symmetric bound.
    extend : str
        'both' if clipping/saturation occurs, otherwise 'neither'.
    """

    valid_vort = np.abs(vorticity[~np.isnan(vorticity)])
    if vlim is not None:
        if isinstance(vlim, (tuple, list)):
            limit = float(max(abs(vlim[0]), abs(vlim[1])))
        else:
            limit = float(abs(vlim))
        extend = 'both'
    elif clip_percentile is not None and clip_percentile < 100.0:
        limit = float(np.percentile(valid_vort, clip_percentile)) if valid_vort.size > 0 else 1.0
        extend = 'both'
    else:
        limit = float(np.max(valid_vort)) if valid_vort.size > 0 else 1.0
        extend = 'neither'

    if limit == 0.0:
        limit = 1.0

    return limit, extend

def _compute_pressure_limits(p: np.ndarray,
                             vlim: float | tuple[float, float] | None,
                             clip_percentile: float | None) -> tuple[tuple[float, float], str]:
    """
    Compute bounds (p_min, p_max) and colorbar extension for the pressure field.

    Parameters
    ----------
    p : np.ndarray
        2D array of pressure values.
    vlim : float or tuple of (float, float) or None
        Explicit limits. If scalar, symmetric [-vlim, vlim]. If tuple, (p_min, p_max).
    clip_percentile : float or None
        Percentile threshold. If specified, computes (100 - pct) and pct percentiles.

    Returns
    -------
    bounds : tuple[float, float]
        (p_min, p_max) for the colormap.
    extend : str
        'both' if clipping/saturation occurs, otherwise 'neither'.
    """

    valid_p = p[~np.isnan(p)]
    if vlim is not None:
        if isinstance(vlim, (tuple, list)):
            p_min, p_max = float(vlim[0]), float(vlim[1])
        else:
            lim = float(abs(vlim))
            p_min, p_max = -lim, lim
        extend = 'both'
    elif clip_percentile is not None and clip_percentile < 100.0:
        if valid_p.size > 0:
            p_min = float(np.percentile(valid_p, 100.0 - clip_percentile))
            p_max = float(np.percentile(valid_p, clip_percentile))
            if p_min == p_max:
                p_min, p_max = p_min - 1.0, p_max + 1.0
        else:
            p_min, p_max = -1.0, 1.0
        extend = 'both'
    else:
        if valid_p.size > 0:
            p_min = float(np.min(valid_p))
            p_max = float(np.max(valid_p))
            if p_min == p_max:
                p_min, p_max = p_min - 1.0, p_max + 1.0
        else:
            p_min, p_max = -1.0, 1.0
        extend = 'neither'

    return (p_min, p_max), extend

def plot_cavity_flow(u: np.ndarray,
                     v: np.ndarray,
                     p: np.ndarray,
                     grid: Grid,
                     mode: str = 'velocity',
                     nu: float = None,
                     t: float = None,
                     reynolds: float = None,
                     quiver_density: int = 20,
                     quiver_scale: float = None,
                     vlim: float | tuple[float, float] = None,
                     clip_percentile: float = 98.0,
                     show_streamlines: bool = True,
                     show_axes: bool = True,
                     save_path: str = None) -> None:
    """
    Plot the flow field inside a lid-driven cavity.

    Render cavity velocity magnitude overlaid with quiver vectors, pressure contours with streamlines, and symmetric vorticity fields.

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
        - 'velocity': Velocity magnitude field with 'turbo' colormap and directional quiver vectors.
        - 'pressure': Velocity streamlines overlaid on pressure contour field ('turbo' colormap).
        - 'vorticity': Vorticity field with symmetric 'coolwarm' colormap.
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
    vlim : float or tuple of (float, float), optional
        Explicit colormap limits for 'vorticity' (scalar symmetric bound) or 'pressure'
        (scalar symmetric or (vmin, vmax) tuple). If specified, overrides `clip_percentile`.
    clip_percentile : float, optional, default=98.0
        Percentile threshold for colormap scaling in 'vorticity' and 'pressure' modes.
        If set to None or 100.0, uses the unclipped full dynamic range.
    show_streamlines : bool, default=True
        Whether to overlay velocity streamlines on the pressure field in 'pressure' mode.
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

    mode_titles = {
        'velocity': 'Velocity Field',
        'pressure': 'Streamlines & Pressure Field',
        'vorticity': 'Vorticity Field'
    }
    if mode not in mode_titles:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'velocity', 'pressure', or 'vorticity'.")

    field_name = mode_titles[mode]
    title_str = f"Lid-Driven Cavity - {field_name}"
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

    elif mode == 'pressure':
        (p_min, p_max), extend_mode = _compute_pressure_limits(p, vlim, clip_percentile)
        levels = np.linspace(p_min, p_max, 101)
        p_clipped = np.clip(p, p_min, p_max)
        cf = plt.contourf(X, Y, p_clipped, levels=levels, alpha=0.5, cmap='turbo')
        plt.colorbar(cf, label='Pressure (Pa)', fraction=0.046, pad=0.04, extend=extend_mode)
        plt.contour(X, Y, p_clipped, levels=15, cmap='turbo', linewidths=0.8)
        if show_streamlines:
            plt.streamplot(X, Y, u, v, density=1.2, color='white')

    elif mode == 'vorticity':
        dv_dx = centered_diff_x(v, grid.dx)
        du_dy = centered_diff_y(u, grid.dy)
        vorticity = np.zeros_like(u)
        vorticity[1:-1, 1:-1] = dv_dx - du_dy

        limit, extend_mode = _compute_vorticity_limits(vorticity, vlim, clip_percentile)
        levels = np.linspace(-limit, limit, 101)
        cf = plt.contourf(X, Y, np.clip(vorticity, -limit, limit), levels=levels, cmap='coolwarm')
        plt.colorbar(cf, label='Vorticity (rad/s)', fraction=0.046, pad=0.04, extend=extend_mode)

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
                       vlim: float | tuple[float, float] = None,
                       clip_percentile: float = 98.0,
                       show_streamlines: bool = True,
                       show_axes: bool = True,
                       save_path: str = None) -> None:
    r"""
    Plot the flow field around a cylinder.

    Render channel flow past the cylinder obstacle with velocity streamlines,
    pressure contour fields, and vorticity fields, while masking the cylinder region.

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
        Visualization mode:
        - 'vorticity': Vorticity field with symmetric 'coolwarm' colormap and masked cylinder obstacle.
        - 'velocity': Velocity magnitude field with 'turbo' colormap, overlaid with velocity streamlines and masked cylinder obstacle.
        - 'pressure': Velocity streamlines overlaid on pressure contour field ('turbo' colormap) and masked cylinder obstacle.
    t : float, optional
        Elapsed simulation time in seconds.
    reynolds : float, optional
        Reynolds number of the simulation.
    vlim : float or tuple of (float, float), optional
        Explicit colormap limits for 'vorticity' (scalar symmetric bound) or 'pressure'
        (scalar symmetric or (vmin, vmax) tuple). If specified, overrides `clip_percentile`.
    clip_percentile : float, optional, default=98.0
        Percentile threshold for colormap scaling in 'vorticity' and 'pressure' modes.
        If set to None or 100.0, uses the unclipped full dynamic range.
    show_streamlines : bool, default=True
        Whether to overlay velocity streamlines on the pressure field in 'pressure' mode.
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

    mode_titles = {
        'vorticity': 'Vorticity Field',
        'velocity': 'Velocity Field',
        'pressure': 'Streamlines & Pressure Field'
    }
    if mode not in mode_titles:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'vorticity', 'velocity', or 'pressure'.")

    field_name = mode_titles[mode]
    title_str = f"Cylinder Flow - {field_name}"
    if reynolds is not None:
        title_str += f" ($Re = {reynolds:.1f}$"
        if t is not None:
            title_str += f", $t = {t:.2f}\\text{{ s}}$)"
        else:
            title_str += ")"
    elif t is not None:
        title_str += f" ($t = {t:.2f}\\text{{ s}}$)"

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
        limit, extend_mode = _compute_vorticity_limits(vorticity, vlim, clip_percentile)
        levels = np.linspace(-limit, limit, 101)
        cf = plt.contourf(X, Y, np.clip(vorticity, -limit, limit), levels=levels, cmap='coolwarm')
        plt.colorbar(cf, label='Vorticity (rad/s)', extend=extend_mode)

    elif mode == 'velocity':
        # Compute velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)

        # Mask the obstacle region to avoid plotting it
        vel_mag[obstacle_mask] = np.nan
        
        cf = plt.contourf(X, Y, vel_mag, levels=100, cmap='turbo')
        plt.colorbar(cf, label='Velocity Magnitude (m/s)')
        plt.streamplot(X, Y, u, v, color='white', linewidth=0.8, density=1.5)

    elif mode == 'pressure':
        # Compute pressure field
        p_masked = p.copy()

        # Mask the obstacle region to avoid plotting it
        p_masked[obstacle_mask] = np.nan
        (p_min, p_max), extend_mode = _compute_pressure_limits(p_masked, vlim, clip_percentile)
        levels = np.linspace(p_min, p_max, 101)
        p_clipped = np.clip(p_masked, p_min, p_max)
        cf = plt.contourf(X, Y, p_clipped, levels=levels, alpha=0.5, cmap='turbo')
        plt.colorbar(cf, label='Pressure (Pa)', extend=extend_mode)
        plt.contour(X, Y, p_clipped, levels=15, cmap='turbo', linewidths=0.8)
        if show_streamlines:
            plt.streamplot(X, Y, u, v, color='white', linewidth=0.8, density=1.5)

    # Fill the obstacle region
    plt.contourf(X, Y, obstacle_mask.astype(float), levels=[0.5, 1.5], colors=['#333333'])

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