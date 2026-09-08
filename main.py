#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAVTILVS: NAVier-Stokes Two-dimensional IncompressibLe Visual Solver
Command-Line Interface (CLI).
"""

__author__ = ['Filippo Di Ludovico']
__email__ = ['filippo.diludovico@studio.unibo.it']

import argparse
from pathlib import Path
from ns_solver import Grid, CavitySimulation, CylinderSimulation
from plots import (
    plot_cavity_flow,
    plot_cylinder_flow,
    plot_saved_fields,
    plot_checkpoint_series,
)

def run_cavity(reynolds: float = 100.0,
               nx: int = 128,
               ny: int = 128,
               t_end: float = 2.5,
               output_dir: str = "imgs",
               save_data: bool = False,
               save_interval: float = None) -> None:
    """
    Run the Lid-Driven Cavity simulation and generate plots.

    Parameters
    ----------
    reynolds : float, default=100.0
        Target Reynolds number :math:`Re = u_{\\text{lid}} l_x / \\nu`.
    nx : int, default=128
        Number of grid points along the x-direction.
    ny : int, default=128
        Number of grid points along the y-direction.
    t_end : float, default=2.5
        Target physical simulation time in seconds.
    output_dir : str, default="imgs"
        Directory where generated diagnostic figures will be saved.
    save_data : bool, default=False
        Whether to save final simulation fields to a compressed NumPy .npz archive.
    save_interval : float, optional
        Physical time interval in seconds for saving intermediate field checkpoints.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lx, ly = 1.0, 1.0
    u_lid = 1.0
    nu = (u_lid * lx) / reynolds
    rho = 1.0

    print("=" * 65)
    print(" NAVTILVS - Lid-Driven Cavity Benchmark")
    print(f" Grid: {nx} x {ny} | Domain: {lx} m x {ly} m")
    print(f" Reynolds number: Re = {reynolds:.1f} | Kinematic Viscosity: nu = {nu:.5f} m^2/s")
    print(f" Target Simulation Time: t_end = {t_end:.2f} s")
    if save_interval is not None:
        print(f" Checkpoint Interval: every {save_interval:.3f} s")
    print("=" * 65)

    grid = Grid(lx=lx, ly=ly, nx=nx, ny=ny)
    sim = CavitySimulation(grid=grid, rho=rho, nu=nu, dt=0.001)

    print("Running Navier-Stokes solver...")
    chk_dir = out_path / "checkpoints" if save_interval is not None else None
    u, v, p = sim.solve(t_end=t_end, save_interval=save_interval, save_dir=chk_dir)
    print(f"Simulation completed! Simulated physical time: {sim.t:.3f} s\n")

    # Save final fields if requested
    if save_data and save_interval is None:
        save_file = out_path / "cavity_fields.npz"
        sim.save_fields(save_file)
        print(f"  -> Saved simulation fields -> {save_file}")
    elif save_interval is not None:
        print(f"  -> Periodic field checkpoints saved to '{chk_dir}/'")

    # Generate diagnostic plots
    modes = [
        ('velocity', 'cavity_velocity.png'),
        ('pressure', 'cavity_pressure.png'),
        ('vorticity', 'cavity_vorticity.png'),
    ]

    for mode, filename in modes:
        save_file = str(out_path / filename)
        print(f"  -> Generating plot [{mode}] -> {save_file}")
        plot_cavity_flow(u=u, v=v, p=p, grid=grid, mode=mode,
                         nu=nu, t=sim.t, reynolds=sim.reynolds_number,
                         save_path=save_file)

    print(f"\nAll cavity plots saved successfully to '{output_dir}/'!")

def run_cylinder(reynolds: float = 100.0,
                 nx: int = 256,
                 ny: int = 64,
                 t_end: float = 3.5,
                 output_dir: str = "imgs",
                 save_data: bool = False,
                 save_interval: float = None) -> None:
    """
    Run the Flow Past a Cylinder simulation and generate plots.

    Parameters
    ----------
    reynolds : float, default=100.0
        Target Reynolds number :math:`Re = u_{\\text{inlet}} (2 r) / \\nu`.
    nx : int, default=256
        Number of grid points along the x-direction.
    ny : int, default=64
        Number of grid points along the y-direction.
    t_end : float, default=3.5
        Target physical simulation time in seconds.
    output_dir : str, default="imgs"
        Directory where generated diagnostic figures will be saved.
    save_data : bool, default=False
        Whether to save final simulation fields to a compressed NumPy .npz archive.
    save_interval : float, optional
        Physical time interval in seconds for saving intermediate field checkpoints.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lx, ly = 2.0, 0.5
    u_inlet = 1.0
    center = (0.4, 0.225)
    radius = 0.05
    diameter = 2.0 * radius  # 0.1 m
    nu = (u_inlet * diameter) / reynolds
    rho = 1.0

    print("=" * 65)
    print(" NAVTILVS - Flow Around a Circular Cylinder")
    print(f" Grid: {nx} x {ny} | Domain: {lx} m x {ly} m")
    print(f" Obstacle: Center {center}, Diameter D = {diameter:.2f} m")
    print(f" Reynolds number: Re = {reynolds:.1f} | Kinematic Viscosity: nu = {nu:.5f} m^2/s")
    print(f" Target Simulation Time: t_end = {t_end:.2f} s")
    if save_interval is not None:
        print(f" Checkpoint Interval: every {save_interval:.3f} s")
    print("=" * 65)

    grid = Grid(lx=lx, ly=ly, nx=nx, ny=ny)
    sim = CylinderSimulation(grid=grid, rho=rho, nu=nu, dt=0.002,
                             cylinder_center=center, cylinder_radius=radius,
                             u_inlet=u_inlet)

    print("Running Navier-Stokes solver...")
    chk_dir = out_path / "checkpoints" if save_interval is not None else None
    u, v, p = sim.solve(t_end=t_end, save_interval=save_interval, save_dir=chk_dir)
    print(f"Simulation completed! Simulated physical time: {sim.t:.3f} s\n")

    # Save final fields if requested
    if save_data and save_interval is None:
        save_file = out_path / "cylinder_fields.npz"
        sim.save_fields(save_file)
        print(f"  -> Saved simulation fields -> {save_file}")
    elif save_interval is not None:
        print(f"  -> Periodic field checkpoints saved to '{chk_dir}/'")

    # Generate all three diagnostic plots
    modes = [
        ('vorticity', 'cylinder_vorticity.png'),
        ('velocity', 'cylinder_velocity.png'),
        ('pressure', 'cylinder_pressure.png'),
    ]

    for mode, filename in modes:
        save_file = str(out_path / filename)
        print(f"  -> Generating plot [{mode}] -> {save_file}")
        plot_cylinder_flow(u=u, v=v, p=p, grid=grid,
                           obstacle_mask=sim.obstacle_mask, mode=mode,
                           t=sim.t, reynolds=sim.reynolds_number,
                           save_path=save_file)

    print(f"\nAll cylinder plots saved successfully to '{output_dir}/'!")

def run_postprocess_file(filepath: str,
                         output_dir: str = None) -> None:
    """
    Generate flow field plots from a single saved .npz archive.

    Parameters
    ----------
    filepath : str
        Path to the saved .npz archive file.
    output_dir : str, optional
        Directory where generated plots will be saved. Defaults to source file directory.
    """

    print("=" * 65)
    print(" NAVTILVS - Post-Processing Single File")
    print(f" Source File: {filepath}")
    if output_dir is not None:
        print(f" Target Directory: {output_dir}")
    print("=" * 65)

    saved_plots = plot_saved_fields(filepath=filepath, output_dir=output_dir)
    print("Plots generated successfully:")
    for p in saved_plots:
        print(f"  -> {p}")
    print()

def run_postprocess_checkpoints(checkpoint_dir: str,
                                output_dir: str = None,
                                step: int = 1) -> None:
    """
    Generate flow field plots for all .npz checkpoints in a directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing .npz checkpoints.
    output_dir : str, optional
        Target directory for saved plot images. Defaults to 'checkpoint_dir/plots'.
    step : int, default=1
        Sampling stride.
    """

    print("=" * 65)
    print(" NAVTILVS - Post-Processing Checkpoints Batch Mode")
    print(f" Checkpoint Directory: {checkpoint_dir}")
    print(f" Sampling Stride (step): {step}")
    if output_dir is not None:
        print(f" Target Directory: {output_dir}")
    print("=" * 65)

    saved_plots = plot_checkpoint_series(checkpoint_dir=checkpoint_dir,
                                         output_dir=output_dir,
                                         step=step)
    print(f"Batch plotting completed! Total figures generated: {len(saved_plots)}\n")

def parse_args():
    """
    Parse command-line interface arguments for NAVTILVS simulations.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="NAVTILVS: 2D Incompressible Navier-Stokes Fluid Flow Solver"
    )
    parser.add_argument(
        "--sim",
        type=str,
        choices=["cavity", "cylinder"],
        default="cavity",
        help="Simulation case to run (default: cavity)",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=100.0,
        help="Target Reynolds number (default: 100.0)",
    )
    parser.add_argument(
        "--tend",
        type=float,
        default=None,
        help="Final simulation physical time in seconds (default: 2.5 for cavity, 3.5 for cylinder)",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help="Grid nodes in x-direction (default: 128 for cavity, 256 for cylinder)",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=None,
        help="Grid nodes in y-direction (default: 128 for cavity, 64 for cylinder)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where output plots will be saved (default: imgs for simulations)",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        default=False,
        help="Save final simulation fields (u, v, p) to a compressed NumPy .npz archive",
    )
    parser.add_argument(
        "--save-interval",
        type=float,
        default=None,
        help="Physical time interval in seconds for saving intermediate field snapshots (default: None)",
    )
    parser.add_argument(
        "--plot-data",
        type=str,
        default=None,
        help="Post-processing: generate plots from a single saved .npz field archive",
    )
    parser.add_argument(
        "--plot-checkpoints",
        type=str,
        default=None,
        help="Post-processing: generate plots for all .npz checkpoints in a directory",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Sampling stride for checkpoint batch plotting (default: 1)",
    )
    return parser.parse_args()

def main():
    """
    CLI entry point: parse arguments and execute the selected fluid simulation or post-processing.
    """
    
    args = parse_args()

    # Post-processing: single .npz file
    if args.plot_data is not None:
        run_postprocess_file(filepath=args.plot_data, output_dir=args.output_dir)
        return

    # Post-processing: checkpoints directory
    if args.plot_checkpoints is not None:
        run_postprocess_checkpoints(checkpoint_dir=args.plot_checkpoints,
                                    output_dir=args.output_dir,
                                    step=args.step)
        return

    # Standard simulation execution
    out_dir = args.output_dir if args.output_dir is not None else "imgs"

    if args.sim == "cavity":
        nx = args.nx if args.nx is not None else 128
        ny = args.ny if args.ny is not None else 128
        t_end = args.tend if args.tend is not None else 2.5
        run_cavity(reynolds=args.re, nx=nx, ny=ny, t_end=t_end,
                   output_dir=out_dir,
                   save_data=args.save_data,
                   save_interval=args.save_interval)

    elif args.sim == "cylinder":
        nx = args.nx if args.nx is not None else 256
        ny = args.ny if args.ny is not None else 64
        t_end = args.tend if args.tend is not None else 3.5
        run_cylinder(reynolds=args.re, nx=nx, ny=ny, t_end=t_end,
                   output_dir=out_dir,
                   save_data=args.save_data,
                   save_interval=args.save_interval)

if __name__ == "__main__":
    main()
