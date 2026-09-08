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
from plots import plot_cavity_flow, plot_cylinder_flow

def run_cavity(reynolds: float = 100.0,
               nx: int = 128,
               ny: int = 128,
               t_end: float = 2.5,
               output_dir: str = "imgs") -> None:
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
    print("=" * 65)

    grid = Grid(lx=lx, ly=ly, nx=nx, ny=ny)
    sim = CavitySimulation(grid=grid, rho=rho, nu=nu, dt=0.001)

    print("Running Navier-Stokes solver...")
    u, v, p = sim.solve(t_end=t_end)
    print(f"Simulation completed! Simulated physical time: {sim.t:.3f} s\n")

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
                 output_dir: str = "imgs") -> None:
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
    print("=" * 65)

    grid = Grid(lx=lx, ly=ly, nx=nx, ny=ny)
    sim = CylinderSimulation(grid=grid, rho=rho, nu=nu, dt=0.002,
                             cylinder_center=center, cylinder_radius=radius,
                             u_inlet=u_inlet)

    print("Running Navier-Stokes solver...")
    u, v, p = sim.solve(t_end=t_end)
    print(f"Simulation completed! Simulated physical time: {sim.t:.3f} s\n")

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
        default="imgs",
        help="Directory where output plots will be saved (default: imgs)",
    )
    return parser.parse_args()

def main():
    """
    CLI entry point: parse arguments and execute the selected fluid simulation.
    """
    
    args = parse_args()

    if args.sim == "cavity":
        nx = args.nx if args.nx is not None else 128
        ny = args.ny if args.ny is not None else 128
        t_end = args.tend if args.tend is not None else 2.5
        run_cavity(reynolds=args.re, nx=nx, ny=ny, t_end=t_end, output_dir=args.output_dir)

    elif args.sim == "cylinder":
        nx = args.nx if args.nx is not None else 256
        ny = args.ny if args.ny is not None else 64
        t_end = args.tend if args.tend is not None else 3.5
        run_cylinder(reynolds=args.re, nx=nx, ny=ny, t_end=t_end, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
