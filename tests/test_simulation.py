import pytest
import math
import numpy as np
from ns_solver import Grid
from ns_solver import SimulationClass, CavitySimulation
TOLERANCE = 5e-5

def test_simulation_invalid_inputs():
    """
    Check that SimulationClass raises errors for non-physical parameters.
    """
    
    # Create a grid
    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)
    
    # Negative density
    with pytest.raises(ValueError):
        SimulationClass(grid=grid, rho=-1.0, nu=0.1, dt=0.001)
        
    # Negative viscosity
    with pytest.raises(ValueError):
        SimulationClass(grid=grid, rho=1.0, nu=-0.1, dt=0.001)

    # Negative time step
    with pytest.raises(ValueError):
        SimulationClass(grid=grid, rho=1.0, nu=0.1, dt=-0.001)

def test_simulation_initialization():
    """
    Test if the base SimulationClass initializes grids and fields correctly.
    """

    # Create a grid and SimulationClass
    grid = Grid(lx=2.0, ly=1.0, nx=15, ny=10)
    sim = SimulationClass(grid=grid, rho=1.2, nu=0.01, dt=0.002)

    # Check that parameters are assigned correctly
    assert sim.grid.lx == 2.0
    assert sim.grid.ly == 1.0
    assert sim.grid.nx == 15
    assert sim.grid.ny == 10
    assert sim.rho == 1.2
    assert sim.nu == 0.01
    assert sim.dt == 0.002

    # Check that velocity and pressure fields are initialized correctly
    assert sim.u.shape == (10, 15)
    assert sim.v.shape == (10, 15)
    assert sim.p.shape == (10, 15)
    assert np.all(sim.u == 0.0)
    assert np.all(sim.v == 0.0)
    assert np.all(sim.p == 0.0)

def test_dynamic_time_stepping():
    """
    Test that the time step :math:`dt` is dynamically reduced if the input :math:`dt` is unstable.

    For a 128x128 square grid with :math:`\\l=1.0` and :math:`\\nu=0.1`:
    * the CFL stability limit is :math:`dt_{CFL} = 0.0019` and
    * the viscous stability limit with safety factor 0.9 is :math:`dt_{visc} \\approx 0.0001395`.

    We initialize the simulation with an unstable time step of :math:`dt=1.0` and verify that after one step,
    the computed :math:`dt` is reduced to the viscous stability limit.
    """

    # Create a 128x128 grid
    grid = Grid(lx=1.0, ly=1.0, nx=128, ny=128)
    
    # Initialize with dt = 1.0
    sim = CavitySimulation(grid=grid, rho=1.0, nu=0.1, dt=1.0)
    
    assert sim.dt == 1.0
    
    # Compute one step
    sim.step()
    
    # Verify that dt is reduced to the viscous stability limit
    dx = grid.dx
    dy = grid.dy
    expected_dt_visc = 0.9 * (dx**2 * dy**2) / (2.0 * sim.nu * (dx**2 + dy**2))
    
    assert math.isclose(sim.dt, expected_dt_visc, rel_tol=TOLERANCE)
    assert sim.dt > 0.0

def test_simulation_base_bc_identity():
    """
    Test if the SimulationClass boundary conditions do not change inputs.
    """

    # Create a grid and SimulationClass
    grid = Grid(lx=1.0, ly=1.0, nx=5, ny=5)
    sim = SimulationClass(grid=grid, rho=1.0, nu=0.1, dt=0.001)

    # Create fields
    dummy_p = np.random.rand(5, 5)
    dummy_u = np.random.rand(5, 5)
    dummy_v = np.random.rand(5, 5)

    # Fields should remain unchanged after applying boundary conditions
    assert np.array_equal(sim.pressure_bc(dummy_p), dummy_p)
    
    res_u, res_v = sim.velocity_bc(dummy_u, dummy_v)
    assert np.array_equal(res_u, dummy_u)
    assert np.array_equal(res_v, dummy_v)

def test_cavity_boundary_conditions():
    """
    Test if boundary conditions are applied correctly for the CavitySimulation.
    """

    # Create a grid and CavitySimulation
    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)
    sim = CavitySimulation(grid=grid, rho=1.0, nu=0.1, dt=0.001)

    # Set all fields to a non-zero values
    sim.u[:, :] = 5.0
    sim.v[:, :] = 5.0
    sim.p[:, :] = 5.0

    # Apply boundary conditions
    sim.u, sim.v = sim.velocity_bc(sim.u, sim.v)
    sim.p = sim.pressure_bc(sim.p)

    # Check that the velocity at the boundaries is zero
    assert np.all(sim.u[0, :] == 0.0)    # Bottom wall
    assert np.all(sim.u[:, 0] == 0.0)    # Left wall
    assert np.all(sim.u[:, -1] == 0.0)   # Right wall
    
    assert np.all(sim.v[0, :] == 0.0)    # Bottom wall
    assert np.all(sim.v[-1, :] == 0.0)   # Top lid
    assert np.all(sim.v[:, 0] == 0.0)    # Left wall
    assert np.all(sim.v[:, -1] == 0.0)   # Right wall

    # Top lid should have u = 1.0, corners should be 0 due to no-slip conditions
    assert np.all(sim.u[-1, 1:-1] == 1.0)
    assert sim.u[-1, 0] == 0.0
    assert sim.u[-1, -1] == 0.0

    # Check pressure
    assert np.all(sim.p[:, -1] == sim.p[:, -2])  # Right wall
    assert np.all(sim.p[0, :] == sim.p[1, :])    # Bottom wall
    assert np.all(sim.p[:, 0] == sim.p[:, 1])    # Left wall
    assert np.all(sim.p[-1, :] == 0.0)           # Top lid

def test_mass_conservation_coarse():
    """
    Physics validation test: verify that divergence of velocity tends to zero.

    This test checks that the numerical divergence of the velocity field
    :math:`(\frac{du}{dx} + \frac{dv}{dy})` remains close to zero on a coarse 20x20 grid.
    
    Due to the low spatial resolution and transient flow state, a wider 
    absolute tolerance of 0.1 is required to accommodate truncation 
    errors and geometric singularities at the top lid corners.
    A 4-node buffer layer (20% of the domain) is excluded to avoid boundary numerical noise.
    """

    # Create a grid and CavitySimulation
    grid = Grid(lx=1.0, ly=1.0, nx=20, ny=20)
    sim = CavitySimulation(grid=grid, rho=1.0, nu=0.1, dt=0.0005)

    # Solve for 50 time steps
    u, v, _ = sim.solve(nt=50)

    # Compute the divergence using central differences
    du_dx = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * grid.dx)
    dv_dy = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * grid.dy)
    divergence = du_dx + dv_dy

    # Divergence should be close to zero for incompressibility.
    # We exclude 4 nodes at the corners where localized high gradients occur.
    center_div = divergence[4:-4, 4:-4]
    
    # Check that divergence far from the boundaries is close to zero
    assert np.allclose(center_div, 0.0, atol=1e-1)

def test_mass_conservation_fine():
    """
    Physics validation test: verify that divergence of velocity tends to zero.

    This test demonstrates grid convergence. By refining the spatial resolution 
    to 128x128 grid points and using a smaller time step (dt=0.00005), the local 
    numerical divergence drops by two orders of magnitude compared to the coarse grid.
    
    An absolute tolerance of 0.001 is demanded in the core of the domain.
    A 10-node buffer layer (7% of the domain) is excluded to avoid boundary numerical noise.
    """

    # Create a grid and CavitySimulation
    grid = Grid(lx=1.0, ly=1.0, nx=128, ny=128)
    sim = CavitySimulation(grid=grid, rho=1.0, nu=0.1, dt=0.00005)

    # Solve for 500 time steps
    u, v, _ = sim.solve(nt=500)

    # Compute the divergence using central differences
    du_dx = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * grid.dx)
    dv_dy = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * grid.dy)
    divergence = du_dx + dv_dy

    # Divergence should be close to zero for incompressibility.
    # We exclude 4 nodes at the corners where localized high gradients occur.
    center_div = divergence[10:-10, 10:-10]
    
    # Check that divergence far from the boundaries is close to zero
    assert np.allclose(center_div, 0.0, atol=1e-3)