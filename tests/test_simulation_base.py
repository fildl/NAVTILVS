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

    # Non-positive p_max_iter
    with pytest.raises(ValueError):
        SimulationClass(grid=grid, rho=1.0, nu=0.1, dt=0.001, p_max_iter=-1)

    # Non-positive p_tol
    with pytest.raises(ValueError):
        SimulationClass(grid=grid, rho=1.0, nu=0.1, dt=0.001, p_tol=-1e-4)

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
    assert sim.p_max_iter == 500
    assert sim.p_tol is None

    # Check custom Poisson parameters
    sim_custom = SimulationClass(grid=grid, rho=1.2, nu=0.01, dt=0.002,
                                 p_max_iter=300, p_tol=1e-3)
    assert sim_custom.p_max_iter == 300
    assert sim_custom.p_tol == 1e-3

    # Check that velocity and pressure fields are initialized correctly
    assert sim.u.shape == (10, 15)
    assert sim.v.shape == (10, 15)
    assert sim.p.shape == (10, 15)
    assert np.all(sim.u == 0.0)
    assert np.all(sim.v == 0.0)
    assert np.all(sim.p == 0.0)

def test_dynamic_dt_visc_limited():
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

def test_dynamic_dt_cfl_limited():
    """
    Test that when velocity is high, the time step :math:`dt` returns a CFL-limited time step.

    For an 11x11 square grid with :math:`l=1.0`, :math:`\\nu=0.1`, and velocity :math:`u=100.0`:
    * the CFL stability limit is :math:`dt_{CFL} = 0.0005` and
    * the viscous stability limit with safety factor 0.9 is :math:`dt_{visc} = 0.025`.

    We initialize the simulation with an unstable time step of :math:`dt=1.0` and verify that after one step,
    the computed :math:`dt` is reduced to the CFL stability limit.
    """
    
    grid = Grid(lx=1.0, ly=1.0, nx=11, ny=11)  # dx = 0.1, dy = 0.1
    sim = SimulationClass(grid=grid, rho=1.0, nu=0.1, dt=1.0)
    
    # Set velocity fields to high values to trigger CFL limit
    sim.u[:, :] = 100.0
    sim.v[:, :] = 0.0

    # Compute one step
    sim.step()
    
    # Expected CFL time step: 0.5 / (max_u/dx + max_v/dy) = 0.5 / (100.0 / 0.1) = 0.0005
    expected_dt_cfl = 0.5 / (100.0 / grid.dx)
    
    assert math.isclose(sim.dt, expected_dt_cfl, rel_tol=TOLERANCE)
    assert sim.dt < 1.0  # Should be less than dt_max

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

def test_simulation_solve_zero_steps():
    """
    Verify that solving for 0 steps works and leaves fields initialized to zero.
    """

    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)
    sim = SimulationClass(grid=grid, rho=1.0, nu=0.1, dt=0.001)
    
    u, v, p = sim.solve(nt=0)
    
    assert np.all(u == 0.0)
    assert np.all(v == 0.0)
    assert np.all(p == 0.0)

def test_simulation_solve_target_time():
    """
    Verify that solving to a target time stops exactly at t_end.
    """

    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)
    sim = SimulationClass(grid=grid, rho=1.0, nu=0.1, dt=0.001)
    
    # Run simulation to t_end = 0.005 seconds
    t_target = 0.005
    u, v, p = sim.solve(t_end=t_target)
    
    # Check that the simulated time is exactly t_target
    assert math.isclose(sim.t, t_target, abs_tol=TOLERANCE)

def test_simulation_solve_invalid_args():
    """
    Verify that calling solve with invalid combinations of t_end and nt raises ValueError.
    """
    
    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)
    sim = SimulationClass(grid=grid, rho=1.0, nu=0.1, dt=0.001)
    
    # Neither t_end nor nt specified
    with pytest.raises(ValueError):
        sim.solve()
        
    # Both t_end and nt specified
    with pytest.raises(ValueError):
        sim.solve(t_end=0.005, nt=10)

def test_simulation_solve_dt_not_too_small():
    """
    Verify that when self.t is extremely close to t_end,
    the solver does not take a small time step (dt < 1e-8) that would cause division by a number close to zero.
    """

    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)
    
    class SimulationTest(SimulationClass):
        """
        A class to store time steps
        """

        def __post_init__(self):
            super().__post_init__()
            # Create a list to store time steps
            self.dts = []

        def step(self, dt_override=None):
            super().step(dt_override)
            self.dts.append(self.dt)

    sim = SimulationTest(grid=grid, rho=1.0, nu=0.1, dt=0.0001)
    # Set t_end slightly larger than a multiple of dt by 1e-15, since 10 steps of dt=0.0001 reach exactly 0.001.
    t_target = 0.001 + 1e-15

    sim.solve(t_end=t_target)
    
    # Verify that no time step was smaller than the 1e-8 tolerance
    for dt in sim.dts:
        assert dt >= 1e-8