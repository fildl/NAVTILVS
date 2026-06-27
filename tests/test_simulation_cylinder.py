import numpy as np
from ns_solver import Grid
from ns_solver import CylinderSimulation

TOLERANCE = 5e-5

def test_cylinder_simulation_init():
    """
    Test that CylinderSimulation correctly initializes.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=32, ny=16)
    
    # Define an obstacle centered at (xc = 0.5, yc = 0.25) with radius 0.1
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1,
        u_inlet=1.5
    )
    
    # Check that parameters are correctly initialized
    assert sim.cylinder_center == (0.5, 0.25)
    assert sim.cylinder_radius == 0.1
    assert sim.u_inlet == 1.5

def test_cylinder_simulation_mask_dimensions():
    """
    Test that CylinderSimulation correctly creates the obstacle mask.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=32, ny=16)
    
    # Define an obstacle centered at (xc = 0.5, yc = 0.25) with radius 0.1
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )

    # Check the dimension of the obstacle mask is the same as the grid
    assert sim.obstacle_mask.shape == (16, 32)

def test_cylinder_simulation_mask_center():
    """
    Test that the center of the cilinder has the correct value in the obstacle mask.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=21, ny=21)
    
    # Define an obstacle centered at (xc = 0.5, yc = 0.25) with radius 0.1
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )
    
    # The central point of the cyclinder (0.5, 0.25) must be True
    # The grid point closest to the central point should be (Y = 10, X = 5)
    center_x = 5    # 0.5  / 0.1,   where 0.1 is grid.dx
    center_y = 10   # 0.25 / 0.025, where 0.025 is grid.dy
    
    assert sim.obstacle_mask[center_y, center_x] == True

def test_cylinder_simulation_mask_inside():
    """
    Test that grid point inside the cilinder have the correct value in the obstacle mask.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=21, ny=21)
    
    # Define an obstacle centered at (xc = 0.5, yc = 0.25) with radius 0.1
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )
    
    # The grid point closest to the central point should be (Y = 10, X = 5)
    center_x = 5    # 0.5  / 0.1,   where 0.1 is grid.dx
    center_y = 10   # 0.25 / 0.025, where 0.025 is grid.dy

    # With these Grid properties the Cylinder should cover 3 cells in x direction and 10 cells in y direction
    
    # Vertical boundaries
    assert sim.obstacle_mask[center_y - 4, center_x] == True
    assert sim.obstacle_mask[center_y + 4, center_x] == True

    # Horizontal boundaries
    assert sim.obstacle_mask[center_y, center_x - 1] == True
    assert sim.obstacle_mask[center_y, center_x + 1] == True
    

def test_cylinder_simulation_mask_outside():
    """
    Test that grid point outside the cilinder have the correct value in the obstacle mask.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=21, ny=21)
    
    # Define an obstacle centered at (xc = 0.5, yc = 0.25) with radius 0.1
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )
    
    # The grid point closest to the central point should be (Y = 10, X = 5)
    center_x = 5    # 0.5  / 0.1,   where 0.1 is grid.dx
    center_y = 10   # 0.25 / 0.025, where 0.025 is grid.dy

    # Test one more grid point respect to previous test
    
    # Vertical boundaries
    assert sim.obstacle_mask[center_y - 5, center_x] == False
    assert sim.obstacle_mask[center_y + 5, center_x] == False

    # Horizontal boundaries
    assert sim.obstacle_mask[center_y, center_x - 2] == False
    assert sim.obstacle_mask[center_y, center_x + 2] == False

def test_cylinder_simulation_mask_grid_corners():
    """
    Test that grid corners have the correct value in the obstacle mask.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=21, ny=21)
    
    # Define an obstacle centered at (xc = 0.5, yc = 0.25) with radius 0.1
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )

    assert sim.obstacle_mask[0, 0] == False
    assert sim.obstacle_mask[0, -1] == False
    assert sim.obstacle_mask[-1, 0] == False
    assert sim.obstacle_mask[-1, -1] == False

def test_cylinder_simulation_obstacle_coordinates_length():
    """
    Test that obstacle coordinates array have the same lenght as the number of obstacle nodes.
    """
    
    grid = Grid(lx=2.0, ly=0.5, nx=21, ny=21)
    
    # Define an obstacle centered at (xc = 0.5, yc = 0.25) with radius 0.1
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )
    
    # Check that coordinates array have the same lenght as the number of obstacle nodes
    n_obstacle_nodes = np.sum(sim.obstacle_mask) # It sums nodes == True
    assert len(sim.obs_x) == n_obstacle_nodes
    assert len(sim.obs_y) == n_obstacle_nodes

def test_cylinder_simulation_obstacle_coordinates_values():
    """
    Test that obstacle coordinates array have the correct values.
    """
    
    grid = Grid(lx=2.0, ly=1.0, nx=21, ny=21)
    
    # Define an obstacle centered at (xc = 0.5, yc = 0.25) with radius 0.1
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )
    
    # With these Grid and Cylinder properties
    # the x coordinates should be [0.4, 0.5, 0.6]
    # the y coordinates should be [0.15, 0.2, 0.25, 0.3, 0.35]
    # Calculating the indexes at which the cylinder should be masked,
    # the x indexes should be [5, 5, 4, 5, 6, 5, 5]
    # the y indexes should be [3, 4, 5, 5, 5, 6, 7]
    expected_obs_x = np.array([5, 5, 4, 5, 6, 5, 5]) 
    expected_obs_y = np.array([3, 4, 5, 5, 5, 6, 7])
                               
    assert np.array_equal(sim.obs_x, expected_obs_x)
    assert np.array_equal(sim.obs_y, expected_obs_y)
    
def test_cylinder_simulation_partition_and_disjointness():
    """
    Test that the grid is correctly partitioned into obstacle and fluid nodes, covering the entire grid.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=21, ny=21)

    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )
    
    total_nodes = grid.nx * grid.ny
    n_obs = len(sim.obs_x)
    n_fluid = len(sim.fluid_x)
    
    # Check that grid is 100% covered
    assert n_obs + n_fluid == total_nodes
    
    # Check that there are no overlaps
    obs_set = set(zip(sim.obs_y, sim.obs_x))
    fluid_set = set(zip(sim.fluid_y, sim.fluid_x))
    assert obs_set.isdisjoint(fluid_set)

def test_nearest_fluid_shapes():
    """
    Test that the computed nearest fluid nodes have valid shapes.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=21, ny=21)

    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )
    
    # Check that nearest_fluid_x and nearest_fluid_y have the same dimensions of the obstacle indexes 
    assert sim.nearest_fluid_y.shape == sim.obs_y.shape
    assert sim.nearest_fluid_x.shape == sim.obs_x.shape

def test_nearest_fluid_are_inside_fluid():
    """
    Test that the computed nearest fluid nodes reside inside the fluid.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=21, ny=21)

    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.1
    )
    
    # Check that mapped nodes are actually fluid nodes
    mapped_nodes_in_obstacle = sim.obstacle_mask[sim.nearest_fluid_y, sim.nearest_fluid_x]
    assert np.all(mapped_nodes_in_obstacle == False)

def test_nearest_fluid_coordinates_simple():
    """
    Test the nearest fluid mapping on a 3x3 grid with a single central obstacle node.

    This test verifies that a single central obstacle node surrounded by fluid
    is mapped to one of its 4 immediate neighbors.
    """

    grid = Grid(lx=2.0, ly=2.0, nx=3, ny=3) # dx = 1.0, dy = 1.0
    
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(1.0, 1.0),
        cylinder_radius=0.1
    )

    # Check there is only one obstacle node
    assert len(sim.obs_y) == 1

    # Check its coordinates
    assert sim.obs_y[0] == 1
    assert sim.obs_x[0] == 1
    
    mapped_y = sim.nearest_fluid_y[0]
    mapped_x = sim.nearest_fluid_x[0]
    
    # The nearest fluid node to (1, 1) should be one among:
    # (1, 0), (1, 2), (0, 1) or (2, 1).
    valid_neighbors = {(1, 0), (1, 2), (0, 1), (2, 1)}
    
    assert (mapped_y, mapped_x) in valid_neighbors

def test_nearest_fluid_coordinates():
    """
    Test the nearest fluid mapping on a 10x10 grid.

    This test validates the mapping correctness under a resolved cylinder:
    1. The central node (5, 5) must map to one of the 8 equidistant fluid nodes 
       lying at diagonal locations (distance = sqrt(5) cells).
    2. An edge node (5, 3) must map to one of its 3 immediate fluid neighbors 
       (distance = 1 cell).

    """
    
    grid = Grid(lx=1.0, ly=1.0, nx=11, ny=11) # dx = 0.1, dy = 0.1
    
    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.5),
        cylinder_radius=0.2
    )
    
    # Check there are 13 obstacle nodes
    assert len(sim.obs_x) == 13
    assert len(sim.obs_y) == 13

    # Check there are 13 nearest fluid nodes
    assert len(sim.nearest_fluid_x) == 13
    assert len(sim.nearest_fluid_y) == 13

    # Find index in nearest_fluid_x and nearest_fluid_y corresponding to the center (5, 5)
    idx_center = np.where((sim.obs_y == 5) & (sim.obs_x == 5))[0][0]
    mapped_center_y = sim.nearest_fluid_y[idx_center]
    mapped_center_x = sim.nearest_fluid_x[idx_center]
    
    # The center (5, 5) must be mapped to one of these 8 nodes
    valid_center_neighbors = {
        (3, 4), (3, 6), (7, 4), (7, 6),
        (4, 3), (4, 7), (6, 3), (6, 7)
        }
    assert (mapped_center_y, mapped_center_x) in valid_center_neighbors

    # Find index in nearest_fluid_x and nearest_fluid_y corresponding to the a node on the left (5, 3)
    idx_edge = np.where((sim.obs_y == 5) & (sim.obs_x == 3))[0][0]
    mapped_edge_y = sim.nearest_fluid_y[idx_edge]
    mapped_edge_x = sim.nearest_fluid_x[idx_edge]
    
    # This left ndoe (5, 3) must be mapped to one of these 3 nodes
    valid_edge_neighbors = {(4, 3), (6, 3), (5, 2)}
    assert (mapped_edge_y, mapped_edge_x) in valid_edge_neighbors

def test_cylinder_simulation_pressure_bc():
    """
    Test if pressure boundary conditions are correctly applied.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=32, ny=32)

    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.08
    )

    # Initalize p with random values to check the effect of BC
    dummy_p = np.random.rand(grid.ny, grid.nx)

    p_bc = sim.pressure_bc(dummy_p)

    # Inlet (left): zero normal gradient
    assert np.allclose(p_bc[:, 0], p_bc[:, 1], rtol=TOLERANCE)

    # Top and bottom walls: zero normal gradient
    assert np.allclose(p_bc[0, :], p_bc[1, :], rtol=TOLERANCE)
    assert np.allclose(p_bc[-1, :], p_bc[-2, :], rtol=TOLERANCE)

    # Outlet (right): constant pressure
    assert np.all(p_bc[:, -1] == 0.0)

    # Cylinder obstacle: copy pressure from the pre-computed nearest fluid cells
    expected_obstacle_p = p_bc[sim.nearest_fluid_y, sim.nearest_fluid_x]
    assert np.allclose(p_bc[sim.obs_y, sim.obs_x], expected_obstacle_p, rtol=TOLERANCE)

def test_cylinder_simulation_velocity_bc():
    """
    Test if velocity boundary conditions are correctly applied.
    """

    grid = Grid(lx=2.0, ly=0.5, nx=32, ny=32)

    sim = CylinderSimulation(
        grid=grid,
        rho=1.0,
        nu=0.1,
        dt=0.001,
        cylinder_center=(0.5, 0.25),
        cylinder_radius=0.08,
        u_inlet=1.5
    )

    # Initalize u and v with random values to check the effect of BC
    dummy_u = np.random.rand(grid.ny, grid.nx)
    dummy_v = np.random.rand(grid.ny, grid.nx)

    u_bc, v_bc = sim.velocity_bc(dummy_u, dummy_v)

    # Top and bottom walls (No-Slip -> 0)
    assert np.all(u_bc[0, :] == 0.0)
    assert np.all(u_bc[-1, :] == 0.0)
    assert np.all(v_bc[0, :] == 0.0)
    assert np.all(v_bc[-1, :] == 0.0)

    # Inlet (left): uniform flow on internal nodes, no-slip on corner boundary nodes
    assert np.allclose(u_bc[1:-1, 0], sim.u_inlet, rtol=TOLERANCE)
    assert u_bc[0, 0] == 0.0
    assert u_bc[-1, 0] == 0.0
    assert np.all(v_bc[:, 0] == 0.0)

    # Outlet (right): zero gradient (flow continues out of the channel)
    assert np.allclose(u_bc[:, -1], u_bc[:, -2])
    assert np.allclose(v_bc[:, -1], v_bc[:, -2])

    # Cylinder obstacle: no-slip
    assert np.all(u_bc[sim.obstacle_mask] == 0.0)
    assert np.all(v_bc[sim.obstacle_mask] == 0.0)