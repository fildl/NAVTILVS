from ns_solver import Grid, CylinderSimulation
from plots import plot_cylinder_flow

# Physics parameters
RHO = 1.0 # Density
NU = 0.001  # Kinematic viscosity

# Define grid
grid = Grid(lx = 2.0,
            ly = 0.5,
            nx = 512,
            ny = 128)

# Define simulation
simulation = CylinderSimulation(grid=grid,
                                rho=RHO,
                                nu=NU,
                                dt=0.002,
                                cylinder_center=(0.4, 0.225),
                                cylinder_radius=0.05,
                                u_inlet=1.0)

# Run simulation
print("Running solver...")
print(f"Reynolds number: {simulation.reynolds_number:.2f}")
u, v, p = simulation.solve(nt=None, t_end=3.5)
print(f"Simulation completed! Simulated time: {simulation.t:.3f} s")

# 1. Plot Velocity Field
plot_cylinder_flow(u=u, v=v, p=p,
                   grid=grid,
                   obstacle_mask=simulation.obstacle_mask,
                   mode='velocity',
                   t=simulation.t,
                   reynolds=simulation.reynolds_number,
                   save_path='imgs/cylinder_velocity.png')

# 2. Plot Streamlines & Pressure Field
plot_cylinder_flow(u=u, v=v, p=p,
                   grid=grid,
                   obstacle_mask=simulation.obstacle_mask,
                   mode='pressure',
                   t=simulation.t,
                   reynolds=simulation.reynolds_number,
                   save_path='imgs/cylinder_pressure.png')

# 3. Plot Vorticity Field
plot_cylinder_flow(u=u, v=v, p=p,
                   grid=grid,
                   obstacle_mask=simulation.obstacle_mask,
                   mode='vorticity',
                   t=simulation.t,
                   reynolds=simulation.reynolds_number,
                   save_path='imgs/cylinder_vorticity.png')