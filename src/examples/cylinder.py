from ns_solver import Grid
from ns_solver import CylinderSimulation
from plots import plot_cylinder_flow

# Physics parameters
RHO = 1.0 # Density
NU = 0.005  # Kinematic viscosity

# Define grid
grid = Grid(lx = 2.0,
            ly = 0.5,
            nx = 128,
            ny = 32)

# Define simulation
simulation = CylinderSimulation(grid=grid,
                                rho=RHO,
                                nu=NU,
                                dt=0.005,
                                cylinder_center=(0.4, 0.25),
                                cylinder_radius=0.05,
                                u_inlet=1.0)

# Run simulation
print("Running solver...")
u, v, p = simulation.solve(nt=None, t_end=5.0)
print(f"Simulation completed! Simulated time: {simulation.t:.3f} s")

# Plot Vorticity
plot_cylinder_flow(u=u, v=v, p=p,
                   grid=grid,
                   obstacle_mask=simulation.obstacle_mask,
                   mode='vorticity')

# Plot Velocity
plot_cylinder_flow(u=u, v=v, p=p,
                   grid=grid,
                   obstacle_mask=simulation.obstacle_mask,
                   mode='velocity')

# Plot Pressure
plot_cylinder_flow(u=u, v=v, p=p,
                   grid=grid,
                   obstacle_mask=simulation.obstacle_mask,
                   mode='pressure')