from ns_solver import Grid
from ns_solver import CavitySimulation
from plots import plot_cavity_flow

# Physics parameters
RHO = 1.0 # Density
NU = 0.01  # Kinematic viscosity

# Define grid
grid = Grid(lx = 1.0,
            ly = 1.0,
            nx = 256,
            ny = 256)

# Define simulation
simulation = CavitySimulation(grid=grid,
                              rho=RHO,
                              nu=NU,
                              dt=0.001)

# Run simulation
print("Running solver...")
print(f"Reynolds number: {simulation.reynolds_number:.2f}")
u, v, p = simulation.solve(nt=None, t_end=2.5)
print(f"Simulation completed! Simulated time: {simulation.t:.3f} s")

# 1. Plot velocity field with directional quiver arrows (turbo colormap)
plot_cavity_flow(u=u, v=v, p=p,
                 grid=grid,
                 mode='velocity',
                 t=simulation.t,
                 reynolds=simulation.reynolds_number,
                 save_path="imgs/cavity_velocity.png")

# 2. Plot streamlines and pressure contours
plot_cavity_flow(u=u, v=v, p=p,
                 grid=grid,
                 mode='stream',
                 nu=NU,
                 t=simulation.t,
                 reynolds=simulation.reynolds_number,
                 save_path="imgs/cavity_flow.png")

# 3. Plot Vorticity Field
plot_cavity_flow(u=u, v=v, p=p,
                 grid=grid,
                 mode='vorticity',
                 t=simulation.t,
                 reynolds=simulation.reynolds_number,
                 save_path="imgs/cavity_vorticity.png")

# 4. Plot Pressure Field
plot_cavity_flow(u=u, v=v, p=p,
                 grid=grid,
                 mode='pressure',
                 t=simulation.t,
                 reynolds=simulation.reynolds_number,
                 save_path="imgs/cavity_pressure.png")