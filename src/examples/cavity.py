from ns_solver import Grid, CavitySimulation
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

# 1. Plot Velocity Field
plot_cavity_flow(u=u, v=v, p=p,
                 grid=grid,
                 mode='velocity',
                 t=simulation.t,
                 reynolds=simulation.reynolds_number,
                 save_path="imgs/cavity_velocity.png")

# 2. Plot Streamlines & Pressure Field
plot_cavity_flow(u=u, v=v, p=p,
                 grid=grid,
                 mode='pressure',
                 nu=NU,
                 t=simulation.t,
                 reynolds=simulation.reynolds_number,
                 save_path="imgs/cavity_pressure.png")

# 3. Plot Vorticity Field
plot_cavity_flow(u=u, v=v, p=p,
                 grid=grid,
                 mode='vorticity',
                 t=simulation.t,
                 reynolds=simulation.reynolds_number,
                 save_path="imgs/cavity_vorticity.png")