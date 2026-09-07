from ns_solver import Grid
from ns_solver import CavitySimulation
from plots import plot_stream

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
u, v, p = simulation.solve(nt=None, t_end=2.0)
print(f"Simulation completed! Simulated time: {simulation.t:.3f} s")

# Plot
plot_stream(u=u, v=v, p=p,
            grid=grid,
            rho=RHO,
            nu=NU)