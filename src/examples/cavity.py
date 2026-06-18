from ns_solver import Grid
from ns_solver import CavitySimulation
from plots import plot_stream

# Physics parameters
RHO = 1.0 # Density
NU = 0.1  # Kinematic viscosity

# Define grid
grid = Grid(lx = 1.0,
             ly = 1.0,
             nx = 41,
             ny = 41)

# Define simulation
simulation = CavitySimulation(grid=grid,
                              rho=RHO,
                              nu=NU,
                              dt=0.001)

# Run simulation
u, v, p = simulation.solve(nt=500)

# Plot
plot_stream(
    u=u, v=v, p=p,
    lx=grid.lx, ly=grid.ly,
    nx=grid.nx, ny=grid.ny,
    rho=RHO,
    nu=NU
)