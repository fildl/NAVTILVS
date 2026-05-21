import numpy as np
from ns_solver.solver import NS_solver
from plots.plot import plot_stream

# Parameters
# Geometric parameters
NX = 41
NY = 41
LX = 2.0
LY = 1.0
NT = 500
DT = 0.001

# Physics parameters
RHO = 1.0 # Density
NU = 0.1  # Kinematic viscosity

# Spacial steps
dx = LX / (NX - 1)
dy = LY / (NY - 1)

# Phisical fileds
u = np.zeros((NY, NX))
v = np.zeros((NY, NX))
p = np.zeros((NY, NX))

# Run simulaion
print("Cavity Flow Simulation running")
u, v, p = NS_solver(
    u=u, v=v, p=p,
    dx=dx, dy=dy,
    nx=NX, ny=NY,
    rho=RHO,
    nu=NU,
    nt=NT,
    dt=DT
)

# Plot
plot_stream(
    u=u, v=v, p=p,
    lx=LX, ly=LY,
    nx=NX, ny=NY,
    rho=RHO,
    nu=NU
)