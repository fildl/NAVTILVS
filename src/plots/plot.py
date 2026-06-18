import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_stream(u : np.ndarray,
                v : np.ndarray,
                p : np.ndarray,
                lx : float,
                ly : float,
                nx : int,
                ny : int,
                rho : float,
                nu : float
                ) -> None:
    """
    Plot the streamlines of the velocity field and the pressure contours.

    This function generates a 2D visualization of the simulation results,
    displaying the pressure field as a contour plot and
    the velocity field as streamlines.

    Parameters
    ----------
    u : np.ndarray
        Velocity field in the x-direction.
    v : np.ndarray
        Velocity field in the y-direction.
    p : np.ndarray
        Pressure field.
    lx : float
        Length of the domain in the x-direction.
    ly : float
        Length of the domain in the y-direction.
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity.
    """
    
    # Create spatial coordinates
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(11, 7), dpi=100)

    # Plot pressure contours
    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar(label='Pressure')
    plt.contour(X, Y, p, cmap=cm.viridis)

    # Plot velocity streamlines
    plt.streamplot(X, Y, u, v)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Final State\nRho: {rho} Nu: {nu}")

    # Show plot
    plt.show()