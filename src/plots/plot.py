import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_stream(u, v,
                p,
                lx, ly,
                nx, ny,
                rho, nu
                ):
    
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(11, 7), dpi=100)
    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, p, cmap=cm.viridis)
    plt.streamplot(X, Y, u, v)
    
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title(f"Final State\nRho: {rho} Nu: {nu}")

    # create file name using variabiles
    rho_str = str(rho).replace('.', '_')
    nu_str = str(nu).replace('.', '_')
    file_name = f"flow_rho_{rho_str}_nu_{nu_str}.png"
    #plt.savefig(file_name, bbox_inches="tight")
    plt.show()