import numpy as np

def backward_diff_x(f: np.ndarray,
                    dx: float
                    ) -> np.ndarray:
    """
    Compute the backward difference of a 2D array along the x-axis.

    Parameters
    ----------

    f : np.ndarray
        2D array for which to compute the backward difference.
    dx : float
        Grid spacing in the x-direction.

    Returns
    -------
    np.ndarray
        2D array with backward difference along the x-axis.

    Raises
    ------
    ValueError
        If the input array is not 2-dimensional.
    """

    if f.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional. Got {f.ndim}D instead.")
    
    return (f[1:-1, 1:-1] - f[1:-1, 0:-2]) / dx