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

def backward_diff_y(f: np.ndarray,
                    dy: float
                    ) -> np.ndarray:
    """
    Compute the backward difference of a 2D array along the y-axis.

    Parameters
    ----------

    f : np.ndarray
        2D array for which to compute the backward difference.
    dy : float
        Grid spacing in the y-direction.

    Returns
    -------
    np.ndarray
        2D array with backward difference along the y-axis.

    Raises
    ------
    ValueError
        If the input array is not 2-dimensional.
    """

    if f.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional. Got {f.ndim}D instead.")
    
    return (f[1:-1, 1:-1] - f[0:-2, 1:-1]) / dy

def centered_diff_x(f: np.ndarray,
                    dx: float
                    ) -> np.ndarray:
    """
    Compute the centered difference of a 2D array along the x-axis.

    Parameters
    ----------

    f : np.ndarray
        2D array for which to compute the centered difference.
    dx : float
        Grid spacing in the x-direction.

    Returns
    -------
    np.ndarray
        2D array with centered difference along the x-axis.

    Raises
    ------
    ValueError
        If the input array is not 2-dimensional.
    """

    if f.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional. Got {f.ndim}D instead.")
    
    return (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * dx)

def centered_diff_y(f: np.ndarray,
                    dy: float
                    ) -> np.ndarray:
    """
    Compute the centered difference of a 2D array along the y-axis.

    Parameters
    ----------

    f : np.ndarray
        2D array for which to compute the centered difference.
    dy : float
        Grid spacing in the y-direction.

    Returns
    -------
    np.ndarray
        2D array with centered difference along the y-axis.

    Raises
    ------
    ValueError
        If the input array is not 2-dimensional.
    """

    if f.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional. Got {f.ndim}D instead.")
    
    return (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * dy)