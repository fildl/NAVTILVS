import pytest
import numpy as np
from ns_solver import backward_diff_x


def test_backward_diff_x_invalid_dimensions():
    """
    Test that the backward difference function raises a ValueError for non-2D arrays.
    """

    # 1D array
    f_1d = np.zeros(5)
    with pytest.raises(ValueError):
        backward_diff_x(f_1d, 1.0)

    # 3D array
    f_3d = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        backward_diff_x(f_3d, 1.0)

def test_backward_diff_x_zero_array():
    """
    Test that the backward difference of a zero array returns a zero array.
    """

    f = np.zeros((5, 5))

    assert np.all(backward_diff_x(f, 1.0) == 0)

def test_backward_diff_x_uniform_array():
    """
    Test that the backward difference of a uniform array returns a zero array.
    """

    f = np.ones((5, 5))

    assert np.all(backward_diff_x(f, 1.0) == 0.0)

def test_backward_diff_x_linear_array():
    """
    Test that the backward difference of a linear array returns the correct constant value.
    """

    # Linear array in x-direction
    f = np.zeros((5, 5))
    for i in range(5):
        f[:, i] = i

    # The backward difference should be 1 everywhere
    expected_result = np.ones((3, 3))

    assert np.all(backward_diff_x(f, 1.0) == expected_result)

def test_backward_diff_x_non_uniform_array():
    """
    Test that the backward difference function correctly computes differences
    for a non-uniformly increasing array.
    """

    # Non-uniform array in x-direction
    f = np.array([
        [1.0, 3.0, 6.0, 10.0],
        [1.0, 3.0, 6.0, 10.0],
        [1.0, 3.0, 6.0, 10.0],
        [1.0, 3.0, 6.0, 10.0]
        ])

    # Inner nodes
    # col 1 diff: (3.0 - 1.0) / 1.0 = 2.0
    # col 2 diff: (6.0 - 3.0) / 1.0 = 3.0
    expected_result = np.array([
        [2.0, 3.0],
        [2.0, 3.0]
        ])

    assert np.allclose(backward_diff_x(f, 1.0), expected_result)