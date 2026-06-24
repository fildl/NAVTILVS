import pytest
import numpy as np
from ns_solver import backward_diff_x, backward_diff_y, centered_diff_x, centered_diff_y, laplacian_2d

TOLERANCE = 1e-5

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

def test_backward_diff_y_invalid_dimensions():
    """
    Test that the backward difference function raises a ValueError for non-2D arrays.
    """

    # 1D array
    f_1d = np.zeros(5)
    with pytest.raises(ValueError):
        backward_diff_y(f_1d, 1.0)

    # 3D array
    f_3d = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        backward_diff_y(f_3d, 1.0)

def test_backward_diff_y_zero_array():
    """
    Test that the backward difference of a zero array returns a zero array.
    """

    f = np.zeros((5, 5))

    assert np.all(backward_diff_y(f, 1.0) == 0)

def test_backward_diff_y_uniform_array():
    """
    Test that the backward difference of a uniform array returns a zero array.
    """

    f = np.ones((5, 5))

    assert np.all(backward_diff_y(f, 1.0) == 0.0)

def test_backward_diff_y_linear_array():
    """
    Test that the backward difference of a linear array returns the correct constant value.
    """

    # Linear array in y-direction
    f = np.zeros((5, 5))
    for i in range(5):
        f[i, :] = i

    # The backward difference should be 1 everywhere
    expected_result = np.ones((3, 3))

    assert np.all(backward_diff_y(f, 1.0) == expected_result)

def test_backward_diff_y_non_uniform_array():
    """
    Test that the backward difference function correctly computes differences
    for a non-uniformly increasing array.
    """

    # Non-uniform array in y-direction
    f = np.array([
        [1.0,  1.0,  1.0,  1.0],
        [3.0,  3.0,  3.0,  3.0],
        [6.0,  6.0,  6.0,  6.0],
        [10.0, 10.0, 10.0, 10.0]
        ])

    # Inner nodes
    # row 1 diff: (3.0 - 1.0) / 1.0 = 2.0
    # row 2 diff: (6.0 - 3.0) / 1.0 = 3.0
    expected_result = np.array([
        [2.0, 2.0],
        [3.0, 3.0]
        ])

    assert np.allclose(backward_diff_y(f, 1.0), expected_result)

def test_centered_diff_x_invalid_dimensions():
    """
    Test that the centered difference function raises a ValueError for non-2D arrays.
    """

    # 1D array
    f_1d = np.zeros(5)
    with pytest.raises(ValueError):
        centered_diff_x(f_1d, 1.0)

    # 3D array
    f_3d = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        centered_diff_x(f_3d, 1.0)

def test_centered_diff_x_zero_array():
    """
    Test that the centered difference of a zero array returns a zero array.
    """

    f = np.zeros((5, 5))

    assert np.all(centered_diff_x(f, 1.0) == 0)

def test_centered_diff_x_uniform_array():
    """
    Test that the centered difference of a uniform array returns a zero array.
    """

    f = np.ones((5, 5))

    assert np.all(centered_diff_x(f, 1.0) == 0.0)

def test_centered_diff_x_linear_array():
    """
    Test that the centered difference of a linear array returns the correct constant value.
    """

    # Linear array in x-direction
    f = np.zeros((5, 5))
    for i in range(5):
        f[:, i] = i * 2

    # The centered difference should be 2 everywhere
    expected_result = np.ones((3, 3)) * 2

    assert np.all(centered_diff_x(f, 1.0) == expected_result)

def test_centered_diff_x_quadratic_array():
    """
    Test that the centered difference of a quadratic array returns the correct values.
    """

    # Quadratic array in x-direction
    f = np.array([
        [2.0,  4.0,  8.0,  16.0, 32.0],
        [2.0,  4.0,  8.0,  16.0, 32.0],
        [2.0,  4.0,  8.0,  16.0, 32.0],
        [2.0,  4.0,  8.0,  16.0, 32.0],
        [2.0,  4.0,  8.0,  16.0, 32.0],
        ])
    
    expected_result = np.array([
        [3.0, 6.0, 12.0],
        [3.0, 6.0, 12.0],
        [3.0, 6.0, 12.0]
        ])

    assert np.allclose(centered_diff_x(f, 1.0), expected_result, rtol=TOLERANCE)

def test_centered_diff_y_invalid_dimensions():
    """
    Test that the centered difference function raises a ValueError for non-2D arrays.
    """

    # 1D array
    f_1d = np.zeros(5)
    with pytest.raises(ValueError):
        centered_diff_y(f_1d, 1.0)

    # 3D array
    f_3d = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        centered_diff_y(f_3d, 1.0)

def test_centered_diff_y_zero_array():
    """
    Test that the centered difference of a zero array returns a zero array.
    """

    f = np.zeros((5, 5))

    assert np.all(centered_diff_y(f, 1.0) == 0)

def test_centered_diff_y_uniform_array():
    """
    Test that the centered difference of a uniform array returns a zero array.
    """

    f = np.ones((5, 5))

    assert np.all(centered_diff_y(f, 1.0) == 0.0)

def test_centered_diff_y_linear_array():
    """
    Test that the centered difference of a linear array returns the correct constant value.
    """

    # Linear array in y-direction
    f = np.zeros((5, 5))
    for i in range(5):
        f[i, :] = i * 2

    # The centered difference should be 2 everywhere
    expected_result = np.ones((3, 3)) * 2

    assert np.all(centered_diff_y(f, 1.0) == expected_result)

def test_centered_diff_y_quadratic_array():
    """
    Test that the centered difference of a quadratic array returns the correct values.
    """

    # Quadratic array in y-direction
    f = np.array([
        [2.0,  2.0,  2.0,  2.0,  2.0],
        [4.0,  4.0,  4.0,  4.0,  4.0],
        [8.0,  8.0,  8.0,  8.0,  8.0],
        [16.0, 16.0, 16.0, 16.0, 16.0],
        [32.0, 32.0, 32.0, 32.0, 32.0],
        ])
    
    expected_result = np.array([
        [3.0,  3.0,  3.0],
        [6.0,  6.0,  6.0],
        [12.0, 12.0, 12.0]
        ])

    assert np.allclose(centered_diff_y(f, 1.0), expected_result, rtol=TOLERANCE)

def test_laplacian_2d_invalid_dimensions():
    """
    Test that the 2d Laplacian function raises a ValueError for non-2D arrays.
    """

    # 1D array
    f_1d = np.zeros(5)
    with pytest.raises(ValueError):
        laplacian_2d(f_1d, 1.0, 1.0)

    # 3D array
    f_3d = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        laplacian_2d(f_3d, 1.0, 1.0)

def test_laplacian_2d_zero_array():
    """
    Test that the Laplacian of a zero array returns a zero array.
    """

    f = np.zeros((5, 5))

    assert np.all(laplacian_2d(f, 1.0, 1.0) == 0)

def test_laplacian_2d_uniform_array():
    """
    Test that the Laplacian of a uniform array returns a zero array.
    """

    f = np.ones((5, 5))

    assert np.all(laplacian_2d(f, 1.0, 1.0) == 0.0)

def test_laplacian_2d_linear_array():
    """
    Test that the Laplacian of a linear array is zero.
    """

    # Linear array
    f = np.array([
        [1.0,  2.0,  3.0,  4.0, 5.0],
        [1.0,  2.0,  3.0,  4.0, 5.0],
        [1.0,  2.0,  3.0,  4.0, 5.0],
        [1.0,  2.0,  3.0,  4.0, 5.0],
        [1.0,  2.0,  3.0,  4.0, 5.0],
        ])

    expected_result = np.zeros((3, 3))

    assert np.all(laplacian_2d(f, 1.0, 1.0) == expected_result)

def test_laplacian_2d_quadratic_array():
    """
    Test that the Laplacian of a quadratic array returns the correct values.
    """

    # Quadratic array in x-direction: f(x) = x^2
    # The second derivative should be 2.0 everywhere.
    f = np.array([
        [0.0,  1.0,  4.0,  9.0, 16.0],
        [0.0,  1.0,  4.0,  9.0, 16.0],
        [0.0,  1.0,  4.0,  9.0, 16.0],
        [0.0,  1.0,  4.0,  9.0, 16.0],
        [0.0,  1.0,  4.0,  9.0, 16.0],
        ])
    
    expected_result = np.ones((3, 3)) * 2.0

    assert np.allclose(laplacian_2d(f, 1.0, 1.0), expected_result, rtol=TOLERANCE)