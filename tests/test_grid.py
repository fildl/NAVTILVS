import pytest
import math
from ns_solver import Grid

TOLERANCE = 1e-5

def test_grid_creation():
    """
    Test if the grid is created correctly and the parameters are assigned correctly.
    """

    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)

    assert grid.lx == 1.0
    assert grid.ly == 1.0
    assert grid.nx == 10
    assert grid.ny == 10

@pytest.mark.parametrize("lx, ly, nx, ny", [
    (-1.0, 1.0, 10, 10),  # Negative grid length in x-direction
    (1.0, -1.0, 10, 10),  # Negative grid length in y-direction
    (0, 1.0, 10, 10),     # Zero grid length in x-direction
    (1.0, 0, 10, 10),     # Zero grid length in y-direction
])
def test_grid_invalid_length(lx, ly, nx, ny):
    """
    Validate that negative or zero grid lengths raise an error.
    """

    with pytest.raises(ValueError):
        Grid(lx=lx, ly=ly, nx=nx, ny=ny)

@pytest.mark.parametrize("lx, ly, nx, ny, expected_exception", [
    (1.0, 1.0, 1.0, 10, TypeError),   # Float grid points in x-direction
    (1.0, 1.0, 10, 1.0, TypeError),   # Float grid points in y-direction
    (1.0, 1.0, 1, 10, ValueError),    # Only 1 grid point in x-direction
    (1.0, 1.0, 10, 1, ValueError),    # Only 1 grid point in y-direction
    (1.0, 1.0, 0, 10, ValueError),    # 0 grid points in x-direction
    (1.0, 1.0, 10, 0, ValueError),    # 0 grid points in y-direction
    (1.0, 1.0, -1, 10, ValueError),   # Negative grid points in x-direction
    (1.0, 1.0, 10, -1, ValueError),   # Negative grid points in y-direction
])
def test_grid_invalid_number_of_grid_points(lx, ly, nx, ny, expected_exception):
    """
    Validate that the number of grid points is valid.
    """

    with pytest.raises(expected_exception):
        Grid(lx=lx, ly=ly, nx=nx, ny=ny)

def test_grid_discretization():
    """
    Test if spatial discretization is correct.
    """

    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)

    assert grid.dx == 1.0 / 9.0
    assert grid.dy == 1.0 / 9.0

    grid = Grid(lx=1.0, ly=1.0, nx=128, ny=128)

    assert math.isclose(grid.dx, 1.0 / 127.0, rel_tol=TOLERANCE)
    assert math.isclose(grid.dy, 1.0 / 127.0, rel_tol=TOLERANCE)

def test_grid_discretization_asymmetric_length():
    """
    Test if spatial discretization is correct with different grid lengths.
    """

    grid = Grid(lx=1.0, ly=5.0, nx=128, ny=128)

    assert math.isclose(grid.dx, 1.0 / 127.0, rel_tol=TOLERANCE)
    assert math.isclose(grid.dy, 5.0 / 127.0, rel_tol=TOLERANCE)

def test_grid_discretization_asymmetric_grid_points():
    """
    Test if spatial discretization is correct with different grid points.
    """

    grid = Grid(lx=1.0, ly=1.0, nx=128, ny=256)

    assert math.isclose(grid.dx, 1.0 / 127.0, rel_tol=TOLERANCE)
    assert math.isclose(grid.dy, 1.0 / 255.0, rel_tol=TOLERANCE)