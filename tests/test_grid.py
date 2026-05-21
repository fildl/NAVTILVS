from ns_solver.grid import Grid
import pytest

def test_grid_creation():
    """
    Test if the grid is created correctly and the parameters are assigned correctly.
    """

    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)

    assert grid.lx == 1.0
    assert grid.ly == 1.0
    assert grid.nx == 10
    assert grid.ny == 10

def test_grid_invalid_lengh():
    """
    Validate that negative grid lengths raise an error.
    """

    # Test with negative grid length in x-direction
    with pytest.raises(ValueError):
        Grid(lx=-1.0, ly=1.0, nx=10, ny=10)

    # Test with negative grid length in y-direction
    with pytest.raises(ValueError):
        Grid(lx=1.0, ly=-1.0, nx=10, ny=10)

def test_grid_invalid_number_of_grid_points():
    """
    Validate that the number of grid points is valid.
    """

    # Test with float grid points in x-direction
    with pytest.raises(TypeError):
        Grid(lx=1.0, ly=1.0, nx=1.0, ny=10)

    # Test with float grid points in y-direction
    with pytest.raises(TypeError):
        Grid(lx=1.0, ly=1.0, nx=10, ny=1.0)

    # Test with only 1 grid point in x-direction
    with pytest.raises(ValueError):
        Grid(lx=1.0, ly=1.0, nx=1, ny=10)

    # Test with only 1 grid point in y-direction
    with pytest.raises(ValueError):
        Grid(lx=1.0, ly=1.0, nx=10, ny=1)

    # Test with 0 grid points in x-direction
    with pytest.raises(ValueError):
        Grid(lx=1.0, ly=1.0, nx=0, ny=10)

    # Test with 0 grid points in y-direction
    with pytest.raises(ValueError):
        Grid(lx=1.0, ly=1.0, nx=10, ny=0)

    # Test with negative grid points in x-direction
    with pytest.raises(ValueError):
        Grid(lx=1.0, ly=1.0, nx=-1, ny=10)

    # Test with negative grid points in y-direction
    with pytest.raises(ValueError):
        Grid(lx=1.0, ly=1.0, nx=10, ny=-1)

def test_grid_discretization():
    """
    Test if spatial discretization is correct.
    """

    grid = Grid(lx=1.0, ly=1.0, nx=10, ny=10)

    assert grid.dx == 1.0 / 9.0
    assert grid.dy == 1.0 / 9.0