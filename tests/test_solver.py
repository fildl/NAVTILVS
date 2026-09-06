import numpy as np
from ns_solver import build_up_b, pressure_poisson, update_velocity

TOLERANCE = 1e-5

# ==========================================
# build_up_b Tests
# ==========================================

def test_build_up_b_zero_velocity():
    """
    Verify that if the velocity fields are zero, the source term b is zero.
    """

    u = np.zeros((10, 10))
    v = np.zeros((10, 10))
    b = build_up_b(dx=0.1, dy=0.1, u=u, v=v, rho=1.0, dt=0.01)
    
    assert np.all(b == 0.0)

def test_build_up_b_uniform_flow():
    """
    Verify that if the velocity fields are uniform, the source term b is zero, since spatial gradients are zero.
    """

    u = np.ones((10, 10))
    v = np.ones((10, 10))
    
    b = build_up_b(dx=0.1, dy=0.1, u=u, v=v, rho=1.0, dt=0.01)
    
    assert np.all(b == 0.0)

def test_build_up_b_shear_flow_u():
    """
    Verify that for a shear flow :math:`u=y`, the source term b is zero.
    """

    u = np.zeros((10, 10))
    v = np.zeros((10, 10))
    
    # Shear flow: u increases linearly in y from 0 to 0.9
    for i in range(10):
        u[i, :] = i * 0.1
    
    b = build_up_b(dx=0.1, dy=0.1, u=u, v=v, rho=1.0, dt=0.01)
    
    assert np.all(b == 0.0)

def test_build_up_b_shear_flow_v():
    """
    Verify that for a shear flow :math:`v=x`, the source term b is zero.
    """

    u = np.zeros((10, 10))
    v = np.zeros((10, 10))
    
    # Shear flow: u increases linearly in y from 0 to 0.9
    for j in range(10):
        v[:, j] = j * 0.1
    
    b = build_up_b(dx=0.1, dy=0.1, u=u, v=v, rho=1.0, dt=0.01)
    
    assert np.all(b == 0.0)

def test_build_up_b_linear_divergence():
    """
    Verify that for a linear divergence field, the source term b is computed correctly.
    
    Here, :math:`u=x` and :math:`v=y`, so that :math:`du/dx = 1`, :math:`dv/dy = 1`,
    while cross terms :math:`du/dy = 0`, :math:`dv/dx = 0`.
    The analytical source term is :math:`b = rho * ( 2/dt - 2 )`.

    For rho = 1.0, dt = 0.5, we expect b = 2.0 on inner nodes.
    """
    nx, ny = 10, 10
    dx, dy = 0.1, 0.2
    
    # Grid coordinates
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    
    u = X
    v = Y
    
    b = build_up_b(dx=dx, dy=dy, u=u, v=v, rho=1.0, dt=0.5)
    
    # Boundary nodes of b are initialized to 0 and remain 0.
    # Check boundary nodes are zero
    assert np.all(b[0, :]  == 0.0)
    assert np.all(b[-1, :] == 0.0)
    assert np.all(b[:, 0]  == 0.0)
    assert np.all(b[:, -1] == 0.0)

    # Check inner nodes: expected value is 2.0
    assert np.allclose(b[1:-1, 1:-1], 2.0, rtol=TOLERANCE)

def test_build_up_b_cross_shear():
    """
    Verify that for a cross shear field, the source term b is computed correctly.

    Here, :math:`u=y` and :math:`v=x`, so that cross terms :math:`du/dy = 1`, :math:`dv/dx = 1`,
    while :math:`du/dx = 0`, :math:`dv/dy = 0`.
    The analytical source term is :math:`b = -2 * rho`.

    For rho = 1.0, we expect b = -2.0 on inner nodes.
    """
    
    nx, ny = 10, 10
    dx, dy = 0.1, 0.1
    
    # Grid coordinates
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    
    u = Y
    v = X
    
    b = build_up_b(dx=dx, dy=dy, u=u, v=v, rho=1.0, dt=0.1)
    
    # Boundary nodes of b are initialized to 0 and remain 0.
    # Check boundary nodes are zero
    assert np.all(b[0, :]  == 0.0)
    assert np.all(b[-1, :] == 0.0)
    assert np.all(b[:, 0]  == 0.0)
    assert np.all(b[:, -1] == 0.0)

    # Check inner nodes: expected value is -2.0
    assert np.allclose(b[1:-1, 1:-1], -2.0, rtol=TOLERANCE)

# ==========================================
# pressure_poisson Tests
# ==========================================

def test_pressure_poisson_flat():
    """
    Verify that if the source term :math:`b = 0` and pressure is constant at the boundaries,
    i.e. the boundary conditions are identity,
    pressure field remains constant (Poisson equation reduces to Laplace equation with zero source term).
    """

    p = np.ones((10, 10)) * 3.0
    b = np.zeros((10, 10))
    
    def identity_bc(x):
        return x
        
    p = pressure_poisson(p, dx=0.1, dy=0.1, b=b, boundary_conditions=identity_bc, max_iter=10)
    
    assert np.allclose(p, 3.0, rtol=TOLERANCE)

def test_pressure_poisson_linear():
    """
    Verify that if the source term :math:`b = 0` and boundary conditions match a linear
    pressure field (p = x + y), pressure field remains constant.
    """

    nx, ny = 10, 10
    dx, dy = 0.1, 0.1
    
    # Grid coordinates
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    
    # Linear field p = x + y
    p = X + Y

    b = np.zeros((ny, nx))
    
    def identity_bc(x):
        return x
        
    p = pressure_poisson(p, dx=dx, dy=dy, b=b, boundary_conditions=identity_bc, max_iter=10)
    
    assert np.allclose(p, X + Y, rtol=TOLERANCE)

def test_pressure_poisson_quadratic():
    """
    Verify that for a quadratic pressure field (p = x^2 + y^2) and constant source
    term (b = 4.0), pressure field remains constant.
    """

    nx, ny = 10, 10
    dx, dy = 0.1, 0.1
    
    # Grid coordinates
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    
    # Quadratic field p = x^2 + y^2
    p = X**2 + Y**2

    # Laplacian: d2p/dx2 + d2p/dy2 = 2.0 + 2.0 = 4.0
    b = np.ones((ny, nx)) * 4.0
    
    def identity_bc(x):
        return x
        
    p = pressure_poisson(p, dx=dx, dy=dy, b=b, boundary_conditions=identity_bc, max_iter=10)
    
    assert np.allclose(p, X**2 + Y**2, rtol=TOLERANCE)

# ==========================================
# update_velocity Tests
# ==========================================

def test_update_velocity_uniform_flow():
    """
    Verify that velocity updater preserves uniform flow fields in the absence of pressure gradients.
    """

    u = np.ones((10, 10))
    v = np.zeros((10, 10))
    un = np.ones((10, 10))
    vn = np.zeros((10, 10))
    p = np.zeros((10, 10))
    
    u, v = update_velocity(u, v, un, vn, dt=0.01, dx=0.1, dy=0.1, p=p, rho=1.0, nu=0.1)
    
    assert np.allclose(u[1:-1, 1:-1], 1.0, rtol=TOLERANCE)
    assert np.allclose(v[1:-1, 1:-1], 0.0, rtol=TOLERANCE)

def test_update_velocity_negative_uniform_flow():
    """
    Verify that velocity updater preserves negative uniform velocity fields.
    """

    u = np.full((10, 10), -1.0)
    v = np.full((10, 10), -0.5)
    un = u.copy()
    vn = v.copy()
    p = np.zeros((10, 10))

    u, v = update_velocity(u, v, un, vn, dt=0.001, dx=0.1, dy=0.1, p=p, rho=1.0, nu=0.01)

    assert np.allclose(u[1:-1, 1:-1], -1.0, rtol=TOLERANCE)
    assert np.allclose(v[1:-1, 1:-1], -0.5, rtol=TOLERANCE)

def test_update_velocity_upwind_selection_u():
    """
    Verify that update_velocity selects backward difference for u > 0 and forward difference for u < 0.
    """

    vn = np.zeros((3, 3))
    p = np.zeros((3, 3))
    dt = 0.01

    # backward_diff_x: (5 - 0) / 1.0 = 5.0
    # forward_diff_x:  (20 - 5) / 1.0 = 15.0
    f = np.array([
        [0.0, 5.0, 20.0],
        [0.0, 5.0, 20.0],
        [0.0, 5.0, 20.0]
    ])
    
    # Case 1: u > 0, it should select backward difference (5.0)
    u = np.ones((3, 3))
    un = f.copy()
    
    u, v = update_velocity(u, vn.copy(), un, vn, dt=dt, dx=1.0, dy=1.0, p=p, rho=1.0, nu=0.0)

    # u = un - dt * 5.0
    u_expected = un[1, 1] - dt * (un[1, 1] * 5.0)
    assert np.isclose(u[1, 1], u_expected, rtol=TOLERANCE)

    # Case 2: u < 0, it should select forward difference (15.0)
    u = np.full((3, 3), -1.0)
    un = f.copy()
    un[1, 1] = -1.0 # set the central node to negative (-1.0) to trigger the < 0 case

    # forward_diff_x = (20.0 - (-1.0)) / 1.0 = 21.0
    u, v = update_velocity(u, vn.copy(), un, vn, dt=dt, dx=1.0, dy=1.0, p=p, rho=1.0, nu=0.0)

    # u = un - dt * (u_mid * forward) = un - dt * (-1.0 * 21.0)
    u_expected = un[1, 1] - dt * (un[1, 1] * 21.0)
    assert np.isclose(u[1, 1], u_expected, rtol=TOLERANCE)

def test_update_velocity_upwind_selection_v():
    """
    Verify that update_velocity selects backward difference for v > 0 and forward difference for v < 0.
    """

    un = np.zeros((3, 3))
    p = np.zeros((3, 3))
    dt = 0.01

    # backward_diff_y: (5.0 - 0.0) / 1.0 = 5.0
    # forward_diff_y:  (20.0 - 5.0) / 1.0 = 15.0
    f = np.array([
        [0.0,  0.0,  0.0],
        [5.0,  5.0,  5.0],
        [20.0, 20.0, 20.0]
    ])

    # Case 1: v > 0, it should select backward difference (5.0)
    v = np.ones((3, 3))
    vn = f.copy()

    u, v_res = update_velocity(un.copy(), v, un, vn, dt=dt, dx=1.0, dy=1.0, p=p, rho=1.0, nu=0.0)

    # v = vn - dt * 5.0
    v_expected = vn[1, 1] - dt * (vn[1, 1] * 5.0)
    assert np.isclose(v_res[1, 1], v_expected, rtol=TOLERANCE)

    # Case 2: v < 0, it should select forward difference (15.0)
    v = np.full((3, 3), -1.0)
    vn = f.copy()
    vn[1, 1] = -1.0  # set the central node to negative (-1.0) to trigger the < 0 case

    # forward_diff_y = (20.0 - (-1.0)) / 1.0 = 21.0
    u, v = update_velocity(un.copy(), v, un, vn, dt=dt, dx=1.0, dy=1.0, p=p, rho=1.0, nu=0.0)

    # v = vn - dt * 21.0
    v_expected = vn[1, 1] - dt * (vn[1, 1] * 21.0)
    assert np.isclose(v[1, 1], v_expected, rtol=TOLERANCE)