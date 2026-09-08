import numpy as np
from plots.plot import _compute_vorticity_limits, _compute_pressure_limits

TOLERANCE = 1e-5

# ==========================================
# _compute_vorticity_limits
# ==========================================

def test_compute_vorticity_limits_default():
    """
    Test default vorticity scaling using the 98th percentile.
    """

    # Create array with an outlier
    vort = np.linspace(-10, 10, 1000).reshape((25, 40))
    vort[0, 0] = 500.0  # outlier

    limit, extend = _compute_vorticity_limits(vort, vlim=None, clip_percentile=98.0)

    expected_limit = float(np.percentile(np.abs(vort), 98.0))
    assert np.isclose(limit, expected_limit, rtol=TOLERANCE)
    assert limit < 500.0
    assert extend == 'both'

def test_compute_vorticity_limits_no_clip():
    """
    Test vorticity scaling when clipping is disabled (clip_percentile=None or 100.0).
    """

    vort = np.array([[-5.0, 2.0], [1.0, -10.0]])

    limit, extend = _compute_vorticity_limits(vort, vlim=None, clip_percentile=None)
    assert limit == 10.0
    assert extend == 'neither'

    limit, extend = _compute_vorticity_limits(vort, vlim=None, clip_percentile=100.0)
    assert limit == 10.0
    assert extend == 'neither'

def test_compute_vorticity_limits_explicit_vlim():
    """
    Test explicit vlim overrides for vorticity (scalar and tuple).
    """

    vort = np.array([[-5.0, 2.0], [1.0, -10.0]])

    # Scalar override
    limit, extend = _compute_vorticity_limits(vort, vlim=7.5, clip_percentile=98.0)
    assert limit == 7.5
    assert extend == 'both'

    # Negative scalar
    limit, extend = _compute_vorticity_limits(vort, vlim=-6.0, clip_percentile=None)
    assert limit == 6.0

    # Tuple override: symmetric limit is max of absolute values
    limit, extend = _compute_vorticity_limits(vort, vlim=(-4.0, 8.0), clip_percentile=None)
    assert limit == 8.0
    assert extend == 'both'

def test_compute_vorticity_limits_all_zeros():
    """
    Test fallback to 1.0 when vorticity field is entirely zero.
    """

    vort = np.zeros((10, 10))

    limit, extend = _compute_vorticity_limits(vort, vlim=None, clip_percentile=98.0)
    assert limit == 1.0

# ==========================================
# _compute_pressure_limits
# ==========================================

def test_compute_pressure_limits_default():
    """
    Test default pressure scaling using 2nd and 98th percentiles.
    """

    p = np.linspace(-5, 5, 1000).reshape((25, 40))
    p[0, 0] = -100.0  # low outlier
    p[-1, -1] = 100.0  # high outlier

    (p_min, p_max), extend = _compute_pressure_limits(p, vlim=None, clip_percentile=98.0)

    expected_min = float(np.percentile(p, 2.0))
    expected_max = float(np.percentile(p, 98.0))

    assert np.isclose(p_min, expected_min, rtol=TOLERANCE)
    assert np.isclose(p_max, expected_max, rtol=TOLERANCE)
    assert p_min > -100.0
    assert p_max < 100.0
    assert extend == 'both'

def test_compute_pressure_limits_no_clip():
    """
    Test pressure scaling when clipping is disabled (clip_percentile=None or 100.0).
    """

    p = np.array([[-3.0, 1.0], [4.0, -1.0]])

    (p_min, p_max), extend = _compute_pressure_limits(p, vlim=None, clip_percentile=None)
    assert p_min == -3.0
    assert p_max == 4.0
    assert extend == 'neither'

    (p_min_100, p_max_100), extend = _compute_pressure_limits(p, vlim=None, clip_percentile=100.0)
    assert p_min_100 == -3.0
    assert p_max_100 == 4.0
    assert extend == 'neither'

def test_compute_pressure_limits_explicit_vlim():
    """
    Test explicit vlim overrides for pressure (tuple and scalar).
    """

    p = np.array([[-3.0, 1.0], [4.0, -1.0]])

    # Tuple override (p_min, p_max)
    bounds, extend = _compute_pressure_limits(p, vlim=(-2.0, 2.0), clip_percentile=98.0)
    assert bounds == (-2.0, 2.0)
    assert extend == 'both'

    # Scalar override -> symmetric [-vlim, vlim]
    bounds, extend = _compute_pressure_limits(p, vlim=5.0, clip_percentile=None)
    assert bounds == (-5.0, 5.0)
    assert extend == 'both'

def test_compute_pressure_limits_constant_field():
    """
    Test fallback expansion when pressure field is uniform/constant.
    """

    p = np.full((10, 10), 5.0)

    bounds, extend = _compute_pressure_limits(p, vlim=None, clip_percentile=98.0)
    assert bounds == (4.0, 6.0)

    bounds, extend = _compute_pressure_limits(p, vlim=None, clip_percentile=None)
    assert bounds == (4.0, 6.0)
