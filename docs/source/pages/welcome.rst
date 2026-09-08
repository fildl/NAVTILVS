Welcome to NAVTILVS
===================

**NAVTILVS** (**NAV**\ ier-Stokes **T**\ wo-dimensional **I**\ ncompressib\ **L**\ e **V**\ isual **S**\ olver) is a Python library designed to simulate and visualize 2D incompressible fluid flows using finite difference methods on structured Cartesian grids.

.. figure:: /_static/cylinder_velocity.png
   :align: center
   :width: 90%
   :alt: Cylinder Wake Velocity Field

   *Flow past a circular cylinder (Re = 100): Kármán vortex street resolved with the upwind finite difference scheme.*

Features
--------

* **Incompressible Flow Solver**: Solves the 2D unsteady incompressible Navier–Stokes equations under the Boussinesq/constant-density approximation.
* **First-Order Upwind Scheme**: Automatically selects backward or forward spatial derivatives based on local velocity direction, eliminating unphysical downwind oscillations and preserving numerical stability.
* **Vectorized Poisson Pressure Solver**: Pressure is computed via an iterative Poisson solver optimized for NumPy vectorization.
* **Adaptive Dynamic Time Stepping**: Continuously monitors numerical stability constraints at each iteration, satisfying both the convective Courant–Friedrichs–Lewy (CFL) condition and the 2D viscous diffusion limit:

  .. math::

     \Delta t \le \min \left( \frac{C_{\text{cfl}}}{\dfrac{|u|_{\max}}{\Delta x} + \dfrac{|v|_{\max}}{\Delta y}}, \; C_{\text{visc}} \frac{\Delta x^2 \Delta y^2}{2\nu (\Delta x^2 + \Delta y^2)} \right)

* **Modular Object-Oriented Design**:

  * :class:`~ns_solver.grid.Grid`: Handles spatial domain discretization, metric scales, and mesh coordinates.
  * :class:`~ns_solver.simulation.SimulationClass`: Base class managing time integration, adaptive time-stepping, and numerical stability.
  * :class:`~ns_solver.simulation.CavitySimulation`: Solver for the classical Lid-Driven Cavity benchmark problem.
  * :class:`~ns_solver.simulation.CylinderSimulation`: Channel flow past an immersed circular cylinder with nearest-fluid Neumann pressure and no-slip velocity conditions.

* **CFD Visual Analytics**: Dedicated visualization tools (:func:`~plots.plot.plot_cavity_flow`, :func:`~plots.plot.plot_cylinder_flow`) generating directional quiver velocity fields, streamlines, pressure contours and vorticity fields.

Installation
------------

To install and use **NAVTILVS**, clone the GitHub repository and install the package in editable (development) mode.

Clone via HTTPS:

.. code-block:: bash

   git clone https://github.com/fildl/NAVTILVS.git
   cd NAVTILVS

Or clone via SSH:

.. code-block:: bash

   git clone git@github.com:fildl/NAVTILVS.git
   cd NAVTILVS

Create and activate a virtual environment (recommended):

.. code-block:: bash

   conda create -n navtilvs python=3.12
   conda activate navtilvs

Install dependencies and the package:

.. code-block:: bash

   python -m pip install -r requirements.txt

Then choose the installation mode:

* **Standard Installation (User Mode)**:

  .. code-block:: bash

     python -m pip install .

* **Development Installation (Editable Mode)**:

  .. code-block:: bash

     python -m pip install -e .

  To also install testing tools (``pytest``):

  .. code-block:: bash

     python -m pip install -e ".[dev]"

Requirements
------------

**NAVTILVS** requires Python **>= 3.10** and the following dependencies:

* **Core Libraries**:

  * ``numpy >= 1.20.0``
  * ``matplotlib >= 3.4.0``

* **Testing**:

  * ``pytest >= 7.0.0``

Quickstart & Examples
---------------------

NAVTILVS can be used both through its command-line interface (CLI) and programmatically via its Python API.

Command-Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script ``main.py`` provides a command-line entrypoint to configure, run, and save plots for both cases.

**Command-line Arguments**:

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Argument
     - Type
     - Default
     - Description
   * - ``--sim``
     - ``str``
     - ``cavity``
     - Simulation benchmark to execute: ``cavity`` (Lid-Driven Cavity) or ``cylinder`` (Flow Around a Cylinder).
   * - ``--re``
     - ``float``
     - ``100.0``
     - Target Reynolds number (:math:`Re`). Determines the kinematic viscosity :math:`\nu`.
   * - ``--tend``
     - ``float``
     - *auto*
     - Final physical simulation time in seconds (defaults to ``2.5`` for cavity, ``3.5`` for cylinder).
   * - ``--nx``
     - ``int``
     - *auto*
     - Number of grid points (nodes) along the :math:`x`-direction (defaults to ``128`` for cavity, ``256`` for cylinder).
   * - ``--ny``
     - ``int``
     - *auto*
     - Number of grid points (nodes) along the :math:`y`-direction (defaults to ``128`` for cavity, ``64`` for cylinder).
   * - ``--output-dir``
     - ``str``
     - ``imgs``
     - Directory path where generated diagnostic figures will be stored.

**CLI Usage Examples**:

1. **Lid-Driven Cavity Benchmark**:

   Runs the square cavity problem and generates 4 plots:

   .. code-block:: bash

      python main.py --sim cavity --re 100.0 --tend 2.5

   Generated files in ``imgs/``:

   * ``cavity_velocity.png``: Velocity magnitude field ('turbo' colormap) overlaid with directional quiver vectors.
   * ``cavity_flow.png``: Velocity streamlines overlaid on pressure contour field ('turbo' colormap).
   * ``cavity_vorticity.png``: Vorticity field with symmetric 'coolwarm' colormap highlighting recirculating vortex cores.
   * ``cavity_pressure.png``: Pressure field distribution with 'coolwarm' colormap.

2. **Flow Past a Circular Cylinder**:

   Solves channel flow past an obstacle, resolving wake separation and vortex shedding:

   .. code-block:: bash

      python main.py --sim cylinder --re 100.0 --tend 3.5

   Generated files in ``imgs/``:

   * ``cylinder_vorticity.png``: Vorticity field with symmetric 'coolwarm' colormap, resolving alternating Kármán vortex street shedding.
   * ``cylinder_velocity.png``: Velocity magnitude field ('turbo' colormap) overlaid with streamlines and recirculation bubbles.
   * ``cylinder_pressure.png``: Pressure field with 'coolwarm' colormap, highlighting upstream stagnation front and low-pressure wake depression.

3. **Custom Resolution & High Reynolds Number**:

   .. code-block:: bash

      python main.py --sim cylinder --re 150.0 --nx 512 --ny 128 --tend 5.0 --output-dir results/

Python API
^^^^^^^^^^

For advanced scripting, custom geometries, or pipeline integration, use the Python API directly:

1. Lid-Driven Cavity Flow
~~~~~~~~~~~~~~~~~~~~~~~~~

Simulates recirculating vortex dynamics in a square cavity induced by a moving top lid:

.. code-block:: bash

   python src/examples/cavity.py

Or directly from Python:

.. code-block:: python

   from ns_solver import Grid, CavitySimulation
   from plots import plot_cavity_flow

   grid = Grid(lx=1.0, ly=1.0, nx=256, ny=256)
   sim = CavitySimulation(grid=grid, rho=1.0, nu=0.01, dt=0.001)

   u, v, p = sim.solve(t_end=2.5)

   # Plot velocity magnitude with vectors
   plot_cavity_flow(u, v, p, grid, mode='velocity', t=sim.t,
                    reynolds=sim.reynolds_number, save_path="imgs/cavity_velocity.png")

2. Flow Around a Circular Cylinder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulates channel flow around an immersed cylinder, resolving boundary layer separation, wake recirculation, and vortex shedding:

.. code-block:: bash

   python src/examples/cylinder.py

Or from Python:

.. code-block:: python

   from ns_solver import Grid, CylinderSimulation
   from plots import plot_cylinder_flow

   grid = Grid(lx=2.0, ly=0.5, nx=512, ny=128)
   sim = CylinderSimulation(grid=grid, rho=1.0, nu=0.001, dt=0.002,
                            cylinder_center=(0.4, 0.225), cylinder_radius=0.05,
                            u_inlet=1.0)

   u, v, p = sim.solve(t_end=3.5)

   # Plot vorticity field
   plot_cylinder_flow(u, v, p, grid, sim.obstacle_mask, mode='vorticity',
                      t=sim.t, reynolds=sim.reynolds_number,
                      save_path="imgs/cylinder_vorticity.png")

Simulation Gallery
------------------

Lid-Driven Cavity Flow (:math:`Re = 100`, :math:`t = 2.50\text{ s}`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/cavity_velocity.png
   :align: center
   :width: 80%
   :alt: Lid-Driven Cavity Velocity Field

   **Velocity Field**: Velocity magnitude field overlaid with directional velocity vectors, showing the primary vortex centered at :math:`(x \approx 0.62\text{ m}, y \approx 0.78\text{ m})`.

.. figure:: /_static/cavity_flow.png
   :align: center
   :width: 80%
   :alt: Lid-Driven Cavity Streamlines

   **Streamlines & Pressure Field**: Velocity streamlines and pressure contours depicting corner stagnation zones and recirculation.

Flow Past a Circular Cylinder (:math:`Re = 100`, :math:`t = 3.50\text{ s}`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/cylinder_vorticity.png
   :align: center
   :width: 95%
   :alt: Cylinder Wake Vorticity Field

   **Vorticity Field**: Alternating vortex street demonstrating boundary layer detachment and Kármán wake dynamics.

.. figure:: /_static/cylinder_velocity.png
   :align: center
   :width: 95%
   :alt: Cylinder Velocity Field

   **Velocity Field**: Velocity magnitude field with streamlines displaying stagnation, lateral acceleration, and recirculation bubbles.

.. figure:: /_static/cylinder_pressure.png
   :align: center
   :width: 95%
   :alt: Cylinder Pressure Field

   **Pressure Field**: High-pressure stagnation zone on the upstream face and low-pressure wake zone.

Testing Suite
-------------

**NAVTILVS** includes a test suite managed by ``pytest``:

.. code-block:: bash

   pytest tests/ -v

Test files are organized into modular suites:

* ``tests/test_grid.py``: Validates grid spacing, mesh generation, and metric dimensions.
* ``tests/test_finite_differences.py``: Tests accuracy of centered, backward, forward difference and laplacian operators.
* ``tests/test_solver.py``: Tests iterative Poisson solver convergence and upwind velocity advection.
* ``tests/test_simulation_base.py``: Tests base simulation parameters, time integration, and adaptive stability limits (CFL and diffusion).
* ``tests/test_simulation_cavity.py``: Verifies Dirichlet/Neumann boundary conditions and global mass conservation.
* ``tests/test_simulation_cylinder.py``: Checks obstacle geometric masking and immersed boundary conditions.

How to Cite
-----------

If you use **NAVTILVS** in academic coursework, research, or presentations, please cite it as:

.. code-block:: bibtex

   @software{navtilvs_2026,
     author       = {Filippo Di Ludovico},
     title        = {NAVTILVS: NAVier-Stokes Two-dimensional IncompressibLe Visual Solver},
     year         = {2026},
     publisher    = {GitHub},
     url          = {https://github.com/fildl/NAVTILVS}
   }

References
----------

1. Barba, L. A., & Forsyth, G. F. (2018). *CFD Python: the 12 steps to Navier-Stokes equations*. Journal of Open Source Education, 1(9), 21. `https://doi.org/10.21105/jose.00021 <https://doi.org/10.21105/jose.00021>`_
2. Hirsch, C. (2007). *Numerical Computation of Internal and External Flows: The Fundamentals of Computational Fluid Dynamics* (Second Edition). Butterworth-Heinemann.