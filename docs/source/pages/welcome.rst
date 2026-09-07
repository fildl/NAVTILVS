Welcome to NAVTILVS
===================

**NAVTILVS** (**NAV**\ ier-Stokes **T**\ wo-dimensional **I**\ ncompressib\ **L**\ e **V**\ isual **S**\ olver) is a Python library designed to simulate and visualize 2D incompressible fluid flows using finite difference methods on structured Cartesian grids.

Features
--------

* **Incompressible Flow Solver**: Solves the 2D unsteady incompressible Navier–Stokes equations under the Boussinesq/constant-density approximation.
* **Vectorized Poisson Pressure Solver**: Pressure is computed via an iterative Poisson solver optimized for NumPy vectorization.
* **First-Order Upwind Scheme**: Automatically selects backward or forward spatial derivatives based on local velocity direction, eliminating unphysical downwind oscillations and preserving numerical stability.
* **Adaptive Dynamic Time Stepping**: Continuously monitors numerical stability constraints at each iteration, satisfying both the convective Courant–Friedrichs–Lewy (CFL) condition and the 2D viscous diffusion limit:

  .. math::

     \Delta t \le \min \left( \text{CFL} \cdot \min\left(\frac{\Delta x}{|u|_{\max}}, \frac{\Delta y}{|v|_{\max}}\right), \; \frac{1}{4\nu} \frac{\Delta x^2 \Delta y^2}{\Delta x^2 + \Delta y^2} \right)

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

NAVTILVS comes with two simulations:

1. Lid-Driven Cavity Flow
^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

References
----------

The core numerical solver of **NAVTILVS** is based on the educational program `CFDPython: 12 steps to Navier-Stokes <https://github.com/barbagroup/CFDPython>`_ by Prof. Lorena A. Barba's group.

Barba, Lorena A., and Forsyth, Gilbert F. (2018). CFD Python: the 12 steps to Navier-Stokes equations. *Journal of Open Source Education*, 1(9), 21, `https://doi.org/10.21105/jose.00021 <https://doi.org/10.21105/jose.00021>`_.