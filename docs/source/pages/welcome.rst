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

Requirements
------------

**TODO**

Simulation Gallery
------------------

Lid-Driven Cavity Flow
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/cavity_velocity.png
   :align: center
   :width: 80%
   :alt: Lid-Driven Cavity Velocity Field

   **Velocity Field** (:math:`Re = 100`, :math:`t = 2.50\text{ s}`): Velocity magnitude overlaid with velocity vectors.

.. figure:: /_static/cavity_flow.png
   :align: center
   :width: 80%
   :alt: Lid-Driven Cavity Streamlines

   **Streamlines & Pressure Field** (:math:`Re = 100`, :math:`t = 2.50\text{ s}`): Primary recirculating vortex and pressure contours.

Flow Around a Circular Cylinder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/cylinder_vorticity.png
   :align: center
   :width: 95%
   :alt: Cylinder Wake Vorticity Field

   **Vorticity Field** (:math:`Re = 100`, :math:`t = 3.50\text{ s}`): Alternating vortex demonstrating wake dynamics.

.. figure:: /_static/cylinder_velocity.png
   :align: center
   :width: 95%
   :alt: Cylinder Velocity Field

   **Velocity Field** (:math:`Re = 100`, :math:`t = 3.50\text{ s}`): Velocity magnitude field with streamlines displaying stagnation and lateral acceleration.

.. figure:: /_static/cylinder_pressure.png
   :align: center
   :width: 95%
   :alt: Cylinder Pressure Field

   **Pressure Field** (:math:`Re = 100`, :math:`t = 3.50\text{ s}`): High-pressure stagnation zone upstream and low-pressure wake zone.

References
----------

The core numerical solver of **NAVTILVS** is based on the educational program `CFDPython: 12 steps to Navier-Stokes <https://github.com/barbagroup/CFDPython>`_ by Prof. Lorena A. Barba's group.

Barba, Lorena A., and Forsyth, Gilbert F. (2018). CFD Python: the 12 steps to Navier-Stokes equations. *Journal of Open Source Education*, 1(9), 21, `https://doi.org/10.21105/jose.00021 <https://doi.org/10.21105/jose.00021>`_.