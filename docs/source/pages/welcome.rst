Welcome to NAVTILVS
===================

**NAVTILVS** (**NAV**\ ier-Stokes **T**\ wo-dimensional **I**\ ncompressib\ **L**\ e **V**\ isual **S**\ olver) is a Python library designed to simulate and visualize 2D incompressible fluid flows using finite difference methods.

Features
--------

* **Incompressible Flow Solver**: Solves the 2D unsteady incompressible Navier–Stokes equations using finite difference discretizations on structured Cartesian grids.
* **Vectorized Poisson Pressure Solver**: Pressure is computed via an iterative Poisson solver optimized for NumPy vectorization.
* **First-Order Upwind Scheme**: Automatically selects forward or backward spatial differences according to local flow direction, eliminating downwind instabilities.
* **Adaptive Dynamic Time Stepping**: Dynamically evaluates stability constraints at every time step, adhering to both the convective Courant–Friedrichs–Lewy (CFL) condition and the 2D viscous diffusion limit.
* **Modular Object-Oriented Design**:
  * :class:`~ns_solver.grid.Grid`: Handles spatial discretization, metric scale precomputations, and coordinate meshes.
  * :class:`~ns_solver.simulation.SimulationClass`: Base class for time integration, stability, and solver execution.
  * :class:`~ns_solver.simulation.CavitySimulation`: Implementation of the classic Lid-Driven Cavity benchmark.
  * :class:`~ns_solver.simulation.CylinderSimulation`: Channel flow around a circular obstacle with no-slip and nearest-fluid Neumann boundary conditions.
* **Visualization**: Plotting tools (:func:`~plots.plot.plot_cavity_flow`, :func:`~plots.plot.plot_cylinder_flow`) generating velocity vector fields, streamlines, velocity magnitude, pressure contours, and symmetric vorticity fields.

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