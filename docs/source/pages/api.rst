Code Documentation (API)
========================

The **NAVTILVS** API is structured into five modular components.

.. toctree::
   :maxdepth: 1
   :hidden:

   api/grid
   api/finite_differences
   api/solver
   api/simulation
   api/plots

Overview of Modules
-------------------

1. :doc:`api/grid`
   **Spatial Domain Discretization & Mesh Coordinates**

   Handles the 2D Cartesian spatial discretization and metric dimensions of the flow domain.

   * :class:`~ns_solver.grid.Grid`: Core class managing physical domain lengths (:math:`L_x, L_y`), grid resolution (:math:`n_x, n_y`), spatial grid spacing (:math:`\Delta x, \Delta y`), and coordinate meshgrids (:math:`X, Y`).

2. :doc:`api/finite_differences`
   **Vectorized Spatial Differential Operators**

   Provides vectorized finite difference operators on 2D scalar and vector fields:

   * **Upwind Differences**: First-order operators (:func:`~ns_solver.finite_differences.backward_diff_x`, :func:`~ns_solver.finite_differences.forward_diff_x`, :func:`~ns_solver.finite_differences.backward_diff_y`, :func:`~ns_solver.finite_differences.forward_diff_y`).
   * **Centered Differences**: Second-order central approximations (:func:`~ns_solver.finite_differences.centered_diff_x`, :func:`~ns_solver.finite_differences.centered_diff_y`).
   * **Laplacian**: 5-point operator (:func:`~ns_solver.finite_differences.laplacian_2d`).

3. :doc:`api/solver`
   **Numerical Solver**

   Contains routines for Navier–Stokes time advancement and pressure Poisson relaxation:

   * :func:`~ns_solver.solver.build_up_b`: Computes the source term (:math:`b`) of the pressure Poisson equation.
   * :func:`~ns_solver.solver.pressure_poisson`: Solves the Poisson equation for pressure (:math:`p`).
   * :func:`~ns_solver.solver.update_velocity`: Solves momentum equations for both velocity components (:math:`u, v`).

4. :doc:`api/simulation`
   **Object-Oriented Physical Simulation**

   Provides classes that configure and execute physical flow simulations:

   * :class:`~ns_solver.simulation.SimulationClass`: Base class managing the time integration of the 2D incompressible Navier-Stokes equations.
   * :class:`~ns_solver.simulation.CavitySimulation`: Simulation solver for the classical 2D Lid-Driven Cavity benchmark problem.
   * :class:`~ns_solver.simulation.CylinderSimulation`: Simulation solver for 2D channel flow past an immersed circular cylinder.

5. :doc:`api/plots`
   **CFD Visual Analytics & Diagnostics**

   Dedicated plotting routines for visualizing and exporting publication-ready flow fields:

   * :func:`~plots.plot.plot_cavity_flow`: Plot the flow field inside a lid-driven cavity.
   * :func:`~plots.plot.plot_cylinder_flow`: Plot the flow field around a cylinder.