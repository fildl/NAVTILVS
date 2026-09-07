Navier–Stokes Theory and Numerical Methods
===========================================

This section provides the mathematical formulations, discretization schemes, and physical modeling principles implemented in **NAVTILVS**.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _momentum_equation:

Governing Equations
-------------------

Momentum Equation
~~~~~~~~~~~~~~~~~

The momentum conservation equation for an incompressible Newtonian fluid in vector notation is:

.. math::
   \frac{\partial \vec v}{\partial t} + (\vec v \cdot \nabla) \vec v = - \frac{1}{\rho} \nabla p + \nu \nabla^2 \vec v

where:

* :math:`\vec v = (u, v)` is the velocity vector field,
* :math:`p` is the static pressure field,
* :math:`\rho` is the constant fluid density,
* :math:`\nu = \frac{\mu}{\rho}` is the kinematic viscosity.

In two dimensions, the vector equation decomposes into two coupled non-linear partial differential equations:

.. math::
   \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} =
   - \frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)

.. math::
   \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} =
   - \frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)

.. _incompressible_fluid:

Mass Conservation and Pressure Poisson Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an incompressible fluid, conservation of mass reduces to the divergence-free kinematic constraint:

.. math::
   \nabla \cdot \vec v = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0

Taking the divergence of the momentum equations and setting :math:`\nabla \cdot \vec v = 0` yields the **Poisson equation for pressure**:

.. math::
   \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} = b(x, y)

where the source term :math:`b` is defined by:

.. math::
   b = - \rho \left[
   \left(\frac{\partial u}{\partial x}\right)^2 +
   2 \frac{\partial u}{\partial y}\frac{\partial v}{\partial x} +
   \left(\frac{\partial v}{\partial y}\right)^2
   \right] + \frac{\rho}{\Delta t} \left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)

The second term on the right-hand side represents compressibility/divergence correction, preventing the numerical accumulation of dilatation errors over time.

.. _convective_upwind:

Convective Discretization: The Upwind Scheme
--------------------------------------------

The non-linear advection terms :math:`(\vec v \cdot \nabla)\vec v` describe the transport of momentum by the velocity field.

Recirculating Flow
~~~~~~~~~~~~~~~~~~

When using a fixed backward difference scheme on a Cartesian grid:

.. math::
   u \frac{\partial f}{\partial x} \approx u_i \frac{f_i - f_{i-1}}{\Delta x}

the scheme is strictly stable only when the local flow direction is positive (:math:`u_i > 0`).
In regions of flow separation and recirculation, such as the wake behind an obstacle or the primary vortex of a cavity, the local velocity becomes negative (:math:`u_i < 0`).

In these negative velocity regions, a backward difference points *downwind*, which introduces a negative numerical diffusion and leads to divergence.

First-Order Upwind Differencing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To guarantee numerical stability across all flow regimes, **NAVTILVS** implements a **First-Order Upwind Scheme**. The spatial difference operator automatically adapts its direction based on the local sign of the advecting velocity:

Along the :math:`x`-axis:

.. math::
   \left( u \frac{\partial f}{\partial x} \right)_{i,j} \approx
   \begin{cases} 
   u_{i,j} \dfrac{f_{i,j} - f_{i-1,j}}{\Delta x} & \text{if } u_{i,j} > 0 \quad (\text{Backward}) \\[10pt]
   u_{i,j} \dfrac{f_{i+1,j} - f_{i,j}}{\Delta x} & \text{if } u_{i,j} < 0 \quad (\text{Forward})
   \end{cases}

Along the :math:`y`-axis:

.. math::
   \left( v \frac{\partial f}{\partial y} \right)_{i,j} \approx
   \begin{cases} 
   v_{i,j} \dfrac{f_{i,j} - f_{i,j-1}}{\Delta y} & \text{if } v_{i,j} > 0 \quad (\text{Backward}) \\[10pt]
   v_{i,j} \dfrac{f_{i,j+1} - f_{i,j}}{\Delta y} & \text{if } v_{i,j} < 0 \quad (\text{Forward})
   \end{cases}

Modified Equation Analysis and Numerical Diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Taylor series expansion reveals the truncation error of the first-order upwind scheme (for :math:`u > 0`):

.. math::
   u \frac{f_i - f_{i-1}}{\Delta x} = u \left( \frac{\partial f}{\partial x} - \frac{\Delta x}{2} \frac{\partial^2 f}{\partial x^2} + \mathcal{O}(\Delta x^2) \right)

Substituting this into the continuous momentum equation yields the *modified equation*:

.. math::
   \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \dots = \left(\nu + \nu_{\text{num}}\right) \frac{\partial^2 u}{\partial x^2}

where:

.. math::
   \nu_{\text{num}} = \frac{|u| \Delta x}{2}

This intrinsic **numerical viscosity** (:math:`\nu_{\text{num}}`) stabilizes the discrete equations at high Reynolds numbers by damping high-frequency spatial oscillations, enabling stable computations at Reynolds numbers previously unattainable with unconditioned schemes.

.. _numerical_stability:

Numerical Stability and Adaptive Time Stepping
----------------------------------------------

To ensure stability in explicit time marching, the time step :math:`\Delta t` must satisfy two independent stability criteria:

1. **Convective Stability (CFL Condition)**:
   Information cannot propagate more than one spatial grid cell per time step:

   .. math::
      \Delta t_{\text{cfl}} = C_{\text{cfl}} \frac{1}{\dfrac{|u|_{\text{max}}}{\Delta x} + \dfrac{|v|_{\text{max}}}{\Delta y}}

   where :math:`C_{\text{cfl}} < 1` is the Courant safety factor (default: :math:`0.5`).

2. **Viscous Stability (Diffusion Limit)**:
   In explicit diffusion schemes, von Neumann stability analysis in 2D establishes (Hirsch, 2007) [1]_:

   .. math::
      \nu \Delta t \left( \frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} \right) \le \frac{1}{2}

   yielding the viscous time step limit:

   .. math::
      \Delta t_{\text{visc}} = C_{\text{visc}} \frac{\Delta x^2 \Delta y^2}{2 \nu (\Delta x^2 + \Delta y^2)}

   where :math:`C_{\text{visc}} < 1` is a safety factor (default: :math:`0.9`).

At each simulation step, the active time step is dynamically evaluated as:

.. math::
   \Delta t = \min(\Delta t_{\text{cfl}}, \Delta t_{\text{visc}})

.. _reynolds_number:

Dimensionless Analysis and Reynolds Number
------------------------------------------

The **Reynolds number** (:math:`Re`) represents the ratio of inertial forces to viscous forces in a fluid:

.. math::
   Re = \frac{U L}{\nu}

where :math:`U` is a characteristic flow velocity and :math:`L` is a characteristic length scale.

Lid-Driven Cavity Flow
~~~~~~~~~~~~~~~~~~~~~~

For the square cavity benchmark (:class:`~ns_solver.simulation.CavitySimulation`):

* Characteristic velocity: lid velocity :math:`U = u_{\text{lid}}` (default: :math:`1.0`),
* Characteristic length: cavity width :math:`L = L_x`.

.. math::
   Re = \frac{u_{\text{lid}} L_x}{\nu}

Flow Around a Cylinder
~~~~~~~~~~~~~~~~~~~~~~

For external flow past an obstacle (:class:`~ns_solver.simulation.CylinderSimulation`):

* Characteristic velocity: uniform inlet velocity :math:`U = u_{\text{inlet}}`,
* Characteristic length: cylinder diameter :math:`D = 2 r`.

.. math::
   Re = \frac{u_{\text{inlet}} (2 r)}{\nu}

The physical flow regimes past a circular cylinder vary drastically with :math:`Re`:

* (:math:`Re < 5`):
  Inertial forces are negligible; streamlines are fully attached and no flow separation occurs.
* (:math:`5 \le Re < 47`):
  Boundary layer separation occurs on the surface of the cylinder, forming two steady symmetric recirculation vortices. The recirculation length grows linearly with :math:`Re`.
* (:math:`47 \le Re < 180`):
  The steady wake undergoes a bifurcation; alternating vortices detach periodically.
* (:math:`Re > 180`):
  In physical experiments, secondary three-dimensional instabilities emerge, transitioning toward turbulence. Because **NAVTILVS** solves the two-dimensional Navier–Stokes equations, it cannot capture spanwise vortex stretching.

.. _obstacle_modeling:

Obstacle Modeling and Boundary Conditions
-----------------------------------------

To simulate arbitrary geometry within a structured Cartesian grid, a boolean mask :math:`M` is defined over all spatial nodes:

.. math::
   M_{i,j} = 
   \begin{cases} 
   1 & \text{if node } (i,j) \text{ lies within the solid obstacle} \\
   0 & \text{otherwise (fluid)}
   \end{cases}

Boundary conditions on the immersed obstacle are enforced as follows:

1. **Velocity No-Slip**:
   Fluid adheres to the solid surface:

   .. math::
      u_{i,j} = 0, \quad v_{i,j} = 0 \quad \forall (i,j) \text{ such that } M_{i,j} = 1

2. **Pressure Zero Normal Gradient**:
   The physical boundary condition on an impermeable wall is :math:`\frac{\partial p}{\partial n} = 0`. On a Cartesian staircased grid, this is enforced by projecting the pressure of each obstacle node to its closest fluid neighbor:

   .. math::
      p_{i,j} = p_{\text{nearest fluid}(i,j)} \quad \forall (i,j) \text{ such that } M_{i,j} = 1

   The mapping from obstacle cells to the closest fluid nodes is precomputed once during initialization using Euclidean distances:

   .. math::
      \text{nearest fluid}(i, j) = \arg\min_{(k, l) \notin M} \left[ \left((i - k)\Delta y\right)^2 + \left((j - l)\Delta x\right)^2 \right]

   During each iteration of the pressure Poisson solver, this mapping is applied as a vectorized 1-to-1 array copy, maintaining zero overhead while ensuring a smooth pressure profile across the solid interface.

References
----------

.. [1] Hirsch, C. (2007). *Numerical Computation of Internal and External Flows: The Fundamentals of Computational Fluid Dynamics* (Second Edition). Butterworth-Heinemann.