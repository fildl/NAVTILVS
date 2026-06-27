Navier–Stokes Equation
======================

Momentum Equation
-----------------

The momentum equation in vector form for a velocity field :math:`\vec v` is:

.. math::
   \frac{\partial \vec v}{\partial t} +
   (\vec v \cdot \nabla) \vec v =
   - \frac{1}{\rho} \nabla p + \nu \nabla^2 \vec v

This represents three scalar equations, one for each velocity component :math:`(u, v, w)`.
In this solver, we consider the two-dimensional case, which reduces the system to two scalar equations:

.. math::
   \frac{\partial u}{\partial t} +
   u \frac{\partial u}{\partial x} +
   v \frac{\partial u}{\partial y} =
   - \frac{1}{\rho} \frac{\partial p}{\partial x} +
   \nu \left(\frac{\partial^2 u}{\partial x^2} +
   \frac{\partial^2 u}{\partial y^2}\right)

.. math::
   \frac{\partial v}{\partial t} +
   u \frac{\partial v}{\partial x} +
   v \frac{\partial v}{\partial y} =
   - \frac{1}{\rho} \frac{\partial p}{\partial y} +
   \nu \left(\frac{\partial^2 v}{\partial x^2} +
   \frac{\partial^2 v}{\partial y^2}\right)

.. _incompressible_fluid:

Incompressible Fluid
--------------------

The equation

.. math::
   \nabla \cdot \vec v = 0

represents mass conservation at constant density.
In incompressible flow it provides a kinematic constraint that requires the pressure field to evolve so that :math:`\nabla \cdot \vec v = 0` everywhere.
Taking the divergence of the momentum equation leads to a Poisson equation for pressure:

.. math::
   \frac{\partial^2 p}{\partial^2 x} +
   \frac{\partial^2 p}{\partial^2 y} = 
   - \rho \left[
      \left(\frac{\partial u}{\partial x}\right)^2 +
      2\frac{\partial u}{\partial y}\frac{\partial v}{\partial x} +
      \left(\frac{\partial v}{\partial y}\right)^2
      \right] = b

.. _numerical_stability:

Numerical Stability and Time Stepping
-------------------------------------

To ensure numerical stability of the explicit finite difference solver, the time step :math:`\Delta t` must satisfy stability criteria for both convection and diffusion.

1. **Convective Stability (CFL Condition)**:
   The Courant-Friedrichs-Lewy (CFL) condition ensures that physical information does not propagate faster than the grid speed:

   .. math::
      \Delta t_{cfl} = C_{cfl} \frac{1}{\frac{|u|_{max}}{\Delta x} + \frac{|v|_{max}}{\Delta y}}

   where :math:`C_{cfl} < 1` is a safety factor (typically :math:`0.5`).

2. **Viscous Stability (Diffusion Limit)**:
   In a 1D diffusion equation, the explicit scheme is stable under the condition (Hirsch, 2007) [1]_:

   .. math::
      \frac{\nu \Delta t}{\Delta x^2} \le \frac{1}{2} \implies \Delta t \le \frac{\Delta x^2}{2\nu}

   Extending the von Neumann stability analysis to a 2D grid yields:

   .. math::
      \nu \Delta t \left( \frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} \right) \le \frac{1}{2}

   This limits the viscous time step to:

   .. math::
      \Delta t_{visc} = C_{visc} \frac{\Delta x^2 \Delta y^2}{2 \nu (\Delta x^2 + \Delta y^2)}

   where :math:`C_{visc} < 1` is a safety factor (typically :math:`0.9`).
   
   If the grid is isotropic (:math:`\Delta x = \Delta y`), the limit simplifies to 
   
   .. math::
      \Delta t_{visc} \le \frac{\Delta x^2}{4\nu}

To guarantee stable execution across all regimes, the active time step is dynamically computed at each step as:

.. math::
   \Delta t = \min(\Delta t_{cfl}, \Delta t_{visc})

Obstacle Modeling and Boundary Conditions
-----------------------------------------

To simulate flow around an obstacle (such as a cylinder) on a Cartesian grid, a boolean mask :math:`M` is defined over the domain:

.. math::
   M_{i,j} = 
   \begin{cases} 
   1 & \text{if node } (i,j) \text{ is inside the obstacle} \\
   0 & \text{otherwise (fluid)}
   \end{cases}

Solid boundary conditions are enforced on the obstacle through two mechanisms:

1. **Velocity No-Slip**:
   The fluid velocity is forced to zero on all masked nodes:

   .. math::
      u_{i,j} = 0, \quad v_{i,j} = 0 \quad \forall (i,j) \text{ where } M_{i,j} = 1

2. **Pressure Zero Normal Gradient**:
   The physical boundary condition for pressure on a solid wall is :math:`\frac{\partial p}{\partial n} = 0`. On a staircoded Cartesian grid, this is approximated by setting the pressure of each obstacle node to the pressure of its nearest fluid neighbor:

   .. math::
      p_{i,j} = p_{\text{nearest fluid}(i,j)} \quad \forall (i,j) \text{ where } M_{i,j} = 1

   This 1-to-1 mapping allows a vectorized update of the pressure field during the Poisson iterations, ensuring computational efficiency and a smooth pressure field inside the solid body for visualization.


References
----------

.. [1] Hirsch, C. (2007). *Chapter 7 - Consistency, Stability and Error Analysis of Numerical Schemes*. In *Numerical Computation of Internal and External Flows (Second Edition)* (pp. 283-335). Oxford: Butterworth-Heinemann. doi: https://doi.org/10.1016/B978-075066594-0/50049-7