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