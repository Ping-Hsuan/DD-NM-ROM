import numpy as np
import scipy.sparse as sp

from typing import Dict

from .. import mesh as mesh_mod
from .dirichlet import DirichletBC
from .utils import get_axis_method


class NeumannBC(DirichletBC):
  """
  Neumann boundary conditions class.

  This class handles the construction and application of Neumann boundary 
  conditions for a given mesh and boundaries function values.

  :param nu: Viscosity or diffusion coefficient.
  :type nu: float
  :param mesh: Mesh object on which the boundary conditions are applied.
  :type mesh: MESH_TYPES
  :param funval: A dictionary defining the function values for 
                 the boundary conditions. The keys areand the 
                 values are dictionaries with axis ('x' or 'y') 
                 and method ('fwd' or 'bwd').
  :type funval: Dict[str, Dict[str, callable]]
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    nu: float,
    mesh: mesh_mod.MESH_TYPES,
    funval: Dict[str,Dict[str,callable]],
    advection: bool = True
  ) -> None:
    super(NeumannBC, self).__init__(nu, mesh, funval, advection)
    self.name = "neumann"
    self.update_built = False

  # Building
  # ===================================
  def build(self) -> None:
    """
    Build the boundary condition operators and source terms.
    """
    super(NeumannBC, self).build()
    self.op = {"x": {}, "y": {}}
    for side in self.funval.keys():
      axis, method = get_axis_method(side)
      self.op[axis][method] = self._build_op(
        axis=axis,
        method=method
      )
    self.built = True

  def _build_src(
    self,
    funval: callable,
    axis: str = "x",
    method: str = "fwd"
  ) -> np.ndarray:
    """
    Build the source term for the given boundary condition value.

    :param funval: Function value for the boundary condition.
    :type funval: callable
    :param axis: Axis for the boundary condition ('x' or 'y').
    :type axis: str
    :param method: Method for building the boundary condition ('fwd' or 'bwd').
    :type method: str

    :return: Source term array.
    :rtype: np.ndarray
    """
    # Uses the shared _eval_at_true_edge helper directly, NOT
    # super()._build_src() -- DirichletBC's own _build_src now applies its
    # OWN ghost-value scaling (the r*g term), which must not leak in here.
    f = self._eval_at_true_edge(funval, axis, method)
    sign = -1 if (method == "fwd") else 1
    # Ghost-value scaling on the g-term, derived via Taylor expansion of u about the TRUE
    # physical boundary (not self.mesh.lim -- irrelevant for our homogeneous funval=0 cases,
    # but this is the physically correct reference point in general). d0 = distance from the
    # first real node x0 to the true edge = (0.5+shift)*h:
    #   shift=0 (with_bounds=True):  d0=0.5h -> ghost = u0 - h*g            (ratio: 1.0)
    #   shift=1 (with_bounds=False): d0=1.5h -> ghost = 1.5*u0 - 0.5*u1 - 0.5*h*g
    # MMS-validated (operator-consistency test, u=cos(2pi x)cos(2pi y), homogeneous Neumann):
    # shift=0 -> ratio=1.0 gives clean 2nd-order convergence (1.87,1.94,1.97,1.99);
    # shift=1 -> ratio=0.5 gives clean 2nd-order convergence (1.81,1.92,1.97,1.98).
    # The old (2/3) scaling converges at NEITHER shift value (rate ~0, non-convergent boundary
    # error) -- it does not correspond to either mesh convention's actual node geometry.
    ratio = 1.0 if (self.mesh.shift == 0) else 0.5
    f *= (sign * ratio * self.mesh.h[axis])
    # f *= (sign * 2.0/3.0 * self.mesh.h[axis])  # WRONG for both shift=0 and shift=1: MMS test
    # shows non-convergent (~0 rate) boundary error at either with_bounds setting.
    return f

  def _build_op(
    self,
    axis: str = "x",
    method: str = "fwd"
  ) -> sp.spmatrix:
    """
    Build the boundary condition operator.

    :param axis: Axis for the boundary condition ('x' or 'y').
    :type axis: str
    :param method: Method for building the boundary condition ('fwd' or 'bwd').
    :type method: str

    :return: The operator matrix.
    :rtype: sp.spmatrix
    """
    n = self.mesh.n[axis]
    # Remove inner points
    index = 0 if (method == "fwd") else -1
    e0 = np.zeros(n)
    e0[index] = 1.0
    I0 = sp.diags(e0, 0)
    # Assemble
    e = np.ones(n)
    # Ghost-value coefficients on (u0, u1), shift-dependent -- see _build_src for the full
    # derivation and MMS validation. shift=0 (with_bounds=True): ghost=1.0*u0 (simple mirror,
    # d0=0.5h). shift=1 (with_bounds=False): ghost=1.5*u0-0.5*u1 (d0=1.5h).
    if (self.mesh.shift == 0):
      op = sp.spdiags([1.0*e], [0], n, n)
    else:
      op = sp.spdiags([1.5*e, -0.5*e], [0, 1], n, n)
    # op = sp.spdiags([4.0/3.0*e, -1.0/3.0*e], [0, 1], n, n)  # WRONG for both shift=0 and
    # shift=1: MMS test shows non-convergent (~0 rate) boundary error at either with_bounds
    # setting -- doesn't correspond to either mesh convention's actual node geometry.
    if (method == "bwd"):
      op = op.T
    return I0 @ op
