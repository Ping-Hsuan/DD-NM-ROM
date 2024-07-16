import numpy as np
import scipy.sparse as sp

from .di import DirichletBC
from .utils import get_axis_method


class NeumannBC(DirichletBC):

  # Initialization
  # ===================================
  def __init__(
    self,
    nu,
    mesh,
    funval
  ):
    super(NeumannBC, self).__init__(nu, mesh, funval)
    self.name = "neumann"
    self.update_built = False

  # Build
  # ===================================
  def build(self):
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
    funval,
    axis="x",
    method="fwd"
  ):
    f = super(NeumannBC, self)._build_src(funval, axis, method)
    sign = -1 if (method == "fwd") else 1
    f *= (sign * 2.0/3.0 * self.mesh.h[axis])
    return f

  def _build_op(
    self,
    axis="x",
    method="fwd"
  ):
    n = self.mesh.n[axis]
    # Remove inner points
    index = 0 if (method == "fwd") else -1
    e0 = np.zeros(n)
    e0[index] = 1.0
    I0 = sp.diags(e0, 0)
    # Assemble
    e = np.ones(n)
    op = sp.spdiags([4.0/3.0*e, -1.0/3.0*e], [0, 1], n, n)
    if (method == "bwd"):
      op = op.T
    return I0 @ op
