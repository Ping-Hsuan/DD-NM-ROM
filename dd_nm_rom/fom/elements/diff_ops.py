import numpy as np
import scipy.sparse as sp

from typing import List, Union

from .mesh import MeshDD, MeshMono
from .bound_cond import DirichletBC, NeumannBC


class DiffOperators(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    nu: float,
    bc: Union[DirichletBC, NeumannBC],
    mesh: Union[MeshDD, MeshMono]
  ) -> None:
    self.nu = nu
    self.bc = bc
    self.mesh = mesh
    self.built = False

  def is_built(self):
    if (not self.built):
      raise ValueError(
        "Differential operators not built. Please, call 'build' method first."
      )

  # Build
  # ===================================
  def build(self) -> None:
    self.ops = {"D": 0.0}
    for axis in ("x", "y"):
      h = self.mesh.h[axis]
      Ai = self._build_op(axis, stencil=[-1,  0, 1], diags=[-1, 0, 1])
      Di = self._build_op(axis, stencil=[ 1, -2, 1], diags=[-1, 0, 1])
      self.ops[f"A{axis}"] = (-0.5/h) * Ai
      self.ops["D"] = self.ops["D"] + (self.nu/h**2) * Di
    self.built = True

  def _build_op(
    self,
    axis: str,
    stencil: List[int],
    diags: List[int]
  ) -> sp.spmatrix:
    n = self.mesh.n
    e = np.ones(n[axis])
    # Operator
    op = sp.spdiags([c*e for c in stencil], diags, n[axis], n[axis])
    # Update with BC
    if (self.bc.op is not None):
      for (method, bc_op) in self.bc.op[axis].items():
        index = 0 if (method == "fwd") else -1
        op += stencil[index] * bc_op
    # Map over 2D grid
    if (axis == "x"):
      op = sp.kron(sp.eye(n["y"]), op)
    else:
      op = sp.kron(op, sp.eye(n["x"]))
    return op.tocsr()
