import numpy as np
import scipy.sparse as sp

from typing import List

from . import mesh as mesh_mod
from . import bound_cond as bc_mod


class DiffOperators(object):
  """
  A class to build and manage differential operators for a given mesh 
  and boundary conditions.

  :param nu: Viscosity or diffusion coefficient.
  :type nu: float
  :param bc: Boundary conditions.
  :type bc: BC_TYPES
  :param mesh: Mesh object containing grid information.
  :type mesh: MESH_TYPES
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    nu: float,
    bc: bc_mod.BC_TYPES,
    mesh: mesh_mod.MESH_TYPES,
  ) -> None:
    self.nu = nu
    self.bc = bc
    self.mesh = mesh
    self.built = False

  # Building
  # ===================================
  def is_built(self) -> None:
    """
    Check if the differential operators have been built.

    :raises ValueError: If the differential operators are not built.
    """
    if (not self.built):
      raise ValueError(
        "Differential operators not built. Please, call 'build' method first."
      )

  def build(self) -> None:
    """
    Build the differential operators for the mesh and boundary conditions.
    """
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
    """
    Build a differential operator for a given axis.

    :param axis: The axis for which to build the operator ('x' or 'y').
    :type axis: str
    :param stencil: Coefficients for the finite difference stencil.
    :type stencil: List[int]
    :param diags: Diagonals for the sparse matrix representation.
    :type diags: List[int]

    :return: The constructed sparse matrix operator.
    :rtype: sp.spmatrix
    """
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
