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
    upwind: bool,
  ) -> None:
    self.nu = nu
    self.bc = bc
    self.mesh = mesh
    self.built = False
    self.upwind = upwind

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

    if self.upwind:
        return self.build_upwind_2nd()

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


  def build_upwind_2nd(self) -> None:
      """
      Build differential operators using second-order upwind scheme,
      assuming positive flow direction for both x and y.
      """
      self.ops = {"D": 0.0}
      for axis in ("x", "y"):
          h = self.mesh.h[axis]
          # Second-order backward difference for first derivative (for positive flow)
          # Formula: (3f_i - 4f_{i-1} + f_{i-2})/(2h)
          upwind_stencil = [1, -4, 3, 0]  # Coefficients for points i-2, i-1, i, i+1
          upwind_diags = [-2, -1, 0, 1]   # Diagonal positions
          Ai = self._build_extended_op(axis, upwind_stencil, upwind_diags)
          self.ops[f"A{axis}"] = (-1.0/(2*h)) * Ai
          # Standard second-order central for diffusion
          Di = self._build_op(axis, stencil=[1, -2, 1], diags=[-1, 0, 1])
          self.ops["D"] = self.ops["D"] + (self.nu/h**2) * Di
      self.built = True

  def _build_extended_op(
      self,
      axis: str,
      stencil: List[float],
      diags: List[int]
  ) -> sp.spmatrix:
      """
      Build a differential operator for wider stencils (second-order upwind).

      :param axis: The axis for which to build the operator ('x' or 'y')
      :param stencil: Coefficients for the finite difference stencil
      :param diags: Diagonals for the sparse matrix representation
      :return: The constructed sparse matrix operator
      """
      n = self.mesh.n
      e = np.ones(n[axis])
      # Create basic operator with the stencil
      op = sp.spdiags([c*e for c in stencil], diags, n[axis], n[axis])

      # Handle periodic boundary conditions for wider stencil
      if isinstance(self.bc, bc_mod.PeriodicBC):

          op = op.tolil()
          # For positive flow with backward bias [1, -4, 3, 0]
          # We need to connect:
          # - Point 0 needs data from points n-2 and n-1
          # - Point 1 needs data from point n-1
          op[0, n[axis]-2] = stencil[0]  # Connect point 0 to point n-2
          op[0, n[axis]-1] = stencil[1]  # Connect point 0 to point n-1
          op[1, n[axis]-1] = stencil[0]  # Connect point 1 to point n-1
      elif self.bc.op is not None:
          # For non-periodic boundaries, apply standard BC handling
          for (method, bc_op) in self.bc.op[axis].items():
              index = 0 if (method == "fwd") else -1
              op += stencil[index] * bc_op

      # Map to 2D grid as before
      if axis == "x":
          op = sp.kron(sp.eye(n["y"]), op)
      else:
          op = sp.kron(op, sp.eye(n["x"]))

      return op.tocsr()