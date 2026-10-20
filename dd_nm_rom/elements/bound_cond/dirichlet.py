import numpy as np
import scipy.sparse as sp

from typing import Dict

from .. import mesh as mesh_mod
from .utils import get_axis_method, SIDES


class DirichletBC(object):
  """
  Dirichlet boundary conditions class.

  This class handles the construction and application of Dirichlet boundary 
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
    self.name = "dirichlet"
    self.nu = nu
    self.mesh = mesh
    self.funval = funval
    self.f = None
    self.op = None
    # Control variables
    self.built = False
    self.update_built = True
    self.advection = advection

  # Building
  # ===================================
  def is_built(self) -> None:
    """
    Check if the boundary conditions have been built.

    :raises ValueError: If the boundary conditions have not been built.
    """
    if (not self.built):
      raise ValueError("BC not built. Please, call 'build' method first.")

  def build(self) -> None:
    """
    Build the boundary condition operators and source terms.
    """
    self.f = {}
    # Loop over variables
    for k in ("u", "v"):
      f_k = {}
      # Loop over sides
      for (side, sfunval) in self.funval.items():
        axis, method = get_axis_method(side)
        f_k[side] = self._build_src(
          funval=sfunval[k],
          axis=axis,
          method=method
        )
      self.f[k] = {"D": self._compose_dif_src(f_k)}
      if self.advection:
        self.f[k]["A"] = self._compose_adv_src(f_k)
    # Ghost-value coefficients on (u0, u1) -- see _build_src/_build_op for
    # the derivation. Needed because the true boundary is never exactly at
    # the first real node (x0) for either with_bounds setting, so the ghost
    # value can't be read off directly -- it has to be reconstructed from
    # u0, u1, and the true boundary value g via Taylor matching.
    self.op = {"x": {}, "y": {}}
    for side in self.funval.keys():
      axis, method = get_axis_method(side)
      self.op[axis][method] = self._build_op(axis=axis, method=method)
    if self.update_built:
      self.built = True

  def _eval_at_true_edge(
    self,
    funval: callable,
    axis: str = "x",
    method: str = "fwd"
  ) -> np.ndarray:
    """
    Evaluate funval at the TRUE physical boundary (self.mesh.phylim), spread
    across the full grid. NOT self.mesh.lim -- that is a fixed, shift-
    independent offset point (phylim + 0.5h) that only coincides with a
    geometrically meaningful location (the ghost point) for with_bounds=
    False, and with neither the true edge nor the ghost point for
    with_bounds=True. Shared by DirichletBC and NeumannBC (Neumann calls
    this directly rather than through DirichletBC._build_src, so it isn't
    affected by Dirichlet's own ghost-value scaling in _build_src below).

    :param funval: Function value for the boundary condition.
    :type funval: callable
    :param axis: Axis for the boundary condition ('x' or 'y').
    :type axis: str
    :param method: Method for building the boundary condition ('fwd' or 'bwd').
    :type method: str

    :return: Boundary value spread across the grid, zero off that side.
    :rtype: np.ndarray
    """
    n = self.mesh.n
    index = 0 if (method == "fwd") else -1
    e0 = np.zeros(n[axis])
    e0[index] = 1.0
    true_edge = self.mesh.phylim[axis][index]
    if (axis == "x"):
      x = np.full(n["y"], true_edge)
      y = self.mesh.nodes_1d["y"]
      return np.kron(funval(x, y), e0)
    else:
      x = self.mesh.nodes_1d["x"]
      y = np.full(n["x"], true_edge)
      return np.kron(e0, funval(x, y))

  def _build_src(
    self,
    funval: callable,
    axis: str = "x",
    method: str = "fwd"
  ) -> np.ndarray:
    """
    Build the source term for the given boundary condition value.

    Ghost = p*u0 + q*u1 + r*g, derived via Taylor expansion about the TRUE
    boundary (matching value/slope/curvature -- see _build_op for p, q).
    This method supplies the r*g piece (a known constant, so it goes into
    the source term); _build_op supplies the (p, q) coefficients on the
    real unknowns u0, u1 (state-dependent, so they modify the operator
    matrix instead). MMS-validated (full-solve convergence test): r=8/3 for
    with_bounds=True (d0=0.5h) and r=8/15 for with_bounds=False (d0=1.5h)
    both give clean 2nd-order convergence; the previous version (ghost = g
    evaluated at self.mesh.lim directly, no u0/u1 correction) only achieves
    1st order at BOTH settings.

    :param funval: Function value for the boundary condition.
    :type funval: callable
    :param axis: Axis for the boundary condition ('x' or 'y').
    :type axis: str
    :param method: Method for building the boundary condition ('fwd' or 'bwd').
    :type method: str

    :return: Source term array.
    :rtype: np.ndarray
    """
    g = self._eval_at_true_edge(funval, axis, method)
    r = (8.0/3.0) if (self.mesh.shift == 0) else (8.0/15.0)
    return r * g
    # n, lim = self.mesh.n, self.mesh.lim               # OLD: only 1st-order
    # index = 0 if (method == "fwd") else -1             # accurate at the
    # e0 = np.zeros(n[axis]); e0[index] = 1.0             # boundary for
    # if (axis == "x"):                                   # BOTH with_bounds
    #   x = np.full(n["y"], lim["x"][index])               # settings
    #   y = self.mesh.nodes_1d["y"]                        # (verified via
    #   return np.kron(funval(x, y), e0)                   # full-solve MMS
    # else:                                                 # convergence
    #   x = self.mesh.nodes_1d["x"]                         # test, not just
    #   y = np.full(n["x"], lim["y"][index])                # local
    #   return np.kron(e0, funval(x, y))                    # residual)

  def _build_op(
    self,
    axis: str = "x",
    method: str = "fwd"
  ) -> sp.spmatrix:
    """
    Build the ghost-value coefficients on (u0, u1) -- see _build_src.

    Derived by Taylor-matching value/slope/curvature about the true
    boundary, using the real distances d0 = (0.5+shift)*h from x0 to the
    true edge:
      with_bounds=True  (shift=0, d0=0.5h): ghost = -2*u0 + (1/3)*u1 + (8/3)*g
      with_bounds=False (shift=1, d0=1.5h): ghost = (2/3)*u0 - (1/5)*u1 + (8/15)*g

    :param axis: Axis for the boundary condition ('x' or 'y').
    :type axis: str
    :param method: Method for building the boundary condition ('fwd' or 'bwd').
    :type method: str

    :return: The operator matrix.
    :rtype: sp.spmatrix
    """
    n = self.mesh.n[axis]
    index = 0 if (method == "fwd") else -1
    e0 = np.zeros(n)
    e0[index] = 1.0
    I0 = sp.diags(e0, 0)
    e = np.ones(n)
    if (self.mesh.shift == 0):
      p, q = -2.0, 1.0/3.0
    else:
      p, q = 2.0/3.0, -1.0/5.0
    op = sp.spdiags([p*e, q*e], [0, 1], n, n)
    if (method == "bwd"):
      op = op.T
    return I0 @ op

  def _compose_adv_src(
    self,
    values: Dict[str, np.ndarray]
  ) -> Dict[str, np.ndarray]:
    """
    Compose the advection source term.

    :param values: Dictionary of source term values for each side.
    :type values: Dict[str, np.ndarray]

    :return: Advection source term for each axis.
    :rtype: Dict[str, np.ndarray]
    """
    f = {}
    for axis in ("x", "y"):
      si = SIDES[axis]
      fi = values[si[0]] - values[si[1]]
      f[axis] = -(0.5/self.mesh.h[axis]) * fi
    return f

  def _compose_dif_src(
    self,
    values: Dict[str, np.ndarray]
  ) -> Dict[str, np.ndarray]:
    """
    Compose the diffusion source term.

    :param values: Dictionary of source term values for each side.
    :type values: Dict[str, np.ndarray]

    :return: Diffusion source term for each axis.
    :rtype: Dict[str, np.ndarray]
    """
    f = 0.0
    for axis in ("x", "y"):
      si = SIDES[axis]
      fi = values[si[0]] + values[si[1]]
      f = f + (self.nu/self.mesh.h[axis]**2) * fi
    return f
