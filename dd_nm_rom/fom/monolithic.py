import numpy as np
import scipy.sparse as sp

from time import time
from dd_nm_rom import ops, solvers
from typing import Dict, List, Tuple, Union

from .elements import *


class Burgers2D(object):
  """
  Generate FOM for 2D Burgers equation with Dirichlet BC on the rectangle
  determined by x_lim x y_lim using finite differences.

  inputs:
  nx: number of grid points in x method
  ny: number of grid points in y method
  x_lim: x_lim[0] = x-coordinate of left boundary
         x_lim[1] = x-coordinate of right boundary
  y_lim: y_lim[0] = y-coordinate of bottom boundary
         y_lim[1] = y-coordinate of top boundary
  nu: positive parameter corresponding to nu
  self.u_bc: Dirichlet BC function for u states
  self.v_bc: Dirichlet BC function for v states

  methods:
  set_bc: update boundary condition data
  residual: compute residual of PDE
  res_jac: compute jacobian of residual with respect to [u, v]
  solve: solves for the state u and v using Newton"s method.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: Union[MeshDD, MeshMono],
    nu: float
  ) -> None:
    # Mesh
    self.mesh = mesh
    # Viscosity
    self.nu = nu
    # Integration
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Runtime
    self.runtime = {k: 0.0 for k in ("total", "linalg", "rhs_jac")}
    self.built = False

  # Building
  # ===================================
  def is_built(self) -> None:
    self.mesh.is_built()
    if (not self.built):
      raise ValueError(
        "FOM model not built. Please, call 'build' method first."
      )

  def build(
    self,
    field
  ):
    self.mesh.is_built()
    # BC
    # -------------
    bc_cls = NeumannBC if (field.bc_type == "neumann") else DirichletBC
    self.bc = bc_cls(
      nu=self.nu,
      mesh=self.mesh,
      funval=field.get_bc_funval()
    )
    self.bc.build()
    self.bc_f = self.bc.f
    # Operators
    # -------------
    self.diff_ops = DiffOperators(
      nu=self.nu,
      bc=self.bc,
      mesh=self.mesh
    )
    self.diff_ops.build()
    self.ops = self.diff_ops.ops
    self.ops_names = list(self.ops.keys())
    self.iden = sp.eye(self.get_ndof()).tocsr()
    self.built = True

  def get_ndof(self) -> int:
    return 2*self.mesh.nxy

  # RHS/Jacobian
  # ===================================
  def rhs_jac(
    self,
    x: np.ndarray
  ) -> Tuple[np.ndarray, sp.spmatrix]:
    start = time()
    rhs, jac = self._rhs_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      rhs = x - self.x_old - self.dt*rhs
      jac = self.iden - self.dt*jac
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["rhs_jac"] += delta
    return rhs, jac

  def _rhs_jac(
    self,
    x: np.ndarray
  ) -> Tuple[np.ndarray, sp.spmatrix]:
    # Extract u and v
    uv, uv_diag = self.extract_uv(x)
    # Action of advection operator on vectors
    adv_act = {}
    for axis in ("x", "y"):
      adv_act[axis] = {}
      for k in ("u", "v"):
        adv_act[axis][k] = self.ops[f"A{axis}"] @ uv[k] \
                         - self.bc_f[k]["A"][axis]
    # Compute RHS
    dx = []
    for k in ("u", "v"):
      dx_k = uv_diag["u"] @ adv_act["x"][k] \
           + uv_diag["v"] @ adv_act["y"][k] \
           + self.ops["D"] @ uv[k] + self.bc_f[k]["D"]
      dx.append(dx_k)
    rhs = np.concatenate(dx)
    # Compute Jacobian
    jac_xx = uv_diag["u"] @ self.ops["Ax"] \
           + uv_diag["v"] @ self.ops["Ay"] \
           + self.ops["D"]
    jac_uu = ops.sp_diag(adv_act["x"]["u"]) + jac_xx
    jac_uv = ops.sp_diag(adv_act["y"]["u"])
    jac_vu = ops.sp_diag(adv_act["x"]["v"])
    jac_vv = ops.sp_diag(adv_act["y"]["v"]) + jac_xx
    jac = sp.bmat(
      [[jac_uu, jac_uv],
       [jac_vu, jac_vv]],
      format="csr"
    )
    return rhs, jac

  def extract_uv(
    self,
    x: np.ndarray,
    diag: bool = True
  ) -> Union[
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    Dict[str, np.ndarray]
  ]:
    uv = {"u": x[:self.mesh.nxy], "v": x[self.mesh.nxy:]}
    if diag:
      uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
      return uv, uv_diag
    else:
      return uv

  # Solution
  # ===================================
  def solve(
    self,
    x0: Union[np.ndarray, None] = None,
    dt: float = 0.0,
    nt: int = 1,
    steady: bool = True,
    tol: float = 1e-8,
    maxit: int = 50,
    stepsize_min: float = 1e-10,
    verbose: bool = False
  ) -> Tuple[Dict[str, np.ndarray], Union[np.ndarray, List[np.ndarray]], bool]:
    """
    Solves for the u and v states of the FOM using Newton"s method.

    inputs:
    u0: (nx*ny,) initial u vector
    v0: (nx*ny,) initial v vector
    tol: [optional] stopping tolerance for Newton solver. Default is 1e-10
    maxit: [optional] max number of iterations for newton solver. Default is 100
    print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False

    outputs:
    u: (nx*ny,) u final solution vector
    v: (nx*ny,) v final solution vector
    res_vecs: (it, nx*ny) array where res_vecs[i] is the PDE residual evaluated at the ith Newton iteration
    """
    self.is_built()
    self.runtime = ops.map_nested_dict(self.runtime, lambda _: 0.0)
    # Initialize solution
    start = time()
    if (x0 is None):
      x0 = np.zeros(self.get_ndof())
    self.runtime["total"] += time()-start
    # Initialize solver
    solver = solvers.Newton(
      model=self,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, rhs, *_, flag = solver(x0, dt, nt)
    # Return solution
    uv = self.extract_uv(x, diag=False)
    converged = True if (flag[-1] == 0) else False
    return uv, rhs, converged
