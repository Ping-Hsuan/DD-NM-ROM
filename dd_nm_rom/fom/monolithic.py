import numpy as np
import scipy.sparse as sp


from time import time
from dd_nm_rom import ops, solvers
from dd_nm_rom import field as field_mod
from typing import Dict, List, Tuple, Union

from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import DiffOperators
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod

# Data types
RES_JAC_TYPE = Tuple[np.ndarray, sp.spmatrix]
UV_TYPE = Dict[str, Union[np.ndarray, sp.spmatrix]]
SOL_TYPE = Tuple[UV_TYPE, Union[np.ndarray, List[np.ndarray]], bool]


class Burgers2D(object):
  """
  Generate FOM for 2D Burgers equation solutions.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    nu: float,
    upwind: bool = False,
    upwind_order: int =1,
    compact: bool = False
  ) -> None:
    # Mesh
    self.mesh = mesh
    self.mesh.is_built()
    # Viscosity
    self.nu = nu
    # Integration
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Runtime
    self.runtime = {k: 0.0 for k in ("total", "lin_solve", "res_jac")}
    self.built = False
    # Scheme
    self.upwind = upwind
    self.upwind_order = upwind_order
    self.compact = compact

  # Building
  # ===================================
  def is_built(self) -> None:
    if (not self.built):
      raise ValueError(
        "FOM model not built. Please, call 'build' method first."
      )

  def build(
    self,
    field: field_mod.FIELD_TYPES
  ) -> None:
    # BC
    self.bc = self.build_bc(field)
    self.bc_f = self.bc.f
    # Operators
    self.diff_ops = DiffOperators(
      nu=self.nu,
      bc=self.bc,
      mesh=self.mesh,
      upwind=self.upwind,
      upwind_order=self.upwind_order
    )
    self.diff_ops.build()
    self.ops = self.diff_ops.ops
    self.ops_names = list(self.ops.keys())
    # Identities
    self.iden = sp.eye(self.mesh.nxy).tocsr()
    self.iden_uv = sp.eye(self.get_ndof()).tocsr()
    self.built = True

  def build_bc(
    self,
    field: field_mod.FIELD_TYPES
  ) -> bc_mod.BC_TYPES:
    if (field.bc_type == "dirichlet"):
      bc_cls = bc_mod.DirichletBC
    elif (field.bc_type == "neumann"):
      bc_cls = bc_mod.NeumannBC
    else:
      bc_cls = bc_mod.PeriodicBC
    bc = bc_cls(
      nu=self.nu,
      mesh=self.mesh,
      funval=field.get_bc_funval()
    )
    bc.build()
    return bc

  def get_ndof(self) -> int:
    return 2*self.mesh.nxy

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    start = time()
    if self.compact:
      res, jac = self.compute_res_jac_compact(x)
    else:
      res, jac = self.compute_res_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      res = x - self.x_old - self.dt*res
      jac = self.iden_uv - self.dt*jac
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["res_jac"] += delta
    return res, jac

  def compute_res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    # Extract u and v
    uv, uv_diag = self.extract_uv(x, diag=True)
    # Action of advection operator on vectors
    adv_act = {}
    for axis in ("x", "y"):
      adv_act[axis] = {}
      for k in ("u", "v"):
        adv_act[axis][k] = self.ops[f"A{axis}"] @ uv[k] \
                         - self.bc_f[k]["A"][axis]
    # Compute residual
    dx = []
    for k in ("u", "v"):
      dx_k = uv_diag["u"] @ adv_act["x"][k] \
           + uv_diag["v"] @ adv_act["y"][k] \
           + self.ops["D"] @ uv[k] + self.bc_f[k]["D"]
      dx.append(dx_k)
    res = np.concatenate(dx)
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
    return res, jac

  def extract_uv(
    self,
    x: np.ndarray,
    diag: bool = True
  ) -> Union[Tuple[UV_TYPE, UV_TYPE], UV_TYPE]:
    uv = {"u": x[:self.mesh.nxy], "v": x[self.mesh.nxy:]}
    if diag:
      uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
      return uv, uv_diag
    else:
      return uv

  # Solving
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
    iostep: int = 1,
    verbose: bool = False
  ) -> SOL_TYPE:
    """
    Solves for the u and v states of the FOM using Newton"s method.
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
      iostep=iostep,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, res, *_, flag = solver(x0, dt, nt)
    # Return solution
    uv = self.extract_uv(x, diag=False)
    converged = True if (flag[-1] == 0) else False
    return uv, res, converged

  def compute_res_jac_compact(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    """Compute residual and jacobian for the compact upwind scheme
    """
    # Extract u and v
    uv, uv_diag = self.extract_uv(x, diag=True)
    # Action of advection operator on vectors
    adv_act = {}
    for axis in ("x", "y"):
      # Determine which velocity component to check
      vel_component = "u" if axis == "x" else "v"
      vel = uv[vel_component]

      # Create masks for positive and negative velocities
      pos_mask = (vel >= 0)
      neg_mask = (vel < 0)

      adv_act[axis] = {}
      for k in ("u", "v"):
        # Apply both operators to the solution vector
        pos_result = self.ops[f"A{axis}_pos"] @ uv[k]
        neg_result = self.ops[f"A{axis}_neg"] @ uv[k]

        if np.all(pos_mask):
            result = pos_result
        elif np.all(neg_mask):
            result = neg_result
        else:
            result = np.zeros_like(pos_result)
            result[pos_mask] = pos_result[pos_mask]
            result[neg_mask] = neg_result[neg_mask]

        # Apply boundary condition adjustments
        adv_act[axis][k] = result - self.bc_f[k]["A"][axis]

    # Compute residual
    dx = []
    for k in ("u", "v"):
      dx_k = uv_diag["u"] @ adv_act["x"][k] \
           + uv_diag["v"] @ adv_act["y"][k] \
           + self.ops["D"] @ uv[k] + self.bc_f[k]["D"]
      dx.append(dx_k)
    res = np.concatenate(dx)

    # Compute Jacobian with direction-dependent operators
    jac_operators = {}
    for axis in ("x", "y"):
      vel_component = "u" if axis == "x" else "v"
      vel = uv[vel_component]
      pos_mask = (vel >= 0).astype(float)
      neg_mask = 1-pos_mask #(vel < 0)

      # Diagonal selection matrices
      P = sp.diags(pos_mask)  # shape (n, n)
      N = sp.diags(neg_mask)  # shape (n, n)

      A_pos = self.ops[f"A{axis}_pos"]
      A_neg = self.ops[f"A{axis}_neg"]

      # Efficient blending
      A_blend = P @ A_pos + N @ A_neg

      jac_operators[f"A{axis}"] = A_blend

    # Compute Jacobian with blended operators
    jac_xx = uv_diag["u"] @ jac_operators["Ax"] \
          + uv_diag["v"] @ jac_operators["Ay"] \
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
    return res, jac

class Poisson2D(object):
  """
  Generate FOM for 2D Poisson equation solutions.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    nu: float
  ) -> None:
    # Mesh
    self.mesh = mesh
    self.mesh.is_built()
    # Viscosity
    self.nu = nu
    # Integration
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Runtime
    self.runtime = {k: 0.0 for k in ("total", "lin_solve", "res_jac")}
    self.built = False
    # Force term in the Poisson
    self.f = None

  # Building
  # ===================================
  def is_built(self) -> None:
    if (not self.built):
      raise ValueError(
        "FOM model not built. Please, call 'build' method first."
      )

  def build(
    self,
    field: field_mod.FIELD_TYPES,
    force: np.ndarray = None
  ) -> None:
    # BC
    self.bc = self.build_bc(field)
    self.bc_f = self.bc.f
    # Operators
    self.diff_ops = DiffOperators(
      nu=self.nu,
      bc=self.bc,
      mesh=self.mesh,
      upwind=False,
      upwind_order=1
    )
    self.diff_ops.build()
    self.ops = self.diff_ops.ops
    self.ops_names = list(self.ops.keys())
    # Identities
    self.iden = sp.eye(self.mesh.nxy).tocsr()
    self.iden_uv = sp.eye(self.get_ndof()).tocsr()

    # Set forcing term
    if force is None:
      # Default to zero force if not provided
      self.f = np.zeros(self.get_ndof())
    else:
      self.f = force
    self.built = True

  def build_bc(
    self,
    field: field_mod.FIELD_TYPES
  ) -> bc_mod.BC_TYPES:
    if (field.bc_type == "dirichlet"):
      bc = bc_mod.DirichletBC(
        nu=self.nu,
        mesh=self.mesh,
        funval=field.get_bc_funval(),
        advection=False
      )
    elif (field.bc_type == "neumann"):
      bc = bc_mod.NeumannBC(
        nu=self.nu,
        mesh=self.mesh,
        funval=field.get_bc_funval(),
        advection=False
      )
    elif (field.bc_type == "mixed"):
      bc = bc_mod.MixedBC(
        nu=self.nu,
        mesh=self.mesh,
        funval=field.get_bc_funval(),
        bc_spec=field.get_bc_spec(),
        advection=False
      )
    else:
      bc = bc_mod.PeriodicBC(
        nu=self.nu,
        mesh=self.mesh,
        funval=field.get_bc_funval(),
        advection=False
      )
    bc.build()
    return bc

  def get_ndof(self) -> int:
    return 2*self.mesh.nxy

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    start = time()
    res, jac = self.compute_res_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      res = x - self.x_old - self.dt*res
      jac = self.iden_uv - self.dt*jac
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["res_jac"] += delta
    return res, jac

  def compute_res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    # Extract u and v
    uv, uv_diag = self.extract_uv(x, diag=True)

    # Split force into u and v components
    f_uv = {"u": self.f[:self.mesh.nxy], "v": self.f[self.mesh.nxy:]}
#   # Action of advection operator on vectors
#   adv_act = {}
#   for axis in ("x", "y"):
#     adv_act[axis] = {}
#     for k in ("u", "v"):
#       adv_act[axis][k] = self.ops[f"A{axis}"] @ uv[k] \
#                        - self.bc_f[k]["A"][axis]
    # Compute residual
    dx = []
    for k in ("u", "v"):
      dx_k = self.ops["D"] @ uv[k] + self.bc_f[k]["D"] - f_uv[k]
      dx.append(dx_k)
    res = np.concatenate(dx)
    # Compute Jacobian
    jac_xx = self.ops["D"]
    jac_uu = jac_xx
    jac_uv = sp.csr_matrix((self.mesh.nxy, self.mesh.nxy))
    jac_vu = sp.csr_matrix((self.mesh.nxy, self.mesh.nxy))
    jac_vv = jac_xx
    jac = sp.bmat(
      [[jac_uu, jac_uv],
       [jac_vu, jac_vv]],
      format="csr"
    )
    return res, jac

  def extract_uv(
    self,
    x: np.ndarray,
    diag: bool = True
  ) -> Union[Tuple[UV_TYPE, UV_TYPE], UV_TYPE]:
    uv = {"u": x[:self.mesh.nxy], "v": x[self.mesh.nxy:]}
    if diag:
      uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
      return uv, uv_diag
    else:
      return uv

  # Solving
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
    iostep: int = 1,
    verbose: bool = False
  ) -> SOL_TYPE:
    """
    Solves for the u and v states of the FOM using Newton"s method.
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
      iostep=iostep,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, res, *_, flag = solver(x0, dt, nt)
    # Return solution
    uv = self.extract_uv(x, diag=False)
    converged = True if (flag[-1] == 0) else False
    return uv, res, converged


class AdvDiff2D(object):
  """
  Generate FOM for the 2D unsteady linear advection-diffusion equation:

    du/dt + cx*du/dx + cy*du/dy - nu*Lap(u) = f

  Unlike Burgers2D, the advection speed (cx, cy) is a fixed pair of
  scalars (either passed in directly or read from `field.get_advection()`),
  not the state itself -- so the residual is linear in u/v and the two
  components are decoupled (same structure as Poisson2D, plus advection
  operator terms).

  By default, Dirichlet BC is imposed exactly like Poisson2D/Burgers2D:
  plain ghost-point elimination, with the boundary value folded additively
  into the residual (bc_f) -- every node, including the outermost ring,
  remains a genuine unknown. Passing `pin_boundary=True` additionally
  freezes the outermost node ring at its initial value for all time
  (matching the convention used by the standalone
  examples/LLM_data/scripts/generate_linear_unsteady.py generator, kept
  only for reproducing/comparing against data generated that way).
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    nu: float,
    cx: float = 1.0,
    cy: float = 0.0,
    pin_boundary: bool = False
  ) -> None:
    # Mesh
    self.mesh = mesh
    self.mesh.is_built()
    # Viscosity / advection
    self.nu = nu
    self.cx, self.cy = float(cx), float(cy)
    self.pin_boundary = bool(pin_boundary)
    # Integration
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Runtime
    self.runtime = {k: 0.0 for k in ("total", "lin_solve", "res_jac")}
    self.built = False
    # Force term
    self.f = None

  # Building
  # ===================================
  def is_built(self) -> None:
    if (not self.built):
      raise ValueError(
        "FOM model not built. Please, call 'build' method first."
      )

  def build(
    self,
    field: field_mod.FIELD_TYPES,
    force: np.ndarray = None
  ) -> None:
    # BC
    self.bc = self.build_bc(field)
    self.bc_f = self.bc.f
    # Advection coefficients (fixed per case, sampled by the field if it
    # provides them)
    if hasattr(field, "get_advection"):
      self.cx, self.cy = field.get_advection()
    # Operators
    self.diff_ops = DiffOperators(
      nu=self.nu,
      bc=self.bc,
      mesh=self.mesh,
      upwind=False,
      upwind_order=1
    )
    self.diff_ops.build()
    self.ops = self.diff_ops.ops
    self.ops_names = list(self.ops.keys())
    # Identities
    self.iden = sp.eye(self.mesh.nxy).tocsr()
    self.iden_uv = sp.eye(self.get_ndof()).tocsr()

    # Set forcing term: explicit argument takes precedence, else pulled
    # from the field (force-driven unsteady datasets), else zero.
    if force is not None:
      self.f = force
    elif hasattr(field, "get_force"):
      self.f = field.get_force()
    else:
      self.f = np.zeros(self.get_ndof())

    # Outermost node ring: opt-in only (pin_boundary=True). Off by default,
    # so Dirichlet BC is imposed the same way as Poisson2D/Burgers2D (plain
    # ghost elimination, every node a genuine unknown). See class docstring.
    if self.pin_boundary and self.bc.name == "dirichlet":
      x, y = self.mesh.nodes_val.T
      tol = 1e-10
      self._bdry = ((x <= x.min()+tol) | (x >= x.max()-tol) |
                    (y <= y.min()+tol) | (y >= y.max()-tol))
    else:
      self._bdry = np.zeros(self.mesh.nxy, dtype=bool)
    # Constant (linear) spatial Jacobian, with pinned rows zeroed so that,
    # after the backward-Euler transform in res_jac(), those rows reduce
    # to the identity (i.e. x[bdry] stays fixed at its initial value).
    jac_xx = self.cx*self.ops["Ax"] + self.cy*self.ops["Ay"] + self.ops["D"]
    if self._bdry.any():
      jac_xx = jac_xx.tolil()
      jac_xx[self._bdry, :] = 0.0
      jac_xx = jac_xx.tocsr()
    self._jac_xx = jac_xx
    self.built = True

  def build_bc(
    self,
    field: field_mod.FIELD_TYPES
  ) -> bc_mod.BC_TYPES:
    if (field.bc_type == "dirichlet"):
      bc_cls = bc_mod.DirichletBC
    elif (field.bc_type == "neumann"):
      bc_cls = bc_mod.NeumannBC
    else:
      bc_cls = bc_mod.PeriodicBC
    bc = bc_cls(
      nu=self.nu,
      mesh=self.mesh,
      funval=field.get_bc_funval()
    )
    bc.build()
    return bc

  def get_ndof(self) -> int:
    return 2*self.mesh.nxy

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    start = time()
    res, jac = self.compute_res_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      res = x - self.x_old - self.dt*res
      jac = self.iden_uv - self.dt*jac
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["res_jac"] += delta
    return res, jac

  def compute_res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    # Extract u and v
    uv = self.extract_uv(x, diag=False)

    # Split force into u and v components
    f_uv = {"u": self.f[:self.mesh.nxy], "v": self.f[self.mesh.nxy:]}
    # Compute residual
    dx = []
    for k in ("u", "v"):
      dx_k = self.cx * (self.ops["Ax"] @ uv[k] - self.bc_f[k]["A"]["x"]) \
           + self.cy * (self.ops["Ay"] @ uv[k] - self.bc_f[k]["A"]["y"]) \
           + self.ops["D"] @ uv[k] + self.bc_f[k]["D"] + f_uv[k]
      dx_k[self._bdry] = 0.0
      dx.append(dx_k)
    res = np.concatenate(dx)
    # Jacobian (constant: linear, decoupled u/v; pinned rows already zeroed)
    jac_xx = self._jac_xx
    jac_uv = sp.csr_matrix((self.mesh.nxy, self.mesh.nxy))
    jac = sp.bmat(
      [[jac_xx, jac_uv],
       [jac_uv, jac_xx]],
      format="csr"
    )
    return res, jac

  def extract_uv(
    self,
    x: np.ndarray,
    diag: bool = True
  ) -> Union[Tuple[UV_TYPE, UV_TYPE], UV_TYPE]:
    uv = {"u": x[:self.mesh.nxy], "v": x[self.mesh.nxy:]}
    if diag:
      uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
      return uv, uv_diag
    else:
      return uv

  # Solving
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
    iostep: int = 1,
    verbose: bool = False
  ) -> SOL_TYPE:
    """
    Solves for the u and v states of the FOM using Newton's method.
    (Linear residual, so this converges in a single Newton step per
    time instant.)
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
      iostep=iostep,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, res, *_, flag = solver(x0, dt, nt)
    # Return solution
    uv = self.extract_uv(x, diag=False)
    converged = True if (flag[-1] == 0) else False
    return uv, res, converged


class ReactAdvDiff2D(object):
  """
  Generate FOM for the 2D unsteady linear reaction-advection-diffusion
  equation:

    du/dt + cx*du/dx + cy*du/dy - nu*Lap(u) + sigma_u*u = f
    dv/dt + cx*dv/dx + cy*dv/dy - nu*Lap(v) + sigma_v*v = f

  Identical to AdvDiff2D plus a reaction term. sigma_u/sigma_v may differ
  (independent per-component reaction) or be equal (the original shared-
  sigma design). Set directly via the constructor, or read from
  `field.get_reaction()`, which may return either a single scalar (shared
  by both components, backward-compatible with fields written against the
  original single-sigma design, e.g. ReactAdvDiffUnsteadyField) or a
  (sigma_u, sigma_v) pair (e.g. ReactAdvDiffUnsteadyFieldMaskedBump). Still
  linear and decoupled in u/v.

  Dirichlet BC handling mirrors AdvDiff2D: plain ghost-point elimination by
  default (same as Poisson2D/Burgers2D); pass `pin_boundary=True` to
  additionally freeze the outermost node ring, matching the convention used
  by examples/LLM_data/scripts/generate_linear_unsteady.py for
  ReactAdvDiff_unsteady.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    nu: float,
    sigma: float = 1.0,
    cx: float = 1.0,
    cy: float = 0.0,
    pin_boundary: bool = False
  ) -> None:
    # Mesh
    self.mesh = mesh
    self.mesh.is_built()
    # Viscosity / reaction / advection
    self.nu = nu
    self.sigma_u = self.sigma_v = float(sigma)
    self.cx, self.cy = float(cx), float(cy)
    self.pin_boundary = bool(pin_boundary)
    # Integration
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Runtime
    self.runtime = {k: 0.0 for k in ("total", "lin_solve", "res_jac")}
    self.built = False
    # Force term
    self.f = None

  # Building
  # ===================================
  def is_built(self) -> None:
    if (not self.built):
      raise ValueError(
        "FOM model not built. Please, call 'build' method first."
      )

  def build(
    self,
    field: field_mod.FIELD_TYPES,
    force: np.ndarray = None
  ) -> None:
    # BC
    self.bc = self.build_bc(field)
    self.bc_f = self.bc.f
    # Advection / reaction coefficients (fixed per case, sampled by the
    # field if it provides them)
    if hasattr(field, "get_advection"):
      self.cx, self.cy = field.get_advection()
    if hasattr(field, "get_reaction"):
      sigma_react = field.get_reaction()
      if isinstance(sigma_react, (tuple, list, np.ndarray)):
        self.sigma_u, self.sigma_v = float(sigma_react[0]), float(sigma_react[1])
      else:
        self.sigma_u = self.sigma_v = float(sigma_react)
    # Operators
    self.diff_ops = DiffOperators(
      nu=self.nu,
      bc=self.bc,
      mesh=self.mesh,
      upwind=False,
      upwind_order=1
    )
    self.diff_ops.build()
    self.ops = self.diff_ops.ops
    self.ops_names = list(self.ops.keys())
    # Identities
    self.iden = sp.eye(self.mesh.nxy).tocsr()
    self.iden_uv = sp.eye(self.get_ndof()).tocsr()

    # Set forcing term: explicit argument takes precedence, else pulled
    # from the field (force-driven unsteady datasets), else zero.
    if force is not None:
      self.f = force
    elif hasattr(field, "get_force"):
      self.f = field.get_force()
    else:
      self.f = np.zeros(self.get_ndof())

    # Outermost node ring: opt-in only (pin_boundary=True). Off by default,
    # so Dirichlet BC is imposed the same way as Poisson2D/Burgers2D (plain
    # ghost elimination, every node a genuine unknown). See AdvDiff2D.
    if self.pin_boundary and self.bc.name == "dirichlet":
      x, y = self.mesh.nodes_val.T
      tol = 1e-10
      self._bdry = ((x <= x.min()+tol) | (x >= x.max()-tol) |
                    (y <= y.min()+tol) | (y >= y.max()-tol))
    else:
      self._bdry = np.zeros(self.mesh.nxy, dtype=bool)
    # Constant (linear) spatial Jacobian, with pinned rows zeroed so that,
    # after the backward-Euler transform in res_jac(), those rows reduce
    # to the identity (i.e. x[bdry] stays fixed at its initial value).
    # Built separately per component since sigma_u may differ from sigma_v.
    def _build_jac_xx(sigma_k):
      jac_xx_k = (self.cx*self.ops["Ax"] + self.cy*self.ops["Ay"] + self.ops["D"]
                  - sigma_k*self.iden)
      if self._bdry.any():
        jac_xx_k = jac_xx_k.tolil()
        jac_xx_k[self._bdry, :] = 0.0
        jac_xx_k = jac_xx_k.tocsr()
      return jac_xx_k
    self._jac_xx_u = _build_jac_xx(self.sigma_u)
    self._jac_xx_v = _build_jac_xx(self.sigma_v)
    self.built = True

  def build_bc(
    self,
    field: field_mod.FIELD_TYPES
  ) -> bc_mod.BC_TYPES:
    if (field.bc_type == "dirichlet"):
      bc_cls = bc_mod.DirichletBC
    elif (field.bc_type == "neumann"):
      bc_cls = bc_mod.NeumannBC
    else:
      bc_cls = bc_mod.PeriodicBC
    bc = bc_cls(
      nu=self.nu,
      mesh=self.mesh,
      funval=field.get_bc_funval()
    )
    bc.build()
    return bc

  def get_ndof(self) -> int:
    return 2*self.mesh.nxy

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    start = time()
    res, jac = self.compute_res_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      res = x - self.x_old - self.dt*res
      jac = self.iden_uv - self.dt*jac
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["res_jac"] += delta
    return res, jac

  def compute_res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    # Extract u and v
    uv = self.extract_uv(x, diag=False)

    # Split force into u and v components
    f_uv = {"u": self.f[:self.mesh.nxy], "v": self.f[self.mesh.nxy:]}
    sigma_uv = {"u": self.sigma_u, "v": self.sigma_v}
    # Compute residual
    dx = []
    for k in ("u", "v"):
      dx_k = self.cx * (self.ops["Ax"] @ uv[k] - self.bc_f[k]["A"]["x"]) \
           + self.cy * (self.ops["Ay"] @ uv[k] - self.bc_f[k]["A"]["y"]) \
           + self.ops["D"] @ uv[k] + self.bc_f[k]["D"] \
           - sigma_uv[k] * uv[k] + f_uv[k]
      dx_k[self._bdry] = 0.0
      dx.append(dx_k)
    res = np.concatenate(dx)
    # Jacobian (constant: linear, decoupled u/v; pinned rows already zeroed)
    jac_uv = sp.csr_matrix((self.mesh.nxy, self.mesh.nxy))
    jac = sp.bmat(
      [[self._jac_xx_u, jac_uv],
       [jac_uv, self._jac_xx_v]],
      format="csr"
    )
    return res, jac

  def extract_uv(
    self,
    x: np.ndarray,
    diag: bool = True
  ) -> Union[Tuple[UV_TYPE, UV_TYPE], UV_TYPE]:
    uv = {"u": x[:self.mesh.nxy], "v": x[self.mesh.nxy:]}
    if diag:
      uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
      return uv, uv_diag
    else:
      return uv

  # Solving
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
    iostep: int = 1,
    verbose: bool = False
  ) -> SOL_TYPE:
    """
    Solves for the u and v states of the FOM using Newton's method.
    (Linear residual, so this converges in a single Newton step per
    time instant.)
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
      iostep=iostep,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, res, *_, flag = solver(x0, dt, nt)
    # Return solution
    uv = self.extract_uv(x, diag=False)
    converged = True if (flag[-1] == 0) else False
    return uv, res, converged


class FisherKPPAdvDiff2D(object):
  """
  Generate FOM for the 2D unsteady linear-advection + nonlinear
  (Fisher-KPP logistic) reaction-diffusion equation:

    du/dt + cx*du/dx + cy*du/dy = nu*Lap(u) + r_u*u*(1-u)
    dv/dt + cx*dv/dx + cy*dv/dy = nu*Lap(v) + r_v*v*(1-v)

  Identical to AdvDiff2D/ReactAdvDiff2D's advection+diffusion structure,
  but with a Fisher-KPP logistic reaction term in place of the linear
  reaction: r_u/r_v may differ (independent per-component growth rate),
  set directly via the constructor or read from field.get_reaction() (a
  single scalar shared by both components, or an (r_u, r_v) pair -- same
  convention as ReactAdvDiff2D).

  Unlike AdvDiff2D/ReactAdvDiff2D (both exactly linear, constant spatial
  Jacobian), the logistic term r*u*(1-u) is genuinely nonlinear in u: its
  Jacobian contribution r*(1-2u) depends on the current state and must be
  rebuilt every Newton iteration (same pattern as Burgers2D), not
  precomputed once at build() time -- so only the advection+diffusion part
  of the spatial Jacobian (_jac_lin) is cached; the reaction block is added
  fresh in compute_res_jac. Still fully decoupled in u/v (no cross term,
  unlike Burgers2D) -- the Jacobian stays block-diagonal, just with a
  state-dependent reaction block instead of a constant one.

  Dirichlet BC handling mirrors AdvDiff2D: plain ghost-point elimination by
  default; pass pin_boundary=True to additionally freeze the outermost
  node ring. (Datasets built on this FOM are expected to use Neumann or
  periodic BC, matching the *MaskedBump field convention, so pin_boundary
  is a no-op there.)
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    nu: float,
    r: float = 1.0,
    cx: float = 1.0,
    cy: float = 0.0,
    pin_boundary: bool = False
  ) -> None:
    # Mesh
    self.mesh = mesh
    self.mesh.is_built()
    # Viscosity / growth rate / advection
    self.nu = nu
    self.r_u = self.r_v = float(r)
    self.cx, self.cy = float(cx), float(cy)
    self.pin_boundary = bool(pin_boundary)
    # Integration
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Runtime
    self.runtime = {k: 0.0 for k in ("total", "lin_solve", "res_jac")}
    self.built = False
    # Force term
    self.f = None

  # Building
  # ===================================
  def is_built(self) -> None:
    if (not self.built):
      raise ValueError(
        "FOM model not built. Please, call 'build' method first."
      )

  def build(
    self,
    field: field_mod.FIELD_TYPES,
    force: np.ndarray = None
  ) -> None:
    # BC
    self.bc = self.build_bc(field)
    self.bc_f = self.bc.f
    # Advection / growth-rate coefficients (fixed per case, sampled by the
    # field if it provides them)
    if hasattr(field, "get_advection"):
      self.cx, self.cy = field.get_advection()
    if hasattr(field, "get_reaction"):
      r = field.get_reaction()
      if isinstance(r, (tuple, list, np.ndarray)):
        self.r_u, self.r_v = float(r[0]), float(r[1])
      else:
        self.r_u = self.r_v = float(r)
    # Operators
    self.diff_ops = DiffOperators(
      nu=self.nu,
      bc=self.bc,
      mesh=self.mesh,
      upwind=False,
      upwind_order=1
    )
    self.diff_ops.build()
    self.ops = self.diff_ops.ops
    self.ops_names = list(self.ops.keys())
    # Identities
    self.iden = sp.eye(self.mesh.nxy).tocsr()
    self.iden_uv = sp.eye(self.get_ndof()).tocsr()

    # Set forcing term: explicit argument takes precedence, else pulled
    # from the field (force-driven unsteady datasets), else zero.
    if force is not None:
      self.f = force
    elif hasattr(field, "get_force"):
      self.f = field.get_force()
    else:
      self.f = np.zeros(self.get_ndof())

    # Outermost node ring: opt-in only (pin_boundary=True). Off by default,
    # so Dirichlet BC is imposed the same way as Poisson2D/Burgers2D (plain
    # ghost elimination, every node a genuine unknown). See AdvDiff2D.
    if self.pin_boundary and self.bc.name == "dirichlet":
      x, y = self.mesh.nodes_val.T
      tol = 1e-10
      self._bdry = ((x <= x.min()+tol) | (x >= x.max()-tol) |
                    (y <= y.min()+tol) | (y >= y.max()-tol))
    else:
      self._bdry = np.zeros(self.mesh.nxy, dtype=bool)
    # Constant linear (advection+diffusion) part of the spatial Jacobian,
    # with pinned rows zeroed -- the reaction term is genuinely nonlinear
    # (r*(1-2u), state-dependent), so it is NOT included here; it's added
    # fresh every Newton iteration in compute_res_jac instead.
    jac_lin = self.cx*self.ops["Ax"] + self.cy*self.ops["Ay"] + self.ops["D"]
    if self._bdry.any():
      jac_lin = jac_lin.tolil()
      jac_lin[self._bdry, :] = 0.0
      jac_lin = jac_lin.tocsr()
    self._jac_lin = jac_lin
    self.built = True

  def build_bc(
    self,
    field: field_mod.FIELD_TYPES
  ) -> bc_mod.BC_TYPES:
    if (field.bc_type == "dirichlet"):
      bc_cls = bc_mod.DirichletBC
    elif (field.bc_type == "neumann"):
      bc_cls = bc_mod.NeumannBC
    else:
      bc_cls = bc_mod.PeriodicBC
    bc = bc_cls(
      nu=self.nu,
      mesh=self.mesh,
      funval=field.get_bc_funval()
    )
    bc.build()
    return bc

  def get_ndof(self) -> int:
    return 2*self.mesh.nxy

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    start = time()
    res, jac = self.compute_res_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      res = x - self.x_old - self.dt*res
      jac = self.iden_uv - self.dt*jac
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["res_jac"] += delta
    return res, jac

  def compute_res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    # Extract u and v
    uv = self.extract_uv(x, diag=False)

    # Split force into u and v components
    f_uv = {"u": self.f[:self.mesh.nxy], "v": self.f[self.mesh.nxy:]}
    r_uv = {"u": self.r_u, "v": self.r_v}
    # Compute residual and per-component nonlinear Jacobian block
    dx = []
    jac_blocks = {}
    for k in ("u", "v"):
      u_k = uv[k]
      dx_k = self.cx * (self.ops["Ax"] @ u_k - self.bc_f[k]["A"]["x"]) \
           + self.cy * (self.ops["Ay"] @ u_k - self.bc_f[k]["A"]["y"]) \
           + self.ops["D"] @ u_k + self.bc_f[k]["D"] \
           + r_uv[k] * u_k * (1.0 - u_k) + f_uv[k]
      dx_k[self._bdry] = 0.0
      dx.append(dx_k)
      # d/du [r*u*(1-u)] = r*(1-2u) -- state-dependent, rebuilt every call.
      jac_xx_k = self._jac_lin + r_uv[k] * sp.diags(1.0 - 2.0*u_k)
      if self._bdry.any():
        jac_xx_k = jac_xx_k.tolil()
        jac_xx_k[self._bdry, :] = 0.0
        jac_xx_k = jac_xx_k.tocsr()
      jac_blocks[k] = jac_xx_k
    res = np.concatenate(dx)
    # Jacobian (nonlinear reaction block, still decoupled u/v)
    jac_uv = sp.csr_matrix((self.mesh.nxy, self.mesh.nxy))
    jac = sp.bmat(
      [[jac_blocks["u"], jac_uv],
       [jac_uv, jac_blocks["v"]]],
      format="csr"
    )
    return res, jac

  def extract_uv(
    self,
    x: np.ndarray,
    diag: bool = True
  ) -> Union[Tuple[UV_TYPE, UV_TYPE], UV_TYPE]:
    uv = {"u": x[:self.mesh.nxy], "v": x[self.mesh.nxy:]}
    if diag:
      uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
      return uv, uv_diag
    else:
      return uv

  # Solving
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
    iostep: int = 1,
    verbose: bool = False
  ) -> SOL_TYPE:
    """
    Solves for the u and v states of the FOM using Newton's method.
    Genuinely nonlinear (Fisher-KPP reaction), so -- unlike AdvDiff2D/
    ReactAdvDiff2D -- this generally needs more than one Newton iteration
    per time instant (same as Burgers2D).
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
      iostep=iostep,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, res, *_, flag = solver(x0, dt, nt)
    # Return solution
    uv = self.extract_uv(x, diag=False)
    converged = True if (flag[-1] == 0) else False
    return uv, res, converged
