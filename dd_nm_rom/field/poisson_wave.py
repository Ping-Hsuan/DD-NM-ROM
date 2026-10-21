import numpy as np
from typing import List, Dict, Union, Tuple

from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod
from dd_nm_rom.elements.bound_cond.utils import SIDES as _SIDES
from .basic import BasicField


class PoissonWaveDirichlet(BasicField):
  """
  Global plane-wave forcing + global plane-wave Dirichlet BC.

  For each component z ∈ {u,v}:
    forcing:  f_z(x,y) = Af_z * sin(2π (kx_z x + ky_z y + th_f_z))
    boundary: g_z(x,y) = Ab_z * sin(2π (kbx_z x + kby_z y + th_b_z))

  If shared_uv=True:
    u and v share the same 8 parameters:
      mu = [Af, kx, ky, th_f,  Ab, kbx, kby, th_b]

  If shared_uv=False:
    u and v have independent parameters (16 total):
      mu = [mu_u(8), mu_v(8)]
  """

  def __init__(
    self,
    mesh: mesh_mod.MeshDD,
    amp_lim: List[float] = [0.9, 1.1],
    k_lim: List[float] = [-0.5, 0.5],
    theta_lim: List[float] = [0.0, 1.0],
    bc_type: str = "dirichlet",
    shared_uv: bool = True,
  ) -> None:
    super(PoissonWaveDirichlet, self).__init__(mesh)
    bc_mod.check_bc_type(bc_type)
    self.bc_type = bc_type

    self.amp_lim = list(amp_lim)
    self.k_lim = list(k_lim)
    self.theta_lim = list(theta_lim)

    self.shared_uv = bool(shared_uv)

    # Store params as dicts so extending is trivial
    self.params_u = self._default_params()
    self.params_v = self._default_params()

  # -------------------------
  # Helpers
  # -------------------------
  @staticmethod
  def _default_params() -> Dict[str, float]:
    return dict(Af=1.0, kx=0.0, ky=0.0, th_f=0.0,
                Ab=1.0, kbx=0.0, kby=0.0, th_b=0.0)

  def _mu_dim(self) -> int:
    return 8 if self.shared_uv else 16

  def _split_mu(self, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu).reshape(-1)
    if mu.size != self._mu_dim():
      raise ValueError(f"Expected mu length {self._mu_dim()}, got {mu.size}.")
    if self.shared_uv:
      return mu, mu
    return mu[:8], mu[8:]

  def _vec_to_params(self, mu8: np.ndarray) -> Dict[str, float]:
    Af, kx, ky, th_f, Ab, kbx, kby, th_b = mu8
    return dict(Af=float(Af), kx=float(kx), ky=float(ky), th_f=float(th_f),
                Ab=float(Ab), kbx=float(kbx), kby=float(kby), th_b=float(th_b))

  # -------------------------
  # Design space
  # -------------------------
  def _init_design_space(self) -> None:
    # base 8 params
    base = np.array([
      self.amp_lim,    # Af
      self.k_lim,      # kx
      self.k_lim,      # ky
      self.theta_lim,  # th_f
      self.amp_lim,    # Ab
      self.k_lim,      # kbx
      self.k_lim,      # kby
      self.theta_lim,  # th_b
    ]).T

    # If shared -> 8 dims. If independent -> 16 dims (u then v).
    self.design_space = base if self.shared_uv else np.hstack([base, base])

  def sample_design_space(self) -> np.ndarray:
    def sample8() -> np.ndarray:
      Af = np.random.uniform(*self.amp_lim)
      kx = np.random.uniform(*self.k_lim)
      ky = np.random.uniform(*self.k_lim)
      th_f = np.random.uniform(*self.theta_lim)
      Ab = np.random.uniform(*self.amp_lim)
      kbx = np.random.uniform(*self.k_lim)
      kby = np.random.uniform(*self.k_lim)
      th_b = np.random.uniform(*self.theta_lim)
      return np.array([Af, kx, ky, th_f, Ab, kbx, kby, th_b], dtype=float)

    mu_u = sample8()
    if self.shared_uv:
      return mu_u
    mu_v = sample8()
    return np.concatenate([mu_u, mu_v])

  def construct_design_mat(
    self,
    n_samples: int
  ) -> np.ndarray:
    """
    Construct a design matrix using Latin Hypercube Sampling (LHS).
    
    Samples parameters according to:
      k, kb ∈ U[-0.5, 0.5]² (2D wave vectors)
      θ, θb ∈ U[0, 1] (scalar phase shifts)
    
    :param n_samples: Number of samples to generate.
    :type n_samples: int
    :return: Design matrix with sampled points.
    :rtype: np.ndarray
    """
    # Generate Latin Hypercube samples using parent class method
    dmat = super(PoissonWaveDirichlet, self).construct_design_mat(n_samples)
    
    # No additional constraints needed - the design space already
    # defines the proper ranges for all parameters:
    # - Af, Ab: amplitude limits
    # - kx, ky, kbx, kby: k_lim (wave vector components)
    # - th_f, th_b: theta_lim (phase shifts)
    
    return dmat

  # -------------------------
  # Params
  # -------------------------
  def set_params(self, mu: np.ndarray) -> None:
    mu_u, mu_v = self._split_mu(mu)
    self.params_u = self._vec_to_params(mu_u)
    self.params_v = self._vec_to_params(mu_v)

  # -------------------------
  # Forcing generation
  # -------------------------
  @staticmethod
  def _plane_wave(x: np.ndarray, y: np.ndarray, A: float, kx: float, ky: float, th: float) -> np.ndarray:
    return A * np.sin(2.0 * np.pi * (kx * x + ky * y + th))

  def _forcing_uv(self) -> Tuple[np.ndarray, np.ndarray]:
    x, y = self.mesh.nodes_val.T
    pu, pv = self.params_u, self.params_v
    fu = self._plane_wave(x, y, pu["Af"], pu["kx"], pu["ky"], pu["th_f"])
    fv = self._plane_wave(x, y, pv["Af"], pv["kx"], pv["ky"], pv["th_f"])
    shape = (self.mesh.n["y"], self.mesh.n["x"])
    return fu.reshape(shape), fv.reshape(shape)

  def u(self) -> np.ndarray:
    fu, _ = self._forcing_uv()
    return fu

  def v(self) -> np.ndarray:
    _, fv = self._forcing_uv()
    return fv

  def get_force(self, mu: Union[np.ndarray, None] = None) -> np.ndarray:
    if mu is not None:
      self.set_params(mu)
    fu, fv = self._forcing_uv()
    return np.concatenate([fu.reshape(-1), fv.reshape(-1)])

  # -------------------------
  # Dirichlet boundary values (callables)
  # -------------------------
  def g_dir_u(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = self.params_u
    return self._plane_wave(x, y, p["Ab"], p["kbx"], p["kby"], p["th_b"])

  def g_dir_v(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = self.params_v
    return self._plane_wave(x, y, p["Ab"], p["kbx"], p["kby"], p["th_b"])

  def get_bc_funval(self) -> Dict[str, Dict[str, callable]]:
    funval = {}
    for side in bc_mod.SIDES["all"]:
      funval[side] = {"u": self.g_dir_u, "v": self.g_dir_v}
    return funval


class PoissonWaveMixed(PoissonWaveDirichlet):
  """
  Poisson with mixed Dirichlet + homogeneous Neumann BCs.

  Dirichlet sides: u = Ab * sin(2π(kbx·x + kby·y + θ_b))  (parameterized wave)
  Neumann  sides:  ∂u/∂n = 0                               (zero flux)
  Forcing:         f = Af * sin(2π(kx·x + ky·y + θ_f))    (same as parent)

  bc_sides (optional):
    If provided, the BC layout is fixed for all samples.
    If None (default), the BC layout is sampled per case: 4 binary bits
    [left, right, bottom, top] (0=Dirichlet, 1=Neumann) are appended to mu.
    At least one Dirichlet side is always enforced (all-Neumann is ill-posed).

  mu layout when bc_sides=None:
    [continuous wave params (8 or 16), left_bit, right_bit, bottom_bit, top_bit]
  """

  _SIDE_ORDER = ("left", "right", "bottom", "top")

  def __init__(
    self,
    mesh: mesh_mod.MeshDD,
    bc_sides: Dict[str, str] = None,
    amp_lim: List[float] = [0.9, 1.1],
    k_lim:   List[float] = [-0.5, 0.5],
    theta_lim: List[float] = [0.0, 1.0],
    shared_uv: bool = True,
  ) -> None:
    super().__init__(mesh, amp_lim=amp_lim, k_lim=k_lim,
                     theta_lim=theta_lim, bc_type="dirichlet",
                     shared_uv=shared_uv)
    self.bc_type  = "mixed"
    self._param_bc = (bc_sides is None)
    self.bc_sides  = ({s: bc_sides[s] for s in _SIDES["all"]}
                      if bc_sides is not None
                      else {s: "dirichlet" for s in _SIDES["all"]})

  def _n_cont(self) -> int:
    return super()._mu_dim()

  def get_bc_spec(self) -> Dict[str, str]:
    return self.bc_sides

  def get_bc_funval(self) -> Dict[str, Dict[str, callable]]:
    parent_funval = super().get_bc_funval()
    _zero = lambda x, y: np.zeros_like(x)
    return {
      side: (parent_funval[side]
             if self.bc_sides[side] == "dirichlet"
             else {"u": _zero, "v": _zero})
      for side in _SIDES["all"]
    }

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    cont_mat = super().construct_design_mat(n_samples)   # (n, 8 or 16)
    if not self._param_bc:
      return cont_mat
    # Append 4 bc_sides bits: order = left, right, bottom, top; 0=D, 1=N
    bits = np.random.randint(0, 2, (n_samples, 4)).astype(float)
    # Prevent all-Neumann (ill-posed for Poisson): force left to Dirichlet
    bits[bits.sum(axis=1) == 4, 0] = 0.0
    return np.hstack([cont_mat, bits])

  def sample_design_space(self) -> np.ndarray:
    return self.construct_design_mat(1).reshape(-1)

  def set_params(self, mu: np.ndarray) -> None:
    mu = np.asarray(mu).reshape(-1)
    n_cont = self._n_cont()
    super().set_params(mu[:n_cont])
    if self._param_bc:
      bits = mu[n_cont:n_cont + 4].round().astype(int)
      self.bc_sides = {s: ("neumann" if b else "dirichlet")
                       for s, b in zip(self._SIDE_ORDER, bits)}
      # Safety: ensure at least one Dirichlet
      if all(v == "neumann" for v in self.bc_sides.values()):
        self.bc_sides[self._SIDE_ORDER[0]] = "dirichlet"
