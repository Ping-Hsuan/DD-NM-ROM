import numpy as np

from dd_nm_rom import ops
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod

from .basic import BasicField

_SIDES = bc_mod.SIDES


def _zero_bc(x, y):
  return np.zeros_like(x)


class GaussianField(object):
  """
  Gaussian source field with zero Dirichlet BC.

  One Gaussian per subdomain, independent u/v amplitudes.
  Forcing is explicitly zeroed at physical boundary nodes so the
  DirichletBC assembly is not corrupted regardless of Gaussian width.

  Parameters (mu): [A_u_0..n, A_v_0..n, x0_0..n, y0_0..n, sigma]
    - A_u, A_v : per-subdomain amplitudes (in amp_lim)
    - (x0_i, y0_i): Gaussian centre sampled within subdomain i's bounding box
    - sigma      : shared width (in sigma_lim)
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma_lim=(0.05, 0.15)
  ) -> None:
    self.mesh = mesh
    self.bc_type = "dirichlet"
    self.amp_lim = list(amp_lim)
    self.sigma_lim = list(sigma_lim)
    self.mu = None
    x, y = mesh.nodes_val.T
    tol = 1e-10
    self._bdry = ((x <= x.min() + tol) | (x >= x.max() - tol) |
                  (y <= y.min() + tol) | (y >= y.max() - tol))
    self._cx, self._cy = self._sub_bounds()

  def _sub_bounds(self):
    x, y = self.mesh.nodes_val.T
    cx, cy = [], []
    for ind in self.mesh.res_nodes:
      cx.append((x[ind].min(), x[ind].max()))
      cy.append((y[ind].min(), y[ind].max()))
    return cx, cy

  def _n_params(self):
    n = self.mesh.n_sub
    return 4 * n + 1

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    from pyDOE import lhs
    n = self.mesh.n_sub
    raw = lhs(self._n_params(), n_samples)
    mat = np.empty_like(raw)
    lo_a, hi_a = self.amp_lim
    lo_s, hi_s = self.sigma_lim
    mat[:, :n]    = raw[:, :n]    * (hi_a - lo_a) + lo_a
    mat[:, n:2*n] = raw[:, n:2*n] * (hi_a - lo_a) + lo_a
    for i, ((xl, xh), (yl, yh)) in enumerate(zip(self._cx, self._cy)):
      mat[:, 2*n+i] = raw[:, 2*n+i] * (xh - xl) + xl
      mat[:, 3*n+i] = raw[:, 3*n+i] * (yh - yl) + yl
    mat[:, -1] = raw[:, -1] * (hi_s - lo_s) + lo_s
    return mat

  def sample_design_space(self) -> np.ndarray:
    return self.construct_design_mat(1).reshape(-1)

  def construct_design_mat_test(self, n_samples, mu_train, tol=0.1):
    ref = [np.asarray(r) for r in mu_train]
    out = []
    while len(out) < n_samples:
      s = self.sample_design_space()
      all_ref = ref + out
      if not all_ref or min(np.linalg.norm(r - s) for r in all_ref) >= tol:
        out.append(s)
        ref.append(s)
    return np.array(out)

  def set_params(self, mu: np.ndarray) -> None:
    self.mu = np.asarray(mu).reshape(-1)

  def _make_comp(self, amps):
    n = self.mesh.n_sub
    sigma = self.mu[-1]
    x0s, y0s = self.mu[2*n:3*n], self.mu[3*n:4*n]
    x, y = self.mesh.nodes_val.T
    f = np.zeros(self.mesh.nxy)
    for a, x0, y0 in zip(amps, x0s, y0s):
      f += a * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    f[self._bdry] = 0.0
    return f

  def get_force(self) -> np.ndarray:
    n = self.mesh.n_sub
    return np.concatenate([self._make_comp(self.mu[:n]), self._make_comp(self.mu[n:2*n])])

  def get_bc_funval(self):
    return {s: {"u": _zero_bc, "v": _zero_bc} for s in _SIDES["all"]}


class AdvDiffUnsteadyField(GaussianField):
  """
  Gaussian forcing + Gaussian IC for unsteady AdvDiff with parameterized (cx, cy).

  Shared Gaussian centres and sigma for both forcing and IC;
  independent amplitudes so the IC and forcing can differ in sign/magnitude.

  mu = [A_u_f_0..n, A_v_f_0..n,    forcing amplitudes
        B_u_ic_0..n, B_v_ic_0..n,  IC amplitudes
        x0_0..n, y0_0..n,          shared Gaussian centres
        sigma,                      shared Gaussian width
        cx, cy]                     advection coefficients
  Total length: 6*n_sub + 3
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma_lim=(0.05, 0.15),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0)
  ) -> None:
    super().__init__(mesh, amp_lim, sigma_lim)
    self.cx_lim = list(cx_lim)
    self.cy_lim = list(cy_lim)

  def _n_params(self):
    n = self.mesh.n_sub
    return 6 * n + 3  # 4n (force+IC amps) + 2n (centers) + 1 sigma + 2 adv

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    from pyDOE import lhs
    n = self.mesh.n_sub
    raw = lhs(self._n_params(), n_samples)
    mat = np.empty_like(raw)
    lo_a, hi_a = self.amp_lim
    lo_s, hi_s = self.sigma_lim
    lo_cx, hi_cx = self.cx_lim
    lo_cy, hi_cy = self.cy_lim
    # forcing amplitudes: [0..2n)
    mat[:, :n]    = raw[:, :n]    * (hi_a - lo_a) + lo_a
    mat[:, n:2*n] = raw[:, n:2*n] * (hi_a - lo_a) + lo_a
    # IC amplitudes: [2n..4n)
    mat[:, 2*n:3*n] = raw[:, 2*n:3*n] * (hi_a - lo_a) + lo_a
    mat[:, 3*n:4*n] = raw[:, 3*n:4*n] * (hi_a - lo_a) + lo_a
    # shared centres: [4n..6n)
    for i, ((xl, xh), (yl, yh)) in enumerate(zip(self._cx, self._cy)):
      mat[:, 4*n+i] = raw[:, 4*n+i] * (xh - xl) + xl
      mat[:, 5*n+i] = raw[:, 5*n+i] * (yh - yl) + yl
    # sigma, cx, cy
    mat[:, 6*n]   = raw[:, 6*n]   * (hi_s - lo_s) + lo_s
    mat[:, 6*n+1] = raw[:, 6*n+1] * (hi_cx - lo_cx) + lo_cx
    mat[:, 6*n+2] = raw[:, 6*n+2] * (hi_cy - lo_cy) + lo_cy
    return mat

  # override _make_comp: evaluate each Gaussian globally so contributions
  # decay smoothly across subdomain interfaces (no jump discontinuity).
  def _make_comp(self, amps):
    n = self.mesh.n_sub
    sigma = self.mu[6*n]
    x0s   = self.mu[4*n:5*n]
    y0s   = self.mu[5*n:6*n]
    x, y  = self.mesh.nodes_val.T
    f = np.zeros(self.mesh.nxy)
    for a, x0, y0 in zip(amps, x0s, y0s):
      f += a * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    f[self._bdry] = 0.0
    return f

  def get_force(self) -> np.ndarray:
    n = self.mesh.n_sub
    return np.concatenate([self._make_comp(self.mu[:n]),
                            self._make_comp(self.mu[n:2*n])])

  def get_ic(self):
    """Return (u0, v0) -- initial condition arrays of shape (nxy,)."""
    n = self.mesh.n_sub
    return (self._make_comp(self.mu[2*n:3*n]),
            self._make_comp(self.mu[3*n:4*n]))

  def get_advection(self):
    n = self.mesh.n_sub
    return float(self.mu[6*n+1]), float(self.mu[6*n+2])


class PeriodicAdvDiffUnsteadyField(object):
  """Fourier forcing + Fourier IC for unsteady periodic AdvDiff.

  Both forcing and IC use the same 6-mode Fourier basis (all zero-mean):
    sin(2*pi*x), cos(2*pi*x), sin(2*pi*y), cos(2*pi*y),
    sin(2*pi*x)sin(2*pi*y), sin(4*pi*x)sin(2*pi*y)
  Independent amplitudes for forcing vs IC. cx, cy also parameterized.

  mu = [a_u_f_0..5, a_v_f_0..5,    forcing amps  (12)
        a_u_ic_0..5, a_v_ic_0..5,   IC amps       (12)
        cx, cy]                      advection      (2)
  Total: 26 params
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0)
  ) -> None:
    self.mesh = mesh
    self.bc_type = "periodic"
    self.amp_lim = list(amp_lim)
    self.cx_lim  = list(cx_lim)
    self.cy_lim  = list(cy_lim)
    self.mu = None
    x, y = mesh.nodes_val.T
    # Wavenumber scaled by the actual physical domain length so exactly one
    # period spans the whole domain, regardless of n_sub (Lx=Ly=1 at the
    # original 2x2 training domain, so kx=ky=2*pi there -- this is a no-op
    # at 2x2 and only changes behavior at other domain sizes, e.g. a 10x10
    # spatial-extrapolation test). Without this, the basis's period stays
    # fixed at 1.0 physical unit regardless of domain size, so at a larger
    # domain it just repeats the same small pattern over and over instead
    # of spanning the domain.
    Lx = mesh.phylim["x"][1] - mesh.phylim["x"][0]
    Ly = mesh.phylim["y"][1] - mesh.phylim["y"][0]
    kx, ky = 2*np.pi/Lx, 2*np.pi/Ly
    self._basis = np.stack([
      np.sin(kx*x), np.cos(kx*x),
      np.sin(ky*y), np.cos(ky*y),
      np.sin(kx*x)*np.sin(ky*y),
      np.sin(2*kx*x)*np.sin(ky*y),
    ])  # (6, nxy)

  def _n_modes(self):  return self._basis.shape[0]
  def _n_params(self): return 4 * self._n_modes() + 2  # 26

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    from pyDOE import lhs
    nm = self._n_modes()
    raw = lhs(self._n_params(), n_samples)
    mat = np.empty_like(raw)
    lo_a, hi_a = self.amp_lim
    lo_cx, hi_cx = self.cx_lim
    lo_cy, hi_cy = self.cy_lim
    mat[:, :4*nm] = raw[:, :4*nm] * (hi_a - lo_a) + lo_a
    mat[:, -2] = raw[:, -2] * (hi_cx - lo_cx) + lo_cx
    mat[:, -1] = raw[:, -1] * (hi_cy - lo_cy) + lo_cy
    return mat

  def sample_design_space(self) -> np.ndarray:
    return self.construct_design_mat(1).reshape(-1)

  def construct_design_mat_test(self, n_samples, mu_train, tol=0.1):
    ref = [np.asarray(r) for r in mu_train]
    out = []
    while len(out) < n_samples:
      s = self.sample_design_space()
      all_ref = ref + out
      if not all_ref or min(np.linalg.norm(r - s) for r in all_ref) >= tol:
        out.append(s); ref.append(s)
    return np.array(out)

  def set_params(self, mu: np.ndarray) -> None:
    self.mu = np.asarray(mu).reshape(-1)

  def _field_from_amps(self, amps):
    return amps @ self._basis  # (nxy,)

  def get_force(self) -> np.ndarray:
    nm = self._n_modes()
    fu = self._field_from_amps(self.mu[:nm])
    fv = self._field_from_amps(self.mu[nm:2*nm])
    return np.concatenate([fu, fv])

  def get_ic(self):
    nm = self._n_modes()
    u0 = self._field_from_amps(self.mu[2*nm:3*nm])
    v0 = self._field_from_amps(self.mu[3*nm:4*nm])
    return u0, v0

  def get_advection(self):
    return float(self.mu[-2]), float(self.mu[-1])

  def get_bc_funval(self):
    return {s: {"u": _zero_bc, "v": _zero_bc} for s in _SIDES["all"]}


class ReactAdvDiffUnsteadyField(AdvDiffUnsteadyField):
  """
  AdvDiffUnsteadyField extended with a parameterized reaction coefficient.

  Identical layout to AdvDiffUnsteadyField with one extra trailing parameter:
    mu = [A_u_f x n, A_v_f x n, B_u_ic x n, B_v_ic x n, x0 x n, y0 x n,
          sigma_gauss, cx, cy, sigma_react]
  Total length: 6*n_sub + 4
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma_lim=(0.05, 0.15),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    sigma_react_lim=(0.1, 2.0)
  ) -> None:
    super().__init__(mesh, amp_lim, sigma_lim, cx_lim, cy_lim)
    self.sigma_react_lim = list(sigma_react_lim)

  def _n_params(self):
    n = self.mesh.n_sub
    return 6 * n + 4  # base (6n+3) + sigma_react

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    from pyDOE import lhs
    n = self.mesh.n_sub
    raw = lhs(self._n_params(), n_samples)
    mat = np.empty_like(raw)
    lo_a, hi_a = self.amp_lim
    lo_s, hi_s = self.sigma_lim
    lo_cx, hi_cx = self.cx_lim
    lo_cy, hi_cy = self.cy_lim
    lo_sr, hi_sr = self.sigma_react_lim
    mat[:, :n]      = raw[:, :n]      * (hi_a - lo_a) + lo_a
    mat[:, n:2*n]   = raw[:, n:2*n]   * (hi_a - lo_a) + lo_a
    mat[:, 2*n:3*n] = raw[:, 2*n:3*n] * (hi_a - lo_a) + lo_a
    mat[:, 3*n:4*n] = raw[:, 3*n:4*n] * (hi_a - lo_a) + lo_a
    for i, ((xl, xh), (yl, yh)) in enumerate(zip(self._cx, self._cy)):
      mat[:, 4*n+i] = raw[:, 4*n+i] * (xh - xl) + xl
      mat[:, 5*n+i] = raw[:, 5*n+i] * (yh - yl) + yl
    mat[:, 6*n]   = raw[:, 6*n]   * (hi_s  - lo_s)  + lo_s
    mat[:, 6*n+1] = raw[:, 6*n+1] * (hi_cx - lo_cx) + lo_cx
    mat[:, 6*n+2] = raw[:, 6*n+2] * (hi_cy - lo_cy) + lo_cy
    mat[:, 6*n+3] = raw[:, 6*n+3] * (hi_sr - lo_sr) + lo_sr
    return mat

  def get_reaction(self):
    n = self.mesh.n_sub
    return float(self.mu[6*n+3])


class AdvDiffUnsteadyFieldNeumann(AdvDiffUnsteadyField):
  """
  Same as AdvDiffUnsteadyField, but homogeneous NEUMANN BC (zero flux)
  instead of Dirichlet.

  Neumann only constrains the *solution's* gradient at the boundary, not
  the forcing/IC value there -- unlike Dirichlet, there's no value the
  forcing/IC needs to be consistent with at the edge. So the artificial
  zero-clip AdvDiffUnsteadyField applies at the boundary-adjacent nodes
  (needed there to avoid a large, arbitrary mismatch against a *fixed*
  Dirichlet value) isn't needed here: the Gaussian bumps keep their
  natural, unclipped value everywhere, including near the edge.
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma_lim=(0.05, 0.15),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0)
  ) -> None:
    super().__init__(mesh, amp_lim, sigma_lim, cx_lim, cy_lim)
    self.bc_type = "neumann"

  def _make_comp(self, amps):
    n = self.mesh.n_sub
    sigma = self.mu[6*n]
    x0s   = self.mu[4*n:5*n]
    y0s   = self.mu[5*n:6*n]
    x, y  = self.mesh.nodes_val.T
    f = np.zeros(self.mesh.nxy)
    for a, x0, y0 in zip(amps, x0s, y0s):
      f += a * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return f


class ReactAdvDiffUnsteadyFieldNeumann(ReactAdvDiffUnsteadyField):
  """
  Same as ReactAdvDiffUnsteadyField, but homogeneous Neumann BC instead of
  Dirichlet -- see AdvDiffUnsteadyFieldNeumann for rationale.
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma_lim=(0.05, 0.15),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    sigma_react_lim=(0.1, 2.0)
  ) -> None:
    super().__init__(mesh, amp_lim, sigma_lim, cx_lim, cy_lim, sigma_react_lim)
    self.bc_type = "neumann"

  def _make_comp(self, amps):
    n = self.mesh.n_sub
    sigma = self.mu[6*n]
    x0s   = self.mu[4*n:5*n]
    y0s   = self.mu[5*n:6*n]
    x, y  = self.mesh.nodes_val.T
    f = np.zeros(self.mesh.nxy)
    for a, x0, y0 in zip(amps, x0s, y0s):
      f += a * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return f


class AdvDiffUnsteadyFieldPeriodicBump(AdvDiffUnsteadyField):
  """
  Same as AdvDiffUnsteadyField (one independently-sampled Gaussian bump per
  subdomain), but with periodic BC instead of Dirichlet.

  Unlike PeriodicAdvDiffUnsteadyField (a global few-mode Fourier IC/forcing,
  whose wavelength is fixed relative to the domain -- extrapolating it to a
  larger domain just stretches/dilutes the same smooth pattern, so more
  subdomains does not add any genuinely new local complexity), this keeps
  the per-subdomain bump construction: each subdomain gets its own
  independently-sampled amplitude/center, so scaling up n_sub (e.g. for a
  10x10 spatial-extrapolation test) genuinely increases the number of
  independent local features and inter-subdomain interactions, mirroring
  how the Burgers/AdvDiff Neumann bump fields scale. Only the BC differs
  from AdvDiffUnsteadyFieldNeumann (periodic wrap-around instead of
  zero-flux); bump placement/amplitude sampling is otherwise identical, and
  needs no boundary clip for the same reason Neumann doesn't: periodic BC
  only constrains matching between opposite edges, not a prescribed
  boundary value the IC/forcing would need to be consistent with.
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma_lim=(0.05, 0.15),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0)
  ) -> None:
    super().__init__(mesh, amp_lim, sigma_lim, cx_lim, cy_lim)
    self.bc_type = "periodic"

  def _make_comp(self, amps):
    n = self.mesh.n_sub
    sigma = self.mu[6*n]
    x0s   = self.mu[4*n:5*n]
    y0s   = self.mu[5*n:6*n]
    x, y  = self.mesh.nodes_val.T
    f = np.zeros(self.mesh.nxy)
    for a, x0, y0 in zip(amps, x0s, y0s):
      f += a * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return f


class ReactAdvDiffUnsteadyFieldPeriodicBump(ReactAdvDiffUnsteadyField):
  """
  Same as ReactAdvDiffUnsteadyField, but periodic BC instead of Dirichlet --
  see AdvDiffUnsteadyFieldPeriodicBump for rationale (per-subdomain
  independent bumps, unlike PeriodicReactAdvDiffUnsteadyField's global
  fixed-wavelength Fourier IC).
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma_lim=(0.05, 0.15),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    sigma_react_lim=(0.1, 2.0)
  ) -> None:
    super().__init__(mesh, amp_lim, sigma_lim, cx_lim, cy_lim, sigma_react_lim)
    self.bc_type = "periodic"

  def _make_comp(self, amps):
    n = self.mesh.n_sub
    sigma = self.mu[6*n]
    x0s   = self.mu[4*n:5*n]
    y0s   = self.mu[5*n:6*n]
    x, y  = self.mesh.nodes_val.T
    f = np.zeros(self.mesh.nxy)
    for a, x0, y0 in zip(amps, x0s, y0s):
      f += a * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return f


class PeriodicReactAdvDiffUnsteadyField(PeriodicAdvDiffUnsteadyField):
  """
  PeriodicAdvDiffUnsteadyField extended with a parameterized reaction
  coefficient.

  mu = [a_u_f x6, a_v_f x6,   forcing amps  (12)
        a_u_ic x6, a_v_ic x6, IC amps        (12)
        cx, cy,                advection      (2)
        sigma_react]           reaction        (1)
  Total: 27 params
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    sigma_react_lim=(0.1, 2.0)
  ) -> None:
    super().__init__(mesh, amp_lim, cx_lim, cy_lim)
    self.sigma_react_lim = list(sigma_react_lim)

  def _n_params(self):
    return super()._n_params() + 1   # base 26 + sigma_react

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    from pyDOE import lhs
    nm = self._n_modes()
    raw = lhs(self._n_params(), n_samples)
    mat = np.empty_like(raw)
    lo_a, hi_a = self.amp_lim
    lo_cx, hi_cx = self.cx_lim
    lo_cy, hi_cy = self.cy_lim
    lo_sr, hi_sr = self.sigma_react_lim
    mat[:, :4*nm] = raw[:, :4*nm] * (hi_a  - lo_a)  + lo_a
    mat[:, -3]    = raw[:, -3]    * (hi_cx - lo_cx) + lo_cx
    mat[:, -2]    = raw[:, -2]    * (hi_cy - lo_cy) + lo_cy
    mat[:, -1]    = raw[:, -1]    * (hi_sr - lo_sr) + lo_sr
    return mat

  def get_advection(self):
    return float(self.mu[-3]), float(self.mu[-2])

  def get_reaction(self):
    return float(self.mu[-1])


class AdvDiffUnsteadyFieldPeriodicWave(object):
  """
  Periodic AdvDiff field with genuinely per-subdomain-independent forcing
  AND IC, both built from the same subdomain-calibrated sin(kx*x)*sin(ky*y)
  wave shape (kx=pi/lx_sub, so kx*lx_sub=pi exactly: one half-period per
  subdomain, exact zero at every subdomain boundary regardless of n_sub --
  same scale-invariance as SinMultiPeak/Burgers' periodic field).

  An earlier version of this field kept forcing restricted to a single
  shared-amplitude global 6-mode Fourier basis, based on ad-hoc isolation
  tests that appeared to show per-subdomain-independent forcing always
  diverges for AdvDiff2D+periodic. Those tests relied on
  AdvDiff2D.solve()'s default steady=True (which silently discards dt/nt
  and solves the steady-state system instead of time-marching, where a
  near-singular Jacobian for some (cx,cy) is unsurprising). Once corrected
  to steady=False (matching what generate_advdiff.py actually uses),
  genuinely per-subdomain-independent forcing converges reliably across
  the full production cx,cy range -- so forcing can safely carry the same
  per-subdomain richness as the IC.

  forcing_amp_lim is still kept separate from (and much smaller than)
  amp_lim: a persistent forcing's long-time response can still be
  amplified by a large, (cx,cy,nu)-dependent gain relative to its own
  bounded magnitude (u_ss = -f/lambda_k for each excited mode), and
  per-subdomain forcing has broadband spatial content (a slope
  discontinuity decays only ~1/k^2 in Fourier space, unlike a narrow
  6-mode basis), so this gain risk is if anything larger here. Keeping
  forcing_amp_lim small keeps the solution's dynamic range reasonable
  without needing to solve for the exact gain per case.

  mu = [a_u_f_0..n-1, a_v_f_0..n-1,   forcing amplitudes (2*n_sub, per-subdomain wave)
        B_u_ic_0..n-1, B_v_ic_0..n-1, IC amplitudes       (2*n_sub, per-subdomain wave)
        cx, cy]                       advection             (2)
  Total length: 4*n_sub + 2
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    forcing_amp_lim=(-0.1, 0.1),
    cx_random_sign=False,
    cy_random_sign=False
  ) -> None:
    self.mesh = mesh
    self.bc_type = "periodic"
    self.amp_lim = list(amp_lim)
    self.cx_lim  = list(cx_lim)
    self.cy_lim  = list(cy_lim)
    self.forcing_amp_lim = list(forcing_amp_lim)
    self.cx_random_sign = bool(cx_random_sign)
    self.cy_random_sign = bool(cy_random_sign)
    self.mu = None
    # Subdomain-calibrated wavenumber, independent per-subdomain amplitude,
    # shared shape for both forcing and IC.
    self._kx_ic = np.pi / mesh.l_sub["x"]
    self._ky_ic = np.pi / mesh.l_sub["y"]

  def _n_params(self):
    n = self.mesh.n_sub
    return 4*n + 2

  def _apply_cxcy(self, mat, raw, cx_col=-2, cy_col=-1):
    """Map the cx, cy design-mat columns (default: last two). Normally a
    plain linear map onto [lo,hi]. If cx_random_sign/cy_random_sign is set,
    cx_lim/cy_lim instead give a MAGNITUDE range (e.g. [1.5,3.0]) and each
    sample independently gets a uniformly random sign -- so cx/cy lands in
    [-hi,-lo] union [lo,hi], never in the excluded near-zero gap."""
    n_samples = mat.shape[0]
    lo_cx, hi_cx = self.cx_lim
    lo_cy, hi_cy = self.cy_lim
    cx = raw[:, cx_col] * (hi_cx - lo_cx) + lo_cx
    cy = raw[:, cy_col] * (hi_cy - lo_cy) + lo_cy
    if self.cx_random_sign:
      cx *= np.random.choice([-1.0, 1.0], size=n_samples)
    if self.cy_random_sign:
      cy *= np.random.choice([-1.0, 1.0], size=n_samples)
    mat[:, cx_col] = cx
    mat[:, cy_col] = cy
    return mat

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    from pyDOE import lhs
    n = self.mesh.n_sub
    raw = lhs(self._n_params(), n_samples)
    mat = np.empty_like(raw)
    lo_f, hi_f = self.forcing_amp_lim
    lo_a, hi_a = self.amp_lim
    mat[:, :2*n]    = raw[:, :2*n]    * (hi_f - lo_f) + lo_f  # forcing amps
    mat[:, 2*n:4*n] = raw[:, 2*n:4*n] * (hi_a - lo_a) + lo_a  # IC amps
    mat = self._apply_cxcy(mat, raw)
    return self._apply_ic_mask(mat)

  def _apply_ic_mask(self, mat):
    """Per-subdomain binary on/off mask on the IC amplitude block only (like
    SinMultiPeak): each subdomain's IC component is either active (its
    sampled amplitude) or exactly zero -- so not every subdomain starts
    with a hump. All-zero draws are retried. Forcing block untouched (every
    subdomain always contributes some forcing, however small)."""
    n = self.mesh.n_sub
    ic0 = 2*n
    for s in range(mat.shape[0]):
      while True:
        mask = np.random.binomial(1, 0.5, size=n)
        if mask.sum() > 0:
          break
      mat[s, ic0:ic0+n]     *= mask
      mat[s, ic0+n:ic0+2*n] *= mask
    return mat

  def sample_design_space(self) -> np.ndarray:
    return self.construct_design_mat(1).reshape(-1)

  def construct_design_mat_test(self, n_samples, mu_train, tol=0.1):
    ref = [np.asarray(r) for r in mu_train]
    out = []
    while len(out) < n_samples:
      s = self.sample_design_space()
      all_ref = ref + out
      if not all_ref or min(np.linalg.norm(r - s) for r in all_ref) >= tol:
        out.append(s); ref.append(s)
    return np.array(out)

  def set_params(self, mu: np.ndarray) -> None:
    self.mu = np.asarray(mu).reshape(-1)

  def _make_comp_field(self, amps):
    n = self.mesh.n_sub
    x, y = self.mesh.nodes_val.T
    f = np.zeros(self.mesh.nxy)
    for i in range(n):
      ind = self.mesh.res_nodes[i]
      f[ind] = amps[i] * np.sin(self._kx_ic*x[ind]) * np.sin(self._ky_ic*y[ind])
    return f

  def get_force(self) -> np.ndarray:
    n = self.mesh.n_sub
    fu = self._make_comp_field(self.mu[:n])
    fv = self._make_comp_field(self.mu[n:2*n])
    return np.concatenate([fu, fv])

  def get_ic(self):
    n = self.mesh.n_sub
    u0 = self._make_comp_field(self.mu[2*n : 3*n])
    v0 = self._make_comp_field(self.mu[3*n : 4*n])
    return u0, v0

  def get_advection(self):
    return float(self.mu[-2]), float(self.mu[-1])

  def get_bc_funval(self):
    return {s: {"u": _zero_bc, "v": _zero_bc} for s in _SIDES["all"]}


class ReactAdvDiffUnsteadyFieldPeriodicWave(AdvDiffUnsteadyFieldPeriodicWave):
  """
  AdvDiffUnsteadyFieldPeriodicWave extended with a parameterized reaction
  coefficient -- see AdvDiffUnsteadyFieldPeriodicWave for rationale.

  mu = [a_u_f_0..n-1, a_v_f_0..n-1, B_u_ic_0..n-1, B_v_ic_0..n-1,
        cx, cy, sigma_react]
  Total length: 4*n_sub + 3
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    forcing_amp_lim=(-0.1, 0.1),
    cx_random_sign=False,
    cy_random_sign=False,
    sigma_react_lim=(0.1, 2.0)
  ) -> None:
    super().__init__(mesh, amp_lim, cx_lim, cy_lim, forcing_amp_lim,
                      cx_random_sign, cy_random_sign)
    self.sigma_react_lim = list(sigma_react_lim)

  def _n_params(self):
    return super()._n_params() + 1

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    from pyDOE import lhs
    n = self.mesh.n_sub
    raw = lhs(self._n_params(), n_samples)
    mat = np.empty_like(raw)
    lo_f, hi_f = self.forcing_amp_lim
    lo_a, hi_a = self.amp_lim
    lo_sr, hi_sr = self.sigma_react_lim
    mat[:, :2*n]    = raw[:, :2*n]    * (hi_f  - lo_f)  + lo_f
    mat[:, 2*n:4*n] = raw[:, 2*n:4*n] * (hi_a  - lo_a)  + lo_a
    mat = self._apply_cxcy(mat, raw, cx_col=-3, cy_col=-2)
    mat[:, -1]      = raw[:, -1]      * (hi_sr - lo_sr) + lo_sr
    return self._apply_ic_mask(mat)

  def get_advection(self):
    return float(self.mu[-3]), float(self.mu[-2])

  def get_reaction(self):
    return float(self.mu[-1])


class AdvDiffUnsteadyFieldMaskedBump(BasicField):
  """
  Minimal-parameter AdvDiff field: fixed-shape Gaussian bumps sitting at
  each subdomain's geometric center, switched on/off per subdomain by a
  Burgers-style config index (see SinMultiPeak) -- one independent mask +
  amplitude draw for u, another independent one for v -- plus a
  globally-sampled advection velocity (cx, cy). No forcing (get_force
  returns an explicit all-zero array -- BasicField already defines a
  get_force stub, so it must be overridden with a real array rather than
  left undefined, or AdvDiff2D.build would call the inherited stub and get
  None).

  Unlike AdvDiffUnsteadyField/Neumann (independent per-subdomain amplitude
  AND center for both forcing and IC -- 6*n_sub+3 params), here only the
  center and width are fixed constants -- amplitude is sampled per
  subdomain (in amp_lim) and masked on/off by a config index, *exactly*
  like SinMultiPeak's Burgers field. That amplitude+mask combination
  collapses to one parameter type per component under the same convention
  that gives SinMultiPeak a count of 1 (a quantity instantiated
  independently per subdomain, but conceptually a single richness axis,
  not one that should scale with how many subdomains happen to tile the
  domain) -- but u and v are two distinct physical fields, not two
  instances of the same mesh-tiling artifact, so giving them independent
  amplitudes is a genuinely new richness axis (exactly how AdvDiffUnsteady
  Field's original design counted A_u/A_v as two separate parameters, not
  one) -- hence collapsed count 4, not 3. cx, cy additionally drive
  qualitatively different dynamics here (this PDE is linear, so amplitude
  alone only rescales the solution; direction/speed of transport is what
  changes behavior beyond that).

  mu = [amp_u_0..n-1, amp_v_0..n-1, cx, cy]
    - amp_u_i / amp_v_i: sampled in amp_lim if subdomain i is "on" (from an
      independently-sampled config index per component), else 0 -- same
      mask+amplitude mechanism as SinMultiPeak, applied separately to u
      and v (independent masks, so u and v need not even be active in the
      same subdomains)
    - cx, cy: global advection velocity
  Stored length: 2*n_sub + 2. Collapsed/effective parameter count: 4
  (amplitude_u, amplitude_v, cx, cy).

  bc_type selects the BC family the FOM builds (see AdvDiff2D.build_bc,
  which reads field.bc_type directly) -- "neumann" or "periodic" both work
  with this same field/class, since the bump is fixed away from the domain
  boundary (sigma small relative to subdomain size) and needs no
  BC-specific construction, unlike AdvDiffUnsteadyField (Dirichlet, needs
  boundary clipping) or PeriodicAdvDiffUnsteadyField (needs a globally
  periodic-consistent Fourier basis).

  cx_random_sign/cy_random_sign (same mechanism as
  AdvDiffUnsteadyFieldPeriodicWave, used for the 10x10 extrapolation test):
  when set, cx_lim/cy_lim are interpreted as a MAGNITUDE range (e.g.
  [1.0, 2.0]) instead of a signed range, and each sample independently
  gets a uniformly random sign -- so cx/cy lands in [-hi,-lo] union
  [lo,hi], never in the near-zero gap that would otherwise make advection
  negligibly slow for some fraction of samples.
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma: float = 0.1,
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    cx_random_sign=False,
    cy_random_sign=False,
    bc_type: str = "neumann",
    forced_config=None,
    speed_lim=None,
    theta_lim=(0.0, 360.0),
    shared_uv: bool = False
  ) -> None:
    """
    shared_uv (optional): if True, v's bump mask+amplitude is an exact
    copy of u's (instead of an independently-sampled mask+amplitude) --
    same mechanism as PoissonWaveMixed's shared_uv, making u(x,y,t) and
    v(x,y,t) identical for every case (same IC, same governing equation,
    same cx/cy, no coupling between u,v in this PDE). This creates the
    same exact row-duplication in the pooled (u,v) snapshot matrix that
    PoissonWaveMixed's shared_uv relies on, letting the trained POD basis
    exploit it to roughly halve the required dimension for a given
    energy_min. mu keeps the same length as shared_uv=False (amp_v is
    still stored, just always equal to amp_u), so nothing downstream
    (u()/v(), get_advection, plot_datasets.py) needs to change.

    speed_lim (optional): if given, advection direction/speed is sampled
    as (speed, theta) instead of independently-bounded (cx, cy) -- speed
    ~ Uniform(*speed_lim), theta ~ Uniform(*theta_lim) degrees, then
    cx = speed*cos(theta), cy = speed*sin(theta). This decouples direction
    (theta) from speed entirely: unlike cx_lim/cy_lim + random_sign (which
    only ever covers 4 diagonal-centered wedges, since bounding |cx| and
    |cy| away from zero independently excludes near-axis directions),
    theta can take any value in theta_lim, including near-horizontal/
    near-vertical, while speed is guaranteed to be exactly speed_lim
    regardless of theta. cx_lim/cy_lim/*_random_sign are ignored when
    speed_lim is not None. mu still stores (cx, cy) at the same position
    as always -- theta is discarded right after conversion, so every
    downstream consumer (get_advection, plot_datasets.py, etc.) is
    unaffected by which sampling mode produced them.
    """
    super().__init__(mesh)
    bc_mod.check_bc_type(bc_type)
    self.bc_type = bc_type
    self.amp_lim = list(amp_lim)
    self.sigma = sigma
    self.cx_lim = list(cx_lim)
    self.cy_lim = list(cy_lim)
    self.cx_random_sign = bool(cx_random_sign)
    self.cy_random_sign = bool(cy_random_sign)
    self.speed_lim = list(speed_lim) if (speed_lim is not None) else None
    self.theta_lim = list(theta_lim)
    self.shared_uv = bool(shared_uv)
    self.configs = None
    self.forced_config = forced_config
    if self.forced_config is not None:
      self.forced_config = np.array(self.forced_config).reshape(-1)
    # Fixed bump centers: geometric center of each subdomain's bounding box.
    x, y = mesh.nodes_val.T
    x0s, y0s = [], []
    for ind in mesh.res_nodes:
      x0s.append(0.5 * (x[ind].min() + x[ind].max()))
      y0s.append(0.5 * (y[ind].min() + y[ind].max()))
    self._x0 = np.array(x0s)
    self._y0 = np.array(y0s)

  def _n_params(self):
    return 2 * self.mesh.n_sub + 2

  # Design space
  # ===================================
  def _init_design_space(self) -> None:
    self.configs = ops.generate_combs([np.arange(2)] * self.mesh.n_sub)[1:]
    if self.forced_config is not None:
      self.configs += self.forced_config.reshape(1, -1)
      self.configs = self.configs.astype(bool).astype(int)
      self.configs = np.unique(self.configs, axis=0)
    n = self.mesh.n_sub
    last_two = ([list(self.speed_lim), list(self.theta_lim)] if (self.speed_lim is not None)
                else [list(self.cx_lim), list(self.cy_lim)])
    self.design_space = np.array(
      [[0, len(self.configs)]] + [self.amp_lim] * n +   # cfg_u, amp_u
      [[0, len(self.configs)]] + [self.amp_lim] * n +   # cfg_v, amp_v
      last_two
    ).T

  def _sample_one_mask(self):
    s = 0.0
    while (s == 0.0):
      config = np.random.binomial(1, p=0.5, size=self.mesh.n_sub)
      if (self.forced_config is not None):
        config = config + self.forced_config
        config = config.astype(bool).astype(int)
      s = np.sum(config)
    return config * np.random.uniform(*self.amp_lim, size=self.mesh.n_sub)

  def _sample_cx_cy(self):
    if self.speed_lim is not None:
      speed = np.random.uniform(*self.speed_lim)
      theta = np.deg2rad(np.random.uniform(*self.theta_lim))
      return speed * np.cos(theta), speed * np.sin(theta)
    cx = np.random.uniform(*self.cx_lim)
    cy = np.random.uniform(*self.cy_lim)
    if self.cx_random_sign:
      cx *= np.random.choice([-1.0, 1.0])
    if self.cy_random_sign:
      cy *= np.random.choice([-1.0, 1.0])
    return cx, cy

  def sample_design_space(self) -> np.ndarray:
    amps_u = self._sample_one_mask()
    amps_v = amps_u if self.shared_uv else self._sample_one_mask()
    cx, cy = self._sample_cx_cy()
    return np.concatenate([amps_u, amps_v, [cx, cy]])

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    dmat = super().construct_design_mat(n_samples)
    return self._convert_dmat_to_mu(dmat)

  def _convert_dmat_to_mu(self, dmat: np.ndarray) -> np.ndarray:
    n = self.mesh.n_sub
    n_samples = dmat.shape[0]
    cfg_u = np.floor(dmat[:, 0]).astype(np.int32)
    amps_u = self.configs[cfg_u] * dmat[:, 1:1 + n]
    if self.shared_uv:
      amps_v = amps_u
    else:
      cfg_v = np.floor(dmat[:, 1 + n]).astype(np.int32)
      amps_v = self.configs[cfg_v] * dmat[:, 2 + n:2 + 2 * n]
    col_a = dmat[:, 2 + 2 * n].copy()
    col_b = dmat[:, 3 + 2 * n].copy()
    if self.speed_lim is not None:
      speed, theta = col_a, np.deg2rad(col_b)
      cx, cy = speed * np.cos(theta), speed * np.sin(theta)
    else:
      cx, cy = col_a, col_b
      if self.cx_random_sign:
        cx = cx * np.random.choice([-1.0, 1.0], size=n_samples)
      if self.cy_random_sign:
        cy = cy * np.random.choice([-1.0, 1.0], size=n_samples)
    # dmat[:, 4+2n:] passes through any trailing columns a subclass appends
    # (e.g. ReactAdvDiffUnsteadyFieldMaskedBump's sigma_react) unchanged.
    extra = dmat[:, 4 + 2 * n:]
    return np.concatenate([amps_u, amps_v, cx[:, None], cy[:, None], extra], axis=1)

  def set_params(self, mu: np.ndarray) -> None:
    self.mu = np.asarray(mu).reshape(-1)

  # Velocity fields
  # ===================================
  def u(self) -> np.ndarray:
    n = self.mesh.n_sub
    return self._make_bump(self.mu[:n])

  def v(self) -> np.ndarray:
    n = self.mesh.n_sub
    return self._make_bump(self.mu[n:2 * n])

  def get_init(self, mu=None) -> np.ndarray:
    if mu is not None:
      self.set_params(mu)
    return np.concatenate([self.u().reshape(-1), self.v().reshape(-1)])

  def _make_bump(self, amps: np.ndarray) -> np.ndarray:
    x, y = self.mesh.nodes_val.T
    f = np.zeros(self.mesh.nxy)
    for a, x0, y0 in zip(amps, self._x0, self._y0):
      if a == 0.0:
        continue
      f += a * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))
    return f

  def get_ic(self):
    n = self.mesh.n_sub
    return self._make_bump(self.mu[:n]), self._make_bump(self.mu[n:2 * n])

  def get_force(self) -> np.ndarray:
    # No forcing. Must be a real zero array, not just "absent" -- BasicField
    # already defines a get_force stub (returns None), so
    # hasattr(field, "get_force") is True regardless and AdvDiff2D.build
    # would call it and crash on self.f = None otherwise.
    return np.zeros(2 * self.mesh.nxy)

  def get_advection(self):
    n = 2 * self.mesh.n_sub
    return float(self.mu[n]), float(self.mu[n + 1])


class ReactAdvDiffUnsteadyFieldMaskedBump(AdvDiffUnsteadyFieldMaskedBump):
  """
  AdvDiffUnsteadyFieldMaskedBump extended with independent per-component
  reaction coefficients sigma_react_u, sigma_react_v (ReactAdvDiff2D
  supports either a single shared sigma or a (sigma_u, sigma_v) pair from
  field.get_reaction() -- this field provides the latter). sigma_u != sigma_v
  is a genuinely new richness axis (two distinct physical decay rates, not
  a per-subdomain-tiling duplicate), so it costs 2 parameters on top of the
  base 4, not 1.

  mu = [amp_u_0..n-1, amp_v_0..n-1, cx, cy, sigma_react_u, sigma_react_v]
  Stored length: 2*n_sub + 4. Collapsed/effective parameter count: 6
  (amplitude_u, amplitude_v, cx, cy, sigma_react_u, sigma_react_v).
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(-1.5, 1.5),
    sigma: float = 0.1,
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    cx_random_sign=False,
    cy_random_sign=False,
    sigma_react_lim=(0.1, 2.0),
    bc_type: str = "neumann",
    forced_config=None,
    speed_lim=None,
    theta_lim=(0.0, 360.0),
    shared_uv: bool = False
  ) -> None:
    super().__init__(mesh, amp_lim, sigma, cx_lim, cy_lim,
                      cx_random_sign, cy_random_sign, bc_type, forced_config,
                      speed_lim, theta_lim, shared_uv)
    self.sigma_react_lim = list(sigma_react_lim)

  def _n_params(self):
    return super()._n_params() + 2

  def _init_design_space(self) -> None:
    super()._init_design_space()
    extra = np.array([list(self.sigma_react_lim)] * 2).T
    self.design_space = np.concatenate([self.design_space, extra], axis=1)

  def sample_design_space(self) -> np.ndarray:
    base = super().sample_design_space()
    sr_u = np.random.uniform(*self.sigma_react_lim)
    sr_v = sr_u if self.shared_uv else np.random.uniform(*self.sigma_react_lim)
    return np.concatenate([base, [sr_u, sr_v]])

  def _convert_dmat_to_mu(self, dmat: np.ndarray) -> np.ndarray:
    # Base class passes dmat[:, 4+2n:] (sigma_react_u, sigma_react_v)
    # through unchanged as "extra" -- when shared_uv, force sigma_react_v
    # to copy sigma_react_u here, same mechanism as amp_v copying amp_u.
    if not self.shared_uv:
      return super()._convert_dmat_to_mu(dmat)
    n = self.mesh.n_sub
    sr_u_col = 4 + 2 * n
    dmat = dmat.copy()
    dmat[:, sr_u_col + 1] = dmat[:, sr_u_col]
    return super()._convert_dmat_to_mu(dmat)

  def get_reaction(self):
    n = 2 * self.mesh.n_sub
    return float(self.mu[n + 2]), float(self.mu[n + 3])


class FisherKPPUnsteadyFieldMaskedBump(AdvDiffUnsteadyFieldMaskedBump):
  """
  AdvDiffUnsteadyFieldMaskedBump extended with independent per-component
  Fisher-KPP logistic growth rates r_u, r_v (FisherKPPAdvDiff2D supports
  either a single shared r or an (r_u, r_v) pair from field.get_reaction()
  -- this field provides the latter). Same structure/cost as
  ReactAdvDiffUnsteadyFieldMaskedBump's sigma_react_u/v (a genuinely new
  richness axis, +2 params on top of the base 4), just a different
  (nonlinear) reaction mechanism.

  Unlike ReactAdvDiffUnsteadyFieldMaskedBump, amp_lim defaults to a
  non-negative range: classic Fisher-KPP treats u as a population density
  in [0,1] (u=0 unstable, u=1 the stable carrying capacity), and the
  traveling-front behavior this field is meant to exercise is cleanest in
  that regime -- allowing negative amplitudes would just give strong decay
  toward 0 (since u(1-u)<0 for u<0), not the classic front dynamics.

  mu = [amp_u_0..n-1, amp_v_0..n-1, cx, cy, r_u, r_v]
  Stored length: 2*n_sub + 4. Collapsed/effective parameter count: 6
  (amplitude_u, amplitude_v, cx, cy, r_u, r_v).
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    amp_lim=(0.0, 1.0),
    sigma: float = 0.1,
    cx_lim=(-1.0, 1.0),
    cy_lim=(-1.0, 1.0),
    cx_random_sign=False,
    cy_random_sign=False,
    r_lim=(1.0, 2.5),
    bc_type: str = "neumann",
    forced_config=None
  ) -> None:
    super().__init__(mesh, amp_lim, sigma, cx_lim, cy_lim,
                      cx_random_sign, cy_random_sign, bc_type, forced_config)
    self.r_lim = list(r_lim)

  def _n_params(self):
    return super()._n_params() + 2

  def _init_design_space(self) -> None:
    super()._init_design_space()
    extra = np.array([list(self.r_lim)] * 2).T
    self.design_space = np.concatenate([self.design_space, extra], axis=1)

  def sample_design_space(self) -> np.ndarray:
    base = super().sample_design_space()
    r_u = np.random.uniform(*self.r_lim)
    r_v = np.random.uniform(*self.r_lim)
    return np.concatenate([base, [r_u, r_v]])

  def get_reaction(self):
    n = 2 * self.mesh.n_sub
    return float(self.mu[n + 2]), float(self.mu[n + 3])


class FisherKPPUnsteadyFieldStep(BasicField):
  """
  Single-front Fisher-KPP field: ONE smoothed circular step function,
  fixed at the domain center, instead of FisherKPPUnsteadyFieldMaskedBump's
  per-subdomain Gaussian bumps (up to 4 simultaneously, each starting well
  below saturation). Meant to isolate the classic Fisher-KPP "compact,
  already-saturated region expanding outward as a single front" behavior
  from the confounds present in the masked-bump family: multiple
  simultaneously-active sources colliding into each other (band/ridge
  artifacts), and a narrow Gaussian peak that spends most of its time
  still growing toward saturation rather than already sitting at it.

  IC (shared by u and v): u(x,y,0) = v(x,y,0) =
    0.5*(1 - tanh((dist - radius) / width)),  dist = ||(x,y) - (x0,y0)||,
  (x0,y0) = domain center (fixed, not sampled -- keeps the front's
  radial-expansion geometry simple/symmetric and away from boundary
  effects at early times). This is ~1 well inside the disk of the given
  radius, ~0 well outside it, with a smooth (not discontinuous)
  transition of the given width -- a numerically well-behaved stand-in
  for the classic Heaviside/step IC (a literal jump would trigger
  Gibbs-type oscillations in this centered finite-difference scheme).

  mu = [radius, cx, cy, r_u, r_v]. Stored length: 5.

  cx_random_sign/cy_random_sign: same convention as
  AdvDiffUnsteadyFieldMaskedBump -- cx_lim/cy_lim are interpreted as a
  magnitude range and each sample independently gets a random sign.
  """

  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    radius_lim=(0.05, 0.15),
    width: float = 0.02,
    cx_lim=(0.2, 0.4),
    cy_lim=(0.2, 0.4),
    cx_random_sign: bool = False,
    cy_random_sign: bool = False,
    r_lim=(1.0, 2.5),
    bc_type: str = "neumann"
  ) -> None:
    super().__init__(mesh)
    bc_mod.check_bc_type(bc_type)
    self.bc_type = bc_type
    self.radius_lim = list(radius_lim)
    self.width = float(width)
    self.cx_lim = list(cx_lim)
    self.cy_lim = list(cy_lim)
    self.cx_random_sign = bool(cx_random_sign)
    self.cy_random_sign = bool(cy_random_sign)
    self.r_lim = list(r_lim)
    # Fixed step center: geometric center of the full domain.
    x, y = mesh.nodes_val.T
    self._x0 = 0.5 * (x.min() + x.max())
    self._y0 = 0.5 * (y.min() + y.max())

  def _n_params(self):
    return 5

  # Design space
  # ===================================
  def _init_design_space(self) -> None:
    self.design_space = np.array(
      [list(self.radius_lim), list(self.cx_lim), list(self.cy_lim),
       list(self.r_lim), list(self.r_lim)]
    ).T

  def sample_design_space(self) -> np.ndarray:
    radius = np.random.uniform(*self.radius_lim)
    cx = np.random.uniform(*self.cx_lim)
    cy = np.random.uniform(*self.cy_lim)
    if self.cx_random_sign:
      cx *= np.random.choice([-1.0, 1.0])
    if self.cy_random_sign:
      cy *= np.random.choice([-1.0, 1.0])
    r_u = np.random.uniform(*self.r_lim)
    r_v = np.random.uniform(*self.r_lim)
    return np.array([radius, cx, cy, r_u, r_v])

  def construct_design_mat(self, n_samples: int) -> np.ndarray:
    dmat = super().construct_design_mat(n_samples)
    if self.cx_random_sign:
      dmat[:, 1] *= np.random.choice([-1.0, 1.0], size=n_samples)
    if self.cy_random_sign:
      dmat[:, 2] *= np.random.choice([-1.0, 1.0], size=n_samples)
    return dmat

  def set_params(self, mu: np.ndarray) -> None:
    self.mu = np.asarray(mu).reshape(-1)

  # Velocity fields
  # ===================================
  def _make_step(self) -> np.ndarray:
    x, y = self.mesh.nodes_val.T
    radius = self.mu[0]
    dist = np.sqrt((x - self._x0)**2 + (y - self._y0)**2)
    return 0.5 * (1.0 - np.tanh((dist - radius) / self.width))

  def u(self) -> np.ndarray:
    return self._make_step()

  def v(self) -> np.ndarray:
    return self._make_step()

  def get_init(self, mu=None) -> np.ndarray:
    if mu is not None:
      self.set_params(mu)
    return np.concatenate([self.u().reshape(-1), self.v().reshape(-1)])

  def get_ic(self):
    step = self._make_step()
    return step, step

  def get_force(self) -> np.ndarray:
    return np.zeros(2 * self.mesh.nxy)

  def get_advection(self):
    return float(self.mu[1]), float(self.mu[2])

  def get_reaction(self):
    return float(self.mu[3]), float(self.mu[4])
