import numpy as np

from .poisson_wave import PoissonWaveDirichlet


class PoissonWaveDirichletPositive(PoissonWaveDirichlet):
  """
  Variant of `PoissonWaveDirichlet` targeting positive Poisson solutions.

  Construction:
  - forcing is non-positive everywhere
      f(x,y) = -Af * 0.5 * (sin(2π(...)) + 1) <= 0
  - Dirichlet boundary is strictly positive everywhere
      g(x,y) = bc_floor + Ab * 0.5 * (sin(2π(...)) + 1) >= bc_floor > 0

  With the code convention for Poisson in this repo, this sign choice is the
  one that enforces a positive solution profile (discrete maximum principle).
  """

  def __init__(
    self,
    *args,
    bc_floor: float = 0.1,
    **kwargs
  ) -> None:
    super(PoissonWaveDirichletPositive, self).__init__(*args, **kwargs)
    self.bc_floor = float(bc_floor)
    if (self.bc_floor <= 0):
      raise ValueError("'bc_floor' must be > 0 to enforce positive BC values.")

  @staticmethod
  def _shifted_wave(
    x: np.ndarray,
    y: np.ndarray,
    A: float,
    kx: float,
    ky: float,
    th: float
  ) -> np.ndarray:
    wave = np.sin(2.0 * np.pi * (kx * x + ky * y + th))
    return A * 0.5 * (wave + 1.0)

  def _forcing_uv(self):
    x, y = self.mesh.nodes_val.T
    pu, pv = self.params_u, self.params_v
    fu = -self._shifted_wave(x, y, pu["Af"], pu["kx"], pu["ky"], pu["th_f"])
    fv = -self._shifted_wave(x, y, pv["Af"], pv["kx"], pv["ky"], pv["th_f"])
    shape = (self.mesh.n["y"], self.mesh.n["x"])
    return fu.reshape(shape), fv.reshape(shape)

  def g_dir_u(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = self.params_u
    return self.bc_floor + self._shifted_wave(
      x, y, p["Ab"], p["kbx"], p["kby"], p["th_b"]
    )

  def g_dir_v(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = self.params_v
    return self.bc_floor + self._shifted_wave(
      x, y, p["Ab"], p["kbx"], p["kby"], p["th_b"]
    )
