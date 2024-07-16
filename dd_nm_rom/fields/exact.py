import numpy as np

from .basic import BasicField
from dd_nm_rom.fom.elements.bound_cond import SIDES


class Burgers2DExact(BasicField):
  """
  See:
    - https://onlinelibrary.wiley.com/doi/epdf/10.1002/fld.1650030302
    - https://doi.org/10.1016/j.cma.2021.113997
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh,
    nu=1e-2,
    a_lim=[0.9,1.1],
    k_lim=[0.9,1.1],
  ):
    super(Burgers2DExact, self).__init__(mesh)
    self.x0 = 1.0
    self.a = np.zeros(4)
    self.nu = float(nu)
    self.Re = 1/self.nu
    self.a_lim = list(a_lim)
    self.k_lim = list(k_lim)

  # Design space
  # ===================================
  def _init_design_space(self):
    self.design_space = np.array([self.a_lim, self.k_lim]).T

  def set_params(self, mu):
    mu = mu.reshape(-1)
    self.a[:2] = mu[0]
    self.k = mu[1]

  # Velocity field
  # ===================================
  def u(self, x, y):
    f = self.a[1] + self.a[3]*y + self.k*self._psi(x,-1)*np.cos(self.k*y)
    return f/self._phi(x, y)

  def v(self, x, y):
    f = self.a[2] + self.a[3]*x - self.k*self._psi(x)*np.sin(self.k*y)
    return f/self._phi(x, y)

  def _phi(self, x, y):
    f = self.a[0] + self.a[1]*x + self.a[2]*y \
      + self.a[3]*x*y + self._psi(x)*np.cos(self.k*y)
    return -0.5*self.Re*f

  def _psi(self, x, sign=1):
    f = self.k*(x-self.x0)
    return np.exp(f) + sign*np.exp(-f)

  def get_init(self):
    raise NotImplementedError

  # Boundary conditions
  # ===================================
  def get_bc_funval(self):
    funval = {}
    for side in SIDES["all"]:
      funval[side] = {}
      for z in ("u", "v"):
        funval[side][z] = self.u if (z == "u") else self.v
    return funval
