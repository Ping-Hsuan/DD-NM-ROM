import numpy as np

from dd_nm_rom import ops
from .basic import BasicField


class SinMultiPeak(BasicField):

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh,
    mu_lim=[0.9,1.1],
    forced_config=None
  ):
    super(SinMultiPeak, self).__init__(mesh)
    self.bc_type = "neumann"
    self.mu_lim = mu_lim
    self.configs = None
    self.forced_config = forced_config
    if (self.forced_config is not None):
      self.forced_config = np.array(self.forced_config).reshape(-1)

  # Design space
  # ===================================
  def _init_design_space(self):
    # Define possible combinations
    self.configs = ops.generate_combs([np.arange(2)]*self.mesh.n_sub)[1:]
    if (self.forced_config is not None):
      self.configs += self.forced_config.reshape(1,-1)
      self.configs = self.configs.astype(bool).astype(int)
      self.configs = np.unique(self.configs, axis=0)
    # Define design space
    self.design_space = [[0,len(self.configs)]] + [self.mu_lim]*self.mesh.n_sub
    self.design_space = np.array(self.design_space).T

  def sample_design_space(self):
    s = 0.0
    while (s == 0.0):
      config = np.random.binomial(1, p=0.5, size=self.mesh.n_sub)
      if (self.forced_config is not None):
        config += self.forced_config
        config = config.astype(bool).astype(int)
      s = np.sum(config)
    mu = config * np.random.uniform(*self.mu_lim, size=self.mesh.n_sub)
    return mu

  def construct_design_mat(self, n_samples):
    dmat = super(SinMultiPeak, self).construct_design_mat(n_samples)
    return self._convert_dmat_to_mu(dmat)

  def _convert_dmat_to_mu(self, dmat):
    cfg = np.floor(dmat[:,0]).astype(np.int32)
    return self.configs[cfg] * dmat[:,1:]

  def set_params(self, mu):
    self.mu = mu.reshape(-1)

  # Velocity field
  # ===================================
  def u(self):
    return self.generate_field()

  def v(self):
    return self.generate_field()

  def generate_field(self):
    f = np.zeros(self.mesh.nxy)
    x, y = self.mesh.nodes_val.T
    for (i, mu_i) in enumerate(self.mu):
      ind = self.mesh.res_nodes[i]
      f[ind] = self._phi(x[ind], y[ind], mu_i)
    return f.reshape(self.mesh.n["y"], self.mesh.n["x"])

  def _phi(self, x, y, mu):
    return np.abs(mu*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))

  def get_init(self, mu=None):
    if (mu is not None):
      self.set_params(mu)
    return np.concatenate([self.u().reshape(-1), self.v().reshape(-1)])
