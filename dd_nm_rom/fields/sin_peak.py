import numpy as np

from .sin_multi_peak import SinMultiPeak


class SinPeak(SinMultiPeak):

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh,
    mu_lim=[0.9,1.1],
  ):
    super(SinPeak, self).__init__(mesh, mu_lim)

  # Design space
  # ===================================
  def _init_design_space(self):
    # Define possible combinations
    self.configs = np.array([1,0,0,0])
    # Define design space
    self.design_space = np.array(self.mu_lim).reshape(-1,1)

  def sample_design_space(self):
    self.init_design_space()
    return self.configs * np.random.uniform(*self.mu_lim)

  def _convert_dmat_to_mu(self, dmat):
    return self.configs.reshape(1,-1) * dmat
