import abc
import numpy as np

from pyDOE import lhs
from dd_nm_rom.fom.elements.bound_cond import SIDES


class BasicField(object):

  # Initialization
  # ===================================
  def __init__(self, mesh):
    self.mu = None
    self.mesh = mesh
    self.design_space = None
    self.bc_type = "dirichlet"

  @property
  def mesh(self):
    return self._mesh

  @mesh.setter
  def mesh(self, value):
    self._mesh = value

  # Design space
  # ===================================
  def init_design_space(self):
    if (self.design_space is None):
      self._init_design_space()

  @abc.abstractmethod
  def _init_design_space(self):
    pass

  def sample_design_space(self):
    self.init_design_space()
    return self.construct_design_mat(n_samples=1).reshape(-1)

  def construct_design_mat(self, n_samples):
    self.init_design_space()
    # Construct
    ddim = self.design_space.shape[1]
    dmat = lhs(ddim, int(n_samples))
    # Rescale
    amin, amax = self.design_space
    return dmat * (amax - amin) + amin

  def construct_design_mat_test(self, n_samples, dmat_train=[], tol=1e-3):
    dmat_test = []
    for _ in range(n_samples):
      sample, dist = self._compute_sample_dist(dmat_test, dmat_train)
      if (len(dist) == 0):
        dist = 1.0 + tol
      while (np.any(dist < tol)):
        sample, dist = self._compute_sample_dist(dmat_test, dmat_train)
      if (len(dmat_test) == 0):
        dmat_test = sample
      else:
        dmat_test = np.vstack([dmat_test, sample])
    return dmat_test

  def _compute_sample_dist(self, dmat_test, dmat_train):
    dist = np.array([])
    sample = self.sample_design_space().reshape(1,-1)
    if (len(dmat_test) > 0):
      dist = np.append(dist, np.linalg.norm(dmat_test - sample, axis=-1))
    if (len(dmat_train) > 0):
      dist = np.append(dist, np.linalg.norm(dmat_train - sample, axis=-1))
    return sample, dist

  @abc.abstractmethod
  def set_params(self, *args, **kwargs):
    pass

  # Velocity field
  # ===================================
  @abc.abstractmethod
  def u(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def v(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def get_init(self, *args, **kwargs):
    pass

  # Boundary conditions
  # ===================================
  def get_bc_funval(self):
    funval = {}
    for side in SIDES["all"]:
      funval[side] = {}
      for z in ("u", "v"):
        funval[side][z] = lambda x, y: np.zeros_like(x)
    return funval
