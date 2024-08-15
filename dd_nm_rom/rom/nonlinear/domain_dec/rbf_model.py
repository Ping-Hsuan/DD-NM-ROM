import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

from time import time
from scipy.interpolate import RBFInterpolator


class RBFModel(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    subdomains,
    n_constraints
  ):
    self.interpolator = None
    self.subdomains = subdomains
    self.n_constraints = n_constraints

  def __call__(
    self,
    mu,
    runtime=0.0
  ):
    return self.call(mu, runtime)

  def call(
    self,
    mu,
    runtime=0.0
  ):
    if (mu.ndim == 1):
      return self.forward(mu, runtime)
    else:
      x = []
      for mu_i in mu:
        x_i, runtime_i = self.forward(mu_i)
        x.append(x_i)
        runtime += runtime_i
      return np.vstack(x), runtime

  # Solution
  # ===================================
  def forward(
    self,
    mu,
    runtime=0.0
  ):
    """
    Computes initial iterate used for solving NM-ROM optimization subproblem.
    """
    start = time()
    if (self.interpolator is None):
      raise ValueError("RBF interpolator not initialized.")
    mu = mu.reshape(1,-1)
    z, res, jac = [], [], []
    lambdas = np.zeros(self.n_constraints)
    runtime += time()-start
    runtime_s = 0.0
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      z_s = {}
      for e_k in ("interior", "interface"):
        z_s[e_k] = self.interpolator[e_k][s](mu).squeeze()
      res_s, *_, cjac_s = sub.res_jac(z_s, lambdas)
      runtime_s = max(time()-start_s, runtime_s)
      start = time()
      # Solution
      # -----------
      for e_k in ("interior", "interface"):
        z.append(z_s[e_k])
      # Lambdas
      # -----------
      size = sub.rom_dim["interface"]
      res.append(res_s[-size:])
      # Jacobian
      jac.append(cjac_s.T[-size:])
      runtime += time()-start
    runtime += runtime_s
    # Get initial ROM solution and lambdas
    start = time()
    z = np.concatenate(z)
    jac = sp.vstack(jac).toarray()
    res = np.concatenate(res)
    lambdas = la.lstsq(jac, -res)[0]
    x0 = np.concatenate([z, lambdas])
    runtime += time()-start
    return x0, runtime

  # Interpolator
  # ===================================
  def build(
    self,
    x,
    mu,
    smoothing=0.0,
    kernel='linear'
  ):
    # Loop over elements
    self.interpolator = {}
    for e_k in ("interior", "interface"):
      # Loop over subdomains
      self.interpolator[e_k] = []
      for (s, sub) in enumerate(self.subdomains):
        if (len(x[e_k][s]) != len(mu)):
          raise ValueError(
            "The number of data points (first dimension) in both the input " \
              "parameter space matrix and the FOM solution matrix must match."
          )
        z = []
        for xi in x[e_k][s]:
          z.append(sub.elem_states[e_k].encode(xi, with_jac=False))
        z = np.vstack(z)
        self.interpolator[e_k].append(
          RBFInterpolator(y=mu, d=z, smoothing=smoothing, kernel=kernel)
        )
