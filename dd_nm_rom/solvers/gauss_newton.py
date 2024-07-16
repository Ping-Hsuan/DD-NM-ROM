import numpy as np
import scipy as sp

from time import time

from .basic import Solver


class GaussNewton(Solver):

  def __init__(
    self,
    model,
    tol=1e-3,
    maxit=20,
    stepsize_min=1e-10,
    verbose=False
  ):
    super(GaussNewton, self).__init__(
      model=model,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    self.squared_res = True

  def solve(
    self,
    x0
  ):
    """
    Solve min 0.5*||r(x)||^2 using Gauss-Newton method.

    inputs:
      x0: initial guess for Newton"s method
      tol: [optional] solver tolerance. Default is 1e-10
      self.maxit: [optional] maximum number of iterations. Default is 20
      verbose: [optional] Set to True to print iteration history. Default is False

    outputs:
      x: solution of min ||r(x)||
      conv_hist: convergence iteration history: conv_hist[i] = ||R"r(x_i)||
      step_hist: stepsize history
      it: number of iterations
    """
    # Initialize
    # ---------------
    # > Set first step
    it, x = 0, x0
    rhs, jac, res = self.evaluate(x)
    # > Set histories
    start = time()
    rhs_hist = [rhs]
    conv_hist = [np.linalg.norm(jac.T@rhs)]
    step_hist = [0.0]
    self.model.runtime["total"] += time()-start
    # > Print first step
    self.print_step(it, step_hist[-1], conv_hist[-1], header=True)
    # Loop until convergence
    # ---------------
    flag = 0
    while ((conv_hist[-1] >= self.tol) & (it < self.maxit)):
      # > Initialize line search
      start = time()
      dx, minval = sp.linalg.lstsq(jac,-rhs)[:2]
      delta = time()-start
      self.model.runtime["total"] += delta
      self.model.runtime["linalg"] += delta
      # > Armijo line search
      eval_res_tol = lambda stepsize: res + 2e-4*stepsize*(minval-res)
      x, rhs, jac, res, stepsize = self.line_search(x, dx, eval_res_tol)
      # > Update
      start = time()
      it += 1
      rhs_hist.append(rhs)
      conv_hist.append(np.linalg.norm(jac.T@rhs))
      step_hist.append(stepsize)
      self.model.runtime["total"] += time()-start
      # > Print step
      self.print_step(it, step_hist[-1], conv_hist[-1])
      # > Check convergence
      if (stepsize < self.stepsize_min):
        flag = 1
        break
      if np.isnan(res):
        flag = 2
        break
    if (it == self.maxit):
      flag = 3
    # Return result
    # ---------------
    start = time()
    out = (
      x,
      np.vstack(rhs_hist),
      np.array(conv_hist),
      np.array(step_hist),
      np.array(it).reshape(1),
      np.array(flag).reshape(1)
    )
    self.model.runtime["total"] += time()-start
    return out
