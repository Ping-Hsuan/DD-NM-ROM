import numpy as np
import scipy.sparse as sp

from time import time

from .basic import Solver


class Newton(Solver):

  def __init__(
    self,
    model,
    tol=1e-3,
    maxit=20,
    stepsize_min=1e-10,
    verbose=False
  ):
    super(Newton, self).__init__(
      model=model,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )

  def solve(
    self,
    x0
  ):
    """
    Solve RHS(x) = 0 using Newton's method.
    inputs:
      rhs_jac: rhsction that returns F(y) and the jacobian of F at y, i.e.
          F, Jac = rhs_jac(y)
      x0: initial guess for Newton's method
      self.tol: [optional] solver tolerance. Default is 1e-10
      self.maxit: [optional] maximum number of iterations. Default is 100
      self.verbose: [optional] Set to True to print iteration history. Default is False

    outputs:
      y: solution of F(y) = 0
      res_vecs: array of residual vectors F(y_i) at each iteration, i.e.
            res_vecs[i] = F(y_i) = residual at ith iteration
      res_hist: residual iteration history
      step_hist: stepsize history
      iter: number of iterations
    """
    # Initialize
    # ---------------
    # > Set first step
    it, x = 0, x0
    rhs, jac, res = self.evaluate(x)
    # > Set histories
    start = time()
    rhs_hist = [rhs]
    res_hist = [res]
    step_hist = [0.0]
    self.model.runtime["total"] += time()-start
    # > Choose a sparse or dense linear solver depending on the problem
    solve = sp.linalg.spsolve if sp.issparse(jac) else np.linalg.solve
    # > Print first step
    self.print_step(it, step_hist[-1], res_hist[-1], header=True)
    # Loop until convergence
    # ---------------
    flag = 0
    while ((res_hist[-1] >= self.tol) and (it < self.maxit)):
      # > Initialize line search
      start = time()
      dx = solve(jac,-rhs)
      delta = time()-start
      self.model.runtime["total"] += delta
      self.model.runtime["linalg"] += delta
      # > Armijo line search
      eval_res_tol = lambda stepsize: (1.0 - 2e-4*stepsize)*res_hist[-1]
      x, rhs, jac, res, stepsize = self.line_search(x, dx, eval_res_tol)
      # > Update
      start = time()
      it += 1
      rhs_hist.append(rhs)
      res_hist.append(res)
      step_hist.append(stepsize)
      self.model.runtime["total"] += time()-start
      # > Print step
      self.print_step(it, step_hist[-1], res_hist[-1])
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
      np.array(res_hist),
      np.array(step_hist),
      np.array(it).reshape(1),
      np.array(flag).reshape(1)
    )
    self.model.runtime["total"] += time()-start
    return out
