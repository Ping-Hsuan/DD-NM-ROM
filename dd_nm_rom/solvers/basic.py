import abc
import numpy as np

from time import time

_PRINT_FMT = "| {0:11d} | {1:11.4e} | {2:11.4e} |"


class Solver(object):

  def __init__(
    self,
    model,
    tol=1e-3,
    maxit=20,
    stepsize_min=1e-10,
    verbose=False
  ):
    self.model = model
    self.tol = tol
    self.maxit = maxit
    self.stepsize_min = stepsize_min
    self.verbose = verbose
    self.squared_res = False
    self.set_header()

  def set_header(self):
    self.header = "|   Iteration |    Stepsize |    Residual |\n| "
    n_chars = len(self.header.split("|")[1])-2
    for _ in range(3):
      self.header += "-"*n_chars + " | "
    self.header = self.header[:-1]

  # Call function
  # ===================================
  def __call__(
    self,
    x0,
    dt=0.0,
    nt=1,
    guess=None,
    use_guess=False
  ):
    return self.integrate(x0, dt, nt, guess, use_guess)

  def integrate(
    self,
    x0,
    dt=0.0,
    nt=1,
    guess=None,
    use_guess=False
  ):
    # Initialize
    start = time()
    x = [x0]
    self.model.t = 0.0
    self.model.dt = dt
    self.model.x_old = x0
    self.model.runtime["total"] += time()-start
    # Loop over time steps
    for i in range(nt):
      if self.verbose:
        print("Time step {0:4d}/{1:d}".format(i+1,nt))
        texec = time()
      # Solve
      self.model.t += dt
      xi, *step = self.solve(self.model.x_old)
      # Check convergence
      res, it, flag = step[1][-1], int(step[-2]), int(step[-1])
      self.print_conv(res, it, flag)
      start = time()
      # Update
      self.model.x_old = xi
      if use_guess:
        self.model.x_old = guess[i]
      # Store
      x.append(xi)
      if (i == 0):
        steps = [[] for _ in step]
      for (j, obj) in enumerate(step):
        steps[j].append(obj)
      if (flag != 0):
        break
      self.model.runtime["total"] += time()-start
      if self.verbose:
        print("Execution time: {:.5e} s".format(time()-texec))
    # Return
    start = time()
    x = np.vstack(x).T
    if ((dt == 0.0) and (nt == 1)):
      x = x[:,-1]
      steps = [obj[-1] for obj in steps]
    self.model.runtime["total"] += time()-start
    return x, *steps

  @abc.abstractmethod
  def solve(self, x0):
    pass

  # Util functions
  # ===================================
  # Solving
  # -----------------------------------
  def line_search(
    self,
    x0,
    dx,
    eval_res_tol
  ):
    # Initialize
    # -------------
    start = time()
    stepsize = 1.0
    x = x0 + stepsize*dx
    self.model.runtime["total"] += time()-start
    rhs, jac, res = self.evaluate(x)
    # Condition
    # -------------
    start = time()
    cond_fun = lambda res, stepsize: (
      (res >= eval_res_tol(stepsize)) and (stepsize >= self.stepsize_min)
    )
    cond = cond_fun(res, stepsize)
    self.model.runtime["total"] += time()-start
    while cond:
      # Update solution
      # -------------
      start = time()
      stepsize *= 0.5
      x = x0 + stepsize*dx
      self.model.runtime["total"] += time()-start
      rhs, jac, res = self.evaluate(x)
      # Condition
      # -------------
      start = time()
      cond = cond_fun(res, stepsize)
      self.model.runtime["total"] += time()-start
    return x, rhs, jac, res, stepsize

  def evaluate(
    self,
    x
  ):
    rhs, jac = self.model.rhs_jac(x)
    start = time()
    res = np.dot(rhs,rhs)
    if (not self.squared_res):
      res = np.sqrt(res)
    self.model.runtime["total"] += time()-start
    return rhs, jac, res

  # Printing
  # -----------------------------------
  def print_step(
    self,
    it,
    stepsize,
    res,
    header=False
  ):
    if self.verbose:
      if header:
        print(self.header)
      print(_PRINT_FMT.format(it, stepsize, res))

  def print_conv(
    self,
    res,
    it,
    flag
  ):
    if (flag == 1):
      print(f"Too small stepsize found at iteration {it}.")
    elif (flag == 2):
      print("The residual value is 'nan'.")
    elif (flag == 3):
      print(f"Solver failed to converge in {self.maxit} iterations.")
    else:
      if self.verbose:
        print(
          f"Solver terminated after {it} iterations " \
            f"with residual norm of {res:1.4e}."
        )
