import numpy as np

from .utils import get_axis_method, SIDES


class DirichletBC(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    nu,
    mesh,
    funval
  ):
    self.name = "dirichlet"
    self.nu = nu
    self.mesh = mesh
    self.funval = funval
    self.f = None
    self.op = None
    # Control variables
    self.built = False
    self.update_built = True

  def is_built(self):
    if (not self.built):
      raise ValueError("BC not built. Please, call 'build' method first.")

  # Build
  # ===================================
  def build(self):
    self.f = {}
    # Loop over variables
    for k in ("u", "v"):
      f_k = {}
      # Loop over sides
      for (side, sfunval) in self.funval.items():
        axis, method = get_axis_method(side)
        f_k[side] = self._build_src(
          funval=sfunval[k],
          axis=axis,
          method=method
        )
      self.f[k] = {
        "A": self._compose_adv_src(f_k),
        "D": self._compose_dif_src(f_k)
      }
    if self.update_built:
      self.built = True

  def _build_src(
    self,
    funval,
    axis="x",
    method="fwd"
  ):
    n, lim = self.mesh.n, self.mesh.lim
    index = 0 if (method == "fwd") else -1
    e0 = np.zeros(n[axis])
    e0[index] = 1.0
    if (axis == "x"):
      x = np.full(n["y"], lim["x"][index])
      y = self.mesh.nodes_1d["y"]
      return np.kron(funval(x, y), e0)
    else:
      x = self.mesh.nodes_1d["x"]
      y = np.full(n["x"], lim["y"][index])
      return np.kron(e0, funval(x, y))

  def _compose_adv_src(
    self,
    values
  ):
    f = {}
    for axis in ("x", "y"):
      si = SIDES[axis]
      fi = values[si[0]] - values[si[1]]
      f[axis] = -(0.5/self.mesh.h[axis]) * fi
    return f

  def _compose_dif_src(
    self,
    values
  ):
    f = 0.0
    for axis in ("x", "y"):
      si = SIDES[axis]
      fi = values[si[0]] + values[si[1]]
      f = f + (self.nu/self.mesh.h[axis]**2) * fi
    return f
