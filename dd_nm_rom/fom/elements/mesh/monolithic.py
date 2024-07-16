import numpy as np

from dd_nm_rom import ops


class MeshMono(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    nx,
    ny,
    x_lim,
    y_lim
  ):
    # Number of nodes
    self.n = {"x": int(nx), "y": int(ny)}
    # Physical limits
    self.phylim = {
      "x": np.sort(x_lim).astype(float),
      "y": np.sort(y_lim).astype(float)
    }
    # Control variables
    self.built = False
    self.update_built = True

  def is_built(self):
    if (not self.built):
      raise ValueError("Mesh not built. Please, call 'build' method first.")

  # Building
  # ===================================
  def build(self):
    # Total number of nodes
    self.nxy = self.n["x"] * self.n["y"]
    # Spacing and nodes location
    self.h, self.lim, self.nodes_1d = {}, {}, {}
    for (axis, lim) in self.phylim.items():
      n = self.n[axis]
      h = (lim[1]-lim[0])/(n+2)
      self.h[axis] = h
      self.lim[axis] = [lim[0]+0.5*h, lim[1]-0.5*h]
      self.nodes_1d[axis] = lim[0] + (1.5+np.arange(n))*h
    self.hxy = self.h["x"] * self.h["y"]
    self.grid = np.meshgrid(self.nodes_1d["x"], self.nodes_1d["y"])
    # Nodes indices
    self.nodes_val = [self.nodes_1d["y"], self.nodes_1d["x"]]
    self.nodes_val = ops.generate_combs(self.nodes_val)
    self.nodes_ind = np.arange(self.nxy).reshape(self.n["y"], self.n["x"])
    # Update control variable
    if self.update_built:
      self.built = True

  # Configuration
  # ===================================
  def get_config(self):
    return self.get_config_mono()

  def get_config_mono(self):
    return {
      "nx": self.n["x"],
      "nx": self.n["y"],
      "x_lim": self.phylim["x"],
      "y_lim": self.phylim["y"]
    }

  def get_config_dd(
    self,
    n_sub_x=2,
    n_sub_y=2
  ):
    return {
      "nx_intr": self._get_n_intr(self.n["x"], n_sub_x),
      "ny_intr": self._get_n_intr(self.n["y"], n_sub_y),
      "lx_sub": self._get_l_sub(self.phylim["x"], n_sub_x),
      "ly_sub": self._get_l_sub(self.phylim["y"], n_sub_y),
      "x0": self.phylim["x"][0],
      "y0": self.phylim["y"][0],
      "n_sub_x": n_sub_x,
      "n_sub_y": n_sub_y
    }

  def _get_n_intr(self, n, n_sub):
    """Compute number of interior nodes for each subdomain."""
    return int((n+2)/n_sub-2)

  def _get_l_sub(self, lim, n_sub):
    """Compute the physical length for each subdomain."""
    return np.abs(np.diff(lim))/n_sub
