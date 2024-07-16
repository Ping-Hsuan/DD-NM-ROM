import numpy as np

from dd_nm_rom import ops

from .monolithic import MeshMono


class MeshDD(MeshMono):

  # Initialization
  # ===================================
  def __init__(
    self,
    nx_intr,
    ny_intr,
    lx_sub,
    ly_sub,
    x0=0.0,
    y0=0.0,
    n_sub_x=2,
    n_sub_y=2
  ):
    # Number of interior nodes per subdomain
    self.n_intr = {"x": int(nx_intr), "y": int(ny_intr)}
    # Physical length per subdomain
    self.l_sub = {"x": float(lx_sub), "y": float(ly_sub)}
    # Full domain origin
    self.origin = {"x": float(x0), "y": float(y0)}
    # Number of subdomains
    self.n_subs = {"x": int(n_sub_x), "y": int(n_sub_y)}
    self.n_sub = self.n_subs["x"]*self.n_subs["y"]
    # Control variables
    self.built = False
    self.update_built = False

  # Building
  # ===================================
  def build(self):
    self.n, self.phylim = {}, {}
    for axis in ("x", "y"):
      self.n[axis] = int(self.n_subs[axis]*(self.n_intr[axis]+2)-2)
      self.phylim[axis] = [
        self.origin[axis],
        self.origin[axis] + float(self.l_sub[axis]*self.n_subs[axis])
      ]
    super(MeshDD, self).build()
    # Subdomains indices
    self.sub_ind = {axis: self.get_indices(axis) for axis in ("x", "y")}
    # x-y subdomains combinations
    self.sub_combs = [np.arange(self.n_subs["y"]), np.arange(self.n_subs["x"])]
    self.sub_combs = ops.generate_combs(self.sub_combs)
    self.set_res_nodes()
    # Update control variable
    self.built = True

  def get_indices(self, axis):
    sub_ind = np.arange(self.n[axis])
    sub_ind_split = []
    si, ei = 0, self.n_intr[axis]+1
    for _ in range(self.n_subs[axis]-1):
      sub_ind_split.append(sub_ind[si:ei])
      si, ei = ei, ei+self.n_intr[axis]+2
    sub_ind_split.append(sub_ind[si:])
    return sub_ind_split

  def set_res_nodes(self):
    self.res_ind = []
    self.res_nodes = []
    for sub in self.sub_combs:
      ind_s = np.ix_(self.sub_ind["y"][sub[0]], self.sub_ind["x"][sub[1]])
      res_s = self.nodes_ind[ind_s].flatten()
      self.res_ind.append(ind_s)
      self.res_nodes.append(np.sort(res_s))

  # Configuration
  # ===================================
  def get_config(self):
    return self.get_config_dd()

  def get_config_dd(self):
    return {
      "nx_intr": self.n_intr["x"],
      "ny_intr": self.n_intr["y"],
      "lx_sub": self.l_sub["x"],
      "ly_sub": self.l_sub["y"],
      "x0": self.phylim["x"][0],
      "y0": self.phylim["y"][0],
      "n_sub_x": self.n_subs["x"],
      "n_sub_y": self.n_subs["y"]
    }
