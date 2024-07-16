import numpy as np

from time import time

class DDIndices(object):
  """
  Class for generating residual, interior, and interface subdomain indices for a steady-state 2D Burgers FOM.

  inputs:
  monolithic: instance of `Burgers2D` class
  n_sub_x: integer number of subs in x direction
  n_sub_y: integer number of subs in y direction

  fields:
  rhs:       list, rhs[i] = array of residual indices on subdomain i
  interior:  list, interior[i] = array of interior indices on subdomain i
  interface: list, interface[i] = array of interface indices on subdomain i
  full:      list, full[i] = array of full state (interior and interface) indices on subdomain i
  self.skeleton:  array of indices of self.skeleton, i.e. all interface states for all subs
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh,
    monolithic
  ):
    self.mesh = mesh
    self.monolithic = monolithic
    self.ops = list(self.monolithic.ops.values())
    # Set indices
    self.res = self.mesh.res_nodes # Active residual nodes for each subdomain
    self.set_allactive_nodes()
    self.set_interior_interface_nodes()
    # Set ports
    self.set_ports_nodes()
    self.set_ports_maps()

  # Nodes indices
  # ===================================
  # All-active (interior+interface) nodes
  # -----------------------------------
  def set_allactive_nodes(self):
    self.allactive = []
    for res_s in self.res:
      # Initialize subdomain "all" indices
      all_s = set(res_s)
      # For each row (i.e., node in the grid),
      # finds nonzero columns (due to FD stencil)
      for op in self.ops:
        cols = op[res_s].nonzero()[1]
        cols = np.unique(cols).tolist()
        all_s = all_s.union(set(cols))
      # Store indices for current subdomain
      all_s = np.sort(np.array(list(all_s)))
      self.allactive.append(all_s)

  # Interior and interface nodes
  # -----------------------------------
  def set_interior_interface_nodes(self):
    self.skeleton = set()     # All interface nodes
    self.interior = []        # Interior nodes for each subdomain
    self.interface = []       # Interface nodes for each subdomain
    # Loop over subs
    subs = np.arange(self.mesh.n_sub)
    for i in subs:
      # Set i-th subdomain
      sub_i = set(self.allactive[i])
      # Set remaining subs
      subs_left = np.delete(subs, i)
      # Define interface nodes for i-th subdomain
      intf_i = set()
      for j in subs_left:
        # > Take intersection between subdomain i and j
        sub_j = set(self.allactive[j])
        intf_ij = sub_i.intersection(sub_j)
        intf_i = intf_i.union(intf_ij)
      # Define interior nodes for i-th subdomain
      intr_i = sub_i.difference(intf_i)
      # Store i-th subdomain indices
      self.skeleton = self.skeleton.union(intf_i)
      self.interior.append(np.sort(np.array(list(intr_i))))
      self.interface.append(np.sort(np.array(list(intf_i))))
    self.skeleton = np.array(list(self.skeleton))

  # Ports nodes
  # -----------------------------------
  def set_ports_nodes(self):
    # Assign each interface node to subdomains
    # -------------
    node_intf_to_subs = np.zeros(
      shape=(len(self.skeleton), self.mesh.n_sub),
      dtype=bool
    )
    for (i, node_i) in enumerate(self.skeleton):
      for (j, intf_j) in enumerate(self.interface):
        node_intf_to_subs[i,j] = node_i in intf_j
    # Assign subdomains to each port
    # -------------
    subs = np.arange(self.mesh.n_sub)
    port_to_subs = set([])
    for mask in node_intf_to_subs:
      port_to_subs.add(frozenset(subs[mask]))
    # > Convert to dictionary
    self.port_to_subs = {}
    for (p, subs_p) in enumerate(list(port_to_subs)):
      self.port_to_subs[p] = np.array(list(subs_p))
    # > List all the ports
    self.ports = np.array(list(self.port_to_subs.keys()))
    # Assign nodes to each port
    # -------------
    self.port_to_nodes = {}
    for (p, subs_p) in self.port_to_subs.items():
      indices = np.zeros(self.mesh.n_sub, dtype=bool)
      indices[subs_p] = True
      mask = (node_intf_to_subs == indices).all(axis=1)
      self.port_to_nodes[p] = np.sort(self.skeleton[mask])

  def set_ports_maps(self):
    self._set_map_sub_to_ports()
    self._set_map_size_to_ports()
    self._set_map_orient_to_ports()
    self._set_map_orientsize_to_ports()

  def _set_map_sub_to_ports(self):
    """
    Assigns each port to multiple subdomains
    """
    self.sub_to_ports = {}
    for s in range(self.mesh.n_sub):
      sub = set([])
      for (p, subs_p) in self.port_to_subs.items():
        if (s in subs_p):
          sub.add(p)
      self.sub_to_ports[s] = np.sort(list(sub))

  def _set_map_size_to_ports(self):
    """
    Assigns each port to its size.
    """
    self.size_to_ports = {}
    for p in self.ports:
      size = self.port_to_nodes[self.ports[p]].size
      if (size not in self.size_to_ports):
        self.size_to_ports[size] = []
      self.size_to_ports[size].append(p)

  def _set_map_orient_to_ports(self):
    """
    Assigns each port to its orientation.
    """
    self.orient_to_ports = {"vert": [], "horiz": [], "inner": []}
    for p in self.ports:
      subs_p = self.port_to_subs[p]
      if (subs_p.size > 2):
        self.orient_to_ports["inner"].append(p)
      else:
        if (self.mesh.n_subs["y"] != 1) and (self.mesh.n_subs["x"] == 1):
          self.orient_to_ports["horiz"].append(p)
        elif ((self.mesh.n_subs["y"] == 1) and (self.mesh.n_subs["x"] != 1)):
          self.orient_to_ports["vert"].append(p)
        elif (np.abs(subs_p[1]-subs_p[0]) == 1):
          self.orient_to_ports["horiz"].append(p)
        else:
          self.orient_to_ports["vert"].append(p)

  def _set_map_orientsize_to_ports(self):
    """
    Assigns each port to its orientation and size.
    """
    # From orientation-size to ports
    self.orientsize_to_ports = {}
    for (orient, ports) in self.orient_to_ports.items():
      self.orientsize_to_ports[orient] = {}
      for p in ports:
        size = self.port_to_nodes[self.ports[p]].size
        if (size not in self.orientsize_to_ports[orient]):
          self.orientsize_to_ports[orient][size] = []
        self.orientsize_to_ports[orient][size].append(p)
    # From port to orientation-size
    self.port_to_orientsize = {}
    for p in self.ports:
      for (orient, size_to_ports) in self.orientsize_to_ports.items():
        for (size, ports) in size_to_ports.items():
          if (p in ports):
            self.port_to_orientsize[p] = (orient, size)
