import numpy as np
import scipy.sparse as sp

from dd_nm_rom import ops


class SubdomainElementState(object):
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
    name,
    monolithic,
    nodes_res,
    nodes_state
  ):
    self.name = name
    self.monolithic = monolithic
    self.nodes_res = nodes_res
    self.nodes_state = nodes_state
    self.n_nodes_res = self.nodes_res.size
    self.n_nodes_state = self.nodes_state.size
    self.submat = np.ix_(self.nodes_res, self.nodes_state)
    # Intersection between state and residual nodes
    self.intersect = {
      "state_res": np.nonzero(np.isin(self.nodes_state, self.nodes_res))[0],
      "res_state": np.nonzero(np.isin(self.nodes_res, self.nodes_state))[0]
    }
    # Operators
    self.set_ops_bc()

  # Operators
  # ===================================
  def set_ops_bc(self):
    # Inclusion operators
    self.iden = self.monolithic.iden[self.submat]
    self.iden_uv = sp.block_diag([self.iden, self.iden])
    # Differential operators
    self.ops = {}
    for (op_k, op_v) in self.monolithic.ops.items():
      self.ops[op_k] = op_v[self.submat]
    # Boundary conditions
    self.bc_f = ops.map_nested_dict(
      self.monolithic.bc_f, lambda x: x[self.nodes_res]
    )
