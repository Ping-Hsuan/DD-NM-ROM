import numpy as np
import scipy.sparse as sp

from dd_nm_rom import ops
from dd_nm_rom import backend as bkd

from .state import SubdomainElementState


class Subdomain(object):
  """
  Class for generating a subdomain of the DD-FOM for the 2D steady-state Burgers" Equation with Dirichlet BC.

  inputs:
  self.monolithic: instance of Burgers2D class representing full domain problem
  nodes_ind["res"]: array of residual nodes_ind corresponding to the subdomain to be generated
  nodes_ind["interior"]: array of interior nodes_ind corresponding to the subdomain to be generated
  nodes_ind["interface"]: array of nodes_ind["interface"] nodes_ind corresponding to the subdomain to be generated
  cmat_intf: constraint matrix corresponding to the interface states of the subdomain
  ports: array containing which ports the subdomain belongs to

  methods:
  set_bc: update boundary condition data on subdomain
  res_jac: compute residual and its jacobian on the subdomain
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    monolithic,
    nodes_ind,
    cmat,
    ports,
    port_to_nodes,
    scaling=1.0
  ):
    self.monolithic = monolithic
    for k in ("ops_names",):
      setattr(self, k, getattr(self.monolithic, k))
    self.nodes_ind = nodes_ind
    self.cmat = cmat
    self.ports = ports
    self.port_to_nodes = port_to_nodes
    self.scaling = scaling
    # Set states
    self.elem_states = {e_k: SubdomainElementState(
      name=e_k,
      monolithic=self.monolithic,
      nodes_res=self.nodes_ind["res"],
      nodes_state=self.nodes_ind[e_k]
    ) for e_k in ("res", "interior", "interface")}

  # RHS/Jacobian
  # ===================================
  def rhs_jac(
    self,
    uv,
    lambdas,
    steady=True,
    dt=0.0,
    uv_old=None
  ):
    """
    Compute residual and its jacobians with respect to interior and interface states.

    inputs:
    u_interior: (n_interior,) vector of u interior states
    v_interior: (n_interior,) vector of v interior states
    u_interface: (n_interface,) vector of u interface states
    v_interface: (n_interface,) vector of  v interface states
    lambdas        : (n_constraints,) vector of lagrange multipliers

    outputs:
    rhs: (2*n_res,) residual vector with u and v residuals concatenated
    jac: (2*n_res, n_interior+n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
    H:   Hessian submatrix for SQP solver
    rhs: RHS block vector in SQP solver
    Ax : constraint matrix times interface state

    """
    # Assemble u and v on residual region
    uv = self.map_on_res(uv)
    if (not steady):
      uv_old = self.map_on_res(uv_old)
    # RHS and Jacobian
    rhs, jac = self.compute_rhs_jac(
      uv=uv,
      elem_states=self.elem_states,
      steady=steady,
      dt=dt,
      uv_old=uv_old
    )
    crhs, cjac = self.compute_crhs_cjac(uv=uv)
    # Return KKT system
    return self.assemble_kkt(
      rhs=rhs,
      crhs=crhs,
      lambdas=lambdas,
      jac=jac,
      cjac=cjac,
      scaling=self.scaling
    )

  def map_on_res(
    self,
    uv
  ):
    uv["res"] = {}
    for x_k in ("u", "v"):
      x_v = 0.0
      for e_k in ("interior", "interface"):
        x_v = x_v + self.elem_states[e_k].iden @ uv[e_k][x_k]
      uv["res"][x_k] = x_v
    return uv

  # RHS/Jacobian - PDE
  # -----------------------------------
  def compute_rhs_jac(
    self,
    uv,
    elem_states,
    steady=True,
    dt=0.0,
    uv_old=None,
    jac_fun=None
  ):
    # Precompute actions of operators
    ops_uv = self.action_ops(uv, elem_states)
    # RHS and Jacobian
    rhs = self.compute_rhs(uv, elem_states, ops_uv, steady, dt, uv_old)
    jac = self.compute_jac(uv, elem_states, ops_uv, steady, dt, jac_fun)
    return rhs, jac

  def action_ops(
    self,
    uv,
    elem_states
  ):
    ops_uv = {}
    for x_k in ("u", "v"):
      ops_uv[x_k] = {}
      for op_k in self.ops_names:
        op_v = 0.0
        for e_k in ("interior", "interface"):
          op_i = elem_states[e_k].ops[op_k]
          if isinstance(op_i, dict):
            op_i = op_i[x_k]
          op_v = op_v + op_i @ uv[e_k][x_k]
        ops_uv[x_k][op_k] = op_v
    return ops_uv

  def compute_rhs(
    self,
    uv,
    elem_states,
    ops_uv,
    steady=True,
    dt=0.0,
    uv_old=None
  ):
    # Compute
    rhs = self._compute_rhs(uv, elem_states, ops_uv)
    # Backward Euler for integration
    if (not steady):
      for x_k in ("u", "v"):
        x_kk = x_k if (x_k in uv["res"].keys()) else x_k+"_"+x_k
        rhs[x_k] = uv["res"][x_kk] - uv_old["res"][x_kk] - dt * rhs[x_k]
    # Return
    return np.concatenate([rhs[x_k] for x_k in ("u", "v")])

  def _compute_rhs(
    self,
    uv,
    elem_states,
    ops_uv
  ):
    bc_f = elem_states["res"].bc_f
    dx = {}
    for x_k in ("u", "v"):
      if (x_k not in uv["res"].keys()):
        u, v = [uv["res"][x_k+"_"+x_i] for x_i in ("u", "v")]
      else:
        u, v = [uv["res"][x_i] for x_i in ("u", "v")]
      adv_act_x = ops_uv[x_k]["Ax"] - bc_f[x_k]["A"]["x"]
      adv_act_y = ops_uv[x_k]["Ay"] - bc_f[x_k]["A"]["y"]
      dx[x_k] = ops.sp_diag(u) @ adv_act_x \
              + ops.sp_diag(v) @ adv_act_y \
              + ops_uv[x_k]["D"] + bc_f[x_k]["D"]
    return dx

  def compute_jac(
    self,
    uv,
    elem_states,
    ops_uv,
    steady=True,
    dt=0.0,
    jac_fun=None
  ):
    # Compute
    jac_fun = self._compute_jac if (jac_fun is None) else jac_fun
    jac = jac_fun(uv, elem_states, ops_uv)
    # Backward Euler for integration
    if (not steady):
      for e_k in ("interior", "interface"):
        jac[e_k] = elem_states[e_k].iden_uv - dt * jac[e_k]
    # Return
    return jac

  def _compute_jac(
    self,
    uv,
    elem_states,
    ops_uv
  ):
    jac = {}
    bc_f = elem_states["res"].bc_f
    jac_uu = ops.sp_diag(ops_uv["u"]["Ax"] - bc_f["u"]["A"]["x"])
    jac_uv = ops.sp_diag(ops_uv["u"]["Ay"] - bc_f["u"]["A"]["y"])
    jac_vu = ops.sp_diag(ops_uv["v"]["Ax"] - bc_f["v"]["A"]["x"])
    jac_vv = ops.sp_diag(ops_uv["v"]["Ay"] - bc_f["v"]["A"]["y"])
    uv_diag = ops.map_nested_dict(uv["res"], ops.sp_diag)
    for e_k in ("interior", "interface"):
      state_k = elem_states[e_k]
      jac_xx_k = uv_diag["u"] @ state_k.ops["Ax"] \
               + uv_diag["v"] @ state_k.ops["Ay"] \
               + state_k.ops["D"]
      jac_uu_k = jac_uu @ state_k.iden + jac_xx_k
      jac_uv_k = jac_uv @ state_k.iden
      jac_vu_k = jac_vu @ state_k.iden
      jac_vv_k = jac_vv @ state_k.iden + jac_xx_k
      jac[e_k] = sp.bmat(
        [[jac_uu_k, jac_uv_k],
         [jac_vu_k, jac_vv_k]],
        format="csr"
      )
    return jac

  # RHS/Jacobian - Constraint
  # -----------------------------------
  def compute_crhs_cjac(
    self,
    uv
  ):
    cx = np.concatenate([uv["interface"][x_k] for x_k in ("u", "v")])
    crhs = self.cmat["interface"] @ cx
    return crhs, self.cmat

  # KKT system
  # -----------------------------------
  def assemble_kkt(
    self,
    rhs,
    crhs,
    lambdas,
    jac,
    cjac,
    scaling
  ):
    # To sparse
    jac = ops.map_nested_dict(jac, bkd.to_sparse)
    cjac = ops.map_nested_dict(cjac, bkd.to_sparse)
    # RHS
    rhs = np.concatenate([
      scaling*(jac["interior"].T@rhs),
      scaling*(jac["interface"].T@rhs) + cjac["interface"].T@lambdas
    ])
    # Constraints
    cjac = sp.hstack([cjac["interior"], cjac["interface"]])
    # Hessian
    jac = sp.hstack([jac["interior"], jac["interface"]])
    hess = scaling*(jac.T@jac)
    return rhs, crhs, hess, cjac
