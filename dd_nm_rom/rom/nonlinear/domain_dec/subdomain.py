import copy
import numpy as np
import scipy.sparse as sp

from dd_nm_rom import ops
from dd_nm_rom.rom.utils import hyper_red as hr_mod

from .state import SubdomainElementStateROM


class SubdomainROM(object):
  '''
  Class for generating a non-hyper-reduced subdomain of the DD NM-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.

  inputs:
  subdomain: subdomain class of full-order DD autoencoder
  interior_dict: dictionary for interior states with fields
        decoder:    state_dict with decoder parameters
        latent_dim: latent dimension for decoder
        scale:      scaling vector for normalizing data
        ref:        reference vector for shifting data
  interface_dict: dictionary for interface states with fields
        decoder:    state_dict with decoder parameters
        latent_dim: latent dimension for decoder
        scale:      scaling vector for normalizing data
        ref:        reference vector for shifting data
  cmat: constraint matrix

  methods:
  en_interior:  encoder function for interior state
  en_interface:  encoder function for interface state
  de_interior : decoder function for interior state
  de_interface : decoder function for interface state
  res_jac: compute residual its jacobian on the subdomain
  '''

  def __init__(
    self,
    rom_dim,
    sub_fom,
    nn_models,
    scaling=1.0,
    cmat=None,
    constraint_type="strong",
    res_bases=None,
    hr_n_samples=1,
    hr_n_edge_samples_ratio=0.75,
    hr_sample_small_ports=False,
    hr_small_ports_dim=5
  ):
    # ROM dimensions
    # -------------
    self.rom_dim = rom_dim
    # FOM subdomain
    # -------------
    self.sub_fom = sub_fom
    for k in ("ports", "port_to_nodes"):
      setattr(self, k, getattr(self.sub_fom, k))
    # Operators/BC
    # -------------
    self.scaling = float(scaling)
    self.cmat = cmat
    self.constraint_type = constraint_type
    # Compose null matrix for interior
    for k in ("interior",):
      self.cmat[k] = self.cmat[k][:,:self.rom_dim[k]]
    # Hyper-reduction (HR)
    # -------------
    self.res_bases = res_bases
    self.hr_n_samples = hr_n_samples
    self.hr_n_edge_samples_ratio = hr_n_edge_samples_ratio
    self.hr_sample_small_ports = hr_sample_small_ports
    self.hr_small_ports_dim = hr_small_ports_dim
    # Control variables
    self.hr_init = False
    self.hr_active = False
    # Element states
    # -------------
    self.elem_states = {e_k: SubdomainElementStateROM(
      state_fom=self.sub_fom.elem_states[e_k],
      nn_model=nn_models[e_k] if (e_k != "res") else None
    ) for e_k in ("res", "interior", "interface")}
    # Set HR mode
    # -------------
    self.set_hr_mode(active=False)

  # Hyper-reduction (HR)
  # ===================================
  def set_hr_mode(self, active=False):
    self.hr_active = active
    if self.hr_active:
      if (not self.hr_init):
        self.init_hr_mode()
      self.map_on_res = self._map_on_res_hr
      self.compute_jac = self._compute_jac_hr
    else:
      self.map_on_res = self._map_on_res
      self.compute_jac = None
    for state_k in self.elem_states.values():
      state_k.set_hr_mode(active=self.hr_active)

  def init_hr_mode(self):
    self.set_hr_dim()
    hr_nodes_res = self.sample_hr_nodes_res()
    for state_k in self.elem_states.values():
      state_k.init_hr_mode(hr_nodes_res)
    self.hr_init = True

  def set_decoder_hr(self, active):
    for state_k in self.elem_states.values():
      state_k.set_decoder_hr(active=active)

  # Number of samples
  # -----------------------------------
  def set_hr_dim(self):
    # Check hyper reduction inputs
    if (self.res_bases is None):
      raise ValueError("Please, provide residual bases for HR.")
    # Compute parameters for hyper reduction
    n_samples_max, self.n_res_bases = self.res_bases.shape
    self.hr_n_samples = np.clip(
      self.hr_n_samples, self.n_res_bases, n_samples_max
    )
    # Set number of nodes on residual edges
    n_nodes_intr = self.sub_fom.elem_states["interior"].n_nodes_state
    n_edges_max = n_samples_max - 2*n_nodes_intr
    self.hr_n_edge_samples = self.hr_n_edge_samples_ratio * self.hr_n_samples
    self.hr_n_edge_samples = min(int(self.hr_n_edge_samples), n_edges_max)

  # Indices
  # -----------------------------------
  def sample_hr_nodes_res(self):
    '''
    Greedy algorithm to select sample nodes for hyper reduction.

    outputs:
      samples: array of sample nodes
    '''
    samples = np.array([], dtype=np.int32)
    nodes_res = self.sub_fom.elem_states["res"].nodes_state
    n_nodes_res = self.sub_fom.elem_states["res"].n_nodes_state
    # Include nodes from small ports
    if self.hr_sample_small_ports:
      for p in self.ports:
        nodes = self.port_to_nodes[p]
        if (nodes.size <= self.hr_small_ports_dim):
          inter = np.nonzero(np.isin(nodes_res, nodes))[0]
          inter = np.concatenate([inter, inter + n_nodes_res])
          samples = np.union1d(samples, inter)
    # Greedily sample nodes on residual edges and interior regions
    for k in ("interface", "interior"):
      if (k == "interior"):
        max_samples = self.hr_n_samples
      else:
        max_samples = self.hr_n_edge_samples
      nodes_k = self.sub_fom.elem_states[k].intersect["res_state"]
      nodes_k = np.concatenate([nodes_k, nodes_k + n_nodes_res])
      nodes_k = np.setdiff1d(nodes_k, samples)
      indices = hr_mod.select_sample_nodes(
        bases=self.res_bases[nodes_k],
        n_samples=max_samples-len(samples)
      )
      samples = np.union1d(samples, nodes_k[indices])
    return samples

  # RHS/Jacobian
  # ===================================
  def rhs_jac(
    self,
    z,
    lambdas,
    steady=True,
    dt=0.0,
    z_old=None
  ):
    """
    Compute residual and its jacobian on subdomain.

    inputs:
    w_interior: (n_interior,) vector - reduced interior state
    w_interface: (n_interface,) vector - reduced interface state
    lam:    (n_constraints,) vector of lagrange multipliers

    outputs:
    res: (nz,) vector - residual on subdomain
    jac: (nz, n_interior + n_interface) array - jacobian of residual w.r.t. (w_interior, w_interface)
    H:   Hessian submatrix for SQP solver
    rhs: RHS block vector in SQP solver
    Ag : constraint matrix times output of interface decoder
    Adg: constraint matrix times jacobian of interface decoder
    """
    # Reconstruct u and v
    # -------------
    uv, dec_jac = self.reconstruct_uv(z, with_jac=True, map_on_res=True)
    uv_old = None
    if (not steady):
      uv_old = self.reconstruct_uv(z_old, with_jac=False, map_on_res=True)
    # Compute RHS/Jacobian
    # -------------
    rhs, jac = self.sub_fom.compute_rhs_jac(
      uv=uv,
      elem_states=self.elem_states,
      steady=steady,
      dt=dt,
      uv_old=uv_old,
      jac_fun=self.compute_jac
    )
    for e_k in ("interior", "interface"):
      jac[e_k] = jac[e_k] @ dec_jac[e_k]
    # Compute constraints RHS/Jacobian
    # -------------
    crhs, cjac = self.compute_crhs_cjac(
      z=z,
      uv=uv,
      dec_jac=dec_jac
    )
    # Assemble KKT
    # -------------
    return self.sub_fom.assemble_kkt(
      rhs=rhs,
      crhs=crhs,
      lambdas=lambdas,
      jac=jac,
      cjac=cjac,
      scaling=self.scaling
    )

  def reconstruct_uv(
    self,
    z,
    with_jac=True,
    map_on_res=True
  ):
    uv, dec_jac = {}, {}
    for e_k in ("interior", "interface"):
      state_k = self.elem_states[e_k]
      # Reconstruct u and v on element
      if with_jac:
        uv_k, dec_jac[e_k] = state_k.decode(z[e_k], with_jac=True)
      else:
        uv_k = state_k.decode(z[e_k], with_jac=False)
      uv[e_k] = {
        "u": uv_k[:state_k.size["u"]],
        "v": uv_k[state_k.size["u"]:]
      }
    if map_on_res:
      # Reconstruct u and v on residual
      uv = self.map_on_res(uv)
    if with_jac:
      return uv, dec_jac
    else:
      return uv

  def compute_crhs_cjac(
    self,
    z,
    uv,
    dec_jac
  ):
    celem = "interface"
    cjac = copy.deepcopy(self.cmat)
    cstate = self.elem_states[celem]
    if (self.constraint_type == "weak"):
      if (self.hr_active):
        cstate.set_decoder_hr(active=False)
        cx, cdec_jac = cstate.decode(z[celem], with_jac=True)
        cstate.set_decoder_hr(active=True)
      else:
        cx = np.concatenate([uv[celem][x_k] for x_k in ("u", "v")])
        cdec_jac = dec_jac[celem]
      crhs = self.cmat[celem] @ cx
      cjac[celem] = cjac[celem] @ cdec_jac
    else:
      crhs = self.cmat[celem] @ z[celem]
    return crhs, cjac

  # HR not active
  # -----------------------------------
  def _map_on_res(self, uv):
    uv["res"] = {}
    for e_k in ("interior", "interface"):
      state_k = self.elem_states[e_k]
      for x_k in ("u", "v"):
        if (x_k not in uv["res"]):
          uv["res"][x_k] = 0.0
        uv["res"][x_k] = uv["res"][x_k] + state_k.iden @ uv[e_k][x_k]
    return uv

  # HR active
  # -----------------------------------
  def _map_on_res_hr(self, uv):
    uv["res"] = {}
    for e_k in ("interior", "interface"):
      state_k = self.elem_states[e_k]
      for x_i in ("u", "v"):
        for x_j in ("u", "v"):
          k = x_i+"_"+x_j
          if (k not in uv["res"]):
            uv["res"][k] = 0.0
          uv["res"][k] = uv["res"][k] + state_k.iden[k] @ uv[e_k][x_j]
    return uv

  def _compute_jac_hr(
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
      jac_xx_k = {}
      for x_k in ("u", "v"):
        jac_xx_k[x_k] = uv_diag[x_k+"_u"] @ state_k.ops["Ax"][x_k] \
                      + uv_diag[x_k+"_v"] @ state_k.ops["Ay"][x_k] \
                      + state_k.ops["D"][x_k]
      jac_uu_k = jac_uu @ state_k.iden["u_u"] + jac_xx_k["u"]
      jac_uv_k = jac_uv @ state_k.iden["u_v"]
      jac_vu_k = jac_vu @ state_k.iden["v_u"]
      jac_vv_k = jac_vv @ state_k.iden["v_v"] + jac_xx_k["v"]
      jac[e_k] = sp.bmat(
        [[jac_uu_k, jac_uv_k],
         [jac_vu_k, jac_vv_k]],
        format="csr"
      )
    return jac
