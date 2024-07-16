import copy
import torch
import numpy as np
import scipy.sparse as sp

from time import time
from dd_nm_rom import ops
from dd_nm_rom import solvers
from dd_nm_rom import backend as bkd

from .rbf_model import RBFModel
from .subdomain import SubdomainROM
from ..autoencoder import AutoencoderNP, MultiAutoencoderNP


class DD_NM_ROM(object):
  '''
  Compute DD NM-ROM for the 2D steady-state Burgers' equation with Dirichlet BC.

  inputs:
  dd_fom: DD model class corresponding to full order DD model.
  intr_net_list: list of paths to trained networks for interior states
  intf_net_list: list of paths to trained networks for interface states
  port_net_list:  [optional] list of paths to trained networks for port states. Required for constraint_type=='strong'
  residual_bases: [optional] list of residual bases where residual_bases[i] is the residual basis for the ith subdomain
  hr: [optional] Boolean to specify if hyper reduction is applied. Default is False
  hr_type: [optional] Hyper-reduction type. Either 'gappy_POD' or 'collocation'.
        Only takes effect if hr=True. Default is 'collocation'
  sample_ratio: [optional] ratio of number of hyper-reduction samples to residual basis size. Default is 2
  n_samples: [optional] specify number of hyper reduction sample nodes.
        If n_samples is an array with length equal to the number of subdomains, then
        n_samples[i] is the number of HR samples on the ith subdomain.

        If n_samples is a positive integer, then each subdomain has n_samples HR nodes.

        Otherwise, the number of samples is determined by the sample ratio.
        Default is -1.

  n_corners: [optional] Number of interface nodes included in the HR sample nodes.
        If n_corners is an array with length equal to the number of subdomains, then
        n_corners[i] is the number of interface HR nodes on the ith subdomain.

        If n_corners is a positive integer, then each subdomain has n_corners interface HR nodes.

        Otherwise, the number of interface HR nodes on each subdomain is determined by n_samples
        multiplied by the ratio of the number of interface nodes contained in the residual nodes
        to the total number of residual nodes.

        Default is -1.

  n_constraints: [optional] number of weak constraints for NLP. Default is 1
  constraint_type: [optional] 'weak' or 'strong' port constraints. Default is 'weak'.

  fields:
  subdomain: list of subdomain_LS_ROM or subdomain_LS_ROM_HR classes corresponding to each subdomain, i.e.
         subdomain[i] = reduced subdomain class corresponding to subdomain [i]

  methods:
  FJac: computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver.
  solve: solves for the reduced states of the DD NM-ROM using the Lagrange-Newton-SQP method.
  '''

  def __init__(
    self,
    dd_fom,
    nn_configfiles,
    res_bases=None,
    hr_active=False,
    hr_n_samples=1,
    hr_n_edge_samples_ratio=0.75,
    hr_sample_small_ports=False,
    hr_small_ports_dim=5,
    constraint_type='weak',
    n_constraints_weak=1,
    scaling=1.0
  ):
    # DD-FOM
    # -------------
    self.dd_fom = dd_fom
    for k in ("runtime", "mesh"):
      setattr(self, k, getattr(self.dd_fom, k))
    # Scaling factor for residual
    self.scaling = self.mesh.hxy if (scaling <= 0) else scaling
    # Autoencoders
    # -------------
    self.nn_configs = ops.map_nested_dict(nn_configfiles, torch.load)
    self.nn_models = self.init_nn_models(self.nn_configs)
    # DD-ROM Constraints
    # -------------
    self.constraint_type = constraint_type
    if (self.constraint_type not in ("weak", "strong")):
      raise ValueError(
        f"Could not interpret constraint type: '{self.constraint_type}'. " \
          "Valid options are: ['weak', 'strong']."
      )
    self.n_constraints_weak = n_constraints_weak
    self.compute_rom_dim()
    if (self.constraint_type == "strong"):
      self.set_port_indices()
      self.init_nn_model_intf()
    self.assemble_cmat()
    # HR
    # -------------
    self.res_bases = res_bases
    self.hr_active = hr_active
    self.hr_n_samples = int(hr_n_samples)
    self.hr_n_edge_samples_ratio = np.clip(hr_n_edge_samples_ratio, 0, 1)
    self.hr_sample_small_ports = float(hr_sample_small_ports)
    self.hr_small_ports_dim = int(hr_small_ports_dim)
    # DD-ROM subdomains
    # -------------
    self.subdomains = []
    for (s, sub) in enumerate(self.dd_fom.subdomains):
      inputs_s = {}
      for input_k in ("cmat", "rom_dim", "nn_models"):
        attr_k = getattr(self, input_k)
        inputs_s[input_k] = {
          e_k: attr_k[e_k][s] for e_k in ("interior", "interface")
        }
      self.subdomains.append(
        SubdomainROM(
          sub_fom=sub,
          scaling=self.scaling,
          constraint_type=self.constraint_type,
          res_bases=self.res_bases[s] if (self.res_bases is not None) else None,
          hr_n_samples=self.hr_n_samples,
          hr_n_edge_samples_ratio=self.hr_n_edge_samples_ratio,
          hr_sample_small_ports=self.hr_sample_small_ports,
          hr_small_ports_dim=self.hr_small_ports_dim,
          **inputs_s
        )
      )
      if self.hr_active:
        self.subdomains[-1].set_hr_mode(active=True)
    # Interpolator
    # -------------
    self.rbf_model = RBFModel(self.subdomains, self.n_constraints)
    # Integration
    # -------------
    self.steady = True
    self.x_old = None
    self.dt = 0.0

  def get_ndof(self):
    ndof = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        ndof += sub.rom_dim[e_k]
    ndof += self.n_constraints
    return ndof

  # ROM dimensions
  # ===================================
  def compute_rom_dim(self):
    # Read latent dimension of each autoencoder
    self.rom_dim = {}
    for (e_k, cfg_k) in self.nn_configs.items():
      if (e_k not in self.rom_dim):
        self.rom_dim[e_k] = []
      for cfg_ki in cfg_k:
        dim = cfg_ki["decoder"]["latent_dim"]
        self.rom_dim[e_k].append(dim)
    # Compute interface latent dimension form ports ones
    if (self.constraint_type == 'strong'):
      self.rom_dim["interface"] = []
      for sub in self.dd_fom.subdomains:
        dim = 0
        for p in sub.ports:
          dim += self.rom_dim["port"][p]
        self.rom_dim["interface"].append(dim)

  # Port to nodes indices
  # ===================================
  def set_port_indices(self):
    # Check number of port autoencoders
    if (len(self.nn_configs["port"]) != len(self.dd_fom.dd_indices.ports)):
      raise ValueError(
        "The number of port autoencoders doesn't " \
          "match the number of ports available."
      )
    # Set port nodes indices
    self.port_to_nodes = []
    for sub in self.dd_fom.subdomains:
      port_to_nodes_s = {}
      shift = 0
      for p in sub.ports:
        # FOM
        # > Port/interface indices
        port_ind = self.dd_fom.dd_indices.port_to_nodes[p]
        intf_ind = sub.elem_states["interface"].nodes_state
        # > Duplicate for u and v
        port_ind = np.concatenate([port_ind, port_ind+self.mesh.nxy])
        intf_ind = np.concatenate([intf_ind, intf_ind+self.mesh.nxy])
        fom_ind = np.nonzero(np.isin(intf_ind, port_ind))[0]
        # ROM
        port_dim = self.rom_dim["port"][p]
        rom_ind = np.arange(port_dim)+shift
        # Update
        port_to_nodes_s[p] = {"fom": fom_ind, "rom": rom_ind}
        shift += port_dim
      self.port_to_nodes.append(port_to_nodes_s)

  # Autoencoders
  # ===================================
  def init_nn_models(
    self,
    configs
  ):
    # Loop over elements: interior and interface/ports
    nn_models = {}
    for (key_i, cfg_i) in configs.items():
      # Loop over instances in each element
      if (not isinstance(cfg_i, (list, tuple))):
        cfg_i = [cfg_i]
      nn_models[key_i] = [AutoencoderNP(cfg_ij) for cfg_ij in cfg_i]
    return nn_models

  # Interface from ports
  # -----------------------------------
  def init_nn_model_intf(self):
    models = []
    for (s, sub) in enumerate(self.dd_fom.subdomains):
      n_nodes_intf = sub.elem_states["interface"].n_nodes_state
      models.append(
        MultiAutoencoderNP(
          indices=self.port_to_nodes[s],
          input_dim=2*n_nodes_intf,
          autoencoders={p: self.nn_models["port"][p] for p in sub.ports}
        )
      )
    self.nn_models["interface"] = models

  # Constraint matrices
  # ===================================
  def assemble_cmat(self):
    if (self.constraint_type == "weak"):
      cmat = copy.deepcopy(self.dd_fom.cmat)
      if (self.dd_fom.constraint_type != "weak"):
        cmat, self.n_constraints = self.dd_fom.assemble_cmat_weak(
          cmat=cmat,
          n_constraints_weak=self.n_constraints_weak,
          n_constraints=self.dd_fom.n_constraints
        )
    else:
      cmat = self._assemble_cmat_strong()
    self.cmat = ops.map_nested_dict(cmat, bkd.to_sparse)

  def _assemble_cmat_strong(self):
    # Compute total number of constraints
    self.n_constraints = 0
    for (p, subs_p) in self.dd_fom.dd_indices.port_to_subs.items():
      port_dim = self.rom_dim["port"][p]
      self.n_constraints += (len(subs_p)-1) * port_dim
    # Assemble constraints matrices
    cmat = {
      "interior": self._init_cmat(element="interior"),
      "interface": self._assemble_cmat_intf()
    }
    return cmat

  def _init_cmat(
    self,
    element
  ):
    cmat = []
    for dim in self.rom_dim[element]:
      cmat.append(sp.coo_matrix((self.n_constraints, dim)))
    return cmat

  def _assemble_cmat_intf(self):
    # Initialize matrices
    cmat = self._init_cmat(element="interface")
    # Fill matrices
    shift = 0
    for (p, subs_p) in self.dd_fom.dd_indices.port_to_subs.items():
      port_dim = self.rom_dim["port"][p]
      for i in range(len(subs_p)-1):
        for (j, l) in enumerate((i,i+1)):
          col = self.port_to_nodes[subs_p[l]][p]["rom"]
          row = np.arange(port_dim) + shift
          data = (-1)**j * np.ones(port_dim)
          cmat[subs_p[l]].col  = np.concatenate((cmat[subs_p[l]].col,  col))
          cmat[subs_p[l]].row  = np.concatenate((cmat[subs_p[l]].row,  row))
          cmat[subs_p[l]].data = np.concatenate((cmat[subs_p[l]].data, data))
        shift += port_dim
    return cmat

  # RHS/Jacobian
  # ===================================
  def rhs_jac(
    self,
    x
  ):
    '''
    Computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver.

    inputs:
    w: vector of all interior and interface states for each subdomain
       and the lagrange multipliers lam in the order
      [intr[0], intf[0], ..., intr[n_subs], intf[n_subs], lam]

    outputs:
    val: RHS of the KKT system
    full_jac: KKT matrix
    runtime: "parallel" runtime to assemble KKT system
    '''
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    rhs, hess, cjac = [], [], []
    crhs = np.zeros(self.n_constraints)
    # > Set Lagrangian multipliers
    lambdas = x[-self.n_constraints:]
    runtime += time()-start
    # Loop over subdomains
    # -------------
    z = self.extract_z_sub_from_vec(x)
    z_old = None
    if (not self.steady):
      z_old = self.extract_z_sub_from_vec(self.x_old)
    runtime_s = 0.0
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      # > Compute quantities needed for KKT system
      rhs_s, crhs_s, hess_s, cjac_s = sub.rhs_jac(
        z=z[s],
        lambdas=lambdas,
        steady=self.steady,
        dt=self.dt,
        z_old=z_old[s] if (z_old is not None) else None
      )
      runtime_s = max(time()-start_s, runtime_s)
      # > Store subdomain-related quantities
      start = time()
      rhs.append(rhs_s)
      crhs += crhs_s
      cjac.append(cjac_s)
      hess.append(hess_s)
      runtime += time()-start
    runtime += runtime_s
    # Assemble
    # -------------
    start = time()
    rhs, jac = self.dd_fom.assemble_kkt(rhs, crhs, cjac, hess)
    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["rhs_jac"] += runtime
    return rhs, jac

  def extract_z_sub_from_vec(
    self,
    x
  ):
    z = []
    si = 0
    runtime_s = 0.0
    for sub in self.subdomains:
      start_s = time()
      z_s = {}
      for e_k in ("interior", "interface"):
        ei = si + sub.rom_dim[e_k]
        z_s[e_k] = x[si:ei]
        si = ei
      z.append(z_s)
      runtime_s = max(time()-start_s, runtime_s)
    self.runtime["total"] += runtime_s
    self.runtime["rhs_jac"] += runtime_s
    return z

  # Encode/Decode
  # ===================================
  def encode(
    self,
    x
  ):
    is_2d = (x.ndim == 2)
    if (is_2d and (x.shape[0] == 2*self.mesh.nxy)):
      x = x.T
    if is_2d:
      return np.vstack([self._encode(xi) for xi in x]).T
    else:
      return self._encode(x)

  def _encode(
    self,
    x
  ):
    # Map on elements
    uv = self.dd_fom.map_sol_on_elements(x.reshape(1,-1))
    # Loop over subdomains/elements
    z = []
    for (s, sub) in enumerate(self.subdomains):
      for e_k in ("interior", "interface"):
        xi = uv[e_k][s].reshape(-1)
        zi = sub.elem_states[e_k].encode(xi, with_jac=False)
        z.append(zi)
    z.append(np.zeros(self.n_constraints))
    return np.concatenate(z)

  def decode(
    self,
    x,
    map_on_res=False
  ):
    is_2d = (x.ndim == 2)
    shape = [self.mesh.nxy, x.shape[1]] if is_2d else [self.mesh.nxy]
    # Initialize solution containers
    uv = self.dd_fom.init_uv(shape=shape, map_on_res=map_on_res)
    z = {k: [] for k in ("interior", "interface")}
    # Loop over subdomains/elements
    si = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        # Extract latent space subvector
        ei = si + sub.rom_dim[e_k]
        xi = x[si:ei]
        si = ei
        # Set element state
        state_k = sub.elem_states[e_k]
        state_k.set_decoder_hr(active=False)
        # Store latent space
        z[e_k].append(xi)
        # Reconstruct/store physical space
        if is_2d:
          uv_i = [state_k.decode(xj, with_jac=False) for xj in xi.T]
          uv_i = np.vstack(uv_i).T
        else:
          uv_i = state_k.decode(xi, with_jac=False)
        uv = self.dd_fom.extract_uv_sub_from_vec(
          uv=uv,
          uv_i=uv_i,
          elem_state=sub.sub_fom.elem_states[e_k],
          map_on_res=map_on_res
        )
    lambdas = x[-self.n_constraints:]
    return uv, z, lambdas

  def reconstruct_static(
    self,
    x
  ):
    return self.decode(self.encode(x), map_on_res=True)[0]

  # Solution
  # ===================================
  def solve(
    self,
    x0=None,
    mu=None,
    dt=0.0,
    nt=1,
    steady=True,
    guess=None,
    use_guess=False,
    runtime=0.0,
    tol=1e-8,
    maxit=50,
    stepsize_min=1e-10,
    verbose=False
  ):
    """
    Solves for the u and v states of the FOM using Newton"s method.

    inputs:
    u0: (nx*ny,) initial u vector
    v0: (nx*ny,) initial v vector
    tol: [optional] stopping tolerance for Newton solver. Default is 1e-10
    maxit: [optional] max number of iterations for newton solver. Default is 100
    print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False

    outputs:
    u: (nx*ny,) u final solution vector
    v: (nx*ny,) v final solution vector
    res_vecs: (it, nx*ny) array where res_vecs[i] is the PDE residual evaluated at the ith Newton iteration
    """
    self.runtime = ops.map_nested_dict(self.runtime, lambda _: 0.0)
    self.runtime["total"] += runtime
    # Initialize solution
    if (mu is not None):
      x0, runtime = self.rbf_model(mu)
      self.runtime["total"] += runtime
    # Initialize solver
    solver = solvers.Newton(
      model=self,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, rhs, *_, flag = solver(x0, dt, nt, guess, use_guess)
    converged = True if (flag[-1] == 0) else False
    # Assemble solution
    uv, z, lambdas = self.assemble_sol(x, map_on_res=True)
    return uv, z, lambdas, rhs, converged

  def get_init_sol(
    self,
    x
  ):
    return self.encode(x)

  def assemble_sol(
    self,
    x,
    map_on_res=False
  ):
    return self.decode(x, map_on_res)

  def compute_error(
    self,
    uv_fom,
    uv_rom,
    scaling=False,
    relative=True,
    axis=None
  ):
    '''
    Compute error between DD-ROM and DD-FOM DD solutions.

    inputs:
    w_intr: list of reduced interior states where
           w_intr[i] = (self.subdomains[i].rom_dim["interior"],) vector of ROM solution interior states
    w_intf: list of reduced interface states where
           w_intf[i] = (self.subdomains[i].rom_dim["interface"],) vector of ROM solution interface states

    u_intr: list of FOM u interior states where
           u_intr[i] = (dd_fom.subdomains[i].n_nodes["interior"],) vector of FOM u solution interior states
    v_intr: list of FOM v interior states where
           v_intr[i] = (dd_fom.subdomains[i].n_nodes["interior"],) vector of FOM v solution interior states
    u_intf: list of FOM u interface states where
           u_intf[i] = (dd_fom.subdomains[i].n_interface,) vector of FOM u solution interface states
    v_intf: list of FOM v interface states where
           v_intf[i] = (dd_fom.subdomains[i].n_interface,) vector of FOM v solution interface states

    output:
    error: square root of mean squared relative error on each subdomain
    '''
    err = 0.0
    for s in range(self.mesh.n_sub):
      num_s, den_s = 0.0, 0.0
      for e_k in ("interior", "interface"):
        for x_k in ("u", "v"):
          x_fom = uv_fom[e_k][x_k][s]
          x_rom = uv_rom[e_k][x_k][s]
          num_s += np.sum(np.square(x_rom - x_fom), axis=axis)
          if relative:
            den_s += np.sum(np.square(x_fom), axis=axis)
      err += num_s/den_s if relative else num_s
    if scaling:
      err *= self.mesh.hxy
    return np.amax(np.sqrt(err/self.mesh.n_sub))
