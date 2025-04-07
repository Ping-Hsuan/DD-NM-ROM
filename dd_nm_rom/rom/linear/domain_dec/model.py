import copy
import torch
import numpy as np
import scipy.sparse as sp
import numpy_indexed as npi
import dill as pickle

from time import time
from dd_nm_rom import ops
from dd_nm_rom import solvers
from dd_nm_rom import backend as bkd

from dd_nm_rom.rom.nonlinear.domain_dec.rbf_model import RBFModel
from .subdomain import SubdomainROM
from dd_nm_rom.rom.linear.nn import Autoencoder as AutoencoderNP
from dd_nm_rom.rom.linear.nn import MultiAutoencoder as MultiAutoencoderNP


class DD_LS_ROM(object):
  """
  Compute DD LS-ROM for the 2D steady-state Burgers' equation with Dirichlet BC.
  """

  def __init__(
    self,
    dd_fom,
    ls_configfiles,
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
    tmp_dict = ops.map_nested_dict(ls_configfiles, load_pickle_file)
    self.ls_configs = create_dict(tmp_dict)
    print("Keys of self.ls_configs:", self.ls_configs.keys())
    print("Keys and values in self.nn_configs['interior']:")
    print(self.ls_configs['interior'])
#   for value in self.ls_configs['interior']:
#     print(f"Value: {value}")
#   print(self.ls_configs['interior'][0]['interior'][0]['u'].shape)
#   print(self.ls_configs['interior'][0]['interior'][0]['s'].shape)
    # Load NN for interior and port -> Load Basis in LS-ROM case
    self.ls_models = self.init_ls_models(self.ls_configs)
    print("Keys of self.ls_models:", self.ls_models.keys())
    print("items of self.ls_models:", self.ls_models['interior'])
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
      self.init_ls_model_intf()
    1/o
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
          res_bases=self.get_res_bases(s),
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

  def get_res_bases(self, index=0):
    if (self.res_bases is not None):
      if isinstance(self.res_bases, (list, tuple)):
        if (len(self.res_bases) != self.mesh.n_sub):
          raise ValueError(
            "The number of residual bases matrices provided " \
            "does not match the number of subdomains."
          )
        else:
          return self.res_bases[index]
      elif isinstance(self.res_bases, np.ndarray):
        return copy.deepcopy(self.res_bases)
      else:
        raise ValueError(
          "The 'res_bases' input must be either a list or a NumPy array."
        )

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
    for (e_k, cfg_k) in self.ls_configs.items():
      if (e_k not in self.rom_dim):
        self.rom_dim[e_k] = []
      for cfg_ki in cfg_k:
#       dim = cfg_ki[e_k][0]['u'].shape[1]
#       self.rom_dim[e_k].append(dim)
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
    if (len(self.ls_configs["port"]) != len(self.dd_fom.dd_indices.ports)):
      raise ValueError(
        "The number of port linear subspaces doesn't " \
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
        fom_ind = npi.indices(intf_ind, port_ind, missing="mask").compressed()
        # ROM
        port_dim = self.rom_dim["port"][p]
        rom_ind = np.arange(port_dim)+shift
        # Update
        port_to_nodes_s[p] = {"fom": fom_ind, "rom": rom_ind}
        shift += port_dim
      self.port_to_nodes.append(port_to_nodes_s)

  # Autoencoders
  # ===================================
  def init_ls_models(
    self,
    configs
  ):
    # Loop over elements: interior and interface/ports
    ls_models = {}
    for (key_i, cfg_i) in configs.items():
      # Loop over instances in each element
      if (not isinstance(cfg_i, (list, tuple))):
        cfg_i = [cfg_i]
#     ls_models[key_i] = [AutoencoderNP(cfg_ij[key_i][0]) for cfg_ij in cfg_i]
      ls_models[key_i] = [AutoencoderNP(cfg_ij) for cfg_ij in cfg_i]
    return ls_models

  # Interface from ports
  # -----------------------------------
  def init_ls_model_intf(self):
    models = []
    for (s, sub) in enumerate(self.dd_fom.subdomains):
      n_nodes_intf = sub.elem_states["interface"].n_nodes_state
      models.append(
        MultiAutoencoderNP(
          indices=self.port_to_nodes[s],
          input_dim=2*n_nodes_intf,
          autoencoders={p: self.ls_models["port"][p] for p in sub.ports}
        )
      )
    for model in models:
      print(model.latent_dim)
      print(model.input_dim)
      print(model.decoder._w['W1'].shape)
      print(model.encoder._w['W1_scale'].shape)
    self.ls_models["interface"] = models

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

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x
  ):
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    res, hess, cjac = [], [], []
    cres = np.zeros(self.n_constraints)
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
      res_s, cres_s, hess_s, cjac_s = sub.res_jac(
        z=z[s],
        lambdas=lambdas,
        steady=self.steady,
        dt=self.dt,
        z_old=z_old[s] if (z_old is not None) else None
      )
      runtime_s = max(time()-start_s, runtime_s)
      # > Store subdomain-related quantities
      start = time()
      res.append(res_s)
      cres += cres_s
      cjac.append(cjac_s)
      hess.append(hess_s)
      runtime += time()-start
    runtime += runtime_s
    # Assemble
    # -------------
    start = time()
    res, jac = self.dd_fom.assemble_kkt(res, cres, cjac, hess)
    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["res_jac"] += runtime
    return res, jac

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
    self.runtime["res_jac"] += runtime_s
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
    Solves for the u and v states of the FOM using Newton's method.
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
    x, res, *_, flag = solver(x0, dt, nt, guess, use_guess)
    converged = True if (flag[-1] == 0) else False
    # Assemble solution
    uv, z, lambdas = self.assemble_sol(x, map_on_res=True)
    return uv, z, lambdas, res, converged

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
    """
    Compute error between DD-ROM and DD-FOM DD solutions.
    """
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

def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def create_dict(configs):
  # Loop over elements: interior and interface/ports
  cfg = {}
  for (key_i, cfg_i) in configs.items():
    print(key_i)
    if key_i not in cfg:
      cfg[key_i] = []

    # Loop over instances in each element
    if (not isinstance(cfg_i, (list, tuple))):
      cfg_i = [cfg_i]
    for cfg_ij in cfg_i:
      print(cfg_ij)
      tmp = {'encoder': 
        {'weights': {'W1': cfg_ij[key_i][0]['u'].T},
         'input_dim': cfg_ij[key_i][0]['u'].shape[0],
         'latent_dim': cfg_ij[key_i][0]['u'].shape[1]
         },
         'decoder':
        {'weights': {'W2': cfg_ij[key_i][0]['u']},
         'input_dim': cfg_ij[key_i][0]['u'].shape[0],
         'latent_dim': cfg_ij[key_i][0]['u'].shape[1]
         }
        }
      cfg[key_i].append(tmp)
  return cfg