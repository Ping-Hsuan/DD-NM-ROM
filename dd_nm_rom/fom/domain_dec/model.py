import numpy as np
import scipy.sparse as sp

from time import time
from dd_nm_rom import ops
from dd_nm_rom import solvers
from dd_nm_rom import backend as bkd

from .indices import DDIndices
from .subdomain import Subdomain


class DDBurgers2D(object):
  """
  Class to compute domain decomposition model from steady-state 2D Burgers FOM

  inputs:
  monolithic: instance of Burgers2D class
  n_sub_x: number of subdomains in x direction. Must divide monolithic.nx
  n_sub_y: number of subdmoains in y direction. Must divide monolithic.ny

  fields:
  nxy:       total number of nodes (finite difference grid points) in full domain model
  n_sub:     number of subdomains
  skeleton:  self.dd_indices of each node in the skeleton
  ports:     list of frozensets corresponding to each port in the DD model.
          ports[i] = frozenset of the subdomains contained in port i
  ports_dict: dictionary of port self.dd_indices where
          ports_dict[port[i]] = self.dd_indices in port[i]
  n_constraints_weak: number of (equality) constraints for DD model
  subdomain: list of instances of Subdomain class where
          subdomain[i] = Subdomain instance corresponding to ith subdomain

  methods:
  set_bc: update boundary condition data
  FJac: computes the KKT system to be solved at each iteration of the Lagrange-solvers.Newton SQP solver
  solve: solves for the states of the DD model using the Lagrange-solvers.Newton-SQP method
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    monolithic,
    n_constraints_weak=1,
    constraint_type="strong",
    scaling=1.0
  ):
    # FOM monolithic
    # -------------
    self.monolithic = monolithic
    for k in ("runtime", "mesh"):
      setattr(self, k, getattr(self.monolithic, k))
    # > Scaling factor for residual
    self.scaling = self.mesh.hxy if (scaling <= 0) else scaling
    # Constraints
    # -------------
    self.constraint_type = constraint_type
    if (self.constraint_type not in ("weak", "strong")):
      raise ValueError(
        f"Could not interpret constraint type: '{self.constraint_type}'. " \
          "Valid options are: ['weak', 'strong']."
      )
    self.n_constraints_weak = int(n_constraints_weak)
    # Integration
    # -------------
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Control variables
    # -------------
    self.built = False

  # Building
  # ===================================
  def is_built(self) -> None:
    self.monolithic.is_built()
    if (not self.built):
      raise ValueError(
        "DD-FOM model not built. Please, call 'build' method first."
      )

  def build(self):
    self.monolithic.is_built()
    # DD-FOM subdomains indices
    self.dd_indices = DDIndices(self.mesh, self.monolithic)
    # Constraints
    self.cmat = self.assemble_cmat()
    # DD-FOM subdomains
    self.subdomains = []
    for s in range(self.mesh.n_sub):
      cmat_s, indices_s = {}, {}
      for e_k in ("res", "interior", "interface"):
        indices_s[e_k] = getattr(self.dd_indices, e_k)[s]
        if (e_k != "res"):
          cmat_s[e_k] = self.cmat[e_k][s]
      self.subdomains.append(
        Subdomain(
          monolithic=self.monolithic,
          nodes_ind=indices_s,
          cmat=cmat_s,
          ports=self.dd_indices.sub_to_ports[s],
          port_to_nodes=self.dd_indices.port_to_nodes,
          scaling=self.scaling
        )
      )
    # Update control variables
    self.built = True

  def get_ndof(self):
    ndof = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        ndof += sub.elem_states[e_k].n_nodes_state
    ndof *= 2
    ndof += self.n_constraints
    return ndof

  # Constraint matrices
  # -----------------------------------
  def assemble_cmat(self):
    # Compute total number of constraints
    self.n_constraints = 0
    for (p, subs_p) in self.dd_indices.port_to_subs.items():
      n_ports = len(self.dd_indices.port_to_nodes[p])
      self.n_constraints += (len(subs_p)-1) * n_ports
    # Assemble constraints matrices
    cmat = {
      "interior": self.init_cmat(element="interior"),
      "interface": self.assemble_cmat_intf()
    }
    # > Make constraint matrices block diagonal for u and v components
    for (e_k, cmat_k) in cmat.items():
      cmat[e_k] = [sp.block_diag([m, m]) for m in cmat_k]
    self.n_constraints *= 2
    # Convert to weak constraints
    if (self.constraint_type == "weak"):
      cmat, self.n_constraints = self.assemble_cmat_weak(
        cmat=cmat,
        n_constraints_weak=self.n_constraints_weak,
        n_constraints=self.n_constraints
      )
    return cmat

  def init_cmat(self, element):
    cmat = []
    for nodes_ind in getattr(self.dd_indices, element):
      cmat.append(sp.coo_matrix((self.n_constraints, len(nodes_ind))))
    return cmat

  def assemble_cmat_intf(self):
    # Initialize matrices
    cmat = self.init_cmat(element="interface")
    # Fill matrices
    shift = 0
    for (p, subs_p) in self.dd_indices.port_to_subs.items():
      port_nodes = self.dd_indices.port_to_nodes[p]
      port_size = port_nodes.size
      for i in range(len(subs_p)-1):
        for (j, l) in enumerate((i,i+1)):
          s = subs_p[l]
          intf_nodes = self.dd_indices.interface[s]
          col = np.where(np.isin(intf_nodes, port_nodes))[0]
          row = np.arange(port_size) + shift
          dat = (-1)**j * np.ones(port_size)
          cmat[s].col = np.concatenate((cmat[s].col, col))
          cmat[s].row = np.concatenate((cmat[s].row, row))
          cmat[s].data = np.concatenate((cmat[s].data, dat))
        shift += port_size
    return cmat

  def assemble_cmat_weak(
    self,
    cmat,
    n_constraints_weak,
    n_constraints
  ):
    n_constraints_weak = max(n_constraints_weak, 1)
    n_constraints_weak = min(n_constraints_weak, n_constraints)
    rgen = np.random.default_rng(bkd.seed())
    rmat = rgen.standard_normal((n_constraints_weak, n_constraints))
    for e_k in ("interior", "interface"):
      cmat[e_k] = [rmat @ m for m in cmat[e_k]]
    cmat = ops.map_nested_dict(cmat, bkd.to_sparse)
    return cmat, n_constraints_weak

  # RHS/Jacobian
  # ===================================
  def rhs_jac(
    self,
    x
  ):
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    rhs, hess, cjac = [], [], []
    crhs = np.zeros(self.n_constraints)
    runtime += time()-start
    # Assemble solution
    # -------------
    start = time()
    uv, lambdas = self.assemble_sol(x, map_on_res=False)
    if (not self.steady):
      uv_old, _ = self.assemble_sol(self.x_old, map_on_res=False)
    runtime += (time()-start) / self.mesh.n_sub
    # Loop over subdomains
    # -------------
    runtime_s = 0.0
    uv_s_old = None
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      # > Get u and v at interior and interface nodes for subdomain 's'
      uv_s = self.extract_uv_sub_from_dict(uv, s)
      if (not self.steady):
        uv_s_old = self.extract_uv_sub_from_dict(uv_old, s)
      # > Compute quantities needed for KKT system
      rhs_s, crhs_s, hess_s, cjac_s = sub.rhs_jac(
        uv=uv_s,
        lambdas=lambdas,
        steady=self.steady,
        dt=self.dt,
        uv_old=uv_s_old
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
    rhs, jac = self.assemble_kkt(rhs, crhs, cjac, hess)
    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["rhs_jac"] += runtime
    return rhs, jac

  def assemble_sol(
    self,
    x,
    map_on_res=False
  ):
    shape = [self.mesh.nxy]
    if (x.ndim == 2):
      shape.append(x.shape[1])
    uv = self.init_uv(shape=shape, map_on_res=map_on_res)
    # Loop over subdomains
    si = 0
    for sub in self.subdomains:
      # Loop over elements
      for e_k in ("interior", "interface"):
        state_k = sub.elem_states[e_k]
        ei = si + 2*state_k.n_nodes_state
        uv = self.extract_uv_sub_from_vec(
          uv=uv,
          uv_i=x[si:ei],
          elem_state=state_k,
          map_on_res=map_on_res
        )
        si = ei
    lambdas = x[-self.n_constraints:]
    return uv, lambdas

  def init_uv(
    self,
    shape,
    map_on_res=True
  ):
    uv = {}
    for e_k in ("interior", "interface"):
      uv[e_k] = {}
      for x_k in ("u", "v"):
        uv[e_k][x_k] = []
    if map_on_res:
      uv["res"] = {}
      for x_k in ("u", "v"):
        uv["res"][x_k] = np.zeros(shape)
    return uv

  def extract_uv_sub_from_vec(
    self,
    uv,
    uv_i,
    elem_state,
    map_on_res=True
  ):
    size = elem_state.n_nodes_state
    indices = elem_state.nodes_state
    # u velocity
    u = uv_i[:size]
    uv[elem_state.name]["u"].append(u)
    if map_on_res:
      uv["res"]["u"][indices] = u
    # v velocity
    v = uv_i[size:]
    uv[elem_state.name]["v"].append(v)
    if map_on_res:
      uv["res"]["v"][indices] = v
    return uv

  def extract_uv_sub_from_dict(
    self,
    uv,
    index
  ):
    uv_s = {}
    for e_k in ("interior", "interface"):
      uv_s[e_k] = {}
      for x_k in ("u", "v"):
        uv_s[e_k][x_k] = uv[e_k][x_k][index]
    return uv_s

  def assemble_kkt(
    self,
    rhs,
    crhs,
    cjac,
    hess
  ):
    # > RHS
    rhs.append(crhs)
    rhs = np.concatenate(rhs)
    # > Constraints Jacobian
    cjac = sp.hstack(cjac)
    # > Hessians
    hess = sp.block_diag(hess)
    # > Full Jacobian
    jac = sp.bmat(
      [[hess, cjac.T],
       [cjac,   None]],
      format="csr"
    )
    return rhs, jac

  # Solution
  # ===================================
  def solve(
    self,
    x0=None,
    dt=0.0,
    nt=1,
    steady=True,
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
    self.is_built()
    self.runtime = ops.map_nested_dict(self.runtime, lambda _: 0.0)
    # Initialize solution
    start = time()
    if (x0 is None):
      x0 = np.zeros(self.get_ndof())
    self.runtime["total"] += time()-start
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
    x, rhs, *_, flag = solver(x0, dt, nt)
    converged = True if (flag[-1] == 0) else False
    # Assemble solution
    uv, lambdas = self.assemble_sol(x, map_on_res=True)
    return uv, lambdas, rhs, converged

  def get_init_sol(
    self,
    x
  ):
    self.is_built()
    # Map init sol on elements
    x = self.map_sol_on_elements(x.reshape(1,-1))
    # Initial guess
    x_dd = []
    # Loop over subdomains
    for s in range(self.mesh.n_sub):
      # Loop over elements
      for e_k in ("interior", "interface"):
        x_dd.append(x[e_k][s].reshape(-1))
    x_dd.append(np.zeros(self.n_constraints))
    return np.concatenate(x_dd)

  def map_sol_on_elements(
    self,
    solutions,
    map_on_ports=False
  ):
    self.is_built()
    if (solutions.shape[1] != self.monolithic.get_ndof()):
      solutions = solutions.T
    data = {}
    for e_k in ("res", "interior", "interface"):
      data[e_k] = []
      for sub in self.subdomains:
        indices = sub.elem_states[e_k].nodes_state
        indices = np.concatenate([indices, indices+self.mesh.nxy])
        data[e_k].append(solutions[:,indices])
    if map_on_ports:
      data["port"] = []
      for p in self.dd_indices.ports:
        indices = self.dd_indices.port_to_nodes[p]
        indices = np.concatenate([indices, indices+self.mesh.nxy])
        data["port"].append(solutions[:,indices])
    return data
