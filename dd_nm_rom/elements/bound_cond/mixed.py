import numpy as np

from typing import Dict

from .. import mesh as mesh_mod
from .dirichlet import DirichletBC
from .neumann import NeumannBC
from .utils import get_axis_method, check_side


class MixedBC(NeumannBC):
  """
  Mixed Dirichlet/Neumann boundary conditions class.

  Generalizes DirichletBC/NeumannBC to a per-side choice of BC type: each
  side of the domain independently gets either the plain ghost-elimination
  (Dirichlet, source-only, no stencil change) or the Neumann ghost-elimination
  (source + stencil correction) treatment, selected via `bc_spec`. Reuses
  DirichletBC/NeumannBC's own per-side source/operator formulas unchanged --
  only the per-side dispatch is new.

  :param nu: Viscosity or diffusion coefficient.
  :type nu: float
  :param mesh: Mesh object on which the boundary conditions are applied.
  :type mesh: MESH_TYPES
  :param funval: Per-side function values, same contract as DirichletBC/NeumannBC.
  :type funval: Dict[str, Dict[str, callable]]
  :param bc_spec: Per-side BC type, one of "dirichlet"/"neumann" for each side.
  :type bc_spec: Dict[str, str]
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    nu: float,
    mesh: mesh_mod.MESH_TYPES,
    funval: Dict[str, Dict[str, callable]],
    bc_spec: Dict[str, str],
    advection: bool = True
  ) -> None:
    super(MixedBC, self).__init__(nu, mesh, funval, advection)
    self.name = "mixed"
    for side, side_type in bc_spec.items():
      check_side(side)
      if side_type not in ("dirichlet", "neumann"):
        raise ValueError(
          f"Could not interpret b.c. side type: '{side_type}'. " \
            "Valid options are: ('dirichlet', 'neumann')"
        )
    self.bc_spec = bc_spec

  # Building
  # ===================================
  def build(self) -> None:
    """
    Build the boundary condition operators and source terms, using the
    Dirichlet or Neumann treatment per side according to `bc_spec`.
    """
    self.f = {}
    for k in ("u", "v"):
      f_k = {}
      for (side, sfunval) in self.funval.items():
        axis, method = get_axis_method(side)
        if self.bc_spec[side] == "neumann":
          f_k[side] = NeumannBC._build_src(self, sfunval[k], axis, method)
        else:
          f_k[side] = DirichletBC._build_src(self, sfunval[k], axis, method)
      self.f[k] = {"D": self._compose_dif_src(f_k)}
      if self.advection:
        self.f[k]["A"] = self._compose_adv_src(f_k)

    self.op = {"x": {}, "y": {}}
    for side in self.funval.keys():
      if self.bc_spec[side] != "neumann":
        continue
      axis, method = get_axis_method(side)
      self.op[axis][method] = self._build_op(axis=axis, method=method)

    self.built = True
