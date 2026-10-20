import os
import numpy as np

from matplotlib import pyplot as plt
from typing import List, Union, Tuple
from dd_nm_rom.fom import domain_dec as dd_mod
from dd_nm_rom.elements import mesh as mesh_mod

from .utils import set_style
plt = set_style(plt)


def plot_field(
  x: np.ndarray,
  y: np.ndarray,
  z: np.ndarray,
  label: str,
  show_labels: bool = True,
  lim: Union[List[float], None] = None,
  figsize: Union[Tuple[int], None] = None,
  cmap: str = "viridis",
  filename: str = "./state.png",
  save: bool = True,
  show: bool = False
) -> None:
  if (figsize is not None):
    plt.figure(figsize=figsize)
  else:
    plt.figure()
  style = dict(
    cmap=cmap,
    shading="auto"
  )
  if (lim is not None):
    style["vmin"] = lim[0]
    style["vmax"] = lim[1]
  plt.pcolormesh(x, y, z, **style)
  if show_labels:
    plt.xlabel("$x$", labelpad=4)
    plt.ylabel("$y$", labelpad=12)
  label = label if show_labels else None
  cbar = plt.colorbar(orientation="vertical", label=label)
  # cbar.formatter.set_powerlimits((0, 0))
  if save:
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
  if show:
    plt.show()
  plt.close()

def plot_field_fom_rom(
  path: str,
  mesh: mesh_mod.MESH_TYPES,
  uv_fom: dd_mod.dtypes.UV_TYPE,
  uv_rom: Union[dd_mod.dtypes.UV_TYPE, None] = None,
  index: Union[int, None] = None,
  figsize: Union[Tuple[int], None] = None,
  show_labels: bool = True,
  use_fom_snapshot_limits: bool = False
) -> None:
  # Per-timestep MEAN-SQUARE FOM value, pooled over BOTH u and v and all
  # nodes -- i.e. den(t) = (1/N) * sum_x u(x,t)^2, N = 2*mesh.nxy. This is
  # the RMS-squared: the *local, per-node* reference to compare a pointwise
  # error against (units match u itself, so err_rel below is genuinely
  # dimensionless), as opposed to the plain (extensive, whole-domain) L2
  # norm used by compute_error, whose units are [u]*[length] and which
  # scales with node count/domain size -- meaningless as a per-pixel
  # denominator.
  #
  # err_rel(x,t) = |diff(x,t)|/sqrt(den(t)) still satisfies the exact
  # identity mean_x[err_rel(x,t)^2] = err(t) = compute_error's per-timestep
  # relative-L2-squared error, so max_t RMS_x[err_rel(x,t)] reproduces the
  # reported scalar error/error_timeavg metrics exactly, and
  # max_t max_x[err_rel(x,t)] reproduces compute_error_linf -- while also
  # being numerically safe (no local/pointwise division, so no blow-up near
  # masked-off/near-zero regions). Both this and the plain absolute error
  # are saved side by side: absolute error can look like it's growing over
  # time purely because the reference solution itself decayed (den
  # shrinking), not because the ROM got worse -- keeping both lets that be
  # told apart.
  den = None
  if uv_rom is not None:
    n_terms = 2 * mesh.nxy
    den = sum(np.sum(np.square(uv_fom["res"][k]), axis=0) for k in ("u", "v"))
    den = den / n_terms

  def _snapshot_lim(arr):
    # Scale to THIS timestep's own range (use_fom_snapshot_limits), not the
    # worst/widest range over the whole trajectory -- otherwise every
    # plotted instant would share one range capped by whichever single
    # moment (possibly one of the unplotted intermediate timesteps) is most
    # extreme, washing out later, smaller-but-still-real changes.
    if use_fom_snapshot_limits and index is not None:
      arr_snap = arr.T[index]
      return [arr_snap.min(), arr_snap.max()]
    return [arr.min(), arr.max()]

  def _save_field(z_full, lim, label, filename):
    z_plot = z_full.T[index] if (index is not None) else z_full
    if (index is not None):
      filename += f"_i{str(index).zfill(5)}"
    filename += ".png"
    plot_field(
      *mesh.grid,
      z=z_plot.reshape(mesh.n["y"], mesh.n["x"]),
      lim=lim,
      figsize=figsize,
      label=label,
      show_labels=show_labels,
      filename=filename,
      save=True,
      show=False
    )

  for x_k in ("u", "v"):
    path_k = path + f"/{x_k}/"
    os.makedirs(path_k, exist_ok=True)

    zt = uv_fom["res"][x_k]
    lim = _snapshot_lim(zt)

    if (uv_rom is not None):
      z = uv_rom["res"][x_k]
      label = "$\hat{%s}$" % x_k
      err_abs = np.abs(z - zt)
      # Broadcasts correctly for both unsteady arrays (err_abs:
      # (n_nodes, n_time), den: (n_time,) -> per-column division) and
      # steady arrays (err_abs: (n_nodes,), den: scalar).
      err_rel = err_abs / np.sqrt(den)
      err_abs_lim = _snapshot_lim(err_abs)
      err_rel_lim = _snapshot_lim(err_rel)
    else:
      z = zt
      label = "$%s$" % x_k
      err_abs, err_rel = None, None

    # Solution
    _save_field(z, lim, label, path_k + "state")

    # Error (absolute)
    if (err_abs is not None):
      label = "$|\hat{%s}-%s|$" % (x_k, x_k)
      _save_field(err_abs, err_abs_lim, label, path_k + "state_err")

    # Error (relative)
    if (err_rel is not None):
      label = "$|\hat{%s}-%s|\,/\,\mathrm{RMS}(\mathrm{FOM})$" % (x_k, x_k)
      _save_field(err_rel, err_rel_lim, label, path_k + "state_err_rel")
