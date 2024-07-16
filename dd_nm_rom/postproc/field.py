import os
import numpy as np

from matplotlib import pyplot as plt

from .utils import set_style
plt = set_style(plt)


def plot_field(
  x,
  y,
  z,
  label,
  lim=None,
  figsize=None,
  cmap="viridis",
  filename="./state.png",
  save=True,
  show=False
):
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
  plt.xlabel("$x$")
  plt.ylabel("$y$")
  cbar = plt.colorbar(orientation="vertical", label=label)
  cbar.formatter.set_powerlimits((0, 0))
  if save:
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
  if show:
    plt.show()
  plt.close()

def plot_field_fom_rom(
  path,
  mesh,
  uv_fom,
  uv_rom=None,
  index=None,
  figsize=None,
  use_label=True
):
  for x_k in ("u", "v"):
    path_k = path + f"/{x_k}/"
    os.makedirs(path_k, exist_ok=True)
    zt = uv_fom["res"][x_k]
    lim = [zt.min(), zt.max()]
    if (uv_rom is not None):
      z = uv_rom["res"][x_k]
      label = "$\hat{%s}$" % x_k
      err = np.abs(z - zt)
      err_lim = [err.min(), err.max()]
    else:
      z = zt
      label = "$%s$" % x_k
      err, err_lim = None, None
    # Solution
    filename = path_k + "state"
    if (index is not None):
      z = z.T[index]
      filename += f"_i{str(index).zfill(5)}"
    filename += ".png"
    plot_field(
      *mesh.grid,
      z=z.reshape(mesh.n["y"], mesh.n["x"]),
      lim=lim,
      figsize=figsize,
      label=label if use_label else None,
      filename=filename,
      save=True,
      show=False
    )
    # Error
    if (err is not None):
      filename = path_k + "state_err"
      if (index is not None):
        err = err.T[index]
        filename += f"_i{str(index).zfill(5)}"
      filename += ".png"
      label = "$|\hat{%s}-%s|$" % (x_k, x_k)
      plot_field(
        *mesh.grid,
        z=err.reshape(mesh.n["y"], mesh.n["x"]),
        lim=err_lim,
        figsize=figsize,
        label=label if use_label else None,
        filename=filename,
        save=True,
        show=False
      )
