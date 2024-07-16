import os
import numpy as np

from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from .utils import set_style
plt = set_style(plt)


def _create_animation(
  x,
  y,
  z,
  lim,
  label,
  frames
):
  # Initialize a figure in which the graphs will be plotted
  fig, ax = plt.subplots()
  # Set up axes
  ax.set_xlabel("$x$")
  ax.set_ylabel("$y$")
  # Initialize lines
  style = dict(
    cmap="viridis",
    shading="auto",
    vmin=lim[0],
    vmax=lim[1]
  )
  data = [plt.pcolormesh(x, y, z[0], **style)]
  # Add color bar
  cbar = plt.colorbar(orientation="vertical", label=label)
  cbar.formatter.set_powerlimits((0, 0))
  # Tight layout
  plt.tight_layout()
  # Animate function
  def _animate(frame):
    # Set data
    data[0].set_array(z[frame+1])
    # Rescale axis limits
    ax.relim()
    ax.autoscale_view(tight=True)
    return data
  # Get animation
  anim = FuncAnimation(
    fig,
    _animate,
    frames=frames-1,
    blit=True
  )
  plt.close("all")
  return anim

def animate(
  x,
  y,
  z,
  lim,
  label,
  frames=None,
  fps=10,
  filename="./state.gif",
  dpi=600,
  save=True,
  show=False
):
  if (frames is None):
    frames = len(z)
  # Create animation
  anim = _create_animation(x, y, z, lim, label, frames)
  # Save animation
  if save:
    anim.save(filename, writer="imagemagick", fps=fps, dpi=dpi)
  # Display animation
  if show:
    HTML(anim.to_jshtml())

def animate_fom_rom(
  path,
  mesh,
  uv_fom,
  uv_rom=None,
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
    z = z.T.reshape(-1, mesh.n["y"], mesh.n["x"])
    animate(
      *mesh.grid,
      z=z,
      lim=lim,
      label=label if use_label else None,
      filename=path_k + "/state.gif",
      dpi=300,
      save=True,
      show=False
    )
    # Error
    if (err is not None):
      z = err.T.reshape(-1, mesh.n["y"], mesh.n["x"])
      label = "$|\hat{%s}-%s|$" % (x_k, x_k)
      animate(
        *mesh.grid,
        z=z,
        lim=err_lim,
        label=label if use_label else None,
        filename=path_k + "/state_err.gif",
        dpi=300,
        save=True,
        show=False
      )
