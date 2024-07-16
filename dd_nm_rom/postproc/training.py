import numpy as np
import matplotlib.pyplot as plt


def plot_loss(
  hist_df=None,
  path=None,
  scale="log",
  title="Loss",
  vars=["loss","val_loss"],
  fig_name="loss",
  label=["Training", "Validation"],
  window=1
):
  plt.figure()
  plt.yscale(scale)
  for (i, var_i) in enumerate(vars):
    if (var_i in hist_df):
      plt.plot(
        moving_average(hist_df[var_i], w=window),
        lw=1,
        label=label[i]
      )
      if (window != 1):
        plt.plot(
          hist_df[var_i],
          lw=1,
          c=plt.gca().lines[-1].get_color(),
          alpha=0.2
        )
  plt.xlabel("Epoch")
  plt.legend()
  plt.title(title + " History")
  plt.savefig(path+f"/{fig_name}.png", bbox_inches="tight", pad_inches=0.1)
  plt.close()

def moving_average(x, w=20):
  return np.convolve(x, np.ones(w), "valid") / w

def plot_lr(hist_df, path):
  plt.figure()
  plt.plot(hist_df[["lr"]], "r", lw=1)
  plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
  plt.xlabel("Epoch")
  plt.title("Learning Rate History")
  plt.savefig(path+"/lr.png", bbox_inches="tight", pad_inches=0.1)
  plt.close()
