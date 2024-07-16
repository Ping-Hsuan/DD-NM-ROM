import copy
import torch


_OPTIM_IDS = ("sgd", "rmsprop", "adam")

def get(
  params,
  optimizer,
  lr=None,
  decay=None
):
  if (isinstance(optimizer, str) and (optimizer.lower() in _OPTIM_IDS)):
    if (lr is None):
      raise ValueError(
        f"No learning rate for '{optimizer}' optimizer."
      )
    optim = {
      "sgd":     torch.optim.SGD,
      "rmsprop": torch.optim.RMSprop,
      "adam":    torch.optim.Adam
    }[optimizer.lower()](params, lr=lr)
  else:
    raise ValueError(
      f"Could not interpret optimizer identifier: '{optimizer}'"
    )
  lr_scheduler = _get_lr_scheduler(optim, decay)
  return optim, lr_scheduler

def _get_lr_scheduler(
  optim,
  decay
):
  if (decay is None):
    return None
  decay = copy.deepcopy(decay)
  name = decay.pop('name')
  lr_scheduler = {
    "exponential":       torch.optim.lr_scheduler.ExponentialLR,
    "polynomial":        torch.optim.lr_scheduler.PolynomialLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "step":              torch.optim.lr_scheduler.StepLR
  }[name]
  scheduler = lr_scheduler(optim, **decay)
  scheduler.metrics_needed = True if (name == "reduce_on_plateau") else False
  return scheduler
