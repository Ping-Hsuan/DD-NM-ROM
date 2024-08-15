import torch
import inspect
import numpy as np
import scipy.sparse as sp


# Mask
# =====================================
def generate_mask(
  output_dim,
  row_shift,
  row_nonzero
):
  """
  Generates a sparsity mask for decoder.
  """
  # Compute hidden layer dimension
  hidden_dim = int(row_nonzero + row_shift*(output_dim-1))
  # Initialize indices
  e = np.ones(row_nonzero, dtype=np.int32)
  indices = np.arange(row_nonzero)
  # Get row/col indices
  row, col = [], []
  for i in range(output_dim):
    row.append(i*e)
    col.append(indices + i*row_shift)
    if (((row_shift+1)*row_nonzero + i*row_shift - 1) < hidden_dim):
      row.append(i*e)
      col.append(indices + i*row_shift + row_shift*row_nonzero)
    if ((-row_shift*row_nonzero + i*row_shift) >= 0):
      row.append(i*e)
      col.append(indices + i*row_shift - row_shift*row_nonzero)
  # Assemble mask
  row, col = [np.concatenate(x, dtype=np.int32) for x in (row, col)]
  data = np.ones(row.size, dtype=np.int32)
  mask = sp.coo_matrix((data, (row, col)), shape=(output_dim, hidden_dim))
  return mask, hidden_dim


# Activation function
# =====================================
_ACT_IDS = ("elu", "linear", "sigmoid", "swish")

def get_activation(identifier="sigmoid", *args, **kwargs):
  if (isinstance(identifier, str) and (identifier.lower() in _ACT_IDS)):
    return {
      "elu":     torch.nn.ELU,
      "linear":  Linear,
      "sigmoid": torch.nn.Sigmoid,
      "swish":   Swish
    }[identifier.lower()](*args, **kwargs)
  elif callable(identifier):
    if inspect.isclass(identifier):
      return identifier()
    else:
      return identifier
  else:
    raise ValueError(
      f"Could not interpret activation function identifier: '{identifier}'."
    )

# Linear
# -------------------------------------
class Linear(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x

# Swish
# -------------------------------------
class Swish(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x * torch.sigmoid(x)
