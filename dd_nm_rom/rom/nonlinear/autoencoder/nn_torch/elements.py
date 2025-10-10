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


def generate_mask2(
    output_dim,
    input_dim,
    row_shift,
    row_nonzero
):
  """
  Generates a sparsity mask for decoder with specified input and output dimensions.
  Each output neuron is connected to a block of `row_nonzero` input neurons
  """
  # Gradually shift block so every output neuron has valid, evenly distributed connections
  row, col = [], []
  block_size = row_nonzero if row_nonzero > 0 else 1
  max_start = input_dim - block_size
  for i in range(output_dim):
    if output_dim == 1:
      start = 0
    else:
      start = int(round(i * max_start / (output_dim - 1))) if max_start > 0 else 0
    end = start + block_size
    cols = np.arange(start, end)
    row.append(np.full(cols.shape, i, dtype=np.int32))
    col.append(cols)
  row = np.concatenate(row, dtype=np.int32)
  col = np.concatenate(col, dtype=np.int32)
  data = np.ones(row.size, dtype=np.int32)
  mask = sp.coo_matrix((data, (row, col)), shape=(output_dim, input_dim))
  return mask

# Activation function
# =====================================
_ACT_IDS = ("elu", "linear", "sigmoid", "swish", "softplus")

def get_activation(identifier="sigmoid", *args, **kwargs):
  if (isinstance(identifier, str) and (identifier.lower() in _ACT_IDS)):
    return {
      "elu":     torch.nn.ELU,
      "linear":  Linear,
      "sigmoid": torch.nn.Sigmoid,
      "swish":   Swish,
      "softplus": torch.nn.Softplus
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
