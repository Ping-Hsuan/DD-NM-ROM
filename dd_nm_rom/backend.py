import os
import torch
import random
import numpy as np
import scipy as sp

from typing import Any, Union


# Global
# -------------------------------------
_SEED = None
_VALID_BKD = {"numpy", "torch"}
_VALID_DEVICE = {"cpu", "cuda"}
_VALID_DTYPE = {"float32", "float64"}

# Setting
# -------------------------------------
def set(
  backend: str = "numpy",
  device: str = "cpu",
  device_idx: int = 0,
  nb_threads: int = 8,
  epsilon: Union[float, None] = 1e-10,
  floatx: str = "float64",
  seed: Union[int, None] = None
) -> None:
  set_backend(backend)
  set_seed(seed)
  set_device(device, device_idx, nb_threads)
  set_floatx(floatx)
  set_epsilon(epsilon)

def get_backend() -> str:
  return _BKD

def set_backend(
  value: str = "numpy"
) -> None:
  global _BKD
  _BKD = value
  if (value not in _VALID_BKD):
    raise ValueError(
      f"Unknown backend: '{value}'. Valid options are: {_VALID_BKD}"
    )

# Conversion
# -------------------------------------
def to_numpy(x: Any) -> Any:
  if (x is not None):
    if isinstance(x, np.ndarray):
      return x
    elif (torch.is_tensor(x)):
      return x.numpy(force=True)
    elif isinstance(x, (int, float, list, tuple)):
      return np.array(x, dtype=floatx("numpy"))
    else:
      return x

def to_backend(x: Any) -> Any:
  if (x is not None):
    if (_BKD == "torch"):
      if torch.is_tensor(x):
        return x
      else:
        return torch.as_tensor(to_numpy(x), dtype=floatx("torch"))
    else:
      return to_numpy(x)

def to_sparse(
  x: Union[np.ndarray, sp.sparse.spmatrix]
) -> sp.sparse.spmatrix:
  return x.tocsr() if sp.sparse.issparse(x) else sp.sparse.csr_matrix(x)

# Device
# -------------------------------------
def device() -> str:
  return _DEVICE

def set_device(
  value: str = None,
  index: int = 0,
  nb_threads: int = 8,
) -> None:
  if ((value is None) or (value == "cuda")):
    value = "cuda" if torch.cuda.is_available() else "cpu"
  if (value not in _VALID_DEVICE):
    raise ValueError(
      f"Unknown device: '{value}'. Valid options are: {_VALID_DEVICE}"
    )
  if (value == "cuda"):
    value += f":{index}"
  global _DEVICE
  _DEVICE = value
  # Set default device
  try:
    torch.set_default_device(torch.device(_DEVICE))
    torch.set_num_interop_threads(nb_threads)
    torch.set_num_threads(nb_threads)
  except:
    pass

# Epsilon
# -------------------------------------
def machine_eps() -> float:
  return float(np.finfo(
    {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  ).eps)

def epsilon() -> float:
  return _EPSILON

def set_epsilon(
  value: Union[float, None] = None
) -> None:
  if (value is None):
    value = machine_eps()
  global _EPSILON
  _EPSILON = value

# Float
# -------------------------------------
def floatx(
  bkd: str = "torch"
) -> Union[str, type, torch.dtype]:
  if (bkd == "torch"):
    return {
      "float16": torch.float16,
      "float32": torch.float32,
      "float64": torch.float64
    }[_FLOATX]
  elif (bkd == "numpy"):
    return {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  else:
    return _FLOATX

def set_floatx(
  value: str
) -> None:
  global _FLOATX
  _FLOATX = value
  if (value not in _VALID_DTYPE):
    raise ValueError(
      f"Unknown dtype: '{value}'. Valid options are: {_VALID_DTYPE}"
    )
  try:
    torch.set_default_dtype(floatx())
  except:
    pass

# Seed
# -------------------------------------
def seed() -> Union[int, None]:
  return _SEED

def set_seed(
  value: Union[int, None] = None
) -> None:
  """
  Set random number generator seeds for reproducibility.

  :param value: An integer seed for random number generators.
  :type value: int or None

  This function sets the seed for Python's built-in random module, NumPy,
  PyTorch, and ensures deterministic operations. It's essential for
  achieving reproducible results in data processing and machine learning
  tasks. If `value` is provided, all random generators will use the same seed.
  """
  global _SEED
  _SEED = value
  if (value is not None):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    # torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(value)
