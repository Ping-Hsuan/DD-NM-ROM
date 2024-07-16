import os

from typing import Union


def set(
  backend: str = "numpy",
  device: str = "cpu",
  device_idx: int = 0,
  nb_threads: int = 8,
  epsilon: Union[float, None] = 1e-10,
  floatx: str = "float64",
  seed: Union[int, None] = None
) -> None:
  nb_threads = int(nb_threads)
  _set_cpu_threads(nb_threads)
  from . import backend as bkd
  bkd.set(
    backend=backend,
    device=device,
    device_idx=device_idx,
    nb_threads=nb_threads,
    epsilon=epsilon,
    floatx=floatx,
    seed=seed
  )

def _set_cpu_threads(
  nb_threads: int
) -> None:
  for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
  ):
    os.environ[k] = str(nb_threads)
