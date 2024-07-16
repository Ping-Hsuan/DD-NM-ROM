import os
import sys
import types
import inspect
import joblib as jl
import dill as pickle

from tqdm import tqdm
from dd_nm_rom import ops
from typing import Any, List, Union
from dd_nm_rom import fom as fom_mod


# Classes
# =====================================
def get_class(
  modules: Union[types.ModuleType, List[types.ModuleType]],
  name: Union[str, None] = None,
  kwargs: Union[dict, None] = None
) -> callable:
  """
  Return a class object given its name and the module it belongs to.

  :param modules: A module or a list of modules to search for the class.
  :type modules: list or module

  :param name: The name of the class to retrieve.
  :type name: str, optional

  :param kwargs: Optional keyword arguments to pass when initializing the
                 class (it can contain the name of the class if 'name'
                 is not provided).
  :type kwargs: dict, optional

  :return: An instance of the class if found, or the class itself.
  :rtype: object or class

  This function searches for a class with the specified name within the given
  module(s). If found, it can return an instance of the class with optional
  keyword arguments provided in `kwargs`. If no class is found, an error is
  raised.
  """
  # Check class name
  if ((name is None) and (kwargs is not None)):
    if ("name" in kwargs.keys()):
      name = kwargs.pop("name")
    else:
      raise ValueError("Class name not provided.")
  # Loop over modules to find class
  if (not isinstance(modules, (list, tuple))):
    modules = [modules]
  for module in modules:
    members = inspect.getmembers(module, inspect.isclass)
    for (name_i, cls_i) in members:
      if (name_i == name):
        if (kwargs is not None):
          return cls_i(**kwargs)
        else:
          return cls_i
  # Raise error if class not found
  names = [module.__name__ for module in modules]
  raise ValueError(f"Class `{name}` not found in modules: {names}.")

def check_path(path):
  if (not os.path.exists(path)):
    raise IOError(f"Path '{path}' does not exist.")

# Data
# =====================================
def save_case(
  path: str,
  index: int,
  data: Any
) -> None:
  filename = path + f"/case_{str(index+1).zfill(5)}.p"
  pickle.dump(data, open(filename, "wb"))

def load_case(
  path: str,
  index: int,
  key: Union[str, None] = None
) -> Any:
  filename = path + f"/case_{str(index+1).zfill(5)}.p"
  if os.path.exists(filename):
    data = pickle.load(open(filename, "rb"))
    if (key is None):
      return data
    else:
      return data[key]

def load_case_parallel(
  path: str,
  ranges: List[int],
  key: Union[str, None] = None,
  n_workers: int = 1
) -> Any:
  iterable = tqdm(
    iterable=range(*ranges),
    ncols=80,
    desc="> Cases",
    file=sys.stdout
  )
  if (n_workers > 1):
    return jl.Parallel(n_workers)(
      jl.delayed(load_case)(path=path, index=i, key=key) for i in iterable
    )
  else:
    return [load_case(path=path, index=i, key=key) for i in iterable]

def generate_case_parallel(
  sol_fun: callable,
  n_samples: int,
  n_workers: int = 1,
  desc: str = "> Cases",
  verbose: bool = True
) -> None:
  """
  The 'sol_fun' callable function needs to return
  if the solver has converged or not as 0 or 1.
  """
  iterable = tqdm(
    iterable=range(n_samples),
    ncols=80,
    desc=desc,
    file=sys.stdout
  )
  if (n_workers > 1):
    converged = jl.Parallel(n_workers)(
      jl.delayed(sol_fun)(i) for i in iterable
    )
  else:
    converged = [sol_fun(i) for i in iterable]
  if verbose:
    print(f"> Total converged cases: {sum(converged)}/{n_samples}")
