__all__ = [
  "DirichletBC",
  "NeumannBC",
  "PeriodicBC",
  "MixedBC"
]

from .dirichlet import DirichletBC
from .neumann import NeumannBC
from .periodic import PeriodicBC
from .mixed import MixedBC
from .utils import *

# Data types
from typing import Union
BC_TYPES = Union[DirichletBC, NeumannBC, PeriodicBC, MixedBC]
