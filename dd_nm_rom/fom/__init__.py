__all__ = [
  "Burgers2D",
  "DDBurgers2D"
]

from .domain_dec import DDBurgers2D
from .monolithic import Burgers2D

# Data types
from typing import Union
FOM_TYPES = Union[Burgers2D, DDBurgers2D]
