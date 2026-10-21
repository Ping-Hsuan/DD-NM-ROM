__all__ = [
  "Burgers2D",
  "DDBurgers2D",
  "Poisson2D",
  "DDPoisson2D",
  "AdvDiff2D",
  "DDAdvDiff2D",
  "ReactAdvDiff2D",
  "DDReactAdvDiff2D",
  "FisherKPPAdvDiff2D"
]

from .domain_dec import DDBurgers2D, DDPoisson2D, DDAdvDiff2D, DDReactAdvDiff2D
from .monolithic import Burgers2D, Poisson2D, AdvDiff2D, ReactAdvDiff2D, FisherKPPAdvDiff2D

# Data types
from typing import Union
FOM_TYPES = Union[Burgers2D, DDBurgers2D, Poisson2D, DDPoisson2D, AdvDiff2D, DDAdvDiff2D, ReactAdvDiff2D, DDReactAdvDiff2D, FisherKPPAdvDiff2D]
