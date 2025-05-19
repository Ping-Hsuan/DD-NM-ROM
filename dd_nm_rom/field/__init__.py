__all__ = [
  "Burgers2DExact",
  "SinMultiPeak",
  "SinPeak",
  "PoissonForce"
]

from .exact import Burgers2DExact
from .sin_multi_peak import SinMultiPeak
from .sin_peak import SinPeak
from .poisson_force import PoissonForce

# Data types
from typing import Union
FIELD_TYPES = Union[Burgers2DExact, SinMultiPeak, SinPeak, PoissonForce]
