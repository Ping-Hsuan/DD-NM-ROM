__all__ = [
  "Burgers2DExact",
  "SinMultiPeak",
  "SinPeak",
  "PoissonForce",
  "PoissonWaveDirichlet",
  "PoissonWaveMixed",
  "PoissonWaveDirichletPositive",
  "MultiPeak",
  "MultiPeakGen",
  "MultiPeakGen2"
]

from .exact import Burgers2DExact
from .sin_multi_peak import SinMultiPeak
from .sin_peak import SinPeak
from .poisson_force import PoissonForce
from .poisson_wave import PoissonWaveDirichlet, PoissonWaveMixed
from .poisson_wave_positive import PoissonWaveDirichletPositive
from .multi_peak import MultiPeak
from .multi_peak_sgn import MultiPeakSgn
from .multi_peak_gen import MultiPeakGen
from .multi_peak_gen2 import MultiPeakGen2

# Data types
from typing import Union
FIELD_TYPES = Union[Burgers2DExact, SinMultiPeak, SinPeak, PoissonForce, MultiPeak, MultiPeakSgn, MultiPeakGen, MultiPeakGen2, PoissonWaveDirichlet, PoissonWaveMixed, PoissonWaveDirichletPositive]
