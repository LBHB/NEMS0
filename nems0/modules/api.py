"""
modules for composing modelspecs
"""

from .weight_channels import basic, gaussian
from .fir import basic, filter_bank
from .nonlinearity import double_exponential
from .sum import sum_channels
from .signal_mod import replicate_channels, merge_channels
from .stp import short_term_plasticity
from .state import state_dc_gain
from .scale import null
