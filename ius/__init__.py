"""Information Uniqueness Score package.

See ``paper/main.pdf`` for the methodology and the headline empirical result.
"""
from .ils import IlsEstimate, block_bootstrap_ils, estimate_ils
from .leadership import GrangerResult, granger_f_ratio
from .simulate import channel_names, simulate_market
from .stats import PermutationResult, permutation_spearman

__all__ = [
    "IlsEstimate",
    "GrangerResult",
    "PermutationResult",
    "block_bootstrap_ils",
    "channel_names",
    "estimate_ils",
    "granger_f_ratio",
    "permutation_spearman",
    "simulate_market",
]

__version__ = "0.1.0"
