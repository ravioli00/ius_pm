"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A reproducible RNG for synthetic-DGP tests."""
    return np.random.default_rng(20260502)


@pytest.fixture
def K() -> int:
    """Default number of financial channels in synthetic markets."""
    return 5


@pytest.fixture
def n_obs() -> int:
    """Default time-series length for synthetic markets used in tests.

    Kept small so the suite runs in seconds; large enough for the regression
    estimators to be informative.
    """
    return 600
