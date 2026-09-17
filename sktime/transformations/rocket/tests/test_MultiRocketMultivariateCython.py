"""Groundtruth tests for MultiRocketMultivariateCython vs the numba reference."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.rocket import (
    MultiRocketMultivariate,
    MultiRocketMultivariateCython,
)

# At large num_kernels a value can sit on the bias threshold and be classified
# differently by the two float32 summation orders, flipping the PPV/MPV/MIPV
# triplet of one feature. Measured rate is ~2e-5 of elements; allow 1e-4.
_MISMATCH_BUDGET = 1e-4


@pytest.mark.skipif(
    not run_test_for_class(MultiRocketMultivariateCython),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_columns", [1, 4])
@pytest.mark.parametrize(
    "num_kernels,max_dilations_per_kernel,random_state,n_timepoints",
    [(84, 32, 42, 60), (168, 16, 7, 60), (6250, 32, 0, 137)],
)
def test_cython_matches_numba(
    n_columns, num_kernels, max_dilations_per_kernel, random_state, n_timepoints
):
    """Cython transform must match the numba implementation (groundtruth)."""
    rng = np.random.RandomState(random_state)
    X = rng.normal(size=(6, n_columns, n_timepoints))

    kw = dict(
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
    )
    numba_out = MultiRocketMultivariate(**kw).fit_transform(X)
    cython_out = MultiRocketMultivariateCython(**kw).fit_transform(X)

    assert cython_out.shape == numba_out.shape
    close = np.isclose(
        cython_out.to_numpy(), numba_out.to_numpy(), rtol=1e-4, atol=1e-5
    )
    assert 1 - close.mean() <= _MISMATCH_BUDGET


@pytest.mark.skipif(
    not run_test_for_class(MultiRocketMultivariateCython),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fitted_parameters_migrate_from_numba():
    """The fitted parameter tuples are interchangeable with the numba class."""
    X = np.random.RandomState(0).normal(size=(6, 3, 60))
    kw = dict(num_kernels=84, max_dilations_per_kernel=32, random_state=0)

    src = MultiRocketMultivariate(**kw).fit(X)
    dst = MultiRocketMultivariateCython(**kw).fit(X)
    dst.parameter, dst.parameter1 = src.parameter, src.parameter1

    close = np.isclose(
        dst.transform(X).to_numpy(),
        src.transform(X).to_numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    assert 1 - close.mean() <= _MISMATCH_BUDGET
