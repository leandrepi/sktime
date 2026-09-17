"""Multivariate MiniRocket transformer, Cython implementation (no numba)."""

__author__ = ["sssilvar"]
__all__ = ["MiniRocketMultivariateCython"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class MiniRocketMultivariateCython(BaseTransformer):
    """MiniRocket multivariate transform, Cython backend (no numba).

    Numerically equivalent to ``MiniRocketMultivariate`` but uses ahead-of-time
    compiled Cython kernels instead of numba, eliminating JIT "warmup" latency
    (which can exceed 30s on the first transform of larger series). The numba
    ``MiniRocketMultivariate`` is retained as the reference/groundtruth.

    MiniRocketMultivariate [1]_ is an almost deterministic version of Rocket. It
    creates convolutions of length 9 with weights restricted to two values, and
    uses 84 fixed convolutions with six of one weight, three of the second weight
    to seed dilations. Works with univariate and multivariate time series.

    This transformer fits one set of parameters per individual series, and
    applies the transform with fitted parameter i to the i-th series in transform.
    Vanilla use requires the same number of series in fit and transform.

    Migrating fitted parameters to/from ``MiniRocketMultivariate``
    --------------------------------------------------------------
    ``save``/``load`` are *not* interchangeable between this class and the numba
    ``MiniRocketMultivariate`` (each restores its own type). The fitted
    ``parameters`` tuple is, however, identical in format, so a fitted estimator
    can be migrated by copying that attribute (transforms then match to ~1e-5,
    a float32 rounding difference)::

        src = MiniRocketMultivariate(num_kernels=..., random_state=...).fit(X)
        dst = MiniRocketMultivariateCython(num_kernels=..., random_state=...)
        dst.fit(X)                 # any panel with the same n_columns
        dst.parameters = src.parameters

    The direction also works in reverse (Cython -> numba).

    Parameters
    ----------
    num_kernels : int, default=10,000
       number of random convolutional kernels. This should be a multiple of 84.
       If it is lower than 84, it will be set to 84. If it is higher than 84
       and not a multiple of 84, the number of kernels used to transform the
       data will be rounded down to the next positive multiple of 84.
    max_dilations_per_kernel : int, default=32
        maximum number of dilations per kernel.
    n_jobs : int, default=1
        Number of threads used in ``transform`` (the GIL-releasing Cython kernel
        is run over disjoint instance chunks). ``-1`` uses all processors.
    random_state : None or int, default = None

    Attributes
    ----------
    num_kernels_ : int
        The true number of kernels used in the rocket transform. This is
        num_kernels rounded down to the nearest multiple of 84. It is 84 if
        num_kernels is less than 84.

    See Also
    --------
    MiniRocketMultivariate, MultiRocketMultivariate, MiniRocket, Rocket

    References
    ----------
    .. [1] Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I,
        "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series
        Classification",2020,
        https://dl.acm.org/doi/abs/10.1145/3447548.3467231,
        https://arxiv.org/abs/2012.08791

    Examples
    --------
     >>> from sktime.transformations.rocket import MiniRocketMultivariateCython
     >>> from sktime.datasets import load_basic_motions
     >>> X_train, y_train = load_basic_motions(split="train") # doctest: +SKIP
     >>> trf = MiniRocketMultivariateCython(num_kernels=512) # doctest: +SKIP
     >>> trf.fit(X_train) # doctest: +SKIP
     MiniRocketMultivariateCython(...)
     >>> X_train = trf.transform(X_train) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["sssilvar"],
        "maintainers": ["sssilvar"],
        "python_dependencies": ["sktime-cython"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # test and CI flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        num_kernels=10_000,
        max_dilations_per_kernel=32,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.num_kernels_ = None
        self.n_jobs = n_jobs
        self.random_state = random_state

        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(
                f"random_state in MiniRocketMultivariateCython must be int or None, "
                f"but found {type(random_state)}"
            )
        if isinstance(random_state, int):
            self.random_state_ = np.int32(random_state)
        else:
            self.random_state_ = random_state

        super().__init__()

    def _fit(self, X, y=None):
        """Fit dilations and biases to input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        from sktime_cython.transformations.rocket import rocket_fit

        self.parameters = rocket_fit(
            X, self.num_kernels, self.max_dilations_per_kernel, self.random_state_
        )
        if self.num_kernels < 84:
            self.num_kernels_ = 84
        else:
            self.num_kernels_ = (self.num_kernels // 84) * 84

        return self

    def _transform(self, X, y=None):
        """Transform input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        pandas DataFrame, transformed features
        """
        from sktime_cython.transformations.rocket import rocket_transform

        return pd.DataFrame(rocket_transform(X, self.parameters, self.n_jobs))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params = [
            {
                "num_kernels": 84,
                "random_state": 42,
                "max_dilations_per_kernel": 32,
            },
            {
                "num_kernels": 42,
                "random_state": 84,
                "max_dilations_per_kernel": 16,
            },
        ]
        return params
