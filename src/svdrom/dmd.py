import numpy as np
import xarray as xr
from pydmd import BOPDMD

from svdrom.logger import setup_logger

logger = setup_logger("DMD", "dmd.log")


class OptDMD:
    def __init__(
        self,
        n_modes: int = -1,
        time_dimension: str = "time",
        time_units: str = "s",
        num_trials: int = 0,
        trial_size: int | float = 0.6,
    ) -> None:
        """Optimized Dynamic Mode Decomposition (DMD) via variable
        projection method for nonlinear least squares, with optional
        bootstrap aggregation (bagging) for uncertainty quantification.

        This class makes use of the BOPDMD class from the PyDMD library
        (https://pydmd.github.io/PyDMD/bopdmd.html).

        Parameters
        ----------
        n_modes: int
            Number of DMD modes to compute (must be positive or -1 for all
            available modes).
        time_dimension: str
            Name of the time dimension in the input data. Default is "time".
        time_units: str
            Units of the time dimension. Default is "s".
            Must be one of {"s", "h"}, where "s" is seconds and "h" is hours.
        num_trials: int
            Number of bagging trials to perform. Default is 0 (no bagging).
        trial_size: int | float
            Size of the randomly selected subset of snapshots to use for
            each trial of bagged optimized DMD. If it's a positive integer,
            "trial_size" many snapshots will be used per trial. If it's a
            float between 0 and 1, then "trial_size" denotes the fraction of
            snapshots to be used per trial. Default is 0.6.

        Notes
        -----
        This class is a wrapper of the `BOPDMD.fit_econ()` method, which fits
        an approximate Optimized DMD on an array X by operating on the SVD of X.
        """
        if n_modes != -1 and n_modes < 1:
            msg = "'n_modes' must be a positive integer or -1."
            logger.exception(msg)
            raise ValueError(msg)
        if time_units not in {"s", "h"}:
            msg = "'time_units' must be one of {'s', 'h'}."
            logger.exception(msg)
            raise ValueError(msg)
        if num_trials < 0 or not isinstance(num_trials, int):
            msg = "'num_trials' must be an integer greater than zero."
            logger.exception(msg)
            raise ValueError(msg)
        if (
            isinstance(trial_size, float)
            and (trial_size <= 0 or trial_size >= 1)
            or isinstance(trial_size, int)
            and trial_size <= 0
        ):
            msg = "'trial_size' must be a positive integer or a float between 0 and 1."
            logger.exception(msg)
            raise ValueError(msg)

        self._n_modes = n_modes
        self._time_dimension = time_dimension
        self._time_units = time_units
        self._num_trials = num_trials
        self._trial_size = trial_size
        self._eigs: np.ndarray | None = None
        self._eigs_std: np.ndarray | None = None
        self._amplitudes: np.ndarray | None = None
        self._amplitudes_std: np.ndarray | None = None
        self._modes: xr.DataArray | None = None
        self._modes_std: xr.DataArray | None = None
        self._solver: BOPDMD | None = None
        self._is_fitted: bool = False
        self._t_fit: np.ndarray | None = None
        self._t_forecast: np.ndarray | None = None

    @property
    def n_modes(self) -> int:
        """Number of DMD modes (read-only)."""
        return self._n_modes

    @property
    def time_dimension(self) -> str:
        """Name of the time dimension (read-only)."""
        return self._time_dimension

    @property
    def eigs(self) -> np.ndarray | None:
        """The DMD eigenvalues (read-only)."""
        return self._eigs

    @property
    def modes(self) -> xr.DataArray | None:
        """The DMD modes (read-only)."""
        return self._modes

    @property
    def amplitudes(self) -> np.ndarray | None:
        """The DMD amplitudes (read-only)."""
        return self._amplitudes

    @property
    def eigs_std(self) -> np.ndarray | None:
        """The standard deviation of the DMD eigenvalues,
        when using bagging (read-only).
        """
        return self._eigs_std

    @property
    def modes_std(self) -> xr.DataArray | None:
        """The standard deviation of the DMD modes,
        when using bagging (read-only)."""
        return self._modes_std

    @property
    def amplitudes_std(self) -> np.ndarray | None:
        """The standard deviation of the DMD amplitudes,
        when using bagging (read-only)."""
        return self._amplitudes_std

    @property
    def num_trials(self) -> int:
        """The number of bagging trials (read-only)."""
        return self._num_trials

    @property
    def trial_size(self) -> int | float:
        """The bagging trial size (read-only)."""
        return self._trial_size

    @property
    def time_units(self) -> str:
        """The time units to use in the DMD fit and forecast (read-only)."""
        return self._time_units

    @property
    def solver(self) -> BOPDMD | None:
        """The DMD solver instance (read-only)."""
        return self._solver

    @property
    def is_fitted(self) -> bool:
        """Whether an optimized DMD model has been fitted (read-only)."""
        return self._is_fitted

    def _check_svd_inputs(self, u: xr.DataArray, s: np.ndarray, v: xr.DataArray):
        """Check that the passed SVD results are valid."""
        if not isinstance(u.data, np.ndarray):
            msg = "The left singular vectors have not been computed."
            logger.exception(msg)
            raise ValueError(msg)
        if not isinstance(v.data, np.ndarray):
            msg = "The right singular vectors have not been computed."
            logger.exception(msg)
            raise ValueError(msg)
        if len(u.components) != len(v.components) or len(u.components) != len(s):
            msg = "'u', 's' and 'v' must have the same number of components."
            logger.exception(msg)
            raise ValueError(msg)
        if self._time_dimension not in v.dims:
            msg = (
                f"Specified time dimension '{self._time_dimension}' not "
                "a dimension of the right singular vectors 'v'."
            )
            logger.exception(msg)
            raise ValueError(msg)

        def is_sorted(x):
            return np.all(x[:-1] <= x[1:])

        if not is_sorted(v[self._time_dimension].values):
            msg = (
                f"Time dimension '{self._time_dimension}' is not "
                "sorted in the right singular vectors 'v'."
            )
            logger.exception(msg)
            raise ValueError(msg)

    def _generate_fit_time_vector(self, v: xr.DataArray) -> np.ndarray:
        """Given the right singular vectors containing the temporal
        information, generate the time vector for the DMD fit.
        """
        time_deltas = np.diff(v[self._time_dimension].values).astype(
            f"timedelta64[{self._time_units}]"
        )
        time_vector = np.cumsum(time_deltas)
        start_time = np.array([0], dtype=f"timedelta64[{self._time_units}]")
        time_vector = np.concat((start_time, time_vector))
        return time_vector.astype("float64")

    def _extract_results(self, bopdmd: BOPDMD, u: xr.DataArray) -> None:
        """Given the fitted BOPDMD instance and the left singular vectors
        containing the spatial information, extract the DMD results and
        store them in the instance attributes."""
        self._solver = bopdmd
        self._modes = u.copy(data=bopdmd.modes)  # use new data with original structure
        self._modes.name = "dmd_modes"
        self._eigs = bopdmd.eigs
        self._amplitudes = bopdmd.amplitudes
        if self.num_trials > 0:
            self._modes_std = u.copy(data=bopdmd.modes)
            self._modes_std.name = "dmd_modes_std"
            self._eigs_std = bopdmd.eigenvalues_std
            self._amplitudes_std = bopdmd.amplitudes_std

    def fit(
        self, u: xr.DataArray, s: np.ndarray, v: xr.DataArray, **kwargs
    ) -> "OptDMD":
        """Fit a OptDMD model to the results of a Singular Value Decomposition (SVD).

        Parameters
        ----------
        u: xarray.DataArray, shape (n_spatial_points, n_components)
            The left singular vectors, containing the spatial information.
        s: np.ndarray, shape (n_components,)
            The singular values.
        v: xarray.DataArray, shape (n_components, n_timesteps)
            The right singular vectors, containing the temporal information.
        **kwargs:
            Additional keyword arguments to pass to PyDMD's BOPDMD constructor.
            See https://pydmd.github.io/PyDMD/bopdmd.html for more info.
        """
        self._check_svd_inputs(u, s, v)
        if self._n_modes > len(s):
            msg = (
                "The requested number of DMD modes exceeds the number "
                "of available SVD components."
            )
            logger.exception(msg)
            raise ValueError(msg)
        if self._n_modes == -1:
            self._n_modes = len(s)

        bopdmd = BOPDMD(
            svd_rank=self._n_modes,
            use_proj=True,
            proj_basis=u.data[:, : self._n_modes],
            num_trials=self._num_trials,
            trial_size=self._trial_size,
            **kwargs,
        )
        self._t_fit = self._generate_fit_time_vector(v)
        logger.info("Computing the DMD fit...")
        try:
            bopdmd.fit_econ(
                s[: self._n_modes],
                v.data[: self._n_modes, :],
                self._t_fit.astype("float64"),
            )
        except Exception as e:
            msg = "Error computing the DMD fit."
            logger.exception(msg)
            raise RuntimeError(msg) from e
        logger.info("Done.")
        self._is_fitted = True
        self._extract_results(bopdmd, u)

        return self
