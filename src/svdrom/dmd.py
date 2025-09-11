from typing import Literal

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
        time_units: Literal["s", "h"] = "s",
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
            Units in which to treat the time dimension. Default is "s".
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
        self._time_fit: np.ndarray | None = None
        self._t_fit: np.ndarray | None = None  # internal use only
        self._time_forecast: np.ndarray | None = None
        self._t_forecast: np.ndarray | None = None  # internal use only

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
    def time_fit(self) -> np.ndarray | None:
        """The time vector for the DMD fit (read-only)."""
        return self._time_fit

    @property
    def time_forecast(self) -> np.ndarray | None:
        """The time vector for the DMD forecast (read-only)."""
        return self._time_forecast

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
        time_fit = v[self._time_dimension].values
        time_deltas = np.diff(time_fit).astype(f"timedelta64[{self._time_units}]")
        time_vector = np.cumsum(time_deltas)
        start_time = np.array([0], dtype=f"timedelta64[{self._time_units}]")
        self._time_fit = time_fit  # vector representing true time of the training data
        self._t_fit = np.concat(
            (start_time, time_vector)
        )  # vector to be fed to the `fit()` call
        return self._t_fit

    def _generate_forecast_time_vector(
        self, forecast_span: str | int, dt: str | int | None = None
    ) -> np.ndarray:
        """Given a forecast time span and time step, generate the time vector
        for the DMD forecast.

        Parameters
        ----------
        forecast_span: str | int
            The total time span for the forecast. If int, interpreted as time
            in the model's time units. If str, should be in the format "value units",
            e.g. "30 D" for 30 days.
        dt: str | int | None, optional
            The time step or number of points for the forecast. If str, should be in
            the format "value units", e.g. "1 h" for 1 hour. If int, interpreted as
            number of forecast points. If None, uses the average time step of the
            training data.
        """
        if self._t_fit is None or self._time_fit is None:
            msg = "The DMD fit time vector is not initialized."
            raise ValueError(msg)

        # parse forecast_span
        if isinstance(forecast_span, int):
            span = np.timedelta64(forecast_span, self._time_units)
        else:
            # forecast_span is a string
            span_parts = forecast_span.split(" ")
            if len(span_parts) != 2:
                msg = (
                    "String forecast_span must be in the format 'value units', "
                    "e.g. '30 D'."
                )
                raise ValueError(msg)
            span = np.timedelta64(int(span_parts[0]), span_parts[1])  # type: ignore[call-overload]

        # determine time step
        if dt is None:
            # use average time step of training data
            if len(self._t_fit) > 1:
                time_step = np.mean(np.diff(self._t_fit))
            else:
                msg = (
                    "Cannot determine time step from training data "
                    "with only one time point."
                )
                raise ValueError(msg)
        elif isinstance(dt, str):
            # parse time step string
            dt_parts = dt.split(" ")
            if len(dt_parts) != 2:
                msg = "String dt must be in the format 'value units', e.g. '1 h'."
                raise ValueError(msg)
            time_step = np.timedelta64(int(dt_parts[0]), dt_parts[1])  # type: ignore[call-overload]
        else:
            # dt is an int and specifies number of points
            if dt <= 0:
                msg = "Number of forecast points must be positive."
                raise ValueError(msg)
            # calculate time step to get the requested number of points
            time_step = span / dt

        # generate the time vector representing true time of the forecast
        forecast_start = self._time_fit[-1]
        forecast_end = forecast_start + span
        self._time_forecast = np.arange(
            forecast_start + time_step,
            forecast_end + time_step,
            time_step,
        )

        # generate the time vector to be fed to the `forecast()` call
        forecast_start = self._t_fit[-1]
        forecast_end = forecast_start + span
        self._t_forecast = np.arange(
            forecast_start + time_step,
            forecast_end + time_step,
            time_step,
        )

        return self._t_forecast

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
            self._modes_std = u.copy(data=bopdmd.modes_std)
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
        u, s, v = u[:, : self._n_modes], s[: self._n_modes], v[: self._n_modes, :]

        bopdmd = BOPDMD(
            svd_rank=self._n_modes,
            use_proj=True,
            proj_basis=u.data,
            num_trials=self._num_trials,
            trial_size=self._trial_size,
            **kwargs,
        )
        t_fit = self._generate_fit_time_vector(v)
        logger.info("Computing the DMD fit...")
        try:
            bopdmd.fit_econ(
                s,
                v.data,
                t_fit.astype("float64"),
            )
        except Exception as e:
            msg = "Error computing the DMD fit."
            logger.exception(msg)
            raise RuntimeError(msg) from e
        logger.info("Done.")
        self._extract_results(bopdmd, u)

        return self

    def _forecast_to_dataarray(
        self, forecast: np.ndarray | tuple[np.ndarray, np.ndarray]
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """Given a single numpy.ndarray consisting of a single DMD forecast,
        or a tuple of two numpy.ndarray where the first one corresponds to
        the ensemble mean forecast and the second one corresponds to the
        ensemble variance, convert the input into xarray.DataArray with the
        corresponding format.
        """
        if self._modes is None:
            msg = "The DMD modes have not been computed."
            raise RuntimeError(msg)

        dims = (self._modes.dims[0], self._time_dimension)
        coords = {k: v for k, v in self._modes.coords.items() if k != "components"}
        coords[self._time_dimension] = self._time_forecast
        if isinstance(forecast, np.ndarray):
            return xr.DataArray(forecast, dims=dims, coords=coords, name="dmd_forecast")
        forecast_mean = xr.DataArray(
            forecast[0], dims=dims, coords=coords, name="dmd_forecast_mean"
        )
        forecast_var = xr.DataArray(
            forecast[1], dims=dims, coords=coords, name="dmd_forecast_var"
        )
        return forecast_mean, forecast_var

    def forecast(
        self, forecast_span: str | int, dt: str | int | None
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """
        Generates a forecast using the fitted OptDMD model over a specified time span.
        The model must be fitted before calling this method.

        Parameters
        ----------
        forecast_span: str | int
            The total time span for the forecast. If int, interpreted as time
            in the model's time units. If str, should be in the format "value units",
            e.g. "30 D" for 30 days.
        dt: str | int | None, optional
            The time step or number of time points for the forecast. If str, should be
            in the format "value units", e.g. "1 h" for 1 hour. If int, interpreted as
            number of forecast points. If None, uses the average time step of the
            training data on which the DMD model was fitted.

        Returns
        -------
        xarray.DataArray | tuple[xarray.DataArray, xarray.DataArray]
            The forecasted data as an xarray.DataArray. If bagging is used, two
            xarray.DataArray are returned where the first one is the ensemble mean
            and the second one is the ensemble variance.
        """
        if self._solver is None:
            msg = "The OptDMD model must be fitted before forecasting."
            logger.exception(msg)
            raise RuntimeError(msg)
        try:
            t_forecast = self._generate_forecast_time_vector(forecast_span, dt)
        except Exception as e:
            msg = "Error trying to generate the forecast time vector."
            logger.exception(msg)
            raise RuntimeError(msg) from e
        logger.info("Computing the DMD forecast...")
        try:
            forecast = self._solver.forecast(t_forecast.astype("float64"))
            if self.num_trials == 0:
                assert isinstance(forecast, np.ndarray), (
                    "Without bagging, expected the forecast to return "
                    "a single numpy.ndarray."
                )
            else:
                assert isinstance(forecast, tuple), (
                    "With bagging, expected the forecast to return "
                    "two numpy.ndarray."
                )
                assert len(forecast) == 2, (
                    "With bagging, expected the forecast to return "
                    "two numpy.ndarray."
                )
        except Exception as e:
            msg = "Error computing the DMD forecast."
            logger.exception(msg)
            raise RuntimeError(msg) from e
        logger.info("Done.")
        try:
            forecast = self._forecast_to_dataarray(forecast)
        except Exception as e:
            msg = "Error trying to convert forecast into xarray.DataArray."
            logger.exception(msg)
            raise RuntimeError(msg) from e

        return forecast
